"""
generate_logs.py
Synthetic Microsoft 365 Unified Audit Log generator.

Produces three JSONL files that mirror the real M365 audit log schema:
  data/train.jsonl        — normal behaviour only
  data/val.jsonl          — normal behaviour only
  data/anomaly_test.jsonl — mixed normal + labeled anomalies

Each line is one JSON event. Anomaly records carry an extra key:
  "_anomaly": {"label": true, "type": "impossible_travel"}

Architecture
------------
  UserProfile      — role-based behavioural archetype with personal variation.
                     Each user has a fixed role (developer, hr, manager, etc.)
                     that determines their core operation weights, activity level,
                     and workload preferences. Personal quirks are layered on top.
  EventSampler     — draws realistic events for a given user + timestamp,
                     respecting that user's role-specific operation distribution.
  AnomalyInjector  — injects labeled anomalies into the test set only.

User roles
----------
  developer    — SharePoint/OneDrive heavy, file sync, high volume
  hr           — Exchange dominant, sensitive files, moderate volume
  manager      — Mixed email + SharePoint, occasional AAD, low-moderate volume
  it_admin     — AAD heavy, device/user management, password ops
  sales        — Teams + Exchange heavy, some SharePoint
  finance      — SharePoint heavy, specific file types (xlsx/pdf), low AAD
  employee     — Baseline mixed behaviour, catch-all role

Absence modelling
-----------------
  Holidays:        30% of users take 1 block of 3-10 days per 90-day period.
                   Activity during holiday: ~2% (mobile email check only).
  Sick leave:      15% of users have 1 sick episode of 1-3 days.
                   Activity during sick leave: ~8% (light mobile activity).
  Public holidays: Fixed dates — complete silence for all users.
  Weekends:        ~5% of weekday volume.

These are intentionally rare so the model learns absence is occasionally
normal, not that it is suspicious.

Flags
-----
  --train-only   Generate train.jsonl only. Use when resuming training.
"""

import argparse
import json
import random
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SEED           = 42
N_USERS        = 500
N_TRAIN        = 5_000_000
N_VAL          = 1_000_000
N_ANOMALY_TEST = 50_000
ANOMALY_RATIO  = 0.001
OUT_DIR        = Path("data")

ORG_DOMAIN     = "contoso.com"
ORG_ID         = str(uuid.uuid4())

WORK_HOURS_START = 7
WORK_HOURS_END   = 18

random.seed(SEED)

# ── Public holidays ───────────────────────────────────────────────────────────

PUBLIC_HOLIDAYS = {
    "2024-10-14",
    "2024-11-01",
    "2024-12-24",
    "2024-12-25",
    "2024-12-26",
    "2024-12-31",
}

# ── Role definitions ──────────────────────────────────────────────────────────

ROLES = {
    "developer": {
        "daily_range": (80, 300),
        "role_weight": 12,
        "base_ops": {
            "FileAccessed":               300,
            "FileSyncDownloadedFull":     180,
            "FileModified":               120,
            "FileModifiedExtended":        60,
            "FileUploaded":                80,
            "FileUploadedPartial":         60,
            "FileDownloaded":              40,
            "FileMoved":                   20,
            "FileDeleted":                 15,
            "UserLoggedIn":                30,
            "UserLoggedOut":               20,
            "Update StsRefreshTokenValidFrom Timestamp.": 15,
            "MailItemsAccessed":           10,
            "MessageSent":                  8,
            "SharingSet":                  10,
        },
    },
    "hr": {
        "daily_range": (40, 150),
        "role_weight": 2,
        "base_ops": {
            "MailItemsAccessed":          200,
            "Send":                       120,
            "MoveToDeletedItems":          20,
            "FileAccessed":               100,
            "FilePreviewed":               60,
            "FileDownloaded":              30,
            "MessageSent":                 40,
            "UserLoggedIn":                25,
            "UserLoggedOut":               15,
            "Update PasswordProfile.":     10,
            "Change user password.":        5,
            "Reset user password.":        10,
            "Update user.":                15,
        },
    },
    "manager": {
        "daily_range": (30, 120),
        "role_weight": 8,
        "base_ops": {
            "MailItemsAccessed":          160,
            "Send":                        80,
            "MessageSent":                 60,
            "MeetingStarted":              40,
            "FileAccessed":               120,
            "FilePreviewed":               80,
            "FileDownloaded":              20,
            "UserLoggedIn":                20,
            "UserLoggedOut":               15,
            "SharingSet":                  15,
            "Update user.":                10,
            "Consent to application.":      5,
        },
    },
    "it_admin": {
        "daily_range": (50, 200),
        "role_weight": 5,
        "base_ops": {
            "Update user.":               200,
            "Update device.":             180,
            "Update PasswordProfile.":    100,
            "Reset user password.":        60,
            "Change user password.":       30,
            "Add delegated permission grant.":        40,
            "Add app role assignment grant to user.": 30,
            "Add service principal.":      20,
            "Set Company Information.":    15,
            "Consent to application.":     25,
            "UserLoggedIn":                40,
            "UserLoggedOut":               30,
            "Update StsRefreshTokenValidFrom Timestamp.": 50,
            "FileAccessed":                40,
            "MailItemsAccessed":           20,
        },
    },
    "sales": {
        "daily_range": (60, 180),
        "role_weight": 15,
        "base_ops": {
            "MessageSent":                180,
            "MeetingStarted":              60,
            "MailItemsAccessed":          150,
            "Send":                        90,
            "FileAccessed":                80,
            "FilePreviewed":               50,
            "FileDownloaded":              30,
            "SharingSet":                  40,
            "UserLoggedIn":                25,
            "UserLoggedOut":               15,
            "MoveToDeletedItems":          10,
        },
    },
    "finance": {
        "daily_range": (50, 160),
        "role_weight": 5,
        "base_ops": {
            "FileAccessed":               250,
            "FilePreviewed":              120,
            "FileDownloaded":              80,
            "FileModified":                60,
            "FileUploadedPartial":         30,
            "FileUploaded":                25,
            "MailItemsAccessed":           80,
            "Send":                        40,
            "UserLoggedIn":                20,
            "UserLoggedOut":               15,
            "FileSyncDownloadedFull":      20,
            "SharingSet":                   8,
        },
    },
    "employee": {
        "daily_range": (20, 100),
        "role_weight": 53,
        "base_ops": {
            "FileAccessed":               200,
            "FilePreviewed":               90,
            "FileModified":                35,
            "FileModifiedExtended":        15,
            "FileSyncDownloadedFull":      18,
            "FileDownloaded":               8,
            "FileUploaded":                12,
            "FileUploadedPartial":         10,
            "FileDeleted":                  6,
            "FileMoved":                    4,
            "SharingSet":                   5,
            "UserLoggedIn":                40,
            "UserLoggedOut":               12,
            "Update user.":                 5,
            "Update PasswordProfile.":      3,
            "MailItemsAccessed":           28,
            "Send":                        10,
            "MoveToDeletedItems":           4,
            "MessageSent":                 10,
            "MeetingStarted":               4,
        },
    },
}

WORKLOAD_MAP = {
    "FileAccessed":                           "SharePoint",
    "FilePreviewed":                          "SharePoint",
    "FileModified":                           "SharePoint",
    "FileModifiedExtended":                   "SharePoint",
    "FileSyncDownloadedFull":                 "OneDrive",
    "FileDownloaded":                         "SharePoint",
    "FileUploaded":                           "SharePoint",
    "FileUploadedPartial":                    "OneDrive",
    "FileDeleted":                            "SharePoint",
    "FileMoved":                              "SharePoint",
    "SharingSet":                             "SharePoint",
    "UserLoggedIn":                           "AzureActiveDirectory",
    "UserLoginFailed":                        "AzureActiveDirectory",
    "UserLoggedOut":                          "AzureActiveDirectory",
    "Update user.":                           "AzureActiveDirectory",
    "Update device.":                         "AzureActiveDirectory",
    "Update PasswordProfile.":                "AzureActiveDirectory",
    "Change user password.":                  "AzureActiveDirectory",
    "Reset user password.":                   "AzureActiveDirectory",
    "Consent to application.":                "AzureActiveDirectory",
    "Add delegated permission grant.":        "AzureActiveDirectory",
    "Set Company Information.":               "AzureActiveDirectory",
    "Add app role assignment grant to user.": "AzureActiveDirectory",
    "Add service principal.":                 "AzureActiveDirectory",
    "Update StsRefreshTokenValidFrom Timestamp.": "AzureActiveDirectory",
    "Disable account":                        "AzureActiveDirectory",
    "Disable Strong Authentication":          "AzureActiveDirectory",
    "MailItemsAccessed":                      "Exchange",
    "Send":                                   "Exchange",
    "MoveToDeletedItems":                     "Exchange",
    "MessageSent":                            "MicrosoftTeams",
    "MeetingStarted":                         "MicrosoftTeams",
}

RECORD_TYPE_MAP = {
    "AzureActiveDirectory": 15,
    "SharePoint":            6,
    "OneDrive":              6,
    "Exchange":              2,
    "MicrosoftTeams":       25,
}

APP_MAP = {
    "AzureActiveDirectory": [
        "Microsoft 365 portal", "Azure portal",
        "Microsoft Authenticator", "Microsoft Teams", "Microsoft Outlook",
    ],
    "SharePoint":     ["SharePoint Online", "OneDrive for Business", "Microsoft Teams"],
    "OneDrive":       ["OneDrive for Business", "Microsoft Teams"],
    "Exchange":       ["Microsoft Outlook", "Microsoft Teams", "Outlook Web App"],
    "MicrosoftTeams": ["Microsoft Teams"],
}

USER_TYPE_WEIGHTS = {0: 65, 2: 8, 4: 20, 5: 7}

DEVICE_COMBOS = [
    ("Windows 11", "Edge"),
    ("Windows 10", "Edge"),
    ("Windows 10", "Chrome"),
    ("macOS 14",   "Safari"),
    ("macOS 13",   "Chrome"),
    ("iOS 17",     "Safari"),
    ("Android 13", "Chrome"),
]

DK_LOCATIONS = [
    {"city": "Copenhagen", "country": "Denmark", "countryCode": "DK", "ip_prefix": "185.20"},
    {"city": "Aarhus",     "country": "Denmark", "countryCode": "DK", "ip_prefix": "195.249"},
    {"city": "Odense",     "country": "Denmark", "countryCode": "DK", "ip_prefix": "80.167"},
    {"city": "Aalborg",    "country": "Denmark", "countryCode": "DK", "ip_prefix": "91.208"},
]

FOREIGN_LOCATIONS = [
    {"city": "Moscow",    "country": "Russia",  "countryCode": "RU", "ip_prefix": "91.108"},
    {"city": "Beijing",   "country": "China",   "countryCode": "CN", "ip_prefix": "118.26"},
    {"city": "Lagos",     "country": "Nigeria", "countryCode": "NG", "ip_prefix": "197.255"},
    {"city": "Bucharest", "country": "Romania", "countryCode": "RO", "ip_prefix": "79.112"},
    {"city": "Kyiv",      "country": "Ukraine", "countryCode": "UA", "ip_prefix": "91.200"},
]

# Legitimate travel destinations — normal users occasionally visit these
TRAVEL_LOCATIONS = [
    {"city": "London",    "country": "UK",          "countryCode": "GB", "ip_prefix": "81.102"},
    {"city": "Berlin",    "country": "Germany",     "countryCode": "DE", "ip_prefix": "87.123"},
    {"city": "Amsterdam", "country": "Netherlands", "countryCode": "NL", "ip_prefix": "84.241"},
    {"city": "Stockholm", "country": "Sweden",      "countryCode": "SE", "ip_prefix": "85.224"},
    {"city": "Oslo",      "country": "Norway",      "countryCode": "NO", "ip_prefix": "88.89"},
    {"city": "Paris",     "country": "France",      "countryCode": "FR", "ip_prefix": "90.63"},
    {"city": "New York",  "country": "USA",         "countryCode": "US", "ip_prefix": "74.125"},
]

SP_SITES = [
    "https://contoso.sharepoint.com/sites/Finance",
    "https://contoso.sharepoint.com/sites/HR",
    "https://contoso.sharepoint.com/sites/IT",
    "https://contoso.sharepoint.com/sites/Sales",
    "https://contoso-my.sharepoint.com/personal",
]

ROLE_FILES = {
    "developer":  ["code-review.docx", "architecture.pdf", "api-spec.xlsx",
                   "deployment-plan.docx", "test-results.xlsx", "README.md",
                   "sprint-backlog.xlsx", "system-design.docx"],
    "hr":         ["Employee-Handbook.pdf", "Onboarding-Template.docx",
                   "Performance-Review.xlsx", "Salary-Survey.xlsx",
                   "Job-Description.docx", "HR-Policy.pdf",
                   "Termination-Checklist.docx", "Benefits-Overview.pdf"],
    "manager":    ["Q4-Report.xlsx", "Team-OKRs.docx", "Budget-2024.xlsx",
                   "Project-Plan.xlsx", "Meeting-Notes.docx", "Strategy.pptx",
                   "Headcount-Plan.xlsx", "Risk-Register.xlsx"],
    "it_admin":   ["IT-Policy.pdf", "Security-Audit.xlsx", "Asset-Register.xlsx",
                   "Incident-Report.docx", "Network-Diagram.pdf",
                   "Patch-Schedule.xlsx", "Access-Review.xlsx"],
    "sales":      ["Sales-Pipeline.xlsx", "Client-Contracts.docx",
                   "Proposal-Template.docx", "Customer-Data.csv",
                   "Pricing-Sheet.xlsx", "Case-Study.pdf",
                   "Account-Plan.docx", "Competitor-Analysis.xlsx"],
    "finance":    ["Budget-2024.xlsx", "Q4-Report.xlsx", "Audit-Report.pdf",
                   "Invoice-Register.xlsx", "Forecast-2025.xlsx",
                   "Customer-Data.csv", "Tax-Filing.pdf", "P&L-Statement.xlsx"],
    "employee":   ["Meeting-Notes.docx", "Project-Plan.xlsx",
                   "Employee-Handbook.pdf", "Q4-Report.xlsx",
                   "Client-Contracts.docx", "Audit-Report.pdf"],
}

DEPARTMENTS = {
    "developer":  ["Engineering", "IT", "Product"],
    "hr":         ["HR", "People & Culture"],
    "manager":    ["Management", "Operations", "Finance", "Sales", "Engineering"],
    "it_admin":   ["IT", "Operations"],
    "sales":      ["Sales", "Marketing", "Business Development"],
    "finance":    ["Finance", "Legal", "Accounting"],
    "employee":   ["Operations", "Marketing", "Legal", "Finance", "HR", "Engineering"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def random_ip(prefix: str) -> str:
    return f"{prefix}.{random.randint(1,254)}.{random.randint(1,254)}"

def random_guid() -> str:
    return str(uuid.uuid4())

def fmt_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def weighted_choice(d: dict) -> str:
    keys    = [k for k, v in d.items() if v > 0]
    weights = [d[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]

def date_range(start: datetime, end: datetime):
    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while current < end:
        yield current
        current += timedelta(days=1)

def random_weekday(start: datetime, end: datetime) -> datetime:
    candidates = [d for d in date_range(start, end) if d.weekday() < 5]
    return random.choice(candidates) if candidates else start


# ── User profiles ─────────────────────────────────────────────────────────────

class UserProfile:
    """
    Role-based user with personal operation weight variation,
    fixed behavioural patterns, and realistic absence modelling.
    """

    def __init__(self, user_id: int, start: datetime, end: datetime):
        first_names = [
            "Oliver", "Emma", "Noah", "Sophia", "Liam", "Mia",
            "Lucas", "Isabella", "Ethan", "Charlotte", "Magnus",
            "Astrid", "Frederik", "Maja", "Christian", "Laura",
            "Thomas", "Anna", "Mikkel", "Sara", "Jonas", "Ida",
            "Kasper", "Julie", "Andreas", "Louise", "Martin", "Camilla",
            "Søren", "Line", "Peter", "Maria", "Rasmus", "Cecilie",
            "Henrik", "Katrine", "Mads", "Sofie", "Jakob", "Nanna",
        ]
        last_names = [
            "Nielsen", "Jensen", "Hansen", "Pedersen", "Andersen",
            "Christensen", "Larsen", "Sørensen", "Rasmussen", "Jørgensen",
            "Petersen", "Madsen", "Kristensen", "Olsen", "Thomsen",
            "Mortensen", "Poulsen", "Johansen", "Knudsen", "Møller",
        ]

        # Role
        role_names   = list(ROLES.keys())
        role_weights = [ROLES[r]["role_weight"] for r in role_names]
        self.role    = random.choices(role_names, weights=role_weights, k=1)[0]
        role_cfg     = ROLES[self.role]

        # Identity
        self.display_name = (
            f"{random.choice(first_names)} {random.choice(last_names)}"
        )
        slug = (self.display_name.lower()
                .replace(" ", ".")
                .replace("ø", "o").replace("æ", "ae").replace("å", "a"))
        self.upn     = f"{slug}.{user_id}@{ORG_DOMAIN}"
        self.user_id = self.upn
        self.dept    = random.choice(DEPARTMENTS[self.role])
        self.is_admin = self.role == "it_admin" or random.random() < 0.04

        # Location
        self.home_location   = random.choice(DK_LOCATIONS)
        self.office_location = random.choice(DK_LOCATIONS)
        self.known_ips = (
            [random_ip(self.home_location["ip_prefix"])
             for _ in range(random.randint(2, 3))]
            + [random_ip(self.office_location["ip_prefix"])]
        )

        # Devices — developers and IT admins have more
        n_devices = (random.randint(2, 3)
                     if self.role in ("developer", "it_admin")
                     else random.randint(1, 2))
        self.devices = random.sample(DEVICE_COMBOS, k=n_devices)

        # Work hours
        self.work_start = random.randint(WORK_HOURS_START, WORK_HOURS_START + 2)
        self.work_end   = random.randint(WORK_HOURS_END - 2, WORK_HOURS_END)

        # Activity level
        lo, hi = role_cfg["daily_range"]
        self.daily_events = random.randint(lo, hi)

        # Operation weights: role base + personal quirks
        self.op_weights = dict(role_cfg["base_ops"])

        # Boost 2-4 operations personally — creates distinct behavioural fingerprints
        boostable = [op for op in self.op_weights if self.op_weights[op] > 0]
        n_quirks  = random.randint(2, min(4, len(boostable)))
        for op in random.sample(boostable, k=n_quirks):
            self.op_weights[op] = int(
                self.op_weights[op] * random.uniform(1.5, 3.5)
            )

        # 15% chance of a cross-role quirk — e.g. a developer who is also
        # very active on email, or a finance user who uses Teams heavily
        if random.random() < 0.15:
            cross_op = random.choice([
                "MailItemsAccessed", "MessageSent", "Send",
                "FileDownloaded", "FileSyncDownloadedFull", "MeetingStarted",
            ])
            self.op_weights[cross_op] = (
                self.op_weights.get(cross_op, 0) + random.randint(20, 60)
            )

        # Role files
        self.files = ROLE_FILES[self.role]

        # Travel probability: 3% chance of being abroad on any given day
        self.travel_probability = 0.03

        # Holiday: 30% of users take one block of 3-10 days
        self.holiday_days: set[str] = set()
        if random.random() < 0.30:
            length = random.randint(3, 10)
            hstart = random_weekday(start, end - timedelta(days=length))
            for i in range(length):
                self.holiday_days.add(
                    (hstart + timedelta(days=i)).strftime("%Y-%m-%d")
                )

        # Sick leave: 15% of users have one episode of 1-3 days
        self.sick_days: set[str] = set()
        if random.random() < 0.15:
            length = random.randint(1, 3)
            sstart = random_weekday(start, end - timedelta(days=length))
            for i in range(length):
                self.sick_days.add(
                    (sstart + timedelta(days=i)).strftime("%Y-%m-%d")
                )

        # App preferences
        self.preferred_apps = {
            wl: random.choice(apps) for wl, apps in APP_MAP.items()
        }

    def is_working_hour(self, hour: int) -> bool:
        return self.work_start <= hour < self.work_end

    def typical_ip(self) -> str:
        return random.choice(self.known_ips)

    def typical_device(self) -> tuple:
        return random.choice(self.devices)

    def typical_location(self) -> dict:
        return random.choice([self.home_location, self.office_location])

    def absence_factor(self, dt: datetime) -> float:
        """Activity multiplier: 1.0=normal, 0.0=silent, 0.02=holiday trickle."""
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in PUBLIC_HOLIDAYS:     return 0.0
        if dt.weekday() >= 5:              return 0.05
        if date_str in self.holiday_days:  return 0.02
        if date_str in self.sick_days:     return 0.08
        return 1.0

    def travel_location(self, dt: datetime) -> dict | None:
        """Returns a travel location if the user is legitimately abroad today."""
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in self.holiday_days or date_str in self.sick_days:
            return None
        if random.random() < self.travel_probability:
            return random.choice(TRAVEL_LOCATIONS)
        return None


# ── Event sampler ─────────────────────────────────────────────────────────────

class EventSampler:
    def __init__(self, users: list[UserProfile]):
        self.users = users

    def sample(
        self,
        user: UserProfile,
        dt: datetime,
        operation: str | None = None,
        ip: str | None = None,
        location: dict | None = None,
        result_status: str | None = None,
        extra: dict | None = None,
        mobile: bool = False,
    ) -> dict:

        if operation is None:
            ops = dict(user.op_weights)
            if not user.is_working_hour(dt.hour):
                for op in ["FileAccessed", "FilePreviewed", "FileModified",
                           "FileModifiedExtended", "FileSyncDownloadedFull"]:
                    if op in ops:
                        ops[op] = max(0, ops[op] - 60)
            if mobile:
                ops = {
                    "MailItemsAccessed": 60,
                    "UserLoggedIn":      20,
                    "MessageSent":       15,
                    "FilePreviewed":     10,
                }
            operation = weighted_choice(ops)

        workload    = WORKLOAD_MAP.get(operation, "SharePoint")
        record_type = RECORD_TYPE_MAP[workload]
        os_, browser = user.typical_device()

        if ip is None:
            travel_loc = user.travel_location(dt)
            if travel_loc:
                ip       = random_ip(travel_loc["ip_prefix"])
                location = location or travel_loc
            else:
                ip = user.typical_ip() if random.random() < 0.78 else None

        if location is None:
            location = user.typical_location()

        user_type     = weighted_choice(USER_TYPE_WEIGHTS)
        result_status = result_status or (
            "Success" if random.random() < 0.23 else None
        )

        event = {
            "Id":             random_guid(),
            "CreationTime":   fmt_time(dt),
            "RecordType":     record_type,
            "Operation":      operation,
            "OrganizationId": ORG_ID,
            "UserType":       user_type,
            "UserKey":        user.upn,
            "UserId":         user.upn,
            "Workload":       workload,
        }

        if ip:                event["ClientIP"]       = ip
        if result_status:     event["ResultStatus"]   = result_status
        if user_type in (0,2): event["UserDisplayName"] = user.display_name

        if workload == "AzureActiveDirectory" and random.random() < 0.15:
            event["AppDisplayName"] = random.choice(APP_MAP[workload])
        elif workload != "AzureActiveDirectory" and random.random() < 0.03:
            event["AppDisplayName"] = random.choice(APP_MAP[workload])

        if random.random() < 0.05:
            event["DeviceProperties"] = [
                {"Name": "OS",          "Value": os_},
                {"Name": "BrowserType", "Value": browser},
                {"Name": "IsCompliant", "Value": "True"},
                {"Name": "IsManaged",   "Value": "True"},
            ]

        if random.random() < 0.08:
            event["Location"] = {
                "City":        location["city"],
                "Country":     location["country"],
                "CountryCode": location["countryCode"],
            }

        if random.random() < 0.23:
            event["ExtendedProperties"] = [
                {"Name": "UserAuthenticationMethod",
                 "Value": "MFA" if random.random() > 0.05 else "Password"},
            ]

        if workload in ("SharePoint", "OneDrive"):
            site  = random.choice(SP_SITES)
            fname = random.choice(user.files)
            event["ObjectId"]       = f"{site}/{fname}"
            event["SiteUrl"]        = site
            event["SourceFileName"] = fname
        elif workload == "Exchange":
            event["MailboxOwnerUPN"]  = user.upn
            event["ClientInfoString"] = random.choice(APP_MAP["Exchange"])
        elif workload == "MicrosoftTeams":
            event["TeamName"] = f"{user.dept} General"

        if extra:
            event.update(extra)

        return event


# ── Timestamp generation ──────────────────────────────────────────────────────

def _sample_hour(user: UserProfile, absence_factor: float) -> int:
    """Sample hour-of-day using a smooth activity curve respecting user hours."""
    ws, we = user.work_start, user.work_end

    if absence_factor <= 0.08:
        # Holiday/sick: random mobile-like hour
        weights = [3]*6 + [8]*3 + [4]*9 + [6]*3 + [3]*3
        return random.choices(range(24), weights=weights, k=1)[0]

    weights = []
    for h in range(24):
        if h < ws - 1:              w = 1
        elif h == ws - 1:           w = 15
        elif h == ws:               w = 60
        elif h == ws + 1:           w = 90
        elif h == ws + 2:           w = 100
        elif ws + 3 <= h < we - 2:  w = 70
        elif h == we - 2:           w = 55
        elif h == we - 1:           w = 45
        elif h == we:               w = 20
        elif we < h < we + 3:       w = 8
        else:                       w = 2
        weights.append(w)

    return random.choices(range(24), weights=weights, k=1)[0]


def generate_user_timestamps(
    user: UserProfile,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    """Generate all event timestamps for one user across the date range."""
    timestamps = []
    span_days  = (end - start).days

    for day_offset in range(span_days):
        day    = start + timedelta(days=day_offset)
        factor = user.absence_factor(day)
        if factor == 0.0:
            continue

        n_today = max(0, int(
            random.gauss(
                user.daily_events * factor,
                user.daily_events * factor * 0.2
            )
        ))

        for _ in range(n_today):
            hour   = _sample_hour(user, factor)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            dt     = day.replace(hour=hour, minute=minute, second=second)
            if start <= dt < end:
                timestamps.append(dt)

    return sorted(timestamps)


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_normal_dataset(
    sampler: EventSampler,
    users: list[UserProfile],
    n_target: int,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """
    Generate n_target normal events across all users.
    Each user's contribution is proportional to their daily_events rate,
    scaled so the total hits n_target.
    """
    # Estimate natural total events
    span_days     = (end - start).days
    natural_total = sum(u.daily_events * span_days for u in users)
    scale         = n_target / max(natural_total, 1)

    events = []
    for user in users:
        original_daily   = user.daily_events
        user.daily_events = max(1, int(user.daily_events * scale))

        timestamps = generate_user_timestamps(user, start, end)
        for dt in timestamps:
            factor = user.absence_factor(dt)
            mobile = 0.0 < factor < 0.1
            events.append(sampler.sample(user, dt, mobile=mobile))

        user.daily_events = original_daily

    events.sort(key=lambda e: e["CreationTime"])
    return events


def build_anomaly_test_dataset(
    sampler: EventSampler,
    injector: "AnomalyInjector",
    users: list[UserProfile],
    n: int,
    start: datetime,
    end: datetime,
) -> list[dict]:
    n_normal    = int(n * (1 - ANOMALY_RATIO))
    normal_evs  = build_normal_dataset(sampler, users, n_normal, start, end)

    anomaly_evs = []
    n_anomalies = n - n_normal
    span        = (end - start).total_seconds()
    for _ in range(n_anomalies):
        dt = start + timedelta(seconds=random.random() * span)
        anomaly_evs.extend(injector.random_anomaly(dt))

    all_events = normal_evs + anomaly_evs
    all_events.sort(key=lambda e: e["CreationTime"])
    return all_events


# ── Anomaly injector ──────────────────────────────────────────────────────────

class AnomalyInjector:
    def __init__(self, sampler: EventSampler, users: list[UserProfile]):
        self.sampler = sampler
        self.users   = users

    def _tag(self, event: dict, anomaly_type: str) -> dict:
        event["_anomaly"] = {"label": True, "type": anomaly_type}
        return event

    def impossible_travel(self, dt: datetime) -> list[dict]:
        user    = random.choice(self.users)
        dk_loc  = user.home_location
        for_loc = random.choice(FOREIGN_LOCATIONS)
        e1 = self.sampler.sample(user, dt, operation="UserLoggedIn",
                                  ip=random_ip(dk_loc["ip_prefix"]),
                                  location=dk_loc)
        e2 = self.sampler.sample(user, dt + timedelta(minutes=8),
                                  operation="UserLoggedIn",
                                  ip=random_ip(for_loc["ip_prefix"]),
                                  location=for_loc)
        return [e1, self._tag(e2, "impossible_travel")]

    def off_hours_admin(self, dt: datetime) -> list[dict]:
        admins = [u for u in self.users if u.is_admin] or self.users
        user   = random.choice(admins)
        night  = dt.replace(hour=3, minute=random.randint(0, 59))
        op     = random.choice(["Disable account", "Reset user password.",
                                 "Disable Strong Authentication"])
        return [self._tag(self.sampler.sample(user, night, operation=op),
                          "off_hours_admin")]

    def mass_download(self, dt: datetime) -> list[dict]:
        user   = random.choice(self.users)
        events = []
        for _ in range(random.randint(8, 12)):
            t = dt + timedelta(seconds=random.randint(0, 300))
            events.append(self._tag(
                self.sampler.sample(user, t, operation="FileDownloaded"),
                "mass_download"
            ))
        return events

    def mfa_disabled(self, dt: datetime) -> list[dict]:
        admins = [u for u in self.users if u.is_admin] or self.users
        actor  = random.choice(admins)
        target = random.choice([u for u in self.users if not u.is_admin])
        event  = self.sampler.sample(actor, dt,
                                     operation="Disable Strong Authentication")
        event["Target"] = [{"Type": 2, "ID": target.upn}]
        if "ExtendedProperties" not in event:
            event["ExtendedProperties"] = []
        event["ExtendedProperties"].append(
            {"Name": "targetUserDisplayName", "Value": target.display_name}
        )
        return [self._tag(event, "mfa_disabled")]

    def new_country_login(self, dt: datetime) -> list[dict]:
        user    = random.choice(self.users)
        for_loc = random.choice(FOREIGN_LOCATIONS)
        return [self._tag(
            self.sampler.sample(user, dt, operation="UserLoggedIn",
                                ip=random_ip(for_loc["ip_prefix"]),
                                location=for_loc),
            "new_country_login"
        )]

    def brute_force(self, dt: datetime) -> list[dict]:
        user    = random.choice(self.users)
        for_loc = random.choice(FOREIGN_LOCATIONS)
        ip      = random_ip(for_loc["ip_prefix"])
        events  = []
        for _ in range(random.randint(10, 20)):
            t = dt + timedelta(seconds=random.randint(0, 60))
            events.append(self._tag(
                self.sampler.sample(user, t, operation="UserLoginFailed",
                                    ip=ip, location=for_loc,
                                    result_status="Failed"),
                "brute_force"
            ))
        return events

    def random_anomaly(self, dt: datetime) -> list[dict]:
        return random.choice([
            self.impossible_travel, self.off_hours_admin,
            self.mass_download,     self.mfa_disabled,
            self.new_country_login, self.brute_force,
        ])(dt)


# ── Writer ────────────────────────────────────────────────────────────────────

def write_jsonl(events: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(events):>8,} events → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic M365 audit log generator v2"
    )
    parser.add_argument(
        "--train-only", action="store_true", dest="train_only",
        help=(
            "Generate train.jsonl only. Skips val.jsonl and anomaly_test.jsonl. "
            "Use when resuming — val/test tensors are already in the repo."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED}). Use a different seed to simulate "
             "a new tenant with different users and behaviour patterns."
    )
    parser.add_argument(
        "--n-users", type=int, default=N_USERS, dest="n_users",
        help=f"Number of synthetic users (default: {N_USERS}). "
             "Use 20-50 to simulate a small SMB tenant for fine-tuning tests."
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUT_DIR), dest="output_dir",
        help=f"Output directory for JSONL files (default: {OUT_DIR}). "
             "Use a separate dir when generating tenant test data."
    )
    args = parser.parse_args()

    # Apply CLI overrides to module-level constants
    random.seed(args.seed)
    n_users    = args.n_users
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scale event counts proportionally when n_users differs from default
    scale      = n_users / N_USERS
    n_train    = max(1000, int(N_TRAIN * scale))
    n_val      = max(200,  int(N_VAL   * scale))
    n_test     = N_ANOMALY_TEST   # keep test set size fixed

    print("=" * 60)
    print("  Synthetic M365 Audit Log Generator  v2")
    if args.train_only:
        print("  Mode: --train-only")
    if args.seed != SEED:
        print(f"  Seed: {args.seed}  (custom — simulating new tenant)")
    if n_users != N_USERS:
        print(f"  Users: {n_users}  (custom)")
    print("=" * 60)

    end   = datetime(2024, 12, 31, 23, 59, 59)
    start = end - timedelta(days=90)
    print(f"\n  Period:  {start.date()} -> {end.date()}  (90 days)")
    print(f"  Users:   {n_users}")
    print(f"  Domain:  {ORG_DOMAIN}")
    print(f"  Output:  {out_dir}")

    print("\nGenerating user profiles...")
    users = [UserProfile(i, start, end) for i in range(n_users)]

    role_counts = Counter(u.role for u in users)
    for role, count in sorted(role_counts.items()):
        avg_daily = sum(u.daily_events for u in users
                        if u.role == role) // max(count, 1)
        print(f"  {role:<12} {count:>4} users  (~{avg_daily} events/day avg)")

    holiday_users = sum(1 for u in users if u.holiday_days)
    sick_users    = sum(1 for u in users if u.sick_days)
    print(f"\n  {holiday_users} users have a holiday block")
    print(f"  {sick_users} users have a sick leave episode")

    sampler  = EventSampler(users)
    injector = AnomalyInjector(sampler, users)

    print(f"\nGenerating training set (target {n_train:,} events)...")
    train_events = build_normal_dataset(sampler, users, n_train, start, end)
    print(f"  Generated {len(train_events):,} events")

    if not args.train_only:
        print(f"\nGenerating validation set (target {n_val:,} events)...")
        val_events = build_normal_dataset(sampler, users, n_val, start, end)
        print(f"  Generated {len(val_events):,} events")

        print(f"\nGenerating anomaly test set ({n_test:,} events)...")
        test_events = build_anomaly_test_dataset(
            sampler, injector, users, n_test, start, end
        )
        n_anomalous = sum(1 for e in test_events if "_anomaly" in e)
        print(f"  Anomalous events: {n_anomalous} / {len(test_events)}")

    print("\nWriting JSONL files...")
    write_jsonl(train_events, out_dir / "train.jsonl")

    if not args.train_only:
        write_jsonl(val_events,  out_dir / "val.jsonl")
        write_jsonl(test_events, out_dir / "anomaly_test.jsonl")

        print("\nAnomaly breakdown in test set:")
        counts = Counter(
            e["_anomaly"]["type"] for e in test_events if "_anomaly" in e
        )
        for atype, count in sorted(counts.items()):
            print(f"  {atype:<25} {count:>4} events")

    print("\nDone. Data ready in", out_dir)


if __name__ == "__main__":
    main()