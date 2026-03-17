"""
generate_logs.py
Synthetic Microsoft 365 Unified Audit Log generator.

Produces three JSONL files that mirror the real M365 audit log schema:
  data/train.jsonl        — normal behaviour only  (~80k events)
  data/val.jsonl          — normal behaviour only  (~20k events)
  data/anomaly_test.jsonl — mixed normal + labeled anomalies (~2k events)

Each line is one JSON event.  Anomaly records carry an extra key:
  "_anomaly": {"label": true, "type": "impossible_travel"}

Architecture
------------
  UserProfile   — fixed attributes per synthetic user (home/office IPs,
                  typical hours, known devices, location, app usage weights)
  EventSampler  — draws realistic events for a given user + timestamp
  AnomalyInjector — wraps sampler, occasionally injects labeled anomalies

Anomaly types implemented
--------------------------
  impossible_travel   — login from DK then US within 10 minutes
  off_hours_admin     — privileged operation at 3 am
  mass_download       — 50+ file downloads in 5 minutes
  mfa_disabled        — MFA removed from an account
  new_country_login   — first-ever login from a foreign country
  brute_force         — 10+ failed logins in 60 seconds
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SEED            = 42
N_USERS         = 1000
N_TRAIN         = 1_000_000  # ~1000 events per user across 1000 users -- sufficient for convergence
N_VAL           = 200_000    # 10% of train
N_ANOMALY_TEST  = 50_000  # 1000 users x ~50 events, 0.1% anomaly rate = ~50 incidents
ANOMALY_RATIO   = 0.001  # 0.1% — realistic enterprise incident rate
OUT_DIR         = Path("data")

# Synthetic org identity
ORG_DOMAIN      = "contoso.com"
ORG_ID          = str(uuid.uuid4())
TENANT_COUNTRY  = "Denmark"
TENANT_CITY     = "Copenhagen"

# Office hours (local time, 24h)
WORK_HOURS_START = 7
WORK_HOURS_END   = 18

random.seed(SEED)

# ── Reference data ────────────────────────────────────────────────────────────

# Realistic M365 operations with frequency weights calibrated to real tenant data.
# SharePoint/OneDrive dominates (~65%), AAD ~20%, Exchange ~10%, Teams ~5%.
OPERATIONS = {
    # SharePoint / OneDrive file ops (dominant in real logs)
    "FileAccessed":                         200,
    "FilePreviewed":                         90,
    "FileModified":                          35,
    "FileModifiedExtended":                  15,
    "FileSyncDownloadedFull":                18,
    "FileDownloaded":                         8,
    "FileUploaded":                          12,
    "FileUploadedPartial":                   10,
    "FileDeleted":                            6,
    "FileMoved":                              4,
    "SharingSet":                             5,
    # Azure AD / sign-in
    "UserLoggedIn":                          40,
    "UserLoginFailed":                        4,
    "UserLoggedOut":                         12,
    "Update user.":                          25,
    "Update device.":                        18,
    "Update PasswordProfile.":               12,
    "Change user password.":                  6,
    "Reset user password.":                   3,
    "Consent to application.":                3,
    "Add delegated permission grant.":        3,
    "Set Company Information.":               3,
    "Add app role assignment grant to user.": 2,
    "Add service principal.":                 2,
    "Update StsRefreshTokenValidFrom Timestamp.": 9,
    # Anomaly-only ops — weight 0 so never drawn in normal sampling
    "Disable account":                        0,
    "Disable Strong Authentication":          0,
    # Exchange
    "MailItemsAccessed":                     28,
    "Send":                                  10,
    "MoveToDeletedItems":                     4,
    # Teams
    "MessageSent":                           10,
    "MeetingStarted":                         4,
}

# Workload mapping for operations.
# File ops split ~80/20 SharePoint/OneDrive to match real tenant data.
WORKLOAD_MAP = {
    "FileAccessed":                          "SharePoint",
    "FilePreviewed":                         "SharePoint",
    "FileModified":                          "SharePoint",
    "FileModifiedExtended":                  "SharePoint",
    "FileSyncDownloadedFull":                "OneDrive",
    "FileDownloaded":                        "SharePoint",
    "FileUploaded":                          "SharePoint",
    "FileUploadedPartial":                   "OneDrive",
    "FileDeleted":                           "SharePoint",
    "FileMoved":                             "SharePoint",
    "SharingSet":                            "SharePoint",
    "UserLoggedIn":                          "AzureActiveDirectory",
    "UserLoginFailed":                       "AzureActiveDirectory",
    "UserLoggedOut":                         "AzureActiveDirectory",
    "Update user.":                          "AzureActiveDirectory",
    "Update device.":                        "AzureActiveDirectory",
    "Update PasswordProfile.":               "AzureActiveDirectory",
    "Change user password.":                 "AzureActiveDirectory",
    "Reset user password.":                  "AzureActiveDirectory",
    "Consent to application.":               "AzureActiveDirectory",
    "Add delegated permission grant.":       "AzureActiveDirectory",
    "Set Company Information.":              "AzureActiveDirectory",
    "Add app role assignment grant to user.":"AzureActiveDirectory",
    "Add service principal.":               "AzureActiveDirectory",
    "Update StsRefreshTokenValidFrom Timestamp.": "AzureActiveDirectory",
    "Disable account":                       "AzureActiveDirectory",
    "Disable Strong Authentication":         "AzureActiveDirectory",
    "MailItemsAccessed":                     "Exchange",
    "Send":                                  "Exchange",
    "MoveToDeletedItems":                    "Exchange",
    "MessageSent":                           "MicrosoftTeams",
    "MeetingStarted":                        "MicrosoftTeams",
}

# RecordType values per workload (official M365 values).
# OneDrive shares RecordType 6 with SharePoint.
RECORD_TYPE_MAP = {
    "AzureActiveDirectory": 15,
    "SharePoint":            6,
    "OneDrive":              6,
    "Exchange":              2,
    "MicrosoftTeams":       25,
}

# App display names per workload.
# Made optional in sampler — real logs often omit AppDisplayName.
APP_MAP = {
    "AzureActiveDirectory": [
        "Microsoft 365 portal", "Azure portal", "Microsoft Authenticator",
        "Microsoft Teams", "Microsoft Outlook",
    ],
    "SharePoint": [
        "SharePoint Online", "OneDrive for Business", "Microsoft Teams",
    ],
    "OneDrive": [
        "OneDrive for Business", "Microsoft Teams",
    ],
    "Exchange": [
        "Microsoft Outlook", "Microsoft Teams", "Outlook Web App",
    ],
    "MicrosoftTeams": [
        "Microsoft Teams",
    ],
}

# UserType values seen in real logs:
#   0 = Regular user, 2 = Admin, 4 = System account, 5 = Application
USER_TYPE_WEIGHTS = {0: 65, 2: 8, 4: 20, 5: 7}

# Browser / OS combinations
DEVICE_COMBOS = [
    ("Windows 11", "Edge"),
    ("Windows 10", "Edge"),
    ("Windows 10", "Chrome"),
    ("macOS 14", "Safari"),
    ("macOS 13", "Chrome"),
    ("iOS 17",   "Safari"),
    ("Android 13","Chrome"),
]

# Danish cities and their rough IP ranges (fictional but realistic)
DK_LOCATIONS = [
    {"city": "Copenhagen", "country": "Denmark", "countryCode": "DK",
     "ip_prefix": "185.20"},
    {"city": "Aarhus",     "country": "Denmark", "countryCode": "DK",
     "ip_prefix": "195.249"},
    {"city": "Odense",     "country": "Denmark", "countryCode": "DK",
     "ip_prefix": "80.167"},
    {"city": "Aalborg",    "country": "Denmark", "countryCode": "DK",
     "ip_prefix": "91.208"},
]

# Foreign locations used in anomalies
FOREIGN_LOCATIONS = [
    {"city": "Moscow",        "country": "Russia",        "countryCode": "RU",
     "ip_prefix": "91.108"},
    {"city": "Beijing",       "country": "China",         "countryCode": "CN",
     "ip_prefix": "118.26"},
    {"city": "Lagos",         "country": "Nigeria",       "countryCode": "NG",
     "ip_prefix": "197.255"},
    {"city": "Bucharest",     "country": "Romania",       "countryCode": "RO",
     "ip_prefix": "79.112"},
    {"city": "Kyiv",          "country": "Ukraine",       "countryCode": "UA",
     "ip_prefix": "91.200"},
]

# SharePoint site/file paths
SP_SITES = [
    "https://contoso.sharepoint.com/sites/Finance",
    "https://contoso.sharepoint.com/sites/HR",
    "https://contoso.sharepoint.com/sites/IT",
    "https://contoso.sharepoint.com/sites/Sales",
    "https://contoso-my.sharepoint.com/personal",
]

SP_FILES = [
    "Q4-Report.xlsx", "Budget-2024.xlsx", "Employee-Handbook.pdf",
    "Client-Contracts.docx", "IT-Policy.pdf", "Sales-Pipeline.xlsx",
    "Meeting-Notes.docx", "Project-Plan.xlsx", "Audit-Report.pdf",
    "Customer-Data.csv",
]

DEPARTMENTS = [
    "Finance", "HR", "IT", "Sales", "Marketing",
    "Operations", "Legal", "Engineering", "Management",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def random_ip(prefix: str) -> str:
    return f"{prefix}.{random.randint(1,254)}.{random.randint(1,254)}"

def random_guid() -> str:
    return str(uuid.uuid4())

def fmt_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def weighted_choice(d: dict):
    keys   = [k for k, v in d.items() if v > 0]
    weights = [d[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


# ── User profiles ─────────────────────────────────────────────────────────────

class UserProfile:
    """Fixed behavioural attributes for one synthetic user."""

    def __init__(self, user_id: int):
        first_names = [
            "Oliver", "Emma", "Noah", "Sophia", "Liam", "Mia",
            "Lucas", "Isabella", "Ethan", "Charlotte", "Magnus",
            "Astrid", "Frederik", "Maja", "Christian", "Laura",
            "Thomas", "Anna", "Mikkel", "Sara", "Jonas", "Ida",
            "Kasper", "Julie", "Andreas", "Louise", "Martin", "Camilla",
            "Søren", "Line", "Peter", "Maria",
        ]
        last_names = [
            "Nielsen", "Jensen", "Hansen", "Pedersen", "Andersen",
            "Christensen", "Larsen", "Sørensen", "Rasmussen", "Jørgensen",
            "Petersen", "Madsen", "Kristensen", "Olsen", "Thomsen",
        ]
        self.display_name = (
            f"{random.choice(first_names)} {random.choice(last_names)}"
        )
        slug = self.display_name.lower().replace(" ", ".").replace("ø","o").replace("æ","ae").replace("å","a")
        # Add user_id suffix to guarantee unique UPNs across 1000 users
        self.upn     = f"{slug}.{user_id}@{ORG_DOMAIN}"
        self.user_id = self.upn
        self.dept    = random.choice(DEPARTMENTS)
        self.is_admin = random.random() < 0.08   # ~8% of users are admins

        # Location: one primary DK city
        self.home_location = random.choice(DK_LOCATIONS)
        self.office_location = random.choice(DK_LOCATIONS)

        # Typical IPs: 2-3 known home/mobile IPs + 1 office subnet
        self.known_ips = [
            random_ip(self.home_location["ip_prefix"])
            for _ in range(random.randint(2, 3))
        ] + [random_ip(self.office_location["ip_prefix"])]

        # Device: 1-2 known devices
        self.devices = random.sample(DEVICE_COMBOS, k=random.randint(1, 2))

        # Work hours: slight variation per person
        self.work_start = random.randint(WORK_HOURS_START, WORK_HOURS_START + 2)
        self.work_end   = random.randint(WORK_HOURS_END - 2, WORK_HOURS_END)

        # App preferences by workload
        self.preferred_apps = {
            wl: random.choice(apps)
            for wl, apps in APP_MAP.items()
        }

    def is_working_hour(self, hour: int) -> bool:
        return self.work_start <= hour < self.work_end

    def typical_ip(self) -> str:
        return random.choice(self.known_ips)

    def typical_device(self) -> tuple:
        return random.choice(self.devices)

    def typical_location(self) -> dict:
        return random.choice([self.home_location, self.office_location])


# ── Event sampler ─────────────────────────────────────────────────────────────

class EventSampler:
    """Generates realistic single M365 audit events."""

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
    ) -> dict:
        """
        Build one M365-schema JSON event.

        Field sparsity is calibrated to real tenant observations:
          ClientIP          ~78% present
          ResultStatus      ~23% present (omitted for most file/background ops)
          AppDisplayName    ~0%  present in AAD ops, sparse elsewhere
          DeviceProperties  ~0%  present (omitted by most workloads)
          Location          ~0%  present (omitted by most workloads)
          ExtendedProperties ~23% present
        """

        if operation is None:
            ops = dict(OPERATIONS)
            if not user.is_working_hour(dt.hour):
                # Outside work hours: reduce file and mail activity
                for op in ["FileAccessed", "FilePreviewed", "FileModified",
                           "FileModifiedExtended", "MailItemsAccessed",
                           "Send", "MessageSent"]:
                    ops[op] = max(0, ops.get(op, 0) - 30)
            operation = weighted_choice(ops)

        workload    = WORKLOAD_MAP[operation]
        record_type = RECORD_TYPE_MAP[workload]
        os_, browser = user.typical_device()

        if ip is None:
            # ~78% of real events have a ClientIP
            ip = user.typical_ip() if random.random() < 0.78 else None
        if location is None:
            location = user.typical_location()

        # UserType: draw from real distribution (0=user, 2=admin, 4=system, 5=app)
        user_type = weighted_choice(USER_TYPE_WEIGHTS)

        # ResultStatus: ~23% of real events include this field.
        # When present, real tenant uses "Success" not "Succeeded".
        if result_status is None:
            if random.random() < 0.23:
                result_status = "Success"
            # else omit entirely

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

        # Sparse fields — only include when present
        if ip:
            event["ClientIP"] = ip

        if result_status:
            event["ResultStatus"] = result_status

        # AppDisplayName: present on ~0% of real AAD events, sparse elsewhere.
        # Include ~15% of the time for AAD sign-ins, almost never otherwise.
        if workload == "AzureActiveDirectory" and random.random() < 0.15:
            event["AppDisplayName"] = random.choice(APP_MAP[workload])
        elif workload not in ("AzureActiveDirectory",) and random.random() < 0.03:
            event["AppDisplayName"] = random.choice(APP_MAP[workload])

        # UserDisplayName: always present for human users
        if user_type in (0, 2):
            event["UserDisplayName"] = user.display_name

        # DeviceProperties: rare in real logs (~0%) — include ~5% of the time
        if random.random() < 0.05:
            event["DeviceProperties"] = [
                {"Name": "OS",          "Value": os_},
                {"Name": "BrowserType", "Value": browser},
                {"Name": "IsCompliant", "Value": "True"},
                {"Name": "IsManaged",   "Value": "True"},
            ]

        # Location: rare in real logs (~0%) — include ~8% of the time
        if random.random() < 0.08:
            event["Location"] = {
                "City":        location["city"],
                "Country":     location["country"],
                "CountryCode": location["countryCode"],
            }

        # ExtendedProperties: ~23% of real events
        if random.random() < 0.23:
            event["ExtendedProperties"] = [
                {"Name": "UserAuthenticationMethod",
                 "Value": "MFA" if random.random() > 0.05 else "Password"},
            ]

        # Workload-specific fields
        if workload in ("SharePoint", "OneDrive"):
            site = random.choice(SP_SITES)
            file = random.choice(SP_FILES)
            event["ObjectId"]       = f"{site}/{file}"
            event["SiteUrl"]        = site
            event["SourceFileName"] = file

        elif workload == "Exchange":
            event["MailboxOwnerUPN"]  = user.upn
            event["ClientInfoString"] = random.choice(APP_MAP["Exchange"])

        elif workload == "MicrosoftTeams":
            event["TeamName"] = f"{user.dept} General"

        if extra:
            event.update(extra)

        return event


# ── Timestamp generator ───────────────────────────────────────────────────────

def generate_timestamps(n: int, start: datetime, end: datetime) -> list[datetime]:
    """
    Generate n timestamps calibrated to real tenant observations:
      Morning (07-09):  ~59%  -- sync clients, overnight batch catch-up
      Work    (10-17):  ~36%
      Evening (18-23):   ~4%
      Night   (00-06):   ~2%
    Weekends: ~5% of weekday volume.
    """
    timestamps = []
    span = (end - start).total_seconds()

    while len(timestamps) < n:
        t       = start + timedelta(seconds=random.random() * span)
        weekday = t.weekday()
        hour    = t.hour

        if weekday >= 5:
            p = 0.05
        elif 7 <= hour < 10:    # morning peak (sync clients, overnight catch-up)
            p = 1.0
        elif 10 <= hour < 18:   # normal work hours
            p = 0.55
        elif 18 <= hour < 23:   # evening
            p = 0.06
        else:                   # night
            p = 0.02

        if random.random() < p:
            timestamps.append(t)

    return sorted(timestamps)


# ── Anomaly injector ──────────────────────────────────────────────────────────

class AnomalyInjector:
    """
    Generates labeled anomalous events.
    Each method returns a list of events (some anomalies span multiple records).
    """

    def __init__(self, sampler: EventSampler, users: list[UserProfile]):
        self.sampler = sampler
        self.users   = users

    def _tag(self, event: dict, anomaly_type: str) -> dict:
        event["_anomaly"] = {"label": True, "type": anomaly_type}
        return event

    def impossible_travel(self, dt: datetime) -> list[dict]:
        """
        User logs in from Denmark, then 8 minutes later from a foreign country.
        Physically impossible — flags as credential compromise.
        """
        user      = random.choice(self.users)
        dk_loc    = user.home_location
        for_loc   = random.choice(FOREIGN_LOCATIONS)
        dk_ip     = random_ip(dk_loc["ip_prefix"])
        for_ip    = random_ip(for_loc["ip_prefix"])

        e1 = self.sampler.sample(user, dt,
                                  operation="UserLoggedIn",
                                  ip=dk_ip, location=dk_loc)

        e2 = self.sampler.sample(user, dt + timedelta(minutes=8),
                                  operation="UserLoggedIn",
                                  ip=for_ip, location=for_loc)
        self._tag(e2, "impossible_travel")
        return [e1, e2]

    def off_hours_admin(self, dt: datetime) -> list[dict]:
        """
        Admin performs privileged operation at 3 am.
        Could indicate account takeover or insider threat.
        """
        admins = [u for u in self.users if u.is_admin]
        if not admins:
            admins = self.users

        user    = random.choice(admins)
        night   = dt.replace(hour=3, minute=random.randint(0, 59))
        op      = random.choice(["Disable account", "Reset user password.",
                                  "Disable Strong Authentication"])
        event   = self.sampler.sample(user, night, operation=op)
        return [self._tag(event, "off_hours_admin")]

    def mass_download(self, dt: datetime) -> list[dict]:
        """
        User downloads 50+ files within 5 minutes.
        Flags potential data exfiltration.
        """
        user   = random.choice(self.users)
        events = []
        for i in range(random.randint(8, 12)):
            t     = dt + timedelta(seconds=random.randint(0, 300))
            event = self.sampler.sample(user, t, operation="FileDownloaded")
            self._tag(event, "mass_download")
            events.append(event)
        return events

    def mfa_disabled(self, dt: datetime) -> list[dict]:
        """
        MFA authentication method removed from a user account.
        High-severity security event.
        """
        # Actor is an admin, target is a regular user
        admins = [u for u in self.users if u.is_admin] or self.users
        actor  = random.choice(admins)
        target = random.choice([u for u in self.users if not u.is_admin])

        event = self.sampler.sample(actor, dt,
                                    operation="Disable Strong Authentication")
        event["Target"] = [{"Type": 2, "ID": target.upn}]
        if "ExtendedProperties" not in event:
            event["ExtendedProperties"] = []
        event["ExtendedProperties"].append(
            {"Name": "targetUserDisplayName", "Value": target.display_name}
        )
        return [self._tag(event, "mfa_disabled")]

    def new_country_login(self, dt: datetime) -> list[dict]:
        """
        User logs in from a country they have never used before.
        """
        user    = random.choice(self.users)
        for_loc = random.choice(FOREIGN_LOCATIONS)
        for_ip  = random_ip(for_loc["ip_prefix"])
        event   = self.sampler.sample(user, dt,
                                       operation="UserLoggedIn",
                                       ip=for_ip, location=for_loc)
        return [self._tag(event, "new_country_login")]

    def brute_force(self, dt: datetime) -> list[dict]:
        """
        10+ failed logins from the same IP within 60 seconds.
        Classic credential-stuffing or brute-force pattern.
        """
        user   = random.choice(self.users)
        ip     = random_ip(random.choice(FOREIGN_LOCATIONS)["ip_prefix"])
        loc    = random.choice(FOREIGN_LOCATIONS)
        events = []
        for i in range(random.randint(10, 20)):
            t     = dt + timedelta(seconds=random.randint(0, 60))
            event = self.sampler.sample(user, t,
                                        operation="UserLoginFailed",
                                        ip=ip, location=loc,
                                        result_status="Failed")
            self._tag(event, "brute_force")
            events.append(event)
        return events

    def random_anomaly(self, dt: datetime) -> list[dict]:
        method = random.choice([
            self.impossible_travel,
            self.off_hours_admin,
            self.mass_download,
            self.mfa_disabled,
            self.new_country_login,
            self.brute_force,
        ])
        return method(dt)


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_normal_dataset(
    sampler: EventSampler,
    users: list[UserProfile],
    n: int,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Generate n normal events distributed across users and timestamps."""
    timestamps = generate_timestamps(n, start, end)
    events     = []
    for dt in timestamps:
        user  = random.choice(users)
        event = sampler.sample(user, dt)
        events.append(event)
    return events


def build_anomaly_test_dataset(
    sampler: EventSampler,
    injector: AnomalyInjector,
    users: list[UserProfile],
    n: int,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """
    ~90% normal events, ~10% anomalous events.
    All events interleaved chronologically so the model
    sees a realistic mixed stream.
    """
    n_normal   = int(n * (1 - ANOMALY_RATIO))
    normal_evs = build_normal_dataset(sampler, users, n_normal, start, end)

    # Generate anomalies
    anomaly_evs = []
    n_anomalies = n - n_normal
    anomaly_timestamps = generate_timestamps(n_anomalies, start, end)
    for dt in anomaly_timestamps:
        anomaly_evs.extend(injector.random_anomaly(dt))

    # Merge + sort
    all_events = normal_evs + anomaly_evs
    all_events.sort(key=lambda e: e["CreationTime"])
    return all_events


# ── Writer ────────────────────────────────────────────────────────────────────

def write_jsonl(events: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(events):>8,} events → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Synthetic M365 Audit Log Generator")
    print("=" * 60)

    # Date range: 90 days of activity (matches M365 default retention)
    end   = datetime(2024, 12, 31, 23, 59, 59)
    start = end - timedelta(days=90)
    print(f"\n  Period:  {start.date()} → {end.date()}  (90 days)")
    print(f"  Users:   {N_USERS}")
    print(f"  Domain:  {ORG_DOMAIN}")

    # Build user pool
    print("\nGenerating user profiles...")
    users = [UserProfile(i) for i in range(N_USERS)]
    admins = sum(1 for u in users if u.is_admin)
    print(f"  {N_USERS} users  ({admins} admins)")

    sampler  = EventSampler(users)
    injector = AnomalyInjector(sampler, users)

    # Training set (normal only)
    print(f"\nGenerating training set ({N_TRAIN:,} events)...")
    train_events = build_normal_dataset(sampler, users, N_TRAIN, start, end)

    # Validation set (normal only)
    print(f"Generating validation set ({N_VAL:,} events)...")
    val_events = build_normal_dataset(sampler, users, N_VAL, start, end)

    # Anomaly test set
    print(f"Generating anomaly test set ({N_ANOMALY_TEST:,} events)...")
    test_events = build_anomaly_test_dataset(
        sampler, injector, users, N_ANOMALY_TEST, start, end
    )
    n_anomalous = sum(1 for e in test_events if "_anomaly" in e)
    print(f"  Anomalous events: {n_anomalous} / {len(test_events)}")

    # Write
    print("\nWriting JSONL files...")
    write_jsonl(train_events, OUT_DIR / "train.jsonl")
    write_jsonl(val_events,   OUT_DIR / "val.jsonl")
    write_jsonl(test_events,  OUT_DIR / "anomaly_test.jsonl")

    # Summary
    print("\nAnomaly breakdown in test set:")
    from collections import Counter
    counts = Counter(
        e["_anomaly"]["type"]
        for e in test_events
        if "_anomaly" in e
    )
    for atype, count in sorted(counts.items()):
        print(f"  {atype:<25} {count:>4} events")

    print("\nDone. Data ready in data/")


if __name__ == "__main__":
    main()