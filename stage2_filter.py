"""
stage2_filter.py
Rule-based anomaly detection for LogSentinel.

Three independent rules, each scanning raw M365 audit log events:

  Rule 1 — mass_download  : >= BURST_THRESHOLD FileDownloaded from same user
                             within BURST_WINDOW_MINUTES
  Rule 2 — brute_force    : >= BRUTE_THRESHOLD UserLoginFailed from same IP
                             within BRUTE_WINDOW_SECONDS
  Rule 3 — mfa_disabled   : any occurrence of "Disable Strong Authentication"

No transformer scores required. Works directly on raw JSONL events.

Usage:
    python stage2_filter.py
    python stage2_filter.py --events-file data/tenant_test/anomaly_test.jsonl
    python stage2_filter.py --events-file data/tenant_real/logs.jsonl --out alerts.json
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ── Operation sets ─────────────────────────────────────────────────────────────

DOWNLOAD_OPS = {
    "FileDownloaded",
    "FileSyncDownloadedFull",
    "FileSyncDownloadedPartial",
    "FileAccessed",
}

BRUTE_FORCE_OP  = "UserLoginFailed"
MFA_DISABLED_OP = "Disable Strong Authentication"

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_EVENTS_FILE        = "data/tenant_test/anomaly_test.jsonl"
DEFAULT_BURST_WINDOW       = 5      # minutes
DEFAULT_BURST_THRESHOLD    = 8      # downloads in window
DEFAULT_BRUTE_WINDOW       = 60     # seconds
DEFAULT_BRUTE_THRESHOLD    = 10     # failed logins from same IP in window


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ── Rule 1: Mass download burst ────────────────────────────────────────────────

def detect_mass_download(events, window_minutes, threshold):
    """
    Flag users with >= threshold download events within any window_minutes window.
    Returns list of alert dicts.
    """
    window = timedelta(minutes=window_minutes)

    user_downloads = defaultdict(list)
    for e in events:
        if e.get("Operation") in DOWNLOAD_OPS:
            user_downloads[e.get("UserId", "unknown")].append(e)

    alerts = []
    for user_id, dl_events in user_downloads.items():
        try:
            dl_events.sort(key=lambda e: parse_time(e["CreationTime"]))
        except (KeyError, ValueError):
            continue

        i = 0
        while i < len(dl_events):
            anchor_time = parse_time(dl_events[i]["CreationTime"])
            window_end  = anchor_time + window
            burst_events = [
                e for e in dl_events[i:]
                if parse_time(e["CreationTime"]) <= window_end
            ]
            if len(burst_events) >= threshold:
                is_tp = any("_anomaly" in e for e in burst_events)
                alerts.append({
                    "rule":        "mass_download",
                    "user_id":     user_id,
                    "start":       dl_events[i]["CreationTime"],
                    "end":         burst_events[-1]["CreationTime"],
                    "event_count": len(burst_events),
                    "detail":      f"{len(burst_events)} downloads in {window_minutes}m",
                    "is_true_positive": is_tp,
                })
                i += len(burst_events)
            else:
                i += 1

    alerts.sort(key=lambda a: a["start"])
    return alerts


# ── Rule 2: Brute force login ──────────────────────────────────────────────────

def detect_brute_force(events, window_seconds, threshold):
    """
    Flag source IPs with >= threshold UserLoginFailed events within window_seconds.
    Returns list of alert dicts.
    """
    window = timedelta(seconds=window_seconds)

    ip_failures = defaultdict(list)
    for e in events:
        if e.get("Operation") == BRUTE_FORCE_OP:
            ip = e.get("ClientIP", "unknown")
            ip_failures[ip].append(e)

    alerts = []
    for ip, fail_events in ip_failures.items():
        try:
            fail_events.sort(key=lambda e: parse_time(e["CreationTime"]))
        except (KeyError, ValueError):
            continue

        i = 0
        while i < len(fail_events):
            anchor_time = parse_time(fail_events[i]["CreationTime"])
            window_end  = anchor_time + window
            burst_events = [
                e for e in fail_events[i:]
                if parse_time(e["CreationTime"]) <= window_end
            ]
            if len(burst_events) >= threshold:
                is_tp = any("_anomaly" in e for e in burst_events)
                users = {e.get("UserId", "unknown") for e in burst_events}
                alerts.append({
                    "rule":        "brute_force",
                    "source_ip":   ip,
                    "start":       fail_events[i]["CreationTime"],
                    "end":         burst_events[-1]["CreationTime"],
                    "event_count": len(burst_events),
                    "detail":      f"{len(burst_events)} failed logins from {ip} in {window_seconds}s targeting {len(users)} user(s)",
                    "is_true_positive": is_tp,
                })
                i += len(burst_events)
            else:
                i += 1

    alerts.sort(key=lambda a: a["start"])
    return alerts


# ── Rule 3: MFA disabled ───────────────────────────────────────────────────────

def detect_mfa_disabled(events):
    """
    Flag every occurrence of MFA being disabled on an account.
    Each event is its own alert — no window required.
    Returns list of alert dicts.
    """
    alerts = []
    for e in events:
        if e.get("Operation") == MFA_DISABLED_OP:
            is_tp = "_anomaly" in e
            alerts.append({
                "rule":        "mfa_disabled",
                "user_id":     e.get("UserId", "unknown"),
                "start":       e.get("CreationTime", ""),
                "end":         e.get("CreationTime", ""),
                "event_count": 1,
                "detail":      f"MFA disabled on {e.get('UserId', 'unknown')}",
                "is_true_positive": is_tp,
            })
    alerts.sort(key=lambda a: a["start"])
    return alerts


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LogSentinel Stage 2 — Rule-Based Detection"
    )
    parser.add_argument("--events-file", default=DEFAULT_EVENTS_FILE)
    parser.add_argument("--burst-window",    type=int, default=DEFAULT_BURST_WINDOW,
                        help=f"Mass download window in minutes (default: {DEFAULT_BURST_WINDOW})")
    parser.add_argument("--burst-threshold", type=int, default=DEFAULT_BURST_THRESHOLD,
                        help=f"Min downloads to trigger burst alert (default: {DEFAULT_BURST_THRESHOLD})")
    parser.add_argument("--brute-window",    type=int, default=DEFAULT_BRUTE_WINDOW,
                        help=f"Brute force window in seconds (default: {DEFAULT_BRUTE_WINDOW})")
    parser.add_argument("--brute-threshold", type=int, default=DEFAULT_BRUTE_THRESHOLD,
                        help=f"Min failed logins to trigger brute force alert (default: {DEFAULT_BRUTE_THRESHOLD})")
    parser.add_argument("--out", default=None,
                        help="Output JSON file (default: <events_dir>/stage2_alerts.json)")
    args = parser.parse_args()

    events_path = Path(args.events_file)
    out_path    = Path(args.out) if args.out else events_path.parent / "stage2_alerts.json"

    print("=" * 66)
    print("  LogSentinel Stage 2 — Rule-Based Detection")
    print("=" * 66)
    print(f"\n  Events file:  {events_path}")

    events = load_jsonl(events_path)
    n_users = len({e.get("UserId") for e in events})
    print(f"  Loaded {len(events):,} events across {n_users} users\n")

    # ── Run all rules ─────────────────────────────────────────────────────────
    md_alerts  = detect_mass_download(events, args.burst_window, args.burst_threshold)
    bf_alerts  = detect_brute_force(events, args.brute_window, args.brute_threshold)
    mfa_alerts = detect_mfa_disabled(events)

    all_alerts = md_alerts + bf_alerts + mfa_alerts
    all_alerts.sort(key=lambda a: a["start"])

    # ── Per-rule summary ──────────────────────────────────────────────────────
    has_labels = any("_anomaly" in e for e in events)

    rule_groups = [
        ("mass_download", md_alerts),
        ("brute_force",   bf_alerts),
        ("mfa_disabled",  mfa_alerts),
    ]

    print(f"{'─' * 66}")
    print(f"  {'Rule':<20} {'Alerts':>7}  {'TP':>4}  {'FP':>4}  {'Recall note'}")
    print(f"{'─' * 66}")

    for rule_name, alerts in rule_groups:
        n_alerts = len(alerts)
        tp = sum(1 for a in alerts if a["is_true_positive"])
        fp = sum(1 for a in alerts if not a["is_true_positive"])

        if has_labels:
            labeled_users = {
                e.get("UserId") for e in events
                if e.get("_anomaly", {}).get("type") == rule_name
            }
            if rule_name == "brute_force":
                # brute force alerts are per-IP not per-user
                detected_users = {
                    e.get("UserId") for a in alerts if a["is_true_positive"]
                    for e in events
                    if e.get("ClientIP") == a.get("source_ip") and "_anomaly" in e
                }
            else:
                detected_users = {
                    a.get("user_id") for a in alerts if a["is_true_positive"]
                }
            missed = labeled_users - detected_users
            recall_note = f"{len(detected_users)}/{len(labeled_users)} users" if labeled_users else "no labels"
        else:
            recall_note = ""

        print(f"  {rule_name:<20} {n_alerts:>7}  {tp:>4}  {fp:>4}  {recall_note}")

    print(f"{'─' * 66}")
    total_tp = sum(1 for a in all_alerts if a["is_true_positive"])
    total_fp = sum(1 for a in all_alerts if not a["is_true_positive"])
    print(f"  {'TOTAL':<20} {len(all_alerts):>7}  {total_tp:>4}  {total_fp:>4}")
    print(f"{'─' * 66}")

    # ── Alert detail ──────────────────────────────────────────────────────────
    if all_alerts:
        print(f"\n  Alert detail:")
        for a in all_alerts:
            tp_marker = " [TP]" if a["is_true_positive"] else " [FP]"
            label = a.get("user_id") or a.get("source_ip", "?")
            print(f"  [{a['rule']:<14}]  {label:<35} {a['start'][:19]}  {a['detail']}{tp_marker if has_labels else ''}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "config": {
            "events_file":     str(events_path),
            "burst_window_m":  args.burst_window,
            "burst_threshold": args.burst_threshold,
            "brute_window_s":  args.brute_window,
            "brute_threshold": args.brute_threshold,
        },
        "summary": {
            "total_alerts":  len(all_alerts),
            "total_tp":      total_tp,
            "total_fp":      total_fp,
            "by_rule": {
                name: {
                    "alerts": len(alerts),
                    "tp": sum(1 for a in alerts if a["is_true_positive"]),
                    "fp": sum(1 for a in alerts if not a["is_true_positive"]),
                }
                for name, alerts in rule_groups
            },
        },
        "alerts": [{k: v for k, v in a.items()} for a in all_alerts],
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Alerts saved → {out_path}")
    print("=" * 66)


if __name__ == "__main__":
    main()
