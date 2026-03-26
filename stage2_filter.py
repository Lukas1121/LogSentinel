"""
stage2_filter.py
Rule-based anomaly detection for LogSentinel.

One rule targeting the model's specific blind spot:

  Rule 1 — mass_download  : >= BURST_THRESHOLD download events from same user
                             within BURST_WINDOW_MINUTES

The transformer model handles brute force, MFA changes, impossible travel,
off-hours admin, and new country logins at high recall. This rule covers
bulk file exfiltration, which the model consistently misses.

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

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_EVENTS_FILE     = "data/tenant_test/anomaly_test.jsonl"
DEFAULT_BURST_WINDOW    = 5      # minutes
DEFAULT_BURST_THRESHOLD = 8      # downloads in window


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ── Rule: Mass download burst ──────────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LogSentinel Stage 2 — Mass Download Detection"
    )
    parser.add_argument("--events-file", default=DEFAULT_EVENTS_FILE)
    parser.add_argument("--burst-window",    type=int, default=DEFAULT_BURST_WINDOW,
                        help=f"Download burst window in minutes (default: {DEFAULT_BURST_WINDOW})")
    parser.add_argument("--burst-threshold", type=int, default=DEFAULT_BURST_THRESHOLD,
                        help=f"Min downloads to trigger alert (default: {DEFAULT_BURST_THRESHOLD})")
    parser.add_argument("--out", default=None,
                        help="Output JSON file (default: <events_dir>/stage2_alerts.json)")
    args = parser.parse_args()

    events_path = Path(args.events_file)
    out_path    = Path(args.out) if args.out else events_path.parent / "stage2_alerts.json"

    print("=" * 66)
    print("  LogSentinel Stage 2 — Mass Download Detection")
    print("=" * 66)
    print(f"\n  Events file:  {events_path}")

    events = load_jsonl(events_path)
    n_users = len({e.get("UserId") for e in events})
    print(f"  Loaded {len(events):,} events across {n_users} users\n")

    alerts = detect_mass_download(events, args.burst_window, args.burst_threshold)

    has_labels = any("_anomaly" in e for e in events)
    tp = sum(1 for a in alerts if a["is_true_positive"])
    fp = sum(1 for a in alerts if not a["is_true_positive"])

    print(f"{'─' * 66}")
    print(f"  {'Rule':<20} {'Alerts':>7}  {'TP':>4}  {'FP':>4}  {'Recall note'}")
    print(f"{'─' * 66}")

    if has_labels:
        labeled_users = {
            e.get("UserId") for e in events
            if e.get("_anomaly", {}).get("type") == "mass_download"
        }
        detected_users = {a.get("user_id") for a in alerts if a["is_true_positive"]}
        recall_note = f"{len(detected_users)}/{len(labeled_users)} users"
    else:
        recall_note = ""

    print(f"  {'mass_download':<20} {len(alerts):>7}  {tp:>4}  {fp:>4}  {recall_note}")
    print(f"{'─' * 66}")

    if alerts:
        print(f"\n  Alert detail:")
        for a in alerts:
            tp_marker = " [TP]" if a["is_true_positive"] else " [FP]"
            print(f"  [{a['rule']:<14}]  {a['user_id']:<35} {a['start'][:19]}  {a['detail']}{tp_marker if has_labels else ''}")

    output = {
        "config": {
            "events_file":     str(events_path),
            "burst_window_m":  args.burst_window,
            "burst_threshold": args.burst_threshold,
        },
        "summary": {
            "total_alerts": len(alerts),
            "total_tp":     tp,
            "total_fp":     fp,
        },
        "alerts": [{k: v for k, v in a.items()} for a in alerts],
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Alerts saved → {out_path}")
    print("=" * 66)


if __name__ == "__main__":
    main()
