"""
stage2_filter.py
Burst download detection for LogSentinel.

Scans raw M365 audit log events per user for download bursts — a high
number of file download events within a short rolling time window.

Works directly on raw JSONL events. No transformer, no windows, no model
scores required. Each user's event stream is scanned independently.

A burst is defined as: >= THRESHOLD download events from the same user
within any WINDOW_MINUTES minute interval. All download events within
that interval are flagged and reported.

Multi-platform aware: counts downloads across SharePoint, OneDrive, and
any other workload. Each flagged event records its source platform.

Usage:
    python stage2_filter.py
    python stage2_filter.py --events-file data/tenant_test/anomaly_test.jsonl
    python stage2_filter.py --window-minutes 10 --threshold 15
    python stage2_filter.py --events-file data/tenant_real/logs.jsonl --out alerts.json
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Download operations to monitor ────────────────────────────────────────────
# Add any M365 operation name that represents a file download.
DOWNLOAD_OPS = {
    "FileDownloaded",
    "FileSyncDownloadedFull",
    "FileSyncDownloadedPartial",
    "FileAccessed",          # broad — remove if too noisy for your tenant
}

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_EVENTS_FILE    = "data/tenant_test/anomaly_test.jsonl"
DEFAULT_WINDOW_MINUTES = 5
DEFAULT_THRESHOLD      = 8


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ── Burst detection ────────────────────────────────────────────────────────────

def detect_bursts(
    events: list[dict],
    window_minutes: int,
    threshold: int,
) -> list[dict]:
    """
    Scan all events for per-user download bursts.

    For each user, collect their download events sorted by time.
    Slide a window of `window_minutes` minutes: if any window contains
    >= threshold events, all events in that window are flagged as a burst.

    Overlapping bursts are merged — once a burst starts, all download events
    within window_minutes of the first event are included before looking
    for the next independent burst.

    Returns a list of burst dicts, each containing:
        user_id       : user who triggered the burst
        burst_start   : timestamp of first download in burst
        burst_end     : timestamp of last download in burst
        event_count   : number of download events in burst
        platforms     : set of workloads (SharePoint, OneDrive, ...)
        events        : list of the raw event dicts in the burst
        is_true_positive : True if any event has _anomaly label (eval only)
    """
    window = timedelta(minutes=window_minutes)

    # Group download events by user
    user_downloads: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("Operation") in DOWNLOAD_OPS:
            user_downloads[e.get("UserId", "unknown")].append(e)

    bursts = []

    for user_id, dl_events in user_downloads.items():
        # Sort by time
        try:
            dl_events.sort(key=lambda e: parse_time(e["CreationTime"]))
        except (KeyError, ValueError):
            continue

        i = 0
        while i < len(dl_events):
            anchor_time = parse_time(dl_events[i]["CreationTime"])
            window_end  = anchor_time + window

            # Collect all events within window_minutes of anchor
            burst_events = [
                e for e in dl_events[i:]
                if parse_time(e["CreationTime"]) <= window_end
            ]

            if len(burst_events) >= threshold:
                platforms = {e.get("Workload", "Unknown") for e in burst_events}
                is_tp     = any("_anomaly" in e for e in burst_events)
                bursts.append({
                    "user_id":          user_id,
                    "burst_start":      dl_events[i]["CreationTime"],
                    "burst_end":        burst_events[-1]["CreationTime"],
                    "event_count":      len(burst_events),
                    "platforms":        sorted(platforms),
                    "is_true_positive": is_tp,
                    "events":           burst_events,
                })
                # Skip past this burst before looking for the next one
                i += len(burst_events)
            else:
                i += 1

    # Sort output by time
    bursts.sort(key=lambda b: b["burst_start"])
    return bursts


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LogSentinel Stage 2 — Burst Download Detection"
    )
    parser.add_argument(
        "--events-file", default=DEFAULT_EVENTS_FILE,
        help="Path to raw events JSONL file",
    )
    parser.add_argument(
        "--window-minutes", type=int, default=DEFAULT_WINDOW_MINUTES,
        help=f"Rolling time window in minutes (default: {DEFAULT_WINDOW_MINUTES})",
    )
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help=f"Min download events in window to trigger alert (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON file for alerts (default: <events_dir>/burst_alerts.json)",
    )
    args = parser.parse_args()

    events_path = Path(args.events_file)
    out_path    = Path(args.out) if args.out else events_path.parent / "burst_alerts.json"

    print("=" * 66)
    print("  LogSentinel Stage 2 — Burst Download Detection")
    print("=" * 66)
    print(f"\n  Events:         {events_path}")
    print(f"  Window:         {args.window_minutes} minutes")
    print(f"  Threshold:      >= {args.threshold} downloads in window")
    print(f"  Monitored ops:  {sorted(DOWNLOAD_OPS)}")

    events = load_jsonl(events_path)
    print(f"\n  Loaded {len(events):,} events")

    # Count users and download events
    n_users     = len({e.get("UserId") for e in events})
    n_downloads = sum(1 for e in events if e.get("Operation") in DOWNLOAD_OPS)
    print(f"  Users:          {n_users}")
    print(f"  Download events:{n_downloads:,}")

    # Detect bursts
    bursts = detect_bursts(events, args.window_minutes, args.threshold)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 66}")
    print(f"  Burst alerts detected: {len(bursts)}")
    print(f"{'─' * 66}")

    if not bursts:
        print("  No bursts detected.")
    else:
        for b in bursts:
            tp_marker = " [LABELED ANOMALY]" if b["is_true_positive"] else ""
            print(f"\n  User:     {b['user_id']}{tp_marker}")
            print(f"  Start:    {b['burst_start']}")
            print(f"  End:      {b['burst_end']}")
            print(f"  Events:   {b['event_count']} downloads")
            print(f"  Platform: {', '.join(b['platforms'])}")

    # ── Evaluation summary (only if _anomaly labels present) ──────────────────
    has_labels = any("_anomaly" in e for e in events)
    if has_labels:
        true_bursts  = sum(1 for b in bursts if b["is_true_positive"])
        false_bursts = sum(1 for b in bursts if not b["is_true_positive"])

        # Count labeled mass_download users in events
        labeled_md_users = {
            e.get("UserId") for e in events
            if e.get("_anomaly", {}).get("type") == "mass_download"
        }
        detected_md_users = {
            b["user_id"] for b in bursts if b["is_true_positive"]
        }
        missed_md_users = labeled_md_users - detected_md_users

        print(f"\n{'─' * 66}")
        print(f"  Evaluation (ground truth labels present)")
        print(f"{'─' * 66}")
        print(f"  Labeled mass_download users:   {len(labeled_md_users)}")
        print(f"  Detected (true positives):     {true_bursts} bursts "
              f"across {len(detected_md_users)} users")
        print(f"  False positives:               {false_bursts} bursts")
        print(f"  Missed users:                  {len(missed_md_users)} "
              f"— {sorted(missed_md_users) or 'none'}")
        if labeled_md_users:
            recall = len(detected_md_users) / len(labeled_md_users)
            print(f"  User-level recall:             {recall:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "config": {
            "events_file":    str(events_path),
            "window_minutes": args.window_minutes,
            "threshold":      args.threshold,
            "download_ops":   sorted(DOWNLOAD_OPS),
        },
        "summary": {
            "total_events":   len(events),
            "total_bursts":   len(bursts),
            "users_flagged":  len({b["user_id"] for b in bursts}),
        },
        "bursts": [
            {k: v for k, v in b.items() if k != "events"}
            for b in bursts
        ],
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Alerts saved → {out_path}")
    print("=" * 66)


if __name__ == "__main__":
    main()
