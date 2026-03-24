"""
stage2_filter.py
Hybrid detection stage for LogSentinel.

Combines BitNet transformer perplexity (Stage 1) with a rule-based engine
to catch volume/rate anomalies that perplexity alone misses when anomalous
events are diluted within a large context window.

Rules (ADD flags):
  Rule A — mass_download  : ≥8 FileDownloaded events from same user in 5 min
  Rule B — brute_force    : ≥10 UserLoginFailed from same IP in 60 seconds

FP suppression (applied to model-only flags, never rule flags):
  Suppression 1 — Marginal score : score < threshold + 10% → suppress
  Suppression 2 — Alert storm    : user has >5 model flags → keep only highest

Usage:
    # Default — finetuned model results
    python stage2_filter.py

    # Custom paths / sigma
    python stage2_filter.py \\
        --scores-file data/tenant_test/finetuned/anomaly_scores.json \\
        --events-file data/tenant_test/anomaly_test.jsonl \\
        --sigma 2.0

    # Tune rule sensitivity
    python stage2_filter.py --mass-download-threshold 6 --brute-force-threshold 8
"""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_SCORES_FILE          = "data/tenant_test/finetuned/anomaly_scores.json"
DEFAULT_EVENTS_FILE          = "data/tenant_test/anomaly_test.jsonl"
DEFAULT_SIGMA                = 2.0
DEFAULT_MARGINAL_PCT         = 0.10
DEFAULT_STORM_LIMIT          = 5
DEFAULT_MASS_DL_THRESHOLD    = 8
DEFAULT_MASS_DL_WINDOW_S     = 300
DEFAULT_BRUTE_THRESHOLD      = 10
DEFAULT_BRUTE_WINDOW_S       = 60


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def uid_hash(email: str) -> str:
    return hashlib.md5(email.encode()).hexdigest()[:4]


def parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def metrics(predicted: list[bool], labels: list[bool]) -> dict:
    tp = sum(p and l for p, l in zip(predicted, labels))
    fp = sum(p and not l for p, l in zip(predicted, labels))
    fn = sum(not p and l for p, l in zip(predicted, labels))
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "flagged":   sum(predicted),
    }


def print_metrics(label: str, m: dict):
    print(f"  {label:<42} "
          f"TP={m['tp']:3d}  FP={m['fp']:4d}  FN={m['fn']:3d}  "
          f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
          f"F1={m['f1']:.3f}  Flagged={m['flagged']}")


# ── Rule engine ────────────────────────────────────────────────────────────────

def rule_mass_download(events: list[dict], threshold: int, window_s: int) -> set[str]:
    """
    Flag user hashes where ≥threshold FileDownloaded events occur within
    window_s seconds of each other.
    """
    user_times: dict[str, list[datetime]] = defaultdict(list)
    for e in events:
        if e.get("Operation") == "FileDownloaded":
            uid = e.get("UserId", "")
            try:
                user_times[uid].append(parse_time(e["CreationTime"]))
            except (KeyError, ValueError):
                pass

    triggered: set[str] = set()
    for uid, times in user_times.items():
        times.sort()
        for i, t0 in enumerate(times):
            count = sum(
                1 for t in times[i:]
                if (t - t0).total_seconds() <= window_s
            )
            if count >= threshold:
                triggered.add(uid_hash(uid))
                break
    return triggered


def rule_brute_force(events: list[dict], threshold: int, window_s: int) -> set[str]:
    """
    Flag user hashes where ≥threshold UserLoginFailed events from the same IP
    occur within window_s seconds of each other.
    """
    ip_entries: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for e in events:
        if e.get("Operation") == "UserLoginFailed":
            ip  = e.get("ClientIP", "unknown")
            uid = e.get("UserId", "")
            try:
                ip_entries[ip].append((parse_time(e["CreationTime"]), uid))
            except (KeyError, ValueError):
                pass

    triggered: set[str] = set()
    for ip, entries in ip_entries.items():
        entries.sort()
        for i, (t0, _) in enumerate(entries):
            window = [
                (t, u) for t, u in entries[i:]
                if (t - t0).total_seconds() <= window_s
            ]
            if len(window) >= threshold:
                for _, uid in window:
                    triggered.add(uid_hash(uid))
                break
    return triggered


# ── FP suppression ─────────────────────────────────────────────────────────────

def suppress_marginal(
    flags: list[bool],
    scores: list[float],
    threshold: float,
    rule_flags: list[bool],
    pct: float,
) -> tuple[list[bool], int]:
    """Suppress model flags with scores barely above threshold. Never suppresses rule flags."""
    out = list(flags)
    suppressed = 0
    for i, flagged in enumerate(out):
        if flagged and not rule_flags[i] and scores[i] < threshold * (1 + pct):
            out[i] = False
            suppressed += 1
    return out, suppressed


def suppress_storm(
    flags: list[bool],
    scores: list[float],
    user_ids: list[str],
    rule_flags: list[bool],
    limit: int,
) -> tuple[list[bool], int]:
    """
    For users with >limit model-only flags, keep only the highest-scoring window.
    Never suppresses rule-flagged windows.
    """
    user_model_indices: dict[str, list[int]] = defaultdict(list)
    for i, flagged in enumerate(flags):
        if flagged and not rule_flags[i]:
            user_model_indices[user_ids[i]].append(i)

    out = list(flags)
    suppressed = 0
    for uid, indices in user_model_indices.items():
        if len(indices) > limit:
            best = max(indices, key=lambda i: scores[i])
            for i in indices:
                if i != best:
                    out[i] = False
                    suppressed += 1
    return out, suppressed


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LogSentinel Stage 2 — Hybrid Detection"
    )
    parser.add_argument("--scores-file",
                        default=DEFAULT_SCORES_FILE,
                        help="Path to anomaly_scores.json from finetune.py or detect.py")
    parser.add_argument("--events-file",
                        default=DEFAULT_EVENTS_FILE,
                        help="Path to raw test events JSONL")
    parser.add_argument("--sigma",
                        type=float, default=DEFAULT_SIGMA,
                        help="Model threshold = val_mean + sigma * val_std")
    parser.add_argument("--marginal-pct",
                        type=float, default=DEFAULT_MARGINAL_PCT,
                        help="Suppress model flags within this %% above threshold")
    parser.add_argument("--storm-limit",
                        type=int, default=DEFAULT_STORM_LIMIT,
                        help="Max model flags per user before storm suppression")
    parser.add_argument("--mass-download-threshold",
                        type=int, default=DEFAULT_MASS_DL_THRESHOLD,
                        help="Min FileDownloaded events to trigger mass_download rule")
    parser.add_argument("--mass-download-window",
                        type=int, default=DEFAULT_MASS_DL_WINDOW_S,
                        help="Time window in seconds for mass_download rule")
    parser.add_argument("--brute-force-threshold",
                        type=int, default=DEFAULT_BRUTE_THRESHOLD,
                        help="Min failed logins from same IP to trigger brute_force rule")
    parser.add_argument("--brute-force-window",
                        type=int, default=DEFAULT_BRUTE_WINDOW_S,
                        help="Time window in seconds for brute_force rule")
    args = parser.parse_args()

    scores_path = Path(args.scores_file)
    events_path = Path(args.events_file)
    out_dir     = scores_path.parent

    print("=" * 74)
    print("  LogSentinel Stage 2 — Hybrid Detection")
    print("=" * 74)
    print(f"\n  Scores:  {scores_path}")
    print(f"  Events:  {events_path}")
    print(f"  Sigma:   {args.sigma}")

    # ── Load ──────────────────────────────────────────────────────────────────
    stage1   = json.loads(scores_path.read_text())
    scores   = stage1["test_scores"]
    labels   = [bool(l) for l in stage1["test_labels"]]
    user_ids = stage1["test_user_ids"]

    val_mean = stage1.get("val_mean")
    val_std  = stage1.get("val_std")
    if val_mean is None or val_std is None:
        raise ValueError(
            "anomaly_scores.json is missing val_mean/val_std. "
            "Re-run finetune.py to regenerate."
        )

    threshold = val_mean + args.sigma * val_std
    events    = load_jsonl(events_path)

    print(f"\n  Windows:          {len(scores):,}")
    print(f"  Anomalous:        {sum(labels)}")
    print(f"  Val distribution: mean={val_mean:.2f}  std={val_std:.2f}")
    print(f"  Threshold:        {threshold:.2f}  (sigma={args.sigma})")

    # ── Stage 1: model flags ──────────────────────────────────────────────────
    model_flags = [s > threshold for s in scores]
    m_model     = metrics(model_flags, labels)

    # ── Rule engine ───────────────────────────────────────────────────────────
    print("\n  Running rule engine on raw events...")
    md_users  = rule_mass_download(
        events, args.mass_download_threshold, args.mass_download_window)
    bf_users  = rule_brute_force(
        events, args.brute_force_threshold, args.brute_force_window)

    all_rule_users = md_users | bf_users
    print(f"    mass_download triggered: {len(md_users):2d} users  — {sorted(md_users)}")
    print(f"    brute_force   triggered: {len(bf_users):2d} users  — {sorted(bf_users)}")

    rule_flags = [uid in all_rule_users for uid in user_ids]
    m_rules    = metrics(rule_flags, labels)

    # ── Combined (model OR rules) ──────────────────────────────────────────────
    combined_flags = [m or r for m, r in zip(model_flags, rule_flags)]
    m_combined     = metrics(combined_flags, labels)

    # ── FP suppression (model-only flags only) ────────────────────────────────
    after_marginal, n_marginal = suppress_marginal(
        combined_flags, scores, threshold, rule_flags, args.marginal_pct)
    after_storm, n_storm = suppress_storm(
        after_marginal, scores, user_ids, rule_flags, args.storm_limit)
    m_final = metrics(after_storm, labels)

    # ── Report ────────────────────────────────────────────────────────────────
    W = 74
    print(f"\n{'─' * W}")
    print(f"  {'Stage':<42} {'TP':>4}  {'FP':>5}  {'FN':>4}  "
          f"{'P':>6}  {'R':>6}  {'F1':>6}  {'Flagged':>7}")
    print(f"{'─' * W}")
    print_metrics("1. Model only",               m_model)
    print_metrics("2. Rules only",               m_rules)
    print_metrics("3. Combined (model OR rules)", m_combined)
    print_metrics("4. Combined + FP suppression", m_final)
    print(f"{'─' * W}")

    print(f"\n  FP suppression removed: {n_marginal} marginal + {n_storm} storm "
          f"= {n_marginal + n_storm} total")
    print(f"  Missed anomalies (final): {m_final['fn']}")
    if m_final["tp"] > 0:
        print(f"  FP per TP (final):        {m_final['fp'] / m_final['tp']:.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "config": {
            "sigma":                    args.sigma,
            "threshold":                round(threshold, 4),
            "val_mean":                 val_mean,
            "val_std":                  val_std,
            "mass_download_threshold":  args.mass_download_threshold,
            "mass_download_window_s":   args.mass_download_window,
            "brute_force_threshold":    args.brute_force_threshold,
            "brute_force_window_s":     args.brute_force_window,
            "marginal_pct":             args.marginal_pct,
            "storm_limit":              args.storm_limit,
        },
        "rule_triggered_users": {
            "mass_download": sorted(md_users),
            "brute_force":   sorted(bf_users),
        },
        "metrics": {
            "model_only":   m_model,
            "rules_only":   m_rules,
            "combined":     m_combined,
            "final":        m_final,
        },
    }

    out_path = out_dir / "stage2_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved → {out_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()
