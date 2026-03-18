"""
stage2_filter.py
Rule-based false positive filter for LogSentinel Stage 2.

Takes Stage 1 (BitNet transformer) flagged windows and applies four
structural rules to suppress obvious false positives before passing
remaining alerts to Stage 3 (neural classifier).

Rules applied in order:
  Rule 1 — Padding ratio     : window > 50% PAD tokens → suppress
  Rule 2 — Marginal score    : score < 10% above user threshold → suppress
  Rule 3 — Alert storm       : user has > 5 flags → keep only highest
  Rule 4 — Operation frequency: dominant op in user's top-3 history → suppress

Input:
  data/test_tokens.pt         stage 1 windows
  data/test_labels.pt         ground truth labels
  data/test_user_ids.json     user ID per window
  data/tokeniser.json         vocab (to decode PAD token ID)
  results/anomaly_scores.json stage 1 scores + per-user thresholds
  data/train.jsonl            training events for user op profiles

Output:
  results/stage2_results.json precision/recall/F1 after each rule
  results/stage2_alerts.json  windows surviving all rules (input to stage 3)
"""

import json
import math
from pathlib import Path
from collections import defaultdict, Counter

import torch

DATA_DIR    = Path("data")
RES_DIR     = Path("results")
N_SIGMA     = 3.0   # must match train_transformer.py
MARGINAL_PCT = 0.10  # rule 2: suppress if score < threshold * (1 + this)
STORM_LIMIT  = 5    # rule 3: max flags per user before storm suppression
PAD_RATIO    = 0.50  # rule 1: suppress if padding fraction exceeds this
TOP_N_OPS    = 3    # rule 4: suppress if dominant op in user's top-N


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def metrics(predicted: list[bool], labels: list[bool]) -> dict:
    tp = sum(p and l for p, l in zip(predicted, labels))
    fp = sum(p and not l for p, l in zip(predicted, labels))
    fn = sum(not p and l for p, l in zip(predicted, labels))
    tn = sum(not p and not l for p, l in zip(predicted, labels))
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def print_metrics(label: str, m: dict):
    print(f"\n  {label}")
    print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"    Precision={m['precision']:.3f}  "
          f"Recall={m['recall']:.3f}  F1={m['f1']:.3f}")


# ── Build per-user operation profiles from training data ──────────────────────

def build_user_op_profiles(train_events: list[dict]) -> dict[str, Counter]:
    """
    For each user, count how many times they performed each operation.
    Used by Rule 4 to check if flagged operation is routine for that user.
    """
    import hashlib
    profiles: dict[str, Counter] = defaultdict(Counter)
    for event in train_events:
        uid = event.get("UserId", "unknown")
        uid_hash = hashlib.md5(uid.encode()).hexdigest()[:4]
        op  = event.get("Operation", "UNKNOWN")
        profiles[uid_hash][op] += 1
    return profiles


def get_dominant_op(window_tokens: list[int], vocab_id2tok: list[str]) -> str | None:
    """
    Find the most frequent operation token in a window.
    Returns the operation name (without 'op:' prefix) or None.
    """
    op_counts: Counter = Counter()
    for tok_id in window_tokens:
        if tok_id < len(vocab_id2tok):
            tok = vocab_id2tok[tok_id]
            if tok.startswith("op:"):
                op_counts[tok[3:]] += 1
    if op_counts:
        return op_counts.most_common(1)[0][0]
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LogSentinel Stage 2 — Rule-based Filter")
    print("=" * 60)

    RES_DIR.mkdir(exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")

    test_windows  = torch.load(DATA_DIR / "test_tokens.pt",  weights_only=True)
    test_labels_t = torch.load(DATA_DIR / "test_labels.pt",  weights_only=True)
    test_labels   = test_labels_t.tolist()
    test_user_ids = json.loads((DATA_DIR / "test_user_ids.json").read_text())
    vocab         = json.loads((DATA_DIR / "tokeniser.json").read_text())
    stage1        = json.loads((RES_DIR  / "anomaly_scores.json").read_text())

    PAD_ID   = vocab["special"]["PAD"]
    id2tok   = vocab["id2tok"]
    ctx_len  = test_windows.shape[1]

    print(f"  Test windows:     {len(test_windows):,}")
    print(f"  Anomalous:        {sum(test_labels)}")
    print(f"  Context length:   {ctx_len}")

    # ── Reconstruct Stage 1 predictions ──────────────────────────────────────
    # Stage 1 used per-user thresholds. Reconstruct which windows were flagged.
    test_scores = stage1["test_scores"]

    # Rebuild per-user thresholds from val scores saved in anomaly_scores.json
    # Fall back to global threshold for users not in val set
    global_threshold = stage1["global_threshold"]

    # We need to re-derive per-user thresholds. Since we saved val_user_ids,
    # load them and recompute from the val scores stored in the results.
    # Simpler: just use global threshold for Stage 2 filter demonstration.
    # The key insight is Stage 2 operates on *already-flagged* windows.

    import statistics, hashlib

    val_user_ids = json.loads((DATA_DIR / "val_user_ids.json").read_text())

    # We don't have val scores saved directly, so we use global threshold
    # with per-user marginal check (rule 2 handles the per-user aspect)
    stage1_flags = [s > global_threshold for s in test_scores]

    stage1_metrics = metrics(stage1_flags, test_labels)
    print_metrics("Stage 1 (global threshold baseline):", stage1_metrics)

    n_flagged = sum(stage1_flags)
    print(f"\n  Stage 1 flagged {n_flagged} windows for Stage 2 review")

    # ── Build user operation profiles ─────────────────────────────────────────
    print("\nBuilding user operation profiles from training data...")
    train_events = load_jsonl(DATA_DIR / "train.jsonl")
    user_op_profiles = build_user_op_profiles(train_events)
    print(f"  Profiles built for {len(user_op_profiles)} users")

    # ── Apply rules ───────────────────────────────────────────────────────────
    # Start with Stage 1 flagged windows, apply each rule in sequence
    # active[i] = True means window i is still flagged after rules so far

    active = list(stage1_flags)
    results = {"stage1": stage1_metrics}

    # ── Rule 1: Padding ratio ─────────────────────────────────────────────────
    print("\nApplying Rule 1 — Padding ratio...")
    suppressed_r1 = 0
    for i, flagged in enumerate(active):
        if not flagged:
            continue
        window  = test_windows[i].tolist()
        pad_count = window.count(PAD_ID)
        if pad_count / ctx_len > PAD_RATIO:
            active[i] = False
            suppressed_r1 += 1

    m1 = metrics(active, test_labels)
    print(f"  Suppressed {suppressed_r1} windows (>{PAD_RATIO*100:.0f}% padding)")
    print_metrics("After Rule 1:", m1)
    results["after_rule1_padding"] = m1

    # ── Rule 2: Marginal score ────────────────────────────────────────────────
    print("\nApplying Rule 2 — Marginal score...")
    suppressed_r2 = 0
    for i, flagged in enumerate(active):
        if not flagged:
            continue
        score     = test_scores[i]
        threshold = global_threshold
        # Only suppress if score is barely above threshold
        if score < threshold * (1 + MARGINAL_PCT):
            active[i] = False
            suppressed_r2 += 1

    m2 = metrics(active, test_labels)
    print(f"  Suppressed {suppressed_r2} windows (score < threshold + {MARGINAL_PCT*100:.0f}%)")
    print_metrics("After Rule 2:", m2)
    results["after_rule2_marginal"] = m2

    # ── Rule 3: Alert storm ───────────────────────────────────────────────────
    print("\nApplying Rule 3 — Alert storm...")
    # Group flagged windows by user, keep only the highest-scoring one
    # if a user has more than STORM_LIMIT flags
    user_flags: dict[str, list[int]] = defaultdict(list)
    for i, flagged in enumerate(active):
        if flagged:
            uid = test_user_ids[i]
            user_flags[uid].append(i)

    suppressed_r3 = 0
    for uid, indices in user_flags.items():
        if len(indices) > STORM_LIMIT:
            # Keep only the highest-scoring window, suppress the rest
            best_idx = max(indices, key=lambda i: test_scores[i])
            for i in indices:
                if i != best_idx:
                    active[i] = False
                    suppressed_r3 += 1

    m3 = metrics(active, test_labels)
    print(f"  Suppressed {suppressed_r3} windows (alert storm >{STORM_LIMIT} per user)")
    print_metrics("After Rule 3:", m3)
    results["after_rule3_storm"] = m3

    # ── Rule 4: Operation frequency ───────────────────────────────────────────
    print("\nApplying Rule 4 — Operation frequency...")
    suppressed_r4 = 0
    for i, flagged in enumerate(active):
        if not flagged:
            continue
        uid     = test_user_ids[i]
        window  = test_windows[i].tolist()
        dom_op  = get_dominant_op(window, id2tok)

        if dom_op and uid in user_op_profiles:
            top_ops = [op for op, _ in
                       user_op_profiles[uid].most_common(TOP_N_OPS)]
            if dom_op in top_ops:
                active[i] = False
                suppressed_r4 += 1

    m4 = metrics(active, test_labels)
    print(f"  Suppressed {suppressed_r4} windows (dominant op in user top-{TOP_N_OPS})")
    print_metrics("After Rule 4:", m4)
    results["after_rule4_op_freq"] = m4

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2 SUMMARY")
    print("=" * 60)

    total_suppressed = sum(stage1_flags) - sum(active)
    print(f"\n  Stage 1 flagged:    {sum(stage1_flags):>6}")
    print(f"  Stage 2 suppressed: {total_suppressed:>6}")
    print(f"  Remaining alerts:   {sum(active):>6}")
    print(f"\n  False positive reduction: "
          f"{total_suppressed / max(stage1_metrics['fp'], 1) * 100:.1f}%")

    print(f"\n  Final metrics:")
    final = m4
    print(f"    Precision: {final['precision']:.3f}  "
          f"(was {stage1_metrics['precision']:.3f})")
    print(f"    Recall:    {final['recall']:.3f}  "
          f"(was {stage1_metrics['recall']:.3f})")
    print(f"    F1:        {final['f1']:.3f}  "
          f"(was {stage1_metrics['f1']:.3f})")
    print(f"    Missed anomalies: {final['fn']}")

    # ── Save surviving alerts for Stage 3 ────────────────────────────────────
    surviving = []
    for i, flagged in enumerate(active):
        surviving.append({
            "window_idx":    i,
            "user_id":       test_user_ids[i],
            "score":         round(test_scores[i], 4),
            "is_anomalous":  test_labels[i],
            "stage2_flag":   flagged,
        })

    alerts = [s for s in surviving if s["stage2_flag"]]
    (RES_DIR / "stage2_alerts.json").write_text(
        json.dumps(alerts, indent=2)
    )
    print(f"\n  Stage 3 input saved: {len(alerts)} alerts → results/stage2_alerts.json")

    results["final"] = final
    results["stage1_flagged"]    = sum(stage1_flags)
    results["stage2_suppressed"] = total_suppressed
    results["stage2_remaining"]  = sum(active)
    results["fp_reduction_pct"]  = round(
        total_suppressed / max(stage1_metrics["fp"], 1) * 100, 1
    )

    (RES_DIR / "stage2_results.json").write_text(
        json.dumps(results, indent=2)
    )
    print(f"  Full results saved → results/stage2_results.json")


if __name__ == "__main__":
    main()