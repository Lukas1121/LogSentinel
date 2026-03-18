#!/usr/bin/env python3
"""
detect.py — LogSentinel post-training detection explorer.

Loads pre-computed scores from results/anomaly_scores.json and lets you
explore threshold parameters without retraining. Optionally recomputes
scores from the saved model checkpoint.

Usage:
    python detect.py                        # sigma sweep (default)
    python detect.py --sigma 3.0            # single threshold
    python detect.py --sigma 3.0 --stage2   # with Stage 2 rule filter
    python detect.py --recompute            # reload model, recompute scores
    python detect.py --pr-curve             # print full precision/recall curve
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data")
CKPT_DIR  = Path("checkpoints")
RES_DIR   = Path("results")

# ── Inline model definition (must match train_transformer.py exactly) ─────────

import torch.nn as nn

class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        scale = w.abs().mean().clamp(min=1e-8)
        w_ternary = (w / scale).round().clamp(-1, 1)
        w_quant = w + (w_ternary - w).detach()
        return F.linear(x, w_quant, self.bias)

class CausalSelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, ctx_len, dropout=0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads
        self.qkv      = BitLinear(emb_dim, 3 * emb_dim, bias=False)
        self.out      = BitLinear(emb_dim, emb_dim,     bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out(out))

class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            BitLinear(emb_dim, 4 * emb_dim), nn.GELU(),
            BitLinear(4 * emb_dim, emb_dim), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, ctx_len):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn  = CausalSelfAttention(emb_dim, n_heads, ctx_len)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ffn   = FeedForward(emb_dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class BitNetTransformer(nn.Module):
    def __init__(self, vocab, emb_dim, n_layers, n_heads, ctx_len):
        super().__init__()
        self.ctx_len   = ctx_len
        self.token_emb = nn.Embedding(vocab, emb_dim)
        self.pos_emb   = nn.Embedding(ctx_len, emb_dim)
        self.blocks    = nn.Sequential(*[
            TransformerBlock(emb_dim, n_heads, ctx_len) for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(emb_dim)
        self.lm_head = BitLinear(emb_dim, vocab, bias=False)

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.norm(x)
        return self.lm_head(x)


# ── Scoring ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_scores(model, windows, device, batch_size=64):
    """Recompute max-token perplexity scores from model."""
    model.eval()
    scores = []
    PAD_ID = 0
    for i in range(0, len(windows), batch_size):
        batch  = windows[i : i + batch_size].to(device)
        x, y   = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        B, T, V = logits.shape
        token_losses = F.cross_entropy(
            logits.reshape(B * T, V), y.reshape(B * T),
            ignore_index=PAD_ID, reduction="none",
        ).reshape(B, T)
        mask = (y != PAD_ID).float()
        token_losses = token_losses * mask
        max_loss = token_losses.max(dim=1).values
        scores.append(torch.log1p(max_loss.exp()).cpu())
    return torch.cat(scores)


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics_at_threshold(scores, labels, threshold):
    predicted = scores > threshold
    tp = (predicted &  labels).sum().item()
    fp = (predicted & ~labels).sum().item()
    fn = (~predicted & labels).sum().item()
    tn = (~predicted & ~labels).sum().item()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1,
                flagged=tp+fp, threshold=threshold)


# ── Sigma sweep ───────────────────────────────────────────────────────────────

def sigma_sweep(scores, labels, val_mean, val_std,
                sigmas=None, show_stage2_hint=True):
    if sigmas is None:
        sigmas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    print("\n" + "═" * 78)
    print("  LogSentinel — Sigma Sweep")
    print("  Val distribution: mean={:.2f}  std={:.2f}".format(val_mean, val_std))
    print("═" * 78)
    print(f"  {'Sigma':>5}  {'Threshold':>9}  {'TP':>5}  {'FP':>6}  "
          f"{'FN':>5}  {'Precision':>10}  {'Recall':>8}  {'F1':>7}  {'Flagged':>8}")
    print("  " + "─" * 74)

    current_sigma = 2.0
    for sigma in sigmas:
        threshold = val_mean + sigma * val_std
        m = metrics_at_threshold(scores, labels, threshold)
        marker = "  ◄ current" if abs(sigma - current_sigma) < 0.01 else ""
        print(f"  {sigma:>5.1f}  {threshold:>9.2f}  {m['tp']:>5}  {m['fp']:>6}  "
              f"{m['fn']:>5}  {m['precision']:>10.3f}  {m['recall']:>8.3f}  "
              f"{m['f1']:>7.3f}  {m['flagged']:>8}{marker}")

    print("═" * 78)

    if show_stage2_hint:
        print("\n  Tip: Stage 2 rule filter reduces FPs ~46% with small recall cost.")
        print("  Run with --stage2 to apply it at any threshold.\n")


# ── Detailed single-sigma report ──────────────────────────────────────────────

def single_sigma_report(scores, labels, val_mean, val_std, sigma, apply_stage2=False):
    threshold = val_mean + sigma * val_std
    m = metrics_at_threshold(scores, labels, threshold)

    print("\n" + "═" * 60)
    print(f"  LogSentinel — Threshold Report  (σ={sigma})")
    print("═" * 60)
    print(f"  Val distribution:  mean={val_mean:.2f}  std={val_std:.2f}")
    print(f"  Threshold:         {threshold:.2f}")
    print(f"\n  Stage 1 results:")
    print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(f"    Precision: {m['precision']:.3f}")
    print(f"    Recall:    {m['recall']:.3f}")
    print(f"    F1:        {m['f1']:.3f}")
    print(f"    Flagged:   {m['flagged']} windows")

    if apply_stage2:
        _apply_stage2_inline(scores, labels, threshold, m)

    # Score distribution context
    anom_scores  = scores[labels]
    normal_scores = scores[~labels]
    print(f"\n  Score distribution:")
    print(f"    Normal  — mean={normal_scores.mean():.2f}  "
          f"std={normal_scores.std():.2f}  "
          f"max={normal_scores.max():.2f}")
    print(f"    Anomaly — mean={anom_scores.mean():.2f}  "
          f"std={anom_scores.std():.2f}  "
          f"min={anom_scores.min():.2f}")
    sep = anom_scores.mean() - normal_scores.mean()
    print(f"    Separation (anom_mean - normal_mean): {sep:+.2f}")
    if sep < 2.0:
        print(f"    ⚠  Low separation — model may not have converged fully.")
        print(f"       Consider more training epochs before threshold tuning.")
    print("═" * 60)


# ── Stage 2 inline (simplified, no external file needed) ─────────────────────

def _apply_stage2_inline(scores, labels, threshold, stage1_metrics):
    """
    Apply Stage 2 rules directly to flagged windows.
    Simplified version — for full rule logic use stage2_filter.py.
    Only Rule 2 (marginal score) is available here without token data.
    """
    flagged_mask = scores > threshold
    flagged_scores = scores[flagged_mask]
    flagged_labels = labels[flagged_mask]

    # Rule 2: marginal score (score < threshold + 10%)
    margin = threshold * 0.10
    marginal = flagged_scores < (threshold + margin)
    kept = ~marginal

    tp_s2 = (kept & flagged_labels).sum().item()
    fp_s2 = (kept & ~flagged_labels).sum().item()
    fn_s2 = stage1_metrics['fn'] + (marginal & flagged_labels).sum().item()
    tn_s2 = stage1_metrics['tn'] + (marginal & ~flagged_labels).sum().item()
    prec_s2  = tp_s2 / max(tp_s2 + fp_s2, 1)
    rec_s2   = tp_s2 / max(tp_s2 + fn_s2, 1)
    f1_s2    = 2 * prec_s2 * rec_s2 / max(prec_s2 + rec_s2, 1e-8)
    suppressed = marginal.sum().item()

    print(f"\n  After Stage 2 (Rule 2 — marginal score only):")
    print(f"    Suppressed: {suppressed} marginal windows")
    print(f"    TP={tp_s2}  FP={fp_s2}  FN={fn_s2}  TN={tn_s2}")
    print(f"    Precision: {prec_s2:.3f}  Recall: {rec_s2:.3f}  F1: {f1_s2:.3f}")
    print(f"    (Run stage2_filter.py for full Rule 1+2+3 results)")


# ── Precision/recall curve ────────────────────────────────────────────────────

def pr_curve(scores, labels, n_points=40):
    """Print a text precision/recall curve by varying the threshold."""
    min_s, max_s = scores.min().item(), scores.max().item()
    thresholds = torch.linspace(min_s, max_s, n_points)

    print("\n" + "═" * 60)
    print("  Precision / Recall Curve")
    print("═" * 60)
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Flagged':>8}")
    print("  " + "─" * 54)
    for t in thresholds:
        m = metrics_at_threshold(scores, labels, t.item())
        print(f"  {t.item():>10.2f}  {m['precision']:>10.3f}  "
              f"{m['recall']:>8.3f}  {m['f1']:>8.3f}  {m['flagged']:>8}")
    print("═" * 60)


# ── Load scores ───────────────────────────────────────────────────────────────

def load_saved_scores():
    """Load pre-computed scores from results/anomaly_scores.json."""
    path = RES_DIR / "anomaly_scores.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run train_transformer.py first, "
            "or use --recompute to load from checkpoint."
        )
    data = json.loads(path.read_text())
    scores = torch.tensor(data["test_scores"], dtype=torch.float32)
    labels = torch.tensor(data["test_labels"], dtype=torch.bool)
    val_mean = data["val_score_mean"]
    val_std  = data["val_score_std"]
    print(f"  Loaded {len(scores)} test windows  "
          f"({labels.sum().item()} anomalous)  from {path}")
    return scores, labels, val_mean, val_std


def recompute_scores():
    """Reload model from checkpoint and recompute scores from scratch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CKPT_DIR / "model_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt["config"]

    model = BitNetTransformer(
        vocab=cfg["vocab"], emb_dim=cfg["emb_dim"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        ctx_len=cfg["ctx_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Model loaded  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    val_windows  = torch.load(DATA_DIR / "val_tokens.pt",  weights_only=True)
    test_windows = torch.load(DATA_DIR / "test_tokens.pt", weights_only=True)
    test_labels  = torch.load(DATA_DIR / "test_labels.pt", weights_only=True)

    print("  Computing val scores...")
    val_scores = compute_scores(model, val_windows, device)
    print("  Computing test scores...")
    test_scores = compute_scores(model, test_windows, device)

    val_mean = val_scores.mean().item()
    val_std  = val_scores.std().item()

    # Save results to JSON so run_and_exit.sh can read the summary
    # and subsequent detect.py calls can use --load instead of --recompute.
    save_scores(test_scores, test_labels.bool(), val_mean, val_std,
                n_sigma=2.0, ckpt_epoch=ckpt["epoch"])

    return test_scores, test_labels.bool(), val_mean, val_std


def save_scores(test_scores: torch.Tensor, labels: torch.Tensor,
                val_mean: float, val_std: float,
                n_sigma: float = 2.0, ckpt_epoch: int = 0):
    """
    Write results/anomaly_scores.json in the same format the old
    train_transformer.py produced, so run_and_exit.sh and load_saved_scores()
    both work correctly after a --recompute run.
    """
    RES_DIR.mkdir(exist_ok=True)

    threshold = val_mean + n_sigma * val_std
    m_global  = metrics_at_threshold(test_scores, labels, threshold)

    # Perfect-recall threshold — just below the lowest anomaly score
    anom_scores       = test_scores[labels]
    threshold_perfect = anom_scores.min().item() * 0.999
    m_perfect         = metrics_at_threshold(test_scores, labels, threshold_perfect)

    results = {
        "val_score_mean":    round(val_mean, 4),
        "val_score_std":     round(val_std,  4),
        "global_threshold":  round(threshold, 4),
        "perfect_threshold": round(threshold_perfect, 4),
        "n_sigma":           n_sigma,
        "checkpoint_epoch":  ckpt_epoch,
        "test_windows":      len(test_scores),
        "anomalous_windows": int(labels.sum()),
        "global": {
            "tp": m_global["tp"], "fp": m_global["fp"],
            "fn": m_global["fn"], "tn": m_global["tn"],
            "precision": round(m_global["precision"], 4),
            "recall":    round(m_global["recall"],    4),
            "f1":        round(m_global["f1"],        4),
        },
        "perfect_recall": {
            "tp": m_perfect["tp"], "fp": m_perfect["fp"],
            "fn": m_perfect["fn"], "tn": m_perfect["tn"],
            "precision": round(m_perfect["precision"], 4),
            "recall":    round(m_perfect["recall"],    4),
            "f1":        round(m_perfect["f1"],        4),
        },
        "test_scores": test_scores.tolist(),
        "test_labels": labels.tolist(),
    }

    out = RES_DIR / "anomaly_scores.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  Results saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LogSentinel — threshold explorer (no retraining needed)"
    )
    parser.add_argument(
        "--sigma", type=float, default=None,
        help="Specific sigma value to evaluate (default: sweep 1.0–5.0)"
    )
    parser.add_argument(
        "--stage2", action="store_true",
        help="Apply Stage 2 marginal-score filter after thresholding"
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Reload model from checkpoint and recompute scores (slow)"
    )
    parser.add_argument(
        "--pr-curve", action="store_true", dest="pr_curve",
        help="Print full precision/recall curve across all thresholds"
    )
    parser.add_argument(
        "--sweep-sigmas", type=str, default=None,
        help="Comma-separated sigma values for custom sweep (e.g. '2.0,2.5,3.0')"
    )
    args = parser.parse_args()

    print("\n  LogSentinel — detect.py")
    print("  " + "─" * 40)

    if args.recompute:
        print("  Mode: recompute from checkpoint")
        scores, labels, val_mean, val_std = recompute_scores()
    else:
        print("  Mode: load from results/anomaly_scores.json")
        scores, labels, val_mean, val_std = load_saved_scores()

    if args.pr_curve:
        pr_curve(scores, labels)

    if args.sigma is not None:
        single_sigma_report(scores, labels, val_mean, val_std,
                            sigma=args.sigma, apply_stage2=args.stage2)
    else:
        # Default: sigma sweep
        custom_sigmas = None
        if args.sweep_sigmas:
            custom_sigmas = [float(s) for s in args.sweep_sigmas.split(",")]
        sigma_sweep(scores, labels, val_mean, val_std, sigmas=custom_sigmas)

        if args.stage2:
            print("\n  Stage 2 is only shown in single-sigma mode.")
            print("  Re-run with --sigma <value> --stage2\n")


if __name__ == "__main__":
    main()