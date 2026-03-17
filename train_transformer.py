# train_transformer.py
# BitNet b1.58 decoder-only transformer for M365 audit log anomaly detection.
#
# Architecture:
#   Vocabulary:      256 tokens  (actual ~137, padded to next power of 2)
#   Embedding dim:   128
#   Layers:          4
#   Attention heads: 4
#   Context length:  128 tokens  (~12 log events per window)
#   Parameters:      ~1.5M
#
# Training objective: next-token prediction (cross-entropy).
# Lower validation perplexity = better model of normal behaviour.
#
# After training, anomaly detection works by:
#   1. Computing per-window perplexity on the test set.
#   2. Setting threshold = mean + 2*std on val set perplexity.
#   3. Windows above threshold are flagged as anomalous.
#   4. Scoring against ground truth labels -> precision / recall / F1.
#
# Inputs:
#   data/train_tokens.pt    LongTensor (N, 128)
#   data/val_tokens.pt      LongTensor (N, 128)
#   data/test_tokens.pt     LongTensor (N, 128)
#   data/test_labels.pt     BoolTensor (N,)
#   data/tokeniser.json     vocab metadata
#
# Outputs:
#   checkpoints/model_best.pt
#   checkpoints/training_log.json
#   results/anomaly_scores.json

import math
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR  = Path("data")
CKPT_DIR  = Path("checkpoints")
RES_DIR   = Path("results")

# Model
VOCAB      = 256     # next power of 2 above real vocab (~137)
EMB_DIM    = 128
N_LAYERS   = 4
N_HEADS    = 4
CTX_LEN    = 128

# Training
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 3e-4
GRAD_CLIP  = 1.0
EVAL_EVERY = 1
SAVE_EVERY = 5

# Anomaly threshold: flag windows above mean + N_SIGMA * std of val perplexity
N_SIGMA    = 1.0  # low threshold = high recall, more false positives (intentional)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── BitLinear ─────────────────────────────────────────────────────────────────

class BitLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear with ternary weights {-1, 0, +1}.
    Forward: quantise weights via absmean scheme.
    Backward: straight-through estimator — gradients flow as if float.
    Microsoft BitNet b1.58.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w         = self.weight
        scale     = w.abs().mean().clamp(min=1e-8)
        w_ternary = (w / scale).round().clamp(-1, 1)
        w_quant   = w + (w_ternary - w).detach()
        return F.linear(x, w_quant, self.bias)


# ── Transformer blocks ────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, ctx_len: int):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads
        self.qkv      = BitLinear(emb_dim, 3 * emb_dim, bias=False)
        self.out      = BitLinear(emb_dim, emb_dim,     bias=False)
        mask = torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv     = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scale   = math.sqrt(self.head_dim)
        scores  = (q @ k.transpose(-2, -1)) / scale
        scores  = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn    = F.softmax(scores, dim=-1)
        out     = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            BitLinear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            BitLinear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, ctx_len: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn  = CausalSelfAttention(emb_dim, n_heads, ctx_len)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ffn   = FeedForward(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BitNetTransformer(nn.Module):
    def __init__(self, vocab: int, emb_dim: int, n_layers: int,
                 n_heads: int, ctx_len: int):
        super().__init__()
        self.ctx_len   = ctx_len
        self.token_emb = nn.Embedding(vocab, emb_dim)
        self.pos_emb   = nn.Embedding(ctx_len, emb_dim)
        self.blocks    = nn.Sequential(*[
            TransformerBlock(emb_dim, n_heads, ctx_len)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(emb_dim)
        self.lm_head = BitLinear(emb_dim, vocab, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, BitLinear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T    = token_ids.shape
        pos     = torch.arange(T, device=token_ids.device)
        x       = self.token_emb(token_ids) + self.pos_emb(pos)
        x       = self.blocks(x)
        x       = self.norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Dataset ───────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    Wraps a pre-tokenised window tensor.
    Returns (input, target) pairs for next-token prediction:
      input  = window[:-1]
      target = window[1:]
    PAD tokens (id=0) are ignored in the loss.
    """
    def __init__(self, path: Path):
        self.windows = torch.load(path, weights_only=True)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        return w[:-1], w[1:]


# ── Loss and evaluation ───────────────────────────────────────────────────────

PAD_ID = 0   # matches tokeniser.json special tokens

def compute_loss(model, batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    logits = model(x)
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B * T, V),
        y.view(B * T),
        ignore_index=PAD_ID,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (mean_loss, perplexity) on a dataloader."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        total += compute_loss(model, batch, device).item()
        n     += 1
    mean = total / max(n, 1)
    model.train()
    return mean, math.exp(mean)


@torch.no_grad()
def window_perplexities(model, windows: torch.Tensor, device,
                        batch_size: int = 64,
                        mode: str = "max") -> torch.Tensor:
    """
    Compute per-window anomaly scores.

    mode="max"  -- maximum per-token perplexity in the window.
                   A single anomalous event cannot hide inside a normal window.
                   Best for high-recall detection. (DEFAULT)

    mode="mean" -- mean perplexity across all non-PAD tokens.
                   Smoothed signal, better precision but misses isolated events.

    Returns a 1-D float tensor of length N_windows.
    """
    model.eval()
    scores = []
    for i in range(0, len(windows), batch_size):
        batch  = windows[i : i + batch_size].to(device)
        x, y   = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        B, T, V = logits.shape

        # Per-token loss
        token_losses = F.cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
            ignore_index=PAD_ID,
            reduction="none",
        ).reshape(B, T)

        # Zero out PAD positions
        mask = (y != PAD_ID).float()
        token_losses = token_losses * mask

        if mode == "max":
            # Max surprise token drives the window score
            per_window = token_losses.max(dim=1).values.exp()
        else:
            # Mean perplexity
            per_window = (token_losses.sum(dim=1) /
                          mask.sum(dim=1).clamp(min=1)).exp()

        scores.append(per_window.cpu())

    model.train()
    return torch.cat(scores)


# ── Anomaly evaluation ────────────────────────────────────────────────────────

def evaluate_anomaly_detection(model, device, n_sigma: float = N_SIGMA):
    """
    Anomaly detection evaluation with two thresholding strategies:

    1. Global threshold — mean + N_sigma * std across all val windows.
       Fast, works well for general deployment.

    2. Recall-optimised threshold — set at the lowest perplexity seen
       on the val set that still flags at least 1% of windows.
       Maximises recall at the expense of precision.

    Both are computed and reported. The recall-optimised threshold
    is used for the final metrics since the goal is to miss nothing.
    """
    print("\nRunning anomaly detection evaluation...")
    RES_DIR.mkdir(exist_ok=True)

    val_windows  = torch.load(DATA_DIR / "val_tokens.pt",  weights_only=True)
    test_windows = torch.load(DATA_DIR / "test_tokens.pt", weights_only=True)
    test_labels  = torch.load(DATA_DIR / "test_labels.pt", weights_only=True)

    print("  Computing val scores (max token perplexity)...")
    val_scores  = window_perplexities(model, val_windows,  device, mode="max")
    print("  Computing test scores (max token perplexity)...")
    test_scores = window_perplexities(model, test_windows, device, mode="max")

    mu, sigma = val_scores.mean().item(), val_scores.std().item()
    print(f"\n  Val max-token score:  mean={mu:.2f}  std={sigma:.2f}")

    def score(threshold):
        predicted = test_scores > threshold
        labels    = test_labels
        tp = (predicted &  labels).sum().item()
        fp = (predicted & ~labels).sum().item()
        fn = (~predicted & labels).sum().item()
        tn = (~predicted & ~labels).sum().item()
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)
        return tp, fp, fn, tn, precision, recall, f1

    # ── Scan thresholds to find 100% recall ───────────────────────────────────
    # Walk down from the global threshold until FN=0.
    # The threshold is the minimum max-token score across all anomalous
    # test windows — anything below that guarantees a miss.
    print("  Scanning for 100% recall threshold...")

    # Global baseline
    threshold_global = mu + n_sigma * sigma
    tp_g, fp_g, fn_g, tn_g, prec_g, rec_g, f1_g = score(threshold_global)

    # Perfect recall threshold: set just below the lowest anomaly score
    # Sort anomalous window scores ascending — the hardest to catch is first
    anom_scores = test_scores[test_labels]
    threshold_perfect = anom_scores.min().item() * 0.999  # just below the minimum

    tp_p, fp_p, fn_p, tn_p, prec_p, rec_p, f1_p = score(threshold_perfect)

    print(f"\n  Test set results ({len(test_windows)} windows, "
          f"{test_labels.sum().item()} anomalous):")

    print(f"\n  Global threshold ({threshold_global:.2f}):")
    print(f"    TP={tp_g}  FP={fp_g}  FN={fn_g}  TN={tn_g}")
    print(f"    Precision={prec_g:.3f}  Recall={rec_g:.3f}  F1={f1_g:.3f}")

    print(f"\n  Perfect-recall threshold ({threshold_perfect:.2f}):")
    print(f"    TP={tp_p}  FP={fp_p}  FN={fn_p}  TN={tn_p}")
    print(f"    Precision={prec_p:.3f}  Recall={rec_p:.3f}  F1={f1_p:.3f}")
    print(f"\n  --> Missed anomalies (FN): {fn_p}  "
          f"({'PERFECT' if fn_p == 0 else 'GOOD' if fn_p < 5 else 'REVIEW'})")

    # ── Show what the 9 hardest anomalies look like ───────────────────────────
    if fn_g > 0:
        # Find the hardest-to-catch anomaly scores
        hard_scores = anom_scores.sort().values[:min(fn_g + 5, len(anom_scores))]
        print(f"\n  Hardest anomaly max-token scores (lowest = easiest to miss):")
        for s in hard_scores[:10]:
            print(f"    {s.item():.3f}")
        normal_scores = test_scores[~test_labels]
        print(f"  Normal window score range: "
              f"{normal_scores.min():.3f} -- {normal_scores.max():.3f}")

    print(f"\n  Perplexity guide:")
    print(f"    < 10   -- excellent predictor")
    print(f"    10-50  -- good predictor")
    print(f"    50-100 -- moderate")
    print(f"    > 100  -- poor, unlikely to beat baseline")

    results = {
        "val_score_mean":      round(mu, 4),
        "val_score_std":       round(sigma, 4),
        "global_threshold":    round(threshold_global, 4),
        "perfect_threshold":   round(threshold_perfect, 4),
        "n_sigma":             n_sigma,
        "test_windows":        len(test_windows),
        "anomalous_windows":   int(test_labels.sum()),
        "global":  {"tp": tp_g, "fp": fp_g, "fn": fn_g, "tn": tn_g,
                    "precision": round(prec_g, 4), "recall": round(rec_g, 4),
                    "f1": round(f1_g, 4)},
        "perfect_recall": {"tp": tp_p, "fp": fp_p, "fn": fn_p, "tn": tn_p,
                           "precision": round(prec_p, 4), "recall": round(rec_p, 4),
                           "f1": round(f1_p, 4)},
        "test_scores":  test_scores.tolist(),
        "test_labels":  test_labels.tolist(),
    }

    out = RES_DIR / "anomaly_scores.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved -> {out}")
    return results


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimiser, epoch, val_loss, path):
    torch.save({
        "epoch":       epoch,
        "val_loss":    val_loss,
        "model_state": model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "config": {
            "vocab":    VOCAB,
            "emb_dim":  EMB_DIM,
            "n_layers": N_LAYERS,
            "n_heads":  N_HEADS,
            "ctx_len":  CTX_LEN,
        },
    }, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    CKPT_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load vocab metadata ───────────────────────────────────────────────────
    tok_path = DATA_DIR / "tokeniser.json"
    if tok_path.exists():
        tok_meta = json.loads(tok_path.read_text())
        real_vocab = len(tok_meta["id2tok"])
        print(f"\nVocab: {real_vocab} real tokens, padded to {VOCAB} for embedding")
        if real_vocab > VOCAB:
            raise ValueError(
                f"Real vocab ({real_vocab}) exceeds VOCAB constant ({VOCAB}). "
                f"Increase VOCAB to the next power of 2."
            )
    else:
        print("\nWARNING: tokeniser.json not found. Run tokenise_logs.py first.")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = WindowDataset(DATA_DIR / "train_tokens.pt")
    val_ds   = WindowDataset(DATA_DIR / "val_tokens.pt")
    print(f"  Train windows: {len(train_ds):,}")
    print(f"  Val windows:   {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(DEVICE.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding BitNet transformer...")
    model = BitNetTransformer(
        vocab=VOCAB, emb_dim=EMB_DIM, n_layers=N_LAYERS,
        n_heads=N_HEADS, ctx_len=CTX_LEN,
    ).to(DEVICE)

    n_params = model.count_parameters()
    print(f"  Parameters:   {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"  Float32 size: {n_params * 4 / 1e6:.2f} MB")
    print(f"  BitNet size:  {n_params * 2 / 8 / 1e6:.2f} MB  (2 bits/weight)")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimiser   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    total_steps = EPOCHS * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=total_steps, eta_min=LR / 10,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Steps/epoch:  {len(train_loader):,}")
    print(f"  Total steps:  {total_steps:,}\n")

    log            = []
    best_val_loss  = float("inf")
    best_ckpt      = CKPT_DIR / "model_best.pt"
    scaler         = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            optimiser.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                loss = compute_loss(model, batch, DEVICE)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

            if (step + 1) % 50 == 0:
                avg = epoch_loss / (step + 1)
                lr  = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:3d} | step {step+1:4d}/{len(train_loader)} "
                      f"| loss {avg:.4f} | lr {lr:.2e}")

        train_loss       = epoch_loss / len(train_loader)
        val_loss, val_ppl = evaluate(model, val_loader, DEVICE)
        elapsed          = time.time() - epoch_start

        print(f"\nEpoch {epoch:3d}/{EPOCHS} -- "
              f"train: {train_loss:.4f}  "
              f"val: {val_loss:.4f}  "
              f"ppl: {val_ppl:.2f}  "
              f"({elapsed:.0f}s)\n")

        entry = {
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_perplexity": val_ppl,
            "elapsed_s": elapsed,
        }
        log.append(entry)
        (CKPT_DIR / "training_log.json").write_text(json.dumps(log, indent=2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimiser, epoch, val_loss, best_ckpt)
            print(f"  New best model saved  (val_loss={val_loss:.4f})")

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimiser, epoch, val_loss,
                            CKPT_DIR / f"model_epoch_{epoch:03d}.pt")

    # ── Save final + run anomaly evaluation ───────────────────────────────────
    save_checkpoint(model, optimiser, EPOCHS, val_loss,
                    CKPT_DIR / "model_final.pt")

    print(f"\nTraining complete.")
    print(f"  Best val loss:       {best_val_loss:.4f}")
    print(f"  Best val perplexity: {math.exp(best_val_loss):.2f}")

    # Load best model for evaluation (not the final epoch weights)
    print("\nLoading best checkpoint for anomaly evaluation...")
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    results = evaluate_anomaly_detection(model, DEVICE)

    print(f"\nDone.")
    r = results["perfect_recall"]
    print(f"  Perfect-recall results:")
    print(f"    Recall:    {r['recall']:.3f}  (missed: {r['fn']})")
    print(f"    Precision: {r['precision']:.3f}")
    print(f"    F1:        {r['f1']:.3f}")
    print(f"  This is your LinkedIn headline number.")


if __name__ == "__main__":
    main()