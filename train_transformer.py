# train_transformer.py
# BitNet b1.58 decoder-only transformer for M365 audit log anomaly detection.
#
# Responsibility: training only.
# Anomaly detection, threshold tuning, and Stage 2 filtering live in detect.py.
#
# Architecture:
#   Vocabulary:      2048 tokens  (actual ~1150 with 1000 users, padded to next power of 2)
#   Embedding dim:   192
#   Layers:          4
#   Attention heads: 6
#   Context length:  1024 tokens  (~72 events of per-user history)
#   Parameters:      ~2M
#
# Training objective: next-token prediction (cross-entropy).
# Lower validation perplexity = better model of normal behaviour.
#
# Inputs:
#   data/train_tokens.pt    LongTensor (N, CTX_LEN)
#   data/val_tokens.pt      LongTensor (N, CTX_LEN)
#   data/tokeniser.json     vocab metadata
#
# Outputs:
#   checkpoints/model_best.pt        -- best val loss checkpoint
#   checkpoints/model_final.pt       -- final epoch checkpoint
#   checkpoints/model_epoch_NNN.pt   -- periodic checkpoints
#   checkpoints/training_log.json    -- per-epoch metrics

import argparse
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

# Model
VOCAB      = 2048
EMB_DIM    = 192
N_LAYERS   = 4
N_HEADS    = 6
CTX_LEN    = 1024

# Training
BATCH_SIZE        = 32
EPOCHS            = 60    # used for fresh runs
ADDITIONAL_EPOCHS = 40    # used when resuming with --resume
LR                = 1e-4
GRAD_CLIP         = 0.5
EVAL_EVERY        = 1
SAVE_EVERY        = 5
WARMUP_STEPS      = 500   # linear ramp before cosine decay kicks in

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
    def __init__(self, emb_dim: int, n_heads: int, ctx_len: int,
                 dropout: float = 0.1):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads
        self.qkv      = BitLinear(emb_dim, 3 * emb_dim, bias=False)
        self.out      = BitLinear(emb_dim, emb_dim,     bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
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
        attn    = self.attn_drop(F.softmax(scores, dim=-1))
        out     = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out(out))


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            BitLinear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            BitLinear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
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
        B, T = token_ids.shape
        pos  = torch.arange(T, device=token_ids.device)
        x    = self.token_emb(token_ids) + self.pos_emb(pos)
        x    = self.blocks(x)
        x    = self.norm(x)
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
    parser = argparse.ArgumentParser(description="LogSentinel — BitNet transformer training")
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            f"Resume from checkpoints/model_best.pt and train for "
            f"{ADDITIONAL_EPOCHS} more epochs. "
            f"Restores model + optimiser state exactly."
        ),
    )
    parser.add_argument(
        "--restart-lr", type=float, default=None, metavar="LR",
        help=(
            "Only valid with --resume. Start a fresh cosine decay from this LR "
            "instead of continuing the old schedule. Useful when the model is "
            "still underfitting and you want more aggressive learning on the "
            "continuation run. Recommended range: 3e-5 to 5e-5. "
            "Example: --resume --restart-lr 4e-5"
        ),
    )

    args = parser.parse_args()
    if args.restart_lr and not args.resume:
        parser.error("--restart-lr only makes sense with --resume")

    CKPT_DIR.mkdir(exist_ok=True)

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

    steps_per_epoch = len(train_loader)

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

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    # ── Resume or fresh start ─────────────────────────────────────────────────
    best_ckpt   = CKPT_DIR / "model_best.pt"
    start_epoch = 0
    best_val_loss = float("inf")
    log = []

    if args.resume:
        if not best_ckpt.exists():
            raise FileNotFoundError(
                f"--resume requires {best_ckpt} to exist. "
                f"Run without --resume first."
            )
        print(f"\nResuming from {best_ckpt}...")
        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optim_state"])
        start_epoch   = ckpt["epoch"]
        best_val_loss = ckpt["val_loss"]
        print(f"  Restored epoch {start_epoch}  val_loss={best_val_loss:.4f}")

        # Restore the training log so history is preserved
        log_path = CKPT_DIR / "training_log.json"
        if log_path.exists():
            log = json.loads(log_path.read_text())
            print(f"  Training log restored  ({len(log)} entries)")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    target_epochs = start_epoch + ADDITIONAL_EPOCHS if args.resume else EPOCHS
    total_steps   = target_epochs * steps_per_epoch
    steps_done    = start_epoch * steps_per_epoch

    LR_MIN = LR / 10

    if args.resume and args.restart_lr:
        # Cosine restart — fresh decay from restart_lr → LR_MIN over the new
        # 40 epochs. Weights are stable so no warmup needed. The optimiser LR
        # is patched to the restart value so AdamW momentum is preserved but
        # the step size is reset to where we want it.
        restart_lr    = args.restart_lr
        restart_steps = ADDITIONAL_EPOCHS * steps_per_epoch

        for pg in optimiser.param_groups:
            pg["lr"] = restart_lr

        def lr_lambda(step: int) -> float:
            progress = step / max(restart_steps, 1)
            cosine   = 0.5 * (1 + math.cos(math.pi * progress))
            return (LR_MIN + (restart_lr - LR_MIN) * cosine) / restart_lr

        # Fresh scheduler starting from step 0 of the restart
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimiser, lr_lambda=lr_lambda, last_epoch=-1,
        )
        print(f"\n  LR restart: {restart_lr:.1e} → {LR_MIN:.1e} "
              f"over {ADDITIONAL_EPOCHS} epochs  (fresh cosine)")

    else:
        # Continue the original cosine curve from where it left off.
        # last_epoch fast-forwards the scheduler to steps_done so the LR
        # picks up at the correct position on the decay curve.
        def lr_lambda(step: int) -> float:
            if step < WARMUP_STEPS:
                return (step + 1) / WARMUP_STEPS
            progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
            cosine   = 0.5 * (1 + math.cos(math.pi * progress))
            return (LR_MIN + (LR - LR_MIN) * cosine) / LR

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimiser, lr_lambda=lr_lambda,
            last_epoch=steps_done - 1 if steps_done > 0 else -1,
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    extra = ADDITIONAL_EPOCHS if args.resume else EPOCHS
    print(f"\n{'Resuming' if args.resume else 'Training'} for {extra} epochs "
          f"(epochs {start_epoch + 1} → {target_epochs})...")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Steps/epoch:  {steps_per_epoch:,}")
    print(f"  Total steps:  {target_epochs * steps_per_epoch:,}")
    if not args.resume:
        print(f"  Warmup steps: {WARMUP_STEPS}  (~{WARMUP_STEPS / steps_per_epoch:.1f} epochs)")
    elif not args.restart_lr:
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Resuming LR:  {current_lr:.2e}  (cosine position {steps_done}/{total_steps})")
    print()

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    for epoch in range(start_epoch + 1, target_epochs + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            optimiser.zero_grad()

            with torch.cuda.amp.autocast(enabled=False):
                loss = compute_loss(model, batch, DEVICE)

            if not torch.isfinite(loss):
                optimiser.zero_grad()
                scheduler.step()
                continue

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
                print(f"  Epoch {epoch:3d} | step {step+1:4d}/{steps_per_epoch} "
                      f"| loss {avg:.4f} | lr {lr:.2e}")

        train_loss        = epoch_loss / steps_per_epoch
        val_loss, val_ppl = evaluate(model, val_loader, DEVICE)
        elapsed           = time.time() - epoch_start

        print(f"\nEpoch {epoch:3d}/{target_epochs} -- "
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

    # ── Save final checkpoint ─────────────────────────────────────────────────
    save_checkpoint(model, optimiser, target_epochs, val_loss,
                    CKPT_DIR / "model_final.pt")

    print(f"\nTraining complete.")
    print(f"  Best val loss:       {best_val_loss:.4f}")
    print(f"  Best val perplexity: {math.exp(best_val_loss):.2f}")
    print(f"\n  Run detect.py to evaluate anomaly detection performance.")


if __name__ == "__main__":
    main()