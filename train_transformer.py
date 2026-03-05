# train_transformer.py
# Step 4: Train a BitNet b1.58 decoder-only transformer on tokenised neutron frames
#
# Architecture (per roadmap):
#   Vocabulary:     4,096 tokens
#   Embedding dim:  256
#   Layers:         6
#   Attention heads: 4
#   Context length: 512 tokens
#   Parameters:     ~20-30M
#
# BitLinear uses ternary weights {-1, 0, +1} via absmean quantisation +
# straight-through estimator for gradients. Weights are float32 during
# training — ternary efficiency comes at deployment via bitnet.cpp.
#
# Objective: next-token prediction (cross-entropy).
# Lower validation perplexity = better predictor = better compressor.
#
# Outputs: checkpoints/model.pt, checkpoints/training_log.json

import math
import json
import time
import numpy as np
import h5py
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenise import (
    encode_frame,
    load_tokeniser,
    make_kdtree,
    SPECIAL,
    BIN_ID_OFFSET,
    VOCAB_SIZE,
)

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_FILE  = Path("data/events_train.h5")
VAL_FILE    = Path("data/events_val.h5")
CKPT_DIR    = Path("checkpoints")

# Model
VOCAB       = VOCAB_SIZE    # 4096
EMB_DIM     = 256
N_LAYERS    = 6
N_HEADS     = 4
CTX_LEN     = 512           # max tokens per context window

# Training
BATCH_SIZE  = 32
LR          = 2e-4          # slightly lower than standard — BitNet benefit
EPOCHS      = 50
GRAD_CLIP   = 1.0
EVAL_EVERY  = 1             # evaluate on val set every N epochs
SAVE_EVERY  = 5             # save checkpoint every N epochs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── BitLinear ─────────────────────────────────────────────────────────────────

class BitLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear with ternary weights {-1, 0, +1}.
    Forward pass: quantise weights via absmean scheme.
    Backward pass: straight-through estimator — gradients flow as if float.
    This is Microsoft BitNet b1.58.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight

        # Absmean quantisation: scale so mean(|w|) = 1, then round and clamp
        scale    = w.abs().mean().clamp(min=1e-8)
        w_ternary = (w / scale).round().clamp(-1, 1)

        # Straight-through estimator: forward uses ternary, backward uses float
        w_quantised = w + (w_ternary - w).detach()

        return F.linear(x, w_quantised, self.bias)


# ── Transformer blocks ────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention using BitLinear projections.
    Causal mask ensures token i only attends to tokens 0..i (autoregressive).
    """
    def __init__(self, emb_dim: int, n_heads: int, ctx_len: int):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads

        # Q, K, V projections — all BitLinear
        self.qkv = BitLinear(emb_dim, 3 * emb_dim, bias=False)
        self.out  = BitLinear(emb_dim, emb_dim,     bias=False)

        # Causal mask — upper triangle = -inf, registered as buffer (not a param)
        mask = torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V and split heads
        qkv = self.qkv(x)                                    # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)                        # each (B, T, C)

        # Reshape to (B, heads, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        scale  = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale           # (B, heads, T, T)

        # Apply causal mask — mask future positions
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn   = F.softmax(scores, dim=-1)

        # Weighted sum of values
        out = (attn @ v)                                      # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    """Standard 4x expansion FFN using BitLinear layers."""
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
    """One transformer layer: pre-norm attention + pre-norm FFN."""
    def __init__(self, emb_dim: int, n_heads: int, ctx_len: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn  = CausalSelfAttention(emb_dim, n_heads, ctx_len)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ffn   = FeedForward(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # residual connection
        x = x + self.ffn(self.norm2(x))    # residual connection
        return x


class BitNetTransformer(nn.Module):
    """
    Decoder-only GPT-style transformer with BitLinear layers.
    Predicts probability distribution over next token at each position.
    """
    def __init__(self, vocab: int, emb_dim: int, n_layers: int,
                 n_heads: int, ctx_len: int):
        super().__init__()
        self.ctx_len = ctx_len

        self.token_emb = nn.Embedding(vocab, emb_dim)
        self.pos_emb   = nn.Embedding(ctx_len, emb_dim)

        self.blocks = nn.Sequential(*[
            TransformerBlock(emb_dim, n_heads, ctx_len)
            for _ in range(n_layers)
        ])

        self.norm    = nn.LayerNorm(emb_dim)
        self.lm_head = BitLinear(emb_dim, vocab, bias=False)

        # Weight tying: token embedding and lm_head share weights
        # Standard practice — reduces parameters, improves generalisation
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Small initialisations — important for BitNet stability."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, BitLinear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, T) integer tensor
        Returns logits: (B, T, vocab_size)
        """
        B, T = token_ids.shape
        assert T <= self.ctx_len, f"Sequence length {T} exceeds context {self.ctx_len}"

        positions = torch.arange(T, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(positions)  # (B, T, C)
        x = self.blocks(x)
        x = self.norm(x)
        return self.lm_head(x)                                    # (B, T, vocab)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Dataset ───────────────────────────────────────────────────────────────────

class NeutronFrameDataset(Dataset):
    """
    Loads neutron pulse frames, tokenises them, and returns
    fixed-length context windows for transformer training.

    Each frame is tokenised → token sequence of variable length.
    Long sequences are split into CTX_LEN chunks.
    Short sequences are padded to CTX_LEN.
    """
    def __init__(self, h5_path: Path, kdtree, bin_centres: np.ndarray,
                 tof_scale: float, ctx_len: int = CTX_LEN):
        self.ctx_len     = ctx_len
        self.bin_centres = bin_centres
        self.tof_scale   = tof_scale
        self.kdtree      = kdtree

        # Tokenise all frames and collect context windows
        print(f"  Tokenising {h5_path.name}...")
        self.windows = []
        self._tokenise_all(h5_path)
        print(f"  {len(self.windows):,} context windows created")

    def _tokenise_all(self, path: Path):
        with h5py.File(path, "r") as f:
            n = f.attrs["n_frames"]
            for i in range(n):
                frame = f["frames"][str(i)][:]
                tokens, _ = encode_frame(
                    frame, self.kdtree, self.bin_centres,
                    self.tof_scale, mode_token=SPECIAL["MODE_PRIOR"]
                )
                # Split into CTX_LEN chunks
                for start in range(0, len(tokens), self.ctx_len):
                    chunk = tokens[start:start + self.ctx_len]
                    # Pad if shorter than ctx_len
                    if len(chunk) < self.ctx_len:
                        chunk = chunk + [SPECIAL["PAD"]] * (self.ctx_len - len(chunk))
                    self.windows.append(torch.tensor(chunk, dtype=torch.long))

                if (i + 1) % 10000 == 0:
                    print(f"    {i+1}/{n} frames tokenised...")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        tokens = self.windows[idx]                  # (CTX_LEN,)
        x = tokens[:-1]                             # input:  all but last
        y = tokens[1:]                              # target: all but first
        return x, y


# ── Training helpers ──────────────────────────────────────────────────────────

def compute_loss(model: BitNetTransformer, batch: tuple,
                 device: torch.device) -> torch.Tensor:
    x, y = batch
    x, y = x.to(device), y.to(device)
    logits = model(x)                               # (B, T-1, vocab)

    # Flatten for cross-entropy: ignore PAD tokens in loss
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, V),
        y.view(B * T),
        ignore_index=SPECIAL["PAD"]
    )
    return loss


@torch.no_grad()
def evaluate(model: BitNetTransformer, loader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    """Returns (mean_loss, perplexity) on the given dataloader."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        loss = compute_loss(model, batch, device)
        total_loss += loss.item()
        n_batches  += 1
    mean_loss   = total_loss / max(n_batches, 1)
    perplexity  = math.exp(mean_loss)
    model.train()
    return mean_loss, perplexity


def save_checkpoint(model: BitNetTransformer, optimiser: torch.optim.Optimizer,
                    epoch: int, val_loss: float, path: Path):
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
        }
    }, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    CKPT_DIR.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load tokeniser ────────────────────────────────────────────────────────
    print("\nLoading tokeniser...")
    tok         = load_tokeniser()
    bin_centres = tok["bin_centres"]
    tof_scale   = tok["tof_scale"]
    kdtree      = make_kdtree(bin_centres, tof_scale)
    print(f"  Bins: {len(bin_centres)}  tof_scale: {tof_scale:.4f}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nBuilding training dataset...")
    train_ds = NeutronFrameDataset(TRAIN_FILE, kdtree, bin_centres, tof_scale)
    print("\nBuilding validation dataset...")
    val_ds   = NeutronFrameDataset(VAL_FILE,   kdtree, bin_centres, tof_scale)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(DEVICE.type == "cuda")
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding BitNet transformer...")
    model = BitNetTransformer(
        vocab=VOCAB, emb_dim=EMB_DIM, n_layers=N_LAYERS,
        n_heads=N_HEADS, ctx_len=CTX_LEN
    ).to(DEVICE)

    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}  ({n_params/1e6:.1f}M)")
    print(f"  Float32 size: {n_params * 4 / 1e6:.1f} MB")
    print(f"  BitNet size:  {n_params * 2 / 8 / 1e6:.1f} MB  (2 bits/weight)")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=total_steps, eta_min=LR / 10
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE}  Steps/epoch: {len(train_loader):,}")
    print(f"  Total steps: {total_steps:,}\n")

    log = []
    best_val_loss  = float("inf")
    best_ckpt_path = CKPT_DIR / "model_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            optimiser.zero_grad()
            loss = compute_loss(model, batch, DEVICE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            scheduler.step()
            epoch_loss += loss.item()

            # Print step-level progress every 200 steps
            if (step + 1) % 200 == 0:
                avg = epoch_loss / (step + 1)
                lr  = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:3d} | step {step+1:5d}/{len(train_loader)} "
                      f"| loss {avg:.4f} | lr {lr:.2e}")

        train_loss = epoch_loss / len(train_loader)
        elapsed    = time.time() - epoch_start

        # Validation
        val_loss, val_ppl = evaluate(model, val_loader, DEVICE)

        print(f"\nEpoch {epoch:3d}/{EPOCHS} — "
              f"train loss: {train_loss:.4f}  "
              f"val loss: {val_loss:.4f}  "
              f"val perplexity: {val_ppl:.2f}  "
              f"({elapsed:.0f}s)\n")

        # Log
        entry = {
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_perplexity": val_ppl,
            "elapsed_s": elapsed,
        }
        log.append(entry)
        (CKPT_DIR / "training_log.json").write_text(json.dumps(log, indent=2))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimiser, epoch, val_loss, best_ckpt_path)
            print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            path = CKPT_DIR / f"model_epoch_{epoch:03d}.pt"
            save_checkpoint(model, optimiser, epoch, val_loss, path)

    # Save final model
    save_checkpoint(model, optimiser, EPOCHS, val_loss,
                    CKPT_DIR / "model_final.pt")

    print(f"\nTraining complete.")
    print(f"  Best val loss:       {best_val_loss:.4f}")
    print(f"  Best val perplexity: {math.exp(best_val_loss):.2f}")
    print(f"  Best model saved →   {best_ckpt_path}")
    print(f"\n  Perplexity guide:")
    print(f"    < 10   — excellent predictor, expect strong compression gains")
    print(f"    10–50  — good predictor, expect meaningful gains over gzip")
    print(f"    50–100 — moderate predictor, marginal gains")
    print(f"    > 100  — poor predictor, unlikely to beat gzip")


if __name__ == "__main__":
    main()