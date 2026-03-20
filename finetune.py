"""
finetune.py
Per-tenant fine-tuning of the LogSentinel base model.

The base model learns universal M365 grammar without user identity tokens.
This script fine-tunes it on a specific tenant's logs by:
  1. Expanding the embedding table to include that tenant's user tokens
  2. Freezing all layers except user embeddings for the first N epochs
     (preserves base model knowledge, only learns who the users are)
  3. Unfreezing all layers for remaining epochs
     (refines the full model for this tenant's specific patterns)
  4. Calibrating per-user detection thresholds from the val set

Typical usage
-------------
Synthetic tenant test (prove the pipeline works):
  python generate_logs.py --seed 99 --n-users 30 --output-dir data/tenant_test
  python finetune.py --tenant-dir data/tenant_test --epochs 15

Real tenant (from M365 Unified Audit Log export):
  python finetune.py --tenant-dir /path/to/tenant_logs --epochs 15

Inputs
------
  checkpoints/model_best.pt          -- base model checkpoint
  --tenant-dir/train.jsonl           -- tenant training logs (normal only)
  --tenant-dir/val.jsonl             -- tenant validation logs (normal only)
  --tenant-dir/anomaly_test.jsonl    -- optional labeled test set

Outputs
-------
  --out-dir/model_finetuned.pt       -- fine-tuned model checkpoint
  --out-dir/tokeniser_tenant.json    -- tenant vocab (base + user tokens)
  --out-dir/user_thresholds.json     -- per-user calibrated thresholds
  --out-dir/finetune_log.json        -- per-epoch training metrics
  --out-dir/anomaly_scores.json      -- detection results if test set provided
"""

import argparse
import json
import math
import time
import hashlib
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_CKPT = Path("checkpoints/model_best.pt")
DATA_DIR  = Path("data")

# ── Hyperparameters ───────────────────────────────────────────────────────────

BATCH_SIZE          = 16    # smaller — tenant datasets are small
LR_FROZEN           = 2e-4  # higher LR while only user embeddings train
LR_UNFROZEN         = 3e-5  # conservative LR when full model trains
FREEZE_EPOCHS       = 5     # epochs with only user embeddings unfrozen
GRAD_CLIP           = 0.5
EARLY_STOP_PATIENCE = 5
CTX_LEN             = 1024
STRIDE              = 512
N_SIGMA             = 2.0

# ── Special tokens ────────────────────────────────────────────────────────────

PAD_STR = "<PAD>"
UNK_STR = "<UNK>"
BOS_STR = "<BOS>"
EOS_STR = "<EOS>"

# ── Model (must match train_transformer.py exactly) ───────────────────────────

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
        self.out      = BitLinear(emb_dim, emb_dim, bias=False)
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def expand_vocab(self, new_vocab_size, emb_dim):
        """
        Expand embedding table and lm_head to include new user tokens.
        Existing token rows are preserved exactly.
        New rows are randomly initialised so the model can learn them.
        """
        old_vocab = self.token_emb.num_embeddings
        if new_vocab_size <= old_vocab:
            return

        # Expand token_emb
        old_weight = self.token_emb.weight.data
        new_emb = nn.Embedding(new_vocab_size, emb_dim)
        nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
        new_emb.weight.data[:old_vocab] = old_weight
        self.token_emb = new_emb

        # Expand lm_head
        old_lm_weight = self.lm_head.weight.data
        new_lm = BitLinear(emb_dim, new_vocab_size, bias=False)
        nn.init.normal_(new_lm.weight, mean=0.0, std=0.02)
        new_lm.weight.data[:old_vocab] = old_lm_weight
        self.lm_head = new_lm

        print(f"  Vocab expanded: {old_vocab} -> {new_vocab_size}  "
              f"({new_vocab_size - old_vocab} new user tokens)")


# ── Tenant tokeniser ──────────────────────────────────────────────────────────

def user_hash(uid):
    if not uid:
        return "usr:????"
    return f"usr:{hashlib.md5(uid.encode()).hexdigest()[:4]}"


class TenantVocab:
    """
    Extends the base model vocab with tenant-specific user tokens.
    All base token IDs are preserved exactly.
    New usr:xxxx tokens are appended above the base vocab ceiling.
    """

    def __init__(self, base_vocab_path):
        data = json.loads(Path(base_vocab_path).read_text())
        self.tok2id   = dict(data["tok2id"])
        self.id2tok   = list(data["id2tok"])
        self.base_size = len(self.id2tok)

    def add_user_tokens(self, events):
        added = 0
        for e in events:
            tok = user_hash(e.get("UserId", ""))
            if tok not in self.tok2id:
                self.tok2id[tok] = len(self.id2tok)
                self.id2tok.append(tok)
                added += 1
        return added

    def encode(self, token):
        return self.tok2id.get(token, self.tok2id[UNK_STR])

    def encode_event(self, event):
        """Tokenise with usr:xxxx — reintroduced for fine-tuning."""
        ts = event.get("CreationTime", "")
        try:
            dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
            h  = dt.hour
            tb = ("tb:night"   if h < 6  else
                  "tb:earlyam" if h < 9  else
                  "tb:morning" if h < 12 else
                  "tb:work"    if h < 18 else "tb:evening")
            hr, dw = f"hr:{h}", f"dw:{dt.weekday()}"
        except (ValueError, IndexError):
            hr, dw, tb = "hr:--", "dw:--", "tb:--"

        workload = event.get("Workload", "UNKNOWN")

        ip = event.get("ClientIP", "")
        parts = ip.split(".") if ip else []
        ip_tok = f"ip:{parts[0]}.{parts[1]}" if len(parts) >= 2 else "ip:--"

        loc = event.get("Location")
        cc  = (f"cc:{loc['CountryCode']}"
               if isinstance(loc, dict) and loc.get("CountryCode") else "cc:--")

        rs_raw = event.get("ResultStatus", "")
        rs = f"rs:{rs_raw.lower()}" if rs_raw else "rs:--"

        props    = event.get("DeviceProperties", [])
        os_tok   = "os:--"
        comp_tok = "comp:--"
        if isinstance(props, list):
            for p in props:
                if not isinstance(p, dict):
                    continue
                if p.get("Name") == "OS":
                    v = p.get("Value", "")
                    os_tok = ("os:windows" if "Windows" in v else
                              "os:macos"   if "macOS"   in v else
                              "os:ios"     if "iOS"     in v else
                              "os:android" if "Android" in v else "os:other")
                if p.get("Name") == "IsCompliant":
                    comp_tok = f"comp:{p.get('Value','?').lower()}"

        fname = event.get("SourceFileName", "")
        if workload in ("SharePoint", "OneDrive") and fname:
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            ext_tok = ("ext:exec"        if ext in ("exe","ps1","bat","cmd","msi","vbs","js") else
                       "ext:spreadsheet" if ext in ("xlsx","xls","csv") else
                       "ext:document"    if ext in ("docx","doc","pdf","txt","md") else
                       "ext:archive"     if ext in ("zip","tar","gz","7z","rar") else
                       "ext:media"       if ext in ("png","jpg","jpeg","gif","mp4","mov") else
                       "ext:other" if ext else "ext:--")
        else:
            ext_tok = "ext:--"

        return [self.encode(t) for t in [
            BOS_STR,
            user_hash(event.get("UserId", "")),  # reintroduced for fine-tuning
            f"op:{event.get('Operation', 'UNKNOWN')}",
            f"wl:{workload}",
            f"ut:{event.get('UserType', 0)}",
            rs, ip_tok, cc, hr, dw, tb, os_tok, comp_tok, ext_tok,
        ]]

    @property
    def size(self):
        return len(self.id2tok)

    def save(self, path):
        Path(path).write_text(json.dumps({
            "tok2id":    self.tok2id,
            "id2tok":    self.id2tok,
            "base_size": self.base_size,
        }, indent=2))
        print(f"  Tenant vocab saved -> {path}  "
              f"({self.size} tokens, {self.size - self.base_size} new user tokens)")


# ── Window packing ────────────────────────────────────────────────────────────

PAD_ID = 0

def _sliding_windows(flat, ctx_len, stride, pad_id):
    if len(flat) < ctx_len:
        return [flat + [pad_id] * (ctx_len - len(flat))]
    windows = []
    for start in range(0, len(flat) - ctx_len + 1, stride):
        windows.append(flat[start : start + ctx_len])
    if len(flat) > ctx_len:
        last = flat[-ctx_len:]
        if last != windows[-1]:
            windows.append(last)
    return windows


def pack_windows(events, vocab, per_user=True):
    pad_id = vocab.tok2id[PAD_STR]
    if not per_user:
        events = sorted(events, key=lambda e: e.get("CreationTime", ""))
        flat = [tid for e in events for tid in vocab.encode_event(e)]
        return torch.tensor(_sliding_windows(flat, CTX_LEN, STRIDE, pad_id),
                            dtype=torch.long), []

    user_events = defaultdict(list)
    for e in events:
        user_events[e.get("UserId", "unknown")].append(e)

    all_windows, all_user_ids = [], []
    for uid in sorted(user_events):
        stream = sorted(user_events[uid], key=lambda e: e.get("CreationTime", ""))
        flat   = [tid for e in stream for tid in vocab.encode_event(e)]
        wins   = _sliding_windows(flat, CTX_LEN, STRIDE, pad_id)
        uid_h  = hashlib.md5(uid.encode()).hexdigest()[:4]
        all_windows.extend(wins)
        all_user_ids.extend([uid_h] * len(wins))

    return torch.tensor(all_windows, dtype=torch.long), all_user_ids


def pack_test_windows(events, vocab):
    pad_id = vocab.tok2id[PAD_STR]
    user_events = defaultdict(list)
    for e in events:
        user_events[e.get("UserId", "unknown")].append(e)

    all_windows, all_labels, all_user_ids = [], [], []
    for uid in sorted(user_events):
        uid_h  = hashlib.md5(uid.encode()).hexdigest()[:4]
        stream = sorted(user_events[uid], key=lambda e: e.get("CreationTime", ""))
        flat_tokens, flat_labels = [], []
        for e in stream:
            toks = vocab.encode_event(e)
            flat_tokens.extend(toks)
            flat_labels.extend([("_anomaly" in e)] * len(toks))

        if len(flat_tokens) < CTX_LEN:
            tok_c = flat_tokens + [pad_id] * (CTX_LEN - len(flat_tokens))
            lab_c = flat_labels + [False]  * (CTX_LEN - len(flat_labels))
            all_windows.append(tok_c)
            all_labels.append(any(lab_c))
            all_user_ids.append(uid_h)
            continue

        positions = list(range(0, len(flat_tokens) - CTX_LEN + 1, STRIDE))
        last_start = len(flat_tokens) - CTX_LEN
        if last_start not in positions:
            positions.append(last_start)
        for start in positions:
            all_windows.append(flat_tokens[start : start + CTX_LEN])
            all_labels.append(any(flat_labels[start : start + CTX_LEN]))
            all_user_ids.append(uid_h)

    return (torch.tensor(all_windows, dtype=torch.long),
            torch.tensor(all_labels,  dtype=torch.bool),
            all_user_ids)


# ── Dataset ───────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        w = self.windows[idx]
        return w[:-1], w[1:]


# ── Loss / eval ───────────────────────────────────────────────────────────────

def compute_loss(model, batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    logits = model(x)
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B*T, V), y.view(B*T), ignore_index=PAD_ID)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        total += compute_loss(model, batch, device).item()
        n += 1
    mean = total / max(n, 1)
    model.train()
    return mean, math.exp(mean)


@torch.no_grad()
def score_windows(model, windows, device, batch_size=64):
    model.eval()
    scores = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i : i + batch_size].to(device)
        x, y  = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        B, T, V = logits.shape
        token_losses = F.cross_entropy(
            logits.reshape(B*T, V), y.reshape(B*T),
            ignore_index=PAD_ID, reduction="none",
        ).reshape(B, T)
        mask = (y != PAD_ID).float()
        token_losses = token_losses * mask
        max_loss = token_losses.max(dim=1).values
        scores.append(torch.log1p(max_loss.exp()).cpu())
    model.train()
    return torch.cat(scores)


# ── Per-user threshold calibration ───────────────────────────────────────────

def calibrate_thresholds(model, val_events, vocab, device,
                          n_sigma=N_SIGMA, min_windows=10):
    val_windows, val_user_ids = pack_windows(val_events, vocab, per_user=True)
    val_scores = score_windows(model, val_windows, device)

    mu    = val_scores.mean().item()
    sigma = val_scores.std().item()
    global_threshold = mu + n_sigma * sigma

    user_scores = defaultdict(list)
    for score, uid in zip(val_scores.tolist(), val_user_ids):
        user_scores[uid].append(score)

    per_user = {}
    n_personal, n_fallback = 0, 0
    for uid, scores in user_scores.items():
        if len(scores) >= min_windows:
            u_mu  = sum(scores) / len(scores)
            u_std = (sum((s - u_mu)**2 for s in scores) / len(scores)) ** 0.5
            per_user[uid] = u_mu + n_sigma * u_std
            n_personal += 1
        else:
            per_user[uid] = global_threshold
            n_fallback += 1

    print(f"  Global threshold: {global_threshold:.3f}  "
          f"(mean={mu:.3f}  std={sigma:.3f})")
    print(f"  Per-user: {n_personal} personal, {n_fallback} global fallback")

    return global_threshold, per_user, {
        "val_mean":          round(mu, 4),
        "val_std":           round(sigma, 4),
        "global_threshold":  round(global_threshold, 4),
        "n_sigma":           n_sigma,
        "n_users_personal":  n_personal,
        "n_users_fallback":  n_fallback,
    }


# ── Detection evaluation ──────────────────────────────────────────────────────

def evaluate_detection(model, test_events, vocab, device,
                        global_threshold, per_user_thresholds):
    test_windows, test_labels, test_user_ids = pack_test_windows(test_events, vocab)
    test_scores = score_windows(model, test_windows, device)

    # Per-user thresholds
    predicted = torch.zeros(len(test_scores), dtype=torch.bool)
    for i, (score, uid) in enumerate(zip(test_scores.tolist(), test_user_ids)):
        thresh = per_user_thresholds.get(uid, global_threshold)
        predicted[i] = score > thresh

    labels = test_labels
    tp = (predicted &  labels).sum().item()
    fp = (predicted & ~labels).sum().item()
    fn = (~predicted & labels).sum().item()
    tn = (~predicted & ~labels).sum().item()
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)

    print(f"\n  {'='*50}")
    print(f"  Detection results — per-user thresholds")
    print(f"  {'='*50}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision: {prec:.3f}   Recall: {rec:.3f}   F1: {f1:.3f}")

    # Global threshold comparison
    pred_g = test_scores > global_threshold
    tp_g = (pred_g &  labels).sum().item()
    fp_g = (pred_g & ~labels).sum().item()
    fn_g = (~pred_g & labels).sum().item()
    prec_g = tp_g / max(tp_g + fp_g, 1)
    rec_g  = tp_g / max(tp_g + fn_g, 1)
    f1_g   = 2 * prec_g * rec_g / max(prec_g + rec_g, 1e-8)
    print(f"\n  Global threshold comparison:")
    print(f"  TP={tp_g}  FP={fp_g}  FN={fn_g}")
    print(f"  Precision: {prec_g:.3f}   Recall: {rec_g:.3f}   F1: {f1_g:.3f}")
    print(f"  {'='*50}")

    return {
        "per_user_thresholds": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
        },
        "global_threshold": {
            "tp": tp_g, "fp": fp_g, "fn": fn_g,
            "precision": round(prec_g, 4),
            "recall":    round(rec_g,  4),
            "f1":        round(f1_g,   4),
        },
        "test_scores":   test_scores.tolist(),
        "test_labels":   test_labels.tolist(),
        "test_user_ids": test_user_ids,
    }


# ── JSONL loader ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ── Training loop helper ──────────────────────────────────────────────────────

def run_epoch(model, loader, optimiser, device,
              freeze_base_rows=False, base_size=0):
    """
    Run one training epoch.
    freeze_base_rows=True: zero-grad the base vocab rows after backward
    so only new user token rows accumulate gradients.
    """
    model.train()
    epoch_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    for step, batch in enumerate(loader):
        optimiser.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            loss = compute_loss(model, batch, device)
        if not torch.isfinite(loss):
            optimiser.zero_grad()
            continue
        scaler.scale(loss).backward()

        if freeze_base_rows and base_size > 0:
            with torch.no_grad():
                if model.token_emb.weight.grad is not None:
                    model.token_emb.weight.grad[:base_size] = 0
                if model.lm_head.weight.grad is not None:
                    model.lm_head.weight.grad[:base_size] = 0

        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimiser)
        scaler.update()
        epoch_loss += loss.item()

        if (step + 1) % 20 == 0:
            avg = epoch_loss / (step + 1)
            print(f"    step {step+1:4d}/{len(loader)}  loss {avg:.4f}")

    return epoch_loss / max(len(loader), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LogSentinel per-tenant fine-tuning"
    )
    parser.add_argument("--tenant-dir",   required=True)
    parser.add_argument("--base-ckpt",    default=str(BASE_CKPT))
    parser.add_argument("--base-vocab",   default=str(DATA_DIR / "tokeniser.json"))
    parser.add_argument("--out-dir",      default=None)
    parser.add_argument("--epochs",       type=int,   default=500,
                        help="Safety cap on total epochs (default: 500). "
                             "Training stops early after 5 epochs with no improvement.")
    parser.add_argument("--freeze-epochs",type=int,   default=FREEZE_EPOCHS)
    parser.add_argument("--n-sigma",      type=float, default=N_SIGMA)
    parser.add_argument("--min-windows",  type=int,   default=10)
    args = parser.parse_args()

    tenant_dir = Path(args.tenant_dir)
    out_dir    = Path(args.out_dir) if args.out_dir else tenant_dir / "finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  LogSentinel -- Fine-tuning")
    print("=" * 60)
    print(f"  Tenant dir:     {tenant_dir}")
    print(f"  Base ckpt:      {args.base_ckpt}")
    print(f"  Output dir:     {out_dir}")
    print(f"  Device:         {device}")
    print(f"  Total epochs:   {args.epochs}")
    print(f"  Frozen epochs:  {args.freeze_epochs}  "
          f"(user embeddings only)")
    print(f"  Unfrozen epochs:{args.epochs - args.freeze_epochs}  "
          f"(full model)")
    print()

    # ── Load tenant data ──────────────────────────────────────────────────────
    train_path = tenant_dir / "train.jsonl"
    val_path   = tenant_dir / "val.jsonl"
    test_path  = tenant_dir / "anomaly_test.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"train.jsonl not found in {tenant_dir}")
    if not val_path.exists():
        raise FileNotFoundError(f"val.jsonl not found in {tenant_dir}")

    print("Loading tenant data...")
    train_events = load_jsonl(train_path)
    val_events   = load_jsonl(val_path)
    test_events  = load_jsonl(test_path) if test_path.exists() else None
    print(f"  Train: {len(train_events):,} events")
    print(f"  Val:   {len(val_events):,} events")
    if test_events:
        n_anom = sum(1 for e in test_events if "_anomaly" in e)
        print(f"  Test:  {len(test_events):,} events  ({n_anom} anomalous)")

    # ── Build tenant vocab ────────────────────────────────────────────────────
    print("\nBuilding tenant vocab...")
    vocab = TenantVocab(args.base_vocab)
    all_events = train_events + val_events + (test_events or [])
    n_added = vocab.add_user_tokens(all_events)
    print(f"  Base vocab:  {vocab.base_size} tokens")
    print(f"  New user tokens: {n_added}  ->  total {vocab.size}")
    vocab.save(out_dir / "tokeniser_tenant.json")

    # ── Load and expand base model ────────────────────────────────────────────
    print(f"\nLoading base model from {args.base_ckpt}...")
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=True)
    cfg  = ckpt["config"]
    print(f"  Base: vocab={cfg['vocab']} emb={cfg['emb_dim']} "
          f"layers={cfg['n_layers']} heads={cfg['n_heads']}")

    model = BitNetTransformer(
        vocab=cfg["vocab"], emb_dim=cfg["emb_dim"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        ctx_len=cfg["ctx_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.expand_vocab(vocab.size, cfg["emb_dim"])
    model = model.to(device)

    # ── Pack windows ──────────────────────────────────────────────────────────
    print("\nPacking training windows...")
    train_windows, _ = pack_windows(train_events, vocab, per_user=True)
    val_windows,   _ = pack_windows(val_events,   vocab, per_user=True)
    print(f"  Train: {len(train_windows):,} windows")
    print(f"  Val:   {len(val_windows):,} windows")

    if len(train_windows) < 10:
        print("  WARNING: very few training windows — results may be unreliable")
        print("  Recommend at least 30 days of tenant logs for meaningful fine-tuning")

    train_loader = DataLoader(WindowDataset(train_windows),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(WindowDataset(val_windows),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Training ──────────────────────────────────────────────────────────────
    log               = []
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    best_ckpt_path    = out_dir / "model_finetuned.pt"

    def save_ckpt(epoch, val_loss):
        torch.save({
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model_state": model.state_dict(),
            "config":     cfg,
            "vocab_size": vocab.size,
        }, best_ckpt_path)

    # ── Phase 1: frozen — only user embeddings update ─────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1 -- Frozen ({args.freeze_epochs} epochs, LR={LR_FROZEN:.0e})")
    print(f"  Only new user token rows update.")
    print(f"  Base model knowledge is fully preserved.")
    print(f"{'='*60}\n")

    for param in model.parameters():
        param.requires_grad = False
    model.token_emb.weight.requires_grad = True
    model.lm_head.weight.requires_grad   = True

    optimiser = torch.optim.AdamW(
        [model.token_emb.weight, model.lm_head.weight],
        lr=LR_FROZEN, weight_decay=0.01,
    )

    for epoch in range(1, args.freeze_epochs + 1):
        t0         = time.time()
        train_loss = run_epoch(model, train_loader, optimiser, device,
                               freeze_base_rows=True, base_size=vocab.base_size)
        val_loss, val_ppl = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"\n[Frozen] Epoch {epoch}/{args.freeze_epochs} -- "
              f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
              f"ppl: {val_ppl:.2f}  ({elapsed:.0f}s)")

        entry = {"epoch": epoch, "phase": "frozen",
                 "train_loss": train_loss, "val_loss": val_loss,
                 "val_perplexity": val_ppl}
        log.append(entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_ckpt(epoch, val_loss)
            print(f"  New best saved  (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement -- patience {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

    # ── Phase 2: unfrozen — full model updates ────────────────────────────────
    unfrozen_epochs = args.epochs - args.freeze_epochs
    if unfrozen_epochs > 0:
        print(f"\n{'='*60}")
        print(f"  Phase 2 -- Unfrozen (until {EARLY_STOP_PATIENCE} epochs no improvement, LR={LR_UNFROZEN:.0e})")
        print(f"  All parameters train. Conservative LR preserves base knowledge.")
        print(f"{'='*60}\n")

        for param in model.parameters():
            param.requires_grad = True

        optimiser = torch.optim.AdamW(
            model.parameters(), lr=LR_UNFROZEN, weight_decay=0.01,
        )
        # Cosine decay over unfrozen epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=unfrozen_epochs * len(train_loader),
            eta_min=LR_UNFROZEN / 10,
        )

        epochs_no_improve = 0  # reset patience for phase 2

        for epoch in range(args.freeze_epochs + 1, args.epochs + 1):
            t0         = time.time()
            train_loss = run_epoch(model, train_loader, optimiser, device,
                                   freeze_base_rows=False)
            scheduler.step()
            val_loss, val_ppl = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            print(f"\n[Unfrozen] Epoch {epoch}/{args.epochs} -- "
                  f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
                  f"ppl: {val_ppl:.2f}  ({elapsed:.0f}s)")

            entry = {"epoch": epoch, "phase": "unfrozen",
                     "train_loss": train_loss, "val_loss": val_loss,
                     "val_perplexity": val_ppl}
            log.append(entry)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_ckpt(epoch, val_loss)
                print(f"  New best saved  (val_loss={val_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  No improvement -- patience "
                      f"{epochs_no_improve}/{EARLY_STOP_PATIENCE}")
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"\n  Early stopping triggered.")
                    break

    # ── Save training log ─────────────────────────────────────────────────────
    (out_dir / "finetune_log.json").write_text(json.dumps(log, indent=2))

    print(f"\nFine-tuning complete.")
    print(f"  Best val loss: {best_val_loss:.4f}  "
          f"(ppl={math.exp(best_val_loss):.2f})")

    # ── Load best checkpoint for evaluation ───────────────────────────────────
    print(f"\nLoading best checkpoint for calibration + evaluation...")
    best = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])

    # ── Calibrate per-user thresholds ─────────────────────────────────────────
    print(f"\nCalibrating per-user thresholds from val set...")
    global_thresh, per_user_thresh, thresh_stats = calibrate_thresholds(
        model, val_events, vocab, device,
        n_sigma=args.n_sigma, min_windows=args.min_windows,
    )

    thresholds_out = {
        "stats":     thresh_stats,
        "per_user":  per_user_thresh,
        "global":    round(global_thresh, 4),
    }
    (out_dir / "user_thresholds.json").write_text(
        json.dumps(thresholds_out, indent=2))
    print(f"  Thresholds saved -> {out_dir / 'user_thresholds.json'}")

    # ── Evaluate on test set if available ─────────────────────────────────────
    if test_events:
        print(f"\nEvaluating on test set...")
        detection_results = evaluate_detection(
            model, test_events, vocab, device,
            global_thresh, per_user_thresh,
        )
        detection_results["val_mean"] = thresh_stats["val_mean"]
        detection_results["val_std"]  = thresh_stats["val_std"]
        (out_dir / "anomaly_scores.json").write_text(
            json.dumps(detection_results, indent=2))
        print(f"  Results saved -> {out_dir / 'anomaly_scores.json'}")
    else:
        print(f"\nNo anomaly_test.jsonl found — skipping detection evaluation.")
        print(f"  To evaluate: add anomaly_test.jsonl to {tenant_dir}")

    print(f"\nDone. Fine-tuned model in {out_dir}/")


if __name__ == "__main__":
    main()
