"""
multiscale_score.py
Multi-scale anomaly scoring for LogSentinel.

Scores the test event stream at multiple window sizes using the existing
finetuned model — no retraining required. The transformer supports any
window size from 1 to ctx_len via its learned positional embeddings.

Why this helps
--------------
Burst anomalies (mass_download, brute_force) inject 8-12 anomalous events
into a user's stream. In a CTX_LEN=1024 window (~73 events) they represent
<15% of the context and get diluted. In a CTX_LEN=84 window (~6 events)
they dominate >50% of the context and spike perplexity.

Algorithm
---------
For each scale (window size) S:
  1. Score the val event stream → calibrate val_mean_S, val_std_S
  2. Score the test event stream → z_S(event) = (perplexity - val_mean_S) / val_std_S
  3. Each event gets z_S from the highest-scoring window it appears in

Final event score = max z-score across all scales.
Windows are re-assembled at CTX_LEN=1024 for compatibility; each window
inherits the max event z-score within it.

Output scores are z-scores (val_mean=0.0, val_std=1.0), so sigma thresholds
map directly: sigma=2.0 → flag anything 2 standard deviations above val mean.

Usage:
    python multiscale_score.py
    python multiscale_score.py --tenant-dir data/tenant_test --sigma 2.0
    python multiscale_score.py --scales 42,84,168,336,672,1024

Output:
    data/tenant_test/finetuned/anomaly_scores_multiscale.json
    (compatible with detect.py --scores-file and stage2_filter.py --scores-file)
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────────────
TOKENS_PER_EVENT  = 14        # BOS + 13 field tokens — constant for all events
DEFAULT_SCALES    = [42, 84, 168, 336, 672, 1024]
CANONICAL_CTX     = 1024      # output windows use this size (matches finetune.py)
CANONICAL_STRIDE  = 512
PAD_ID            = 0
BOS_STR           = "<BOS>"
UNK_STR           = "<UNK>"
PAD_STR           = "<PAD>"


# ── Model definition (must match train_transformer.py / finetune.py) ───────────

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


# ── Vocab ──────────────────────────────────────────────────────────────────────

class Vocab:
    """Lightweight vocab loader — encode only, no training."""
    def __init__(self, vocab_path):
        data = json.loads(Path(vocab_path).read_text())
        self.tok2id = dict(data["tok2id"])
        self.unk_id = self.tok2id.get(UNK_STR, 1)

    def encode(self, token):
        return self.tok2id.get(token, self.unk_id)

    def encode_event(self, event):
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

        uid_tok = f"usr:{hashlib.md5(event.get('UserId','').encode()).hexdigest()[:4]}"
        return [self.encode(t) for t in [
            BOS_STR, uid_tok,
            f"op:{event.get('Operation', 'UNKNOWN')}",
            f"wl:{workload}",
            f"ut:{event.get('UserType', 0)}",
            rs, ip_tok, cc, hr, dw, tb, os_tok, comp_tok, ext_tok,
        ]]


# ── Windowing ──────────────────────────────────────────────────────────────────

def make_windows(flat_tokens, ctx_len, stride):
    """
    Slice flat_tokens into overlapping windows of ctx_len.
    Returns (windows_tensor, list of (event_start, event_end) per window).
    Event boundaries are inferred from TOKENS_PER_EVENT.
    """
    n = len(flat_tokens)
    windows, event_ranges = [], []

    if n < ctx_len:
        padded = flat_tokens + [PAD_ID] * (ctx_len - n)
        windows.append(padded)
        event_ranges.append((0, n // TOKENS_PER_EVENT))
    else:
        positions = list(range(0, n - ctx_len + 1, stride))
        if not positions or positions[-1] != n - ctx_len:
            positions.append(n - ctx_len)
        for start in positions:
            windows.append(flat_tokens[start : start + ctx_len])
            e_start = start // TOKENS_PER_EVENT
            e_end   = (start + ctx_len) // TOKENS_PER_EVENT
            event_ranges.append((e_start, e_end))

    return torch.tensor(windows, dtype=torch.long), event_ranges


# ── Scoring ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_windows(model, windows, device, batch_size=64):
    """Max-token perplexity score per window — matches finetune.py score_windows."""
    model.eval()
    scores = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i : i + batch_size].to(device)
        x, y  = batch[:, :-1], batch[:, 1:]
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def group_by_user(events):
    """Returns dict of uid -> events sorted by CreationTime."""
    user_events = defaultdict(list)
    for e in events:
        user_events[e.get("UserId", "")].append(e)
    return {
        uid: sorted(evs, key=lambda e: e.get("CreationTime", ""))
        for uid, evs in user_events.items()
    }


def metrics(scores, labels, threshold):
    tp = sum(s > threshold and l for s, l in zip(scores, labels))
    fp = sum(s > threshold and not l for s, l in zip(scores, labels))
    fn = sum(s <= threshold and l for s, l in zip(scores, labels))
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return dict(tp=tp, fp=fp, fn=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LogSentinel — multi-scale anomaly scoring (no retraining)"
    )
    parser.add_argument("--tenant-dir",    default="data/tenant_test",
                        help="Tenant data directory")
    parser.add_argument("--finetuned-dir", default=None,
                        help="Finetuned model dir (default: <tenant-dir>/finetuned)")
    parser.add_argument("--sigma",         type=float, default=2.0,
                        help="Threshold for quick evaluation summary")
    parser.add_argument("--scales",        default=",".join(map(str, DEFAULT_SCALES)),
                        help="Comma-separated window sizes to score at")
    args = parser.parse_args()

    tenant_dir    = Path(args.tenant_dir)
    finetuned_dir = Path(args.finetuned_dir) if args.finetuned_dir \
                    else tenant_dir / "finetuned"
    scales        = [int(s) for s in args.scales.split(",")]

    print("=" * 66)
    print("  LogSentinel — Multi-Scale Scoring")
    print("=" * 66)
    print(f"  Tenant:  {tenant_dir}")
    print(f"  Model:   {finetuned_dir / 'model_finetuned.pt'}")
    print(f"  Scales:  {scales}  (tokens per window)")
    print(f"  Events/window at each scale: "
          f"{[s // TOKENS_PER_EVENT for s in scales]}")

    # ── Load model ────────────────────────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = finetuned_dir / "model_finetuned.pt"
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg       = ckpt["config"]
    vocab_size = ckpt["vocab_size"]

    model = BitNetTransformer(
        vocab=vocab_size, emb_dim=cfg["emb_dim"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        ctx_len=cfg["ctx_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n  Model loaded  (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}, ctx_len={cfg['ctx_len']})")

    # ── Load vocab and events ─────────────────────────────────────────────────
    vocab       = Vocab(finetuned_dir / "tokeniser_tenant.json")
    val_events  = load_jsonl(tenant_dir / "val.jsonl")
    test_events = load_jsonl(tenant_dir / "anomaly_test.jsonl")

    val_users  = group_by_user(val_events)
    test_users = group_by_user(test_events)

    # Build absolute event index for test set (sorted users, sorted events)
    abs_idx = {}   # (uid, local_event_i) -> absolute index
    n_abs = 0
    for uid in sorted(test_users):
        for local_i in range(len(test_users[uid])):
            abs_idx[(uid, local_i)] = n_abs
            n_abs += 1

    # per_event_z[abs_i] = max z-score across all scales
    per_event_z = [-1e9] * n_abs

    # ── Score at each scale ───────────────────────────────────────────────────
    print()
    for scale in scales:
        # stride aligned to event boundary (multiple of TOKENS_PER_EVENT)
        raw_stride = scale // 2
        stride = max(
            (raw_stride // TOKENS_PER_EVENT) * TOKENS_PER_EVENT,
            TOKENS_PER_EVENT,
        )

        # Calibrate on val set
        val_scores_list = []
        for uid in sorted(val_users):
            flat = []
            for e in val_users[uid]:
                flat.extend(vocab.encode_event(e))
            if not flat:
                continue
            wins, _ = make_windows(flat, scale, stride)
            val_scores_list.extend(score_windows(model, wins, device).tolist())

        if not val_scores_list:
            print(f"  Scale {scale:5d}: no val windows — skipping")
            continue

        val_mean_s = sum(val_scores_list) / len(val_scores_list)
        val_std_s  = max(
            (sum((s - val_mean_s) ** 2 for s in val_scores_list)
             / max(len(val_scores_list) - 1, 1)) ** 0.5,
            1e-6,
        )
        print(f"  Scale {scale:5d} ({scale//TOKENS_PER_EVENT:2d} events/win): "
              f"val_mean={val_mean_s:.3f}  val_std={val_std_s:.3f}  "
              f"({len(val_scores_list)} val windows)")

        # Score test events, propagate max z to each event
        for uid in sorted(test_users):
            flat = []
            for e in test_users[uid]:
                flat.extend(vocab.encode_event(e))
            if not flat:
                continue
            wins, event_ranges = make_windows(flat, scale, stride)
            win_scores = score_windows(model, wins, device).tolist()

            for win_score, (e_start, e_end) in zip(win_scores, event_ranges):
                z = (win_score - val_mean_s) / val_std_s
                n_user_events = len(test_users[uid])
                for local_i in range(e_start, min(e_end, n_user_events)):
                    key = (uid, local_i)
                    if key in abs_idx:
                        ai = abs_idx[key]
                        if z > per_event_z[ai]:
                            per_event_z[ai] = z

    # ── Re-assemble into canonical windows ────────────────────────────────────
    print("\n  Re-assembling canonical windows "
          f"(CTX={CANONICAL_CTX}, stride={CANONICAL_STRIDE})...")

    test_scores, test_labels, test_user_ids = [], [], []

    for uid in sorted(test_users):
        events = test_users[uid]
        flat_tokens  = []
        flat_anomaly = []
        for local_i, e in enumerate(events):
            toks = vocab.encode_event(e)
            flat_tokens.extend(toks)
            flat_anomaly.extend([("_anomaly" in e)] * len(toks))

        n = len(flat_tokens)
        uid_h = hashlib.md5(uid.encode()).hexdigest()[:4]
        n_events = len(events)

        if n < CANONICAL_CTX:
            # Single padded window
            e_end = n // TOKENS_PER_EVENT
            z_vals = [per_event_z[abs_idx[(uid, i)]]
                      for i in range(e_end) if (uid, i) in abs_idx]
            test_scores.append(max(z_vals) if z_vals else 0.0)
            test_labels.append(any(flat_anomaly))
            test_user_ids.append(uid_h)
        else:
            positions = list(range(0, n - CANONICAL_CTX + 1, CANONICAL_STRIDE))
            if not positions or positions[-1] != n - CANONICAL_CTX:
                positions.append(n - CANONICAL_CTX)
            for start in positions:
                end     = start + CANONICAL_CTX
                e_start = start // TOKENS_PER_EVENT
                e_end   = min(end // TOKENS_PER_EVENT, n_events)
                z_vals  = [per_event_z[abs_idx[(uid, i)]]
                           for i in range(e_start, e_end)
                           if (uid, i) in abs_idx]
                test_scores.append(max(z_vals) if z_vals else 0.0)
                test_labels.append(any(flat_anomaly[start:end]))
                test_user_ids.append(uid_h)

    # ── Quick evaluation ──────────────────────────────────────────────────────
    n_windows  = len(test_scores)
    n_anomalous = sum(test_labels)
    print(f"  Windows: {n_windows}   Anomalous: {n_anomalous}")

    print(f"\n{'─' * 66}")
    print(f"  {'Sigma':>5}  {'Threshold':>9}  {'TP':>4}  {'FP':>5}  "
          f"{'FN':>4}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Flagged':>7}")
    print(f"{'─' * 66}")
    for sigma in [1.0, 1.5, 2.0, 2.5, 3.0]:
        threshold = sigma   # scores are already z-scores; val_mean=0, val_std=1
        m = metrics(test_scores, test_labels, threshold)
        marker = "  ◄" if abs(sigma - args.sigma) < 0.01 else ""
        print(f"  {sigma:>5.1f}  {threshold:>9.2f}  {m['tp']:>4}  {m['fp']:>5}  "
              f"{m['fn']:>4}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  "
              f"{m['f1']:>6.3f}  {sum(s > threshold for s in test_scores):>7}{marker}")
    print(f"{'─' * 66}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "val_mean":      0.0,   # scores are z-scores
        "val_std":       1.0,
        "scales":        scales,
        "test_scores":   test_scores,
        "test_labels":   [int(l) for l in test_labels],
        "test_user_ids": test_user_ids,
    }
    out_path = finetuned_dir / "anomaly_scores_multiscale.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved → {out_path}")
    print(f"  Use with:  python stage2_filter.py "
          f"--scores-file {out_path}")
    print("=" * 66)


if __name__ == "__main__":
    main()
