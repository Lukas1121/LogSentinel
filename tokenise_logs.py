"""
tokenise_logs.py
Converts synthetic M365 audit log JSONL into integer token sequences
for BitNet transformer training.

Design
------
Each JSON event is mapped to ~10 namespaced tokens:

  BOS usr:<hash> op:<operation> wl:<workload> ut:<usertype>
      rs:<resultstatus> ip:<prefix> cc:<country> hr:<hour> dw:<dow>

Namespacing (field:value) prevents token collisions across fields.
Values are discretised to keep vocab small (~300 tokens):
  - IP addresses   -> /16 prefix (first two octets)
  - Timestamps     -> hour-of-day + day-of-week
  - User IDs       -> short hash (consistent within dataset)
  - Missing fields -> dedicated ABSENT token per field

Events are sorted per-user then chronologically before packing into
CTX_LEN=128 windows. This gives the model temporal context within
a single user's session — crucial for learning what is "normal"
for a given user.

Outputs
-------
  data/tokeniser.json      -- vocab: token_string -> int id
  data/train_tokens.pt     -- LongTensor of shape (N_windows, CTX_LEN)
  data/val_tokens.pt
  data/test_tokens.pt      -- includes anomaly labels as separate tensor
  data/test_labels.pt      -- BoolTensor (N_windows,) True = window has anomaly

Flags
-----
  --use-existing-vocab   Load data/tokeniser.json instead of rebuilding the
                         vocab from training data. REQUIRED when resuming
                         training — token IDs must match what the model learned.
                         Building a new vocab remaps IDs and silently corrupts
                         the resumed model.

  --train-only           Tokenise train.jsonl only. Skips val.jsonl and
                         anomaly_test.jsonl, and does not write val_tokens.pt,
                         test_tokens.pt, test_labels.pt, or user ID JSONs.
                         Use when resuming — those tensors are already in the
                         repo and must not be replaced.
"""

import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import torch
from torch.utils.data import Dataset

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_FILE = Path("data/train.jsonl")
VAL_FILE   = Path("data/val.jsonl")
TEST_FILE  = Path("data/anomaly_test.jsonl")
OUT_DIR    = Path("data")

CTX_LEN    = 1024   # tokens per window — fits ~17 events of per-user history
STRIDE     = 512    # 50% overlap — each event appears in 2 windows

# Special token strings
PAD_STR = "<PAD>"
UNK_STR = "<UNK>"
BOS_STR = "<BOS>"
EOS_STR = "<EOS>"
SPECIAL  = [PAD_STR, UNK_STR, BOS_STR, EOS_STR]

def absent(field: str) -> str:
    return f"{field}:--"


# ── Discretisation helpers ────────────────────────────────────────────────────

def ip_prefix(ip: str | None) -> str:
    if not ip:
        return absent("ip")
    parts = ip.split(".")
    if len(parts) >= 2:
        return f"ip:{parts[0]}.{parts[1]}"
    return absent("ip")


def user_hash(uid: str | None) -> str:
    if not uid:
        return absent("usr")
    h = hashlib.md5(uid.encode()).hexdigest()[:4]
    return f"usr:{h}"


def time_tokens(ts: str | None) -> tuple[str, str, str]:
    if not ts:
        return absent("hr"), absent("dw"), absent("tb")
    try:
        dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
        h  = dt.hour
        if h < 6:
            tb = "tb:night"
        elif h < 9:
            tb = "tb:earlyam"
        elif h < 12:
            tb = "tb:morning"
        elif h < 18:
            tb = "tb:work"
        else:
            tb = "tb:evening"
        return f"hr:{h}", f"dw:{dt.weekday()}", tb
    except ValueError:
        return absent("hr"), absent("dw"), absent("tb")


def country_code(event: dict) -> str:
    loc = event.get("Location")
    if isinstance(loc, dict):
        cc = loc.get("CountryCode")
        if cc:
            return f"cc:{cc}"
    return absent("cc")


def result_status(event: dict) -> str:
    rs = event.get("ResultStatus")
    if rs:
        return f"rs:{rs.lower()}"
    return absent("rs")


def device_os(event: dict) -> str:
    props = event.get("DeviceProperties")
    if isinstance(props, list):
        for p in props:
            if isinstance(p, dict) and p.get("Name") == "OS":
                val = p.get("Value", "")
                if "Windows" in val:   return "os:windows"
                if "macOS" in val:     return "os:macos"
                if "iOS" in val:       return "os:ios"
                if "Android" in val:   return "os:android"
                if "Linux" in val:     return "os:linux"
                return "os:other"
    return absent("os")


def device_compliance(event: dict) -> str:
    props = event.get("DeviceProperties")
    if isinstance(props, list):
        for p in props:
            if isinstance(p, dict) and p.get("Name") == "IsCompliant":
                return f"comp:{p.get('Value','?').lower()}"
    return absent("comp")


def file_extension(event: dict) -> str:
    fname = event.get("SourceFileName")
    if not fname:
        return absent("ext")
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    if ext in ("exe", "ps1", "bat", "cmd", "msi", "vbs", "js"):
        return "ext:exec"
    if ext in ("xlsx", "xls", "csv"):
        return "ext:spreadsheet"
    if ext in ("docx", "doc", "pdf", "txt", "md"):
        return "ext:document"
    if ext in ("zip", "tar", "gz", "7z", "rar"):
        return "ext:archive"
    if ext in ("png", "jpg", "jpeg", "gif", "mp4", "mov"):
        return "ext:media"
    if ext:
        return "ext:other"
    return absent("ext")


# ── Event -> token strings ────────────────────────────────────────────────────

def event_to_tokens(event: dict) -> list[str]:
    hr, dw, tb = time_tokens(event.get("CreationTime"))
    workload   = event.get("Workload", "UNKNOWN")

    tokens = [
        BOS_STR,
        # usr:xxxx intentionally omitted — base model learns universal M365
        # grammar without user identity. User tokens are added during
        # per-tenant fine-tuning only, where real user hashes are meaningful.
        f"op:{event.get('Operation', 'UNKNOWN')}",
        f"wl:{workload}",
        f"ut:{event.get('UserType', 0)}",
        result_status(event),
        ip_prefix(event.get("ClientIP")),
        country_code(event),
        hr,
        dw,
        tb,
        device_os(event),
        device_compliance(event),
        file_extension(event) if workload in ("SharePoint", "OneDrive")
                              else absent("ext"),
    ]
    return tokens


# ── Vocabulary ────────────────────────────────────────────────────────────────

class Vocab:
    def __init__(self):
        self.tok2id: dict[str, int] = {}
        self.id2tok: list[str]      = []
        for s in SPECIAL:
            self._add(s)

    def _add(self, token: str) -> int:
        if token not in self.tok2id:
            self.tok2id[token] = len(self.id2tok)
            self.id2tok.append(token)
        return self.tok2id[token]

    def build_from_events(self, events: list[dict]):
        for event in events:
            for tok in event_to_tokens(event):
                self._add(tok)
        for field in ["usr", "op", "wl", "ut", "rs", "ip", "cc", "hr", "dw"]:
            self._add(absent(field))

    def encode(self, token: str) -> int:
        return self.tok2id.get(token, self.tok2id[UNK_STR])

    def encode_event(self, event: dict) -> list[int]:
        return [self.encode(t) for t in event_to_tokens(event)]

    @property
    def size(self) -> int:
        return len(self.id2tok)

    def save(self, path: Path):
        data = {
            "tok2id": self.tok2id,
            "id2tok": self.id2tok,
            "special": {
                "PAD": self.tok2id[PAD_STR],
                "UNK": self.tok2id[UNK_STR],
                "BOS": self.tok2id[BOS_STR],
                "EOS": self.tok2id[EOS_STR],
            }
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  Vocab saved -> {path}  ({self.size} tokens)")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        data     = json.loads(path.read_text())
        v        = cls.__new__(cls)
        v.tok2id = data["tok2id"]
        v.id2tok = data["id2tok"]
        return v


# ── Window packer ─────────────────────────────────────────────────────────────

def _sliding_windows(flat: list[int], ctx_len: int, stride: int,
                      pad_id: int) -> list[list[int]]:
    windows = []
    if len(flat) < ctx_len:
        padded = flat + [pad_id] * (ctx_len - len(flat))
        windows.append(padded)
        return windows

    for start in range(0, len(flat) - ctx_len + 1, stride):
        windows.append(flat[start : start + ctx_len])

    if len(flat) > ctx_len:
        last = flat[-ctx_len:]
        if last != windows[-1]:
            windows.append(last)

    return windows


def pack_windows(
    events: list[dict],
    vocab: Vocab,
    ctx_len: int = CTX_LEN,
    stride: int = STRIDE,
    per_user: bool = True,
    return_user_ids: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[str]]:
    PAD_ID = vocab.tok2id[PAD_STR]

    if per_user:
        user_events: dict[str, list[dict]] = defaultdict(list)
        for e in events:
            uid = e.get("UserId", "unknown")
            user_events[uid].append(e)

        all_windows  = []
        all_user_ids = []
        for uid in sorted(user_events):
            user_stream = sorted(user_events[uid],
                                 key=lambda e: e.get("CreationTime", ""))
            flat = []
            for event in user_stream:
                flat.extend(vocab.encode_event(event))
            user_windows = _sliding_windows(flat, ctx_len, stride, PAD_ID)
            uid_hash = hashlib.md5(uid.encode()).hexdigest()[:4]
            all_windows.extend(user_windows)
            all_user_ids.extend([uid_hash] * len(user_windows))

        tensor = torch.tensor(all_windows, dtype=torch.long)
        if return_user_ids:
            return tensor, all_user_ids
        return tensor
    else:
        ordered = sorted(events, key=lambda e: e.get("CreationTime", ""))
        flat = []
        for event in ordered:
            flat.extend(vocab.encode_event(event))
        windows = _sliding_windows(flat, ctx_len, stride, PAD_ID)
        tensor = torch.tensor(windows, dtype=torch.long)
        if return_user_ids:
            return tensor, ["unknown"] * len(windows)
        return tensor


def pack_test_windows(
    events: list[dict],
    vocab: Vocab,
    ctx_len: int = CTX_LEN,
    stride: int = STRIDE,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    PAD_ID = vocab.tok2id[PAD_STR]

    user_events: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        uid = e.get("UserId", "unknown")
        user_events[uid].append(e)

    all_windows  = []
    all_labels   = []
    all_user_ids = []

    for uid in sorted(user_events):
        uid_hash    = hashlib.md5(uid.encode()).hexdigest()[:4]
        user_stream = sorted(user_events[uid],
                             key=lambda e: e.get("CreationTime", ""))

        flat_tokens:     list[int]  = []
        flat_is_anomaly: list[bool] = []

        for event in user_stream:
            toks    = vocab.encode_event(event)
            is_anom = "_anomaly" in event
            flat_tokens.extend(toks)
            flat_is_anomaly.extend([is_anom] * len(toks))

        if len(flat_tokens) < ctx_len:
            tok_chunk  = flat_tokens   + [PAD_ID] * (ctx_len - len(flat_tokens))
            anom_chunk = flat_is_anomaly + [False] * (ctx_len - len(flat_is_anomaly))
            all_windows.append(tok_chunk)
            all_labels.append(any(anom_chunk))
            all_user_ids.append(uid_hash)
            continue

        positions = list(range(0, len(flat_tokens) - ctx_len + 1, stride))
        if len(flat_tokens) > ctx_len:
            last_start = len(flat_tokens) - ctx_len
            if last_start not in positions:
                positions.append(last_start)

        for start in positions:
            tok_chunk  = flat_tokens[start : start + ctx_len]
            anom_chunk = flat_is_anomaly[start : start + ctx_len]
            all_windows.append(tok_chunk)
            all_labels.append(any(anom_chunk))
            all_user_ids.append(uid_hash)

    return (
        torch.tensor(all_windows, dtype=torch.long),
        torch.tensor(all_labels,  dtype=torch.bool),
        all_user_ids,
    )


# ── JSONL loader ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ── Dataset class ─────────────────────────────────────────────────────────────

class LogTokenDataset(Dataset):
    def __init__(self, windows: torch.Tensor):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.windows[idx]
        return w[:-1], w[1:]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="M365 log tokeniser"
    )
    parser.add_argument(
        "--use-existing-vocab", action="store_true", dest="use_existing_vocab",
        help=(
            "Load data/tokeniser.json instead of rebuilding vocab from training "
            "data. Required when resuming — token IDs must be stable across runs. "
            "New tokens in the regenerated training data map to UNK (id=1)."
        ),
    )
    parser.add_argument(
        "--train-only", action="store_true", dest="train_only",
        help=(
            "Tokenise train.jsonl only. Skips val.jsonl and anomaly_test.jsonl "
            "and does not write val/test tensors or user ID JSONs. "
            "Use when resuming — those files are already in the repo."
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  M365 Log Tokeniser")
    if args.use_existing_vocab:
        print("  Mode: --use-existing-vocab (token IDs frozen)")
    if args.train_only:
        print("  Mode: --train-only (val + test skipped)")
    print("=" * 60)

    OUT_DIR.mkdir(exist_ok=True)

    # ── Load raw events ───────────────────────────────────────────────────────
    print(f"\nLoading events...")
    train_events = load_jsonl(TRAIN_FILE)
    print(f"  Train: {len(train_events):>8,}")

    if not args.train_only:
        val_events  = load_jsonl(VAL_FILE)
        test_events = load_jsonl(TEST_FILE)
        print(f"  Val:   {len(val_events):>8,}")
        print(f"  Test:  {len(test_events):>8,}")

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_path = OUT_DIR / "tokeniser.json"

    if args.use_existing_vocab:
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"{vocab_path} not found. Cannot use --use-existing-vocab "
                f"without a tokeniser.json already in the repo."
            )
        print(f"\nLoading existing vocab from {vocab_path}...")
        vocab = Vocab.load(vocab_path)
        print(f"  Vocab size: {vocab.size} tokens  (frozen — no new tokens added)")
        print(f"  Any tokens not in vocab will map to UNK (id=1)")
    else:
        print("\nBuilding vocabulary from training data...")
        vocab = Vocab()
        vocab.build_from_events(train_events)
        print(f"  Vocabulary size: {vocab.size} tokens")

        field_counts = Counter()
        for tok in vocab.id2tok:
            prefix = tok.split(":")[0] if ":" in tok else "special"
            field_counts[prefix] += 1
        print("  Tokens per field:")
        for field, count in sorted(field_counts.items()):
            print(f"    {field:<10} {count:>4}")

        vocab.save(vocab_path)

    # ── Tokenise and pack ─────────────────────────────────────────────────────
    print(f"\nPacking training windows (ctx_len={CTX_LEN})...")
    train_windows = pack_windows(train_events, vocab, per_user=True)
    print(f"  Windows: {len(train_windows):,}  shape: {tuple(train_windows.shape)}")

    if not args.train_only:
        print("Packing validation windows...")
        val_windows, val_user_ids = pack_windows(
            val_events, vocab, per_user=True, return_user_ids=True
        )
        print(f"  Windows: {len(val_windows):,}  shape: {tuple(val_windows.shape)}")

        print("Packing test windows (with anomaly labels)...")
        test_windows, test_labels, test_user_ids = pack_test_windows(
            test_events, vocab
        )
        n_anom_windows = test_labels.sum().item()
        print(f"  Windows: {len(test_windows):,}  Anomalous: {n_anom_windows}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving tensors...")
    torch.save(train_windows, OUT_DIR / "train_tokens.pt")
    size_mb = train_windows.element_size() * train_windows.nelement() / 1e6
    print(f"  train_tokens.pt  {str(tuple(train_windows.shape)):<25} {size_mb:.1f} MB")

    if not args.train_only:
        torch.save(val_windows,  OUT_DIR / "val_tokens.pt")
        torch.save(test_windows, OUT_DIR / "test_tokens.pt")
        torch.save(test_labels,  OUT_DIR / "test_labels.pt")
        (OUT_DIR / "val_user_ids.json").write_text(json.dumps(val_user_ids))
        (OUT_DIR / "test_user_ids.json").write_text(json.dumps(test_user_ids))

        for name, tensor in [
            ("val_tokens.pt",   val_windows),
            ("test_tokens.pt",  test_windows),
            ("test_labels.pt",  test_labels),
        ]:
            size_mb = tensor.element_size() * tensor.nelement() / 1e6
            print(f"  {name:<20} {str(tuple(tensor.shape)):<25} {size_mb:.1f} MB")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\nSample window decode (first 3 events from training):")
    sample = train_windows[0].tolist()
    toks   = [vocab.id2tok[i] for i in sample if i != vocab.tok2id[PAD_STR]]
    event_strs = []
    current    = []
    for t in toks:
        if t == BOS_STR and current:
            event_strs.append(current)
            current = []
        current.append(t)
    if current:
        event_strs.append(current)
    for i, ev in enumerate(event_strs[:3]):
        print(f"  Event {i+1}: {' '.join(ev)}")

    print("\nTokenisation complete. Files in data/")

    if not args.use_existing_vocab:
        print(f"\nNext step: confirm train_transformer.py uses:")
        print(f"  VOCAB    = {vocab.size}")
        print(f"  CTX_LEN  = {CTX_LEN}")


if __name__ == "__main__":
    main()