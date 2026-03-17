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
"""

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

CTX_LEN    = 128   # tokens per training window — matches train_transformer.py

# Special token strings
PAD_STR = "<PAD>"
UNK_STR = "<UNK>"
BOS_STR = "<BOS>"
EOS_STR = "<EOS>"
SPECIAL  = [PAD_STR, UNK_STR, BOS_STR, EOS_STR]

# Field-level ABSENT token — used when a field is missing from the event.
# Distinct per field so the model knows *which* field is absent.
def absent(field: str) -> str:
    return f"{field}:--"


# ── Discretisation helpers ────────────────────────────────────────────────────

def ip_prefix(ip: str | None) -> str:
    """Reduce IP to /16 prefix. 185.20.160.145 -> ip:185.20"""
    if not ip:
        return absent("ip")
    parts = ip.split(".")
    if len(parts) >= 2:
        return f"ip:{parts[0]}.{parts[1]}"
    return absent("ip")


def user_hash(uid: str | None) -> str:
    """Hash user ID to a short 4-char hex token for anonymisation."""
    if not uid:
        return absent("usr")
    h = hashlib.md5(uid.encode()).hexdigest()[:4]
    return f"usr:{h}"


def time_tokens(ts: str | None) -> tuple[str, str]:
    """
    Parse ISO timestamp -> (hr:<hour>, dw:<weekday>)
    Hour: 0-23.  Day of week: 0=Mon, 6=Sun.
    """
    if not ts:
        return absent("hr"), absent("dw")
    try:
        dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
        return f"hr:{dt.hour}", f"dw:{dt.weekday()}"
    except ValueError:
        return absent("hr"), absent("dw")


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


# ── Event -> token strings ────────────────────────────────────────────────────

def event_to_tokens(event: dict) -> list[str]:
    """
    Convert one event dict to a list of namespaced token strings.
    Always starts with BOS. Order is fixed — the model learns position
    of each field implicitly.
    """
    hr, dw = time_tokens(event.get("CreationTime"))

    tokens = [
        BOS_STR,
        user_hash(event.get("UserId")),
        f"op:{event.get('Operation', 'UNKNOWN')}",
        f"wl:{event.get('Workload', 'UNKNOWN')}",
        f"ut:{event.get('UserType', 0)}",
        result_status(event),
        ip_prefix(event.get("ClientIP")),
        country_code(event),
        hr,
        dw,
    ]
    return tokens


# ── Vocabulary builder ────────────────────────────────────────────────────────

class Vocab:
    """
    Builds and stores the token -> int mapping.
    Unknown tokens at inference map to UNK (id=1).
    """
    def __init__(self):
        self.tok2id: dict[str, int] = {}
        self.id2tok: list[str]      = []
        # Add specials first so their IDs are stable
        for s in SPECIAL:
            self._add(s)

    def _add(self, token: str) -> int:
        if token not in self.tok2id:
            self.tok2id[token] = len(self.id2tok)
            self.id2tok.append(token)
        return self.tok2id[token]

    def build_from_events(self, events: list[dict]):
        """Scan all events and register every token string."""
        for event in events:
            for tok in event_to_tokens(event):
                self._add(tok)
        # Always register all absent tokens so they're in vocab
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
        data   = json.loads(path.read_text())
        v      = cls.__new__(cls)
        v.tok2id = data["tok2id"]
        v.id2tok = data["id2tok"]
        return v


# ── Window packer ─────────────────────────────────────────────────────────────

def pack_windows(
    events: list[dict],
    vocab: Vocab,
    ctx_len: int = CTX_LEN,
    sort_by_user: bool = True,
) -> torch.Tensor:
    """
    Encode all events and pack into fixed-length context windows.

    sort_by_user=True  -> group by user then sort by time within each group.
                          Used for train/val: model sees each user's timeline.
    sort_by_user=False -> sort globally by time.
                          Used for test: preserves the real temporal order.

    Returns LongTensor of shape (N_windows, ctx_len).
    PAD token fills any remaining space in the last window.
    """
    PAD_ID = vocab.tok2id[PAD_STR]

    if sort_by_user:
        # Group events by user, sort each group by timestamp
        user_events: dict[str, list[dict]] = defaultdict(list)
        for e in events:
            uid = e.get("UserId", "unknown")
            user_events[uid].append(e)
        ordered = []
        for uid in sorted(user_events):
            user_events[uid].sort(key=lambda e: e.get("CreationTime", ""))
            ordered.extend(user_events[uid])
    else:
        ordered = sorted(events, key=lambda e: e.get("CreationTime", ""))

    # Encode all events into a flat token stream
    flat: list[int] = []
    for event in ordered:
        flat.extend(vocab.encode_event(event))

    # Chunk into ctx_len windows
    windows = []
    for start in range(0, len(flat), ctx_len):
        chunk = flat[start : start + ctx_len]
        if len(chunk) < ctx_len:
            chunk = chunk + [PAD_ID] * (ctx_len - len(chunk))
        windows.append(chunk)

    return torch.tensor(windows, dtype=torch.long)


def pack_test_windows(
    events: list[dict],
    vocab: Vocab,
    ctx_len: int = CTX_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack test events and produce a parallel bool tensor indicating
    which windows contain at least one anomalous event.

    Returns (windows, labels):
      windows: LongTensor (N, ctx_len)
      labels:  BoolTensor (N,)  True = window contains an anomaly
    """
    PAD_ID = vocab.tok2id[PAD_STR]

    # Sort globally by time (preserve interleaved anomalies)
    ordered = sorted(events, key=lambda e: e.get("CreationTime", ""))

    flat_tokens: list[int]  = []
    flat_is_anomaly: list[bool] = []

    for event in ordered:
        toks = vocab.encode_event(event)
        flat_tokens.extend(toks)
        is_anom = "_anomaly" in event
        flat_is_anomaly.extend([is_anom] * len(toks))

    windows = []
    labels  = []

    for start in range(0, len(flat_tokens), ctx_len):
        tok_chunk  = flat_tokens[start : start + ctx_len]
        anom_chunk = flat_is_anomaly[start : start + ctx_len]

        if len(tok_chunk) < ctx_len:
            tok_chunk  = tok_chunk  + [PAD_ID] * (ctx_len - len(tok_chunk))
            anom_chunk = anom_chunk + [False]  * (ctx_len - len(anom_chunk))

        windows.append(tok_chunk)
        labels.append(any(anom_chunk))   # window is anomalous if ANY token was

    return (
        torch.tensor(windows, dtype=torch.long),
        torch.tensor(labels,  dtype=torch.bool),
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


# ── Dataset class (used by train_transformer.py) ──────────────────────────────

class LogTokenDataset(Dataset):
    """
    Wraps a window tensor for use with PyTorch DataLoader.
    Returns (input_ids, target_ids) pairs for next-token prediction:
      input  = window[:-1]   (tokens 0 to T-2)
      target = window[1:]    (tokens 1 to T-1)
    """
    def __init__(self, windows: torch.Tensor):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.windows[idx]
        return w[:-1], w[1:]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  M365 Log Tokeniser")
    print("=" * 60)

    OUT_DIR.mkdir(exist_ok=True)

    # ── Load raw events ───────────────────────────────────────────────────────
    print(f"\nLoading events...")
    train_events = load_jsonl(TRAIN_FILE)
    val_events   = load_jsonl(VAL_FILE)
    test_events  = load_jsonl(TEST_FILE)
    print(f"  Train: {len(train_events):>8,}")
    print(f"  Val:   {len(val_events):>8,}")
    print(f"  Test:  {len(test_events):>8,}")

    # ── Build vocabulary from training data only ──────────────────────────────
    # Never look at val/test when building vocab — simulates real deployment.
    print("\nBuilding vocabulary from training data...")
    vocab = Vocab()
    vocab.build_from_events(train_events)
    print(f"  Vocabulary size: {vocab.size} tokens")

    # Show vocab breakdown
    field_counts = Counter()
    for tok in vocab.id2tok:
        prefix = tok.split(":")[0] if ":" in tok else "special"
        field_counts[prefix] += 1
    print("  Tokens per field:")
    for field, count in sorted(field_counts.items()):
        print(f"    {field:<10} {count:>4}")

    vocab.save(OUT_DIR / "tokeniser.json")

    # ── Tokenise and pack ─────────────────────────────────────────────────────
    print(f"\nPacking training windows (ctx_len={CTX_LEN})...")
    train_windows = pack_windows(train_events, vocab, sort_by_user=True)
    print(f"  Windows: {len(train_windows):,}  shape: {tuple(train_windows.shape)}")

    print("Packing validation windows...")
    val_windows = pack_windows(val_events, vocab, sort_by_user=True)
    print(f"  Windows: {len(val_windows):,}  shape: {tuple(val_windows.shape)}")

    print("Packing test windows (with anomaly labels)...")
    test_windows, test_labels = pack_test_windows(test_events, vocab)
    n_anom_windows = test_labels.sum().item()
    print(f"  Windows: {len(test_windows):,}  Anomalous: {n_anom_windows}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving tensors...")
    torch.save(train_windows, OUT_DIR / "train_tokens.pt")
    torch.save(val_windows,   OUT_DIR / "val_tokens.pt")
    torch.save(test_windows,  OUT_DIR / "test_tokens.pt")
    torch.save(test_labels,   OUT_DIR / "test_labels.pt")

    for name, tensor in [
        ("train_tokens.pt", train_windows),
        ("val_tokens.pt",   val_windows),
        ("test_tokens.pt",  test_windows),
        ("test_labels.pt",  test_labels),
    ]:
        size_mb = tensor.element_size() * tensor.nelement() / 1e6
        print(f"  {name:<20} {str(tuple(tensor.shape)):<25} {size_mb:.1f} MB")

    # ── Sanity check: decode a sample window ─────────────────────────────────
    print("\nSample window decode (first 3 events from training):")
    sample = train_windows[0].tolist()
    toks   = [vocab.id2tok[i] for i in sample if i != vocab.tok2id[PAD_STR]]
    # Split back into events at each BOS
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
    print(f"\nNext step: update train_transformer.py to use:")
    print(f"  VOCAB    = {vocab.size}")
    print(f"  CTX_LEN  = {CTX_LEN}")
    print(f"  Data     = data/train_tokens.pt, data/val_tokens.pt")


if __name__ == "__main__":
    main()
