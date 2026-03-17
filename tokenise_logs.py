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

CTX_LEN    = 256   # tokens per window — fits ~17 events of per-user history
STRIDE     = 128   # 50% overlap — each event appears in 2 windows

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
    """185.20.160.145 -> ip:185.20  (/16 prefix)"""
    if not ip:
        return absent("ip")
    parts = ip.split(".")
    if len(parts) >= 2:
        return f"ip:{parts[0]}.{parts[1]}"
    return absent("ip")


def user_hash(uid: str | None) -> str:
    """Hash user ID to 4-char hex. Consistent within dataset."""
    if not uid:
        return absent("usr")
    h = hashlib.md5(uid.encode()).hexdigest()[:4]
    return f"usr:{h}"


def time_tokens(ts: str | None) -> tuple[str, str, str]:
    """
    Parse ISO timestamp -> (hr:<hour>, dw:<weekday>, tb:<time_bucket>)
    hr:  0-23 hour of day
    dw:  0=Mon .. 6=Sun
    tb:  time bucket — night/earlyam/morning/work/evening
         This gives the model coarse time-of-day without overfitting to exact hours.
    """
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
    """Extract OS family from DeviceProperties."""
    props = event.get("DeviceProperties")
    if isinstance(props, list):
        for p in props:
            if isinstance(p, dict) and p.get("Name") == "OS":
                val = p.get("Value", "")
                # Normalise to OS family — reduces vocab size
                if "Windows" in val:   return "os:windows"
                if "macOS" in val:     return "os:macos"
                if "iOS" in val:       return "os:ios"
                if "Android" in val:   return "os:android"
                if "Linux" in val:     return "os:linux"
                return f"os:other"
    return absent("os")


def device_compliance(event: dict) -> str:
    """IsCompliant flag — unmanaged device logins are high signal."""
    props = event.get("DeviceProperties")
    if isinstance(props, list):
        for p in props:
            if isinstance(p, dict) and p.get("Name") == "IsCompliant":
                return f"comp:{p.get('Value','?').lower()}"
    return absent("comp")


def file_extension(event: dict) -> str:
    """
    Extract file extension from SourceFileName.
    Groups by risk category — .exe/.ps1 are very different from .xlsx.
    """
    fname = event.get("SourceFileName")
    if not fname:
        return absent("ext")
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    # Group into categories rather than raw extension — keeps vocab small
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
    """
    Convert one event to namespaced token strings.

    v2 token set (~15 tokens per event vs 10 in v1):
      BOS usr op wl ut rs ip cc hr dw tb os comp ext

    New in v2:
      tb   — time bucket (night/earlyam/morning/work/evening)
      os   — device OS family
      comp — IsCompliant flag
      ext  — file extension category (SharePoint events only)

    Per-user windowing means the model reads these tokens in the context
    of that specific user's recent history — not a mixed population stream.
    """
    hr, dw, tb = time_tokens(event.get("CreationTime"))
    workload   = event.get("Workload", "UNKNOWN")

    tokens = [
        BOS_STR,
        user_hash(event.get("UserId")),
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
        # File extension only meaningful for SharePoint/OneDrive
        file_extension(event) if workload in ("SharePoint", "OneDrive")
                              else absent("ext"),
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

def _sliding_windows(flat: list[int], ctx_len: int, stride: int,
                      pad_id: int) -> list[list[int]]:
    """
    Slice a flat token list into overlapping windows of ctx_len.
    Short streams (fewer tokens than ctx_len) produce one padded window.
    """
    windows = []
    if len(flat) < ctx_len:
        # Pad short user streams to ctx_len
        padded = flat + [pad_id] * (ctx_len - len(flat))
        windows.append(padded)
        return windows

    for start in range(0, len(flat) - ctx_len + 1, stride):
        windows.append(flat[start : start + ctx_len])

    # Capture final events if not already included
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
    """
    Encode events and pack into overlapping sliding windows.

    per_user=True (default): separate stream per user, sorted chronologically.
    return_user_ids=True: also return a parallel list of user ID hashes,
        one per window, used for per-user threshold calibration.

    Returns LongTensor of shape (N_windows, ctx_len).
    If return_user_ids=True, returns (tensor, list[str]).
    """
    PAD_ID = vocab.tok2id[PAD_STR]

    if per_user:
        user_events: dict[str, list[dict]] = defaultdict(list)
        for e in events:
            uid = e.get("UserId", "unknown")
            user_events[uid].append(e)

        all_windows = []
        all_user_ids = []
        for uid in sorted(user_events):
            user_stream = sorted(user_events[uid],
                                 key=lambda e: e.get("CreationTime", ""))
            flat = []
            for event in user_stream:
                flat.extend(vocab.encode_event(event))
            user_windows = _sliding_windows(flat, ctx_len, stride, PAD_ID)
            # Hash the uid to match the usr: token format
            import hashlib
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack test events per-user with anomaly labels.

    Per-user windowing means anomalies are always evaluated in the context
    of that specific user's own history — not mixed with other users.
    A window is labelled anomalous if ANY event inside it is anomalous.

    With realistic data (~10 anomalies in 10k events) the anomaly windows
    will be very sparse — exactly as in production.

    Returns (windows, labels):
      windows: LongTensor (N, ctx_len)
      labels:  BoolTensor (N,)  True = window contains an anomaly
    """
    PAD_ID = vocab.tok2id[PAD_STR]

    # Group by user, sort each user's events chronologically
    user_events: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        uid = e.get("UserId", "unknown")
        user_events[uid].append(e)

    all_windows  = []
    all_labels   = []
    all_user_ids = []

    for uid in sorted(user_events):
        import hashlib
        uid_hash    = hashlib.md5(uid.encode()).hexdigest()[:4]
        user_stream = sorted(user_events[uid],
                             key=lambda e: e.get("CreationTime", ""))

        flat_tokens:    list[int]  = []
        flat_is_anomaly: list[bool] = []

        for event in user_stream:
            toks    = vocab.encode_event(event)
            is_anom = "_anomaly" in event
            flat_tokens.extend(toks)
            flat_is_anomaly.extend([is_anom] * len(toks))

        # Sliding windows over this user's stream
        if len(flat_tokens) < ctx_len:
            tok_chunk  = flat_tokens  + [PAD_ID] * (ctx_len - len(flat_tokens))
            anom_chunk = flat_is_anomaly + [False] * (ctx_len - len(flat_is_anomaly))
            all_windows.append(tok_chunk)
            all_labels.append(any(anom_chunk))
            all_user_ids.append(uid_hash)
            continue

        positions = list(range(0, len(flat_tokens) - ctx_len + 1, stride))
        # Always include the last window
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
    train_windows = pack_windows(train_events, vocab, per_user=True)
    print(f"  Windows: {len(train_windows):,}  shape: {tuple(train_windows.shape)}")

    print("Packing validation windows (saving user IDs for per-user thresholds)...")
    val_windows, val_user_ids = pack_windows(val_events, vocab, per_user=True, return_user_ids=True)
    print(f"  Windows: {len(val_windows):,}  shape: {tuple(val_windows.shape)}")

    print("Packing test windows (with anomaly labels and user IDs)...")
    test_windows, test_labels, test_user_ids = pack_test_windows(test_events, vocab)
    n_anom_windows = test_labels.sum().item()
    print(f"  Windows: {len(test_windows):,}  Anomalous: {n_anom_windows}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving tensors...")
    torch.save(train_windows, OUT_DIR / "train_tokens.pt")
    torch.save(val_windows,   OUT_DIR / "val_tokens.pt")
    torch.save(test_windows,  OUT_DIR / "test_tokens.pt")
    torch.save(test_labels,   OUT_DIR / "test_labels.pt")
    # Save user ID lists for per-user threshold calibration
    (OUT_DIR / "val_user_ids.json").write_text(json.dumps(val_user_ids))
    (OUT_DIR / "test_user_ids.json").write_text(json.dumps(test_user_ids))

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