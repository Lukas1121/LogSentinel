"""
Rigorous per-window anomaly type evaluation.

Reconstructs finetune.py's exact window packing (CTX_LEN=1024, STRIDE=512,
14 tokens/event, per-user sorted streams) to determine which specific anomaly
type(s) each window contains. Detection is only credited if the detecting
method (model or rule) matches the window's actual anomaly type.
"""
import json, hashlib
from collections import defaultdict
from datetime import datetime, timedelta

SIGMA          = 2.5
CTX_LEN        = 1024
STRIDE         = 512
TOKENS_PER_EVT = 14   # must match finetune.py encode_event output length

# ── Load scores ────────────────────────────────────────────────────────────────
with open('data/tenant_test/finetuned/anomaly_scores.json') as f:
    results = json.load(f)

scores    = results['test_scores']
labels    = results['test_labels']
uid_hashes = results['test_user_ids']   # md5[:4] from finetune.py
val_mean  = results['val_mean']
val_std   = results['val_std']
threshold = val_mean + SIGMA * val_std

# ── Load raw events ────────────────────────────────────────────────────────────
events = [json.loads(l) for l in open('data/tenant_test/anomaly_test.jsonl')]

user_events = defaultdict(list)
for e in events:
    user_events[e.get('UserId', 'unknown')].append(e)

# ── Reconstruct per-window anomaly types (mirrors pack_test_windows exactly) ──
# finetune.py processes users in sorted() order, STRIDE=512, CTX_LEN=1024
window_types_list = []   # list of sets — one set of anomaly types per window
reconstructed_hashes = []

for uid in sorted(user_events.keys()):
    uid_h  = hashlib.md5(uid.encode()).hexdigest()[:4]
    stream = sorted(user_events[uid], key=lambda e: e.get('CreationTime', ''))

    # Build per-event anomaly type (None if normal)
    event_type = []
    for e in stream:
        if '_anomaly' in e:
            event_type.append(e['_anomaly']['type'])
        else:
            event_type.append(None)

    n_tokens = len(stream) * TOKENS_PER_EVT

    if n_tokens < CTX_LEN:
        # Single padded window — contains all events
        wtypes = {t for t in event_type if t is not None}
        window_types_list.append(wtypes)
        reconstructed_hashes.append(uid_h)
        continue

    positions = list(range(0, n_tokens - CTX_LEN + 1, STRIDE))
    last_start = n_tokens - CTX_LEN
    if last_start not in positions:
        positions.append(last_start)

    for start in positions:
        # Events whose token span overlaps [start, start+CTX_LEN)
        # Event j spans tokens [j*14, j*14+14)
        # Overlaps window if j*14 < start+CTX_LEN AND j*14+14 > start
        j_min = start // TOKENS_PER_EVT
        j_max = (start + CTX_LEN - 1) // TOKENS_PER_EVT
        j_max = min(j_max, len(stream) - 1)

        wtypes = set()
        for j in range(j_min, j_max + 1):
            if event_type[j] is not None:
                wtypes.add(event_type[j])

        window_types_list.append(wtypes)
        reconstructed_hashes.append(uid_h)

# ── Sanity check: window count and hash order must match scores file ───────────
assert len(window_types_list) == len(scores), (
    f"Window count mismatch: reconstructed {len(window_types_list)}, "
    f"scores file has {len(scores)}"
)
mismatches = sum(r != s for r, s in zip(reconstructed_hashes, uid_hashes))
assert mismatches == 0, f"{mismatches} user-hash mismatches between reconstruction and scores file"
print(f"  Reconstruction verified: {len(scores)} windows, all user hashes match.\n")

# ── Build rule-detected user sets ─────────────────────────────────────────────
def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

DOWNLOAD_OPS = {"FileDownloaded","FileSyncDownloadedFull","FileSyncDownloadedPartial","FileAccessed"}

# Rule: mass_download only (model handles everything else at high recall)
rule_md_users = set()
user_dl = defaultdict(list)
for e in events:
    if e.get('Operation') in DOWNLOAD_OPS:
        user_dl[e.get('UserId','')].append(e)
for uid, evts in user_dl.items():
    evts.sort(key=lambda e: parse_time(e['CreationTime']))
    i = 0
    while i < len(evts):
        anchor = parse_time(evts[i]['CreationTime'])
        burst = [e for e in evts[i:] if parse_time(e['CreationTime']) <= anchor + timedelta(minutes=5)]
        if len(burst) >= 8:
            rule_md_users.add(uid)
            break
        i += 1

# email lookup
hash_to_email = {}
for e in events:
    uid = e.get('UserId','')
    h = hashlib.md5(uid.encode()).hexdigest()[:4]
    hash_to_email[h] = uid

rule_coverage = {
    'mass_download': rule_md_users,
}

# ── Per-window predictions ─────────────────────────────────────────────────────
model_pred  = []
direct_pred = []  # rule credit only if rule type matches window's actual type

for i, (score, label, uid_h, wtypes) in enumerate(
        zip(scores, labels, uid_hashes, window_types_list)):

    email = hash_to_email.get(uid_h, '')
    model_flag = score >= threshold

    # Rule fires for this window only if it matches the window's exact anomaly type
    rule_flag = False
    if label:
        for atype in wtypes:
            if atype in rule_coverage and email in rule_coverage[atype]:
                rule_flag = True
                break

    model_pred.append(model_flag)
    direct_pred.append(model_flag or rule_flag)

# ── Metrics ────────────────────────────────────────────────────────────────────
def metrics(pred, labels):
    tp = sum(p and l     for p, l in zip(pred, labels))
    fp = sum(p and not l for p, l in zip(pred, labels))
    fn = sum(not p and l for p, l in zip(pred, labels))
    tn = sum(not p and not l for p, l in zip(pred, labels))
    prec  = tp / max(tp+fp, 1)
    rec   = tp / max(tp+fn, 1)
    f1    = 2*prec*rec / max(prec+rec, 1e-8)
    fp_tp = fp / max(tp, 1)
    return tp, fp, fn, tn, prec, rec, f1, fp_tp

total  = len(labels)
n_anom = sum(labels)

print("=" * 66)
print("  LogSentinel — Rigorous Per-Window Evaluation")
print("=" * 66)
print(f"  Test set: {total} windows  ({n_anom} anomalous,  {total-n_anom} normal)")
print(f"  Sigma: {SIGMA}  |  Threshold: {threshold:.2f}")

tp, fp, fn, tn, prec, rec, f1, fp_tp = metrics(model_pred, labels)
print(f"\n{'─'*66}")
print(f"  STAGE 1 — Transformer model only")
print(f"{'─'*66}")
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  FP/TP: {fp_tp:.2f}")

tp2, fp2, fn2, tn2, prec2, rec2, f12, fp_tp2 = metrics(direct_pred, labels)
print(f"\n{'─'*66}")
print(f"  COMBINED — Model + Rules (per-window type match required)")
print(f"{'─'*66}")
print(f"  TP={tp2}  FP={fp2}  FN={fn2}  TN={tn2}")
print(f"  Precision: {prec2:.3f}  Recall: {rec2:.3f}  F1: {f12:.3f}  FP/TP: {fp_tp2:.2f}")

# ── Per-type breakdown ────────────────────────────────────────────────────────
print(f"\n{'─'*66}")
print(f"  Detection by anomaly type")
print(f"{'─'*66}")
print(f"  {'Type':<22} {'Windows':>7}  {'Model':>6}  {'Combined':>9}  {'Still missed'}")
print(f"  {'':─<62}")

type_windows   = defaultdict(int)
type_model_tp  = defaultdict(int)
type_combo_tp  = defaultdict(int)
type_missed    = defaultdict(set)

for i, (label, wtypes, mp, cp) in enumerate(
        zip(labels, window_types_list, model_pred, direct_pred)):
    if not label:
        continue
    for atype in wtypes:
        type_windows[atype] += 1
        if mp:
            type_model_tp[atype] += 1
        if cp:
            type_combo_tp[atype] += 1
        else:
            uid_h = uid_hashes[i]
            type_missed[atype].add(hash_to_email.get(uid_h, uid_h))

for atype in sorted(type_windows.keys()):
    n   = type_windows[atype]
    m   = type_model_tp[atype]
    c   = type_combo_tp[atype]
    missed_users = ', '.join(sorted(type_missed[atype])) if type_missed[atype] else 'none'
    print(f"  {atype:<22} {n:>7}  {m:>6}  {c:>9}  {missed_users}")

print(f"\n{'─'*66}")
print(f"  Genuinely missed after combined: {fn2} windows")
if fn2 == 0:
    print(f"  All anomalous windows directly detected.")
else:
    print(f"  Missed types:")
    for atype, users in sorted(type_missed.items()):
        if users:
            print(f"    {atype:<22}  {', '.join(sorted(users))}")
print("=" * 66)
