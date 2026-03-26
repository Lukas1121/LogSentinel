"""
Honest combined detection analysis.
A window is only counted as detected if:
  - The model flagged it directly, OR
  - A rule fired for the SAME anomaly type as that window contains.
Cross-catches (user flagged for type A, window contains type B) are NOT counted.
"""
import json, hashlib
from collections import defaultdict
from datetime import datetime, timedelta

SIGMA = 2.5

# ── Load model scores ──────────────────────────────────────────────────────────
with open('data/tenant_test/finetuned/anomaly_scores.json') as f:
    results = json.load(f)

scores    = results['test_scores']
labels    = results['test_labels']
user_ids  = results['test_user_ids']
val_mean  = results['val_mean']
val_std   = results['val_std']
threshold = val_mean + SIGMA * val_std

events = [json.loads(l) for l in open('data/tenant_test/anomaly_test.jsonl')]

hash_to_email = {}
for e in events:
    uid = e.get('UserId', '')
    h = hashlib.md5(uid.encode()).hexdigest()[:4]
    hash_to_email[h] = uid

def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

DOWNLOAD_OPS = {"FileDownloaded","FileSyncDownloadedFull","FileSyncDownloadedPartial","FileAccessed"}

# Rule 1: mass_download
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

# Rule 2: brute_force
rule_bf_users = set()
ip_fails = defaultdict(list)
for e in events:
    if e.get('Operation') == 'UserLoginFailed':
        ip_fails[e.get('ClientIP','')].append(e)
for ip, evts in ip_fails.items():
    evts.sort(key=lambda e: parse_time(e['CreationTime']))
    i = 0
    while i < len(evts):
        anchor = parse_time(evts[i]['CreationTime'])
        burst = [e for e in evts[i:] if parse_time(e['CreationTime']) <= anchor + timedelta(seconds=60)]
        if len(burst) >= 10:
            for e in burst:
                rule_bf_users.add(e.get('UserId',''))
            break
        i += 1

# Rule 3: mfa_disabled
rule_mfa_users = {e.get('UserId','') for e in events if e.get('Operation') == 'Disable Strong Authentication'}

# Rule coverage by anomaly type
rule_coverage = {
    'mass_download': rule_md_users,
    'brute_force':   rule_bf_users,
    'mfa_disabled':  rule_mfa_users,
}

# ── Per-window anomaly types ───────────────────────────────────────────────────
# Build: for each user hash, what anomaly types do they have?
hash_to_types = defaultdict(set)
for e in events:
    if '_anomaly' in e:
        uid = e.get('UserId','')
        h = hashlib.md5(uid.encode()).hexdigest()[:4]
        hash_to_types[h].add(e['_anomaly']['type'])

# ── Compute predictions ────────────────────────────────────────────────────────
model_pred  = []
direct_pred = []  # only counts rule if it matches the window's anomaly type

for score, label, uid_h in zip(scores, labels, user_ids):
    email = hash_to_email.get(uid_h, '')
    model_flag = score >= threshold

    # Direct rule detection: rule fires for same type as this window's anomaly
    rule_direct = False
    if label:
        window_types = hash_to_types.get(uid_h, set())
        for atype in window_types:
            if atype in rule_coverage and email in rule_coverage[atype]:
                rule_direct = True
                break

    model_pred.append(model_flag)
    direct_pred.append(model_flag or rule_direct)

def metrics(pred, labels):
    tp = sum(p and l  for p, l in zip(pred, labels))
    fp = sum(p and not l for p, l in zip(pred, labels))
    fn = sum(not p and l  for p, l in zip(pred, labels))
    tn = sum(not p and not l for p, l in zip(pred, labels))
    prec  = tp / max(tp+fp, 1)
    rec   = tp / max(tp+fn, 1)
    f1    = 2*prec*rec / max(prec+rec, 1e-8)
    fp_tp = fp / max(tp, 1)
    return tp, fp, fn, tn, prec, rec, f1, fp_tp

total = len(labels)
n_anom = sum(labels)

print("=" * 66)
print("  LogSentinel — Honest Combined Evaluation")
print("=" * 66)
print(f"  Test set: {total} windows  ({n_anom} anomalous,  {total-n_anom} normal)")
print(f"  Sigma: {SIGMA}  |  Threshold: {threshold:.2f}  (mean={val_mean:.2f}, std={val_std:.2f})")

tp, fp, fn, tn, prec, rec, f1, fp_tp = metrics(model_pred, labels)
print(f"\n{'─'*66}")
print(f"  STAGE 1 — Model only")
print(f"{'─'*66}")
print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  FP/TP: {fp_tp:.2f}")

tp2, fp2, fn2, tn2, prec2, rec2, f12, fp_tp2 = metrics(direct_pred, labels)
print(f"\n{'─'*66}")
print(f"  COMBINED — Model + Rules (direct matches only)")
print(f"{'─'*66}")
print(f"  TP={tp2}  FP={fp2}  FN={fn2}  TN={tn2}")
print(f"  Precision: {prec2:.3f}  Recall: {rec2:.3f}  F1: {f12:.3f}  FP/TP: {fp_tp2:.2f}")

# ── What's genuinely still missed ─────────────────────────────────────────────
print(f"\n{'─'*66}")
print(f"  Genuinely missed ({fn2} windows) — no direct detection by model or rule")
print(f"{'─'*66}")
missed_types = defaultdict(set)
for pred, label, uid_h in zip(direct_pred, labels, user_ids):
    if label and not pred:
        email = hash_to_email.get(uid_h, '')
        for atype in hash_to_types.get(uid_h, set()):
            if atype not in rule_coverage:  # no rule exists for this type
                missed_types[atype].add(email)
            elif email not in rule_coverage[atype]:  # rule exists but didn't fire
                missed_types[atype].add(email)

if not missed_types:
    print("  None.")
else:
    for atype, users in sorted(missed_types.items()):
        print(f"  {atype:<25}  {len(users)} user(s): {', '.join(sorted(users))}")

print("=" * 66)
