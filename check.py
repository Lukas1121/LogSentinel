"""
Combined detection analysis: Model alone vs Model + Stage 2 rules.
All metrics computed at the window level for consistency.
"""
import json, hashlib
from collections import defaultdict
from datetime import datetime, timedelta

SIGMA = 2.5

# ── Load model scores ──────────────────────────────────────────────────────────
with open('data/tenant_test/finetuned/anomaly_scores.json') as f:
    results = json.load(f)

scores   = results['test_scores']
labels   = results['test_labels']
user_ids = results['test_user_ids']   # md5[:4] hashes
val_mean = results['val_mean']
val_std  = results['val_std']
threshold = val_mean + SIGMA * val_std

# ── Load raw events ────────────────────────────────────────────────────────────
events = [json.loads(l) for l in open('data/tenant_test/anomaly_test.jsonl')]

hash_to_email = {}
for e in events:
    uid = e.get('UserId', '')
    h = hashlib.md5(uid.encode()).hexdigest()[:4]
    hash_to_email[h] = uid


# ── Stage 2 rules → sets of detected user emails ──────────────────────────────
def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

DOWNLOAD_OPS = {"FileDownloaded", "FileSyncDownloadedFull", "FileSyncDownloadedPartial", "FileAccessed"}

# Rule 1: mass_download
rule_md_users = set()
user_dl = defaultdict(list)
for e in events:
    if e.get('Operation') in DOWNLOAD_OPS:
        user_dl[e.get('UserId', '')].append(e)
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
        ip_fails[e.get('ClientIP', '')].append(e)
for ip, evts in ip_fails.items():
    evts.sort(key=lambda e: parse_time(e['CreationTime']))
    i = 0
    while i < len(evts):
        anchor = parse_time(evts[i]['CreationTime'])
        burst = [e for e in evts[i:] if parse_time(e['CreationTime']) <= anchor + timedelta(seconds=60)]
        if len(burst) >= 10:
            for e in burst:
                rule_bf_users.add(e.get('UserId', ''))
            break
        i += 1

# Rule 3: mfa_disabled
rule_mfa_users = {e.get('UserId', '') for e in events if e.get('Operation') == 'Disable Strong Authentication'}

all_rule_users = rule_md_users | rule_bf_users | rule_mfa_users


# ── Window-level predictions ───────────────────────────────────────────────────
model_pred    = []
combined_pred = []

for score, label, uid_h in zip(scores, labels, user_ids):
    email = hash_to_email.get(uid_h, '')
    m = score >= threshold
    c = m or (email in all_rule_users)
    model_pred.append(m)
    combined_pred.append(c)

def metrics(pred, labels):
    tp = sum(p and l  for p, l in zip(pred, labels))
    fp = sum(p and not l for p, l in zip(pred, labels))
    fn = sum(not p and l  for p, l in zip(pred, labels))
    tn = sum(not p and not l for p, l in zip(pred, labels))
    prec   = tp / max(tp + fp, 1)
    rec    = tp / max(tp + fn, 1)
    f1     = 2 * prec * rec / max(prec + rec, 1e-8)
    fp_tp  = fp / max(tp, 1)
    return tp, fp, fn, tn, prec, rec, f1, fp_tp


# ── What's still missed after combined? ───────────────────────────────────────
missed_hashes = set()
for pred, label, uid_h in zip(combined_pred, labels, user_ids):
    if label and not pred:
        missed_hashes.add(uid_h)

missed_types = defaultdict(set)
for e in events:
    if '_anomaly' in e:
        uid = e.get('UserId', '')
        h = hashlib.md5(uid.encode()).hexdigest()[:4]
        if h in missed_hashes:
            missed_types[e['_anomaly']['type']].add(uid)


# ── Print report ───────────────────────────────────────────────────────────────
total = len(labels)
n_anom = sum(labels)

print("=" * 66)
print("  LogSentinel — Full Architecture Evaluation")
print("=" * 66)
print(f"  Test set: {total} windows  ({n_anom} anomalous,  {total - n_anom} normal)")
print(f"  Sigma:    {SIGMA}  |  Threshold: {threshold:.2f}  (mean={val_mean:.2f}, std={val_std:.2f})")

# Section 1: Model alone
tp, fp, fn, tn, prec, rec, f1, fp_tp = metrics(model_pred, labels)
print(f"\n{'─' * 66}")
print(f"  STAGE 1 — Transformer model only")
print(f"{'─' * 66}")
print(f"  TP={tp}   FP={fp}   FN={fn}   TN={tn}")
print(f"  Precision: {prec:.3f}   Recall: {rec:.3f}   F1: {f1:.3f}")
print(f"  FP/TP ratio: {fp_tp:.2f}  (false alerts per true alert)")

# Section 2: Rules alone
rule_pred = [hash_to_email.get(uid_h, '') in all_rule_users for uid_h in user_ids]
tp_r, fp_r, fn_r, tn_r, prec_r, rec_r, f1_r, fp_tp_r = metrics(rule_pred, labels)
print(f"\n{'─' * 66}")
print(f"  STAGE 2 — Rules only  (mass_download + brute_force + mfa_disabled)")
print(f"{'─' * 66}")
print(f"  TP={tp_r}   FP={fp_r}   FN={fn_r}   TN={tn_r}")
print(f"  Precision: {prec_r:.3f}   Recall: {rec_r:.3f}   F1: {f1_r:.3f}")
print(f"  FP/TP ratio: {fp_tp_r:.2f}")

# Section 3: Combined
tp_c, fp_c, fn_c, tn_c, prec_c, rec_c, f1_c, fp_tp_c = metrics(combined_pred, labels)
print(f"\n{'─' * 66}")
print(f"  COMBINED — Model + Rules")
print(f"{'─' * 66}")
print(f"  TP={tp_c}   FP={fp_c}   FN={fn_c}   TN={tn_c}")
print(f"  Precision: {prec_c:.3f}   Recall: {rec_c:.3f}   F1: {f1_c:.3f}")
print(f"  FP/TP ratio: {fp_tp_c:.2f}  (false alerts per true alert)")

# Section 4: What's still missed
print(f"\n{'─' * 66}")
print(f"  Still missed after combined ({fn_c} windows)")
print(f"{'─' * 66}")
if not missed_types:
    print("  None — all anomalous users detected.")
else:
    for atype, users in sorted(missed_types.items()):
        print(f"  {atype:<25}  {len(users)} user(s): {', '.join(sorted(users))}")

# Section 5: Honest check — missed windows caught only by a DIFFERENT rule
print(f"\n{'─' * 66}")
print(f"  Honest check: missed windows caught only via a DIFFERENT anomaly rule")
print(f"{'─' * 66}")

# For each missed model window, check what anomaly type it contains
# vs what rule(s) caught the user
user_anomaly_types = defaultdict(set)
for e in events:
    if '_anomaly' in e:
        uid = e.get('UserId', '')
        user_anomaly_types[uid].add(e['_anomaly']['type'])

rule_coverage = {
    'mass_download': rule_md_users,
    'brute_force':   rule_bf_users,
    'mfa_disabled':  rule_mfa_users,
}

cross_caught = []  # windows only caught because user was flagged for something else
directly_caught = []

for score, label, uid_h in zip(scores, labels, user_ids):
    if label and score < threshold:  # model missed this window
        email = hash_to_email.get(uid_h, '')
        if email not in all_rule_users:
            continue  # not caught by rules either — already shown in FN
        # Find which anomaly types this user has
        user_types = user_anomaly_types.get(email, set())
        # Find which rules caught them
        catching_rules = {rt for rt, users in rule_coverage.items() if email in users}
        # Are all their anomaly types covered by a rule that fires for that type?
        directly = user_types & catching_rules  # types caught by correct rule
        cross     = user_types - catching_rules  # types only caught via different rule
        if cross:
            cross_caught.append((email, cross, catching_rules))
        else:
            directly_caught.append((email, directly))

if not cross_caught:
    print("  All rule-assisted detections are for the correct anomaly type.")
    print("  No inflation — every detected anomaly is caught by a matching rule.")
else:
    print(f"  {len(cross_caught)} case(s) where a user is caught by rule for type A")
    print(f"  but their missed windows contain type B:")
    for email, cross_types, catching in cross_caught:
        print(f"    {email}")
        print(f"      Caught via:      {', '.join(sorted(catching))}")
        print(f"      Anomaly missed:  {', '.join(sorted(cross_types))}")

print("=" * 66)
