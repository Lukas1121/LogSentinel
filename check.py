import json, hashlib
from collections import defaultdict, Counter

# ── Load model scores ──────────────────────────────────────────────────────────
with open('data/tenant_test/finetuned/anomaly_scores.json') as f:
    results = json.load(f)

scores   = results['test_scores']
labels   = results['test_labels']
user_ids = results['test_user_ids']
val_mean = results['val_mean']
val_std  = results['val_std']

SIGMA = 2.5
threshold = val_mean + SIGMA * val_std

# ── Load raw events ────────────────────────────────────────────────────────────
events = [json.loads(l) for l in open('data/tenant_test/anomaly_test.jsonl')]

# Map hash -> anomaly types per user
hash_to_email = {}
hash_to_types = defaultdict(set)
for e in events:
    uid = e.get('UserId', '')
    h = hashlib.md5(uid.encode()).hexdigest()[:4]
    hash_to_email[h] = uid
    if '_anomaly' in e:
        hash_to_types[h].add(e['_anomaly']['type'])

# ── Stage 2 rules ──────────────────────────────────────────────────────────────
from datetime import datetime, timedelta

DOWNLOAD_OPS = {"FileDownloaded", "FileSyncDownloadedFull", "FileSyncDownloadedPartial", "FileAccessed"}

def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

# Rule 1: mass_download — flag users
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

# Rule 2: brute_force — flag users targeted
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

# Rule 3: mfa_disabled — flag users
rule_mfa_users = {e.get('UserId', '') for e in events if e.get('Operation') == 'Disable Strong Authentication'}

rule_detected_emails = rule_md_users | rule_bf_users | rule_mfa_users

# ── Model detected users ───────────────────────────────────────────────────────
model_detected_hashes = set()
for score, label, uid_h in zip(scores, labels, user_ids):
    if label and score >= threshold:
        model_detected_hashes.add(uid_h)

model_detected_emails = {hash_to_email[h] for h in model_detected_hashes if h in hash_to_email}

# ── Ground truth ──────────────────────────────────────────────────────────────
all_anomaly_users = defaultdict(set)  # type -> set of emails
for e in events:
    if '_anomaly' in e:
        all_anomaly_users[e['_anomaly']['type']].add(e.get('UserId', ''))

# ── Combined analysis ─────────────────────────────────────────────────────────
combined_detected = model_detected_emails | rule_detected_emails

print("=" * 60)
print("  Combined Detection: Model (sigma=2.5) + Stage 2 Rules")
print("=" * 60)
print(f"\n  {'Anomaly type':<25} {'Total':>5}  {'Model':>6}  {'Rules':>6}  {'Combined':>9}  {'Still missed'}")
print("  " + "-" * 70)

total_users = 0
total_detected = 0

for atype, users in sorted(all_anomaly_users.items()):
    n = len(users)
    m = len(users & model_detected_emails)
    r = len(users & rule_detected_emails)
    c = len(users & combined_detected)
    missed = users - combined_detected
    total_users += n
    total_detected += c
    missed_str = ', '.join(sorted(missed)) if missed else 'none'
    print(f"  {atype:<25} {n:>5}  {m:>6}  {r:>6}  {c:>9}  {missed_str}")

print("  " + "-" * 70)
print(f"  {'TOTAL':<25} {total_users:>5}  {'':>6}  {'':>6}  {total_detected:>9}  {total_users - total_detected} missed")
recall = total_detected / total_users if total_users else 0
print(f"\n  Overall user-level recall: {recall:.3f}  ({total_detected}/{total_users} anomalous users detected)")
print("=" * 60)
