import json
import hashlib
from collections import defaultdict, Counter

with open('data/tenant_test/finetuned/anomaly_scores.json') as f:
    results = json.load(f)

scores = results['test_scores']
labels = results['test_labels']
user_ids = results['test_user_ids']  # these are md5[:4] hashes

THRESHOLD = 11.18  # sigma=1.0

missed_hashes = set()
for score, label, uid_h in zip(scores, labels, user_ids):
    if label == 1 and score < THRESHOLD:
        missed_hashes.add(uid_h)

# Build hash -> email mapping from the JSONL
events = [json.loads(l) for l in open(r'data\tenant_test\anomaly_test.jsonl')]
hash_to_email = {}
for e in events:
    email = e.get('UserId', '')
    h = hashlib.md5(email.encode()).hexdigest()[:4]
    if h in missed_hashes:
        hash_to_email[h] = email

print(f"Missed user hashes: {missed_hashes}")
print(f"Resolved emails:    {hash_to_email}")
print()

# Anomaly types per missed user
user_anomaly_types = defaultdict(list)
for e in events:
    if '_anomaly' in e:
        user_anomaly_types[e['UserId']].append(e['_anomaly']['type'])

print("Anomaly types per missed user:")
for h, email in sorted(hash_to_email.items()):
    print(f"  {h} ({email}): {Counter(user_anomaly_types[email])}")

print()

# Overall count of missed anomaly types across all missed users
all_missed_types = []
for email in hash_to_email.values():
    all_missed_types.extend(user_anomaly_types[email])
print("Missed anomaly types (total):", Counter(all_missed_types))

# Also show detected types for comparison
detected_hashes = set()
for score, label, uid_h in zip(scores, labels, user_ids):
    if label == 1 and score >= THRESHOLD:
        detected_hashes.add(uid_h)

detected_emails = {hashlib.md5(e.get('UserId','').encode()).hexdigest()[:4]: e.get('UserId','')
                   for e in events
                   if hashlib.md5(e.get('UserId','').encode()).hexdigest()[:4] in detected_hashes}

all_detected_types = []
for email in detected_emails.values():
    all_detected_types.extend(user_anomaly_types[email])
print("Detected anomaly types (total):", Counter(all_detected_types))
