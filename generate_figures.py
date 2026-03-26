"""
generate_figures.py
Two publication figures for LogSentinel.

Figure 1 — Score distribution: normal vs anomalous windows with threshold
Figure 2 — Per-anomaly-type recall: model alone vs model + rules (rigorous)

Output: results/figures/score_distribution.png
        results/figures/detection_breakdown.png
"""

import json, hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIGMA          = 2.5
CTX_LEN        = 1024
STRIDE         = 512
TOKENS_PER_EVT = 14

# ── Load scores ────────────────────────────────────────────────────────────────
with open("data/tenant_test/finetuned/anomaly_scores.json") as f:
    results = json.load(f)

scores     = results["test_scores"]
labels     = results["test_labels"]
uid_hashes = results["test_user_ids"]
val_mean   = results["val_mean"]
val_std    = results["val_std"]
threshold  = val_mean + SIGMA * val_std

scores_arr = np.array(scores)
labels_arr = np.array(labels, dtype=bool)

# ── Load raw events ────────────────────────────────────────────────────────────
events = [json.loads(l) for l in open("data/tenant_test/anomaly_test.jsonl")]

user_events = defaultdict(list)
for e in events:
    user_events[e.get("UserId", "unknown")].append(e)

hash_to_email = {}
for e in events:
    uid = e.get("UserId", "")
    h = hashlib.md5(uid.encode()).hexdigest()[:4]
    hash_to_email[h] = uid

def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

# ── Reconstruct per-window anomaly types ──────────────────────────────────────
window_types_list = []
for uid in sorted(user_events.keys()):
    stream = sorted(user_events[uid], key=lambda e: e.get("CreationTime", ""))
    event_type = [e["_anomaly"]["type"] if "_anomaly" in e else None for e in stream]
    n_tokens = len(stream) * TOKENS_PER_EVT

    if n_tokens < CTX_LEN:
        window_types_list.append({t for t in event_type if t is not None})
        continue

    positions = list(range(0, n_tokens - CTX_LEN + 1, STRIDE))
    last_start = n_tokens - CTX_LEN
    if last_start not in positions:
        positions.append(last_start)

    for start in positions:
        j_min = start // TOKENS_PER_EVT
        j_max = min((start + CTX_LEN - 1) // TOKENS_PER_EVT, len(stream) - 1)
        wtypes = {event_type[j] for j in range(j_min, j_max + 1) if event_type[j]}
        window_types_list.append(wtypes)

# ── Rules ──────────────────────────────────────────────────────────────────────
DOWNLOAD_OPS = {"FileDownloaded","FileSyncDownloadedFull","FileSyncDownloadedPartial","FileAccessed"}

rule_md_users = set()
user_dl = defaultdict(list)
for e in events:
    if e.get("Operation") in DOWNLOAD_OPS:
        user_dl[e.get("UserId","")].append(e)
for uid, evts in user_dl.items():
    evts.sort(key=lambda e: parse_time(e["CreationTime"]))
    i = 0
    while i < len(evts):
        burst = [x for x in evts[i:] if parse_time(x["CreationTime"]) <= parse_time(evts[i]["CreationTime"]) + timedelta(minutes=5)]
        if len(burst) >= 8:
            rule_md_users.add(uid); break
        i += 1

rule_bf_users = set()
ip_fails = defaultdict(list)
for e in events:
    if e.get("Operation") == "UserLoginFailed":
        ip_fails[e.get("ClientIP","")].append(e)
for ip, evts in ip_fails.items():
    evts.sort(key=lambda e: parse_time(e["CreationTime"]))
    i = 0
    while i < len(evts):
        burst = [x for x in evts[i:] if parse_time(x["CreationTime"]) <= parse_time(evts[i]["CreationTime"]) + timedelta(seconds=60)]
        if len(burst) >= 10:
            for x in burst: rule_bf_users.add(x.get("UserId",""))
            break
        i += 1

rule_mfa_users = {e.get("UserId","") for e in events if e.get("Operation") == "Disable Strong Authentication"}

rule_coverage = {
    "mass_download": rule_md_users,
    "brute_force":   rule_bf_users,
    "mfa_disabled":  rule_mfa_users,
}

# ── Per-type recall (rigorous) ─────────────────────────────────────────────────
type_total    = defaultdict(int)
type_model_tp = defaultdict(int)
type_rules_tp = defaultdict(int)
type_combo_tp = defaultdict(int)

for score, label, uid_h, wtypes in zip(scores, labels, uid_hashes, window_types_list):
    if not label:
        continue
    email = hash_to_email.get(uid_h, "")
    model_flag = score >= threshold
    rule_flag  = any(atype in rule_coverage and email in rule_coverage[atype] for atype in wtypes)

    for atype in wtypes:
        type_total[atype] += 1
        if model_flag:
            type_model_tp[atype] += 1
        rule_direct = atype in rule_coverage and email in rule_coverage[atype]
        if rule_direct:
            type_rules_tp[atype] += 1
        if model_flag or rule_direct:
            type_combo_tp[atype] += 1

TYPE_DISPLAY = {
    "brute_force":      "Brute\nForce",
    "impossible_travel":"Impossible\nTravel",
    "mass_download":    "Mass\nDownload",
    "mfa_disabled":     "MFA\nDisabled",
    "new_country_login":"New Country\nLogin",
    "off_hours_admin":  "Off-Hours\nAdmin",
}

type_order = ["brute_force","impossible_travel","mass_download",
              "mfa_disabled","new_country_login","off_hours_admin"]

type_order_f  = [t for t in type_order if t in type_total]
xlabels       = [TYPE_DISPLAY[t] for t in type_order_f]
model_recalls = [type_model_tp[t] / type_total[t] for t in type_order_f]
rules_recalls = [type_rules_tp[t] / type_total[t] for t in type_order_f]
combo_recalls = [type_combo_tp[t] / type_total[t] for t in type_order_f]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Score distribution
# ══════════════════════════════════════════════════════════════════════════════
normal_scores  = scores_arr[~labels_arr]
anomaly_scores = scores_arr[labels_arr]

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(scores_arr.min(), min(scores_arr.max(), 30), 60)

ax.hist(normal_scores,  bins=bins, alpha=0.65, color="#4C72B0",
        label=f"Normal windows (n={len(normal_scores)})", zorder=2)
ax.hist(anomaly_scores, bins=bins, alpha=0.75, color="#DD4949",
        label=f"Anomalous windows (n={len(anomaly_scores)})", zorder=3)
ax.axvline(threshold, color="#2ca02c", linewidth=2.0, linestyle="--",
           label=f"Threshold  σ={SIGMA}  ({threshold:.1f})", zorder=4)

tp = int(((scores_arr >= threshold) & labels_arr).sum())
fp = int(((scores_arr >= threshold) & ~labels_arr).sum())
ax.annotate(f"σ=2.5 threshold\nTP={tp}  FP={fp}\nFP/TP = {fp/max(tp,1):.2f}",
            xy=(threshold + 0.25, ax.get_ylim()[1] * 0.70),
            fontsize=9.5, color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.85))

ax.set_xlabel("Anomaly Score  (log-perplexity)", fontsize=12)
ax.set_ylabel("Number of Windows", fontsize=12)
ax.set_title("LogSentinel — Model Score Distribution\nNormal vs Anomalous Windows",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(bins[0], bins[-1])
fig.tight_layout()
out1 = OUT_DIR / "score_distribution.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-type detection breakdown (3 bars)
# ══════════════════════════════════════════════════════════════════════════════
x     = np.arange(len(xlabels))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5.5))
bars1 = ax.bar(x - width, model_recalls, width,
               label="Model alone  (σ=2.5)", color="#4C72B0", alpha=0.85, zorder=2)
bars2 = ax.bar(x,          rules_recalls, width,
               label="Stage 2 rules only",   color="#DD8800", alpha=0.85, zorder=2)
bars3 = ax.bar(x + width,  combo_recalls, width,
               label="Model + Stage 2 rules", color="#2ca02c", alpha=0.85, zorder=2)

ax.set_ylabel("Window-level Recall", fontsize=12)
ax.set_title("LogSentinel — Detection Recall by Anomaly Type\nModel  |  Stage 2 Rules  |  Combined  (rigorous per-window evaluation)",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=10.5)
ax.set_ylim(0, 1.22)
ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":", zorder=1)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bars, color in [(bars1, "#4C72B0"), (bars2, "#DD8800"), (bars3, "#2ca02c")]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.0%}", ha="center", va="bottom",
                    fontsize=8.5, color=color, fontweight="bold")

fig.tight_layout()
out2 = OUT_DIR / "detection_breakdown.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out2}")
print("\n  Both figures saved to results/figures/")
