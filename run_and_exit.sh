#!/bin/bash
# =============================================================================
# run_and_exit.sh -- M365 Log Anomaly Detection Training Runner
#
# USAGE:
#   export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
#   export RUNPOD_API_KEY=xxxxxxxxxxxxxxxx
#   chmod +x run_and_exit.sh
#   nohup ./run_and_exit.sh > training.log 2>&1 &
#
#   Monitor from anywhere:  tail -f training.log
#
# PIPELINE:
#   Step 1 -- Install dependencies
#   Step 2 -- Generate synthetic M365 logs  (generate_logs.py)
#   Step 3 -- Tokenise logs                 (tokenise_logs.py)
#   Step 4 -- Train BitNet transformer      (train_transformer.py)
#   Step 5 -- Push results to GitHub + terminate pod
#
# EXPECTED RUNTIME ON A100:  ~10 minutes total
# =============================================================================

GITHUB_TOKEN="${GITHUB_TOKEN:?ERROR: export GITHUB_TOKEN=ghp_xxx before running}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=xxx before running}"
GITHUB_USER="${GITHUB_USER:-Lukas1121}"
REPO_NAME="${REPO_NAME:-LogSentinel}"

# RunPod sets RUNPOD_POD_ID in most templates, fall back to unknown
RUNPOD_POD_ID="${RUNPOD_POD_ID:-$(curl -sf http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")}"

BRANCH="main"
TRAIN_EXIT=0

# Force Python to flush output immediately — prevents step logs appearing in batches
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =============================================================================
# TRAP -- always push + terminate, even if a step crashes
# =============================================================================
cleanup() {
    log ""
    log "========================================="
    log "  Cleanup triggered -- pushing + terminating"
    log "========================================="
    push_to_github
    log ""
    log "Terminating pod in 10 seconds..."
    log "(Ctrl+C now to keep the pod alive)"
    sleep 10
    terminate_pod
}
trap cleanup EXIT

# =============================================================================
# Push everything to GitHub
# =============================================================================
push_to_github() {
    log "--- GitHub push ---"

    git config user.email "runpod@bot.local"
    git config user.name "RunPod Bot"

    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

    mkdir -p checkpoints results data

    git add -f checkpoints/model_best.pt    2>/dev/null || true
    git add -f checkpoints/model_final.pt   2>/dev/null || true
    git add -f checkpoints/training_log.json 2>/dev/null || true
    git add -f results/anomaly_scores.json  2>/dev/null || true
    git add -f results/stage2_results.json  2>/dev/null || true
    git add -f results/stage2_alerts.json   2>/dev/null || true
    git add -f data/tokeniser.json          2>/dev/null || true
    git add -f data/val_user_ids.json       2>/dev/null || true
    git add -f data/test_user_ids.json      2>/dev/null || true
    git add -f training.log                 2>/dev/null || true

    # data/*.pt files can be large -- only push if under 90MB each
    for pt in data/train_tokens.pt data/val_tokens.pt data/test_tokens.pt data/test_labels.pt; do
        if [ -f "$pt" ]; then
            size=$(du -m "$pt" | cut -f1)
            if [ "$size" -lt 90 ]; then
                git add -f "$pt" 2>/dev/null || true
            else
                log "  Skipping $pt (${size}MB -- too large for GitHub)"
            fi
        fi
    done

    if git diff --cached --quiet; then
        log "  Nothing staged -- check training.log for errors"
    else
        git commit -m "Auto: pod=$RUNPOD_POD_ID train_exit=$TRAIN_EXIT [$(date '+%Y-%m-%d %H:%M')]"
        if git push origin "$BRANCH"; then
            log "  SUCCESS -- results pushed to GitHub on branch $BRANCH"
        else
            log "  ERROR -- git push failed. Check GITHUB_TOKEN has repo write access."
        fi
    fi
}

# =============================================================================
# Terminate the RunPod pod
# =============================================================================
terminate_pod() {
    log "--- Terminating pod $RUNPOD_POD_ID ---"
    RESPONSE=$(curl -s --request POST \
        --url "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        --header "Content-Type: application/json" \
        --data "{\"query\": \"mutation { podTerminate(input: { podId: \\\"$RUNPOD_POD_ID\\\" }) }\"}")
    log "  RunPod response: $RESPONSE"
}

# =============================================================================
# MAIN
# =============================================================================
log "========================================="
log "  M365 Log Anomaly Detection"
log "  BitNet b1.58 Transformer Training"
log "========================================="
log "  Pod ID:  $RUNPOD_POD_ID"
log "  Branch:  $BRANCH"
log "  Time:    $(date)"
log "========================================="

# ── Step 0 -- Clone repo ─────────────────────────────────────────────────────
log ""
log "STEP 0: Cloning repository..."

cd /workspace || cd /root || cd ~

git clone "https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
if [ $? -ne 0 ]; then
    log "ERROR: git clone failed. Check GITHUB_TOKEN and repo name."
    exit 1
fi

cd "$REPO_NAME" || { log "ERROR: could not cd into $REPO_NAME"; exit 1; }
log "  Cloned into $(pwd)"

# ── Step 1 -- Dependencies ────────────────────────────────────────────────────
log ""
log "STEP 1/4: Installing dependencies..."

# RunPod A100 images ship with PyTorch pre-installed.
# Only install if missing to avoid a slow 2GB download.
if python3 -c "import torch" 2>/dev/null; then
    log "  PyTorch already installed -- skipping pip install"
else
    log "  PyTorch not found -- installing from requirements.txt..."
    pip install -r requirements.txt --quiet
    if [ $? -ne 0 ]; then
        log "ERROR: pip install failed -- attempting to continue..."
    fi
fi

# Verify torch is available
python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    log "ERROR: PyTorch not importable after install. Cannot continue."
    exit 1
fi

if python3 -c "import torch; assert torch.cuda.is_available()"; then
    python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
    python3 -c "import torch; print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
else
    log "WARNING: CUDA not available -- training will run on CPU (much slower)"
fi

log "  Dependencies ready"

# ── Step 2 -- Generate synthetic logs ─────────────────────────────────────────
log ""
log "STEP 2/4: Generating synthetic M365 audit logs..."
log "  80k training + 20k val + 2k anomaly test events"

python3 generate_logs.py
if [ $? -ne 0 ]; then
    log "ERROR: generate_logs.py failed -- cannot continue without data"
    exit 1
fi

# Verify output
for f in data/train.jsonl data/val.jsonl data/anomaly_test.jsonl; do
    lines=$(wc -l < "$f")
    log "  $f: $lines events"
done

log "  Log generation complete"

# ── Step 3 -- Tokenise ────────────────────────────────────────────────────────
log ""
log "STEP 3/4: Tokenising logs..."

python3 tokenise_logs.py
if [ $? -ne 0 ]; then
    log "ERROR: tokenise_logs.py failed"
    exit 1
fi

# Print vocab size
python3 -c "
import json
tok = json.load(open('data/tokeniser.json'))
print(f'  Vocab size: {len(tok[\"id2tok\"])} tokens')
"

# Verify tensors exist
for f in data/train_tokens.pt data/val_tokens.pt data/test_tokens.pt data/test_labels.pt; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        log "  $f: $size"
    else
        log "ERROR: $f missing after tokenisation"
        exit 1
    fi
done

log "  Tokenisation complete"

# ── Step 4 -- Train ───────────────────────────────────────────────────────────
log ""
log "STEP 4/4: Training BitNet transformer..."
log "  Architecture: 4 layers, 128 dim, 4 heads, ~1.5M params"
log "  30 epochs on A100 ~= 10 minutes"

python3 train_transformer.py
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    log "WARNING: Training exited with code $TRAIN_EXIT"
    log "  Pushing any checkpoints saved before the crash"
else
    log "  Training complete"

    # Print the headline number
    if [ -f "results/anomaly_scores.json" ]; then
        python3 -c "
import json
r = json.load(open('results/anomaly_scores.json'))
p = r['per_user']
print()
print('  =========================================')
print(f'  Val perplexity:  {r[\"val_score_mean\"]:.2f} (std={r[\"val_score_std\"]:.2f})')
print(f'  Global threshold: {r[\"global_threshold\"]:.2f}')
print(f'  Users calibrated: {r[\"n_users_calibrated\"]}')
print()
print(f'  Per-user threshold results:')
print(f'    Recall:    {p[\"recall\"]:.3f}  (missed: {p[\"fn\"]})')
print(f'    Precision: {p[\"precision\"]:.3f}')
print(f'    F1:        {p[\"f1\"]:.3f}')
print('  =========================================')
print()
print('  This is your LinkedIn headline number.')
"
    fi
fi

log "STEP 5/4: Cleanup trap will push results and terminate pod..."
# trap fires here automatically on EXIT