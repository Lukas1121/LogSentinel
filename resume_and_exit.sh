#!/bin/bash
# =============================================================================
# resume_and_exit.sh -- LogSentinel resume training runner
#
# Resumes from an existing checkpoint — skips log generation and tokenisation
# entirely. The tokeniser vocab and val/test tensors are pulled directly from
# the repo, train_tokens.pt is regenerated using the existing vocab.
#
# USAGE:
#   export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
#   export RUNPOD_API_KEY=xxxxxxxxxxxxxxxx
#   chmod +x resume_and_exit.sh
#   nohup ./resume_and_exit.sh > resume.log 2>&1 &
#
#   Monitor:  tail -f resume.log
#
# PIPELINE:
#   Step 1 -- Install dependencies
#   Step 2 -- Regenerate train_tokens.pt using existing tokeniser vocab
#   Step 3 -- Resume training         (train_transformer.py --resume --restart-lr 4e-5)
#   Step 4 -- Evaluate                (detect.py --recompute)
#   Step 5 -- Push results + terminate pod
#
# WHY A SEPARATE SCRIPT:
#   run_and_exit.sh regenerates logs + rebuilds the tokeniser from scratch.
#   On a resume run this is dangerous — if the vocab is rebuilt from new
#   synthetic data the token IDs can shift, silently corrupting the model
#   (token 42 meant FileAccessed; now it means something else entirely).
#   This script uses --use-existing-vocab to freeze the vocab and only
#   rebuilds train_tokens.pt, leaving val/test tensors from the repo intact.
#
# EXPECTED RUNTIME ON A100:  ~8 minutes (40 epochs, no data generation)
# =============================================================================

GITHUB_TOKEN="${GITHUB_TOKEN:?ERROR: export GITHUB_TOKEN=ghp_xxx before running}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=xxx before running}"
GITHUB_USER="${GITHUB_USER:-Lukas1121}"
REPO_NAME="${REPO_NAME:-LogSentinel}"

RUNPOD_POD_ID="${RUNPOD_POD_ID:-$(curl -sf http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")}"

BRANCH="main"
TRAIN_EXIT=0
DETECT_EXIT=0

# LR for the cosine restart — 40% of original 1e-4.
# High enough to learn meaningfully on an underfitting model,
# low enough not to destabilise already-trained weights.
RESTART_LR="${RESTART_LR:-4e-5}"

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
push_to_github() {
    log "--- GitHub push ---"

    git config user.email "runpod@bot.local"
    git config user.name "RunPod Bot"

    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

    mkdir -p checkpoints results data

    git add -f checkpoints/model_best.pt        2>/dev/null || true
    git add -f checkpoints/model_final.pt       2>/dev/null || true
    git add -f checkpoints/training_log.json    2>/dev/null || true
    git add -f results/anomaly_scores.json      2>/dev/null || true
    git add -f results/stage2_results.json      2>/dev/null || true
    git add -f results/stage2_alerts.json       2>/dev/null || true
    git add -f data/tokeniser.json              2>/dev/null || true
    git add -f data/val_user_ids.json           2>/dev/null || true
    git add -f data/test_user_ids.json          2>/dev/null || true
    git add -f resume.log                       2>/dev/null || true

    for pt in data/val_tokens.pt data/test_tokens.pt data/test_labels.pt; do
        if [ -f "$pt" ]; then
            size=$(du -m "$pt" | cut -f1)
            if [ "$size" -lt 90 ]; then
                git add -f "$pt" 2>/dev/null || true
            else
                log "  Skipping $pt (${size}MB -- too large for GitHub)"
            fi
        fi
    done
    # train_tokens.pt is always too large — never push it

    if git diff --cached --quiet; then
        log "  Nothing staged -- check resume.log for errors"
    else
        git commit -m "Resume: pod=$RUNPOD_POD_ID lr=$RESTART_LR train_exit=$TRAIN_EXIT detect_exit=$DETECT_EXIT [$(date '+%Y-%m-%d %H:%M')]"
        if git push origin "$BRANCH"; then
            log "  SUCCESS -- results pushed to GitHub on branch $BRANCH"
        else
            log "  ERROR -- git push failed. Check GITHUB_TOKEN has repo write access."
        fi
    fi
}

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
log "  LogSentinel — Resume Training"
log "  restart-lr=$RESTART_LR  additional-epochs=40"
log "========================================="
log "  Pod ID:  $RUNPOD_POD_ID"
log "  Branch:  $BRANCH"
log "  Time:    $(date)"
log "========================================="

# ── Step 0 -- Clone repo ──────────────────────────────────────────────────────
log ""
log "STEP 0: Cloning repository..."

cd /workspace || cd /root || cd ~

git clone "https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
if [ $? -ne 0 ]; then
    log "ERROR: git clone failed."
    exit 1
fi

cd "$REPO_NAME" || { log "ERROR: could not cd into $REPO_NAME"; exit 1; }
log "  Cloned into $(pwd)"

# Verify checkpoint exists before going any further
if [ ! -f "checkpoints/model_best.pt" ]; then
    log "ERROR: checkpoints/model_best.pt not found in repo."
    log "  Push your checkpoint first, or use run_and_exit.sh for a fresh run."
    exit 1
fi
log "  Checkpoint found: checkpoints/model_best.pt"

# ── Step 1 -- Dependencies ────────────────────────────────────────────────────
log ""
log "STEP 1/4: Installing dependencies..."

if python3 -c "import torch" 2>/dev/null; then
    log "  PyTorch already installed -- skipping"
else
    pip install -r requirements.txt --quiet
fi

python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    log "ERROR: PyTorch not importable."
    exit 1
fi

if python3 -c "import torch; assert torch.cuda.is_available()"; then
    python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
else
    log "WARNING: CUDA not available -- training will run on CPU"
fi

log "  Dependencies ready"

# ── Step 2 -- Rebuild train_tokens.pt only ────────────────────────────────────
log ""
log "STEP 2/4: Regenerating train_tokens.pt using existing vocab..."
log "  (val/test tensors and tokeniser.json already in repo -- not touched)"

# Generate fresh train.jsonl (we need new training text, just mapped to old IDs)
python3 generate_logs.py --train-only
GEN_EXIT=$?

if [ $GEN_EXIT -ne 0 ]; then
    log "WARNING: generate_logs.py --train-only failed (exit $GEN_EXIT)"
    log "  Falling back to full generation..."
    python3 generate_logs.py
    if [ $? -ne 0 ]; then
        log "ERROR: generate_logs.py failed completely"
        exit 1
    fi
fi

# Tokenise train only, locking vocab to existing tokeniser.json
# --use-existing-vocab  freezes token IDs — no remapping
# --train-only          skips val/test tensors (already in repo)
python3 tokenise_logs.py --use-existing-vocab --train-only
if [ $? -ne 0 ]; then
    log "ERROR: tokenise_logs.py failed"
    exit 1
fi

# Verify everything needed is present
for f in data/train_tokens.pt data/val_tokens.pt data/test_tokens.pt data/test_labels.pt data/tokeniser.json; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        log "  $f: $size"
    else
        log "ERROR: $f missing -- cannot resume without it"
        exit 1
    fi
done

log "  Data ready"

# ── Step 3 -- Resume training ─────────────────────────────────────────────────
log ""
log "STEP 3/4: Resuming training..."
log "  --resume --restart-lr $RESTART_LR  (+40 epochs)"

python3 train_transformer.py --resume --restart-lr "$RESTART_LR"
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    log "WARNING: Training exited with code $TRAIN_EXIT"
else
    log "  Training complete"

    if [ -f "checkpoints/training_log.json" ]; then
        python3 -c "
import json, math
log = json.load(open('checkpoints/training_log.json'))
best = min(log, key=lambda e: e['val_loss'])
last = log[-1]
print()
print('  =========================================')
print(f'  Best epoch:     {best[\"epoch\"]}')
print(f'  Best val loss:  {best[\"val_loss\"]:.4f}')
print(f'  Best val ppl:   {best[\"val_perplexity\"]:.2f}')
print(f'  Last epoch:     {last[\"epoch\"]}')
print(f'  Last val ppl:   {last[\"val_perplexity\"]:.2f}')
print('  =========================================')
"
    fi
fi

# ── Step 4 -- Evaluate ────────────────────────────────────────────────────────
log ""
log "STEP 4/4: Running anomaly detection evaluation..."

python3 detect.py --recompute
DETECT_EXIT=$?

if [ $DETECT_EXIT -ne 0 ]; then
    log "WARNING: detect.py exited with code $DETECT_EXIT"
else
    log "  Detection evaluation complete"

    if [ -f "results/anomaly_scores.json" ]; then
        python3 -c "
import json
r = json.load(open('results/anomaly_scores.json'))
g = r['global']
print()
print('  =========================================')
print(f'  Val score mean:   {r[\"val_score_mean\"]:.2f} (std={r[\"val_score_std\"]:.2f})')
print(f'  Global threshold: {r[\"global_threshold\"]:.2f}  (sigma=2.0)')
print()
print(f'  Global threshold results:')
print(f'    TP={g[\"tp\"]}  FP={g[\"fp\"]}  FN={g[\"fn\"]}  TN={g[\"tn\"]}')
print(f'    Precision: {g[\"precision\"]:.3f}')
print(f'    Recall:    {g[\"recall\"]:.3f}')
print(f'    F1:        {g[\"f1\"]:.3f}')
print('  =========================================')
print()
print('  Tune sigma with: python3 detect.py --sigma <value>')
"
    fi
fi

log ""
log "Cleanup trap will push results and terminate pod..."
# trap fires here on EXIT