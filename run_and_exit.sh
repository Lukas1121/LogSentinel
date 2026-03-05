#!/bin/bash
# =============================================================================
# run_and_exit.sh — Neutron Compression Training Runner
#
# USAGE:
#   export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
#   export RUNPOD_API_KEY=xxxxxxxxxxxxxxxx
#   chmod +x run_and_exit.sh
#   nohup ./run_and_exit.sh > training.log 2>&1 &
#
#   nohup keeps it running if your SSH session disconnects.
#   Monitor progress from anywhere with:  tail -f training.log
# =============================================================================

# No "set -e" — we want the script to ALWAYS reach push+terminate,
# even if training crashes halfway through.

GITHUB_TOKEN="${GITHUB_TOKEN:?ERROR: export GITHUB_TOKEN=ghp_xxx before running}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=xxx before running}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-unknown}"  # RunPod sets this automatically

BRANCH="main"
TRAIN_EXIT=0  # updated after training

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =============================================================================
# TRAP — fires no matter how the script exits:
#   - normal completion
#   - training crash
#   - unhandled error
#   - SSH disconnect
# Guarantees we always push + terminate before dying.
# =============================================================================
cleanup() {
    log ""
    log "========================================="
    log "  Cleanup triggered — pushing + terminating"
    log "========================================="
    push_to_github
    log ""
    log "Terminating pod in 10 seconds..."
    log "(Ctrl+C now if you want to keep the pod alive)"
    sleep 10
    terminate_pod
    log "Done. Billing should stop within 1 minute."
}
trap cleanup EXIT

# =============================================================================
# Push whatever checkpoints exist to GitHub
# =============================================================================
push_to_github() {
    log "--- GitHub push ---"

    git config user.email "runpod@bot.local"
    git config user.name "RunPod Bot"

    # Inject token into remote URL so no password prompt
    REMOTE=$(git remote get-url origin)
    git remote set-url origin "${REMOTE/https:\/\//https:\/\/$GITHUB_TOKEN@}"

    # Ensure directories exist so git add doesn't fail
    mkdir -p checkpoints results

    # || true means missing files won't abort the script
    git add -f checkpoints/model.pt      2>/dev/null || true
    git add -f checkpoints/model_best.pt 2>/dev/null || true
    git add -f results/                  2>/dev/null || true
    git add -f training.log              2>/dev/null || true

    if git diff --cached --quiet; then
        log "  Nothing staged — no checkpoint was saved during training"
        log "  Check training.log for errors"
    else
        git commit -m "Auto: pod=$RUNPOD_POD_ID train_exit=$TRAIN_EXIT"
        if git push origin "$BRANCH"; then
            log "  SUCCESS — model pushed to GitHub on branch $BRANCH"
        else
            log "  ERROR — git push failed. Check GITHUB_TOKEN has repo write access."
        fi
    fi
}

# =============================================================================
# Terminate the RunPod pod via API
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
log "  Neutron Compression — Training Runner"
log "========================================="
log "  Pod ID:  $RUNPOD_POD_ID"
log "  Branch:  $BRANCH"
log "  Time:    $(date)"
log "========================================="

# ── Step 1 — Install dependencies ────────────────────────────────────────────
log ""
log "STEP 1/4: Installing dependencies..."

# arithmeticcoding ships as a local .py file in the repo — not on PyPI
sed -i '/arithmeticcoding/d' requirements.txt

pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    log "ERROR: pip install failed — attempting to continue anyway..."
fi
log "  Dependencies installed"

# ── Step 2 — Generate data ────────────────────────────────────────────────────
log ""
log "STEP 2/4: Generating synthetic event data..."

python simulate_events.py
if [ $? -ne 0 ]; then
    log "ERROR: simulate_events.py failed — cannot train without data"
    exit 1  # trap fires: push (nothing yet) + terminate
fi
log "  Data generation complete"

# ── Step 3 — Train ────────────────────────────────────────────────────────────
log ""
log "STEP 3/4: Training BitNet transformer (this will take several hours)..."
log "  Checkpoints saved every 5 epochs to checkpoints/"

python train_transformer.py
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    log "WARNING: Training exited with code $TRAIN_EXIT"
    log "  Will still push any checkpoints saved before the crash"
else
    log "  Training completed successfully"
fi

# ── Step 4 — trap cleanup EXIT fires automatically here ───────────────────────
log ""
log "STEP 4/4: Handing off to cleanup trap (push + terminate)..."