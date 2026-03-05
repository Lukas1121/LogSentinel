#!/bin/bash
# =============================================================================
# run_and_exit.sh
#
# Runs training, pushes the model checkpoint to GitHub when done,
# then terminates the RunPod pod to stop billing.
#
# SETUP (do this once on the pod before running):
#   1. Set your GitHub token:      export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
#   2. Set your RunPod API key:    export RUNPOD_API_KEY=xxxxxxxxxxxxxxxx
#   3. Make this executable:       chmod +x run_and_exit.sh
#   4. Run it:                     ./run_and_exit.sh
#
# HOW TO GET THESE KEYS:
#   GITHUB_TOKEN:   GitHub → Settings → Developer Settings → Personal Access Tokens
#                   Needs "repo" scope (read + write)
#   RUNPOD_API_KEY: RunPod dashboard → Settings → API Keys
#   RUNPOD_POD_ID:  Shown in your pod URL e.g. runpod.io/console/pods/abc123
#                   Or run: curl -s ifconfig.me  (not the pod ID but helps locate it)
#                   Easiest: copy from the RunPod dashboard pod list
# =============================================================================

set -e  # Exit immediately if any command fails

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN="${GITHUB_TOKEN:?ERROR: Set GITHUB_TOKEN before running}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: Set RUNPOD_API_KEY before running}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:?ERROR: Set RUNPOD_POD_ID before running}"

REPO_DIR="neutron-compression"          # folder cloned from GitHub
CHECKPOINT="checkpoints/model.pt"       # path inside repo dir
RESULTS="results/gzip_baseline.json"    # include results too if present
BRANCH="main"
COMMIT_MSG="Auto: trained model checkpoint from RunPod pod $RUNPOD_POD_ID"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

push_to_github() {
    log "Configuring git..."
    cd "$REPO_DIR"

    git config user.email "runpod-bot@neutron-compression.local"
    git config user.name "RunPod Training Bot"

    # Use token in remote URL so push doesn't require interactive login
    REMOTE=$(git remote get-url origin)
    # Insert token: https://github.com/... → https://TOKEN@github.com/...
    AUTHED_REMOTE="${REMOTE/https:\/\//https:\/\/$GITHUB_TOKEN@}"
    git remote set-url origin "$AUTHED_REMOTE"

    log "Staging checkpoint..."
    git add -f "$CHECKPOINT" 2>/dev/null && log "  Added $CHECKPOINT" || log "  WARNING: $CHECKPOINT not found, skipping"
    git add -f "$RESULTS"    2>/dev/null && log "  Added $RESULTS"    || log "  WARNING: $RESULTS not found, skipping"

    # Only commit if there's something staged
    if git diff --cached --quiet; then
        log "WARNING: Nothing to commit — did training produce a checkpoint?"
    else
        git commit -m "$COMMIT_MSG"
        git push origin "$BRANCH"
        log "SUCCESS: Model pushed to GitHub on branch $BRANCH"
    fi

    cd ..
}

terminate_pod() {
    log "Terminating RunPod pod $RUNPOD_POD_ID to stop billing..."

    RESPONSE=$(curl -s --request POST \
        --url "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        --header "Content-Type: application/json" \
        --data "{\"query\": \"mutation { podTerminate(input: { podId: \\\"$RUNPOD_POD_ID\\\" }) }\"}")

    log "RunPod API response: $RESPONSE"

    if echo "$RESPONSE" | grep -q "error"; then
        log "WARNING: Pod termination may have failed — check RunPod dashboard manually"
        log "Pod ID: $RUNPOD_POD_ID"
    else
        log "Pod termination requested. Billing should stop within 1 minute."
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "========================================="
log "  Neutron Compression — Training Runner"
log "========================================="
log "Pod ID:    $RUNPOD_POD_ID"
log "Repo dir:  $REPO_DIR"
log "Branch:    $BRANCH"
echo ""

# Step 1 — Generate data
log "STEP 1/3: Generating synthetic event data..."
cd "$REPO_DIR"
python simulate_events.py
log "Data generation complete."
cd ..

# Step 2 — Run training
log "STEP 2/3: Starting training (this will take several hours)..."
cd "$REPO_DIR"
python train_transformer.py
TRAIN_EXIT=$?
cd ..

if [ $TRAIN_EXIT -ne 0 ]; then
    log "ERROR: Training exited with code $TRAIN_EXIT"
    log "Attempting to push whatever checkpoints exist before terminating..."
fi

# Step 3 — Push to GitHub regardless of training exit code
log "STEP 3/3: Pushing results to GitHub..."
push_to_github

# Final — Terminate pod
echo ""
log "All done. Requesting pod termination in 10 seconds..."
log "(Press Ctrl+C now if you want to keep the pod running)"
sleep 10
terminate_pod

# This line will likely never execute — pod will be gone
log "Termination signal sent."
