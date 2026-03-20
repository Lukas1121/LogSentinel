#!/bin/bash
# =============================================================================
# finetune_and_exit.sh -- LogSentinel per-tenant fine-tuning runner
#
# Generates a simulated tenant dataset (or uses real logs), fine-tunes
# the base model, evaluates detection metrics, and pushes results.
#
# USAGE:
#   export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
#   export RUNPOD_API_KEY=xxxxxxxxxxxxxxxx
#
#   # Synthetic tenant test (default — seed 99, 30 users)
#   nohup ./finetune_and_exit.sh > finetune.log 2>&1 &
#
#   # Custom synthetic tenant
#   TENANT_SEED=42 TENANT_USERS=50 nohup ./finetune_and_exit.sh > finetune.log 2>&1 &
#
#   # Real tenant logs (place jsonl files in tenant_data/ before running)
#   REAL_TENANT=true nohup ./finetune_and_exit.sh > finetune.log 2>&1 &
#
#   Monitor:  tail -f finetune.log
#
# PIPELINE:
#   Step 1 -- Install dependencies
#   Step 2 -- Generate tenant data  (skipped if REAL_TENANT=true)
#   Step 3 -- Fine-tune base model  (finetune.py)
#   Step 4 -- Push results + terminate pod
#
# CONFIGURATION (set via environment variables):
#   TENANT_SEED    Random seed for synthetic tenant  (default: 99)
#   TENANT_USERS   Number of synthetic users         (default: 30)
#   TENANT_EPOCHS  Fine-tuning epochs                (default: 15)
#   FREEZE_EPOCHS  Frozen phase epochs               (default: 5)
#   REAL_TENANT    Set to "true" to use real logs    (default: false)
#   TENANT_DIR     Directory with real tenant logs   (default: data/tenant_real)
#
# REAL TENANT MODE:
#   Place these files in $TENANT_DIR before running:
#     train.jsonl         -- normal M365 events (70+ days recommended)
#     val.jsonl           -- normal M365 events (14+ days recommended)
#     anomaly_test.jsonl  -- optional, labeled test set
#   Files must follow the M365 Unified Audit Log schema used by LogSentinel.
#
# EXPECTED RUNTIME ON A100:
#   Synthetic (30 users):  ~5 minutes
#   Real tenant (varies):  ~10-20 minutes depending on log volume
# =============================================================================

GITHUB_TOKEN="${GITHUB_TOKEN:?ERROR: export GITHUB_TOKEN=ghp_xxx before running}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=xxx before running}"
GITHUB_USER="${GITHUB_USER:-Lukas1121}"
REPO_NAME="${REPO_NAME:-LogSentinel}"

RUNPOD_POD_ID="${RUNPOD_POD_ID:-$(curl -sf http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")}"

BRANCH="main"
FINETUNE_EXIT=0

# ── Tenant configuration ──────────────────────────────────────────────────────
TENANT_SEED="${TENANT_SEED:-99}"
TENANT_USERS="${TENANT_USERS:-30}"
TENANT_EPOCHS="${TENANT_EPOCHS:-15}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-5}"
REAL_TENANT="${REAL_TENANT:-false}"
TENANT_DIR="${TENANT_DIR:-data/tenant_real}"

# Output directory for fine-tuned model
if [ "$REAL_TENANT" = "true" ]; then
    OUT_DIR="${TENANT_DIR}/finetuned"
    TENANT_LABEL="real"
else
    OUT_DIR="data/tenant_seed${TENANT_SEED}_u${TENANT_USERS}/finetuned"
    TENANT_DATA_DIR="data/tenant_seed${TENANT_SEED}_u${TENANT_USERS}"
    TENANT_LABEL="synthetic_seed${TENANT_SEED}_u${TENANT_USERS}"
fi

export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# =============================================================================
# TRAP
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

    mkdir -p "$OUT_DIR"

    # Fine-tuned model outputs
    git add -f "${OUT_DIR}/model_finetuned.pt"       2>/dev/null || true
    git add -f "${OUT_DIR}/tokeniser_tenant.json"    2>/dev/null || true
    git add -f "${OUT_DIR}/user_thresholds.json"     2>/dev/null || true
    git add -f "${OUT_DIR}/finetune_log.json"        2>/dev/null || true
    git add -f "${OUT_DIR}/anomaly_scores.json"      2>/dev/null || true
    git add -f finetune.log                          2>/dev/null || true

    # Never push raw tenant logs — could contain real user data
    # Only push model outputs and metrics

    if git diff --cached --quiet; then
        log "  Nothing staged -- check finetune.log for errors"
    else
        git commit -m "Finetune: tenant=$TENANT_LABEL pod=$RUNPOD_POD_ID exit=$FINETUNE_EXIT [$(date '+%Y-%m-%d %H:%M')]"
        if git push origin "$BRANCH"; then
            log "  SUCCESS -- results pushed to GitHub on branch $BRANCH"
        else
            log "  ERROR -- git push failed."
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
log "  LogSentinel -- Fine-tuning"
if [ "$REAL_TENANT" = "true" ]; then
log "  Mode: REAL TENANT  (${TENANT_DIR})"
else
log "  Mode: SYNTHETIC  (seed=$TENANT_SEED  users=$TENANT_USERS)"
fi
log "  Epochs: $TENANT_EPOCHS  (frozen=$FREEZE_EPOCHS)"
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

# Verify base model checkpoint exists
if [ ! -f "checkpoints/model_best.pt" ]; then
    log "ERROR: checkpoints/model_best.pt not found."
    log "  Train the base model first with run_and_exit.sh"
    exit 1
fi
log "  Base model found: checkpoints/model_best.pt"

# Verify base vocab exists
if [ ! -f "data/tokeniser.json" ]; then
    log "ERROR: data/tokeniser.json not found."
    log "  Run run_and_exit.sh first to generate base vocab"
    exit 1
fi
log "  Base vocab found: data/tokeniser.json"

# ── Step 1 -- Dependencies ────────────────────────────────────────────────────
log ""
log "STEP 1/3: Installing dependencies..."

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
    log "WARNING: CUDA not available -- fine-tuning will run on CPU (slower)"
fi

log "  Dependencies ready"

# ── Step 2 -- Tenant data ─────────────────────────────────────────────────────
log ""
log "STEP 2/3: Preparing tenant data..."

if [ "$REAL_TENANT" = "true" ]; then
    # Real tenant mode — verify files exist
    log "  Real tenant mode -- checking for logs in ${TENANT_DIR}..."

    if [ ! -f "${TENANT_DIR}/train.jsonl" ]; then
        log "ERROR: ${TENANT_DIR}/train.jsonl not found."
        log "  Export M365 Unified Audit Logs and place them in ${TENANT_DIR}/"
        log "  Required: train.jsonl, val.jsonl"
        log "  Optional: anomaly_test.jsonl"
        exit 1
    fi
    if [ ! -f "${TENANT_DIR}/val.jsonl" ]; then
        log "ERROR: ${TENANT_DIR}/val.jsonl not found."
        exit 1
    fi

    train_lines=$(wc -l < "${TENANT_DIR}/train.jsonl")
    val_lines=$(wc -l < "${TENANT_DIR}/val.jsonl")
    log "  train.jsonl: $train_lines events"
    log "  val.jsonl:   $val_lines events"

    if [ -f "${TENANT_DIR}/anomaly_test.jsonl" ]; then
        test_lines=$(wc -l < "${TENANT_DIR}/anomaly_test.jsonl")
        log "  anomaly_test.jsonl: $test_lines events"
    else
        log "  anomaly_test.jsonl: not found (detection eval will be skipped)"
    fi

    FINETUNE_TENANT_DIR="$TENANT_DIR"

else
    # Synthetic tenant mode — generate fresh data with new seed
    log "  Generating synthetic tenant..."
    log "  Seed=$TENANT_SEED  Users=$TENANT_USERS"
    log "  Output: $TENANT_DATA_DIR"

    python3 generate_logs.py \
        --seed "$TENANT_SEED" \
        --n-users "$TENANT_USERS" \
        --output-dir "$TENANT_DATA_DIR"

    if [ $? -ne 0 ]; then
        log "ERROR: generate_logs.py failed"
        exit 1
    fi

    for f in "${TENANT_DATA_DIR}/train.jsonl" \
              "${TENANT_DATA_DIR}/val.jsonl" \
              "${TENANT_DATA_DIR}/anomaly_test.jsonl"; do
        if [ -f "$f" ]; then
            lines=$(wc -l < "$f")
            log "  $(basename $f): $lines events"
        else
            log "ERROR: $f not generated"
            exit 1
        fi
    done

    FINETUNE_TENANT_DIR="$TENANT_DATA_DIR"
fi

log "  Tenant data ready"

# ── Step 3 -- Fine-tune ───────────────────────────────────────────────────────
log ""
log "STEP 3/3: Fine-tuning base model..."
log "  Tenant dir:    $FINETUNE_TENANT_DIR"
log "  Output dir:    $OUT_DIR"
log "  Total epochs:  $TENANT_EPOCHS  (frozen=$FREEZE_EPOCHS, unfrozen=$((TENANT_EPOCHS - FREEZE_EPOCHS)))"
log ""
log "  Phase 1 (frozen $FREEZE_EPOCHS epochs):"
log "    LR=2e-4  Only new user token rows update"
log "    Base model M365 grammar preserved exactly"
log ""
log "  Phase 2 (unfrozen $((TENANT_EPOCHS - FREEZE_EPOCHS)) epochs):"
log "    LR=3e-5  Full model refines for this tenant"
log "    Early stopping patience=4"
log ""

python3 finetune.py \
    --tenant-dir  "$FINETUNE_TENANT_DIR" \
    --base-ckpt   "checkpoints/model_best.pt" \
    --base-vocab  "data/tokeniser.json" \
    --out-dir     "$OUT_DIR" \
    --epochs      "$TENANT_EPOCHS" \
    --freeze-epochs "$FREEZE_EPOCHS" \
    --n-sigma     2.0 \
    --min-windows 10

FINETUNE_EXIT=$?

if [ $FINETUNE_EXIT -ne 0 ]; then
    log "WARNING: finetune.py exited with code $FINETUNE_EXIT"
else
    log "  Fine-tuning complete"

    # Print summary
    if [ -f "${OUT_DIR}/finetune_log.json" ]; then
        python3 -c "
import json, math
log = json.load(open('${OUT_DIR}/finetune_log.json'))
best = min(log, key=lambda e: e['val_loss'])
last = log[-1]
print()
print('  =========================================')
print(f'  Best epoch:     {best[\"epoch\"]}  ({best[\"phase\"]})')
print(f'  Best val loss:  {best[\"val_loss\"]:.4f}')
print(f'  Best val ppl:   {best[\"val_perplexity\"]:.2f}')
print(f'  Last epoch:     {last[\"epoch\"]}')
print(f'  Phases:')
frozen   = [e for e in log if e[\"phase\"] == \"frozen\"]
unfrozen = [e for e in log if e[\"phase\"] == \"unfrozen\"]
if frozen:
    print(f'    Frozen   ({len(frozen)} epochs): val {frozen[0][\"val_loss\"]:.4f} -> {frozen[-1][\"val_loss\"]:.4f}')
if unfrozen:
    print(f'    Unfrozen ({len(unfrozen)} epochs): val {unfrozen[0][\"val_loss\"]:.4f} -> {unfrozen[-1][\"val_loss\"]:.4f}')
print('  =========================================')
"
    fi

    # Print detection results if available
    if [ -f "${OUT_DIR}/anomaly_scores.json" ]; then
        python3 -c "
import json
r = json.load(open('${OUT_DIR}/anomaly_scores.json'))
pu = r.get('per_user_thresholds', {})
gl = r.get('global_threshold', {})
print()
print('  =========================================')
print('  Detection results')
print()
if pu:
    print(f'  Per-user thresholds:')
    print(f'    TP={pu[\"tp\"]}  FP={pu[\"fp\"]}  FN={pu[\"fn\"]}')
    print(f'    Precision={pu[\"precision\"]:.3f}  Recall={pu[\"recall\"]:.3f}  F1={pu[\"f1\"]:.3f}')
if gl:
    print(f'  Global threshold:')
    print(f'    TP={gl[\"tp\"]}  FP={gl[\"fp\"]}  FN={gl[\"fn\"]}')
    print(f'    Precision={gl[\"precision\"]:.3f}  Recall={gl[\"recall\"]:.3f}  F1={gl[\"f1\"]:.3f}')
print('  =========================================')
"
    fi

    # Print threshold summary
    if [ -f "${OUT_DIR}/user_thresholds.json" ]; then
        python3 -c "
import json
t = json.load(open('${OUT_DIR}/user_thresholds.json'))
s = t['stats']
print()
print(f'  Thresholds calibrated:')
print(f'    Global:          {s[\"global_threshold\"]:.3f}  (mean={s[\"val_mean\"]:.3f}  std={s[\"val_std\"]:.3f})')
print(f'    Per-user:        {s[\"n_users_personal\"]} users with personal threshold')
print(f'    Global fallback: {s[\"n_users_fallback\"]} users (insufficient val data)')
"
    fi
fi

log ""
log "Cleanup trap will push results and terminate pod..."
# trap fires here on EXIT