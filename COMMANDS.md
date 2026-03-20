# LogSentinel — Command Reference

Quick reference for all scripts and commands in the LogSentinel pipeline.

---

## Pipeline Overview

```
generate_logs.py  →  tokenise_logs.py  →  train_transformer.py  →  detect.py
                                                                 →  finetune.py
```

---

## Data Generation — `generate_logs.py`

Generates synthetic M365 audit logs as JSONL files.

```bash
# Standard full run (base model training data)
python generate_logs.py

# Train data only — use when resuming training (preserves val/test tensors)
python generate_logs.py --train-only

# Simulate a new tenant — different seed, small user count, separate output dir
python generate_logs.py --seed 99 --n-users 30 --output-dir data/tenant_test

# Custom configuration
python generate_logs.py --seed 42 --n-users 500 --output-dir data/
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--train-only` | false | Skip val.jsonl and anomaly_test.jsonl |
| `--seed` | 42 | Random seed — change to simulate a new tenant |
| `--n-users` | 500 | Number of synthetic users |
| `--output-dir` | `data/` | Output directory for JSONL files |

---

## Tokenisation — `tokenise_logs.py`

Converts JSONL events into token sequences for training.

```bash
# Standard full tokenisation (fresh run)
python tokenise_logs.py

# Resume run — freeze vocab, rebuild train_tokens.pt only
python tokenise_logs.py --use-existing-vocab --train-only
```

**Flags:**

| Flag | Description |
|---|---|
| `--use-existing-vocab` | Load tokeniser.json instead of rebuilding. REQUIRED when resuming — prevents token ID remapping |
| `--train-only` | Skip val/test tensors (already in repo) |

---

## Training — `train_transformer.py`

Trains the BitNet b1.58 transformer on tokenised windows.

```bash
# Fresh training run
python train_transformer.py

# Resume from checkpoint — continue existing training
python train_transformer.py --resume

# Resume with LR restart — fresh cosine decay from a new LR
python train_transformer.py --resume --restart-lr 4e-5
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--resume` | false | Load model_best.pt and continue training |
| `--restart-lr` | None | New LR for cosine restart on resume. Recommended: 4e-5 |

**Key constants (edit in file):**

| Constant | Value | Description |
|---|---|---|
| `EPOCHS` | 40 | Total training epochs for fresh runs |
| `ADDITIONAL_EPOCHS` | 40 | Extra epochs when resuming |
| `LR` | 1e-4 | Base learning rate |
| `EARLY_STOP_PATIENCE` | 5 | Stop if no improvement for N epochs |
| `WARMUP_STEPS` | 500 | Linear LR warmup steps |
| `BATCH_SIZE` | 32 | Reduce to 16 for local RTX 2060 |

---

## Detection — `detect.py`

Explores detection thresholds on pre-computed scores. No retraining needed.

```bash
# Sigma sweep (default — shows all operating points)
python detect.py

# Load fine-tuned model scores instead of base model
python detect.py --scores-file data/tenant_test/finetuned/anomaly_scores.json

# Recompute scores from checkpoint (after new training)
python detect.py --recompute

# Evaluate a specific sigma value
python detect.py --sigma 3.0

# Sigma + Stage 2 rule filter
python detect.py --sigma 3.0 --stage2

# Full precision/recall curve
python detect.py --pr-curve

# Custom sigma sweep
python detect.py --sweep-sigmas 2.0,2.5,3.0,3.5

# Use a specific checkpoint (with recompute)
python detect.py --recompute --checkpoint data/tenant_test/finetuned/model_finetuned.pt
```

**Flags:**

| Flag | Description |
|---|---|
| `--sigma` | Single threshold report |
| `--stage2` | Apply Stage 2 marginal-score filter |
| `--recompute` | Reload model from checkpoint, recompute all scores |
| `--pr-curve` | Print full precision/recall curve |
| `--sweep-sigmas` | Custom comma-separated sigma values |
| `--checkpoint` | Path to specific model checkpoint |
| `--scores-file` | Path to specific anomaly_scores.json |

---

## Fine-tuning — `finetune.py`

Fine-tunes the base model on a specific tenant's logs.

```bash
# Fine-tune on synthetic test tenant
python finetune.py --tenant-dir data/tenant_test --epochs 15

# Fine-tune on real tenant logs
python finetune.py --tenant-dir data/tenant_real --epochs 15

# Custom freeze/unfreeze split
python finetune.py --tenant-dir data/tenant_test --epochs 20 --freeze-epochs 7

# Higher sigma threshold (fewer alerts, higher precision)
python finetune.py --tenant-dir data/tenant_test --epochs 15 --n-sigma 3.0
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--tenant-dir` | required | Directory with train.jsonl, val.jsonl, anomaly_test.jsonl |
| `--base-ckpt` | `checkpoints/model_best.pt` | Base model checkpoint |
| `--base-vocab` | `data/tokeniser.json` | Base tokeniser vocab |
| `--out-dir` | `<tenant-dir>/finetuned/` | Output directory |
| `--epochs` | 15 | Total fine-tuning epochs |
| `--freeze-epochs` | 5 | Phase 1 epochs (user embeddings only) |
| `--n-sigma` | 2.0 | Threshold sigma for calibration |
| `--min-windows` | 10 | Min val windows for per-user threshold |

**Output files:**

| File | Description |
|---|---|
| `model_finetuned.pt` | Fine-tuned model checkpoint |
| `tokeniser_tenant.json` | Base vocab + tenant user tokens |
| `user_thresholds.json` | Per-user calibrated thresholds |
| `finetune_log.json` | Per-epoch training metrics |
| `anomaly_scores.json` | Detection results on test set |

---

## Analysis — `analyse_logs.py`

Visualises synthetic data distributions to verify data quality.

```bash
# Analyse training data (sample for speed)
python analyse_logs.py --sample 200000

# Analyse a specific file
python analyse_logs.py --file data/val.jsonl --sample 100000
```

Outputs 5 plots to `results/analysis/`:
- `operation_frequency.png`
- `workload_distribution.png`
- `time_heatmap.png`
- `ip_country_dist.png`
- `user_variance.png`

---

## RunPod Scripts

### Fresh base model training
```bash
export GITHUB_TOKEN=ghp_xxx
export RUNPOD_API_KEY=xxx
curl -H "Authorization: token $GITHUB_TOKEN" \
     -o run_and_exit.sh \
     https://raw.githubusercontent.com/Lukas1121/LogSentinel/main/run_and_exit.sh
chmod +x run_and_exit.sh
nohup ./run_and_exit.sh > training.log 2>&1 &
tail -f training.log
```

### Resume training (additional epochs on new data)
```bash
export GITHUB_TOKEN=ghp_xxx
export RUNPOD_API_KEY=xxx
curl -H "Authorization: token $GITHUB_TOKEN" \
     -o resume_and_exit.sh \
     https://raw.githubusercontent.com/Lukas1121/LogSentinel/main/resume_and_exit.sh
chmod +x resume_and_exit.sh
nohup ./resume_and_exit.sh > resume.log 2>&1 &
tail -f resume.log
```

### Fine-tune on RunPod (synthetic or real tenant)
```bash
export GITHUB_TOKEN=ghp_xxx
export RUNPOD_API_KEY=xxx
curl -H "Authorization: token $GITHUB_TOKEN" \
     -o finetune_and_exit.sh \
     https://raw.githubusercontent.com/Lukas1121/LogSentinel/main/finetune_and_exit.sh
chmod +x finetune_and_exit.sh

# Synthetic tenant (default seed=99, 30 users)
nohup ./finetune_and_exit.sh > finetune.log 2>&1 &

# Custom synthetic tenant
TENANT_SEED=42 TENANT_USERS=50 nohup ./finetune_and_exit.sh > finetune.log 2>&1 &

# Real tenant logs (place files in data/tenant_real/ first)
REAL_TENANT=true nohup ./finetune_and_exit.sh > finetune.log 2>&1 &

tail -f finetune.log
```

---

## Git Workflows

```bash
# Force pull — discard local changes and match GitHub exactly
git fetch --all
git reset --hard origin/main

# Check training progress after pod push
git pull
cat checkpoints/training_log.json | python -c "
import json,sys,math
log=json.load(sys.stdin)
best=min(log,key=lambda e:e['val_loss'])
print(f'Best: epoch={best[\"epoch\"]} val_loss={best[\"val_loss\"]:.4f} ppl={best[\"val_perplexity\"]:.2f}')
print(f'Last: epoch={log[-1][\"epoch\"]} ppl={log[-1][\"val_perplexity\"]:.2f}')
"
```

---

## Typical Full Workflows

### Train a new base model from scratch
```bash
# 1. Generate data
python generate_logs.py

# 2. Tokenise
python tokenise_logs.py

# 3. Train locally (overnight) or push to RunPod
python train_transformer.py          # local
# OR run_and_exit.sh on RunPod

# 4. Evaluate
python detect.py --recompute
```

### Fine-tune to a synthetic new tenant (local)
```bash
# 1. Generate tenant data
python generate_logs.py --seed 99 --n-users 30 --output-dir data/tenant_test

# 2. Fine-tune (RTX 2060, ~45 minutes)
python finetune.py --tenant-dir data/tenant_test --epochs 15

# 3. Evaluate
python detect.py --scores-file data/tenant_test/finetuned/anomaly_scores.json
```

### Fine-tune to a real tenant
```bash
# 1. Export M365 Unified Audit Logs from client tenant
#    Split into train.jsonl (70 days) and val.jsonl (20 days)
#    Place in data/tenant_real/

# 2. Fine-tune
python finetune.py --tenant-dir data/tenant_real --epochs 15

# 3. Evaluate
python detect.py --scores-file data/tenant_real/finetuned/anomaly_scores.json
```

### Resume training with more data
```bash
# 1. Update role weights or N_TRAIN in generate_logs.py
# 2. Regenerate training data only
python generate_logs.py --train-only
python tokenise_logs.py --use-existing-vocab --train-only

# 3. Resume
python train_transformer.py --resume --restart-lr 4e-5
```

---

## File Structure

```
LogSentinel/
├── generate_logs.py          # Synthetic M365 log generator
├── tokenise_logs.py          # JSONL → token tensors
├── train_transformer.py      # Base model training
├── detect.py                 # Threshold exploration
├── finetune.py               # Per-tenant fine-tuning
├── stage2_filter.py          # Rule-based FP filter
├── analyse_logs.py           # Data quality visualisation
├── run_and_exit.sh           # RunPod: fresh training
├── resume_and_exit.sh        # RunPod: resume training
├── finetune_and_exit.sh      # RunPod: fine-tuning
├── requirements.txt
├── checkpoints/
│   ├── model_best.pt         # Best base model checkpoint
│   ├── model_final.pt        # Final epoch checkpoint
│   └── training_log.json     # Per-epoch metrics
├── data/
│   ├── tokeniser.json        # Base model vocab
│   ├── train_tokens.pt       # Training windows (too large for GitHub)
│   ├── val_tokens.pt         # Validation windows
│   ├── test_tokens.pt        # Test windows
│   ├── test_labels.pt        # Anomaly labels
│   └── tenant_test/          # Synthetic tenant data
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── anomaly_test.jsonl
│       └── finetuned/
│           ├── model_finetuned.pt
│           ├── tokeniser_tenant.json
│           ├── user_thresholds.json
│           ├── finetune_log.json
│           └── anomaly_scores.json
└── results/
    ├── anomaly_scores.json   # Base model detection results
    ├── stage2_results.json
    └── analysis/             # Data quality plots
```
