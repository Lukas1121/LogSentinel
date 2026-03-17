# LogSentinel

Unsupervised anomaly detection for Microsoft 365 audit logs using a BitNet b1.58 transformer trained on synthetic tenant data.

## How it works

A decoder-only transformer is trained on **normal** M365 audit log sequences using next-token prediction. After training, the model assigns a perplexity score to each window of log events. Windows that surprise the model — scoring above a calibrated threshold — are flagged as anomalous.

No labels are required during training. The model learns what normal looks like, and anomalies surface as statistical outliers.

```
Train  →  synthetic normal logs  →  model learns baseline behaviour
Deploy →  real tenant logs       →  high perplexity = anomaly flag
```

## Architecture

| Component | Detail |
|---|---|
| Model | BitNet b1.58 decoder-only transformer |
| Weights | Ternary {-1, 0, +1} via absmean quantisation |
| Parameters | ~1.5M |
| Vocab | ~137 tokens (namespaced field:value pairs) |
| Context | 128 tokens (~12 log events per window) |
| Layers | 4 × (CausalSelfAttention + FFN) |
| Embedding dim | 128 |

BitNet's ternary weights mean the trained model is **~375KB** on disk — small enough to run on CPU inside a tenant with no GPU required.

## Anomaly types detected

The model is validated against six injected anomaly types:

| Type | Description |
|---|---|
| `impossible_travel` | Login from DK then foreign country within 10 minutes |
| `off_hours_admin` | Privileged operation performed at 3am |
| `mass_download` | 50+ file downloads within 5 minutes |
| `mfa_disabled` | MFA authentication removed from a user account |
| `new_country_login` | First-ever login from an unseen country |
| `brute_force` | 10+ failed logins from same IP within 60 seconds |

## Why BitNet

Standard transformer weights are float32 (4 bytes/param). BitNet weights are 2-bit ternary (~0.25 bytes/param) — a **16x** size reduction. A model that would require a GPU at float32 precision runs comfortably on CPU at inference time. This means:

- No GPU required for production monitoring
- Log data never leaves the tenant (no external API calls)
- GDPR-friendly by design — runs inside the customer environment
- Cost: essentially zero per log line

## Tenant-specific fine-tuning

The base model is trained on synthetic M365 logs calibrated to real tenant schema and field distributions. For production deployment, the base model can be fine-tuned on a specific tenant's 90-day log history, lowering perplexity on that organisation's normal behaviour and sharpening anomaly sensitivity.

## Pipeline

```
generate_logs.py     →  synthetic M365 JSONL (80k train / 20k val / 2k test)
tokenise_logs.py     →  namespaced token vocab + packed .pt tensors
train_transformer.py →  BitNet training + anomaly evaluation (F1 score)
run_and_exit.sh      →  RunPod orchestration (generate → tokenise → train → push → terminate)
```

## Tokenisation

Each log event is encoded as ~10 namespaced tokens:

```
<BOS>  usr:a3f2  op:FileAccessed  wl:SharePoint  ut:0  rs:--  ip:185.20  cc:DK  hr:8  dw:1
```

Fields are discretised to keep vocab small: IP addresses become `/16` prefixes, timestamps become hour-of-day and day-of-week, user IDs are hashed to 4-char hex. Missing fields use a dedicated `field:--` absent token rather than being dropped.

## Quickstart (RunPod)

```bash
git clone https://github.com/YOUR_USERNAME/LogSentinel
cd LogSentinel
export GITHUB_TOKEN=ghp_xxx
export RUNPOD_API_KEY=xxx
chmod +x run_and_exit.sh
nohup ./run_and_exit.sh > training.log 2>&1 &
tail -f training.log
```

Expected runtime on A100: **~10 minutes**. Results pushed to `results/anomaly_scores.json` automatically on completion.

## Quickstart (local GPU)

```bash
pip install -r requirements.txt
python generate_logs.py
python tokenise_logs.py
python train_transformer.py
```

Works on consumer GPUs (tested on RTX 2060 Super, ~30 minutes). Set `BATCH_SIZE = 16` and `num_workers = 0` for Windows.

## Results

After training, `results/anomaly_scores.json` contains:

```json
{
  "val_perplexity_mean": 4.21,
  "threshold": 9.87,
  "precision": 0.91,
  "recall": 0.88,
  "f1": 0.89
}
```

## Data validation

Synthetic log schema is validated against real M365 Unified Audit Log data using `Compare-Logs.ps1`. Key calibrated metrics:

| Field | Real tenant | Synthetic |
|---|---|---|
| ClientIP presence | 77.5% | 78.1% |
| ResultStatus presence | 22.5% | 22.9% |
| ResultStatus value | "Success" | "Success" |
| Workload (SharePoint) | 64.3% | 63.0% |
| Workload (AAD) | 22.5% | 24.8% |

## License

MIT
