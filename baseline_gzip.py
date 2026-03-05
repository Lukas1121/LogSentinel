# baseline_gzip.py
# Step 3: Measure gzip compression on test frames as the baseline to beat
# Tests three representations: raw binary, delta-encoded binary, voronoi-tokenised + gzip
# Metric: bits-per-event (lower is better)
# This number is the target the transformer + arithmetic coder must beat by >30%

import numpy as np
import h5py
import gzip
import json
from pathlib import Path
from tokenise import (
    delta_encode,
    encode_frame,
    load_tokeniser,
    make_kdtree,
    SPECIAL,
)

# ── Configuration ─────────────────────────────────────────────────────────────

TEST_FILE     = Path("data/events_test.h5")
RESULTS_DIR   = Path("results")

# ── Load frames ───────────────────────────────────────────────────────────────

def load_frames(path: Path) -> list[np.ndarray]:
    frames = []
    with h5py.File(path, "r") as f:
        n = f.attrs["n_frames"]
        for i in range(n):
            frames.append(f["frames"][str(i)][:])
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{n} frames loaded ({(i+1)/n*100:.0f}%)")
    return frames

# ── Compression helpers ───────────────────────────────────────────────────────

def gzip_compress(data: bytes) -> int:
    """Returns compressed size in bits."""
    return len(gzip.compress(data, compresslevel=9)) * 8


def frame_to_raw_bytes(frame: np.ndarray) -> bytes:
    """Raw binary — just the int32 array as bytes, no preprocessing."""
    return frame.astype(np.int32).tobytes()


def frame_to_delta_bytes(frame: np.ndarray) -> bytes:
    """Delta-encoded binary — differences from previous event."""
    return delta_encode(frame).tobytes()


def frame_to_voronoi_bytes(frame: np.ndarray, kdtree, bin_centres: np.ndarray,
                            tof_scale: float) -> tuple[int, int]:
    """
    Voronoi tokenised representation.
    Returns (token_bits, residual_bits) separately so we can see
    how much each stream contributes to the total.
    Token stream:    bin IDs as int32 — structure the transformer will learn
    Residual stream: exact offsets from bin centre — gzip compressed separately
    """
    tokens, residuals = encode_frame(
        frame, kdtree, bin_centres, tof_scale,
        mode_token=SPECIAL["MODE_PRIOR"]
    )
    token_bytes    = np.array(tokens, dtype=np.int32).tobytes()
    residual_bytes = residuals.astype(np.int16).tobytes()  # int16 sufficient for small residuals

    token_bits    = gzip_compress(token_bytes)
    residual_bits = gzip_compress(residual_bytes)
    return token_bits, residual_bits


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(frames: list[np.ndarray], kdtree, bin_centres: np.ndarray,
              tof_scale: float) -> dict:
    results = {
        "raw":              [],
        "delta":            [],
        "voronoi_tokens":   [],   # token stream only
        "voronoi_residuals":[],   # residual stream only
        "voronoi_total":    [],   # both streams combined — true comparison point
    }

    n = len(frames)
    for i, frame in enumerate(frames):
        n_events = len(frame)

        raw_bits   = gzip_compress(frame_to_raw_bytes(frame))
        delta_bits = gzip_compress(frame_to_delta_bytes(frame))
        token_bits, residual_bits = frame_to_voronoi_bytes(
            frame, kdtree, bin_centres, tof_scale)

        results["raw"].append(raw_bits / n_events)
        results["delta"].append(delta_bits / n_events)
        results["voronoi_tokens"].append(token_bits / n_events)
        results["voronoi_residuals"].append(residual_bits / n_events)
        results["voronoi_total"].append((token_bits + residual_bits) / n_events)

        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n} frames compressed ({(i+1)/n*100:.0f}%)")

    return results


def print_results(results: dict):
    print("\n" + "="*60)
    print("  GZIP BASELINE — bits per event (lower is better)")
    print("="*60)
    print(f"  {'Method':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*60)
    for method, values in results.items():
        arr = np.array(values)
        print(f"  {method:<25} {arr.mean():>8.2f} {arr.std():>8.2f} "
              f"{arr.min():>8.2f} {arr.max():>8.2f}")
    print("="*60)

    # voronoi_total is the true baseline — both streams must be transmitted
    baseline = np.array(results["voronoi_total"]).mean()
    print(f"\n  TRUE BASELINE (voronoi tokens + residuals): {baseline:.2f} bits/event")
    print(f"  30% improvement target: {baseline * 0.7:.2f} bits/event")
    print(f"  50% improvement target: {baseline * 0.5:.2f} bits/event")
    print("="*60)

    # Note on what the transformer replaces
    token_mean = np.array(results["voronoi_tokens"]).mean()
    resid_mean = np.array(results["voronoi_residuals"]).mean()
    print(f"\n  Token stream:    {token_mean:.2f} bits/event  ← transformer replaces gzip here")
    print(f"  Residual stream: {resid_mean:.2f} bits/event  ← gzip keeps this in production")
    print(f"  If transformer beats gzip on tokens by 30%:")
    print(f"    New total: {token_mean * 0.7 + resid_mean:.2f} bits/event")
    print(f"    vs baseline: {baseline:.2f} bits/event")

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for m, v in results.items()}
    summary["targets"] = {
        "baseline":  float(baseline),
        "minus_30pct": float(baseline * 0.7),
        "minus_50pct": float(baseline * 0.5),
    }
    out = RESULTS_DIR / "gzip_baseline.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  Results saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading tokeniser...")
    tok = load_tokeniser()
    bin_centres = tok["bin_centres"]
    tof_scale   = tok["tof_scale"]
    mode        = tok["mode"]
    print(f"  Mode: {mode}  bins: {len(bin_centres)}  tof_scale: {tof_scale:.4f}")

    kdtree = make_kdtree(bin_centres, tof_scale)

    print("Loading test frames...")
    frames = load_frames(TEST_FILE)
    print(f"  Loaded {len(frames)} test frames")

    print("\nRunning gzip benchmark across all test frames...")
    results = benchmark(frames, kdtree, bin_centres, tof_scale)
    print_results(results)


if __name__ == "__main__":
    main()