# tokenise.py
# Step 2: Voronoi-based tokeniser with residual coding
#
# Two modes:
#   PriorKnowledgeTokeniser  — bins derived from training data density map
#                              hook provided for real instrument geometry files
#   DataDrivenTokeniser      — bins computed per-frame from local event density
#                              generalises to unseen scattering patterns
#
# Each event encodes to:
#   token_id  — which Voronoi bin (transformer sees this)
#   residual  — exact (delta_pixel, delta_tof) offset from bin centre (stored separately)
#
# Streaming safe: no per-frame normalisation. One fixed tof_scale constant
# computed once at training time and baked into tokeniser.json.
#
# Vocabulary layout (4096 total):
#   0          PAD
#   1          UNK  (event outside all bins — should be rare)
#   2          FRAME_START
#   3          FRAME_END
#   4          MODE_PRIOR
#   5          MODE_DATA_DRIVEN
#   6–4095     Voronoi bin tokens (4090 bins)

import numpy as np
import h5py
import json
from pathlib import Path
from scipy.spatial import cKDTree

# ── Constants ─────────────────────────────────────────────────────────────────

VOCAB_SIZE    = 4096
N_BINS        = 4090       # IDs 6–4095
SPECIAL = {
    "PAD":              0,
    "UNK":              1,
    "FRAME_START":      2,
    "FRAME_END":        3,
    "MODE_PRIOR":       4,
    "MODE_DATA_DRIVEN": 5,
}
BIN_ID_OFFSET = 6          # bin 0 → token ID 6

TRAIN_FILE    = Path("data/events_train.h5")
TOKENISER_OUT = Path("tokeniser.json")

# ── Delta encoding ─────────────────────────────────────────────────────────────

def delta_encode(frame: np.ndarray) -> np.ndarray:
    """
    Convert absolute (pixel_id, tof) pairs to delta values.
    First event stored absolute, remainder are differences from previous.
    Frame must be sorted by pixel then tof.
    """
    deltas = np.empty_like(frame)
    deltas[0] = frame[0]
    deltas[1:] = np.diff(frame, axis=0)
    return deltas.astype(np.int32)


def delta_decode(deltas: np.ndarray) -> np.ndarray:
    """Reverse of delta_encode — cumulative sum recovers absolute values."""
    return np.cumsum(deltas, axis=0).astype(np.int32)


# ── Load frames ───────────────────────────────────────────────────────────────

def load_frames(path: Path) -> list[np.ndarray]:
    frames = []
    with h5py.File(path, "r") as f:
        n = f.attrs["n_frames"]
        for i in range(n):
            frames.append(f["frames"][str(i)][:])
            if (i + 1) % 5000 == 0:
                print(f"    {i+1}/{n} frames loaded ({(i+1)/n*100:.0f}%)")
    return frames


# ── TOF scale factor ──────────────────────────────────────────────────────────
# Computed once from training data. Baked into tokeniser.json.
# Used at stream time to equalise pixel and TOF axes for distance computation
# without any per-frame statistics. Safe for streaming.

def compute_tof_scale(frames: list[np.ndarray], n_sample: int = 500) -> float:
    """
    Compute fixed scale factor to equalise pixel and TOF axes.
    Uses median absolute deviation of delta values across a sample of frames.
    Returns tof_scale = mad_pixel / mad_tof
    """
    sample = frames[:n_sample]
    pixel_deltas, tof_deltas = [], []
    for frame in sample:
        d = delta_encode(frame)
        pixel_deltas.append(np.abs(d[1:, 0]))  # skip first absolute event
        tof_deltas.append(np.abs(d[1:, 1]))

    mad_pixel = np.median(np.concatenate(pixel_deltas))
    mad_tof   = np.median(np.concatenate(tof_deltas))

    if mad_tof == 0:
        return 1.0
    scale = float(mad_pixel / mad_tof)
    print(f"  MAD pixel: {mad_pixel:.2f}  MAD tof: {mad_tof:.2f}  tof_scale: {scale:.4f}")
    return scale


# ── Voronoi assignment ────────────────────────────────────────────────────────

def make_kdtree(bin_centres: np.ndarray, tof_scale: float) -> cKDTree:
    """
    Build KD-tree from bin centres with TOF axis scaled.
    Same scaling applied at query time — streaming safe.
    """
    scaled = bin_centres.copy().astype(float)
    scaled[:, 1] *= tof_scale
    return cKDTree(scaled)


def assign_bins(deltas: np.ndarray, kdtree: cKDTree,
                tof_scale: float) -> np.ndarray:
    """
    Assign each delta-encoded event to its nearest Voronoi bin.
    Returns array of bin indices (0-indexed, before adding BIN_ID_OFFSET).
    """
    scaled = deltas.astype(float).copy()
    scaled[:, 1] *= tof_scale
    _, bin_indices = kdtree.query(scaled)
    return bin_indices


# ══════════════════════════════════════════════════════════════════════════════
# PRIOR KNOWLEDGE TOKENISER
# ══════════════════════════════════════════════════════════════════════════════
# In PoC: derives bin centres from training data event density map.
# In production: call from_instrument_file() to load real geometry.
#
# Bin placement strategy:
#   - Build a 2D density histogram of delta values across all training frames
#   - Place fine bins (many, small) in high-density regions — Bragg peaks
#   - Place coarse bins (few, large) in low-density regions — background/edges
#   - Use k-means to find N_BINS centres weighted by density

def _kmeans_weighted(points: np.ndarray, weights: np.ndarray,
                     k: int, n_iter: int = 50,
                     rng: np.random.Generator = None) -> np.ndarray:
    """
    Weighted k-means to place bin centres.
    High-weight (high-density) regions attract more centres → finer bins.
    Low-weight (background) regions get fewer centres → coarser bins.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Initialise centres by sampling points proportional to weight
    probs = weights / weights.sum()
    idx   = rng.choice(len(points), size=k, replace=False, p=probs)
    centres = points[idx].astype(float)

    for iteration in range(n_iter):
        # Assign each point to nearest centre
        tree = cKDTree(centres)
        _, labels = tree.query(points)

        # Update centres as weighted mean of assigned points
        new_centres = np.zeros_like(centres)
        for j in range(k):
            mask = labels == j
            if mask.sum() == 0:
                # Dead centre — reinitialise to random weighted point
                new_centres[j] = points[rng.choice(len(points), p=probs)]
            else:
                w = weights[mask]
                new_centres[j] = (points[mask] * w[:, None]).sum(0) / w.sum()

        shift = np.linalg.norm(new_centres - centres, axis=1).max()
        centres = new_centres

        if (iteration + 1) % 10 == 0:
            print(f"    k-means iter {iteration+1}/{n_iter}  "
                  f"max centre shift: {shift:.3f}")
        if shift < 0.5:
            print(f"    Converged at iteration {iteration+1}")
            break

    return centres


def build_prior_bins_from_training(frames: list[np.ndarray],
                                   tof_scale: float,
                                   n_bins: int = N_BINS,
                                   hist_bins: int = 512) -> np.ndarray:
    """
    Derive Voronoi bin centres from training data density.

    1. Build 2D histogram of delta values (scaled) across training frames
    2. Run weighted k-means: more centres where density is high (signal regions),
       fewer where density is low (background/detector edges)
    3. Return bin centres in UNSCALED delta space (tof_scale applied at query time)
    """
    print("  Building delta density histogram from training frames...")
    all_deltas = []
    for i, frame in enumerate(frames):
        d = delta_encode(frame)
        all_deltas.append(d[1:])  # skip first absolute event
        if (i + 1) % 10000 == 0:
            print(f"    {i+1}/{len(frames)} frames processed")

    all_deltas = np.concatenate(all_deltas, axis=0)
    print(f"  Total delta events for density map: {len(all_deltas):,}")

    # Scale TOF axis for distance computation
    scaled_deltas = all_deltas.astype(float).copy()
    scaled_deltas[:, 1] *= tof_scale

    # 2D histogram in scaled space
    pixel_vals = scaled_deltas[:, 0]
    tof_vals   = scaled_deltas[:, 1]
    pixel_range = (pixel_vals.min(), pixel_vals.max())
    tof_range   = (tof_vals.min(),   tof_vals.max())

    hist, pixel_edges, tof_edges = np.histogram2d(
        pixel_vals, tof_vals,
        bins=hist_bins,
        range=[pixel_range, tof_range]
    )

    # Convert histogram to point cloud: centre of each non-zero bin + its count
    pixel_centres = (pixel_edges[:-1] + pixel_edges[1:]) / 2
    tof_centres   = (tof_edges[:-1]   + tof_edges[1:])   / 2
    pp, tt = np.meshgrid(pixel_centres, tof_centres, indexing="ij")

    mask    = hist > 0
    points  = np.stack([pp[mask], tt[mask]], axis=1)  # in scaled space
    weights = hist[mask].astype(float)

    print(f"  Non-zero histogram bins: {len(points):,}")
    print(f"  Running weighted k-means (k={n_bins})...")

    centres_scaled = _kmeans_weighted(points, weights, k=n_bins)

    # Convert centres back to unscaled space for storage
    # (tof_scale applied at query time — streaming safe)
    centres_unscaled = centres_scaled.copy()
    centres_unscaled[:, 1] /= tof_scale

    return centres_unscaled.astype(np.float32)


def from_instrument_file(path: str, tof_scale: float) -> np.ndarray:
    """
    Production hook: load Voronoi bin centres from instrument geometry file.
    Expected format: JSON or HDF5 with pixel/tof bin centre arrays.

    Not implemented for PoC — raises NotImplementedError with instructions.
    """
    raise NotImplementedError(
        f"Instrument geometry loading not yet implemented.\n"
        f"To use: provide a JSON file at '{path}' with keys:\n"
        f"  'bin_centres': [[pixel, tof], ...] list of N_BINS centres\n"
        f"  'tof_scale': float — must match the value used at training time\n"
        f"For PoC, use build_prior_bins_from_training() instead."
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA DRIVEN TOKENISER
# ══════════════════════════════════════════════════════════════════════════════
# Computes bin centres from the current frame's local event density.
# More expensive per frame but adapts to unseen scattering patterns.
# Streaming safe: only uses current frame statistics, no global normalisation.

def build_datadriven_bins(frame: np.ndarray,
                          tof_scale: float,
                          n_bins: int = N_BINS,
                          hist_bins: int = 128) -> np.ndarray:
    """
    Derive Voronoi bin centres from a single frame's event density.
    Same weighted k-means approach as prior mode but per-frame.
    """
    deltas = delta_encode(frame)
    d = deltas[1:].astype(float)
    d[:, 1] *= tof_scale

    if len(d) < n_bins:
        # Too few events — fall back to uniform grid
        pixel_lin = np.linspace(d[:, 0].min(), d[:, 0].max(), int(np.sqrt(n_bins)))
        tof_lin   = np.linspace(d[:, 1].min(), d[:, 1].max(), int(np.sqrt(n_bins)))
        pp, tt    = np.meshgrid(pixel_lin, tof_lin)
        centres   = np.stack([pp.ravel(), tt.ravel()], axis=1)[:n_bins]
    else:
        hist, pe, te = np.histogram2d(d[:, 0], d[:, 1], bins=hist_bins)
        pc = (pe[:-1] + pe[1:]) / 2
        tc = (te[:-1] + te[1:]) / 2
        pp, tt = np.meshgrid(pc, tc, indexing="ij")
        mask   = hist > 0
        points  = np.stack([pp[mask], tt[mask]], axis=1)
        weights = hist[mask].astype(float)
        centres = _kmeans_weighted(points, weights, k=min(n_bins, len(points)))

    # Return in unscaled space
    centres_out = centres.copy()
    centres_out[:, 1] /= tof_scale
    return centres_out.astype(np.float32)


# ── Encode / decode ───────────────────────────────────────────────────────────

def encode_frame(frame: np.ndarray, kdtree: cKDTree,
                 bin_centres: np.ndarray, tof_scale: float,
                 mode_token: int) -> tuple[list[int], np.ndarray]:
    """
    Encode one frame into (token_ids, residuals).

    token_ids: list of int — FRAME_START, MODE, bin tokens, FRAME_END
    residuals: np.ndarray shape (N_events, 2) — exact delta offsets from bin centre
               First row is absolute (pixel, tof) of first event, not a residual.
    """
    deltas    = delta_encode(frame)
    bin_idx   = assign_bins(deltas, kdtree, tof_scale)

    # Token stream: wrap in frame markers + mode token
    tokens = (
        [SPECIAL["FRAME_START"], mode_token] +
        [int(b) + BIN_ID_OFFSET for b in bin_idx] +
        [SPECIAL["FRAME_END"]]
    )

    # Residuals: difference between actual delta and bin centre
    # Bin centres are in unscaled original delta space
    bin_centre_for_each_event = np.round(bin_centres[bin_idx]).astype(np.int32)
    residuals = (deltas - bin_centre_for_each_event).astype(np.int32)

    return tokens, residuals


def decode_frame(tokens: list[int], residuals: np.ndarray,
                 bin_centres: np.ndarray) -> np.ndarray:
    """
    Decode token_ids + residuals back to absolute (pixel_id, tof) pairs.
    Filters out special tokens, recovers bin centres, adds residuals,
    then cumsum to recover absolute values.
    """
    # Extract bin token IDs (strip special tokens)
    bin_tokens = [
        t for t in tokens
        if t >= BIN_ID_OFFSET and t < BIN_ID_OFFSET + len(bin_centres)
    ]

    bin_idx    = np.array(bin_tokens) - BIN_ID_OFFSET
    centres    = np.round(bin_centres[bin_idx]).astype(np.int32)
    deltas     = (centres + residuals).astype(np.int32)
    frame      = delta_decode(deltas)
    return frame


# ── Save / load ───────────────────────────────────────────────────────────────

def save_tokeniser(mode: str, bin_centres: np.ndarray, tof_scale: float):
    data = {
        "mode":       mode,
        "vocab_size": VOCAB_SIZE,
        "n_bins":     N_BINS,
        "tof_scale":  tof_scale,
        "special_tokens": SPECIAL,
        "bin_id_offset":  BIN_ID_OFFSET,
        # Store centres as list of [pixel, tof] pairs
        "bin_centres": bin_centres.tolist(),
    }
    TOKENISER_OUT.write_text(json.dumps(data, indent=2))
    print(f"Tokeniser saved → {TOKENISER_OUT} "
          f"({TOKENISER_OUT.stat().st_size / 1e3:.1f} KB)")


def load_tokeniser() -> dict:
    data = json.loads(TOKENISER_OUT.read_text())
    data["bin_centres"] = np.array(data["bin_centres"], dtype=np.float32)
    return data


# ── Round-trip verification ───────────────────────────────────────────────────

def verify_roundtrip(frames: list[np.ndarray], kdtree: cKDTree,
                     bin_centres: np.ndarray, tof_scale: float,
                     mode_token: int, n_check: int = 20):
    print(f"\nVerifying round-trip on {n_check} frames...")
    for i in range(n_check):
        frame = frames[i]
        tokens, residuals = encode_frame(
            frame, kdtree, bin_centres, tof_scale, mode_token)
        recovered = decode_frame(tokens, residuals, bin_centres)

        if not np.array_equal(frame, recovered):
            print(f"  FAIL frame {i}")
            print(f"    Original:  {frame[:3]}")
            print(f"    Recovered: {recovered[:3]}")
            diff = frame - recovered
            print(f"    Diff:      {diff[:3]}")
            return False

        if i < 3:
            n_bin_tokens = len([t for t in tokens if t >= BIN_ID_OFFSET])
            print(f"  Frame {i}: {len(frame)} events → "
                  f"{n_bin_tokens} bin tokens  "
                  f"+ {residuals.nbytes} residual bytes ✓")

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n_check} frames verified...")

    print(f"  All {n_check} frames round-tripped identically ✓")
    return True


# ── Residual stats ────────────────────────────────────────────────────────────

def report_residual_stats(frames: list[np.ndarray], kdtree: cKDTree,
                          bin_centres: np.ndarray, tof_scale: float,
                          mode_token: int, n_sample: int = 100):
    """
    Report how large residuals are — smaller = better bin placement.
    Tiny residuals compress trivially. Large residuals mean bins are too coarse.
    """
    all_residuals = []
    for frame in frames[:n_sample]:
        _, residuals = encode_frame(
            frame, kdtree, bin_centres, tof_scale, mode_token)
        all_residuals.append(residuals)

    r = np.concatenate(all_residuals, axis=0)
    print(f"\nResidual stats over {n_sample} frames:")
    print(f"  Pixel residuals — mean: {np.abs(r[:,0]).mean():.2f}  "
          f"std: {r[:,0].std():.2f}  "
          f"max: {np.abs(r[:,0]).max()}")
    print(f"  TOF residuals   — mean: {np.abs(r[:,1]).mean():.2f}  "
          f"std: {r[:,1].std():.2f}  "
          f"max: {np.abs(r[:,1]).max()}")
    print(f"  Residuals within ±10 pixel: "
          f"{(np.abs(r[:,0]) <= 10).mean()*100:.1f}%")
    print(f"  Residuals within ±10 TOF:   "
          f"{(np.abs(r[:,1]) <= 10).mean()*100:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading training frames...")
    frames = load_frames(TRAIN_FILE)
    print(f"  Loaded {len(frames)} frames")

    # ── Compute streaming-safe scale factor ───────────────────────────────────
    print("\nComputing TOF scale factor...")
    tof_scale = compute_tof_scale(frames)

    # ── Prior Knowledge Mode ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PRIOR KNOWLEDGE MODE")
    print("="*60)

    # Uncomment to use real instrument geometry instead:
    # bin_centres = from_instrument_file("instrument_geometry.json", tof_scale)

    bin_centres = build_prior_bins_from_training(frames, tof_scale)
    kdtree      = make_kdtree(bin_centres, tof_scale)

    save_tokeniser("prior", bin_centres, tof_scale)

    ok = verify_roundtrip(
        frames, kdtree, bin_centres, tof_scale,
        mode_token=SPECIAL["MODE_PRIOR"]
    )
    if not ok:
        raise RuntimeError("Prior mode round-trip failed.")

    report_residual_stats(
        frames, kdtree, bin_centres, tof_scale,
        mode_token=SPECIAL["MODE_PRIOR"]
    )

    # ── Data Driven Mode ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DATA DRIVEN MODE — sample of 5 frames")
    print("="*60)
    # Data driven builds bins per frame — verify on a small sample
    # At stream time: call build_datadriven_bins(frame) before encoding

    dd_ok = True
    for i in range(5):
        frame       = frames[i]
        dd_centres  = build_datadriven_bins(frame, tof_scale)
        dd_kdtree   = make_kdtree(dd_centres, tof_scale)

        tokens, residuals = encode_frame(
            frame, dd_kdtree, dd_centres, tof_scale,
            mode_token=SPECIAL["MODE_DATA_DRIVEN"]
        )
        recovered = decode_frame(tokens, residuals, dd_centres)

        if np.array_equal(frame, recovered):
            print(f"  Frame {i}: round-trip ✓  "
                  f"({len(frame)} events → "
                  f"{len([t for t in tokens if t >= BIN_ID_OFFSET])} tokens)")
        else:
            print(f"  Frame {i}: FAIL ✗")
            dd_ok = False

    if not dd_ok:
        raise RuntimeError("Data driven mode round-trip failed.")

    print("\nBoth modes verified. Ready to proceed to gzip baseline.")


if __name__ == "__main__":
    main()