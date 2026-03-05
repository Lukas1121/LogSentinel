# simulate_events.py
# Step 1: Generate synthetic ESS-like neutron pulse frame data
# Outputs: data/events_train.h5, data/events_val.h5, data/events_test.h5

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

RNG_SEED        = 42
N_FRAMES = 50_000
SPLIT    = (40_000, 5_000, 5_000)  # train / val / test

N_PIXELS        = 100_000       # detector pixel count (ESS-scale)
TOF_MAX         = 100_000       # max time-of-flight in microseconds
BACKGROUND_RATE = 0.05           # mean background counts per pixel per frame

# Bragg peaks: (pixel_centre, tof_centre, pixel_sigma, tof_sigma, peak_intensity)
# Simplified silicon-like d-spacings mapped to detector geometry
BRAGG_PEAKS = [
    (10_000, 20_000, 300, 500,  80),
    (25_000, 35_000, 400, 600,  60),
    (50_000, 50_000, 350, 550, 100),
    (70_000, 65_000, 300, 400,  70),
    (85_000, 80_000, 250, 450,  50),
]

# ── Simulate one pulse frame ──────────────────────────────────────────────────

def simulate_frame(rng: np.random.Generator) -> np.ndarray:
    """
    Returns an array of shape (N_events, 2) with columns [pixel_id, tof].
    Each row is one neutron detection event.
    """
    events = []

    # Background: sparse Poisson noise across all pixels
    # Expected total background events = N_PIXELS * BACKGROUND_RATE
    n_background = rng.poisson(N_PIXELS * BACKGROUND_RATE)
    bg_pixels = rng.integers(0, N_PIXELS, size=n_background)
    bg_tofs    = rng.integers(0, TOF_MAX,  size=n_background)
    events.append(np.stack([bg_pixels, bg_tofs], axis=1))

    # Bragg peaks: Gaussian clusters in pixel/TOF space
    for px_c, tof_c, px_s, tof_s, intensity in BRAGG_PEAKS:
        n_peak = rng.poisson(intensity)
        px  = rng.normal(px_c,  px_s,  size=n_peak).astype(int)
        tof = rng.normal(tof_c, tof_s, size=n_peak).astype(int)

        # Clip to valid range
        px  = np.clip(px,  0, N_PIXELS - 1)
        tof = np.clip(tof, 0, TOF_MAX  - 1)
        events.append(np.stack([px, tof], axis=1))

    frame = np.concatenate(events, axis=0)

    # Sort by pixel then TOF — matches how real DAQ systems emit events
    order = np.lexsort((frame[:, 1], frame[:, 0]))
    return frame[order].astype(np.int32)


# ── Write frames to HDF5 ─────────────────────────────────────────────────────

def write_split(path: Path, frames: list[np.ndarray]):
    with h5py.File(path, "w") as f:
        grp = f.create_group("frames")
        for i, frame in enumerate(frames):
            grp.create_dataset(str(i), data=frame)
        f.attrs["n_frames"]  = len(frames)
        f.attrs["n_pixels"]  = N_PIXELS
        f.attrs["tof_max"]   = TOF_MAX
    print(f"  Wrote {len(frames)} frames → {path}  "
          f"({path.stat().st_size / 1e6:.1f} MB)")


# ── Sanity check plot ─────────────────────────────────────────────────────────

def plot_sample_frame(frame: np.ndarray, title: str = "Sample pulse frame"):
    plt.figure(figsize=(10, 4))
    plt.scatter(frame[:, 1], frame[:, 0], s=0.3, alpha=0.4, c="steelblue")
    plt.xlabel("Time of Flight (µs)")
    plt.ylabel("Pixel ID")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("data/sample_frame.png", dpi=120)
    plt.close()
    print("  Saved sanity-check plot → data/sample_frame.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path("data").mkdir(exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print(f"Simulating {N_FRAMES} pulse frames...")
    frames = [simulate_frame(rng) for i in range(N_FRAMES)]

    # Report average events per frame
    sizes = [len(f) for f in frames]
    print(f"  Events per frame — mean: {np.mean(sizes):.0f}  "
          f"std: {np.std(sizes):.0f}  "
          f"min: {min(sizes)}  max: {max(sizes)}")

    # Write splits
    train, val, test = SPLIT
    print("Writing HDF5 splits...")
    write_split(Path("data/events_train.h5"), frames[:train])
    write_split(Path("data/events_val.h5"),   frames[train:train+val])
    write_split(Path("data/events_test.h5"),  frames[train+val:])

    # Sanity check — plot first frame
    plot_sample_frame(frames[0])
    print("Done. Run: open data/sample_frame.png to verify peaks are visible.")


if __name__ == "__main__":
    main()