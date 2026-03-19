"""
analyse_logs.py
Visual analysis of synthetic M365 audit log distributions.

Produces 5 plots saved to results/analysis/:
  1. operation_frequency.png   -- top 20 operations by count
  2. workload_distribution.png -- SharePoint/AAD/Exchange/Teams breakdown
  3. time_heatmap.png          -- hour-of-day x day-of-week activity heatmap
  4. ip_country_dist.png       -- IP presence rate + top country codes
  5. user_variance.png         -- per-user event count distribution + top operations per user

Usage:
    python analyse_logs.py                    # analyses data/train.jsonl
    python analyse_logs.py --file data/val.jsonl
    python analyse_logs.py --sample 100000   # analyse first 100k events only (faster)
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

OUT_DIR = Path("results/analysis")

WORKLOAD_COLORS = {
    "SharePoint":          "#2E86AB",
    "OneDrive":            "#A8DADC",
    "AzureActiveDirectory":"#E63946",
    "Exchange":            "#F4A261",
    "MicrosoftTeams":      "#2A9D8F",
    "UNKNOWN":             "#999999",
}

STYLE = {
    "bg":      "#0f1117",
    "panel":   "#1a1d27",
    "text":    "#e8eaf0",
    "subtext": "#8b8fa8",
    "accent":  "#4f8ef7",
    "grid":    "#2a2d3a",
}

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ── Loader ────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path, max_events: int | None = None) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_events and i >= max_events:
                break
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events

# ── Plot helpers ──────────────────────────────────────────────────────────────

def dark_fig(w=12, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])
    ax.tick_params(colors=STYLE["subtext"], labelsize=9)
    ax.spines["bottom"].set_color(STYLE["grid"])
    ax.spines["left"].set_color(STYLE["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=STYLE["grid"], linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax

def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=STYLE["bg"], edgecolor="none")
    plt.close(fig)
    print(f"  Saved -> {path}")

# ── Plot 1: Operation frequency ───────────────────────────────────────────────

def plot_operation_frequency(events: list[dict]):
    print("Plotting operation frequency...")
    counts = Counter(e.get("Operation", "UNKNOWN") for e in events)
    top = counts.most_common(20)
    ops, vals = zip(*top)

    fig, ax = dark_fig(14, 7)
    colors = [WORKLOAD_COLORS.get(
        next((e.get("Workload") for e in events
              if e.get("Operation") == op), "UNKNOWN"),
        STYLE["accent"]
    ) for op in ops]

    bars = ax.barh(ops, vals, color=colors, height=0.65, alpha=0.9)
    ax.invert_yaxis()

    # Value labels
    for bar, val in zip(bars, vals):
        ax.text(val + max(vals) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", color=STYLE["subtext"], fontsize=8)

    ax.set_xlabel("Event count", color=STYLE["subtext"], fontsize=10)
    ax.set_title("Top 20 Operations by Frequency",
                 color=STYLE["text"], fontsize=13, pad=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))

    # Workload legend
    seen = {}
    for e in events:
        wl = e.get("Workload", "UNKNOWN")
        if wl not in seen:
            seen[wl] = WORKLOAD_COLORS.get(wl, STYLE["accent"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.9)
               for wl, c in seen.items()]
    ax.legend(handles, list(seen.keys()), loc="lower right",
              facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
              labelcolor=STYLE["text"], fontsize=8)

    save(fig, "operation_frequency.png")

# ── Plot 2: Workload distribution ─────────────────────────────────────────────

def plot_workload_distribution(events: list[dict]):
    print("Plotting workload distribution...")
    counts = Counter(e.get("Workload", "UNKNOWN") for e in events)
    labels = list(counts.keys())
    vals   = list(counts.values())
    colors = [WORKLOAD_COLORS.get(l, STYLE["accent"]) for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(STYLE["bg"])

    # Pie
    ax1.set_facecolor(STYLE["bg"])
    wedges, texts, autotexts = ax1.pie(
        vals, labels=None, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": STYLE["bg"], "linewidth": 2},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_color(STYLE["bg"])
        t.set_fontsize(9)
        t.set_fontweight("bold")
    ax1.legend(wedges, [f"{l} ({v:,})" for l, v in zip(labels, vals)],
               loc="lower center", bbox_to_anchor=(0.5, -0.12),
               facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
               labelcolor=STYLE["text"], fontsize=8, ncol=2)
    ax1.set_title("Workload Distribution", color=STYLE["text"],
                  fontsize=13, fontweight="bold", pad=15)

    # Bar — events per workload over hours
    hour_workload: dict[str, list[int]] = defaultdict(lambda: [0]*24)
    for e in events:
        ts = e.get("CreationTime", "")
        wl = e.get("Workload", "UNKNOWN")
        try:
            h = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").hour
            hour_workload[wl][h] += 1
        except (ValueError, IndexError):
            pass

    ax2.set_facecolor(STYLE["panel"])
    ax2.tick_params(colors=STYLE["subtext"], labelsize=9)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax2.spines[sp].set_color(STYLE["grid"])
    ax2.grid(axis="y", color=STYLE["grid"], linewidth=0.5, alpha=0.7)
    ax2.set_axisbelow(True)

    hours = list(range(24))
    bottom = np.zeros(24)
    for wl, color in WORKLOAD_COLORS.items():
        if wl in hour_workload:
            vals_h = np.array(hour_workload[wl])
            ax2.bar(hours, vals_h, bottom=bottom, color=color,
                    alpha=0.9, label=wl, width=0.85)
            bottom += vals_h

    ax2.set_xlabel("Hour of day", color=STYLE["subtext"], fontsize=10)
    ax2.set_ylabel("Event count", color=STYLE["subtext"], fontsize=10)
    ax2.set_title("Workload Activity by Hour",
                  color=STYLE["text"], fontsize=13, fontweight="bold", pad=15)
    ax2.set_xticks(hours)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))

    save(fig, "workload_distribution.png")

# ── Plot 3: Time heatmap ──────────────────────────────────────────────────────

def plot_time_heatmap(events: list[dict]):
    print("Plotting time heatmap...")
    grid = np.zeros((7, 24))  # [dow, hour]

    for e in events:
        ts = e.get("CreationTime", "")
        try:
            dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
            grid[dt.weekday(), dt.hour] += 1
        except (ValueError, IndexError):
            pass

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(STYLE["bg"])
    ax.set_facecolor(STYLE["panel"])

    im = ax.imshow(grid, aspect="auto", cmap="Blues",
                   interpolation="nearest")

    # Annotations
    vmax = grid.max()
    for i in range(7):
        for j in range(24):
            val = grid[i, j]
            color = "white" if val > vmax * 0.5 else STYLE["subtext"]
            ax.text(j, i, f"{val/1000:.1f}k" if val >= 1000 else str(int(val)),
                    ha="center", va="center", fontsize=6.5, color=color)

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                        rotation=45, ha="right", fontsize=8,
                        color=STYLE["subtext"])
    ax.set_yticks(range(7))
    ax.set_yticklabels(DAYS, fontsize=9, color=STYLE["subtext"])
    for sp in ax.spines.values():
        sp.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.tick_params(colors=STYLE["subtext"], labelsize=8)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))

    ax.set_title("Event Activity Heatmap — Hour × Day of Week",
                 color=STYLE["text"], fontsize=13, fontweight="bold", pad=15)
    fig.patch.set_facecolor(STYLE["bg"])

    save(fig, "time_heatmap.png")

# ── Plot 4: IP + country distribution ────────────────────────────────────────

def plot_ip_country(events: list[dict]):
    print("Plotting IP / country distribution...")

    has_ip = sum(1 for e in events if e.get("ClientIP"))
    no_ip  = len(events) - has_ip

    country_counts: Counter = Counter()
    for e in events:
        loc = e.get("Location")
        if isinstance(loc, dict):
            cc = loc.get("CountryCode")
            if cc:
                country_counts[cc] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(STYLE["bg"])

    # IP presence donut
    ax1.set_facecolor(STYLE["bg"])
    wedges, _, autotexts = ax1.pie(
        [has_ip, no_ip],
        labels=None,
        colors=[STYLE["accent"], STYLE["grid"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": STYLE["bg"], "linewidth": 2, "width": 0.5},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_color(STYLE["text"])
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax1.legend(wedges, [f"Has IP ({has_ip:,})", f"No IP ({no_ip:,})"],
               loc="lower center", bbox_to_anchor=(0.5, -0.08),
               facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
               labelcolor=STYLE["text"], fontsize=9)
    ax1.set_title(f"ClientIP Presence\n({has_ip/len(events)*100:.1f}% present)",
                  color=STYLE["text"], fontsize=13, fontweight="bold", pad=15)

    # Country bar
    ax2.set_facecolor(STYLE["panel"])
    ax2.tick_params(colors=STYLE["subtext"], labelsize=9)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax2.spines[sp].set_color(STYLE["grid"])
    ax2.grid(axis="y", color=STYLE["grid"], linewidth=0.5, alpha=0.7)
    ax2.set_axisbelow(True)

    if country_counts:
        top_cc = country_counts.most_common(10)
        cc_labels, cc_vals = zip(*top_cc)
        colors_cc = [STYLE["accent"] if cc == "DK" else "#E63946"
                     if cc in ("RU", "CN", "NG") else STYLE["subtext"]
                     for cc in cc_labels]
        bars = ax2.bar(cc_labels, cc_vals, color=colors_cc, alpha=0.9, width=0.6)
        for bar, val in zip(bars, cc_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cc_vals)*0.01,
                     f"{val:,}", ha="center", color=STYLE["subtext"], fontsize=8)
        ax2.set_title("Top Country Codes (Location field)",
                      color=STYLE["text"], fontsize=13, fontweight="bold", pad=15)
        ax2.set_ylabel("Event count", color=STYLE["subtext"], fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))
    else:
        ax2.text(0.5, 0.5, "No Location field found\nin sampled events",
                 ha="center", va="center", color=STYLE["subtext"],
                 transform=ax2.transAxes, fontsize=11)
        ax2.set_title("Country Code Distribution",
                      color=STYLE["text"], fontsize=13, fontweight="bold")

    save(fig, "ip_country_dist.png")

# ── Plot 5: User activity variance ───────────────────────────────────────────

def plot_user_variance(events: list[dict]):
    print("Plotting user activity variance...")

    user_counts:  Counter = Counter()
    user_ops:     dict[str, Counter] = defaultdict(Counter)

    for e in events:
        uid = e.get("UserId", "unknown")
        op  = e.get("Operation", "UNKNOWN")
        user_counts[uid] += 1
        user_ops[uid][op] += 1

    counts = list(user_counts.values())
    n_users = len(counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(STYLE["bg"])

    # Histogram of events per user
    ax1.set_facecolor(STYLE["panel"])
    ax1.tick_params(colors=STYLE["subtext"], labelsize=9)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax1.spines[sp].set_color(STYLE["grid"])
    ax1.grid(axis="y", color=STYLE["grid"], linewidth=0.5, alpha=0.7)
    ax1.set_axisbelow(True)

    ax1.hist(counts, bins=40, color=STYLE["accent"], alpha=0.85, edgecolor=STYLE["bg"])
    ax1.axvline(np.mean(counts), color="#E63946", linewidth=1.5,
                linestyle="--", label=f"Mean: {np.mean(counts):.0f}")
    ax1.axvline(np.median(counts), color="#F4A261", linewidth=1.5,
                linestyle="--", label=f"Median: {np.median(counts):.0f}")
    ax1.set_xlabel("Events per user", color=STYLE["subtext"], fontsize=10)
    ax1.set_ylabel("Number of users", color=STYLE["subtext"], fontsize=10)
    ax1.set_title(f"User Activity Distribution\n({n_users} users)",
                  color=STYLE["text"], fontsize=13, fontweight="bold", pad=15)
    ax1.legend(facecolor=STYLE["panel"], edgecolor=STYLE["grid"],
               labelcolor=STYLE["text"], fontsize=9)

    # Top operations per user — show variance across users
    # Sample 200 users, get their top operation, see how diverse it is
    sample_users = random.sample(list(user_ops.keys()),
                                 min(200, len(user_ops)))
    top_op_per_user = Counter(
        user_ops[u].most_common(1)[0][0] for u in sample_users
    )
    top_ops = top_op_per_user.most_common(12)
    op_labels, op_vals = zip(*top_ops)

    ax2.set_facecolor(STYLE["panel"])
    ax2.tick_params(colors=STYLE["subtext"], labelsize=8.5)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax2.spines[sp].set_color(STYLE["grid"])
    ax2.grid(axis="x", color=STYLE["grid"], linewidth=0.5, alpha=0.7)
    ax2.set_axisbelow(True)

    bars = ax2.barh(op_labels, op_vals, color=STYLE["accent"],
                    alpha=0.85, height=0.6)
    ax2.invert_yaxis()
    for bar, val in zip(bars, op_vals):
        ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", color=STYLE["subtext"], fontsize=8)

    ax2.set_xlabel("Number of users where this is top operation",
                   color=STYLE["subtext"], fontsize=9)
    ax2.set_title("Most Common Top Operation per User\n(sample of 200 users — diversity check)",
                  color=STYLE["text"], fontsize=13, fontweight="bold", pad=15)

    save(fig, "user_variance.png")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse synthetic M365 log distributions")
    parser.add_argument("--file",   default="data/train.jsonl",
                        help="JSONL file to analyse (default: data/train.jsonl)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Analyse first N events only (default: all)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: {path} not found. Run generate_logs.py first.")
        return

    print(f"\nLoading {path}...")
    events = load_jsonl(path, max_events=args.sample)
    print(f"  Loaded {len(events):,} events")

    if args.sample:
        print(f"  (Sampled first {args.sample:,} of full dataset)")

    print(f"\nGenerating plots -> {OUT_DIR}/\n")

    plot_operation_frequency(events)
    plot_workload_distribution(events)
    plot_time_heatmap(events)
    plot_ip_country(events)
    plot_user_variance(events)

    print(f"\nDone. Open results/analysis/ to view plots.")
    print(f"\nKey things to look for:")
    print(f"  Operation frequency  — is FileAccessed dominating too heavily?")
    print(f"  Workload heatmap     — is activity realistic across hours?")
    print(f"  Time heatmap         — are weekends quiet? Morning peak present?")
    print(f"  IP presence          — should be ~78%, check against real tenant")
    print(f"  User variance        — if all users have same top op, data is too uniform")


if __name__ == "__main__":
    main()