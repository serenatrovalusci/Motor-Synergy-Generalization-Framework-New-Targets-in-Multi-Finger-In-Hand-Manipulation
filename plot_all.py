import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

BASE_DIR = "/Users/serenatrovalusci/Documents/UNI/weights"

OBJECTS = {
    "Small Cube":  "smallblock",
    "Large Cube":  "bigblock",
    "Egg":         "egg",
    "Cylinder":    "cup",
    "Sphere":      "ball",
}

SEEDS = [0, 1, 2]

COLOR_BASE = "#E07B54"
COLOR_SYN  = "#5B8DB8"
LABEL_BASE = "Baseline"
LABEL_SYN  = "Synergy"

RELATIVE_THRESHOLD_FRAC = 0.60

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_dirs(prefix, obj_key):
    folder = "full_actionspace" if prefix == "full" else "synergy"
    tag    = "full" if prefix == "full" else "syn"
    return [
        os.path.join(BASE_DIR, folder, f"{tag}_{obj_key}_seed{s}")
        for s in SEEDS
    ]


def load_npz(run_dir):
    path = os.path.join(run_dir, "eval_logs", "evaluations.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find: {path}")
    return np.load(path)


def align_and_stack(run_dirs):
    all_ts, all_vals = [], []
    for d in run_dirs:
        data = load_npz(d)
        ts   = data["timesteps"]
        vals = data["successes"].mean(axis=1)
        all_ts.append(ts)
        all_vals.append(vals)

    min_len   = min(len(ts) for ts in all_ts)
    common_ts = all_ts[np.argmin([len(t) for t in all_ts])][:min_len]

    stacked = np.stack([
        np.interp(common_ts, ts, vals[:len(ts)])
        for ts, vals in zip(all_ts, all_vals)
    ], axis=0)

    return common_ts, stacked


def plot_band(ax, ts, stacked, color, label):
    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0)
    ax.plot(ts, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.2)


def effective_threshold(succ_base, succ_syn):
    peak = max(succ_base.mean(axis=0).max(),
               succ_syn.mean(axis=0).max())
    if peak >= 1.0:
        return 0.80, "80%"
    else:
        thr = RELATIVE_THRESHOLD_FRAC * float(peak)
        return thr, f"{RELATIVE_THRESHOLD_FRAC*100:.0f}% of peak ({peak*100:.1f}%)"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

# IEEE textwidth ~ 7.16 inches. 5 subplots in a row.
# Height kept small so the figure doesn't dominate the page.
fig, axes = plt.subplots(
    1, 5,
    figsize=(7.16, 2.2),
    sharey=False,
)

for idx, (obj_name, obj_key) in enumerate(OBJECTS.items()):
    ax = axes[idx]

    base_dirs = get_dirs("full", obj_key)
    syn_dirs  = get_dirs("syn",  obj_key)

    ts_base, succ_base = align_and_stack(base_dirs)
    ts_syn,  succ_syn  = align_and_stack(syn_dirs)

    thr, thr_label = effective_threshold(succ_base, succ_syn)

    plot_band(ax, ts_base, succ_base, COLOR_BASE, LABEL_BASE)
    plot_band(ax, ts_syn,  succ_syn,  COLOR_SYN,  LABEL_SYN)

    ax.axhline(thr, color="gray", linewidth=0.8, linestyle="--",
               alpha=0.7, label=f"Threshold")

    ax.set_title(obj_name, fontsize=7, fontweight="bold", pad=3)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_xlabel("Timesteps", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1e6))
    ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    # Only show y-axis label on leftmost subplot
    if idx == 0:
        ax.set_ylabel("Success rate", fontsize=6)
    else:
        ax.set_ylabel("")

    # Legend only on first subplot
    if idx == 0:
        ax.legend(
            fontsize=5.5,
            framealpha=0.3,
            loc="upper left",
            handlelength=1.2,
            borderpad=0.4,
        )

plt.tight_layout(pad=0.5, w_pad=0.4)

out_path = os.path.join(BASE_DIR, "learning_curves_1x5.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close()

# ─────────────────────────────────────────────
# Per-seed statistics for LaTeX table
# ─────────────────────────────────────────────

def read_tb_elapsed(run_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError("pip install tensorboard")

    tb_root = os.path.join(run_dir, "tb_logs")
    candidates = []
    for dirpath, _, filenames in os.walk(tb_root):
        event_files = [f for f in filenames if f.startswith("events.out")]
        if event_files:
            total_size = sum(os.path.getsize(os.path.join(dirpath, f)) for f in event_files)
            candidates.append((dirpath, total_size))

    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event files found under {tb_root}")

    tb_dir = max(candidates, key=lambda x: x[1])[0]
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    anchor_tag = "time/fps" if "time/fps" in available else max(available, key=lambda t: len(ea.Scalars(t)))
    events  = ea.Scalars(anchor_tag)
    steps   = np.array([e.step      for e in events], dtype=np.float64)
    wt      = np.array([e.wall_time for e in events], dtype=np.float64)
    elapsed = wt - wt[0]
    return steps, elapsed


def time_to_threshold_per_seed(ts, stacked, threshold):
    """Crossing timestep for each individual seed."""
    crossings = []
    for seed_curve in stacked:
        idx = np.where(seed_curve >= threshold)[0]
        crossings.append(float(ts[idx[0]]) if len(idx) > 0 else float(ts[-1]))
    return np.array(crossings)


def wall_clock_per_seed(run_dirs, crossing_timesteps):
    result = []
    for d, ts_cross in zip(run_dirs, crossing_timesteps):
        tb_steps, tb_elapsed = read_tb_elapsed(d)
        wc = float(np.interp(ts_cross, tb_steps, tb_elapsed,
                             left=tb_elapsed[0], right=tb_elapsed[-1]))
        result.append(wc)
    return np.array(result)


def fmt_h(s):
    s = int(s)
    return f"{s//3600}h {(s%3600)//60:02d}m"


print("\n── Per-seed statistics for LaTeX table ─────────────")
for obj_name, obj_key in OBJECTS.items():
    base_dirs = get_dirs("full", obj_key)
    syn_dirs  = get_dirs("syn",  obj_key)

    ts_base, succ_base = align_and_stack(base_dirs)
    ts_syn,  succ_syn  = align_and_stack(syn_dirs)

    thr, thr_label = effective_threshold(succ_base, succ_syn)

    ttt_base = time_to_threshold_per_seed(ts_base, succ_base, thr)
    ttt_syn  = time_to_threshold_per_seed(ts_syn,  succ_syn,  thr)

    wc_base = wall_clock_per_seed(base_dirs, ttt_base)
    wc_syn  = wall_clock_per_seed(syn_dirs,  ttt_syn)

    print(f"\n{obj_name} (threshold: {thr_label}):")
    print(f"  BL steps: {ttt_base/1e6} → mean={ttt_base.mean()/1e6:.2f}M  std={ttt_base.std()/1e6:.2f}M")
    print(f"  SC steps: {ttt_syn/1e6}  → mean={ttt_syn.mean()/1e6:.2f}M  std={ttt_syn.std()/1e6:.2f}M")
    print(f"  BL time : {[fmt_h(v) for v in wc_base]} → mean={fmt_h(wc_base.mean())}  std={wc_base.std()/3600:.2f}h")
    print(f"  SC time : {[fmt_h(v) for v in wc_syn]}  → mean={fmt_h(wc_syn.mean())}  std={wc_syn.std()/3600:.2f}h")

