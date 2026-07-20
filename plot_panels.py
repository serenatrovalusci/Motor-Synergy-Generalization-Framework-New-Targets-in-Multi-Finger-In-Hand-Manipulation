# Usage
# -----
#   python plot_panels.py --config plot_panels_config.json --out plots/panels/mega_panel.pdf
#
# One combined figure, 3 rows x N objects (N = however many objects are in
# --config): row 0 = object photo, row 1 = reorientation success rate WITH
# touch sensors, row 2 = reorientation success rate WITHOUT touch sensors.
# Reorientation is the only success criterion used (see report.tex Section
# III) -- sustained-hold is no longer plotted.
# See plot_panels_config.example.json for the config layout.
"""
Mega-panel: object photos on top, reorientation success-rate curves (with
and without touch sensors) below, one column per object.

Each curve cell overlays the baseline (full action space) and synergy
curves, mean +/- std over 3 seeds, with the success-rate threshold used for
the sample/wall-clock efficiency numbers drawn as a dashed line. Reuses
plot.py's data-loading and plot_band() directly rather than reimplementing
them, so the curve style matches the rest of the pipeline.

Reorientation curves come from local evaluations_anytime.npz files (written
by eval_checkpoints.py). Wall-clock time at the threshold-crossing step is
still looked up from wandb's time/elapsed_sec (dense, logged throughout
training regardless of how sparse the local checkpoints are), so each
object's with_sensors/no_sensors config blocks must still list a
baseline_wandb_group/synergy_wandb_group even though sustained-hold itself
is no longer plotted.
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker

import plot as plot_lib


ROWS = [
    ("with_sensors", "With touch sensors"),
    ("no_sensors",   "Without touch sensors"),
]

COLOR_BASE = "#E07B54"   # orange — baseline, matches plot.py
COLOR_SYN  = "#5B8DB8"   # blue   — synergy, matches plot.py
LABEL_BASE = "Baseline (full action space)"
LABEL_SYN  = "Synergy"


def load_reorientation_curve(sensor_key: str, obj_cfg: dict, args):
    """
    Return (ts_base, succ_base, ts_syn, succ_syn, base_runs, syn_runs) for
    one object/sensor-condition cell, or None if obj_cfg has no block for
    this sensor_key (e.g. no-sensor data not ready yet -- the panel still
    renders, that cell just shows "no data").

    base_runs/syn_runs are the matching wandb Run objects, fetched purely to
    look up wall-clock time at the reorientation threshold-crossing step
    (see print_efficiency) -- the curves themselves are always local.
    """
    block = obj_cfg.get(sensor_key)
    if block is None:
        return None

    ts_base, succ_base = plot_lib.align_and_stack(block["baseline_dirs"], "successes")
    ts_syn, succ_syn = plot_lib.align_and_stack(block["synergy_dirs"], "successes")
    base_runs = plot_lib.fetch_wandb_group_runs(
        args.wandb_project, args.wandb_entity, block["baseline_wandb_group"])
    syn_runs = plot_lib.fetch_wandb_group_runs(
        args.wandb_project, args.wandb_entity, block["synergy_wandb_group"])
    return ts_base, succ_base, ts_syn, succ_syn, base_runs, syn_runs


def time_to_threshold_per_seed(ts, stacked, threshold: float):
    """
    Per-seed threshold crossing, as opposed to plot.py's time_to_threshold()
    which finds a single crossing on the GROUP MEAN curve and replicates it
    for every seed (giving a meaningless std of exactly 0). Here each seed's
    own curve is checked independently.

    Returns (crossing_times, all_crossed):
      crossing_times[i] = ts[idx] of the first point where seed i's own
        curve reaches `threshold`, or ts[-1] if it never does.
      all_crossed = False if any seed never crosses -- callers should not
        report a mean +- std over crossing_times in that case, since it
        would silently blend a real crossing time with an end-of-training
        placeholder.
    """
    crossing_times = []
    all_crossed = True
    for seed_curve in stacked:
        idx = np.where(seed_curve >= threshold)[0]
        if len(idx) > 0:
            crossing_times.append(float(ts[idx[0]]))
        else:
            crossing_times.append(float(ts[-1]))
            all_crossed = False
    return np.array(crossing_times), all_crossed


def print_efficiency(label: str, curves, relative_threshold_frac: float):
    """
    Print sample- and wall-clock-efficiency (timesteps/time to cross a
    relative threshold) for one reorientation cell, using per-seed crossing
    times so the reported spread is a real mean +- std across seeds (see
    time_to_threshold_per_seed). Explicitly reports "NOT REACHED" instead of
    a mean +- std when at least one seed never crosses -- averaging in an
    end-of-training placeholder would understate the true (unknown,
    censored) crossing time.
    """
    ts_base, succ_base, ts_syn, succ_syn, base_runs, syn_runs = curves

    peak = max(succ_base.mean(axis=0).max(), succ_syn.mean(axis=0).max())
    threshold = relative_threshold_frac * float(peak) if peak < 1.0 else 0.80

    ttt_base, base_all_crossed = time_to_threshold_per_seed(ts_base, succ_base, threshold)
    ttt_syn, syn_all_crossed = time_to_threshold_per_seed(ts_syn, succ_syn, threshold)
    wc_base = plot_lib.wall_clock_to_threshold_wandb(base_runs, ttt_base)
    wc_syn = plot_lib.wall_clock_to_threshold_wandb(syn_runs, ttt_syn)

    def fmt_h(s):
        s = int(s)
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"

    def fmt_group(name, ttt, wc, all_crossed):
        # time_to_threshold_per_seed already substitutes ts[-1] (end of the
        # training budget) for any seed that never crosses, so the mean/std
        # below is always a real number -- no blank "--" cells in the table,
        # per the reviewer feedback that censored dashes made Tables II/III
        # unreadable. `all_crossed` still gates a printed note so we (the
        # authors) always know when a value includes a budget-end censored
        # seed rather than a genuine crossing, even though the table itself
        # shows a plain number either way.
        steps_mean, steps_std = ttt.mean() / 1e6, ttt.std() / 1e6
        wc_mean, wc_std = wc.mean(), wc.std()
        note = "" if all_crossed else "  [right-censored: not all seeds crossed within budget]"
        print(f"    {name:8s} steps: {steps_mean:.3f}M +/- {steps_std:.3f}M  {[f'{v / 1e6:.3f}M' for v in ttt]}{note}")
        print(f"    {name:8s} time : {fmt_h(wc_mean)} +/- {fmt_h(wc_std)}  {[fmt_h(v) for v in wc]}{note}")

    print(f"  --- {label}  (peak={peak * 100:.1f}%, threshold={threshold * 100:.1f}%) ---")
    fmt_group("baseline", ttt_base, wc_base, base_all_crossed)
    fmt_group("synergy", ttt_syn, wc_syn, syn_all_crossed)


def draw_cell(ax, curves, relative_threshold_frac: float):
    """
    Draw one reorientation success-rate cell, threshold line included;
    returns (handles, labels) for the shared legend, or None.
    """
    if curves is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                fontsize=9, color="gray", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return None

    ts_base, succ_base, ts_syn, succ_syn, _base_runs, _syn_runs = curves
    plot_lib.plot_band(ax, ts_base, succ_base, COLOR_BASE, LABEL_BASE)
    plot_lib.plot_band(ax, ts_syn, succ_syn, COLOR_SYN, LABEL_SYN)

    peak = max(succ_base.mean(axis=0).max(), succ_syn.mean(axis=0).max())
    threshold = relative_threshold_frac * float(peak) if peak < 1.0 else 0.80
    ax.axhline(threshold, color="gray", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    return ax.get_legend_handles_labels()


def render_mega_panel(objects: list, args):
    n_objects = len(objects)
    fig, axes = plt.subplots(
        3, n_objects, figsize=(2.6 * n_objects, 6.6),
        gridspec_kw={"height_ratios": [1.0, 1.3, 1.3]},
    )
    if n_objects == 1:
        axes = axes.reshape(3, 1)

    legend_handles, legend_labels = None, None

    for ci, obj_cfg in enumerate(objects):
        ax_photo = axes[0, ci]
        img = mpimg.imread(obj_cfg["photo"])
        ax_photo.imshow(img)
        ax_photo.axis("off")
        ax_photo.set_title(obj_cfg["name"], fontsize=11, fontweight="bold")

        for ri, (sensor_key, row_label) in enumerate(ROWS, start=1):
            ax = axes[ri, ci]
            curves = load_reorientation_curve(sensor_key, obj_cfg, args)
            result = draw_cell(ax, curves, args.relative_threshold_frac)
            if result is not None and legend_handles is None:
                legend_handles, legend_labels = result
            if ci == 0:
                ax.set_ylabel(f"{row_label}\nSuccess rate", fontsize=9)

            if curves is not None:
                label = f"{obj_cfg['name']} -- {row_label}"
                print_efficiency(label, curves, args.relative_threshold_frac)

    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=2,
                  fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Render one combined [photos / reorientation with sensors / "
                     "reorientation without sensors] mega-panel across all objects."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="JSON config; see plot_panels_config.example.json.")
    parser.add_argument("--wandb-project", type=str, default="motor-synergy-generalization")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--relative-threshold-frac", type=float, default=0.80)
    parser.add_argument("--out", type=str, default="plots/panels/mega_panel.pdf")
    args = parser.parse_args()

    # Reorientation is the only metric plotted; always read the any-time
    # success definition (from eval_checkpoints.py), never plain
    # evaluations.npz (EvalCallback's end-of-episode/sustained-hold one).
    plot_lib.NPZ_NAME = "evaluations_anytime.npz"

    with open(args.config) as f:
        config = json.load(f)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig = render_mega_panel(config["objects"], args)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()
