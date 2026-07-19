# Usage
# -----
#   python plot_panels.py --config plot_panels_config.json --out-dir plots/panels
#
# One output image PER OBJECT: [photo] [sustained, sensors] [reach, sensors]
# [sustained, no sensors] [reach, no sensors] laid out in a single row.
# See plot_panels_config.example.json for the config layout.
"""
Per-object results panel: object photo + 4 success-rate plots in one row.

Columns (after the photo)
--------------------------
1. Sustained hold, with touch sensors     (wandb eval/success_rate)
2. Reorientation only, with touch sensors  (local evaluations_anytime.npz)
3. Sustained hold, without touch sensors   (wandb eval/success_rate)
4. Reorientation only, without touch sensors (local evaluations_anytime.npz)

Each plot cell overlays the baseline (full action space) and synergy curves,
mean ± std over 3 seeds. Reuses plot.py's data-loading and plot_band()
directly rather than reimplementing them, so the curve style always matches
the single-object figures from plot.py.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker

import plot as plot_lib


COLUMNS = [
    ("with_sensors", "sustained", "Sustained hold\n(with sensors)"),
    ("with_sensors", "reach",     "Reorientation only\n(with sensors)"),
    ("no_sensors",   "sustained", "Sustained hold\n(no sensors)"),
    ("no_sensors",   "reach",     "Reorientation only\n(no sensors)"),
]

COLOR_BASE = "#E07B54"   # orange — baseline, matches plot.py
COLOR_SYN  = "#5B8DB8"   # blue   — synergy, matches plot.py
LABEL_BASE = "Baseline (full action space)"
LABEL_SYN  = "Synergy"


def load_curve(sensor_key: str, metric: str, obj_cfg: dict, args):
    """
    Return (ts_base, succ_base, ts_syn, succ_syn, base_runs, syn_runs) for one
    cell, or None if obj_cfg has no block for this sensor_key (e.g. no-sensor
    data not ready yet -- the panel still renders, that cell just shows
    "no data").

    base_runs/syn_runs (the wandb Run objects) are returned alongside the
    curves for "sustained" cells only -- they're needed afterwards to look
    up wall-clock time at the threshold-crossing step. "reach" cells return
    None for both: efficiency-to-threshold is only computed on sustained-hold
    (the primary metric; see report.tex Section III), reach is descriptive only.
    """
    block = obj_cfg.get(sensor_key)
    if block is None:
        return None

    if metric == "sustained":
        base_runs = plot_lib.fetch_wandb_group_runs(
            args.wandb_project, args.wandb_entity, block["baseline_wandb_group"])
        syn_runs = plot_lib.fetch_wandb_group_runs(
            args.wandb_project, args.wandb_entity, block["synergy_wandb_group"])
        ts_base, succ_base = plot_lib.align_and_stack_wandb(base_runs, args.wandb_metric)
        ts_syn, succ_syn = plot_lib.align_and_stack_wandb(syn_runs, args.wandb_metric)
        return ts_base, succ_base, ts_syn, succ_syn, base_runs, syn_runs

    ts_base, succ_base = plot_lib.align_and_stack(block["baseline_dirs"], "successes")
    ts_syn, succ_syn = plot_lib.align_and_stack(block["synergy_dirs"], "successes")
    return ts_base, succ_base, ts_syn, succ_syn, None, None


def print_sustained_efficiency(label: str, curves, relative_threshold_frac: float):
    """
    Print sample- and wall-clock-efficiency (timesteps/time to cross a
    relative threshold) for one sustained-hold cell, mirroring plot.py's
    console output. Explicitly reports "NOT REACHED" (right-censored)
    instead of a crossing time when a curve's final mean value never
    reaches the threshold -- printing ts[-1] there would silently claim a
    crossing that didn't happen.
    """
    ts_base, succ_base, ts_syn, succ_syn, base_runs, syn_runs = curves

    peak = max(succ_base.mean(axis=0).max(), succ_syn.mean(axis=0).max())
    threshold = relative_threshold_frac * float(peak) if peak < 1.0 else 0.80

    ttt_base = plot_lib.time_to_threshold(ts_base, succ_base, threshold)
    ttt_syn = plot_lib.time_to_threshold(ts_syn, succ_syn, threshold)
    wc_base = plot_lib.wall_clock_to_threshold_wandb(base_runs, ttt_base)
    wc_syn = plot_lib.wall_clock_to_threshold_wandb(syn_runs, ttt_syn)

    def fmt_h(s):
        s = int(s)
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"

    def fmt_group(name, succ, ttt, wc):
        reached = succ.mean(axis=0)[-1] >= threshold
        steps_str = f"{[f'{v / 1e6:.3f}M' for v in ttt]}"
        time_str = f"{[fmt_h(v) for v in wc]}"
        if not reached:
            steps_str += "  [NOT REACHED within budget -- value is end-of-training, not a crossing]"
            time_str += "  [NOT REACHED within budget -- value is end-of-training, not a crossing]"
        print(f"    {name:8s} steps: {steps_str}")
        print(f"    {name:8s} time : {time_str}")

    print(f"  --- {label}  (peak={peak * 100:.1f}%, threshold={threshold * 100:.1f}%) ---")
    fmt_group("baseline", succ_base, ttt_base, wc_base)
    fmt_group("synergy", succ_syn, ttt_syn, wc_syn)


def draw_cell(ax, curves, show_threshold: bool, relative_threshold_frac: float):
    """Draw one success-rate cell; returns (handles, labels) for the shared legend, or None."""
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

    if show_threshold:
        peak = max(succ_base.mean(axis=0).max(), succ_syn.mean(axis=0).max())
        threshold = relative_threshold_frac * float(peak) if peak < 1.0 else 0.80
        ax.axhline(threshold, color="gray", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    return ax.get_legend_handles_labels()


def render_object_panel(obj_cfg: dict, args):
    n_cols = 1 + len(COLUMNS)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.9 * n_cols, 2.6))

    ax_photo = axes[0]
    img = mpimg.imread(obj_cfg["photo"])
    ax_photo.imshow(img)
    ax_photo.axis("off")
    ax_photo.set_title(obj_cfg["name"], fontsize=11, fontweight="bold")

    legend_handles, legend_labels = None, None
    for ci, (sensor_key, metric, col_title) in enumerate(COLUMNS):
        ax = axes[1 + ci]
        curves = load_curve(sensor_key, metric, obj_cfg, args)
        result = draw_cell(ax, curves, args.show_threshold, args.relative_threshold_frac)
        if result is not None and legend_handles is None:
            legend_handles, legend_labels = result
        ax.set_title(col_title, fontsize=8.5)
        if ci == 0:
            ax.set_ylabel("Success rate", fontsize=9)

        if curves is not None and metric == "sustained":
            label = f"{obj_cfg['name']} -- {col_title.replace(chr(10), ' ')}"
            print_sustained_efficiency(label, curves, args.relative_threshold_frac)

    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=2,
                  fontsize=9, frameon=False, bbox_to_anchor=(0.58, -0.08))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Render one [photo + 4 success-rate plots] panel per object."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="JSON config; see plot_panels_config.example.json.")
    parser.add_argument("--wandb-project", type=str, default="motor-synergy-generalization")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-metric", type=str, default="eval/success_rate")
    parser.add_argument("--show-threshold", action="store_true",
                        help="Off by default -- decide per-batch once you see how "
                             "saturated the real curves are.")
    parser.add_argument("--relative-threshold-frac", type=float, default=0.80)
    parser.add_argument("--out-dir", type=str, default="plots/panels")
    parser.add_argument("--ext", type=str, default="pdf", choices=["pdf", "png"])
    args = parser.parse_args()

    # The "reach" columns always read evaluations_anytime.npz (from
    # eval_checkpoints.py); this script never reads plain evaluations.npz.
    plot_lib.NPZ_NAME = "evaluations_anytime.npz"

    with open(args.config) as f:
        config = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    for obj_cfg in config["objects"]:
        print(f"=== {obj_cfg['name']} ===")
        fig = render_object_panel(obj_cfg, args)
        safe_name = obj_cfg["name"].lower().replace(" ", "_")
        out_path = os.path.join(args.out_dir, f"{safe_name}.{args.ext}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
