import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

SUCCESS_THRESHOLD       = 0.80   # §Metrics: "prescribed success threshold, typically 80 %"
RELATIVE_THRESHOLD_FRAC = 0.60   # fraction of the peak success rate used when absolute is unreachable
N_EVAL_EPISODES         = 200    # §Metrics: "estimated over 200 held-out episodes"


def load_npz(run_dir: str):
    """Load evaluations.npz from a run directory."""
    path = os.path.join(run_dir, "eval_logs", "evaluations.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find: {path}")
    return np.load(path)


def align_and_stack(run_dirs: list[str], key: str):
    """
    Load `key` from each run, interpolate onto a common timestep grid,
    and return (common_timesteps, array of shape [n_seeds, n_evals]).
    """
    all_ts, all_vals = [], []
    for d in run_dirs:
        data = load_npz(d)
        ts   = data["timesteps"]
        vals = data[key]
        all_ts.append(ts)
        all_vals.append(vals.mean(axis=1))

    min_len   = min(len(ts) for ts in all_ts)
    common_ts = all_ts[np.argmin([len(t) for t in all_ts])][:min_len]

    stacked = np.stack([
        np.interp(common_ts, ts, vals[:len(ts)])
        for ts, vals in zip(all_ts, all_vals)
    ], axis=0)

    return common_ts, stacked


def read_tb_elapsed(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse the TensorBoard event file written by TimeLoggingCallback and
    return (steps, elapsed_seconds) for the tag ``time/elapsed_sec``.

    tensorboard is installed automatically alongside stable-baselines3.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as e:
        raise ImportError(
            "tensorboard package not found. Install with:  pip install tensorboard"
        ) from e

    tb_root = os.path.join(run_dir, "tb_logs")
    if not os.path.isdir(tb_root):
        raise FileNotFoundError(
            f"TensorBoard log directory not found: {tb_root}\n"
            "Expected because TimeLoggingCallback was used during training."
        )

    # SB3 writes events into a subdirectory, e.g. tb_logs/SAC_1/events.out.*
    # If a run was resumed, multiple subdirectories may exist (SAC_1, SAC_2,
    # …). Pick the one with the largest total event-file size, which is a
    # reliable proxy for "most complete training log".
    candidates = []
    for dirpath, _dirnames, filenames in os.walk(tb_root):
        event_files = [f for f in filenames if f.startswith("events.out")]
        if event_files:
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, f)) for f in event_files
            )
            candidates.append((dirpath, total_size))

    if not candidates:
        raise FileNotFoundError(
            f"No TensorBoard event files found under {tb_root}"
        )

    tb_dir = max(candidates, key=lambda x: x[1])[0]

    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()

    # Prefer time/fps (logged by TimeLoggingCallback) to derive elapsed time
    # by integration: Δt_i = Δsteps_i / fps_i.  This is robust against SB3
    # batching TF event flushes, which makes raw wall_time unreliable.
    # Fall back to wall_time only if time/fps is absent (old runs without
    # TimeLoggingCallback).
    available = ea.Tags().get("scalars", [])
    if not available:
        raise KeyError(f"No scalar tags found in {tb_dir}.")

    # time/fps is logged by TimeLoggingCallback with an explicit logger.dump()
    # every log_every_steps, so each event's wall_time is a real Unix timestamp
    # of when that training step was reached — no batching issues.
    # Fall back to the densest available tag if time/fps is absent.
    anchor_tag = "time/fps" if "time/fps" in available else \
                 max(available, key=lambda t: len(ea.Scalars(t)))

    events  = ea.Scalars(anchor_tag)
    steps   = np.array([e.step      for e in events], dtype=np.float64)
    wt      = np.array([e.wall_time for e in events], dtype=np.float64)
    elapsed = wt - wt[0]   # seconds since first logged event

    return steps, elapsed


def wall_clock_at_eval_checkpoints(run_dir: str) -> np.ndarray:
    """
    Interpolate ``time/elapsed_sec`` from TensorBoard onto the eval
    timestep grid in evaluations.npz.

    Returns (n_evals,): cumulative wall-clock seconds at each checkpoint.
    """
    eval_ts            = load_npz(run_dir)["timesteps"].astype(np.float64)
    tb_steps, tb_elapsed = read_tb_elapsed(run_dir)
    return np.interp(eval_ts, tb_steps, tb_elapsed,
                     left=0.0, right=tb_elapsed[-1])


def wall_clock_to_threshold(run_dirs: list[str],
                             crossing_timesteps: np.ndarray):
    """
    For each seed, look up its wall-clock time at the group mean crossing
    timestep.  Because all seeds receive the same crossing_timestep (from
    time_to_threshold), the dots show how long each seed had been running
    at the moment the group average crossed the threshold — directly
    consistent with the left subplot.
    """
    result = []
    for d, ts_cross in zip(run_dirs, crossing_timesteps):
        tb_steps, tb_elapsed = read_tb_elapsed(d)
        wc = float(np.interp(ts_cross, tb_steps, tb_elapsed,
                             left=tb_elapsed[0], right=tb_elapsed[-1]))
        result.append(wc)
    return np.array(result)


def mean_crossing_timestep(ts: np.ndarray, stacked: np.ndarray,
                           threshold: float) -> float:
    """
    Return the first timestep where the GROUP MEAN success curve crosses
    ``threshold`` — the same curve drawn in the left subplot.
    Right-censored to ts[-1] if it never crosses.
    """
    mean_curve = stacked.mean(axis=0)
    idx = np.where(mean_curve >= threshold)[0]
    return float(ts[idx[0]]) if len(idx) > 0 else float(ts[-1])


def time_to_threshold(ts: np.ndarray, stacked: np.ndarray,
                      threshold: float = SUCCESS_THRESHOLD):
    """
    Returns the crossing timestep of the GROUP MEAN curve as a constant
    array of shape (n_seeds,) so bar + dots are directly comparable with
    the left subplot.  If the mean never crosses, all values are ts[-1].
    """
    t_cross = mean_crossing_timestep(ts, stacked, threshold)
    return np.full(len(stacked), t_cross)


def plot_band(ax, ts, stacked, color, label):
    """Plot mean ± 1 std band."""
    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0)
    ax.plot(ts, mean, color=color, linewidth=2, label=label)
    ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.2)


def plot_bar_pair(ax, values_base, values_syn, color_base, color_syn,
                  label_base, label_syn, ylabel, fmt_fn=None):
    """
    For each group show:
      - a bar for the mean
      - individual seed dots (jittered) so per-seed values are visible
      - the mean annotated at the bar top
    With only 3 seeds, individual points are more informative than ±std bars.
    """
    groups  = [values_base, values_syn]
    colors  = [color_base,  color_syn]
    centers = [0, 1]

    for vals, color, cx in zip(groups, colors, centers):
        mean = vals.mean()

        # Bar for the mean
        ax.bar(cx, mean, color=color, width=0.5, alpha=0.7, zorder=2)

        # Individual seed dots, jittered horizontally so they don't overlap
        n      = len(vals)
        jitter = np.linspace(-0.12, 0.12, n)
        ax.scatter(cx + jitter, vals, color=color, s=40, zorder=4,
                   edgecolors="white", linewidths=0.6)

        # Mean annotation just above the bar top
        label_str = fmt_fn(mean, None) if fmt_fn is not None else f"{mean:.2f}"
        ax.text(cx, mean * 1.02, label_str,
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(centers)
    ax.set_xticklabels([label_base, label_syn], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(bottom=0)

    if fmt_fn is not None:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_fn))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dirs", nargs=3, required=True,
                        metavar="DIR", help="3 baseline run directories")
    parser.add_argument("--synergy-dirs",  nargs=3, required=True,
                        metavar="DIR", help="3 synergy run directories")
    parser.add_argument("--synergy-label", type=str, default="Synergy")
    parser.add_argument("--threshold",     type=float, default=SUCCESS_THRESHOLD,
                        help="Success threshold for efficiency metrics (default: 0.80)")
    parser.add_argument("--out", type=str, default="plots/comparison.png",
                        help="Base path. Two files are written: "
                             "<base>_success<ext> and <base>_efficiency<ext>.")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Derive two output paths from --out
    base, ext = os.path.splitext(args.out)
    out_success    = f"{base}_success{ext}"
    out_efficiency = f"{base}_efficiency{ext}"

    COLOR_BASE = "#E07B54"   # orange — baseline
    COLOR_SYN  = "#5B8DB8"   # blue   — synergy
    LABEL_BASE = "Baseline (full action space)"
    LABEL_SYN  = args.synergy_label

    # ── 1. Success rate (learning curves) ────────────────────────────────
    ts_base, succ_base = align_and_stack(args.baseline_dirs, "successes")
    ts_syn,  succ_syn  = align_and_stack(args.synergy_dirs,  "successes")

    # ── Resolve effective threshold ───────────────────────────────────────
    # If neither group ever reaches the absolute threshold (args.threshold),
    # all seeds are right-censored and the bar charts collapse to the same
    # value. We detect this and fall back to a relative threshold:
    #   effective_threshold = RELATIVE_THRESHOLD_FRAC * peak_mean_success
    # where peak_mean_success is the highest mean success achieved across
    # both groups (a single shared value keeps the comparison fair).
    # Trigger condition: use the ABSOLUTE threshold only when the runs
    # fully saturate (peak mean success rate == 100%).  Whenever the peak
    # is below 100% the curves haven't converged, so we switch to a
    # peak-relative threshold so the comparison is fair across groups that
    # plateau at different ceilings (and to avoid right-censoring the
    # bar charts when one group never crosses the absolute threshold).
    peak = max(succ_base.mean(axis=0).max(), succ_syn.mean(axis=0).max())
    fully_converged = peak >= 1.0
    if fully_converged:
        effective_threshold = args.threshold
        threshold_label     = f"{int(args.threshold * 100)}%"
        threshold_note      = ""
    else:
        effective_threshold = RELATIVE_THRESHOLD_FRAC * float(peak)
        threshold_label     = f"{RELATIVE_THRESHOLD_FRAC*100:.0f}% of peak ({peak*100:.1f}%)"
        threshold_note      = (
            f"  [peak success {peak*100:.1f}% < 100% — "
            f"using relative threshold = {effective_threshold*100:.1f}%]"
        )
        print(f"INFO: peak mean success = {peak*100:.2f}% (< 100%).")
        print(f"      Using relative threshold: {effective_threshold*100:.2f}% "
              f"(= {RELATIVE_THRESHOLD_FRAC*100:.0f}% of peak)")

    # ── 2 & 3. Sample + wall-clock efficiency ────────────────────────────
    # Both use the same aligned stacked arrays as the success-rate plot,
    # so all three views are guaranteed to be consistent with each other.
    ttt_base = time_to_threshold(ts_base, succ_base, effective_threshold)
    ttt_syn  = time_to_threshold(ts_syn,  succ_syn,  effective_threshold)

    wc_base  = wall_clock_to_threshold(args.baseline_dirs, ttt_base)
    wc_syn   = wall_clock_to_threshold(args.synergy_dirs,  ttt_syn)

    # ── Debug: print every value going into the bar charts ───────────────
    def fmt_h(s):
        s = int(s); return f"{s//3600}h {(s%3600)//60:02d}m"
    print("── Sample efficiency (timesteps) ──────────────────")
    print(f"  baseline seeds : {[f'{v/1e6:.3f}M' for v in ttt_base]}")
    print(f"  synergy  seeds : {[f'{v/1e6:.3f}M' for v in ttt_syn]}")
    print("── Wall-clock efficiency ──────────────────────────")
    print(f"  baseline seeds : {[fmt_h(v) for v in wc_base]}")
    print(f"  synergy  seeds : {[fmt_h(v) for v in wc_syn]}")
    print(f"  effective threshold : {effective_threshold*100:.1f}%")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Figure 1: Success rate (learning curves)
    # ─────────────────────────────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(7, 4.5))
    title1 = f"SAC+HER — {LABEL_BASE} vs {LABEL_SYN}"
    if threshold_note:
        title1 += f"\n{threshold_note}"
    #fig1.suptitle(title1, fontsize=12, fontweight="bold")

    plot_band(ax, ts_base, succ_base, COLOR_BASE, LABEL_BASE)
    plot_band(ax, ts_syn,  succ_syn,  COLOR_SYN,  LABEL_SYN)
    ax.axhline(effective_threshold, color="gray", linewidth=1,
               linestyle="--", alpha=0.6,
               label=f"Threshold ({threshold_label})")
    #ax.set_title("Success rate")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Success rate  (mean ± std)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    ax.legend(framealpha=0.3, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_success, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_success}")
    plt.close(fig1)

    # ─────────────────────────────────────────────────────────────────────
    # Figure 2: Efficiency (sample + wall-clock)
    # ─────────────────────────────────────────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    title2 = f"Efficiency to {threshold_label} — {LABEL_BASE} vs {LABEL_SYN}"
    if threshold_note:
        title2 += f"\n{threshold_note}"
    #fig2.suptitle(title2, fontsize=12, fontweight="bold")

    # Sampling duration
    ax = axes[0]
    plot_bar_pair(
        ax, ttt_base, ttt_syn,
        COLOR_BASE, COLOR_SYN, LABEL_BASE, LABEL_SYN,
        ylabel=f"Timesteps to {threshold_label}  (mean ± std)",
        fmt_fn=lambda x, _: f"{x / 1e6:.2f}M",
    )
    ax.set_title("Sampling duration")

    # Wall-clock efficiency
    ax = axes[1]

    def fmt_time(x, _):
        x = int(x)
        if x >= 3600:
            return f"{x // 3600}h {(x % 3600) // 60:02d}m"
        return f"{x // 60}m {x % 60:02d}s"

    plot_bar_pair(
        ax, wc_base, wc_syn,
        COLOR_BASE, COLOR_SYN, LABEL_BASE, LABEL_SYN,
        ylabel=f"Wall-clock time to {threshold_label}  (mean ± std)",
        fmt_fn=fmt_time,
    )
    ax.set_title("Wall-clock time to threshold")

    plt.tight_layout()
    plt.savefig(out_efficiency, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_efficiency}")
    plt.close(fig2)


if __name__ == "__main__":
    main()