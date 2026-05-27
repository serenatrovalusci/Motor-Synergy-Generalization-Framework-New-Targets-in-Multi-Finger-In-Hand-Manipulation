#!/usr/bin/env python3
"""
Offline synergy extraction and diagnostic-plot pipeline.

This script is the *upstream* counterpart of ``train_sac_her_synergy.py``:
given a recorded dataset of joint-action trajectories with shape ``(N, T, M)``
(N episodes, T timesteps per episode, M joint DOFs), it fits a
``SpatialSynergy`` model and produces the figures used in the thesis to
characterise the resulting low-dimensional basis.

Outputs
-------
- A reconstruction-quality curve  R^2(K)  evaluated on held-out trajectories,
  obtained by re-fitting the synergy model for each K in a sweep
  (train / test split is *by trajectory* to avoid temporal leakage).
- A weight matrix heatmap  W in R^{K x M}  with per-cell numeric annotations.
- One bar plot per individual synergy showing its joint-weight pattern.
- A paper-style ``joints x synergies`` grid of per-joint contribution time
  series  C_{m,k}(t) = a_k(t) * W[k, m]  for a single trajectory.

Module layout
-------------
- R^2 metrics                   :func:`r2_global`, :func:`r2_per_dof_mean_std`
- Range / distribution stats    :func:`summarize_ranges`
- Plotting                      :func:`plot_joint_synergy_contributions_grid`
- CLI / orchestration           :func:`main`

Example
-------
    python synergy_extract_analyze.py \\
        --npz-path data/hand_actions.npz \\
        --array-key actions \\
        --method pca \\
        --n-synergies 6 \\
        --k-list "1,2,4,6,8,10,12,16,20" \\
        --traj-idx 0
"""

import argparse
import os
import pickle  # noqa: F401  (kept for parity with the original script)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Project-local synergy implementation: must expose
# ``extract / encode / decode`` and a ``synergies`` attribute of shape (K, M).
from synergy import SpatialSynergy


# =============================================================================
# R^2 metrics
# =============================================================================

def r2_global(x, y, eps=1e-12):
    """
    Compute the *global* coefficient of determination between data and
    reconstruction.

    R^2 = 1 - SS_res / SS_tot, where SS_tot is computed against the global
    mean of ``x`` (i.e. across ALL elements, not per-feature). This matches
    the definition used in the synergy literature when reporting a single
    scalar reconstruction quality.

    Parameters
    ----------
    x : array-like
        Ground-truth array (any shape).
    y : array-like
        Reconstruction with the same shape as ``x``.
    eps : float
        Small constant added to the denominator for numerical stability.

    Returns
    -------
    float
        Global R^2 in (-inf, 1].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    num = np.sum((x - y) ** 2)
    den = np.sum((x - np.mean(x)) ** 2) + eps
    return 1.0 - float(num / den)


def r2_per_dof_mean_std(x, y):
    """
    Compute R^2 separately for each joint DOF and return summary statistics.

    Each DOF gets its own scalar R^2 (computed via :func:`r2_global` on the
    corresponding slice). The function returns the mean and standard deviation
    across DOFs, plus the full per-DOF array for downstream plotting.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays of shape ``(N, T, M)`` -- N trajectories, T timesteps, M DOFs.

    Returns
    -------
    mean_r2 : float
        Average R^2 across the M DOFs.
    std_r2 : float
        Standard deviation of R^2 across DOFs.
    r2_d : np.ndarray, shape (M,)
        Per-DOF R^2 values.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape, f"Shape mismatch x={x.shape}, y={y.shape}"
    assert x.ndim == 3, f"Expected (N,T,M), got {x.shape}"

    M = x.shape[-1]
    r2_d = np.array(
        [r2_global(x[..., j], y[..., j]) for j in range(M)],
        dtype=np.float64,
    )
    return float(r2_d.mean()), float(r2_d.std()), r2_d


# =============================================================================
# Range / distribution statistics
# =============================================================================

def summarize_ranges(name, x):
    """
    Print and return per-dimension and global range statistics for a 3D array.

    Useful for sanity-checking that the reconstruction stays inside a
    comparable numerical range to the original data (and therefore inside the
    action bounds expected by the simulator).

    Parameters
    ----------
    name : str
        Label used in the printed report (e.g. ``"DATA"`` or ``"RECON"``).
    x : np.ndarray
        Array of shape ``(N, T, M)``.

    Returns
    -------
    dict
        Dictionary with global min/max/p1/p99 plus per-dimension arrays.
    """
    x = np.asarray(x)
    assert x.ndim == 3, f"{name}: expected 3D (N,T,M), got {x.shape}"
    flat = x.reshape(-1, x.shape[-1])

    # Per-dimension stats: one value per joint DOF.
    per_min = flat.min(axis=0)
    per_max = flat.max(axis=0)
    per_p1 = np.percentile(flat, 1, axis=0)
    per_p99 = np.percentile(flat, 99, axis=0)

    # Global stats: a single scalar across all elements.
    gmin = float(flat.min())
    gmax = float(flat.max())
    gp1 = float(np.percentile(flat, 1))
    gp99 = float(np.percentile(flat, 99))

    print(
        f"\n[{name}] global min/max: {gmin:.4f} / {gmax:.4f}  |  "
        f"p1/p99: {gp1:.4f} / {gp99:.4f}"
    )
    print(f"[{name}] per-dim min range: {per_min.min():.4f} .. {per_min.max():.4f}")
    print(f"[{name}] per-dim max range: {per_max.min():.4f} .. {per_max.max():.4f}")
    print(f"[{name}] per-dim p99 range: {per_p99.min():.4f} .. {per_p99.max():.4f}")

    return {
        "global_min": gmin,
        "global_max": gmax,
        "global_p1": gp1,
        "global_p99": gp99,
        "per_min": per_min,
        "per_max": per_max,
        "per_p1": per_p1,
        "per_p99": per_p99,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_joint_synergy_contributions_grid(
    model,
    activities,          # (N, T, K)
    traj_idx=0,
    joints=None,         # list of joint indices (0-based). None -> all
    synergies=None,      # list of synergy indices (0-based). None -> all
    time=None,           # array (T,) in seconds; if None -> linspace(0,1,T)
    y_mode="global",     # "global" or "per_col"
    figsize_scale=(2.6, 1.35),
    save_path=None,
    dpi=200,
):
    """
    Render a paper-style grid of per-joint, per-synergy contribution traces.

    Layout::

                synergy 1   synergy 2   ...   synergy K
        joint 1   C_{1,1}(t)  C_{1,2}(t)  ...   C_{1,K}(t)
        joint 2   C_{2,1}(t)  C_{2,2}(t)  ...   C_{2,K}(t)
        ...
        joint M   C_{M,1}(t)  ...

    where  C_{m,k}(t) = a_k(t) * W[k, m]  is the contribution of synergy
    ``k`` to joint ``m`` at time ``t`` for a single trajectory.

    Parameters
    ----------
    model : SpatialSynergy
        Fitted synergy model. ``model.synergies`` must have shape (K, M).
    activities : np.ndarray
        Per-trajectory synergy activities, shape (N, T, K).
    traj_idx : int
        Index of the trajectory to plot.
    joints : list of int, optional
        Subset of joint indices (0-based). Defaults to all M joints.
    synergies : list of int, optional
        Subset of synergy indices (0-based). Defaults to all K synergies.
    time : np.ndarray, optional
        Time axis of length T. Defaults to ``linspace(0, 1, T)`` (normalised
        time -- absolute units don't matter for the qualitative plot).
    y_mode : {"global", "per_col"}
        - "global":  shared symmetric y-limits across the whole grid.
        - "per_col": symmetric y-limits per synergy column (lets weak
          synergies still be visible when a single one dominates).
    figsize_scale : tuple of (float, float)
        Per-cell width / height multipliers used to compute the figure size.
    save_path : str, optional
        If given, the figure is saved to this path; otherwise it is only
        rendered (and then closed to avoid leaking figures).
    dpi : int
        DPI used when ``save_path`` is provided.
    """
    A = np.asarray(activities)
    a = A[traj_idx]  # (T, K)

    W = np.asarray(model.synergies)  # (K, M)
    K, M = W.shape
    T = a.shape[0]

    if joints is None:
        joints = list(range(M))
    if synergies is None:
        synergies = list(range(K))

    R = len(joints)
    C = len(synergies)

    if time is None:
        time = np.linspace(0.0, 1.0, T)

    # ------------------------------------------------------------------
    # Pre-compute contributions C[r, c, t] for every visible cell.
    # Doing this up-front makes the per-axis y-limit logic below trivial.
    # ------------------------------------------------------------------
    contrib = np.zeros((R, C, T), dtype=np.float64)
    for ri, m in enumerate(joints):
        for ci, k in enumerate(synergies):
            contrib[ri, ci] = a[:, k] * W[k, m]

    # ------------------------------------------------------------------
    # Y-axis limit policy.
    # ------------------------------------------------------------------
    if y_mode == "global":
        ymax = np.max(np.abs(contrib)) + 1e-12
        ylims = [(-ymax, ymax)] * C
    elif y_mode == "per_col":
        ylims = []
        for ci in range(C):
            ymax = np.max(np.abs(contrib[:, ci, :])) + 1e-12
            ylims.append((-ymax, ymax))
    else:
        raise ValueError("y_mode must be 'global' or 'per_col'")

    fig_w = figsize_scale[0] * C
    fig_h = figsize_scale[1] * R
    fig, axes = plt.subplots(R, C, figsize=(fig_w, fig_h), sharex=True, sharey=False)

    # Normalise ``axes`` to a 2D array regardless of R, C so the loop below
    # can index ``axes[ri, ci]`` unconditionally.
    if R == 1 and C == 1:
        axes = np.array([[axes]])
    elif R == 1:
        axes = axes.reshape(1, -1)
    elif C == 1:
        axes = axes.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Draw each cell: zero baseline + filled contribution above / below.
    # ------------------------------------------------------------------
    for ri, m in enumerate(joints):
        for ci, k in enumerate(synergies):
            ax = axes[ri, ci]
            y = contrib[ri, ci]

            ax.axhline(0, linewidth=1)

            ax.fill_between(time, 0, y, where=(y >= 0), interpolate=True, alpha=0.85)
            ax.fill_between(time, 0, y, where=(y <= 0), interpolate=True, alpha=0.85)

            ax.set_ylim(*ylims[ci])
            ax.set_xlim(time[0], time[-1])

            # Column titles: top row only.
            if ri == 0:
                ax.set_title(f"synergy {k+1}", fontsize=18, pad=12)
            # Row labels: left column only, rotated horizontally.
            if ci == 0:
                ax.set_ylabel(
                    f"joint {m+1}",
                    fontsize=22,
                    rotation=0,
                    labelpad=42,
                    va="center",
                )

            # X-axis label only on the bottom row.
            if ri == R - 1:
                ax.set_xlabel("time (s)", fontsize=20)
            else:
                ax.set_xlabel("")

            ax.tick_params(axis="both", which="both", labelsize=16)
            if ri != R - 1:
                ax.set_xticklabels([])

    plt.tight_layout()

    # Always save when a path is provided so the user gets the artefact even
    # in headless / non-interactive environments where ``plt.show()`` is a no-op.
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] Saved grid plot to: {os.path.abspath(save_path)}")

    # Close the figure explicitly to avoid memory leaks when this function is
    # called repeatedly (e.g. inside a sweep).
    plt.close(fig)


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Parse CLI arguments, fit the synergy model and produce all diagnostics."""
    parser = argparse.ArgumentParser(
        description="Offline synergy extraction + reconstruction-quality diagnostics."
    )

    # --- Data ---------------------------------------------------------------
    parser.add_argument(
        "--npz-path",
        type=str,
        required=True,
        help="Path to .npz file containing array (N,T,M).",
    )
    parser.add_argument(
        "--array-key",
        type=str,
        default="actions",
        help="Key inside the .npz file holding the (N,T,M) array.",
    )

    # --- Model --------------------------------------------------------------
    parser.add_argument(
        "--method",
        type=str,
        default="pca",
        choices=["pca", "nmf", "negative-nmf"],
        help="Synergy extraction method.",
    )
    parser.add_argument(
        "--n-synergies",
        type=int,
        default=5,
        help="Number of synergies K used for the bundle and the recon plots.",
    )

    # --- Output -------------------------------------------------------------
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Where to save synergy bundle (.pkl). If None, save next to npz.",
    )
    parser.add_argument(
        "--save-recon",
        action="store_true",
        help="Save original/recon/activities/stats to compressed .npz.",
    )
    parser.add_argument(
        "--recon-out",
        type=str,
        default=None,
        help="Output path for the saved reconstruction artefact (when --save-recon).",
    )

    parser.add_argument(
        "--traj-idx",
        type=int,
        default=0,
        help="Trajectory index used for the contribution-grid plot.",
    )

    # --- R^2 sweep ----------------------------------------------------------
    parser.add_argument(
        "--k-max",
        type=int,
        default=20,
        help="Upper bound of K when --k-list is not given (sweep is 1..k-max).",
    )
    parser.add_argument(
        "--k-list",
        type=str,
        default=None,
        help='Comma-separated list like "1,2,4,6,8,10,12,16,20". '
             "If None uses 1..k-max.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Test fraction. Split is BY EPISODES (trajectories) to avoid leakage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the train/test split.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1) Load data
    # -------------------------------------------------------------------------
    npz = np.load(args.npz_path)

    if args.array_key not in npz:
        raise KeyError(f"Key '{args.array_key}' not found. Keys: {list(npz.keys())}")

    data = np.asarray(npz[args.array_key])
    if data.ndim != 3:
        raise ValueError(f"Expected data (N,T,M), got {data.shape}")

    N, T, M = data.shape
    print(f"Loaded '{args.array_key}': {data.shape}")

    # -------------------------------------------------------------------------
    # 2) Fit the chosen-K model used for the bundle and the per-synergy plots.
    # -------------------------------------------------------------------------
    K = int(args.n_synergies)
    model = SpatialSynergy(K, method=args.method)
    synergies = model.extract(data)
    print("Synergies shape:", np.asarray(synergies).shape)

    activities = model.encode(data)          # (N, T, K)
    recon = model.decode(activities)         # (N, T, M)

    assert activities is not None and recon is not None
    assert activities.shape == (N, T, K), f"activities shape {activities.shape} != {(N,T,K)}"
    assert recon.shape[0] == N and recon.shape[1] == T, f"recon shape {recon.shape} unexpected"

    # -------------------------------------------------------------------------
    # 3) Range / distribution sanity checks on data and reconstruction.
    # -------------------------------------------------------------------------
    summarize_ranges("DATA", data)
    summarize_ranges("RECON", recon)

    # -------------------------------------------------------------------------
    # 4) R^2 vs K with a TRAIN/TEST split *by trajectory* to avoid leakage.
    #    (Splitting by timestep would leak temporally-correlated frames into
    #    both folds and inflate the score.)
    # -------------------------------------------------------------------------
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=float(args.test_split),
        random_state=int(args.seed),
        shuffle=True,
    )
    train_data = data[train_idx]
    test_data = data[test_idx]

    if args.k_list is not None:
        K_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    else:
        K_list = list(range(1, int(args.k_max) + 1))

    r2_mean_list = []
    for k in K_list:
        # Fit a fresh model for each K so the curve is comparable across K.
        model_k = SpatialSynergy(k, method=args.method)
        _ = model_k.extract(train_data)
        act_te = model_k.encode(test_data)
        recon_te = model_k.decode(act_te)
        r2_mean, _, _ = r2_per_dof_mean_std(test_data, recon_te)  # ignore std
        r2_mean_list.append(r2_mean)

    # Paper-style plot: marker line, dotted reference at R^2 = 0.95,
    # integer K on the x-axis, fixed y range so different runs are comparable.
    plt.figure(figsize=(7, 4))
    plt.title(f"Reconstruction quality (R²) vs K — method={args.method}")
    plt.plot(K_list, r2_mean_list, marker="o", linewidth=1)
    plt.axhline(0.95, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("K (number of synergy components)")
    plt.ylabel("R² reconstruction on TEST trajectories")
    plt.xticks(K_list)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 5) Synergy weight matrix as a heatmap with numeric annotations.
    # -------------------------------------------------------------------------
    W = np.asarray(model.synergies)   # (K, M)

    plt.figure(figsize=(12, 3))
    im = plt.imshow(W, aspect="auto")
    plt.colorbar(im, label="Weight")

    plt.yticks(range(W.shape[0]), [f"S{k+1}" for k in range(W.shape[0])])
    plt.xticks(
        range(W.shape[1]),
        [str(i + 1) for i in range(W.shape[1])],
        rotation=90,
    )
    plt.xlabel("Joint (DoF)")
    plt.ylabel("Synergy")
    plt.title("Spatial synergies weight matrix")

    # Annotate every cell with its numeric weight (compact 2-decimal format).
    for i in range(W.shape[0]):      # synergies
        for j in range(W.shape[1]):  # DoF
            plt.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 6) Per-synergy bar plots (one figure per synergy, K figures total).
    # -------------------------------------------------------------------------
    W = np.asarray(model.synergies)   # (K, M)  -- re-bind for clarity
    K, M = W.shape

    joint_labels = [f"J{j+1}" for j in range(M)]

    for k in range(K):
        plt.figure(figsize=(10, 3))
        plt.bar(range(M), W[k], color="tab:purple")

        plt.axhline(0, linewidth=1, color="black")
        plt.xticks(range(M), joint_labels, rotation=90)
        plt.ylabel("Weight")
        plt.xlabel("Joint (DoF)")
        plt.title(f"Spatial Synergy {k+1}")

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 7) Paper-style joint x synergy contribution grid (single trajectory).
    # -------------------------------------------------------------------------
    plot_joint_synergy_contributions_grid(
        model,
        activities,
        traj_idx=int(args.traj_idx),
        joints=list(range(min(20, W.shape[1]))),
        synergies=list(range(min(K, 5))),
        y_mode="per_col",
        save_path="joint_synergy_grid.png",
    )


if __name__ == "__main__":
    main()