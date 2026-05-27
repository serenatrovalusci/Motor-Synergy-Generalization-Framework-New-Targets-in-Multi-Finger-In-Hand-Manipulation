#!/usr/bin/env python3
"""
Random low-dimensional projection as an action-space prior (control condition).

This module provides :class:`RandomProjection`, a drop-in replacement for
:class:`SpatialSynergy` that exposes the *same* ``n_synergies / encode / decode``
interface but uses a **random orthonormal K x M basis** instead of one fitted
to expert rollouts.

Why this exists
---------------
The PCA-derived synergy basis bundles three things into a single experimental
condition:

    1. *Dimensionality reduction* (M-dim joint space -> K-dim action space).
    2. *Task-relevant linear structure* (W learned from successful rollouts).
    3. *Same-object-family structure* (W learned specifically on the block).

A no-PCA baseline (full M-dim action space) tests the union of (1)+(2)+(3)
against (none). To attribute the speed-up to (2)+(3) -- the part that is
actually a "synergy" claim -- we need a control that keeps (1) and removes
(2)+(3). That is exactly this module: same K, same action-box geometry, but
the K directions in joint space are sampled uniformly at random from the
Stiefel manifold (orthonormal frames in R^M).

Pipeline parity
---------------
The pickled bundle produced by :func:`build_bundle` has the exact same layout
as ``synergies/pca_K*.pkl``::

    {
        "synergy_model": RandomProjection(...),   # has encode / decode / n_synergies
        "mu":            None,                    # no per-DOF normalisation
        "sigma":         None,
        "standardized":  False,
    }

so it can be passed to ``train_sac_her_synergy.py --synergy-path`` and to
``hand_synergy_eval.py --synergy-path`` without any modification to those
scripts.

Example
-------
    # Build a random K=5 bundle for the 20-DOF Shadow Hand:
    python synergy_random.py --m 20 --k 5 --seed 0 \\
        --out synergies/random_K5_seed0.pkl

    # Train SAC+HER on the Egg using the random prior (otherwise unchanged):
    python -m synergies.train_sac_her_synergy \\
        --env-id HandManipulateEggRotate_ContinuousTouchSensors-v1 \\
        --synergy-path synergies/random_K5_seed0.pkl \\
        --save-dir runs/random_K5_egg_seed0 \\
        ...
"""

import argparse
import os
import pickle

import numpy as np


# =============================================================================
# Random orthonormal projection
# =============================================================================

class RandomProjection:
    """
    Frozen random orthonormal K x M projection with the SpatialSynergy API.

    The basis ``W`` is a K x M matrix whose rows are orthonormal in R^M, drawn
    uniformly from the Stiefel manifold via a QR decomposition of a Gaussian
    matrix. Because the rows are orthonormal, the pseudoinverse is simply the
    transpose, so::

        encode(X) = X @ W.T          # (N, T, M) -> (N, T, K)
        decode(A) = A @ W            # (N, T, K) -> (N, T, M)

    and a unit step in any activity dimension corresponds to a unit-norm step
    in joint space along the matching basis direction -- the same geometric
    property the PCA basis has, so the [-act_scale, +act_scale]^K action box
    means the same thing across the two conditions.

    Attributes
    ----------
    n_synergies : int
        Number of synergies K (the policy's action-space dimensionality).
    method : str
        ``"random"`` -- used by ``hand_synergy_eval.py`` for printing.
    dof : int
        Native DOF count M of the underlying joint space.
    synergies : np.ndarray
        The K x M basis (rows are unit-norm and mutually orthogonal). Stored
        under this name (rather than ``W``) so that diagnostic plots written
        for SpatialSynergy -- e.g. the heatmap in
        ``synergy_extract_analyze.py`` -- still work without modification.
    """

    def __init__(self, n_synergies, dof, seed=0):
        """
        Parameters
        ----------
        n_synergies : int
            Number of synergies K. Must satisfy ``K <= dof``.
        dof : int
            Native joint dimensionality M (20 for the Shadow Hand).
        seed : int
            RNG seed for reproducibility. Different seeds give different
            random bases; report multi-seed results to avoid attributing a
            speed-up to one lucky draw.
        """
        if n_synergies > dof:
            raise ValueError(
                f"n_synergies (K={n_synergies}) must be <= dof (M={dof})."
            )

        self.n_synergies = n_synergies
        self.method = "random"
        self.dof = dof
        self.seed = seed

        # ---------------------------------------------------------------------
        # Sample a uniform-on-the-Stiefel-manifold K x M orthonormal frame.
        # Construction: take a Gaussian M x M matrix, run QR, keep the first K
        # columns of Q (which are orthonormal), and transpose so that rows of
        # the result are the basis vectors -- matching SpatialSynergy.synergies.
        # ---------------------------------------------------------------------
        rng = np.random.default_rng(seed)
        gaussian = rng.standard_normal((dof, dof))
        q, _ = np.linalg.qr(gaussian)              # (M, M), columns orthonormal
        self.synergies = q[:, :n_synergies].T      # (K, M), rows orthonormal

    # -------------------------------------------------------------------------
    # API parity with SpatialSynergy
    # -------------------------------------------------------------------------

    def encode(self, data):
        """
        Project trajectories into the K-dim activity space.

        Parameters
        ----------
        data : np.ndarray
            Trajectories of shape ``(N, T, M)`` -- or any leading batch shape;
            the projection only touches the last axis.

        Returns
        -------
        np.ndarray
            Activities of shape ``(..., K)``: ``data @ W.T``.
        """
        return data @ self.synergies.T

    def decode(self, activities):
        """
        Reconstruct trajectories from K-dim activities.

        Parameters
        ----------
        activities : np.ndarray
            Activities of shape ``(..., K)`` -- typically ``(N, T, K)`` for
            offline replay or ``(1, 1, K)`` for a single env step.

        Returns
        -------
        np.ndarray
            Reconstructed signal of shape ``(..., M)``: ``activities @ W``.
        """
        return activities @ self.synergies

    def extract(self, data, max_iter=None):
        """
        No-op stub kept for interface symmetry with SpatialSynergy.

        The basis is fixed at construction time -- there is nothing to fit on
        ``data``. Provided so that callers that loop over synergy models and
        call ``extract`` (e.g. some unit tests) don't crash.

        Parameters
        ----------
        data : np.ndarray
            Ignored.
        max_iter : Any
            Ignored.

        Returns
        -------
        np.ndarray
            ``self.synergies`` unchanged.
        """
        del data, max_iter
        return self.synergies


# =============================================================================
# Bundle builder
# =============================================================================

def build_bundle(n_synergies, dof, seed=0):
    """
    Build a pickle-ready bundle dict matching the PCA bundle layout.

    Parameters
    ----------
    n_synergies : int
        K -- size of the random subspace.
    dof : int
        M -- native joint-space dimensionality.
    seed : int
        RNG seed.

    Returns
    -------
    dict
        Same keys as the PCA bundles produced by ``synergy_extract_analyze.py``:
        ``synergy_model``, ``mu``, ``sigma``, ``standardized``. The two
        normalisation fields are ``None`` / ``False`` because a random basis
        does not see expert data and therefore has no per-DOF statistics.
    """
    model = RandomProjection(n_synergies=n_synergies, dof=dof, seed=seed)
    return {
        "synergy_model": model,
        "mu": None,
        "sigma": None,
        "standardized": False,
    }


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    """Build a random-projection bundle and pickle it to disk."""
    parser = argparse.ArgumentParser(
        description="Build a random orthonormal K x M projection bundle "
                    "(control for the synergy ablation)."
    )
    parser.add_argument(
        "--m", type=int, default=20,
        help="Native joint-space dimensionality M (default: 20 for Shadow Hand).",
    )
    parser.add_argument(
        "--k", type=int, required=True,
        help="Number of synergies K (action-space dimensionality).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed -- different seeds give different random bases.",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output .pkl path for the bundle.",
    )

    args = parser.parse_args()

    bundle = build_bundle(n_synergies=args.k, dof=args.m, seed=args.seed)

    # Make sure the parent directory exists, then pickle.
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)

    model = bundle["synergy_model"]
    print(f"> Wrote random bundle: {args.out}")
    print(f"  K = {model.n_synergies}, M = {model.dof}, seed = {model.seed}")
    print(f"  basis shape = {model.synergies.shape}, "
          f"row-norms ~ {np.linalg.norm(model.synergies, axis=1).mean():.4f}, "
          f"max |W W^T - I| = "
          f"{np.max(np.abs(model.synergies @ model.synergies.T - np.eye(model.n_synergies))):.2e}")


if __name__ == "__main__":
    main()