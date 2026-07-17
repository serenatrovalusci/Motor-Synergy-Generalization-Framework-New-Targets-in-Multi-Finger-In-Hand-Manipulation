#!/usr/bin/env python3
"""
Post-hoc checkpoint evaluation with the "any-time success" definition.

Why this exists
---------------
SB3's ``EvalCallback`` (the source of ``eval_logs/evaluations.npz`` and the
``eval/success_rate`` curve on wandb) counts an episode as successful only if
``is_success == 1`` at the episode's FINAL step. HandManipulate episodes run a
fixed number of steps and do not terminate on success, so the hand often
reaches the goal and then drifts off it before the end -- deflating the curve.

``sac_her_pipeline.py --task eval`` instead counts success if the goal was
reached at ANY step of the episode (see ``final_eval_loop``). This script
applies that same definition retroactively to every periodic checkpoint saved
in ``<run_dir>/ckpts/``, producing a success-vs-timesteps curve consistent
with the offline eval numbers.

Each run dir's ``args.json`` (written at training time) is used to rebuild the
exact same eval environment (env id, synergy bundle, act scale) -- no need to
re-specify them on the command line.

Outputs
-------
- Per run dir: ``eval_logs/evaluations_anytime.npz`` with keys ``timesteps``
  (n_ckpts,) and ``successes`` (n_ckpts, n_episodes) -- the same layout as
  SB3's evaluations.npz, so ``plot.py --npz-name evaluations_anytime.npz``
  can build the paper comparison figures from it directly.
- A quick mean±std preview curve across the given run dirs (``--plot-out``).

Example
-------
    python eval_checkpoints.py \\
        --run-dirs runs/big_block_synergy_K5_seed0 \\
                   runs/big_block_synergy_K5_seed1 \\
                   runs/big_block_synergy_K5_seed2 \\
        --n-episodes 50 \\
        --plot-out plots/big_block_synergy_anytime.png
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sac_her_pipeline import (
    load_synergy_bundle,
    make_full_env,
    make_synergy_env,
    register_robotics_envs,
)


CKPT_RE = re.compile(r"_(\d+)_steps\.zip$")


def discover_checkpoints(run_dir: str):
    """
    Return sorted [(timesteps, model_zip, vecnorm_pkl), ...] from run_dir/ckpts.

    CheckpointCallback (save_vecnormalize=True) writes pairs like:
        ckpts/sac_her_hand_500000_steps.zip
        ckpts/sac_her_hand_vecnormalize_500000_steps.pkl
    Checkpoints without a matching vecnormalize file are skipped with a
    warning: evaluating without the right obs normalisation gives garbage.
    """
    ckpt_dir = os.path.join(run_dir, "ckpts")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No ckpts/ directory under: {run_dir}")

    found = []
    for zip_path in glob.glob(os.path.join(ckpt_dir, "*_steps.zip")):
        m = CKPT_RE.search(os.path.basename(zip_path))
        if not m:
            continue
        steps = int(m.group(1))
        vecnorm_path = zip_path[: -len(m.group(0))] + f"_vecnormalize_{steps}_steps.pkl"
        if not os.path.exists(vecnorm_path):
            print(f"  WARNING: no vecnormalize for {os.path.basename(zip_path)} -- skipped")
            continue
        found.append((steps, zip_path, vecnorm_path))

    if not found:
        raise FileNotFoundError(f"No usable checkpoint pairs found in: {ckpt_dir}")
    return sorted(found)


def load_run_config(run_dir: str) -> dict:
    """Load the training-time CLI config saved by sac_her_pipeline.py."""
    path = os.path.join(run_dir, "args.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path} -- cannot reconstruct the eval env. "
            "(It is written automatically by --task train.)"
        )
    with open(path) as f:
        return json.load(f)


def build_raw_env(cfg: dict):
    """Rebuild the same (unnormalised) eval env the run was trained with."""
    if cfg.get("full_action_space"):
        return DummyVecEnv([make_full_env(cfg["env_id"], render=False)])
    synergy_model = load_synergy_bundle(cfg["synergy_path"])
    return DummyVecEnv([make_synergy_env(
        cfg["env_id"], synergy_model,
        act_scale=cfg.get("act_scale", 3.0),
        nonnegative_activities=cfg.get("nonnegative_activities", False),
        render=False,
    )])


def eval_anytime(model, eval_env, n_episodes: int, seed: int | None):
    """
    Deterministic rollouts; an episode succeeds if is_success==1 at ANY step
    (same definition as final_eval_loop in sac_her_pipeline.py).

    When ``seed`` is given, episode i is seeded with seed+i, so every
    checkpoint sees the same goal sequence -- differences between checkpoints
    then reflect the policy, not goal-sampling luck.
    """
    successes = []
    for ep in range(n_episodes):
        if seed is not None:
            eval_env.seed(seed + ep)
        obs = eval_env.reset()
        done, hit = False, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _rewards, dones, infos = eval_env.step(action)
            if not hit and infos[0].get("is_success", 0) == 1:
                hit = True
            done = bool(dones[0])
        successes.append(int(hit))
    return successes


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved checkpoints with the any-time success definition."
    )
    parser.add_argument("--run-dirs", nargs="+", required=True, metavar="DIR",
                        help="Run directories (e.g. the 3 seeds of one config).")
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Eval episodes per checkpoint (default: 50).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for paired goal sequences across checkpoints. "
                             "Pass -1 to disable seeding.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--plot-out", type=str, default="anytime_success_curve.png",
                        help="Where to save the mean±std preview curve.")
    parser.add_argument("--label", type=str, default=None,
                        help="Curve label in the preview plot (default: first run dir name).")
    args = parser.parse_args()

    register_robotics_envs()
    seed = None if args.seed == -1 else int(args.seed)

    all_ts, all_rates = [], []

    for run_dir in args.run_dirs:
        print(f"\n=== {run_dir} ===")
        cfg = load_run_config(run_dir)
        ckpts = discover_checkpoints(run_dir)
        print(f"  Found {len(ckpts)} checkpoints "
              f"({ckpts[0][0]:,} .. {ckpts[-1][0]:,} steps)")

        raw_env = build_raw_env(cfg)

        timesteps, succ_matrix = [], []
        for steps, zip_path, vecnorm_path in ckpts:
            eval_env = VecNormalize.load(vecnorm_path, raw_env)
            eval_env.training = False
            eval_env.norm_reward = False

            # custom_objects replaces the pickled gym spaces with the current
            # env's ones instead of deserialising them -- the SB3-recommended
            # way to keep checkpoints loadable across gym/numpy version drift.
            # buffer_size=1: predict-only, skip allocating the ~3GB HER buffer
            # SAC.load would otherwise recreate for every checkpoint.
            model = SAC.load(
                zip_path, env=eval_env, device=args.device,
                custom_objects={
                    "observation_space": eval_env.observation_space,
                    "action_space": eval_env.action_space,
                    "buffer_size": 1,
                },
            )
            successes = eval_anytime(model, eval_env, args.n_episodes, seed)
            rate = float(np.mean(successes))
            print(f"  {steps:>12,} steps  ->  success {rate:.2%} "
                  f"({int(np.sum(successes))}/{args.n_episodes})")

            timesteps.append(steps)
            succ_matrix.append(successes)

        raw_env.close()

        timesteps = np.asarray(timesteps, dtype=np.int64)
        succ_matrix = np.asarray(succ_matrix, dtype=np.int8)

        out_dir = os.path.join(run_dir, "eval_logs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "evaluations_anytime.npz")
        np.savez_compressed(out_path, timesteps=timesteps, successes=succ_matrix)
        print(f"  [OK] Saved: {out_path}")

        all_ts.append(timesteps.astype(np.float64))
        all_rates.append(succ_matrix.mean(axis=1))

    # ── Preview curve: mean ± std across run dirs on a common grid ──────────
    min_len = min(len(ts) for ts in all_ts)
    common_ts = all_ts[int(np.argmin([len(t) for t in all_ts]))][:min_len]
    stacked = np.stack([
        np.interp(common_ts, ts, vals)
        for ts, vals in zip(all_ts, all_rates)
    ], axis=0)

    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    label = args.label or os.path.basename(os.path.normpath(args.run_dirs[0]))

    plt.figure(figsize=(7, 4.5))
    plt.plot(common_ts, mean, marker="o", linewidth=2, label=label)
    plt.fill_between(common_ts, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel(f"Success rate, any-time  (mean ± std, {args.n_episodes} ep)")
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.gca().xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    plt.grid(True, alpha=0.3)
    plt.legend(framealpha=0.3, fontsize=9)
    plt.tight_layout()

    out_plot_dir = os.path.dirname(args.plot_out)
    if out_plot_dir:
        os.makedirs(out_plot_dir, exist_ok=True)
    plt.savefig(args.plot_out, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Saved preview curve: {args.plot_out}")


if __name__ == "__main__":
    main()
