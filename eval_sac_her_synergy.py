#!/usr/bin/env python3
"""
Evaluation companion to ``train_sac_her_synergy.py``.

This script loads a SAC + HER policy that was trained in a low-dimensional
synergy action space (see ``HandSynergyEnv``) together with the ``VecNormalize``
statistics produced during training, and runs a deterministic rollout loop to
report success rate, episode return and goal-reaching statistics.

Pipeline at a glance
--------------------
    raw obs  -->  SAC policy  -->  a_k in R^K       (K = number of synergies)
                                       |
                                       v
                       synergy_model.decode(a_k)
                                       |
                                       v
                                 a_m in R^M         (M = native joint dim)
                                       |
                                       v
                                env.step(a_m)

Assumptions
-----------
The training script was launched with ``--save-dir <DIR>`` and produced::

    <DIR>/sac_her_hand_synergy.zip   final SAC policy
    <DIR>/vecnorm_synergy.pkl        VecNormalize observation statistics

Both files are required: ``vecnorm_synergy.pkl`` is mandatory for matching the
observation distribution the policy was trained on. If the file is missing the
script falls back to the un-normalised env and prints a warning, but that path
is intended for sanity checks only.

Custom environments
-------------------
In addition to the standard Gymnasium-Robotics suite, this script registers
``HandManipulateGlassRotateXYZ_ContinuousTouchSensors-v1`` so that policies
trained on the block can be cross-evaluated on the glass object.

Example
-------
    python hand_synergy_eval.py \\
        --env-id HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1 \\
        --synergy-path synergies/pca_K6.pkl \\
        --save-dir runs/synergy_K6_seed0 \\
        --n-episodes 100
"""

import argparse
import os
import pickle

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (kept for side-effect parity with train script)
import numpy as np  # noqa: F401  (kept for parity / downstream use)

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Reuse the EXACT wrapper + registration helper from the training script so
# observation / action spaces are guaranteed to match what the policy expects.
from synergies.train_sac_her_synergy import HandSynergyEnv, register_robotics_envs



# =============================================================================
# Utilities
# =============================================================================

def load_synergy_bundle(path: str):
    """
    Load a pickled synergy model from disk.

    The training pipeline saves either a bare ``SpatialSynergy`` instance or a
    dictionary with the model under the key ``"synergy_model"``. Both layouts
    are supported here so the evaluation script stays compatible with older
    bundles.

    NOTE
    ----
    This function is duplicated verbatim from ``train_sac_her_synergy.py`` so
    that the evaluation script does not depend on the training module's
    private helpers; behaviour must stay identical across the two files.

    Parameters
    ----------
    path : str
        Path to the ``.pkl`` file produced by the offline synergy extractor.

    Returns
    -------
    object
        A synergy model exposing ``n_synergies``, ``encode`` and ``decode``.
    """
    with open(path, "rb") as f:
        packed = pickle.load(f)

    if isinstance(packed, dict):
        if "synergy_model" in packed:
            return packed["synergy_model"]
        raise KeyError(
            f"Expected dict with 'synergy_model'. Got keys: {list(packed.keys())}"
        )

    return packed


# =============================================================================
# Environment builder
# =============================================================================

def make_synergy_eval_env(
    env_id,
    actions_synergy_model,
    act_scale=1.0,
    nonnegative_activities=False,
    render=False,
):
    """
    Build a thunk that constructs a Monitor-wrapped ``HandSynergyEnv`` instance.

    The factory pattern (returning a callable) mirrors the training script and
    is the form expected by ``DummyVecEnv``.

    Parameters
    ----------
    env_id : str
        Gymnasium environment id of the *base* dexterous-hand task.
    actions_synergy_model : object
        Synergy model used to map K-dimensional activities into M-dimensional
        joint commands (must be the same one used at training time).
    act_scale : float
        Symmetric bound for the K-dim action box (or upper bound when
        ``nonnegative_activities`` is True).
    nonnegative_activities : bool
        If True, restrict the action space to ``[0, act_scale]`` per dimension.
    render : bool
        If True, request a ``"human"`` render mode from the underlying env.

    Returns
    -------
    Callable[[], Monitor]
        A zero-argument factory ready to be passed to ``DummyVecEnv``.
    """
    def _thunk():
        base_render = "human" if render else None
        env = HandSynergyEnv(
            base_env_id=env_id,
            synergy_model=actions_synergy_model,
            act_scale=act_scale,
            nonnegative_activities=nonnegative_activities,
            render_mode=base_render,
        )
        return Monitor(env)
    return _thunk

# =============================================================
# Final evaluation loop
# =============================================================

def final_eval_loop(model, eval_env, n_episodes=10, deterministic=True):
    successes, ep_returns, ep_lengths, op_times = [], [], [], []
    dt= 0.04

    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        goal_time = None

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = eval_env.step(action)

            ep_ret += float(rewards[0])
            ep_len += 1

            info = infos[0]
            if goal_time is None and info.get("is_success", 0) == 1:
                goal_time = ep_len * dt

            done = bool(dones[0])

        successes.append(int(goal_time is not None))
        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)
        if goal_time is not None:
            op_times.append(goal_time)

    success_rate = float(np.mean(successes)) if successes else 0.0
    mean_op_time = float(np.mean(op_times)) if op_times else float("nan")

    n_success = int(np.sum(successes))
    print(f"Success rate: {success_rate:.3f}")
    print(f"Successes: {n_success}/{len(successes)}")
    print(f"Return mean: {np.mean(ep_returns):.2f}")
    print(f"Episode len mean: {np.mean(ep_lengths):.1f}")
    print(f"Operation time mean (successful eps only): {mean_op_time:.3f}")

# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Parse CLI arguments, load model + VecNormalize, run the eval loop."""
    parser = argparse.ArgumentParser(
        description="Evaluate SAC+HER on SpatialSynergy action space (step-by-step)"
    )

    # --- Environment ---------------------------------------------------------
    # Must match the env-id used during training; otherwise the loaded
    # observation normalization statistics will not align with the real env.
    parser.add_argument(
        "--env-id",
        type=str,
        default="HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1",
        help="Gymnasium-Robotics environment id used during training.",
    )

    # Synergy model used for training (same .pkl passed to train script).
    parser.add_argument(
        "--synergy-path",
        type=str,
        required=True,
        help="Path to the pickled synergy model (.pkl) used at training time.",
    )

    # Directory where the training script saved model + vecnorm.
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory containing sac_her_hand_synergy.zip and vecnorm_synergy.pkl.",
    )

    # --- Evaluation schedule ------------------------------------------------
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes to roll out.",
    )

    # --- Action-space configuration (must match training) ------------------
    parser.add_argument(
        "--act-scale",
        type=float,
        default=1.0,
        help="Symmetric (or upper) bound for the K-dim synergy action box.",
    )
    parser.add_argument(
        "--nonnegative-activities",
        action="store_true",
        help="Restrict the synergy action space to non-negative activities.",
    )

    # --- Rendering ----------------------------------------------------------
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable human rendering during evaluation.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1) Register Gymnasium-Robotics envs (parity with the training script)
    # -------------------------------------------------------------------------
    register_robotics_envs()

    # Register the Glass-rotation variant manually: it lives in
    # gymnasium_robotics.envs.shadow_dexterous_hand.manipulate_glass_touch_sensors
    # and is not part of the default registration batch.
    gym.register(
        id="HandManipulateGlassRotateXYZ_ContinuousTouchSensors-v1",
        entry_point=(
            "gymnasium_robotics.envs.shadow_dexterous_hand."
            "manipulate_glass_touch_sensors:MujocoHandGlassTouchSensorsEnv"
        ),
        max_episode_steps=100,
    )

    # -------------------------------------------------------------------------
    # 2) Load the synergy model
    # -------------------------------------------------------------------------
    actions_synergy_model = load_synergy_bundle(args.synergy_path)
    print(
        f"Loaded synergy model: K={actions_synergy_model.n_synergies}, "
        f"method={getattr(actions_synergy_model, 'method', 'NA')}"
    )

    # -------------------------------------------------------------------------
    # 3) Build the (raw) vectorised eval env -- a single-process DummyVecEnv is
    #    sufficient because we only run rollouts here.
    # -------------------------------------------------------------------------
    raw_eval_env = DummyVecEnv([
        make_synergy_eval_env(
            env_id=args.env_id,
            actions_synergy_model=actions_synergy_model,
            act_scale=args.act_scale,
            nonnegative_activities=args.nonnegative_activities,
            render=args.render,
        )
    ])

    # -------------------------------------------------------------------------
    # 4) Wrap with the SAME VecNormalize used at train time (or warn loudly).
    # -------------------------------------------------------------------------
    model_path = os.path.join(args.save_dir, "sac_her_hand_synergy")
    vecnorm_path = os.path.join(args.save_dir, "vecnorm_synergy.pkl")

    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, raw_eval_env)
        # Freeze running statistics: do not update obs_rms during evaluation,
        # and do not normalize reward (we want raw rewards in the metrics).
        eval_env.training = False
        eval_env.norm_reward = False
        print(f"> Loaded VecNormalize from: {vecnorm_path}")
    else:
        eval_env = raw_eval_env
        print("> WARNING: VecNormalize not found. Using raw env (no obs normalization).")

    # -------------------------------------------------------------------------
    # 5) Load the trained SAC policy and run the shared rollout loop.
    # -------------------------------------------------------------------------
    print(f"> Loading model: {model_path}")
    model = SAC.load(model_path, env=eval_env)

    # ``final_eval_loop`` handles success-rate / return aggregation and pretty
    # printing (defined in SHER_trial so it can be reused by other agents).
    final_eval_loop(model, eval_env, n_episodes=args.n_episodes)
    eval_env.close()


if __name__ == "__main__":
    main()