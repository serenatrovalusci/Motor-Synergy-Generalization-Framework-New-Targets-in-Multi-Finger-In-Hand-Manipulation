#!/usr/bin/env python3
"""
Train Soft Actor-Critic with Hindsight Experience Replay (SAC + HER) in a
low-dimensional synergy action space, on a Gymnasium-Robotics dexterous-hand
environment.

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

The synergy model is loaded from a pickle produced by an offline PCA / NMF
extraction step. SAC therefore searches a K-dimensional action space (typical
K = 6..10), while the underlying simulator still receives full M-dimensional
joint commands (M = 20 for the Shadow Dexterous Hand).

Output layout (under ``--save-dir``)
------------------------------------
    best/         best_model.zip + vecnorm_best.pkl   (kept in sync)
    ckpts/        periodic checkpoints
    eval_logs/    EvalCallback npz logs
    tb_logs/      TensorBoard scalars
    sac_her_hand_synergy.zip   final policy
    vecnorm_synergy.pkl        final VecNormalize stats

Example
-------
    python train_sac_her_synergy.py \\
        --env-id HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1 \\
        --synergy-path synergies/pca_K6.pkl \\
        --save-dir runs/synergy_K6_seed0 \\
        --timesteps 2000000

Notes
-----
- Training uses a single DummyVecEnv (n_envs=1) with gradient_steps=4.
  SubprocVecEnv IPC overhead dominates for fast low-dim MuJoCo envs, so
  parallelising collection does not help. Instead we extract more learning
  signal per transition by doing 4 gradient updates per env step.
- ``batch_size`` is 256 for a more stable gradient estimate.
- ``learning_starts`` is 10_000 
  so HER has enough diverse trajectories before the first update.
- ``gamma`` defaults to 0.99 (Plappert et al. multigoal default).
- ``target_entropy = -0.8 * action_dim`` slightly below the SB3 default of
  ``-K`` to bias the policy toward marginally more deterministic behaviour.
- HER uses the "future" strategy with ``n_sampled_goal = 8``.
- obs_rms is deep-copied (not shared by reference) between train and eval
  envs to prevent eval rollouts from mutating training statistics.
"""

import argparse
import copy
import json
import os
import pickle
import time

import numpy as np
import torch

import gymnasium as gym
import gymnasium_robotics
from gymnasium import spaces

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize



# =============================================================================
# Callbacks
# =============================================================================

class TimeLoggingCallback(BaseCallback):
    """
    Log wall-clock training time and throughput to TensorBoard.

    Emits every ``log_every_steps`` env steps:
        - ``time/elapsed_sec``
        - ``time/fps``
        - ``time/sec_per_100k_steps``
    """

    def __init__(self, log_every_steps: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every_steps = int(log_every_steps)
        self._t0 = None
        self._last_log_t = None
        self._last_log_steps = 0

    def _on_training_start(self) -> None:
        now = time.time()
        self._t0 = now
        self._last_log_t = now
        self._last_log_steps = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log_steps < self.log_every_steps:
            return True

        now = time.time()
        elapsed = now - self._t0
        dt = now - self._last_log_t
        dsteps = self.num_timesteps - self._last_log_steps

        fps = (dsteps / dt) if dt > 0 else 0.0
        sec_per_100k = (100_000.0 / fps) if fps > 0 else float("inf")

        self.logger.record("time/elapsed_sec", float(elapsed))
        self.logger.record("time/fps", float(fps))
        self.logger.record("time/sec_per_100k_steps", float(sec_per_100k))
        self.logger.dump(self.num_timesteps)

        self._last_log_t = now
        self._last_log_steps = self.num_timesteps
        return True


class SaveVecNormalizeOnBest(BaseCallback):
    """
    Two responsibilities:

    1. Every step: deep-copy train_env.obs_rms -> eval_env.obs_rms so that
       evaluation rollouts always use up-to-date normalisation statistics.
       A deep copy (instead of a shared reference) prevents eval rollouts
       from mutating the running statistics that the training policy depends
       on. This fixes the silent performance corruption that occurs with the
       naive ``eval_env.obs_rms = train_env.obs_rms`` assignment.

    2. On new best model: save vecnorm_best.pkl next to best_model.zip so
       the two artefacts stay in sync for deployment. Without the matching
       VecNormalize stats, a loaded policy would receive un-normalised
       observations and its performance would silently collapse.
    """

    def __init__(
        self,
        best_dir: str,
        train_env: VecNormalize,
        eval_env: VecNormalize,
        filename: str = "vecnorm_best.pkl",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.best_dir = best_dir
        self.train_env = train_env
        self.eval_env = eval_env
        self.filename = filename
        self._last_mtime = None

    def _on_step(self) -> bool:
        # --- 1. Sync eval obs_rms with training env -------------------------
        self.eval_env.obs_rms = copy.deepcopy(self.train_env.obs_rms)

        # --- 2. Save vecnorm whenever a new best_model.zip appears ----------
        best_model_path = os.path.join(self.best_dir, "best_model.zip")
        if not os.path.exists(best_model_path):
            return True

        mtime = os.path.getmtime(best_model_path)
        if self._last_mtime is None or mtime != self._last_mtime:
            vec_path = os.path.join(self.best_dir, self.filename)
            try:
                self.train_env.save(vec_path)
                if self.verbose:
                    print(f"[SaveVecNormalizeOnBest] Saved VecNormalize to: {vec_path}")
            except Exception as e:
                print(f"[SaveVecNormalizeOnBest] WARNING: could not save VecNormalize: {e}")
            self._last_mtime = mtime

        return True


# =============================================================================
# Environment wrapper: K-dim synergy activities -> M-dim joint command
# =============================================================================

class HandSynergyEnv(gym.Env):
    """
    Wrap a Gymnasium-Robotics hand environment so that the agent acts in
    K-dimensional synergy space rather than the native M-dimensional joint
    space.

    Parameters
    ----------
    base_env_id : str
        Gymnasium id of the underlying robotics env.
    synergy_model : object
        Offline-fitted model with ``n_synergies`` (int) and
        ``decode(activities)`` (shape T,B,K -> T,B,M).
    act_scale : float
        Half-range of the synergy-activity Box action space.
    nonnegative_activities : bool
        If True, activities in [0, act_scale] (NMF).
        Otherwise [-act_scale, act_scale] (PCA).
    render_mode : str or None
        Forwarded to the base env.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        base_env_id,
        synergy_model,
        act_scale=3.0,
        nonnegative_activities=False,
        render_mode=None,
    ):
        super().__init__()
        self.base_env = gym.make(base_env_id, render_mode=render_mode)
        self.synergy_model = synergy_model

        self.K = int(synergy_model.n_synergies)
        self.M = int(synergy_model.dof)

        self.observation_space = self.base_env.observation_space

        if nonnegative_activities:
            low = np.zeros(self.K, dtype=np.float32)
            high = act_scale * np.ones(self.K, dtype=np.float32)
        else:
            low = -act_scale * np.ones(self.K, dtype=np.float32)
            high = act_scale * np.ones(self.K, dtype=np.float32)

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        base_act_dim = self.base_env.action_space.shape[0]    #make sure that the synergy model's dof matches the base env's action dim
        assert base_act_dim == self.M, (
            f"Synergy model dof={self.M} != base env action_dim={base_act_dim}"
        )

    def reset(self, *, seed=None, options=None):
        return self.base_env.reset(seed=seed, options=options)

    def step(self, a_k):
        a_k = np.asarray(a_k, dtype=np.float32)
        activities = a_k.reshape(1, 1, self.K)
        actions_decoded = self.synergy_model.decode(activities)
        a_m_raw = actions_decoded[0, 0, :].astype(np.float32)

        low = self.base_env.action_space.low
        high = self.base_env.action_space.high
        a_m = np.clip(a_m_raw, low, high).astype(np.float32)

        return self.base_env.step(a_m)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.base_env.unwrapped.compute_reward(
            achieved_goal, desired_goal, info
        )

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()


# =============================================================================
# Env builders
# =============================================================================

def make_synergy_env(
    env_id,
    actions_synergy_model,
    act_scale=3.0,
    nonnegative_activities=False,
    render=False,
    seed=None,
):
    """Return a thunk that builds a Monitor-wrapped HandSynergyEnv."""

    def _thunk():
        env = HandSynergyEnv(
            base_env_id=env_id,
            synergy_model=actions_synergy_model,
            act_scale=act_scale,
            nonnegative_activities=nonnegative_activities,
            render_mode="human" if render else None,
        )
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return Monitor(env)

    return _thunk


def build_train_eval_envs(
    env_id,
    actions_synergy_model,
    act_scale=3.0,
    nonnegative_activities=False,
    seed=0,
):
    """
    Build (train_env, eval_env) as single DummyVecEnv + VecNormalize pairs.

    No SubprocVecEnv: for fast low-dim MuJoCo envs the IPC overhead of
    subprocess communication outweighs the parallelism benefit.

    obs_rms starts as a deep copy so the two envs are statistically
    independent. SaveVecNormalizeOnBest propagates updates from train to
    eval at every training step.
    """
    train_env = DummyVecEnv([
        make_synergy_env(
            env_id, actions_synergy_model,
            act_scale=act_scale,
            nonnegative_activities=nonnegative_activities,
            render=False, seed=seed,
        )
    ])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )

    eval_env = DummyVecEnv([
        make_synergy_env(
            env_id, actions_synergy_model,
            act_scale=act_scale,
            nonnegative_activities=nonnegative_activities,
            render=False, seed=seed + 10_000,
        )
    ])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    eval_env.training = False
    eval_env.norm_reward = False

    # Deep copy: eval never mutates train's running statistics.
    eval_env.obs_rms = copy.deepcopy(train_env.obs_rms)

    return train_env, eval_env

# ---------------------------------------------------------------------------
# env-builder for the full action space case
# ---------------------------------------------------------------------------

def make_full_env(env_id, render=False, seed=None):
    """Return a thunk that builds a Monitor-wrapped base hand env."""
    def _thunk():
        env = gym.make(env_id, render_mode="human" if render else None)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return Monitor(env)
    return _thunk


def build_train_eval_envs_full(env_id, seed=0):
    """
    Same structure as build_train_eval_envs but without the synergy wrapper.
    The agent acts in the native M-dimensional joint space.
    """
    train_env = DummyVecEnv([make_full_env(env_id, render=False, seed=seed)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )

    eval_env = DummyVecEnv([make_full_env(env_id, render=False, seed=seed + 10_000)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.obs_rms = copy.deepcopy(train_env.obs_rms)

    return train_env, eval_env


# =============================================================================
# Utilities
# =============================================================================

def register_robotics_envs():
    """Register Gymnasium-Robotics env namespace if not already done."""
    try:
        gym.register_envs(gymnasium_robotics)
    except Exception:
        pass


def load_synergy_bundle(path: str):
    """
    Load a synergy model from a .pkl produced by the offline extraction
    pipeline. Accepts a bare model object or a dict with a
    ``"synergy_model"`` key.
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
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAC + HER on a PCA / NMF synergy action space"
    )

    # --- Environment / synergy ----------------------------------------------
    parser.add_argument(
        "--env-id", type=str,
        default="HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1",
    )
    # NEW:
    parser.add_argument(
        "--synergy-path", type=str, default=None,
        help="Path to synergy .pkl. Omit when --full-action-space is set.",
    )
    parser.add_argument(
        "--full-action-space", action="store_true",
        help="Train directly in the native joint space (no synergy model). "
            "Use this for the baseline comparison.",
    )
    parser.add_argument("--act-scale", type=float, default=0.5)
    parser.add_argument("--nonnegative-activities", action="store_true")

    # --- Bookkeeping --------------------------------------------------------
    parser.add_argument("--save-dir", type=str, default="synergy_sac_her_results")

    # --- Training schedule --------------------------------------------------
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=50)

    # --- SAC / HER hyperparameters ------------------------------------------
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--n-sampled-goal", type=int, default=8,
        help="HER hindsight goals per real transition ('future' strategy).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="SAC minibatch size. 512 gives a more stable gradient than 256 "
             "without overloading HER's CPU-side relabeling cost.",
    )
    parser.add_argument(
        "--gradient-steps", type=int, default=4,
        help="Gradient updates per collected transition. "
             "4 extracts more learning signal per env step without overfitting "
             "risk given the 1M replay buffer.",
    )
    parser.add_argument(
        "--learning-starts", type=int, default=10_000,
        help="Random-action warm-up before first gradient update. "
             "25k = ~250 complete episodes (100 steps/ep). "
             "Scaled up from 10k to give HER diverse relabeling material "
             "and to satisfy: learning_starts >= 50 * batch_size.",
    )

    # --- Logging ------------------------------------------------------------
    parser.add_argument("--time-log-every", type=int, default=10_000)

    # --- Hardware -----------------------------------------------------------
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device. 'cpu' is often fastest for small 256x256 MLPs. "
             "Consider 'cuda' only if batch_size >= 2048.",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=4,
        help="PyTorch intra-op threads. With a single DummyVecEnv there are "
             "no competing MuJoCo worker processes, so 4 threads is safe.",
    )

    # --- Reproducibility ----------------------------------------------------
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    torch.set_num_threads(int(args.torch_threads))
    set_random_seed(args.seed)

    print("=" * 70)
    print("RUN CONFIGURATION")
    print("=" * 70)
    print(f"  env_id          = {args.env_id}")
    print(f"  synergy_path    = {args.synergy_path}")
    print(f"  save_dir        = {args.save_dir}")
    print(f"  timesteps       = {args.timesteps:_}")
    print(f"  seed            = {args.seed}")
    print(f"  n_envs          = 1  (DummyVecEnv, no IPC overhead)")
    print(f"  gradient_steps  = {args.gradient_steps}")
    print(f"  batch_size      = {args.batch_size}")
    print(f"  learning_starts = {args.learning_starts}")
    print(f"  device          = {args.device}")
    print(f"  torch_threads   = {torch.get_num_threads()}")
    print("=" * 70)

    # --- Output directories -------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    best_dir = os.path.join(args.save_dir, "best")
    ckpt_dir = os.path.join(args.save_dir, "ckpts")
    eval_log_dir = os.path.join(args.save_dir, "eval_logs")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # --- Envs ---------------------------------------------------------------
    register_robotics_envs()

    if args.full_action_space:
        if args.synergy_path is not None:
            print("WARNING: --synergy-path is ignored when --full-action-space is set.")
        print("Mode: FULL action space (no synergy model)")
        train_env, eval_env = build_train_eval_envs_full(args.env_id, seed=args.seed)
    else:
        if args.synergy_path is None:
            raise ValueError("--synergy-path is required unless --full-action-space is set.")
        actions_synergy_model = load_synergy_bundle(args.synergy_path)
        print(
            f"Mode: SYNERGY  K={actions_synergy_model.n_synergies}, "
            f"method={getattr(actions_synergy_model, 'method', 'NA')}"
        )

        train_env, eval_env = build_train_eval_envs(
            args.env_id,
            actions_synergy_model,
            act_scale=args.act_scale,
            nonnegative_activities=args.nonnegative_activities,
            seed=args.seed,
        )
    # --- SAC + HER ----------------------------------------------------------
    action_dim = train_env.action_space.shape[-1]
    target_entropy = -float(action_dim) * 0.8

    model = SAC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=args.n_sampled_goal,
            goal_selection_strategy="future",
        ),
        buffer_size=int(1e6),
        batch_size=args.batch_size,           # 512
        learning_rate=args.lr,
        gamma=args.gamma,
        learning_starts=args.learning_starts, # 25_000
        ent_coef="auto",
        train_freq=1,
        gradient_steps=args.gradient_steps,   # 4
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
        target_entropy=target_entropy,
        verbose=1,
        device=args.device,
        seed=args.seed,
        tensorboard_log=os.path.join(args.save_dir, "tb_logs"),
    )

    # --- Callbacks ----------------------------------------------------------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=eval_log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="sac_her_hand_synergy",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Syncs eval obs_rms at every step (deep copy) and saves vecnorm_best.pkl
    # whenever EvalCallback writes a new best_model.zip.
    save_best_vec_cb = SaveVecNormalizeOnBest(
        best_dir=best_dir,
        train_env=train_env,
        eval_env=eval_env,
        filename="vecnorm_best.pkl",
        verbose=1,
    )

    time_cb = TimeLoggingCallback(log_every_steps=args.time_log_every)

    # --- Train --------------------------------------------------------------
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, ckpt_cb, save_best_vec_cb, time_cb],
    )

    # --- Save final artifacts -----------------------------------------------
    model_path = os.path.join(args.save_dir, "sac_her_hand_synergy")
    vecnorm_path = os.path.join(args.save_dir, "vecnorm_synergy.pkl")
    model.save(model_path)
    train_env.save(vecnorm_path)

    print(f"Saved model to:                 {model_path}")
    print(f"Saved VecNormalize stats to:    {vecnorm_path}")
    print(f"Best model path:                {os.path.join(best_dir, 'best_model.zip')}")
    print(f"Best VecNormalize path:         {os.path.join(best_dir, 'vecnorm_best.pkl')}")


if __name__ == "__main__":
    main()