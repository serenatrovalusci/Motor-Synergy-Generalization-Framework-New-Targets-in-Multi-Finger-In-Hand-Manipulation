#!/usr/bin/env python3
"""
SAC + HER training, evaluation, and trajectory collection.

Supports two action-space modes:
  - Synergy space (default): the agent acts in a K-dimensional latent space
    decoded into M-dimensional joint commands via a pre-fitted PCA/NMF model.
  - Full action space (--full-action-space): the agent acts directly in the
    native M-dimensional joint space (baseline).

Pipeline at a glance (synergy mode)
-------------------------------------
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

Output layout (under --save-dir)
---------------------------------
    best/           best_model.zip + vecnorm_best.pkl   (kept in sync)
    ckpts/          periodic checkpoints every 50k steps
    eval_logs/      EvalCallback npz logs
    tb_logs/        TensorBoard scalars
    args.json       full run configuration

Usage examples
--------------
  # Synergy training
  python train_sac_her_synergy.py \\
      --task train \\
      --env-id HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1 \\
      --synergy-path pca/block_K5 \\
      --save-dir runs/synergy_K5_egg \\
      --timesteps 2000000

  # Full action space (baseline)
  python train_sac_her_synergy.py \\
      --task train \\
      --full-action-space \\
      --env-id HandManipulateEggRotate_ContinuousTouchSensors-v1 \\
      --save-dir runs/full_action_egg \\
      --timesteps 6000000

  # Evaluation
  python train_sac_her_synergy.py \\
      --task eval \\
      --synergy-path pca/block_K5 \\
      --save-dir runs/synergy_K5_egg \\
      --eval-episodes 100

  # Trajectory collection (actions only)
  python train_sac_her_synergy.py \\
      --task collect \\
      --full-action-space \\
      --save-dir runs/full_action_block \\
      --traj-episodes 200 \\
      --traj-save-path trajectory.npz
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
        - time/elapsed_sec
        - time/fps
        - time/sec_per_100k_steps
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

    1. Every step: deep-copy train_env.obs_rms -> eval_env.obs_rms so
       evaluation rollouts always use up-to-date normalisation statistics
       without mutating the training running stats.

    2. On new best model: save vecnorm_best.pkl next to best_model.zip so
       the two artefacts stay in sync for deployment.
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
        self.eval_env.obs_rms = copy.deepcopy(self.train_env.obs_rms)

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
    K-dimensional synergy space rather than the native M-dimensional joint space.

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

        base_act_dim = self.base_env.action_space.shape[0]
        assert base_act_dim == self.M, (
            f"Synergy model dof={self.M} != base env action_dim={base_act_dim}"
        )

    def reset(self, *, seed=None, options=None):
        return self.base_env.reset(seed=seed, options=options)

    def step(self, a_k):
        a_k = np.asarray(a_k, dtype=np.float32)
        activities = a_k.reshape(1, 1, self.K)
        a_m_raw = self.synergy_model.decode(activities)[0, 0, :].astype(np.float32)
        a_m = np.clip(a_m_raw, self.base_env.action_space.low,
                      self.base_env.action_space.high).astype(np.float32)
        return self.base_env.step(a_m)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.base_env.unwrapped.compute_reward(achieved_goal, desired_goal, info)

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()


# =============================================================================
# Environment builders
# =============================================================================

def make_synergy_env(env_id, synergy_model, act_scale=3.0,
                     nonnegative_activities=False, render=False, seed=None):
    """Return a thunk that builds a Monitor-wrapped HandSynergyEnv."""
    def _thunk():
        env = HandSynergyEnv(
            base_env_id=env_id,
            synergy_model=synergy_model,
            act_scale=act_scale,
            nonnegative_activities=nonnegative_activities,
            render_mode="human" if render else None,
        )
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return Monitor(env)
    return _thunk


def make_full_env(env_id, render=False, seed=None):
    """Return a thunk that builds a Monitor-wrapped base hand env."""
    def _thunk():
        env = gym.make(env_id, render_mode="human" if render else None)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return Monitor(env)
    return _thunk


def build_train_eval_envs(env_id, synergy_model, act_scale=3.0,
                           nonnegative_activities=False, seed=0):
    """Build (train_env, eval_env) VecNormalize pairs for synergy mode."""
    train_env = DummyVecEnv([make_synergy_env(
        env_id, synergy_model, act_scale=act_scale,
        nonnegative_activities=nonnegative_activities, render=False, seed=seed,
    )])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = DummyVecEnv([make_synergy_env(
        env_id, synergy_model, act_scale=act_scale,
        nonnegative_activities=nonnegative_activities, render=False, seed=seed + 10_000,
    )])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.obs_rms = copy.deepcopy(train_env.obs_rms)
    return train_env, eval_env


def build_train_eval_envs_full(env_id, seed=0):
    """Build (train_env, eval_env) VecNormalize pairs for full action space mode."""
    train_env = DummyVecEnv([make_full_env(env_id, render=False, seed=seed)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = DummyVecEnv([make_full_env(env_id, render=False, seed=seed + 10_000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
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
    Load a synergy model from a .pkl produced by the offline extraction pipeline.
    Accepts a bare model object or a dict with a ``"synergy_model"`` key.
    """
    with open(path, "rb") as f:
        packed = pickle.load(f)
    if isinstance(packed, dict):
        if "synergy_model" in packed:
            return packed["synergy_model"]
        raise KeyError(f"Expected dict with 'synergy_model'. Got keys: {list(packed.keys())}")
    return packed


# =============================================================================
# Evaluation loop
# =============================================================================

def final_eval_loop(model, eval_env, n_episodes=100, deterministic=True):
    """Run deterministic rollouts and print success rate, return, and timing."""
    successes, ep_returns, ep_lengths, op_times = [], [], [], []
    dt = 0.04  # MuJoCo timestep

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
            if goal_time is None and infos[0].get("is_success", 0) == 1:
                goal_time = ep_len * dt
            done = bool(dones[0])

        successes.append(int(goal_time is not None))
        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)
        if goal_time is not None:
            op_times.append(goal_time)

    print(f"Success rate : {float(np.mean(successes)):.3f}")
    print(f"Successes    : {int(np.sum(successes))}/{n_episodes}")
    print(f"Return mean  : {np.mean(ep_returns):.2f}")
    print(f"Episode len  : {np.mean(ep_lengths):.1f}")
    mean_op = float(np.mean(op_times)) if op_times else float("nan")
    print(f"Op time mean : {mean_op:.3f}s  (successful episodes only)")


# =============================================================================
# Trajectory collection
# =============================================================================

def collect_trajectories(model, eval_env, n_episodes, save_path, deterministic=True):
    """
    Roll out the policy for ``n_episodes`` episodes and save joint actions.

    Output shape: (n_episodes, T, M) saved as ``trajectory.npz`` with key
    ``"actions"``. Only action trajectories are stored (no observations).
    """
    all_ep_actions = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            ep_actions.append(np.array(action[0], dtype=np.float32))
            obs, _, dones, _ = eval_env.step(action)
            done = bool(dones[0])

        all_ep_actions.append(np.stack(ep_actions, axis=0))
        print(f"[collect] Episode {ep + 1}/{n_episodes}  length={len(ep_actions)}")

    actions_arr = np.stack(all_ep_actions, axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez_compressed(save_path, actions=actions_arr)
    print(f"[collect] Saved actions {actions_arr.shape} -> {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAC + HER — synergy or full action space"
    )

    # --- Task ---------------------------------------------------------------
    parser.add_argument(
        "--task",
        choices=["train", "eval", "collect"],
        default="train",
        help="train: run SAC+HER training. "
             "eval: load best model and report metrics. "
             "collect: roll out best model and save joint-action trajectories.",
    )

    # --- Environment / action space -----------------------------------------
    parser.add_argument(
        "--env-id", type=str,
        default="HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1",
    )
    parser.add_argument(
        "--synergy-path", type=str, default=None,
        help="Path to synergy .pkl. Required unless --full-action-space is set.",
    )
    parser.add_argument(
        "--full-action-space", action="store_true",
        help="Train/evaluate in the native M-dimensional joint space (no synergy).",
    )
    parser.add_argument("--act-scale", type=float, default=0.5)
    parser.add_argument("--nonnegative-activities", action="store_true")

    # --- Bookkeeping --------------------------------------------------------
    parser.add_argument(
        "--save-dir", type=str, default="runs/default",
        help="Directory for models, logs, and checkpoints.",
    )

    # --- Training schedule --------------------------------------------------
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=50)

    # --- SAC / HER hyperparameters ------------------------------------------
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-sampled-goal", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--learning-starts", type=int, default=10_000)

    # --- Trajectory collection ----------------------------------------------
    parser.add_argument(
        "--traj-episodes", type=int, default=200,
        help="Number of episodes to record (--task collect).",
    )
    parser.add_argument(
        "--traj-save-path", type=str, default=None,
        help="Output path for trajectory .npz. "
             "Defaults to <save-dir>/trajectory.npz.",
    )

    # --- Logging / hardware -------------------------------------------------
    parser.add_argument("--time-log-every", type=int, default=10_000)
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    torch.set_num_threads(int(args.torch_threads))
    set_random_seed(args.seed)
    register_robotics_envs()

    # Validate synergy/full-action-space flags
    if not args.full_action_space and args.synergy_path is None:
        parser.error("--synergy-path is required unless --full-action-space is set.")
    if args.full_action_space and args.synergy_path is not None:
        print("WARNING: --synergy-path is ignored when --full-action-space is set.")

    # Load synergy model if needed
    synergy_model = None
    if not args.full_action_space:
        synergy_model = load_synergy_bundle(args.synergy_path)
        print(f"Synergy model: K={synergy_model.n_synergies}, "
              f"method={getattr(synergy_model, 'method', 'NA')}")

    # Output directories
    os.makedirs(args.save_dir, exist_ok=True)
    best_dir = os.path.join(args.save_dir, "best")
    ckpt_dir = os.path.join(args.save_dir, "ckpts")
    eval_log_dir = os.path.join(args.save_dir, "eval_logs")
    for d in [best_dir, ckpt_dir, eval_log_dir]:
        os.makedirs(d, exist_ok=True)

    # =========================================================================
    # TRAIN
    # =========================================================================
    if args.task == "train":
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)

        if args.full_action_space:
            print("Mode: FULL action space")
            train_env, eval_env = build_train_eval_envs_full(args.env_id, seed=args.seed)
        else:
            print(f"Mode: SYNERGY  K={synergy_model.n_synergies}")
            train_env, eval_env = build_train_eval_envs(
                args.env_id, synergy_model,
                act_scale=args.act_scale,
                nonnegative_activities=args.nonnegative_activities,
                seed=args.seed,
            )

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
            batch_size=args.batch_size,
            learning_rate=args.lr,
            gamma=args.gamma,
            learning_starts=args.learning_starts,
            ent_coef="auto",
            train_freq=1,
            gradient_steps=args.gradient_steps,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
            target_entropy=target_entropy,
            verbose=1,
            device=args.device,
            seed=args.seed,
            tensorboard_log=os.path.join(args.save_dir, "tb_logs"),
        )

        callbacks = [
            EvalCallback(
                eval_env,
                best_model_save_path=best_dir,
                log_path=eval_log_dir,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            ),
            CheckpointCallback(
                save_freq=50_000,
                save_path=ckpt_dir,
                name_prefix="sac_her_hand",
                save_replay_buffer=False,
                save_vecnormalize=True,
            ),
            SaveVecNormalizeOnBest(
                best_dir=best_dir,
                train_env=train_env,
                eval_env=eval_env,
                verbose=1,
            ),
            TimeLoggingCallback(log_every_steps=args.time_log_every),
        ]

        model.learn(total_timesteps=args.timesteps, callback=callbacks)
        train_env.close()
        eval_env.close()
        print(f"Best model : {os.path.join(best_dir, 'best_model.zip')}")
        print(f"Best vecnorm: {os.path.join(best_dir, 'vecnorm_best.pkl')}")
        return

    # =========================================================================
    # EVAL / COLLECT — load best model + VecNormalize
    # =========================================================================
    model_path = os.path.join(args.save_dir, "best", "best_model.zip")
    vecnorm_path = os.path.join(args.save_dir, "best", "vecnorm_best.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No best model found at {model_path}")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"No VecNormalize found at {vecnorm_path}")

    if args.full_action_space:
        raw_env = DummyVecEnv([make_full_env(args.env_id)])
    else:
        raw_env = DummyVecEnv([make_synergy_env(
            args.env_id, synergy_model,
            act_scale=args.act_scale,
            nonnegative_activities=args.nonnegative_activities,
        )])

    eval_env = VecNormalize.load(vecnorm_path, raw_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = SAC.load(model_path, env=eval_env, device=args.device)
    print(f"Loaded model    : {model_path}")
    print(f"Loaded VecNorm  : {vecnorm_path}")

    # =========================================================================
    # EVAL
    # =========================================================================
    if args.task == "eval":
        final_eval_loop(model, eval_env, n_episodes=args.eval_episodes)
        eval_env.close()
        return

    # =========================================================================
    # COLLECT
    # =========================================================================
    if args.task == "collect":
        save_path = args.traj_save_path or os.path.join(args.save_dir, "trajectory.npz")
        collect_trajectories(model, eval_env, args.traj_episodes, save_path)
        eval_env.close()
        return


if __name__ == "__main__":
    main()
