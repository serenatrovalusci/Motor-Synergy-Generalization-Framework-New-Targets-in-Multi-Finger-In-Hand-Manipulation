#!/usr/bin/env python3
"""
NO-PCA baseline: SAC + HER training script (cleaned + gamma=0.99 + time callback).
"""

from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import argparse
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import os
from glob import glob
import time


# -------------------------------------------------------------
# Register robotics envs
# -------------------------------------------------------------
def register_robotics_envs():
    try:
        gym.register_envs(gymnasium_robotics)
    except Exception:
        pass


# -------------------------------------------------------------
# Time logging callback (TensorBoard)
# -------------------------------------------------------------
class TimeLoggingCallback(BaseCallback):
    """
    Logs wall-clock time and FPS to TensorBoard.

    Scalars:
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


# -------------------------------------------------------------
# Early stopping callback (reads EvalCallback logs)
# -------------------------------------------------------------
class EarlyStopOnEvalPlateau(BaseCallback):
    """
    Stops training when the evaluation metric does not improve
    for `patience` consecutive evaluations.

    It reads EvalCallback's log_path/evaluations.npz.

    If 'successes' exists in evaluations.npz, uses mean success rate.
    Otherwise falls back to mean reward.
    """

    def __init__(
        self,
        eval_log_dir: str,
        patience: int = 10,
        min_delta: float = 0.01,
        min_evals: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_log_dir = eval_log_dir
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_evals = int(min_evals)

        self._best = -np.inf
        self._bad = 0
        self._last_seen_eval_count = 0

    def _read_latest_metric(self):
        path = os.path.join(self.eval_log_dir, "evaluations.npz")
        if not os.path.exists(path):
            return None, 0

        data = np.load(path, allow_pickle=True)
        n_evals = len(data["timesteps"])

        if "successes" in data.files:
            successes = data["successes"]
            metric = float(np.mean(successes[-1]))
            return metric, n_evals

        results = data["results"]
        metric = float(np.mean(results[-1]))
        return metric, n_evals

    def _on_step(self) -> bool:
        metric, n_evals = self._read_latest_metric()
        if metric is None:
            return True

        if n_evals <= self._last_seen_eval_count:
            return True
        self._last_seen_eval_count = n_evals

        if n_evals < self.min_evals:
            if self.verbose:
                print(f"[early-stop] eval {n_evals}/{self.min_evals} (warmup), metric={metric:.4f}")
            return True

        improved = metric > (self._best + self.min_delta)
        if improved:
            self._best = metric
            self._bad = 0
            if self.verbose:
                print(f"[early-stop] improved: best={self._best:.4f} (evals={n_evals})")
        else:
            self._bad += 1
            if self.verbose:
                print(
                    f"[early-stop] no improvement (metric={metric:.4f}, best={self._best:.4f}) "
                    f"bad={self._bad}/{self.patience}"
                )
            if self._bad >= self.patience:
                if self.verbose:
                    print(
                        f"[early-stop] STOPPING: plateau for {self.patience} evals "
                        f"(min_delta={self.min_delta})."
                    )
                return False

        return True


# -------------------------------------------------------------
# Env creation
# -------------------------------------------------------------
def make_env(env_id: str, render: bool = False):
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    env = Monitor(env)
    return env


# -------------------------------------------------------------
# Build NON-PARALLEL train/eval envs
# -------------------------------------------------------------
def build_train_eval_envs(env_id: str):
    train_env = DummyVecEnv([lambda: make_env(env_id, render=False)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = DummyVecEnv([lambda: make_env(env_id, render=False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.obs_rms = train_env.obs_rms
    return train_env, eval_env


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def find_latest_checkpoint_zip(ckpt_dir: str):
    zips = sorted(glob(os.path.join(ckpt_dir, "*.zip")), key=os.path.getmtime)
    return zips[-1] if zips else None


# -------------------------------------------------------------
# Final evaluation loop
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Trajectory collection
# -------------------------------------------------------------
    
def collect_trajectories(model, eval_env, n_episodes, save_path, deterministic=True):
    all_ep_actions = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            ep_actions.append(np.array(action[0], dtype=np.float32))
            obs, rewards, dones, infos = eval_env.step(action)
            done = bool(dones[0])

        all_ep_actions.append(np.stack(ep_actions, axis=0))
        print(f"[collect] Episode {ep + 1}/{n_episodes} length = {len(ep_actions)}")

    actions_arr = np.stack(all_ep_actions, axis=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, actions=actions_arr)
    print(f"[collect] Saved actions with shape {actions_arr.shape} → {save_path}")


def collect_trajectories_with_obs(model, eval_env, n_episodes, save_path, deterministic=True):
    all_ep_actions = []
    all_ep_obs = []
    all_ep_raw_obs = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_actions = []
        ep_obs = []
        ep_raw_obs = []

        while not done:
            ep_obs.append(obs.copy())
            if hasattr(eval_env, "unnormalize_obs"):
                raw_obs = eval_env.unnormalize_obs(obs)
                ep_raw_obs.append(raw_obs.copy())

            action, _ = model.predict(obs, deterministic=deterministic)
            ep_actions.append(np.array(action[0], dtype=np.float32))
            obs, rewards, dones, infos = eval_env.step(action)
            done = bool(dones[0])

        all_ep_actions.append(np.stack(ep_actions, axis=0))
        all_ep_obs.append(np.stack(ep_obs, axis=0))
        if ep_raw_obs:
            all_ep_raw_obs.append(np.stack(ep_raw_obs, axis=0))

        print(f"[collect] Episode {ep + 1}/{n_episodes} length = {len(ep_actions)}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if all_ep_raw_obs:
        np.savez_compressed(
            save_path,
            actions=np.stack(all_ep_actions, axis=0),
            observations=np.stack(all_ep_obs, axis=0),
            raw_observations=np.stack(all_ep_raw_obs, axis=0),
        )
    else:
        np.savez_compressed(
            save_path,
            actions=np.stack(all_ep_actions, axis=0),
            observations=np.stack(all_ep_obs, axis=0),
        )

    print(f"[collect] Saved trajectories to {save_path}")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["train", "eval", "collect", "collect-act-obs"],
        default="train",
        help="Select between training, evaluation, and trajectory collection.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1",
        help="Gymnasium environment ID.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of episodes for final evaluation.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=100_000,
        help="Timesteps between intermediate evaluations during training.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="Manipulation2.0/egg_sac_her_2M",
        help="Directory to save models, logs, and trajectories.",
    )
    parser.add_argument(
        "--n-sampled-goal",
        type=int,
        default=8,
        help="Number of HER sampled goals.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--target-entropy-scale",
        type=float,
        default=0.8,
        help="Scale for automatic entropy tuning.",
    )
    parser.add_argument(
        "--load-best",
        action="store_true",
        help="When evaluating, load the best model instead of the most recent.",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping based on evaluation plateau.",
    )
    parser.add_argument(
        "--es-patience",
        type=int,
        default=10,
        help="Stop after this many evaluations without improvement.",
    )
    parser.add_argument(
        "--es-min-delta",
        type=float,
        default=0.01,
        help="Minimum improvement to reset patience.",
    )
    parser.add_argument(
        "--es-min-evals",
        type=int,
        default=5,
        help="Warmup eval count before early stop can trigger.",
    )
    parser.add_argument(
        "--traj-episodes",
        type=int,
        default=200,
        help="Number of episodes to record for trajectory collection.",
    )
    parser.add_argument(
        "--traj-save-path",
        type=str,
        default=None,
        help="Path to save collected trajectories.",
    )
    parser.add_argument(
        "--time-log-every",
        type=int,
        default=10_000,
        help="Log time/FPS every N steps to TensorBoard.",
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    best_dir = os.path.join(args.save_dir, "best")
    ckpt_dir = os.path.join(args.save_dir, "ckpts")
    eval_log_dir = os.path.join(args.save_dir, "eval_logs")
    for d in [best_dir, ckpt_dir, eval_log_dir]:
        os.makedirs(d, exist_ok=True)

    register_robotics_envs()


    def make_callbacks(eval_env):
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=eval_log_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=max(20, args.eval_episodes),
            deterministic=True,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=50_000,
            save_path=ckpt_dir,
            name_prefix="sac_her_hand",
            save_vecnormalize=True,
            save_replay_buffer=False,
        )

        callbacks = [eval_cb, ckpt_cb, TimeLoggingCallback(log_every_steps=args.time_log_every)]
        if args.early_stop:
            callbacks.append(
                EarlyStopOnEvalPlateau(
                    eval_log_dir=eval_log_dir,
                    patience=args.es_patience,
                    min_delta=args.es_min_delta,
                    min_evals=args.es_min_evals,
                    verbose=1,
                )
            )
        return callbacks

    # =========================================================
    # TRAIN
    # =========================================================
    if args.task == "train":
        train_env, eval_env = build_train_eval_envs(args.env_id)

        action_dim = train_env.action_space.shape[-1]
        target_entropy = -action_dim * args.target_entropy_scale

        model = SAC(
            "MultiInputPolicy",
            train_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=args.n_sampled_goal,
                goal_selection_strategy="future",
            ),
            buffer_size=int(1e6),
            batch_size=256,
            learning_rate=args.lr,
            gamma=args.gamma,
            learning_starts=10_000,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
            target_entropy=target_entropy,
            verbose=1,
            device="cuda",
            tensorboard_log=os.path.join(args.save_dir, "tb_logs"),
        )

        callbacks = make_callbacks(eval_env)

        model.learn(total_timesteps=args.timesteps, callback=callbacks)

        model_path = os.path.join(args.save_dir, "manipulate_block_hand_sher")
        vecnorm_path = os.path.join(args.save_dir, "vecnorm_train.pkl")

        model.save(model_path)
        train_env.save(vecnorm_path)

        print(f"Saved model: {model_path}")
        print(f"Saved VecNormalize stats: {vecnorm_path}")

        final_eval_loop(model, eval_env, n_episodes=args.eval_episodes)
        train_env.close()
        eval_env.close()
        return

    # =========================================================
    # EVAL / COLLECT
    # =========================================================
    raw_eval_env = DummyVecEnv([lambda: make_env(args.env_id, render=False)])

    vecnorm_path = os.path.join(args.save_dir, "vecnorm_train.pkl")
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"No VecNormalize stats found at {vecnorm_path}")

    eval_env = VecNormalize.load(vecnorm_path, raw_eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    print(f"Loaded VecNormalize: {vecnorm_path}")

    best_model = os.path.join(args.save_dir, "best", "best_model.zip")
    latest = os.path.join(args.save_dir, "manipulate_block_hand_sher")
    model_path = best_model if (args.load_best and os.path.exists(best_model)) else latest

    if not os.path.exists(model_path):
        latest_ckpt = find_latest_checkpoint_zip(os.path.join(args.save_dir, "ckpts"))
        if latest_ckpt is None:
            raise FileNotFoundError("No model found to eval/collect.")
        model_path = latest_ckpt

    print(f"Loading model: {model_path}")
    model = SAC.load(model_path, env=eval_env, device="cuda")

    if args.task == "eval":
        final_eval_loop(model, eval_env, n_episodes=args.eval_episodes)
        eval_env.close()
        return

    if args.task == "collect":
        save_path = args.traj_save_path or os.path.join(args.save_dir, "trajectories_actions.npz")
        collect_trajectories(model, eval_env, args.traj_episodes, save_path)
        eval_env.close()
        return

    if args.task == "collect-act-obs":
        save_path = args.traj_save_path or os.path.join(args.save_dir, "trajectories_act_obs.npz")
        collect_trajectories_with_obs(model, eval_env, args.traj_episodes, save_path)
        eval_env.close()
        return


if __name__ == "__main__":
    main()