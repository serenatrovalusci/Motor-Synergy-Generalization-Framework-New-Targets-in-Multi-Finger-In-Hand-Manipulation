import numpy as np
import pickle
import gymnasium as gym
import gymnasium_robotics
import argparse
from typing import Optional

def eval_synergy_replay(
    synergy_pkl_path: Optional[str],
    npz_path: str,
    array_key: str,
    env_id: str,
    n_episodes: int = None,
    compute_mse: bool = True,
    use_synergy: bool = False
):
    """Evaluate a saved policy by replaying actions in an environment.
    If use_synergy=True, actions are encoded/decoded through the synergy model first."""
    # Load synergy model only if specified
    if use_synergy and synergy_pkl_path:
        with open(synergy_pkl_path, "rb") as f:
            bundle = pickle.load(f)
        synergy_model = bundle["synergy_model"]
        mu = bundle.get("mu")
        sigma = bundle.get("sigma")
        standardized = bundle.get("standardized", False)
    else:
        synergy_model = None
        mu = sigma = None
        standardized = False

    # Load trajectories; we expect actions and possibly observations
    data = np.load(npz_path)[array_key]  # (N, T, M)
    N, T, M = data.shape
    if n_episodes is None or n_episodes > N:
        n_episodes = N

    # Create environment
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id, render_mode="human")

    successes, returns, lengths, mse_errors = [], [], [], []
    viewer_cleaned_up = False

    for i in range(n_episodes):
        actions_orig = data[i]  # (T, M)

        if use_synergy and synergy_model:
            # Standardize if necessary
            data_input = actions_orig.copy()
            if standardized and mu is not None and sigma is not None:
                data_input = (data_input - mu) / (sigma + 1e-8)
            # Encode and decode via synergy model
            acts = synergy_model.encode(data_input[None])
            recon = synergy_model.decode(acts)[0]
            if standardized and mu is not None and sigma is not None:
                recon = recon * sigma + mu
        else:
            recon = actions_orig

        # Compute MSE only if synergies were used
        if compute_mse and use_synergy and synergy_model:
            mse_errors.append(np.mean((actions_orig - recon)**2))

        # Execute actions in the environment
        obs, _ = env.reset()
        ep_ret = 0.0
        success = 0
        for t in range(T):
            a = recon[t]
            obs, reward, done, truncated, info = env.step(a)

            # Hide the on-screen HUD overlay for clean video capture. Pure
            # rendering option — does not affect simulation. Applied once,
            # as soon as the viewer window exists.
            if not viewer_cleaned_up:
                viewer = env.unwrapped.mujoco_renderer.viewer
                if viewer is not None:
                    viewer._hide_menu = True
                    viewer_cleaned_up = True

            ep_ret += float(reward)
            if info.get("is_success", 0):
                success = 1
            if done or truncated:
                break

        successes.append(success)
        returns.append(ep_ret)
        lengths.append(t + 1)

    env.close()

    # Print metrics
    success_rate = float(np.mean(successes)) if successes else 0.0
    print(f"Success rate: {success_rate:.3f}")
    print(f"Successes: {int(np.sum(successes))}/{n_episodes}")
    print(f"Return mean: {np.mean(returns):.2f}")
    print(f"Episode length mean: {np.mean(lengths):.1f}")
    if compute_mse and mse_errors:
        print(f"Reconstruction MSE mean: {np.mean(mse_errors):.6e}")
        print(f"Reconstruction MSE std: {np.std(mse_errors):.6e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synergy-pkl-path", type=str, default=None,
                        help="Path to synergy bundle (.pkl); leave blank if not using synergies")
    parser.add_argument("--npz-path", type=str, required=True,
                        help="Path to .npz file with trajectories")
    parser.add_argument("--array-key", type=str, default="actions",
                        help="Key in the .npz file (e.g. 'actions')")
    parser.add_argument("--env-id", type=str, required=True,
                        help="Gym environment ID")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Number of trajectories to evaluate")
    parser.add_argument("--synergy", action="store_true",
                        help="Use synergies for action reconstruction")
    args = parser.parse_args()

    eval_synergy_replay(
        synergy_pkl_path=args.synergy_pkl_path,
        npz_path=args.npz_path,
        array_key=args.array_key,
        env_id=args.env_id,
        n_episodes=args.n_episodes,
        compute_mse=True,
        use_synergy=args.synergy
    )

if __name__ == "__main__":
    main()