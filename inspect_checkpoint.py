import pickle
import os
import numpy as np
import argparse
import yaml
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.algorithms.ppo import PPOConfig

from run_rl_task import build_env_config, promote_nested_keys
from examples.multirobot.envs.multirobot_env import DynamicTwoPhaseNavEnv

def _dummy_obs_from_space(space: spaces.Space):
    if isinstance(space, spaces.Dict):
        return {k: np.zeros(s.shape, dtype=s.dtype) for k, s in space.spaces.items()}
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    raise ValueError(f"Unsupported space: {space}")

def sanity_check_policy(checkpoint_dir: str, task_yaml: str):
    with open(task_yaml, "r") as f:
        root_cfg = yaml.safe_load(f) or {}
    promote_nested_keys(root_cfg)
    env_cfg = build_env_config(root_cfg, headless=True)
    cfg = (
        PPOConfig()
        .environment(env=DynamicTwoPhaseNavEnv, env_config=env_cfg, disable_env_checking=True)
        .framework("torch")
    )
    trainer = PPOTrainer(config=cfg)
    trainer.restore(checkpoint_dir)
    # Grab first policy (shared by default).
    policy = next(iter(trainer.workers.local_worker().policy_map.values()))
    obs = _dummy_obs_from_space(policy.observation_space)
    action, _, _ = policy.compute_single_action(obs, explore=False)
    ok = np.isfinite(action).all()
    print(f"[SanityCheck] Action finite: {ok}")
    if not ok:
        print("[SanityCheck] NaN/Inf action -> checkpoint likely corrupt for export.")
    trainer.stop()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity_check", action="store_true")
    ap.add_argument("--task_yaml", default="examples/multirobot/tasks/multirobot.yaml")
    args = ap.parse_args()

    checkpoint_file = os.path.join("checkpoints", "algorithm_state.pkl")

    if args.sanity_check:
        sanity_check_policy("checkpoints", args.task_yaml)

if not os.path.exists(checkpoint_file):
    print(f"Error: Checkpoint file not found at {checkpoint_file}")
else:
    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract key information
        iterations = data.get("training_iteration", "Not found")
        timesteps = data.get("timesteps_total", "Not found")
        metrics = data.get("metrics", {})
        hist_stats = metrics.get("hist_stats", {}) if metrics else {}

        print("Checkpoint analysis successful:")
        print(f"  - Training Iterations Completed: {iterations}")
        print(f"  - Total Timesteps Trained: {timesteps}")

        # Print training metrics if available
        if metrics:
            rewards = hist_stats.get("episode_reward", [])
            lengths = hist_stats.get("episode_lengths", [])
            print("  - Mean Reward (RLlib):", metrics.get("episode_reward_mean", "N/A"))
            print("  - Max Reward (RLlib):", metrics.get("episode_reward_max", "N/A"))
            print("  - Min Reward (RLlib):", metrics.get("episode_reward_min", "N/A"))
            print("  - Episodes Total:", metrics.get("episodes_total", "N/A"))
            print(f"  - Total Episode Rewards Stored: {len(rewards)}")
            if rewards:
                print(f"  - Historical Mean Reward: {np.mean(rewards):.2f}")
                print(f"  - Historical Max Reward: {np.max(rewards):.2f}")
                print(f"  - Historical Min Reward: {np.min(rewards):.2f}")
                print(f"  - Mean Reward Last 100: {np.mean(rewards[-100:]):.2f}")
                print(f"  - Mean Reward Last 10: {np.mean(rewards[-10:]):.2f}")
                print(f"  - Last 10 Episode Rewards: {rewards[-10:]}")
            else:
                print("  - No episode rewards found in hist_stats.")
            if lengths:
                print(f"  - Mean Episode Length: {np.mean(lengths):.2f}")
                print(f"  - Last 10 Episode Lengths: {lengths[-10:]}")
        else:
            print("  - No training metrics found in checkpoint.")

    except Exception as e:
        print(f"Error: The checkpoint file seems to be broken or corrupted.")
        print(f"Details: {e}")