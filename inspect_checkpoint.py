import pickle
import os
import numpy as np

checkpoint_file = os.path.join("checkpoints", "algorithm_state.pkl")

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