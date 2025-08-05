import pickle
import os

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
        
        print("Checkpoint analysis successful:")
        print(f"  - Training Iterations Completed: {iterations}")
        print(f"  - Total Timesteps Trained: {timesteps}")

    except Exception as e:
        print(f"Error: The checkpoint file seems to be broken or corrupted.")
        print(f"Details: {e}")