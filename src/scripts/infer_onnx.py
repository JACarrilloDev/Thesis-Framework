import argparse
import onnxruntime as ort
import numpy as np

from src.core.navigation_env import NavigationEnv

def run_onnx_inference(onnx_path, env_config, episodes=1):
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    # Create environment
    env = NavigationEnv(env_config)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            obs_np = np.array(obs, dtype=np.float32).reshape(1, -1)
            action = session.run(None, {input_name: obs_np})[0]
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            total_reward += reward
            steps += 1
        print(f"Episode {ep+1}: reward={total_reward}, steps={steps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with ONNX policy")
    parser.add_argument("--onnx_path", required=True, help="Path to exported ONNX model")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--obs_shape", type=int, nargs="+", default=[12484], help="Observation shape")
    args = parser.parse_args()

    env_config = {
        "scene_file": "examples/robot_navigation/ProjectorMove.ttt",
        "robot_type": "AstiPioneerHybrid",
        "robot_name_in_scene": "AstiPioneerHybrid",
        "task_config": {},  # Fill as needed
        "max_episode_steps": 1000,
        "obs_shape": tuple(args.obs_shape)
    }

    run_onnx_inference(args.onnx_path, env_config, episodes=args.episodes)