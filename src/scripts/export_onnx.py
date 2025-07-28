import argparse
import os
import torch
import numpy as np
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.algorithms.ppo import PPOConfig

from src.core.navigation_env import NavigationEnv

def export_policy_to_onnx(checkpoint_dir, env_config, output_path):
    # Restore RLlib trainer
    config = (
        PPOConfig()
        .environment(env=NavigationEnv, env_config=env_config)
        .framework("torch")
    )
    trainer = PPOTrainer(config=config)
    trainer.restore(checkpoint_dir)

    # Get the policy/model
    policy = trainer.get_policy()
    model = policy.model
    model.eval()

    # Dummy input for tracing (match your obs shape)
    obs_shape = env_config.get("obs_shape", (4 + 16 + 64*64*3,))
    dummy_input = torch.randn(1, *obs_shape)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=11
    )
    print(f"Exported ONNX model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RLlib PPO policy to ONNX")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to RLlib checkpoint directory (e.g. checkpoints/)")
    parser.add_argument("--output_path", default="exported_policy.onnx", help="Output ONNX file path")
    parser.add_argument("--obs_shape", type=int, nargs="+", default=[12484], help="Observation shape (default: 4+16+64*64*3)")
    args = parser.parse_args()

    # Example: load env_config as in your training script
    env_config = {
        "scene_file": "examples/robot_navigation/ProjectorMove.ttt",
        "robot_type": "AstiPioneerHybrid",
        "robot_name_in_scene": "AstiPioneerHybrid",
        "task_config": {},  # Fill as needed
        "max_episode_steps": 1000,
        "obs_shape": tuple(args.obs_shape)
    }

    export_policy_to_onnx(args.checkpoint_dir, env_config, args.output_path)