import argparse
import os
import yaml
from examples.navigation.envs.navigation_env import NavigationEnv
from src.core.reinforcement_learning import RLTrainer
import warnings
import logging

# Suppress all deprecation warnings and INFO logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("gymnasium").setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description="Run DRL navigation training for Asti.")
    parser.add_argument("--task_yaml", required=True, help="Path to the task YAML configuration file.")
    parser.add_argument("--iterations", type=int, default=300, help="Number of training iterations.")
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim in headless mode.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint to restore from.")
    args = parser.parse_args()

    # Load task configuration
    with open(args.task_yaml, 'r') as f:
        task_config = yaml.safe_load(f)

    # Prepare environment configuration
    env_config = {
        "scene_file": task_config["scene_file"],
        "robot_type": task_config["robots_setup"][0]["type"],
        "robot_name_in_scene": task_config["robots_setup"][0]["name"],
        "task_config": task_config,
        "max_episode_steps": task_config.get("max_episode_steps", 300),
        "headless": args.headless
    }

    # Initialize trainer
    trainer = RLTrainer(
        env_class=NavigationEnv,
        env_config=env_config
    )

    # Restore from checkpoint if specified
    if args.checkpoint_path:
        trainer.restore(args.checkpoint_path)

    # Train
    trainer.train(iterations=args.iterations)

if __name__ == "__main__":
    main()