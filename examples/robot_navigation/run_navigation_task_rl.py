import argparse
import os
import sys
import yaml
import glob

from src.core.simulation import Simulation
from src.core.robot_controller import RobotController
from src.core.navigation_env import NavigationEnv
from src.core.reinforcement_learning import RLTrainer
from src.core.robot_definitions import get_robot_definition
from src.core.logger import setup_logger
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

main_logger = setup_logger('nav_rl_main', 'logs/nav_rl_main.log', console_output=True)

def get_latest_checkpoint():
    """Find the most recent checkpoint in the checkpoints directory."""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*"))
    if not checkpoints:
        return None
        
    # Sort by modification time
    latest = max(checkpoints, key=os.path.getmtime)
    return latest

def main():
    parser = argparse.ArgumentParser(description="Run DRL navigation training.")
    parser.add_argument("--task_yaml", required=True, help="Path to the task YAML configuration file.")
    parser.add_argument("--iterations", type=int, default=300, help="Number of training iterations.")
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim in headless mode.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to a checkpoint to restore training from.")
    parser.add_argument("--fresh_training", action="store_true", help="Start fresh training ignoring existing checkpoints")
    args = parser.parse_args()

    # Task config loading
    if not os.path.exists(args.task_yaml):
        main_logger.error(f"Task YAML file not found: {args.task_yaml}")
        sys.exit(1)
    with open(args.task_yaml, 'r') as f:
        task_config = yaml.safe_load(f)

    # Scene path setup
    scene_file_path = task_config.get("scene_file")
    if not os.path.isabs(scene_file_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        scene_file_path = os.path.join(project_root, scene_file_path)

    if not os.path.exists(scene_file_path):
        main_logger.error(f"Scene file specified in YAML not found: {scene_file_path}")
        sys.exit(1)

    sim_instance = None
    try:
        main_logger.info("Initializing Simulation...")

        # Environment setup
        robot_setup_config = task_config['robots_setup'][0]
        robot_type_name = robot_setup_config['type']
        robot_def = get_robot_definition(robot_type_name)
        robot_def["robot_name_in_scene"] = robot_setup_config.get("name", robot_def.get("robot_name_in_scene", robot_type_name))

        env_config = {
            "scene_file": scene_file_path,
            "robot_type": robot_type_name,
            "robot_name_in_scene": robot_setup_config.get("name", robot_def.get("robot_name_in_scene", robot_type_name)),
            "task_config": task_config,
            "max_episode_steps": 300,
        }
        
        trainer = RLTrainer(env_class=NavigationEnv, env_config=env_config)

        # Checkpoint handling
        checkpoint_to_load = None
        if not args.fresh_training:
            if args.checkpoint_path:
                checkpoint_to_load = args.checkpoint_path
            else:
                checkpoint_to_load = get_latest_checkpoint()

        if checkpoint_to_load and os.path.exists(checkpoint_to_load):
            main_logger.info(f"Restoring training from checkpoint: {checkpoint_to_load}")
            trainer.restore(checkpoint_to_load)

        # Training loop
        main_logger.info(f"Starting DRL navigation training for {args.iterations} iterations...")
        total_iterations = 0
        
        try:
            for i in range(args.iterations):
                result = trainer.train()
                total_iterations += 1
                
                # Detailed logging
                main_logger.info(
                    f"Iteration {total_iterations}/{args.iterations}: "
                    f"mean_reward={result.get('episode_reward_mean', 'N/A'):.2f}, "
                    f"episodes={result.get('episodes_this_iter', 'N/A')}"
                )
                
                # Regular checkpoints
                if total_iterations % 10 == 0:
                    checkpoint_dir = trainer.save_checkpoint()
                    main_logger.info(f"Checkpoint saved at iteration {total_iterations}")
                    
                if total_iterations >= args.iterations:
                    # Final checkpoint
                    final_checkpoint = trainer.save_checkpoint()
                    main_logger.info(f"Training completed. Final checkpoint saved.")
                    break
                    
        except KeyboardInterrupt:
            main_logger.info("Training interrupted by user.")
            # Save emergency checkpoint
            emergency_checkpoint = trainer.save_checkpoint()
            main_logger.info(f"Emergency checkpoint saved at iteration {total_iterations}")
            raise

        main_logger.info(f"Training complete after {total_iterations} iterations.")

    except Exception as e:
        main_logger.error(f"An error occurred during the RL process: {e}", exc_info=True)
    finally:
        if sim_instance:
            main_logger.info("Shutting down simulation.")
            sim_instance.shutdown()
        main_logger.info("Script finished.")

if __name__ == "__main__":
    main()