import argparse
import os
import sys
import yaml # For loading task config
from src.core.curriculum_manager import CurriculumManager, CurriculumStage

# Ensure src is in python path if running with `python3 src/scripts/...`
# For `python3 -m src.scripts...` this is usually not needed.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.simulation import Simulation
from src.core.robot_controller import RobotController
from src.core.task_manager import TaskManager # Minimal use here
from src.core.robot_env import RobotEnv
from src.core.reinforcement_learning import RLTrainer
from src.core.robot_definitions import get_robot_definition
from src.core.logger import setup_logger

# Setup main logger for this script
main_logger = setup_logger('projector_rl_main', 'logs/projector_rl_main.log', console_output=True)

def main():
    parser = argparse.ArgumentParser(description="Run DRL training for projector pick-and-place task.")
    parser.add_argument("--task_yaml", required=True, help="Path to the task YAML configuration file.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations.")
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim in headless mode.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a checkpoint to restore training from.")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--start-stage", type=str, default="REACH", choices=[stage.name for stage in CurriculumStage], help="Starting curriculum stage")

    args = parser.parse_args()

    # Load task configuration from YAML
    if not os.path.exists(args.task_yaml):
        main_logger.error(f"Task YAML file not found: {args.task_yaml}")
        sys.exit(1)
    with open(args.task_yaml, 'r') as f:
        task_config = yaml.safe_load(f)
    
    scene_file_path = task_config.get("scene_file", "examples/scenes/default_scene.ttt")
    if not os.path.isabs(scene_file_path): # Make path absolute if relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        scene_file_path = os.path.join(project_root, scene_file_path)


    if not os.path.exists(scene_file_path):
        main_logger.error(f"Scene file specified in YAML not found: {scene_file_path}")
        sys.exit(1)

    sim_instance = None
    try:
        main_logger.info("Initializing Simulation...")
        sim_instance = Simulation(scene_file=scene_file_path, headless=args.headless)
        sim_instance.import_environment() # Launches CoppeliaSim
        sim_instance.start()

        # Assuming one robot for this task from YAML
        robot_setup_config = task_config['robots_setup'][0]
        robot_type_name = robot_setup_config['type']
        robot_def = get_robot_definition(robot_type_name)
        if not robot_def:
            main_logger.error(f"Robot definition for '{robot_type_name}' not found.")
            sys.exit(1)
        
        # Update robot_def with instance-specific name if needed (e.g. from YAML)
        robot_def["robot_name_in_scene"] = robot_setup_config.get("name_in_scene", robot_def.get("robot_name_in_scene", robot_type_name))


        main_logger.info("Initializing RobotController...")
        # RobotController needs the PyRep instance from Simulation
        robot_controller = RobotController(robot_definition=robot_def, pyrep_instance=sim_instance.pr)

        main_logger.info("Initializing TaskManager (minimal use for DRL)...")
        task_manager = TaskManager(task_file_path=args.task_yaml) # Loads objectives names

        main_logger.info("Initializing RobotEnv...")
        # RobotEnv needs to be registered with RLlib if not using a string name
        # For direct instantiation:
        env_config_for_rllib = {
            "sim_instance": sim_instance,
            "robot_controller": robot_controller,
            "task_manager": task_manager,
            "task_config": task_config, # Pass the loaded YAML dict
            "max_episode_steps": 1000,
        }
        # To use with RLlib's trainer.env_config, the env class itself is passed
        # or registered with tune.register_env

        main_logger.info("Initializing RLTrainer...")
        # RLlib expects the env to be a string name (if registered) or a class.
        # If passing class, env_config is used by RLlib to instantiate it.
        trainer = RLTrainer(env_class=RobotEnv, env_name="projector_env", env_config=env_config_for_rllib)
        
        if args.checkpoint_path:
            if os.path.exists(args.checkpoint_path):
                main_logger.info(f"Restoring training from checkpoint: {args.checkpoint_path}")
                trainer.restore(args.checkpoint_path)
            else:
                main_logger.warning(f"Checkpoint path {args.checkpoint_path} not found. Starting new training.")


        main_logger.info(f"Starting DRL training for {args.iterations} iterations...")
        for i in range(args.iterations):
            result = trainer.train()
            main_logger.info(f"Iteration {i+1}/{args.iterations}: {result}")
            if (i + 1) % 10 == 0: # Save checkpoint every 10 iterations
                checkpoint_dir = trainer.save_checkpoint()
                main_logger.info(f"Checkpoint saved in directory {checkpoint_dir}")
        
        main_logger.info("Training complete.")

    except Exception as e:
        main_logger.error(f"An error occurred during the RL process: {e}", exc_info=True)
    finally:
        if sim_instance:
            main_logger.info("Shutting down simulation.")
            sim_instance.shutdown()
        main_logger.info("Script finished.")

if __name__ == "__main__":
    main()