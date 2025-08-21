import argparse, os, yaml, logging, warnings, sys
from examples.multirobot.envs.multirobot_env import DynamicTwoPhaseNavEnv
from src.core.reinforcement_learning import RLTrainer
from src.core.logger import setup_logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ray").setLevel(logging.ERROR)

main_logger = setup_logger("multi_nav_rl", "logs/multi_nav_rl.log", console_output=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_yaml", required=True)
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--checkpoint_path")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.task_yaml):
        main_logger.error(f"Task YAML not found: {args.task_yaml}")
        sys.exit(1)

    with open(args.task_yaml, "r") as f:
        task_cfg = yaml.safe_load(f)

    # Ensure exactly 4 targets
    targets = task_cfg.get("targets", [])
    if len(targets) != 4:
        main_logger.error(f"DynamicTwoPhaseNavEnv requires exactly 4 targets, got {len(targets)}: {targets}")
        sys.exit(1)

    if args.verbose:
        task_cfg["verbose"] = True

    # Resolve scene path relative to project root (mirrors navigation script style)
    scene_file = task_cfg["scene_file"]
    if not os.path.isabs(scene_file):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        scene_file = os.path.abspath(os.path.join(project_root, scene_file))
    if not os.path.exists(scene_file):
        main_logger.error(f"Scene file not found: {scene_file}")
        sys.exit(1)

    env_config = {
        "scene_file": scene_file,
        "robot_type": task_cfg["robots_setup"][0]["type"],
        "robot_names_in_scene": [
            task_cfg["robots_setup"][0]["name"],
            task_cfg["robots_setup"][1]["name"]
        ],
        "task_config": task_cfg,
        "target_dummies": targets,
        "start_dummies": task_cfg.get("start_dummies", []),
        "max_episode_steps": task_cfg.get("max_episode_steps", 500),
        "success_dist": task_cfg.get("success_dist", 0.40),
        "collision_dist": task_cfg.get("collision_dist", 0.30),
        "use_camera": task_cfg.get("use_camera", False),
        "camera_key": task_cfg.get("camera_key", "front_camera"),
        "camera_size": task_cfg.get("camera_size", [84,84]),
        "camera_grayscale": task_cfg.get("camera_grayscale", True),
        "frame_stack": task_cfg.get("frame_stack", 4),
        "headless": args.headless,
        "verbose": task_cfg.get("verbose", False),
        "multi_agent": True
    }

    trainer = RLTrainer(
        env_class=DynamicTwoPhaseNavEnv,
        env_config=env_config,
        train_batch_size=16000
    )

    if args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            main_logger.info(f"Restoring from checkpoint: {args.checkpoint_path}")
            trainer.restore(args.checkpoint_path)
        else:
            main_logger.warning(f"Checkpoint not found: {args.checkpoint_path}")

    main_logger.info(f"Starting multi-robot training for {args.iterations} iterations...")
    trainer.train(iterations=args.iterations)
    main_logger.info("Training finished.")

if __name__ == "__main__":
    main()