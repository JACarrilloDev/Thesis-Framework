import argparse, os, sys, yaml, importlib, logging, warnings, copy
from typing import Any, Dict
from src.core.reinforcement_learning import RLTrainer
from src.core.logger import setup_logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ray").setLevel(logging.ERROR)

class _SilenceGetSliceDeprecation(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "_get_slice_indices" not in record.getMessage()

logging.getLogger("ray").addFilter(_SilenceGetSliceDeprecation())
logging.getLogger("py.warnings").addFilter(_SilenceGetSliceDeprecation())

log = setup_logger("generic_rl", "logs/generic_rl.log", console_output=True)

def import_env_class(dotted: str, fallback_file: str = None, fallback_class: str = None):
    """Import environment class.
    Priority:
      1) dotted path (standard import)
      2) fallback_file + fallback_class (direct .py path)
    """
    if dotted:
        try:
            mod_path, cls_name = dotted.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            return getattr(mod, cls_name)
        except Exception as e:
            log.warning(f"Failed dotted import '{dotted}': {e}")
            if not (fallback_file and fallback_class):
                sys.exit(1)
    if fallback_file and fallback_class:
        try:
            spec = importlib.util.spec_from_file_location("user_env_module", fallback_file)
            if spec is None or spec.loader is None:
                raise RuntimeError("Spec/loader not resolved.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, fallback_class)
        except Exception as e:
            log.error(f"Failed file-based import env_file='{fallback_file}' class='{fallback_class}': {e}")
            sys.exit(1)
    log.error("Could not import environment class (no valid strategy).")
    sys.exit(1)

def promote_nested_keys(cfg: Dict[str, Any]) -> None:
    """
    Promote nested task_config.* keys (robots_setup, reward_weights, etc.)
    to top-level so envs and RLTrainer that expect flat configs still work.
    """
    task_cfg = cfg.get("task_config", {})
    # robots_setup
    if "robots_setup" not in cfg and "robots_setup" in task_cfg:
        cfg["robots_setup"] = task_cfg["robots_setup"]
        log.debug("Promoted task_config.robots_setup to top-level.")
    # robot_type
    if "robot_type" not in cfg:
        rs = cfg.get("robots_setup") or task_cfg.get("robots_setup") or []
        if rs:
            rt = rs[0].get("type")
            if rt:
                cfg["robot_type"] = rt
                log.debug(f"Derived robot_type='{rt}' from first robots_setup entry.")
    # robot names
    if "robot_names_in_scene" not in cfg:
        rs = cfg.get("robots_setup", [])
        if len(rs) >= 2:
            cfg["robot_names_in_scene"] = [
                r.get("name") or r.get("robot_name_in_scene") or r.get("type", f"robot{i}")
                for i, r in enumerate(rs[:2])
            ]
    # multi_agent default
    if "multi_agent" not in cfg:
        if len(cfg.get("robots_setup", [])) == 2:
            cfg["multi_agent"] = True

def build_env_config(root_cfg: Dict[str, Any], headless: bool) -> Dict[str, Any]:
    """
    Generate env_config consumed by environment classes and RLTrainer.
    root_cfg is the full YAML (possibly with nested task_config).
    """
    # Use a merged task_config (if nested) for env.task_cfg
    task_cfg = root_cfg.get("task_config", {})
    # Shallow merge: top-level keys override nested for shared names
    merged_task_cfg = copy.deepcopy(task_cfg)
    for k, v in root_cfg.items():
        if k not in ("task_config", "env_class"):
            # Do not overwrite an existing nested dict unless top-level provides value
            if isinstance(v, dict) and k in merged_task_cfg and isinstance(merged_task_cfg[k], dict):
                # merge sub-dict
                merged_task_cfg[k] = {**merged_task_cfg[k], **v}
            else:
                merged_task_cfg[k] = v

    env_cfg: Dict[str, Any] = {
        "scene_file": root_cfg["scene_file"],
        "task_config": merged_task_cfg,   # full (possibly merged) task_config
        "headless": headless,
        "max_episode_steps": root_cfg.get("max_episode_steps",
                                          root_cfg.get("episode_horizon",
                                                       merged_task_cfg.get("max_episode_steps",
                                                                           merged_task_cfg.get("episode_horizon", 500)))),
    }

    # Robots setup (already promoted)
    robots = root_cfg.get("robots_setup", [])
    if len(robots) == 0:
        log.warning("robots_setup empty in YAML; ensure env will not expect robot mapping.")
    elif len(robots) == 1:
        r0 = robots[0]
        env_cfg["robot_type"] = root_cfg.get("robot_type", r0.get("type"))
        env_cfg["robot_names_in_scene"] = [r0.get("name") or r0.get("robot_name_in_scene") or env_cfg["robot_type"]]
    else:
        env_cfg["robot_type"] = root_cfg.get("robot_type", robots[0].get("type"))
        env_cfg["robot_names_in_scene"] = [
            r.get("name") or r.get("robot_name_in_scene") or r.get("type", f"robot{i}")
            for i, r in enumerate(robots[:2])
        ]
        env_cfg["multi_agent"] = True

    # Pass-through optional keys (from either level)
    passthru_keys = [
        "use_camera","camera_key","camera_size","camera_grayscale","frame_stack",
        "success_dist","collision_dist","reward_weights",
        "targets","target_dummies","start_dummies",
        "dynamic_obstacle",
        "defer_camera",
        "force_camera_headless"
    ]
    for k in passthru_keys:
        if k in root_cfg:
            env_cfg[k] = root_cfg[k]
        elif k in merged_task_cfg:
            env_cfg[k] = merged_task_cfg[k]

    return env_cfg

def main():
    ap = argparse.ArgumentParser(description="Universal RL training runner.")
    ap.add_argument("--task_yaml", required=True, help="Path to task YAML (must contain 'env_class').")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--checkpoint_path", help="Checkpoint to restore from.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--train_batch_size", type=int, default=3200)
    ap.add_argument("--sgd_minibatch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--override_env_class", help="Override env_class in YAML.")
    ap.add_argument("--batch_mode", choices=["truncate_episodes","complete_episodes"], default="truncate_episodes")
    ap.add_argument("--rollout_fragment_length", type=int, default=128)
    ap.add_argument("--use_default_curriculum", action="store_true")
    ap.add_argument("--curriculum_offset", type=int, default=0, help="Subtract this from global_iter for curriculum stage selection.")
    args = ap.parse_args()

    if not os.path.exists(args.task_yaml):
        log.error(f"Task YAML not found: {args.task_yaml}")
        sys.exit(1)

    with open(args.task_yaml, "r") as f:
        root_cfg = yaml.safe_load(f) or {}

    # Promote nested keys (if any)
    promote_nested_keys(root_cfg)

    env_class_path = args.override_env_class or root_cfg.get("env_class")
    if not env_class_path:
        log.error("env_class missing in YAML (and no --override_env_class given).")
        sys.exit(1)

    env_class = import_env_class(env_class_path)
    env_config = build_env_config(root_cfg, args.headless)

    log.info(f"Launching training env_class={env_class_path} multi_agent={env_config.get('multi_agent', False)}")

    curriculum_fn = None
    if args.use_default_curriculum:
        # 3-stage plan across GLOBAL PPO iterations (with optional offset)
        def default_curriculum(global_iter: int):
            g = max(0, int(global_iter) - int(args.curriculum_offset))
            # Stage A (0-400): easier success, obstacle hidden
            if g < 400:
                return {
                    "success_dist": 1.0,
                    "enable_obstacle_cfg": True,
                    "obstacle_prob": 0.0,
                    "dual_stagnation_limit": 240,
                }
            # Stage B (400-900): tighten and reintroduce obstacle
            if g < 900:
                return {
                    "success_dist": 0.8,
                    "enable_obstacle_cfg": True,
                    "obstacle_prob": 0.5,
                    "dual_stagnation_limit": 200
                }
            # Stage C (900+): longest horizon
            return {
                "success_dist": 0.7,
                "dual_stagnation_limit": 240
            }
        curriculum_fn = default_curriculum

    trainer = RLTrainer(
        env_class=env_class,
        env_config=env_config,
        train_batch_size=args.train_batch_size,
        sgd_minibatch_size=args.sgd_minibatch_size,
        lr=args.lr,
        curriculum_fn=curriculum_fn,
        batch_mode=args.batch_mode,
        rollout_fragment_length=args.rollout_fragment_length
    )

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        log.info(f"Restoring from checkpoint: {args.checkpoint_path}")
        trainer.restore(args.checkpoint_path)

    trainer.train(iterations=args.iterations)
    log.info("Training complete.")

if __name__ == "__main__":
    main()