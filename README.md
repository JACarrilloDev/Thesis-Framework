# AI Robotics Framework (CoppeliaSim + PyRep + RL)

## 1. Overview
The framework enables simulation and reinforcement learning control of multifunctional robots inside CoppeliaSim. Users can:
- Import or create 3D scenes (Blender/Unity → CoppeliaSim `.ttt`)
- Configure tasks and training via YAML
- Train single or multi‑agent RL policies (Ray RLlib)
- Extend with custom environments without modifying core code

Core example environments include single‑robot navigation and a two‑robot cooperative target reaching scenario implemented in [`DynamicTwoPhaseNavEnv`](examples/multirobot/envs/multirobot_env.py).

## 2. Core Features
- PyRep wrapper for fast reset / step cycles
- Robot abstraction (sensors, wheels, grippers) via `RobotController`
- Multi‑agent RL (shared scene, coordination logic)
- Universal training runner [`run_rl_task.py`](run_rl_task.py)
- Potential‑based & shaped rewards (distance progress, heading alignment, velocity alignment, idle/stuck handling)
- User workspace isolation (`user_workspace/`) for custom additions
- Checkpoint + metric logging (`logs/`, `checkpoints/`)

## 3. Repository Structure (Condensed)
```
.
├── run_rl_task.py                 # Universal RL training runner
├── examples/                      # Example envs, scenes & task YAMLs
│   └── multirobot/envs/multirobot_env.py
├── src/
│   ├── core/                      # Core simulation / control
│   ├── robots/                    # Robot model metadata / helpers
│   ├── scripts/                   # (Legacy) simulation scripts
│   └── utils/                     # Export / utility tools
├── user_workspace/                # User custom envs/scenes/tasks (safe sandbox)
├── docs/                          # Guides (basic + advanced)
├── logs/                          # Training & runtime logs
└── checkpoints/                   # Saved RL checkpoints
```

## 4. Quick Start
```bash
git clone <repository-url> ai_robotics_framework
cd ai_robotics_framework
pip install -r requirements.txt
# (Optional) export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 run_rl_task.py --task_yaml examples/multirobot/tasks/multirobot.yaml --iterations 300
```

## 5. Prerequisites
- Linux (or WSL2) – PyRep requirement
- CoppeliaSim installed (GUI optional if using `--headless`)
- Python 3.9–3.11 recommended
- GPU optional (policies are small by default)

## 6. Universal Training Runner
File: [`run_rl_task.py`](run_rl_task.py)  
Supports two ways to specify the environment in the task YAML:

Dotted class (preferred):
```yaml
env_class: examples.multirobot.envs.multirobot_env.DynamicTwoPhaseNavEnv
```

File fallback:
```yaml
env_file: user_workspace/custom_envs/my_nav_env.py
env_class_name: MyCustomNavEnv
```

Run:
```bash
python3 run_rl_task.py --task_yaml examples/multirobot/tasks/multirobot.yaml --iterations 500
python3 run_rl_task.py --task_yaml examples/navigation/tasks/navigation_easy.yaml --iterations 300
```
Common flags:
- `--headless`
- `--checkpoint_path <dir_or_file>`
- `--train_batch_size 16000`
- `--override_env_file ...` (override YAML)

## 7. Task YAML Essentials
Example (multi‑robot) snippet:
```yaml
scene_file: examples/multirobot/scenes/multi_nav.ttt
env_class: examples.multirobot.envs.multirobot_env.DynamicTwoPhaseNavEnv
robots_setup:
  - { type: "AstiPioneerHybrid", name: "AstiPioneer1" }
  - { type: "AstiPioneerHybrid", name: "AstiPioneer2" }
max_episode_steps: 500
success_dist: 0.40
reward_weights:
  progress: 5.0
  completion: 25.0
dynamic_obstacle:
  enabled: true
  name: MidObstacle
```

Single‑robot navigation minimal:
```yaml
scene_file: examples/navigation/scenes/navigation_easy.ttt
env_class: src.core.navigation_env.NavigationEnv
robots_setup:
  - { type: "AstiPioneerHybrid", name: "AstiPioneer1" }
max_episode_steps: 360
success_dist: 0.2
```

## 8. Reward Shaping (Two‑Robot Environment)
[`DynamicTwoPhaseNavEnv`](examples/multirobot/envs/multirobot_env.py) combines:
- Distance progress (clipped delta; configurable `w_progress`)
- Potential‑based shaping Φ = −α·dist with discount γ (policy invariant)
- Adaptive idle + stuck penalties (distance‑scaled thresholds)
- Heading alignment bonus
- Velocity alignment: forward velocity projected onto target direction (`vel_align_scale`)
- Spin penalty for rotational dithering
- Near‑target ramp shaping: continuous bonus inside `near_target_radius_mult*success_dist`
- Collision penalty
Parameters configurable via `reward_weights` and shaping keys (e.g. `shaping_gamma`, `idle_warmup_steps`, `spin_penalty_scale`).

## 9. Multi‑Agent Phase Logic
Phase 1: Each robot gets one of 4 targets (random distinct).  
Phase 2: First to finish picks the nearer of the two remaining; the other robot gets the last target.  
Episode ends when both have completed two targets or step limit reached.

## 10. User Workspace Workflow
Users create or extend without editing core:
```
user_workspace/
  custom_envs/        # Custom Gym / MultiAgent env classes
  custom_tasks/       # YAML task definitions
  custom_scenes/      # Imported .ttt scenes
```
Reference them via:
```yaml
scene_file: user_workspace/custom_scenes/arena_variant.ttt
env_file: user_workspace/custom_envs/my_nav_env.py
env_class_name: MyNavEnv
```

## 11. Creating a Custom Environment (Template)
See full template in advanced guide or copy minimal form:
```python
class MyNavEnv(gym.Env):
    def __init__(self, env_config): ...
    def reset(self, *, seed=None, options=None): ...
    def step(self, action): ...
```

## 12. Logging & Checkpoints
- Logs: `logs/robot_controller.log`, `logs/rl_training.log`, custom run logs created through `setup_logger` in [`src/core/logger.py`](src/core/logger.py)
- Metrics CSV: `logs/training_metrics.csv`
- Checkpoints: `checkpoints/<date_or_run_id>/`
Restore:
```bash
python3 run_rl_task.py --task_yaml ... --checkpoint_path checkpoints/run_123
```

## 13. Extending Robots
Add models (.ttm) under `src/robots/` (or custom folder) and corresponding definitions in your robot definition factory helper (see existing definitions referenced in controller construction).

## 14. Exporting / Deployment (Future)
Planned:
- ONNX export for trained policies
- Curriculum scheduling hook in runner
- Domain randomization pack (sensor noise, friction)

## 15. Troubleshooting (Condensed)
| Issue | Fix |
|-------|-----|
| PyRep import failure | Confirm running on Linux & CoppeliaSim path set if needed |
| env_class ImportError | Add missing `__init__.py` or use `env_file` / `env_class_name` |
| Object type mismatch (Dummy vs Shape) | Use unified resolver in updated `RobotController.set_object_pose` |
| Low reward progression early | Increase `w_progress`, reduce `idle_warmup_steps`, verify velocity scaling |

More details: see advanced guide.

## 16. Contributing
1. Fork & branch
2. Add tests/examples if applicable
3. Keep docs updated (`README.md` + advanced guide)
4. Submit PR

## 17. License
MIT (see `LICENSE`)

---
For deeper technical notes (potential shaping math, packaging migration, pitfalls) consult `docs/ADVANCED_GUIDE.md`.