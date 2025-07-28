# AI Robotics Framework with CoppeliaSim and Python

## **Project Goal**

Create a Python framework that integrates with CoppeliaSim for simulating multifunctional robots (prebuilt by me) in user-defined environments (imported from Blender/Unity). Users can:

1. Import 3D environments into CoppeliaSim.
2. Assign tasks to preconfigured robots via YAML files.
3. Train AI agents using reinforcement learning (RLlib/Stable-Baselines3).
4. Pause/resume simulations and export data/models.

## **Directory Structure**

```bash
ai_robotics_framework/
├── docs/                       # Guides for users
│   ├── ENVIRONMENT_IMPORT.md
│   └── TASK_CONFIGURATION.md
├── src/
│   ├── robots/                # Prebuilt CoppeliaSim robots (.ttm)
│   ├── tasks/                 # Task templates (.yaml)
│   ├── utils/                 # Blender/Unity integration tools for environments in .fbx or .obj
│   │   ├── blender_exporter.py
│   │   └── unity_exporter.py
│   ├── core/                  # Framework logic
│   │   ├── simulation.py      # CoppeliaSim interface
│   │   ├── robot_controller.py # Robot sensor/motor APIs
│   │   └── task_manager.py    # Parses YAML tasks
│   └── scripts/               # CLI entry points
│       └── run_simulation.py
├── examples/                  # Demo environments/tasks
│   ├── warehouse_sorting/
│   │   ├── warehouse.blend    # Blender scene
│   │   └── sorting_task.yaml
├── requirements.txt           # Python dependencies
└── README.md                  # Setup/usage guide
```

## **Key files**

1. simulation.py: Manages CoppeliaSim connections using PyRep. Its methods could include import_environment(), start(), pause(), export_data().

2. robot_controller.py: Controls prebuilt robots (wheels, grippers, sensors). Its methods could include get_camera_image(), move_wheels(), gripper_open().

3. tasks/sorting_task.yaml: Defines goals (e.g., "move 5 red boxes to Zone A").

4. utils/blender_exporter.py: Converts Blender scenes to a CoppeliaSim format with collision tags.

## **Getting Started**

For the dependencies, there could be in the requirements.txt some like PyRep, ray[rllib], opencv, pyyaml, and more necessary.

For running demos, it could work like python src/scripts/run_simulation.py --env examples/warehouse_sorting/warehouse.blend --task sorting_task.yaml

With this custom worflow, environments can be designed in Blender/Unity and expored as .blend/.fbx/.obj. New objectives can be defined by editing tasks/\*.yaml.

CoppeliaSim fits in by being open during the simulations to visualize robots and environments, being able to be integrated with PyRep for real-time control (maybe even use pr.step() for physics steps) and access camera feeds.
