# src/tasks/README.md

# Task Configuration for AI Robotics Framework

This directory contains YAML files that define tasks for the multifunctional robots in the AI Robotics Framework. Each task file specifies the objectives that the robots need to achieve within the simulated environments.

## Structure of Task YAML Files

A typical task YAML file includes the following sections:

- **goal**: A description of the task objective.
- **parameters**: Specific parameters that may be required for the task execution.
- **robot**: The identifier of the robot assigned to the task.
- **environment**: The environment in which the task will be executed.

### Example Task YAML File

Here is an example of a sorting task YAML file:

```yaml
goal: "Move 5 red boxes to Zone A"
parameters:
  box_color: "red"
  number_of_boxes: 5
robot: "robot_1"
environment: "warehouse"
```

## Adding New Tasks

To add a new task:

1. Create a new YAML file in this directory.
2. Follow the structure outlined above.
3. Ensure that the task is compatible with the robots and environments defined in the framework.

## Usage

Tasks defined in YAML files can be loaded and executed by the `task_manager.py` module in the core directory. You can specify the task file when running simulations through the command-line interface.

For example:

```bash
python src/scripts/run_simulation.py --env examples/warehouse_sorting/warehouse.blend --task sorting_task.yaml
```

This command will load the specified environment and execute the defined task using the appropriate robot.