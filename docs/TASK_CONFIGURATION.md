# TASK_CONFIGURATION.md

# Task Configuration Guide

This document explains how to configure tasks for the robots using YAML files in the AI Robotics Framework. The tasks define the objectives that the robots will perform in the simulated environments.

## YAML Task Structure

A typical YAML task file consists of the following sections:

1. **task_name**: A unique identifier for the task.
2. **description**: A brief explanation of what the task entails.
3. **goals**: A list of objectives that the robot needs to achieve.
4. **parameters**: Additional settings or parameters that may be required for the task.

### Example Task YAML

Here is an example of a YAML task file for a sorting task:

```yaml
task_name: sorting_task
description: Sort red boxes into Zone A and blue boxes into Zone B.
goals:
  - move: 
      object: red_box
      target_zone: Zone A
      quantity: 5
  - move: 
      object: blue_box
      target_zone: Zone B
      quantity: 3
parameters:
  max_time: 300  # Maximum time allowed for the task in seconds
  robot_id: robot1  # Identifier for the robot executing the task
```

## Defining Goals

Each goal can specify different actions for the robot. Common actions include:

- **move**: Move an object from one location to another.
- **pick**: Pick up an object.
- **drop**: Drop an object at a specified location.

### Goal Parameters

Each action may have specific parameters. For example, the `move` action requires:

- `object`: The type of object to move (e.g., red_box).
- `target_zone`: The destination zone for the object (e.g., Zone A).
- `quantity`: The number of objects to move.

## Using Task YAML Files

To use a task YAML file in your simulation, ensure it is placed in the `src/tasks/` directory. You can then reference it in your simulation script or through the command line.

### Running a Simulation with a Task

You can run a simulation with a specific task using the following command:

```bash
python src/scripts/run_simulation.py --env examples/warehouse_sorting/warehouse.blend --task sorting_task.yaml
```

This command will load the specified environment and execute the defined task using the configured robot.

## Conclusion

This guide provides an overview of how to configure tasks for robots using YAML files. By following the structure outlined above, you can create custom tasks tailored to your simulation needs.