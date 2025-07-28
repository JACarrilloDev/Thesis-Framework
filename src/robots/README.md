# Prebuilt Robots in the AI Robotics Framework

This directory contains prebuilt robot models designed for use within the AI Robotics Framework. These robots are compatible with CoppeliaSim and can be utilized in various simulation tasks.

## Available Robots

1. **Robot1.ttm**: A multifunctional robot equipped with wheels and a gripper, suitable for tasks such as object manipulation and navigation.

2. **Robot2.ttm**: A mobile robot designed for exploration and data collection, featuring advanced sensors for environmental interaction.

3. **Robot3.ttm**: A robotic arm optimized for precision tasks, capable of performing complex movements and handling delicate objects.

## Usage

To use the prebuilt robots in your simulations:

1. Ensure that the robot model files (.ttm) are placed in this directory.
2. Configure the desired tasks in the corresponding YAML files located in the `tasks` directory.
3. Use the `robot_controller.py` to interact with the robots during simulations. This includes controlling movements, accessing sensor data, and executing predefined tasks.

## Integration with CoppeliaSim

The robots can be imported into CoppeliaSim using the framework's simulation management tools. Ensure that CoppeliaSim is running and connected to the framework for real-time control and monitoring.

For detailed instructions on task configuration and environment import, please refer to the documentation in the `docs` directory.