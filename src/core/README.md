# Overview of the Core Functionalities

The core module of the AI Robotics Framework is designed to manage the essential functionalities required for simulating multifunctional robots within CoppeliaSim. This module includes three primary components:

1. **Simulation Management (`simulation.py`)**: This component handles the connection to CoppeliaSim using the PyRep library. It provides methods to import 3D environments, start and pause simulations, and export simulation data.

2. **Robot Control (`robot_controller.py`)**: This file contains the logic for controlling the prebuilt robots. It includes methods for interacting with robot sensors and actuators, such as capturing camera images, moving wheels, and operating grippers.

3. **Task Management (`task_manager.py`)**: This component is responsible for parsing YAML task files and managing the execution of tasks assigned to the robots. It ensures that the robots perform their designated objectives as defined in the task configurations.

## Usage

To utilize the core functionalities of the framework, you will typically follow these steps:

1. **Set up the simulation environment** using the `simulation.py` module to connect to CoppeliaSim and import your desired 3D environment.

2. **Control the robots** through the `robot_controller.py` module, allowing you to execute specific actions based on the tasks defined.

3. **Define and manage tasks** using the `task_manager.py` module, which will read from YAML files to determine the objectives for the robots.

This modular approach allows for flexibility and scalability, enabling users to easily integrate new robots, tasks, and environments into the framework.