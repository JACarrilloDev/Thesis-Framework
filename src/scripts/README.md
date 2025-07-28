# Scripts README

This directory contains scripts that serve as command-line interface (CLI) entry points for running simulations within the AI Robotics Framework.

## Available Scripts

### run_simulation.py

- **Purpose**: This script is used to initiate simulations with specified environments and tasks.
- **Usage**: 
  To run a simulation, use the following command:
  ```
  python src/scripts/run_simulation.py --env <path_to_environment> --task <path_to_task>
  ```
  Replace `<path_to_environment>` with the path to your environment file (e.g., `.blend`, `.fbx`, or `.obj`) and `<path_to_task>` with the path to your YAML task file.

## Requirements

Ensure that all necessary dependencies are installed as listed in the `requirements.txt` file before running the scripts.