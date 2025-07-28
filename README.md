# AI Robotics Framework with CoppeliaSim and Python

## Overview

The AI Robotics Framework is designed to facilitate the simulation of multifunctional robots within user-defined environments using CoppeliaSim and Python. This framework allows users to import 3D environments, configure robot tasks, and train AI agents through reinforcement learning.

## Project Structure

The project is organized into several directories, each serving a specific purpose:

- **docs/**: Contains documentation files that provide guidelines on importing environments and configuring tasks.
- **src/**: The source code of the framework, including utilities, core functionalities, and scripts for running simulations.
- **examples/**: Provides example environments and tasks to demonstrate the framework's capabilities.
- **requirements.txt**: Lists the necessary Python dependencies for the project.

## Getting Started

### Prerequisites

Before using the framework, ensure you have the following installed:

- Python 3.x
- CoppeliaSim
- Blender or Unity (for environment creation)
- **Linux or WSL**: PyRep is only supported on Linux. If you're using Windows, set up WSL (Windows Subsystem for Linux) to run the framework.

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai_robotics_framework
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

To run a simulation, use the command:
```
python src/scripts/run_simulation.py --env <path_to_environment> --task <path_to_task>
```

For example:
```
python src/scripts/run_simulation.py --env examples/warehouse_sorting/warehouse.blend --task examples/warehouse_sorting/sorting_task.yaml
```

### Documentation

Refer to the documentation files in the **docs/** directory for detailed instructions on importing environments and configuring tasks.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.