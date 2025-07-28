# Example Environment and Task for Warehouse Sorting

This directory contains an example environment and task for simulating a warehouse sorting scenario using the AI Robotics Framework with CoppeliaSim.

## Contents

- **warehouse.blend**: A Blender scene file representing the warehouse environment.
- **sorting_task.yaml**: A YAML file that defines the sorting task for the robots, including specific goals and parameters.

## Getting Started

To run the warehouse sorting simulation, follow these steps:

1. **Ensure Dependencies are Installed**: Make sure you have all the necessary Python dependencies listed in `requirements.txt` installed.

2. **Open CoppeliaSim**: Launch CoppeliaSim and ensure it is ready to accept connections from the framework.

3. **Run the Simulation**: Use the command line to execute the simulation script with the following command:

   ```bash
   python src/scripts/run_simulation.py --env examples/warehouse_sorting/warehouse.blend --task sorting_task.yaml
   ```

4. **Observe the Simulation**: Watch as the robots perform the sorting task in the provided warehouse environment.

## Customization

You can customize the warehouse environment by modifying the `warehouse.blend` file in Blender. Additionally, you can change the sorting task parameters by editing the `sorting_task.yaml` file to define new goals or modify existing ones.

## Additional Resources

Refer to the documentation in the `docs` directory for more information on importing environments and configuring tasks.