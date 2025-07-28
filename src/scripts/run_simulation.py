import argparse
import os
import sys
from core.simulation import Simulation
from core.task_manager import TaskManager

def main():
    parser = argparse.ArgumentParser(description='Run a simulation with a specified environment and task.')
    parser.add_argument('--env', required=True, help='Path to the environment file (.blend/.fbx/.obj)')
    parser.add_argument('--task', required=True, help='Path to the task configuration file (.yaml)')
    
    args = parser.parse_args()

    if not os.path.exists(args.env):
        print(f"Error: Environment file '{args.env}' does not exist.")
        sys.exit(1)

    if not os.path.exists(args.task):
        print(f"Error: Task file '{args.task}' does not exist.")
        sys.exit(1)

    # Initialize the simulation
    simulation = Simulation()
    simulation.import_environment(args.env)

    # Initialize the task manager
    task_manager = TaskManager()
    task_manager.load_task(args.task)

    # Start the simulation
    simulation.start()

    try:
        while True:
            simulation.step()
    except KeyboardInterrupt:
        print("Simulation paused. Exiting...")
        simulation.pause()
        simulation.export_data()

if __name__ == '__main__':
    main()