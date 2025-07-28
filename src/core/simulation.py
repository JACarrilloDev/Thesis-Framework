from pyrep import PyRep
from pyrep.objects.shape import Shape
import os
import json

# Optional: Import and set up a logger for this module
# from .logger import setup_logger
# sim_logger = setup_logger('simulation_logger', 'logs/simulation.log', console_output=True)
# If you uncomment the above, replace print() calls with sim_logger.info(), sim_logger.error(), etc.

class Simulation:
    def __init__(self, scene_file: str, headless: bool = False):
        self.pr = PyRep()
        self.scene_file = scene_file
        self.headless = headless
        self.robot_instance = None # Will hold the actual robot object from PyRep
        # if sim_logger:
        #     sim_logger.info(f"Simulation instance created for scene: {scene_file}, headless: {headless}")
        # else:
        print(f"Simulation instance created for scene: {scene_file}, headless: {headless}")


    def import_environment(self):
        """Launch the CoppeliaSim scene."""
        if not os.path.exists(self.scene_file):
            message = f"Scene file '{self.scene_file}' not found."
            # if sim_logger: sim_logger.error(message)
            raise FileNotFoundError(message)
        try:
            self.pr.launch(self.scene_file, headless=self.headless)
            message = f"Environment '{self.scene_file}' imported successfully (headless={self.headless})."
            # if sim_logger: sim_logger.info(message)
            # else: print(message)
            print(message) # Using print for now as logger is optional
        except Exception as e:
            message = f"Failed to launch PyRep simulation with scene '{self.scene_file}': {e}"
            # if sim_logger: sim_logger.error(message, exc_info=True)
            raise RuntimeError(message) from e

    def start(self):
        """Start the simulation."""
        try:
            self.pr.start()
            message = "Simulation started."
            # if sim_logger: sim_logger.info(message)
            # else: print(message)
            print(message)
        except Exception as e:
            message = f"Failed to start simulation: {e}"
            # if sim_logger: sim_logger.error(message, exc_info=True)
            raise RuntimeError(message) from e

    def pause(self):
        """Pause the simulation."""
        try:
            self.pr.stop() # In PyRep, stop() effectively pauses the simulation clock.
            message = "Simulation paused."
            # if sim_logger: sim_logger.info(message)
            # else: print(message)
            print(message)
        except Exception as e:
            message = f"Failed to pause simulation: {e}"
            # if sim_logger: sim_logger.error(message, exc_info=True)
            raise RuntimeError(message) from e

    def step(self):
        """Step the simulation forward."""
        try:
            self.pr.step()
        except Exception as e: # Catch a more general exception if PyRep step can fail
            message = f"Failed to step simulation: {e}"
            # if sim_logger: sim_logger.warning(message, exc_info=True)
            # else: print(f"Warning: {message}")
            print(f"Warning: {message}") # Or re-raise depending on desired behavior

    def shutdown(self):
        """Shutdown the simulation and the PyRep connection."""
        try:
            if self.pr.running: # Check if simulation is running before trying to stop
                self.pr.stop()
            self.pr.shutdown()
            message = "Simulation and PyRep connection shutdown."
            # if sim_logger: sim_logger.info(message)
            # else: print(message)
            print(message)
        except Exception as e:
            message = f"Error during simulation shutdown: {e}"
            # if sim_logger: sim_logger.warning(message, exc_info=True)
            # else: print(f"Warning: {message}")
            print(f"Warning: {message}")

    def train_ai(self, steps=100):
        """Example AI training loop. Assumes self.robot_instance is loaded and is an arm."""
        if self.robot_instance is None or not hasattr(self.robot_instance, 'set_joint_target_velocities'):
            message = "Robot not loaded or does not support set_joint_target_velocities. Cannot train_ai."
            # if sim_logger: sim_logger.error(message)
            # else: print(message)
            print(message)
            return

        # This target is conceptual; in a real scenario, it would be defined in the scene
        # or its properties would be dynamically obtained.
        # target_object_name = 'target' # Example name of a target object in your scene
        # target = Shape(target_object_name)
        # if not target.exists():
        #     print(f"Warning: Target object '{target_object_name}' not found in scene for train_ai.")
        
        print(f"Starting example AI training loop for {steps} steps...")
        for step_num in range(steps):
            # Example: Simple action - move joints slightly
            # Ensure the robot has get_joint_count method
            if hasattr(self.robot_instance, 'get_joint_count'):
                num_joints = self.robot_instance.get_joint_count()
                self.robot_instance.set_joint_target_velocities([0.01] * num_joints)
            else:
                print("Warning: Robot instance does not have get_joint_count method.")
                # Apply a generic step if no joint control is available or appropriate
            
            self.step() # Advance simulation
            # if (step_num + 1) % 10 == 0: # Print progress every 10 steps
            #     print(f"AI Training Step {step_num + 1}/{steps} completed.")
        print(f"Example AI training loop finished after {steps} steps.")
    
    def reset(self):
        """Reset the simulation by stopping and starting it.
           Note: This is a basic reset. True object state reset might need more specific logic.
        """
        message_prefix = "Attempting to reset simulation."
        # if sim_logger: sim_logger.info(message_prefix)
        # else: print(message_prefix)
        print(message_prefix)
        try:
            if self.pr.running:
                self.pr.stop()
            # TODO: Implement logic to reset positions of key dynamic objects if needed.
            # This often involves iterating through scene objects and setting their initial poses.
            self.pr.start()
            message = "Simulation reset (simulation clock stopped and started)."
            # if sim_logger: sim_logger.info(message)
            # else: print(message)
            print(message)
        except Exception as e:
            message = f"Failed to reset simulation: {e}"
            # if sim_logger: sim_logger.error(message, exc_info=True)
            raise RuntimeError(message) from e

    def export_data(self, output_path="logs/simulation_data.json"):
        """Export simulation data (e.g., robot pose) to a JSON file."""
        if self.robot_instance is None:
            message = "Cannot export data: Robot not loaded or instance not available."
            # if sim_logger: sim_logger.warning(message)
            # else: print(f"Warning: {message}")
            print(f"Warning: {message}")
            return

        message_export = f"Exporting simulation data to {output_path}."
        # if sim_logger: sim_logger.info(message_export)
        # else: print(message_export)
        print(message_export)
        
        data_to_export = {}
        try:
            # Ensure the robot_instance has these methods; common for PyRep objects
            if hasattr(self.robot_instance, 'get_name'):
                data_to_export["robot_name"] = self.robot_instance.get_name()
            
            if hasattr(self.robot_instance, 'get_position'):
                data_to_export["robot_position"] = self.robot_instance.get_position().tolist()
            
            if hasattr(self.robot_instance, 'get_orientation'):
                data_to_export["robot_orientation"] = self.robot_instance.get_orientation().tolist()
            
            # Add more data as needed, e.g., target positions, task status

            # Ensure the directory for the output file exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_path, "w") as f:
                json.dump(data_to_export, f, indent=4)
            
            message_success = f"Simulation data successfully exported to {output_path}."
            # if sim_logger: sim_logger.info(message_success)
            # else: print(message_success)
            print(message_success)

        except Exception as e:
            message_fail = f"Failed to export simulation data: {e}"
            # if sim_logger: sim_logger.error(message_fail, exc_info=True)
            # else: print(f"Error: {message_fail}")
            print(f"Error: {message_fail}")


    def run_episode(self, task_manager, max_steps=100):
        """
        Run a single episode, typically for reinforcement learning or task execution.
        Args:
            task_manager: An instance of TaskManager (or similar logic provider).
            max_steps: Maximum number of simulation steps for this episode.
        """
        if self.robot_instance is None:
            message = "Cannot run episode: Robot not loaded."
            # if sim_logger: sim_logger.error(message)
            # else: print(message)
            print(message)
            return
        if task_manager is None:
            message = "Cannot run episode: TaskManager not provided."
            # if sim_logger: sim_logger.error(message)
            # else: print(message)
            print(message)
            return

        print(f"Starting new episode. Max steps: {max_steps}")
        # task_manager.reset_current_task() # Ensure task is reset for the episode

        for step_num in range(max_steps):
            # Task manager dictates actions based on current goal
            # This is a simplified interaction. RobotController would typically be involved.
            # task_manager.execute_task_step(self.robot_instance) # Pass robot to task manager
            
            # For now, let's assume task_manager has a method that returns an action
            # or directly manipulates the robot via a RobotController.
            # This part needs to align with your TaskManager and RobotController design.
            # Example: if task_manager.get_next_action(): action.execute(self.robot_instance)
            
            self.step() # Advance simulation
            # print(f"Episode step {step_num + 1}/{max_steps} completed.")

            # Check if task is complete (logic within task_manager)
            # if task_manager.is_task_complete():
            #     print(f"Task completed in {step_num + 1} steps during episode.")
            #     break
        else: # If loop finished without break (max_steps reached or task not completed)
            print(f"Episode ended after {max_steps} steps.")


if __name__ == "__main__":
    # This example usage is for testing the Simulation class directly.
    # Ensure paths are correct if you run this as `python3 src/core/simulation.py`
    # from the Framework root directory.
    
    # Create logs directory if it doesn't exist (for export_data default path)
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Use a .ttt scene file for more reliable direct loading with PyRep.
    # Replace with a valid .ttt scene from your 'examples' or 'user_workspace'
    # example_scene = "examples/scenes/scene_panda_reach_target.ttt" # Make sure this path is valid
    example_scene = "../../examples/simple_reach/scene_panda_reach_target.ttt" # Relative to src/core/
    
    # Check if the example scene file exists from the perspective of this script's location
    # If running `python3 src/core/simulation.py` from `Framework/`, then `example_scene`
    # should be `examples/simple_reach/scene_panda_reach_target.ttt`
    
    # For robust path handling when running from project root:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    scene_for_testing = os.path.join(project_root, "examples", "simple_reach", "scene_panda_reach_target.ttt")


    if not os.path.exists(scene_for_testing):
        print(f"Test scene file not found: {scene_for_testing}. Skipping Simulation example.")
    else:
        print(f"Using test scene: {scene_for_testing}")
        sim_instance = Simulation(scene_file=scene_for_testing, headless=False)
        try:
            sim_instance.import_environment()
            sim_instance.start()

            sim_instance.reset()
            print("Running a few more steps after reset...")
            for _ in range(10):
                sim_instance.step()
            
        except Exception as e_main:
            print(f"An error occurred during Simulation example: {e_main}")
        finally:
            print("Shutting down simulation from __main__ block.")
            sim_instance.shutdown()