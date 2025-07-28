import argparse
import os
import sys
import time
from pyrep import PyRep
from src.core.logger import setup_logger # Assuming logger.py is in src/core

# Setup logger for this script
load_env_logger = setup_logger('load_environment_logger', 'logs/load_environment.log')

def load_environment(scene_filepath: str, headless: bool = False):
    """
    Launches CoppeliaSim and loads the specified scene file.

    Args:
        scene_filepath (str): Path to the scene file (.ttt, .fbx, .obj, .stl, .dae, .blend).
                              Note: .blend files might require Blender to be installed and configured
                              for background conversion if not directly supported by CoppeliaSim's importers.
                              Direct .ttt and .fbx are most reliable.
        headless (bool): Run CoppeliaSim in headless mode.
    """
    if not os.path.exists(scene_filepath):
        load_env_logger.error(f"Scene file not found: {scene_filepath}")
        print(f"Error: Scene file not found: {scene_filepath}")
        sys.exit(1)

    pr = PyRep()
    file_extension = os.path.splitext(scene_filepath)[1].lower()

    try:
        load_env_logger.info(f"Attempting to launch CoppeliaSim with scene: {scene_filepath} (headless={headless})")
        
        # For .ttt files, PyRep can launch them directly.
        # For other importable formats (like .fbx), CoppeliaSim usually imports them into an empty scene.
        # PyRep's launch() can take a .ttt file. For others, we launch empty and then import.
        if file_extension == '.ttt':
            pr.launch(scene_filepath, headless=headless)
        else:
            # Launch an empty scene first
            pr.launch("", headless=headless) # Empty string launches an empty scene
            load_env_logger.info(f"Launched empty scene. Attempting to import: {scene_filepath}")
            # Attempt to import the scene. This works well for .fbx, .obj etc.
            # For .blend, CoppeliaSim's internal importers might be used.
            pr.import_model(scene_filepath) 
            load_env_logger.info(f"Successfully imported scene: {scene_filepath}")

        pr.start()
        load_env_logger.info("Simulation started. CoppeliaSim should be open with the environment.")
        print(f"CoppeliaSim launched with '{os.path.basename(scene_filepath)}'.")
        print("Press Ctrl+C in this terminal to close this PyRep connection and allow CoppeliaSim to be closed (if not headless).")

        # Keep the script running so CoppeliaSim stays open and responsive
        while True:
            pr.step() # Keep stepping the simulation
            time.sleep(0.01) # Small delay

    except ImportError as e: # Specifically catch import errors for non-standard formats if pr.import_scene fails
        load_env_logger.error(f"Failed to import scene '{scene_filepath}'. It might be an unsupported format or corrupted: {e}", exc_info=True)
        print(f"Error: Could not import scene '{scene_filepath}'. Check format and CoppeliaSim compatibility. Error: {e}")
    except RuntimeError as e:
        load_env_logger.error(f"Runtime error during CoppeliaSim launch or scene load: {e}", exc_info=True)
        print(f"Error: A runtime error occurred with CoppeliaSim: {e}")
    except Exception as e:
        load_env_logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
    except KeyboardInterrupt:
        load_env_logger.info("User interrupted. Shutting down PyRep connection.")
        print("\nUser interrupted. Shutting down PyRep connection...")
    finally:
        if pr.running:
            pr.stop()
        pr.shutdown()
        load_env_logger.info("PyRep connection closed.")
        print("PyRep connection closed. You can now close CoppeliaSim.")

def main():
    parser = argparse.ArgumentParser(description="Load a 3D environment into CoppeliaSim.")
    parser.add_argument("--scene_file", required=True, help="Path to the scene file (e.g., .ttt, .fbx, .blend).")
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim in headless mode.")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    load_environment(args.scene_file, args.headless)

if __name__ == "__main__":
    main()