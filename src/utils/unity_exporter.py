# Unity Exporter Script

import os
import json
import subprocess

def export_unity_scene(scene_path, output_path):
    """
    Exports a Unity scene to a format compatible with CoppeliaSim.
    
    Parameters:
        scene_path (str): The path to the Unity scene file.
        output_path (str): The path where the exported file will be saved.
    """
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"The specified scene path does not exist: {scene_path}")

    # Command to export the Unity scene
    command = f"Unity -batchmode -quit -projectPath {os.path.dirname(scene_path)} -executeMethod ExportScene.ExportToCoppeliaSim -outputPath {output_path}"
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully exported Unity scene to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while exporting the scene: {e}")

def main():
    # Example usage
    scene_path = "path/to/your/unity_scene.unity"
    output_path = "path/to/output/exported_scene.json"
    export_unity_scene(scene_path, output_path)

if __name__ == "__main__":
    main()