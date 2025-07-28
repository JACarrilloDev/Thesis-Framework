# ENVIRONMENT_IMPORT.md

# Importing 3D Environments into CoppeliaSim

This document provides guidelines on how to import 3D environments into CoppeliaSim from Blender or Unity. Follow the steps below to ensure a smooth import process.

## Importing from Blender

1. **Prepare Your Scene**:
   - Create your 3D environment in Blender.
   - Ensure that all objects are properly named and organized in the scene.

2. **Exporting the Scene**:
   - Use the provided `blender_exporter.py` script located in the `src/utils/` directory to export your Blender scene.
   - The script will convert your Blender scene into a format compatible with CoppeliaSim, including handling collision tags.

   Example command:
   ```
   python src/utils/blender_exporter.py --input your_scene.blend --output your_scene.ttm
   ```

3. **Importing into CoppeliaSim**:
   - Open CoppeliaSim and navigate to the 'File' menu.
   - Select 'Import' and choose the exported `.ttm` file.
   - Adjust the scene settings as necessary.

## Importing from Unity

1. **Prepare Your Scene**:
   - Design your environment in Unity, ensuring all objects are correctly configured.

2. **Exporting the Scene**:
   - Use the `unity_exporter.py` script located in the `src/utils/` directory to export your Unity scene.
   - This script will convert your Unity scene into a format that CoppeliaSim can utilize.

   Example command:
   ```
   python src/utils/unity_exporter.py --input your_scene.unity --output your_scene.ttm
   ```

3. **Importing into CoppeliaSim**:
   - Similar to the Blender import process, open CoppeliaSim and select 'File' > 'Import'.
   - Choose the exported `.ttm` file and make any necessary adjustments.

## Tips for Successful Import

- Ensure that all textures and materials are properly linked in Blender or Unity before exporting.
- Check for any errors in the console output of the exporter scripts to troubleshoot issues.
- Test the imported environment in CoppeliaSim to ensure that all elements are functioning as expected.

By following these guidelines, you can successfully import 3D environments into CoppeliaSim for your robotic simulations.