# Utilities for Environment Exporting

This directory contains utility scripts that facilitate the export of 3D environments from Blender and Unity into formats compatible with CoppeliaSim.

## Contents

- **blender_exporter.py**: This script provides functions to export Blender scenes, ensuring that collision tags are correctly handled for use in CoppeliaSim.

- **unity_exporter.py**: This script offers functionality to export Unity scenes, allowing for seamless integration with CoppeliaSim.

## Usage

To utilize these utilities, you can import the respective exporter in your Python scripts and call the appropriate functions to convert your 3D environments into the required formats.

For example, to export a Blender scene, you would use:

```python
from utils.blender_exporter import export_scene

export_scene('path_to_your_blender_file.blend')
```

Ensure that you have the necessary dependencies installed as listed in the `requirements.txt` file.