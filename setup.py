from setuptools import setup, find_packages

setup(
    name="ai_robotics_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyRep",
        "ray[rllib]",
        "torch",
        "gym",
        "numpy",
        "pyyaml",
        "opencv-python",
        "opencv-python-headless",
        "matplotlib",
        "imageio",
    ],
    entry_points={
        "console_scripts": [
            "run_simulation=src.scripts.run_simulation:main",
            "stop_simulation=src.scripts.stop_simulation:main",
            "pause_simulation=src.scripts.pause_simulation:main",
            "resume_simulation=src.scripts.resume_simulation:main",
            "add_robot=src.scripts.add_robot:main",
            "export_data=src.scripts.export_data:main",
            "load_environment=src.scripts.load_environment:main",
        ],
    },
)