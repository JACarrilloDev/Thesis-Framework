import json
from pyrep import PyRep
from pyrep.objects.robot import Robot

def export_data(output_path="simulation_data.json"):
    pr = PyRep()
    robot = Robot("Panda")  # Replace with your robot name
    data = {
        "robot_position": robot.get_position(),
        "robot_orientation": robot.get_orientation(),
    }
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Simulation data exported to {output_path}.")

if __name__ == "__main__":
    export_data()