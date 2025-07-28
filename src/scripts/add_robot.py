import argparse
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor

def add_robot(robot_file, position):
    pr = PyRep()
    pr.launch(headless=False)
    robot = pr.import_model(robot_file)
    robot.set_position(position)
    pr.start()
    print(f"Robot {robot_file} added at position {position}.")
    pr.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Add a robot to the simulation.")
    parser.add_argument("--robot", required=True, help="Path to the robot .ttm file.")
    parser.add_argument("--position", required=True, help="Initial position (x,y,z).")
    args = parser.parse_args()

    position = tuple(map(float, args.position.split(",")))
    add_robot(args.robot, position)

if __name__ == "__main__":
    main()