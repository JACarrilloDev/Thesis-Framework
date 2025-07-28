from pyrep.robots.arms import Arm
from pyrep.errors import ConfigurationPathError

class PathPlanner:
    def __init__(self, robot: Arm):
        self.robot = robot

    def compute_path(self, target_position, target_orientation):
        try:
            return self.robot.get_path(position=target_position, euler=target_orientation)
        except ConfigurationPathError:
            print("Path planning failed.")
            return None