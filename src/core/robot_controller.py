from typing import Optional
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.joint import Joint
from pyrep.robots.end_effectors.gripper import Gripper as PyRepGripper
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.camera import Camera
from pyrep.objects.proximity_sensor import ProximitySensor
import numpy as np
import math
import cv2

from .logger import setup_logger

rc_logger = setup_logger('robot_controller_logger', 'logs/robot_controller.log', console_output=True)

class RobotController:
    def __init__(self, robot_definition: dict, pyrep_instance: PyRep):
        self.pr = pyrep_instance
        self.definition = robot_definition
        self.sensor_prefix = self.definition.get("sensor_prefix", "")
        self.robot_name_in_scene = self.definition.get("robot_name_in_scene", "UnknownRobot")
        rc_logger.info(f"Initializing RobotController for: {self.robot_name_in_scene}")

        # Use Shape for the robot model handle
        self.robot_model_handle = Shape(self.robot_name_in_scene)
        if not Shape.exists(self.robot_name_in_scene):
            msg = f"Robot model '{self.robot_name_in_scene}' not found in scene."
            rc_logger.error(msg)
            raise ValueError(msg)

        # Arm components
        self.arm_joint_names = self.definition.get("arm_joints", [])
        self.arm_joint_handles = [Joint(name) for name in self.arm_joint_names]
        for i, name in enumerate(self.arm_joint_names):
            if not Joint.exists(name):
                rc_logger.warning(f"Arm joint '{name}' not found for robot '{self.robot_name_in_scene}'.")

        # End Effector Tip
        self.ee_tip_name = self.definition.get("end_effector_tip_name")
        self.ee_tip_handle = None
        if self.ee_tip_name:
            self.ee_tip_handle = Dummy(self.ee_tip_name)
            if not Dummy.exists(self.ee_tip_name):
                rc_logger.warning(f"End-effector tip '{self.ee_tip_name}' not found.")
                self.ee_tip_handle = None

        # Gripper components
        self.gripper_name = self.definition.get("gripper_name")
        self.gripper_control_joint_name = self.definition.get("gripper_control_joint")
        self.gripper_joint_limits = self.definition.get("gripper_joint_limits", {"closed": 0.0, "open": 0.05})
        self.gripper_handle = None

        if self.gripper_name:
            try:
                self.gripper_handle = PyRepGripper(self.gripper_name)
                if not PyRepGripper.exists(self.gripper_name):
                    self.gripper_handle = None
                    rc_logger.warning(f"PyRep Gripper object '{self.gripper_name}' not found. Will try joint control if defined.")
            except Exception:
                self.gripper_handle = None
                rc_logger.info(f"'{self.gripper_name}' is not a standard PyRep Gripper. Checking for joint control.")

        if self.gripper_handle is None and self.gripper_control_joint_name:
            self.gripper_handle = Joint(self.gripper_control_joint_name)
            if not Joint.exists(self.gripper_control_joint_name):
                rc_logger.warning(f"Gripper control joint '{self.gripper_control_joint_name}' not found or not a Joint.")
                self.gripper_handle = None
            else:
                rc_logger.info(f"Gripper control via joint: '{self.gripper_control_joint_name}'")
        elif self.gripper_handle is not None:
            rc_logger.info(f"Gripper '{self.gripper_name}' initialized as PyRep Gripper object.")

        # Camera components
        self.camera_definitions = self.definition.get("cameras", {})
        self.camera_handles = {}
        for cam_key, cam_def in self.camera_definitions.items():
            cam_name = cam_def.get("name")
            if cam_name:
                if VisionSensor.exists(cam_name):
                    handle = VisionSensor(cam_name)
                    self.camera_handles[cam_key] = handle
                    rc_logger.info(f"VisionSensor '{cam_name}' (key: {cam_key}) initialized.")
                else:
                    rc_logger.warning(f"VisionSensor '{cam_name}' not found.")
            else:
                rc_logger.warning(f"Camera definition for key '{cam_key}' missing 'name'.")

        # Base components (for completeness, but Asti uses script-based walking)
        self.base_actuator_def = self.definition.get("base_actuators", {})
        self.base_wheel_joint_names = self.base_actuator_def.get("wheel_joints", [])
        self.base_wheel_joint_handles = [Joint(name) for name in self.base_wheel_joint_names]
        self.wheel_radius = self.base_actuator_def.get("wheel_radius", 0.05)
        self.base_control_script = self.definition.get("base_control_script")
        self.base_control_function = self.definition.get("base_control_function")

        rc_logger.info(f"RobotController for '{self.robot_name_in_scene}' components initialized.")

    def _resolve_object_handle(self, name: str):
        """Return a PyRep handle (Dummy or Shape) or None. Try Dummy first to avoid type mismatch warnings."""
        try:
            if Dummy.exists(name):
                return Dummy(name)
        except Exception:
            pass
        try:
            if Shape.exists(name):
                return Shape(name)
        except Exception:
            pass
        rc_logger.warning(f"_resolve_object_handle: '{name}' not found as Dummy or Shape.")
        return None

    def get_object_pose(self, object_name: str) -> Optional[list]:
        """Unified getter that works for Shape or Dummy. Returns [x,y,z,qx,qy,qz,qw] or None."""
        handle = self._resolve_object_handle(object_name)
        if not handle:
            return None
        try:
            return handle.get_pose()
        except Exception as e:
            rc_logger.warning(f"get_object_pose failed for '{object_name}': {e}")
            return None

    def set_object_pose(self, object_name: str, pose: list) -> bool:
        """Set pose [x,y,z,qx,qy,qz,qw] of a scene object (Dummy or Shape)."""
        try:
            obj_handle = self._resolve_object_handle(object_name)
            if obj_handle is None:
                rc_logger.warning(f"set_object_pose: object '{object_name}' not found.")
                return False
            if len(pose) == 7:
                obj_handle.set_pose(pose)
            else:
                rc_logger.warning(f"set_object_pose: pose length {len(pose)} invalid (need 7).")
                return False
            return True
        except Exception as e:
            rc_logger.error(f"Error setting pose for '{object_name}': {e}")
            return False

    def set_base_target_velocities(self, linear_velocity_xy: list, angular_velocity_z: float):
        # Only use vx (forward), ignore vy for differential drive
        vx = linear_velocity_xy[0]
        omega_z = angular_velocity_z

        # Get wheel parameters
        wheel_radius = self.base_actuator_def.get("wheel_radius", 0.0975)
        wheel_separation = self.base_actuator_def.get("wheel_separation", 0.31)

        # Differential drive kinematics
        v_left = (vx - (omega_z * wheel_separation / 2.0)) / wheel_radius
        v_right = (vx + (omega_z * wheel_separation / 2.0)) / wheel_radius

        # Set wheel velocities
        if len(self.base_wheel_joint_handles) >= 2:
            self.base_wheel_joint_handles[0].set_joint_target_velocity(v_left)
            self.base_wheel_joint_handles[1].set_joint_target_velocity(v_right)
        else:
            rc_logger.error("Wheel joint handles not properly initialized.")

    def set_arm_joint_target_velocities(self, velocities: list):
        if len(velocities) != len(self.arm_joint_handles):
            rc_logger.warning(f"Arm joint velocities count mismatch. Expected {len(self.arm_joint_handles)}, got {len(velocities)}.")
            return
        for i, handle in enumerate(self.arm_joint_handles):
            if Joint.exists(self.arm_joint_names[i]):
                handle.set_joint_target_velocity(velocities[i])
            else:
                rc_logger.warning(f"Attempted to set velocity for non-existent arm joint: {self.arm_joint_names[i]}")

    def get_proximity_sensor_readings(self):
        readings = []
        for i in range(1, 17):
            # Try prefixed first if provided
            base_name = f"Pioneer_p3dx_ultrasonicSensor{i}"
            candidate = f"{self.sensor_prefix}{i}" if self.sensor_prefix else base_name
            names_to_try = [candidate] if candidate != base_name else [base_name]
            if candidate != base_name:
                names_to_try.append(base_name)
            value = 1.0
            for name in names_to_try:
                try:
                    if ProximitySensor.exists(name):
                        sensor = ProximitySensor(name)
                        dist = sensor.read()
                        value = 1.0 if dist < 0 else dist
                        break
                except Exception:
                    continue
            readings.append(value)
        return readings

    def check_sandwich_grasp(self, object_name: str, threshold: float = 0.05) -> bool:
        # Get positions of both arm tips and the object
        left_tip = self.get_object_pose("leftArmTip")
        right_tip = self.get_object_pose("rightArmTip")
        obj_pos = self.get_object_pose(object_name)
        if left_tip and right_tip and obj_pos:
            # Check if object is between the arms (simple 1D check, expand as needed)
            min_x = min(left_tip[0], right_tip[0])
            max_x = max(left_tip[0], right_tip[0])
            return min_x - threshold < obj_pos[0] < max_x + threshold
        return False                

    def get_gripper_open_amount(self) -> float:
        if isinstance(self.gripper_handle, PyRepGripper):
            return self.gripper_handle.get_open_amount()[0]
        elif isinstance(self.gripper_handle, Joint):
            current_pos = self.gripper_handle.get_joint_position()
            closed_pos = self.gripper_joint_limits["closed"]
            open_pos = self.gripper_joint_limits["open"]
            if (open_pos - closed_pos) == 0:
                return 0.0
            return np.clip((current_pos - closed_pos) / (open_pos - closed_pos), 0.0, 1.0)
        rc_logger.warning("No valid gripper handle for get_gripper_open_amount.")
        return 0.0

    def check_grasp(self, object_to_check_name: str) -> bool:
        if not Shape.exists(object_to_check_name):
            rc_logger.warning(f"Object '{object_to_check_name}' not found for grasp check.")
            return False
        obj_handle = Shape(object_to_check_name)

        if isinstance(self.gripper_handle, PyRepGripper):
            return self.gripper_handle.grasp(obj_handle)
        elif isinstance(self.gripper_handle, Joint):
            is_closed_enough = self.get_gripper_open_amount() < 0.1
            if not is_closed_enough:
                return False
            if self.ee_tip_handle:
                dist_to_tip = np.linalg.norm(np.array(obj_handle.get_position()) - np.array(self.ee_tip_handle.get_position()))
                if dist_to_tip < 0.05:
                    rc_logger.debug(f"Object '{object_to_check_name}' is close and gripper is closed. Assuming grasp (simplified).")
                    return True
            rc_logger.debug(f"Grasp check for '{object_to_check_name}' (joint gripper): not close enough or EE tip not found.")
            return False
        rc_logger.warning(f"Cannot check grasp for '{object_to_check_name}', gripper issue.")
        return False

    def get_robot_base_pose(self) -> list:
        return self.robot_model_handle.get_pose().tolist()

    def get_base_velocities(self):
        """Returns [vx, vy, omega_z] for the robot base."""
        try:
            linear, angular = self.robot_model_handle.get_velocity()
            return [linear[0], linear[1], angular[2]]
        except Exception as e:
            rc_logger.warning(f"Could not get base velocities: {e}")
            return [0.0, 0.0, 0.0]

    def get_arm_joint_positions(self) -> list:
        return [h.get_joint_position() for i, h in enumerate(self.arm_joint_handles) if Joint.exists(self.arm_joint_names[i])]

    def get_end_effector_pose(self) -> list:
        if self.ee_tip_handle:
            return self.ee_tip_handle.get_pose().tolist()
        rc_logger.warning("End-effector tip handle not available for get_end_effector_pose. Returning base pose.")
        return self.robot_model_handle.get_pose().tolist()

    def get_camera_image(self, camera_key: str):
        """Get image from a camera by its key in camera_definitions.
        
        Args:
            camera_key (str): Key of the camera in self.camera_definitions
            
        Returns:
            numpy.ndarray or None: Image array if camera exists and capture succeeds, None otherwise
        """
        if camera_key in self.camera_handles:
            try:
                return self.camera_handles[camera_key].capture_rgb()
            except Exception as e:
                rc_logger.warning(f"Failed to capture image from camera '{camera_key}': {e}")
                return None
        rc_logger.warning(f"Camera '{camera_key}' not found in handles.")
        return None

    def get_camera_image_processed(
        self,
        camera_key: str,
        size=(84, 84),
        grayscale: bool = True,
        normalize: bool = True,
        clip: bool = True
    ):
        """
        Returns processed image (H,W,C) float32 in [0,1] (if normalize) or uint8.
        Falls back to zeros if capture fails.
        """
        raw = self.get_camera_image(camera_key)
        if raw is None:
            if grayscale:
                return np.zeros((size[1], size[0], 1), dtype=np.float32 if normalize else np.uint8)
            return np.zeros((size[1], size[0], 3), dtype=np.float32 if normalize else np.uint8)

        # raw likely already RGB float [0..1] or uint8; make it uint8 first
        arr = raw
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)

        arr_resized = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)

        if grayscale:
            arr_resized = cv2.cvtColor(arr_resized, cv2.COLOR_RGB2GRAY)
            arr_resized = arr_resized[:, :, None]

        if normalize:
            out = arr_resized.astype(np.float32) / 255.0
            if clip:
                out = np.clip(out, 0.0, 1.0)
            return out
        return arr_resized

    def check_collision(self, object_names_to_check_against: list = None) -> bool:
        if object_names_to_check_against:
            other_handles = [Shape(name) for name in object_names_to_check_against if Shape.exists(name)]
            if not other_handles:
                return self.robot_model_handle.check_collision()
            return self.robot_model_handle.check_collision_all(other_handles)
        return self.robot_model_handle.check_collision()