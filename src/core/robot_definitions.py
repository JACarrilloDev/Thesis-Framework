from typing import Optional

PREDEFINED_ROBOTS = {
    "Panda_Default": { # Example for Panda if you have one
        "description": "Franka Emika Panda robot arm.",
        "model_file": "src/robots/Panda.ttm", # Path to the .ttm model file if loaded separately
        "robot_name_in_scene": "Panda",       # Default name of the robot in the CoppeliaSim scene
        "arm_joints": [
            "Panda_joint1", "Panda_joint2", "Panda_joint3", "Panda_joint4",
            "Panda_joint5", "Panda_joint6", "Panda_joint7"
        ],
        "gripper_name": "Panda_gripper", # Name of the gripper object/mechanism
        "gripper_joints": ["Panda_finger_joint1", "Panda_finger_joint2"], # Joints controlling the gripper
        "gripper_actuation_type": "force_closure", # or "joint_control"
        "end_effector_tip_name": "Panda_tip", # Name of the dummy object at the EE tip
        "cameras": {
            "hand_camera": "Panda_handCamera" # Example camera
        },
        "action_space_components": ["arm_joint_velocities", "gripper_command"],
        "observation_components": ["arm_joint_positions", "gripper_state", "ee_pose"]
    },
    "AstiPioneerHybrid": {
        "description": "Hybrid robot: Pioneer base with Asti arms and body, vision sensor on mast.",
        "robot_name_in_scene": "AstiPioneerHybrid",  # Name in CoppeliaSim scene
        "base_type": "differential_drive",
        "base_actuators": {
            "wheel_joints": ["Pioneer_p3dx_leftMotor", "Pioneer_p3dx_rightMotor"],
            "wheel_radius": 0.0975,  # Update if scaled
            "wheel_separation": 0.31  # Update if scaled
        },
        "arm_joints": [
            "leftArmJoint0", "leftArmJoint1", "leftArmJoint2",
            "rightArmJoint0", "rightArmJoint1", "rightArmJoint2"
        ],
        "gripper_name": None,  # No gripper, just arms
        "gripper_joints": None,
        "cameras": {
            "front_camera": {"name": "Vision_sensor"}
        },
        "end_effector_tip_name": None,  # No gripper tip
        "base_control_script": "Pioneer_p3dx",  # Use the Pioneer Lua script for wheel control
        "base_control_function": None,
        "observation_components": ["base_pose", "arm_joint_positions", "camera_image"],
        "action_space_components": ["base_velocities", "arm_joint_velocities"]
    },
}

def get_robot_definition(robot_type: str) -> Optional[dict]:
    """Retrieves a predefined robot definition."""
    return PREDEFINED_ROBOTS.get(robot_type)
