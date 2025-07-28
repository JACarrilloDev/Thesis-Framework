import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from .simulation import Simulation # Assuming Simulation class is defined
from .robot_controller import RobotController
from .task_manager import TaskManager # For getting task info
from .logger import setup_logger

env_logger = setup_logger('robot_env_logger', 'logs/robot_env.log', console_output=True)

def get_relative_pose(pose_a, pose_b):
    """Calculates pose_b relative to pose_a. Both are [x,y,z,qx,qy,qz,qw]."""
    # This is a simplified version. For full SE(3) transform:
    # T_wa = pose_to_matrix(pose_a)
    # T_wb = pose_to_matrix(pose_b)
    # T_ab = np.linalg.inv(T_wa) @ T_wb
    # return matrix_to_pose_vector(T_ab)
    # For now, just relative position and orientation difference (approx)
    pos_rel = np.array(pose_b[:3]) - np.array(pose_a[:3])
    # Quaternion difference is more complex (q_rel = q_a_inv * q_b).
    # For simplicity in observation, could use Euler angle differences or just target orientation.
    # Let's just return relative position and b's full orientation for now.
    return np.concatenate([pos_rel, pose_b[3:]])


class RobotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, sim_instance: Simulation, 
                 robot_controller: RobotController, 
                 task_manager: TaskManager,
                 task_config: dict, # Loaded from the YAML file
                 max_episode_steps=1000,
                 render_mode=None):
        super().__init__()
        self.sim = sim_instance
        self.robot_controller = robot_controller
        self.task_manager = task_manager # Not used heavily for DRL logic, but for setup
        self.task_config = task_config
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        self.object_to_manipulate = task_config['learning_objective']['object_to_manipulate']
        self.target_pose_object = task_config['learning_objective']['target_pose_object']
        self.success_dist_thresh = task_config['learning_objective']['success_threshold_distance']
        self.success_angle_thresh = task_config['learning_objective']['success_threshold_angle']
        
        self.initial_projector_pose = task_config['environment_objects'][0]['initial_pose'] # Assuming projector is first

        # --- Action Space (Example for Asti: base_vel_x, base_vel_y, base_ang_vel_z, 6x arm_joint_vel, gripper_cmd) ---
        # base_vel_x, base_vel_y: [-0.5, 0.5] m/s
        # base_ang_vel_z: [-1.0, 1.0] rad/s
        # arm_joint_vel: [-1.0, 1.0] rad/s (scaled by controller)
        # gripper_cmd: [-1.0 (close), 1.0 (open)]
        num_arm_joints = len(robot_controller.definition.get("arm_joints", []))
        action_dim = 2 + 1 + num_arm_joints + 1 
        act_low = [-0.5, -0.5, -1.0] + [-1.0] * num_arm_joints + [-1.0]
        act_high = [0.5, 0.5, 1.0] + [1.0] * num_arm_joints + [1.0]
        self.action_space = spaces.Box(low=np.array(act_low), high=np.array(act_high), dtype=np.float32)
        env_logger.info(f"Action space defined (dim={action_dim}): {self.action_space}")

        # --- Observation Space ---
        # base_pose (7: x,y,z,qx,qy,qz,qw)
        # arm_joint_positions (num_arm_joints)
        # gripper_open_amount (1)
        # is_gripping_projector (1: 0 or 1)
        # projector_pose_relative_to_ee (7: dx,dy,dz,qx,qy,qz,qw)
        # target_pose_relative_to_projector (7: dx,dy,dz,qx,qy,qz,qw)
        obs_dim = 7 + num_arm_joints + 1 + 1 + 7 + 7 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        env_logger.info(f"Observation space defined (dim={obs_dim}): {self.observation_space}")

        self.render_mode = render_mode
        self._prev_dist_ee_to_projector = None
        self._prev_dist_projector_to_target = None
        self._is_gripping_projector_flag = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        env_logger.info("Resetting RobotEnv.")
        self.current_step = 0
        self._prev_dist_ee_to_projector = None
        self._prev_dist_projector_to_target = None
        self._is_gripping_projector_flag = False
        
        # Reset simulation (robot pose, object poses)
        # This needs a more robust reset in Simulation class or here
        self.sim.reset() # Basic sim reset (stops/starts)
        
        # Set projector to its initial pose (more robust reset)
        projector_handle = self.sim.pr.get_object(self.object_to_manipulate)
        if projector_handle.exists():
            projector_handle.set_pose(self.initial_projector_pose)
        
        # Robot initial pose is set by its definition or task_config,
        # ensure Simulation.reset() or RobotController handles this.

        observation = self._get_observation()
        info = self._get_info()
        
        env_logger.debug(f"Env reset. Initial observation shape: {observation.shape}")
        return observation, info

    def step(self, action):
        self.current_step += 1
        
        # Apply action
        base_linear_vel = action[0:2]
        base_angular_vel = action[2]
        num_arm_joints = len(self.robot_controller.definition.get("arm_joints", []))
        arm_joint_velocities = action[3 : 3 + num_arm_joints]
        gripper_command = action[3 + num_arm_joints]

        self.robot_controller.set_base_velocities(base_linear_vel.tolist(), base_angular_vel)
        self.robot_controller.set_arm_joint_target_velocities(arm_joint_velocities.tolist())
        self.robot_controller.actuate_gripper(gripper_command)

        self.sim.step()

        observation = self._get_observation()
        reward = self._calculate_reward(observation) # Pass current obs for reward calculation
        
        # Check termination conditions
        terminated = self._check_success(observation)
        if terminated:
            reward += 500.0 # Large success bonus
            env_logger.info(f"SUCCESS! Task completed at step {self.current_step}.")

        truncated = False
        if self.current_step >= self.max_episode_steps:
            truncated = True
            env_logger.info(f"Episode truncated at step {self.current_step} (max steps reached).")
        
        info = self._get_info()
        if self.current_step % 100 == 0: # Log periodically
            env_logger.debug(f"Step {self.current_step}. Obs_sample: {observation[:3]}, Reward: {reward:.3f}, Term: {terminated}, Trunc: {truncated}")
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        base_pose_abs = np.array(self.robot_controller.get_robot_base_pose())
        arm_joints_pos = np.array(self.robot_controller.get_arm_joint_positions())
        gripper_open_amount = np.array([self.robot_controller.get_gripper_open_amount()])
        
        ee_pose_abs = np.array(self.robot_controller.get_end_effector_pose())
        projector_pose_abs_list = self.robot_controller.get_object_pose(self.object_to_manipulate)
        projector_pose_abs = np.array(projector_pose_abs_list) if projector_pose_abs_list else np.zeros(7) # Default if not found

        target_marker_pose_abs_list = self.robot_controller.get_object_pose(self.target_pose_object)
        target_marker_pose_abs = np.array(target_marker_pose_abs_list) if target_marker_pose_abs_list else np.zeros(7)

        # Relative poses
        # Projector relative to end-effector
        projector_rel_to_ee = get_relative_pose(ee_pose_abs, projector_pose_abs)
        
        # Target relative to current projector pose (if found, else relative to origin)
        target_rel_to_projector = get_relative_pose(projector_pose_abs, target_marker_pose_abs)

        # Is projector gripped?
        self._is_gripping_projector_flag = self.robot_controller.check_grasp(self.object_to_manipulate)
        is_gripping_projector_obs = np.array([1.0 if self._is_gripping_projector_flag else 0.0])

        obs = np.concatenate([
            base_pose_abs, arm_joints_pos, gripper_open_amount, is_gripping_projector_obs,
            projector_rel_to_ee, target_rel_to_projector
        ]).astype(np.float32)
        
        if obs.shape[0] != self.observation_space.shape[0]:
            env_logger.error(f"Observation shape mismatch! Expected {self.observation_space.shape[0]}, got {obs.shape[0]}. Padding/truncating.")
            # Fallback to zeros/truncate to prevent crash, but this is a critical error
            correct_shape_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            common_len = min(obs.shape[0], self.observation_space.shape[0])
            correct_shape_obs[:common_len] = obs[:common_len]
            return correct_shape_obs
            
        return obs

    def _calculate_reward(self, current_observation):
        reward = -0.01  # Small penalty per step to encourage efficiency

        # Deconstruct observation for easier access
        ee_pose_abs = np.array(self.robot_controller.get_end_effector_pose())
        projector_pose_abs_list = self.robot_controller.get_object_pose(self.object_to_manipulate)
        projector_pose_abs = np.array(projector_pose_abs_list) if projector_pose_abs_list else ee_pose_abs # Avoid None

        target_marker_pose_abs_list = self.robot_controller.get_object_pose(self.target_pose_object)
        target_marker_pose_abs = np.array(target_marker_pose_abs_list) if target_marker_pose_abs_list else projector_pose_abs


        # Phase 1: Reaching for the projector (if not holding it)
        if not self._is_gripping_projector_flag:
            dist_ee_to_projector = np.linalg.norm(ee_pose_abs[:3] - projector_pose_abs[:3])
            if self._prev_dist_ee_to_projector is not None:
                reward += (self._prev_dist_ee_to_projector - dist_ee_to_projector) * 10.0 # Reward getting closer
            self._prev_dist_ee_to_projector = dist_ee_to_projector
            
            # Bonus for being very close to projector
            if dist_ee_to_projector < 0.05:
                reward += 5.0
        else: # Holding the projector
            self._prev_dist_ee_to_projector = None # Reset reach reward shaping
            reward += 2.0 # Small bonus for maintaining grasp

            # Phase 2: Moving projector to target
            dist_projector_to_target = np.linalg.norm(projector_pose_abs[:3] - target_marker_pose_abs[:3])
            if self._prev_dist_projector_to_target is not None:
                reward += (self._prev_dist_projector_to_target - dist_projector_to_target) * 15.0 # Reward getting closer
            self._prev_dist_projector_to_target = dist_projector_to_target

            # Bonus for projector being very close to target
            if dist_projector_to_target < 0.05:
                reward += 10.0
                # Bonus for correct orientation (simplified: dot product of Z axes or quaternion distance)
                # q_proj = projector_pose_abs[3:]
                # q_targ = target_marker_pose_abs[3:]
                # orientation_similarity = ... (complex to calculate simply)
                # reward += orientation_similarity * 5.0
        

        # Penalty for collisions
        if self.robot_controller.check_collision():
            reward -= 20.0
            env_logger.debug("Collision detected, -20 reward.")

        return reward

    def _check_success(self, current_observation):
        # Check if projector is at target and not gripped (meaning it was placed)
        if not self._is_gripping_projector_flag: # Must have released it
            projector_pose_abs_list = self.robot_controller.get_object_pose(self.object_to_manipulate)
            target_marker_pose_abs_list = self.robot_controller.get_object_pose(self.target_pose_object)

            if projector_pose_abs_list and target_marker_pose_abs_list:
                projector_pos = np.array(projector_pose_abs_list[:3])
                target_pos = np.array(target_marker_pose_abs_list[:3])
                
                dist = np.linalg.norm(projector_pos - target_pos)
                
                # Simplified orientation check (e.g. dot product of Z axes from quaternions)
                # q_proj = projector_pose_abs_list[3:]
                # q_targ = target_marker_pose_abs_list[3:]
                # z_proj = ... # vector from q_proj
                # z_targ = ... # vector from q_targ
                # angle_diff = np.arccos(np.clip(np.dot(z_proj, z_targ), -1.0, 1.0))
                angle_diff_placeholder = 0.0 # Needs proper quaternion math

                if dist < self.success_dist_thresh and angle_diff_placeholder < self.success_angle_thresh:
                    return True
        return False

    def _get_info(self):
        return {
            "current_step": self.current_step,
            "is_gripping": self._is_gripping_projector_flag,
            "dist_ee_to_projector": self._prev_dist_ee_to_projector,
            "dist_projector_to_target": self._prev_dist_projector_to_target,
        }

    def render(self): # Assuming RobotController has get_camera_image
        if self.render_mode == 'rgb_array':
            img = self.robot_controller.get_camera_image() 
            return img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        pass # Human mode is CoppeliaSim GUI

    def close(self):
        env_logger.info("Closing RobotEnv.")
        # Simulation shutdown is typically handled by the main script