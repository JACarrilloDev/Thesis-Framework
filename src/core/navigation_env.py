import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from src.core.simulation import Simulation
from src.core.robot_controller import RobotController
from src.core.robot_definitions import get_robot_definition
import time

class NavigationEnv(gym.Env):
    def __init__(self, env_config):
        super().__init__()
        scene_file = env_config["scene_file"]
        robot_type = env_config["robot_type"]
        robot_name_in_scene = env_config["robot_name_in_scene"]
        task_config = env_config["task_config"]
        max_episode_steps = env_config.get("max_episode_steps", 1000)
        headless = env_config.get("headless", False)

        self.stuck_steps = 0
        self.stuck_threshold = 160  # Number of steps with little/no progress before penalty
        self.stuck_delta = 0.03    # What counts as "no progress" (meters)

        self.sim = Simulation(scene_file=scene_file, headless=headless)
        self.sim.import_environment()
        self.sim.start()
        robot_def = get_robot_definition(robot_type)
        robot_def["robot_name_in_scene"] = robot_name_in_scene
        self.robot = RobotController(robot_definition=robot_def, pyrep_instance=self.sim.pr)
        self.task_config = task_config
        self.max_steps = max_episode_steps
        self.current_step = 0

        self.num_prox = 16
        self.img_shape = (64, 64, 3)
        obs_dim = 4 + self.num_prox + np.prod(self.img_shape)
        self.observation_space = spaces.Box(
            low=np.concatenate([np.full(4, -40), np.zeros(self.num_prox), np.zeros(np.prod(self.img_shape))]),
            high=np.concatenate([np.full(4, 40), np.ones(self.num_prox), np.ones(np.prod(self.img_shape))]),
            shape=(obs_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

        self.target_name = task_config['learning_objective']['target_object']
        self.success_dist = task_config['learning_objective']['success_threshold_distance']

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.stuck_steps = 0
        self._prev_dist = None
        self.sim.reset()
        self.sim.start()
        for _ in range(5):  # Multiple steps to ensure stability
            self.sim.step()
        time.sleep(0.1)  # Reduced sleep time since we're using multiple steps
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1
        noise = np.random.normal(0, 0.05, size=action.shape)
        action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        self.robot.set_base_target_velocities(action[:2], action[2])
        self.sim.step()
        obs = self._get_obs()
        reward, terminated = self._compute_reward(obs)

        # Get positions first
        robot_xy = obs[:2]
        target_xy = obs[2:4]
        dist = np.linalg.norm(robot_xy - target_xy)

        # Print more informative status every 30 steps
        if self.current_step % 30 == 0:
            print(f"Step {self.current_step}: Distance={dist:.3f}m, Reward={reward:.3f}")

        robot_pose = self.robot.get_robot_base_pose()
        if robot_pose[2] < 0.1:  # Z position below threshold
            terminated = True
            reward -= 30.0 
            print(f"Robot fell at step {self.current_step}!")

        # Early termination for way off course
        if dist > 15.0:  # If robot is too far
            terminated = True
            reward -= 20.0  # Penalty for going too far

        truncated = self.current_step >= self.max_steps
        info = {"distance_to_target": dist}  # Using already calculated distance
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        robot_pose = self.robot.get_robot_base_pose()
        target_pose = self.robot.get_object_pose(self.target_name)
        prox = self.robot.get_proximity_sensor_readings()
        prox = np.clip(prox, 0.0, 1.0)
        # print(f"Proximity sensor readings at step {self.current_step}: {prox}")
        img = self.robot.get_camera_image("front_camera")
        if img is not None:
            import cv2
            img_resized = cv2.resize(img, (self.img_shape[1], self.img_shape[0]))
            img_flat = (img_resized / 255.0).flatten()
            img_flat = np.clip(img_flat, 0.0, 1.0) 
        else:
            img_flat = np.zeros(np.prod(self.img_shape), dtype=np.float32)
        obs = np.concatenate([robot_pose[:2], target_pose[:2], prox, img_flat])
        return obs

    def _compute_reward(self, obs):
        robot_xy = obs[:2]
        target_xy = obs[2:4]
        dist = np.linalg.norm(robot_xy - target_xy)
        prox = obs[4:4+self.num_prox]
        reward = 0.0

        # Progress reward: encourage getting closer to the goal
        if hasattr(self, '_prev_dist') and self._prev_dist is not None:
            delta_dist = self._prev_dist - dist
            reward += delta_dist * 15.0 # Tuned for stability

            # Stuck detection
            if abs(delta_dist) < self.stuck_delta:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0

            if self.stuck_steps >= self.stuck_threshold:
                reward -= 20.0
                print(f"Stuck penalty at step {self.current_step} (no progress for {self.stuck_steps} steps)")
                self.stuck_steps = 0
        else:
            self.stuck_steps = 0

        self._prev_dist = dist

        # Small step penalty to encourage efficiency
        reward -= 0.01

        # Encourage moving forward, penalize moving backward
        if hasattr(self.robot, "get_base_velocities"):
            vx = self.robot.get_base_velocities()[0]
            # If sensor4 (index 3) is close to an obstacle, penalize forward motion
            if prox[3] < 0.2 and vx > 0.12:
                reward -= 1.5 * vx  # Penalize trying to go forward into a wall
            elif vx > 0.10:
                reward += 0.75 * vx  # Small bonus for forward motion
            elif vx < -0.15:
                reward -= 2.0 * abs(vx)  # Penalize backward motion
                # print(f"Backward penalty at step {self.current_step} (vx={vx:.2f})")

        # Alignment reward: encourage facing the target
        robot_pose = self.robot.get_robot_base_pose()  # [x, y, z, roll, pitch, yaw]
        robot_yaw = robot_pose[5]
        to_target = target_xy - robot_xy
        target_angle = np.arctan2(to_target[1], to_target[0])
        angle_diff = np.arctan2(np.sin(target_angle - robot_yaw), np.cos(target_angle - robot_yaw))  # [-pi, pi]
        alignment_reward = 1.0 - (abs(angle_diff) / np.pi)  # 1 when aligned, 0 when opposite
        reward += 0.07 * alignment_reward  # Small bonus for facing the target

        # If facing a wall, encourage turning toward the target
        if prox[3] < 0.2 and abs(angle_diff) > 0.3:
            reward += 0.5 * alignment_reward  # Extra encouragement to turn toward target

        # Proximity penalty: only penalize for being very close to obstacles
        very_close_threshold = 0.15
        proximity_penalty = np.sum(prox < very_close_threshold)
        if proximity_penalty > 0:
            reward -= proximity_penalty * 3.0  # Strong penalty for being too close
            # print(f"Proximity penalty at step {self.current_step} (close sensors: {proximity_penalty})")

        # Success bonus
        if dist < self.success_dist:
            reward += 100.0
            print(f"Goal reached at step {self.current_step}!")

        return reward, dist < self.success_dist

    def render(self, mode='human'):
        pass

    def close(self):
        pass