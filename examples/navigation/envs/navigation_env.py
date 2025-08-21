import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from src.core.base_env import BaseEnv
from src.core.simulation import Simulation
from src.core.robot_controller import RobotController
from src.core.robot_definitions import get_robot_definition
from collections import deque

class NavigationEnv(BaseEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        # Se extrae la configuración
        self.scene_file = env_config["scene_file"]
        self.robot_type = env_config["robot_type"]
        self.robot_name = env_config["robot_name_in_scene"]
        self.task_config = env_config["task_config"]
        self.max_steps = env_config.get("max_episode_steps", 300)
        self.headless = env_config.get("headless", False)
        self.use_camera = env_config.get("use_camera", True)
        self.camera_key = env_config.get("camera_key", "front_camera")
        self.cam_size = tuple(env_config.get("camera_size", (84, 84)))  # (W,H)
        self.cam_grayscale = env_config.get("camera_grayscale", True)
        self.frame_stack = int(env_config.get("frame_stack", 4)) if self.use_camera else 1
        self._episode_counter = 0

        # Inicialización de simulación y robot
        self.sim = Simulation(scene_file=self.scene_file, headless=self.headless)
        self.sim.import_environment()
        self.sim.start()

        robot_def = get_robot_definition(self.robot_type)
        robot_def["robot_name_in_scene"] = self.robot_name
        self.robot = RobotController(robot_definition=robot_def, pyrep_instance=self.sim.pr)

        # Parámetros específicos de la tarea
        self.target_name = self.task_config['learning_objective']['target_object']
        self.success_dist = self.task_config['learning_objective'].get('success_threshold_distance', 0.2)

        # Inicialización de contadores y estados
        self.current_step = 0
        self.stuck_steps = 0
        self.stuck_threshold = 160
        self.stuck_delta = 0.03
        self._prev_dist = None

        # Definición del espacio de observación
        self.num_prox = 16
        self.vector_dim = 4 + self.num_prox  # [robot_x, robot_y, target_x, target_y, prox...]
        vect_low = np.concatenate([np.full(4, -40.0), np.zeros(self.num_prox)])
        vect_high = np.concatenate([np.full(4, 40.0), np.ones(self.num_prox)])

        if self.use_camera:
            c_channels = 1 if self.cam_grayscale else 3
            stacked_channels = c_channels * self.frame_stack
            img_shape = (self.cam_size[1], self.cam_size[0], stacked_channels)
            img_low = np.zeros(img_shape, dtype=np.float32)
            img_high = np.ones(img_shape, dtype=np.float32)
            self._frame_buffer = deque(maxlen=self.frame_stack)
            self.observation_space = spaces.Dict({
                "vect": spaces.Box(low=vect_low, high=vect_high, dtype=np.float32),
                "img": spaces.Box(low=img_low, high=img_high, dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Box(
                low=vect_low,
                high=vect_high,
                dtype=np.float32
            )

        # Action space unchanged (vx, vy ignored by controller except x; omega)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.stuck_steps = 0
        self._prev_dist = None
        self._episode_counter += 1

        try:
            self.sim.reset()
            self.sim.start()
            time.sleep(0.1)  # Dar tiempo a las físicas

            # Conseguir posición aleatoria del objetivo
            target_dummies = ["TargetPos1", "TargetPos2", "TargetPos3", "TargetPos4"]
            chosen_target = np.random.choice(target_dummies)
            target_pose = self.robot.get_object_pose(chosen_target)
            if target_pose:
                self.robot.set_object_pose(self.target_name, target_pose)
                print(f"\nEpisode {self._episode_counter} - NavTarget placed at {chosen_target}")

            if self.use_camera:
                self._frame_buffer.clear()
                first = self._capture_frame()
                for _ in range(self.frame_stack):
                    self._frame_buffer.append(first.copy())

            obs = self._get_obs()
        return obs, {}

        except Exception as e:
            print(f"Error during reset: {e}")
            return self.reset(seed=seed, options=options)

    def _capture_frame(self):
        img = self.robot.get_camera_image_processed(
            self.camera_key,
            size=self.cam_size,
            grayscale=self.cam_grayscale,
            normalize=True
        )
        return img  # (H,W,1 or 3)

    def _get_stacked_image(self):
        if len(self._frame_buffer) < self.frame_stack:
            # Pad with first frame
            first = self._frame_buffer[0]
            while len(self._frame_buffer) < self.frame_stack:
                self._frame_buffer.appendleft(first.copy())
        # Stack along channel axis
        return np.concatenate(list(self._frame_buffer), axis=2)  # (H,W,C*stack)

    def _build_vector_obs(self):
        robot_pose = self.robot.get_robot_base_pose()
        target_pose = self.robot.get_object_pose(self.target_name)
        prox = np.clip(self.robot.get_proximity_sensor_readings(), 0.0, 1.0)
        return np.concatenate([robot_pose[:2], target_pose[:2], prox]).astype(np.float32)

    def _get_obs(self):
        vect = self._build_vector_obs()
        if not self.use_camera:
            return vect
        # push new frame
        frame = self._capture_frame()
        self._frame_buffer.append(frame)
        stacked = self._get_stacked_image().astype(np.float32)
        return {"vect": vect, "img": stacked}

    def _compute_reward(self, obs):
        reward = 0.0
        robot_xy = obs[:2]
        target_xy = obs[2:4]
        prox = obs[4:4+self.num_prox]
        dist = np.linalg.norm(robot_xy - target_xy)

        # Penalización por cada paso
        reward -= 0.02

        # Calculate heading reward
        robot_yaw = self.robot.get_robot_base_pose()[5]
        desired_yaw = np.arctan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0])
        heading_diff = np.abs(np.arctan2(np.sin(desired_yaw - robot_yaw), np.cos(desired_yaw - robot_yaw)))
        heading_reward = 1.0 - (heading_diff / np.pi)
        reward += heading_reward * 0.4  # slightly reduced

        # Progress reward
        if self._prev_dist is not None:
            delta_dist = self._prev_dist - dist
            delta_dist = np.clip(delta_dist, -0.2, 0.2)
            reward += delta_dist * 8.0  # was 15.0
            if abs(delta_dist) < self.stuck_delta:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0
            if self.stuck_steps >= self.stuck_threshold:
                reward -= 8.0  # slightly reduced
                print(f"Stuck penalty applied at step {self.current_step} (dist={dist:.3f})")
                self.stuck_steps = 0
        self._prev_dist = dist

        # Movement rewards
        vx = self.robot.get_base_velocities()[0]
        if prox[3] < 0.2 and vx > 0.12:  # If facing wall
            reward -= 1.5 * vx
        elif vx > 0.10:  # Moving forward
            reward += 0.8 * vx
        elif vx < -0.15:  # Moving backward
            reward -= 1.2 * abs(vx)

        # Recompensa por éxito
        if dist < self.success_dist:
            reward += 40.0
            print(f"Goal reached at step {self.current_step}!")

        return reward, dist < self.success_dist

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Execute action
        self.robot.set_base_target_velocities(action[:2], action[2])
        self.sim.step()
        
        obs = self._get_obs()

        # Get observation and compute reward
        vect_part = obs["vect"] if self.use_camera else obs
        reward, terminated = self._compute_reward(vect_part)

        # Get distance for info
        robot_xy = vect_part[:2]
        target_xy = vect_part[2:4]
        dist = np.linalg.norm(robot_xy - target_xy)

        if self.current_step % 30 == 0:
            print(f"Step {self.current_step}: Distance={dist:.3f}m, Reward={reward:.3f}")

        truncated = self.current_step >= self.max_steps
        success = bool(terminated and dist < self.success_dist)
        info = {
            "distance_to_target": dist,
            "success": success
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self, 'sim'):
            self.sim.shutdown()