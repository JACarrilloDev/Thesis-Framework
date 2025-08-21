import time, os, random, copy, collections
from typing import Dict, List, Any, Optional
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.core.simulation import Simulation
from src.core.robot_controller import RobotController
from src.core.robot_definitions import get_robot_definition

from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy

AGENTS = ["robot_0", "robot_1"]

class DynamicTwoPhaseNavEnv(MultiAgentEnv):
    """
    Two robots, 4 fixed target dummies.

    Phase 1:
        - Randomly pick two distinct targets (one per robot).
    Phase 2 trigger:
        - First robot that completes its phase-1 target immediately decides both
          second-phase targets: it chooses the closer of the two remaining for itself;
          the other remaining target is assigned to the other robot.
    Completion:
        - Episode ends when both robots complete their second target OR max steps.

    Observations (vector part):
        [ self_x, self_y,
          other_rel_x, other_rel_y,
          cur_tgt_dx, cur_tgt_dy,
          next_tgt_dx, next_tgt_dy,
          phase (0/1),
          self_completed (0..2),
          other_completed (0..2),
          dist_to_cur (clipped),
          heading_alignment (0..1),
          obstacle_present (0/1),
          prox16 ]

    If use_camera=True => Dict observation: { "vect": <vector>, "img": stacked_frames }
    """

    def __init__(self, env_config: Dict[str, Any]):
        self.cfg = env_config
        task_cfg = env_config.get("task_config", {})   # grouped YAML required
        self.task_cfg = task_cfg

        self.scene_file = env_config["scene_file"]

        # ---- Unified robots_setup sourcing (nested or top-level) ----
        robots_setup = task_cfg.get("robots_setup") or env_config.get("robots_setup") or []
        if not robots_setup or len(robots_setup) != 2:
            raise ValueError(f"robots_setup must define exactly 2 entries (found {len(robots_setup)}). "
                             f"Provide under task_config.robots_setup or top-level robots_setup.")
        self.instance_overrides = robots_setup

        # Derive primary robot_type (allow top-level override, else first entry)
        self.robot_type = env_config.get("robot_type") or robots_setup[0].get("type")
        if self.robot_type is None:
            raise ValueError("robot_type missing: set top-level robot_type or first robots_setup.type.")

        self._deferred_camera_names: set = set()

        # Robot names
        self.robot_names = env_config.get(
            "robot_names_in_scene",
            [r.get("name") or r.get("robot_name_in_scene") for r in robots_setup]
        )
        if len(self.robot_names) != 2:
            raise ValueError(f"Need 2 robot names; got {self.robot_names}")

        self.max_steps = env_config.get("max_episode_steps", 500)
        self.success_dist = env_config.get("success_dist", 0.40)
        self.headless = env_config.get("headless", False)
        self.collision_dist = env_config.get("collision_dist", 0.30)

        # Verbosity (allow either nesting)
        self.verbose = task_cfg.get("verbose", env_config.get("verbose", False))
        self.log_every = task_cfg.get("log_every_steps", env_config.get("log_every_steps", 50))

        # Camera
        self.use_camera = env_config.get("use_camera", False)
        self.camera_key = env_config.get("camera_key", "front_camera")
        self.cam_size = tuple(env_config.get("camera_size", (84, 84)))
        self.cam_gray = bool(env_config.get("camera_grayscale", True))
        self.frame_stack = int(env_config.get("frame_stack", 4 if self.use_camera else 1))
        self.defer_camera = bool(env_config.get("defer_camera", False))
        self.capture_enabled = True
        if self.use_camera and (self.headless or self.defer_camera) and not env_config.get("force_camera_headless", False):
            print("[Info] Camera capture deferred (headless or defer_camera=True) -> supplying dummy frames.")
            self.capture_enabled = False

        # Targets & starts
        self.target_names = task_cfg.get("target_dummies", ["TargetPos1","TargetPos2","TargetPos3","TargetPos4"])
        self.start_names  = task_cfg.get("start_dummies",  ["StartPos1","StartPos2","StartPos3","StartPos4"])
        assert len(self.target_names) == 4, "Exactly 4 targets required."

        # Always-on moving nav targets (NavTarget, NavTarget2)
        self.nav_target_objects = task_cfg.get("nav_target_objects", ["NavTarget", "NavTarget2"])
        self.nav_current_indices: Dict[str, Optional[int]] = {aid: None for aid in AGENTS}
        self.nav_hits: Dict[str, int] = {aid: 0 for aid in AGENTS}

        # Reward weights
        rw = task_cfg.get("reward_weights", {})
        self.w_progress       = rw.get("progress", 5.0)
        self.w_time           = rw.get("time_penalty", -0.01)
        self.w_completion     = rw.get("completion", 25.0)
        self.w_final_bonus    = rw.get("final_bonus", 15.0)
        self.w_collision      = rw.get("collision", -20.0)
        self.w_no_progress    = rw.get("no_progress_penalty", -10.0)
        self.heading_weight   = rw.get("heading", 0.4)

        # No-progress detection
        self.stuck_delta = task_cfg.get("stuck_delta", 0.01)
        self.no_progress_threshold = task_cfg.get("no_progress_threshold",
                                                  task_cfg.get("stuck_threshold", 160))

        # Enhanced shaping parameters (all optional; defaults keep prior behavior close)
        self.shaping_gamma = task_cfg.get("shaping_gamma", 0.99)              # Potential-based progress discount
        self.progress_alpha = task_cfg.get("progress_alpha", 1.0)             # Φ = -alpha * distance
        self.near_target_bonus = task_cfg.get("near_target_bonus", 2.0)       # Max ramp bonus near success
        self.near_target_radius_mult = task_cfg.get("near_target_radius_mult", 2.0)
        self.vel_align_scale = task_cfg.get("vel_align_scale", 0.6)           # Forward velocity alignment scale
        self.spin_penalty_scale = task_cfg.get("spin_penalty_scale", -0.2)    # Negative -> penalty   

        # Dynamic obstacle control
        dyn = task_cfg.get("dynamic_obstacle", {})
        self.enable_obstacle_cfg = dyn.get("enabled", False)
        self.obstacle_name = dyn.get("name", "MidObstacle")
        self.obstacle_prob = dyn.get("spawn_prob", 0.5)
        self.hide_offset = dyn.get("hide_offset", 15.0)
        self._obstacle_pose_orig = None

        self.instance_overrides = robots_setup

        # Simulation
        self.sim = Simulation(scene_file=self.scene_file, headless=self.headless)
        self.sim.import_environment()
        self.sim.start()

        # Controllers
        self.controllers: Dict[str, RobotController] = {}
        self._frame_buffers: Dict[str, collections.deque] = {}
        for idx, aid in enumerate(AGENTS):
            inst = self.instance_overrides[idx]
            base_type = inst.get("type", self.robot_type)
            base_def = copy.deepcopy(get_robot_definition(base_type))
            base_def["robot_name_in_scene"] = self.robot_names[idx]

            if inst.get("wheel_joints"):
                base_def["base_actuators"]["wheel_joints"] = inst["wheel_joints"]
            if inst.get("sensor_prefix"):
                base_def["sensor_prefix"] = inst["sensor_prefix"]
            if inst.get("cameras"):
                base_def["cameras"] = inst["cameras"]

            if self.use_camera:
                for cam_def in base_def.get("cameras", {}).values():
                    nm = cam_def.get("name")
                    if nm:
                        self._deferred_camera_names.add(nm)

            if self.use_camera and not self.capture_enabled:
                base_def["cameras"] = {}

            self.controllers[aid] = RobotController(base_def, self.sim.pr)

            if self.use_camera:
                self._frame_buffers[aid] = collections.deque(maxlen=self.frame_stack)

        # Deferred camera handling
        if self.use_camera and not self.capture_enabled and self._deferred_camera_names:
            from pyrep.objects.vision_sensor import VisionSensor
            self._deferred_vs_handles = []
            for cam_name in self._deferred_camera_names:
                try:
                    if VisionSensor.exists(cam_name):
                        vs = VisionSensor(cam_name)
                        vs.set_explicit_handling(True)
                        self._deferred_vs_handles.append(vs)
                        print(f"[Info] Deferred vision sensor '{cam_name}' set to explicit handling.")
                except Exception as e:
                    print(f"[Warn] Could not defer vision sensor '{cam_name}': {e}")

        # Obstacle original pose
        if self.enable_obstacle_cfg:
            orig_pose = self.controllers[AGENTS[0]].get_object_pose(self.obstacle_name)
            if orig_pose is not None and len(orig_pose) >= 7:
                self._obstacle_pose_orig = list(orig_pose[:7])
                self._obstacle_is_shape = Shape.exists(self.obstacle_name)
            else:
                self._obstacle_pose_orig = [0.0,0.0,0.1,0.0,0.0,0.0,1.0]
                self._obstacle_is_shape = False

        # Observation spaces
        self.num_prox = 16
        low_vec = np.array(
            [-40, -40, -80, -80, -80, -80, -80, -80,
             0, 0, 0, 0.0, 0.0, 0.0] + [0.0]*self.num_prox, dtype=np.float32)
        high_vec = np.array(
            [40, 40, 80, 80, 80, 80, 80, 80,
             1, 2, 2, 80.0, 1.0, 1.0] + [1.0]*self.num_prox, dtype=np.float32)
        vect_space = spaces.Box(low=low_vec, high=high_vec, dtype=np.float32)
        if self.use_camera:
            c = 1 if self.cam_gray else 3
            stacked_c = c * self.frame_stack
            img_space = spaces.Box(low=0.0, high=1.0,
                                   shape=(self.cam_size[1], self.cam_size[0], stacked_c),
                                   dtype=np.float32)
            self.observation_space = spaces.Dict({"vect": vect_space, "img": img_space})
        else:
            self.observation_space = vect_space

        self.action_space = spaces.Box(
            low=np.array([-1.0,-1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self._forward_speed_scale = self.task_cfg.get("forward_speed_scale", 0.6)
        self._yaw_speed_scale = self.task_cfg.get("yaw_speed_scale", 1.2)

        if self.enable_obstacle_cfg:
            orig_pose = self.controllers[AGENTS[0]].get_object_pose(self.obstacle_name)

        # Episode state
        self.episode_index = 0
        self.current_step = 0
        self.first_phase_assign: Dict[str, int] = {}
        self.second_phase_assign: Dict[str, Optional[int]] = {}
        self.completed_counts: Dict[str, int] = {}
        self.prev_dist: Dict[str, Optional[float]] = {}
        self.no_progress_steps: Dict[str, int] = {}
        self.second_phase_chosen = False
        self.enable_obstacle_active = False
        self.target_completion_step: List[int] = [-1]*4
        self.target_completed_by: List[Optional[str]] = [None]*4
        self.phase2_decider: Optional[str] = None
        self.start_dummy_chosen: Dict[str, str] = {aid: "" for aid in AGENTS}
        self.last_step_rewards: Dict[str, float] = {aid: 0.0 for aid in AGENTS}
        self.enable_dual_stagnation_terminate = self.task_cfg.get("enable_dual_stagnation_terminate", True)
        self.dual_stagnation_limit = self.task_cfg.get("dual_stagnation_limit", 240)
        self._dual_stagnation_counter = 0

    def enable_camera_capture(self):
        """Call (e.g. via curriculum) to start real image capture mid-training."""
        if self.use_camera and not self.capture_enabled:
            for vs in getattr(self, "_deferred_vs_handles", []):
                try:
                    vs.set_explicit_handling(False)
                except Exception:
                    pass
            for ctrl in self.controllers.values():
                for vs in getattr(ctrl, "camera_handles", {}).values():
                    try:
                        vs.set_explicit_handling(False)
                    except Exception:
                        pass
            self.capture_enabled = True
            print("[Info] Camera capture ENABLED.")
            for aid in AGENTS:
                fb = self._frame_buffers[aid]
                fb.clear()
                real = self._capture_frame(aid)
                for _ in range(self.frame_stack):
                    fb.append(real.copy())

    def safe_set_pose(self, name: str, pose7: List[float]) -> bool:
        """Set pose for Shape or Dummy without assuming type order."""
        try:
            handle = None
            if Shape.exists(name):
                handle = Shape(name)
            elif Dummy.exists(name):
                handle = Dummy(name)
            if not handle:
                if self.verbose:
                    print(f"[Warn] safe_set_pose: '{name}' not found.")
                return False
            if len(pose7) != 7:
                if self.verbose:
                    print(f"[Warn] safe_set_pose: pose len {len(pose7)} != 7 for '{name}'")
                return False
            handle.set_pose(pose7)
            return True
        except Exception as e:
            if self.verbose:
                print(f"[Warn] safe_set_pose failed for '{name}': {e}")
            return False

    # ---------- Helpers ----------
    def _xy(self, aid): return np.array(self.controllers[aid].get_robot_base_pose()[:2], dtype=np.float32)
    def _tpose_xy(self, idx):
        pose = self.controllers[AGENTS[0]].get_object_pose(self.target_names[idx])
        return np.array(pose[:2], dtype=np.float32) if pose is not None else None
    def _distance(self, a,b): return float(np.linalg.norm(a-b))

    def _pick_initial_targets(self):
        perm = list(range(4)); random.shuffle(perm)
        self.first_phase_assign["robot_0"] = perm[0]
        self.first_phase_assign["robot_1"] = perm[1]
        self.second_phase_assign = {"robot_0": None, "robot_1": None}
        return perm[2:]

    def _decide_second_phase(self, finisher, remaining):
        fin_xy = self._xy(finisher)
        d_pairs = [(self._distance(fin_xy, self._tpose_xy(idx)), idx) for idx in remaining]
        d_pairs.sort()
        chosen = d_pairs[0][1]
        other = remaining[0] if remaining[1] == chosen else remaining[1]
        self.second_phase_assign[finisher] = chosen
        other_agent = "robot_1" if finisher == "robot_0" else "robot_0"
        self.second_phase_assign[other_agent] = other
        self.second_phase_chosen = True
        self.phase2_decider = finisher
        if self.verbose:
            print(f"[Episode {self.episode_index}] Phase2 decided by {finisher}: "
                  f"{finisher}->{self.target_names[chosen]}, "
                  f"{other_agent}->{self.target_names[other]}")

    def _teleport_nav_target(self, aid: str, tgt_idx: int):
        """Place the agent's nav target object onto one of the fixed TargetPos* dummies."""
        if tgt_idx < 0 or tgt_idx >= len(self.target_names):
            return
        if aid not in AGENTS:
            return
        obj_name = self.nav_target_objects[AGENTS.index(aid)] if AGENTS.index(aid) < len(self.nav_target_objects) else self.nav_target_objects[0]
        pose = self.controllers[AGENTS[0]].get_object_pose(self.target_names[tgt_idx])
        if pose is not None and len(pose) >= 7:
            self.controllers[AGENTS[0]].set_object_pose(obj_name, pose[:7])
            self.nav_current_indices[aid] = tgt_idx
        elif self.verbose:
            print(f"[Warn] Could not teleport nav target '{obj_name}' to target idx {tgt_idx} (pose invalid: {pose})")

    def _get_nav_target_xy(self, aid: str):
        obj_name = self.nav_target_objects[AGENTS.index(aid)] if AGENTS.index(aid) < len(self.nav_target_objects) else self.nav_target_objects[0]
        pose = self.controllers[AGENTS[0]].get_object_pose(obj_name)
        return np.array(pose[:2], dtype=np.float32) if pose is not None else None

    def _current_target_idx(self, aid):
        if self.completed_counts[aid] == 0: return self.first_phase_assign[aid]
        if self.completed_counts[aid] == 1: return self.second_phase_assign[aid]
        return None

    def _next_target_vec(self, aid, self_xy):
        if self.completed_counts[aid] == 0 and self.second_phase_assign[aid] is not None:
            txy = self._tpose_xy(self.second_phase_assign[aid])
            return txy - self_xy if txy is not None else np.zeros(2,dtype=np.float32)
        return np.zeros(2,dtype=np.float32)

    def _heading_alignment(self, aid, self_xy, target_xy):
        pose = self.controllers[aid].get_robot_base_pose()
        yaw = pose[5] if len(pose) >= 6 else 0.0
        vec = target_xy - self_xy
        tgt_ang = np.arctan2(vec[1], vec[0])
        diff = np.arctan2(np.sin(tgt_ang - yaw), np.cos(tgt_ang - yaw))
        return 1.0 - (abs(diff) / np.pi)

    def _capture_frame(self, aid):
        if self.use_camera and not self.capture_enabled:
            c = 1 if self.cam_gray else 3
            noise = self.task_cfg.get("deferred_cam_noise", 0.0)
            if noise > 0.0:
                return (np.random.rand(self.cam_size[1], self.cam_size[0], c).astype(np.float32) * noise)
            return np.zeros((self.cam_size[1], self.cam_size[0], c), dtype=np.float32)
        img = self.controllers[aid].get_camera_image_processed(
            self.camera_key, size=self.cam_size,
            grayscale=self.cam_gray, normalize=True)
        if img is None:
            c = 1 if self.cam_gray else 3
            img = np.zeros((self.cam_size[1], self.cam_size[0], c), dtype=np.float32)
        return img

    def _stacked_image(self, aid):
        fb = self._frame_buffers[aid]
        if len(fb) < self.frame_stack:
            first = fb[0]
            while len(fb) < self.frame_stack:
                fb.appendleft(first.copy())
        return np.concatenate(list(fb), axis=2)

    def _build_vect_obs(self, aid):
        self_xy = self._xy(aid)
        other = "robot_1" if aid == "robot_0" else "robot_0"
        other_rel = self._xy(other) - self_xy
        tgt_xy = self._get_nav_target_xy(aid)
        cur_vec = np.zeros(2, dtype=np.float32)
        dist_cur = 0.0
        heading_align = 0.0
        if tgt_xy is not None:
            cur_vec = tgt_xy - self_xy
            dist_cur = min(80.0, np.linalg.norm(cur_vec))
            heading_align = self._heading_alignment(aid, self_xy, tgt_xy)
        nxt_vec = self._next_target_vec(aid, self_xy)
        phase = float(self.completed_counts.get(aid, 0))
        prox = np.clip(self.controllers[aid].get_proximity_sensor_readings(), 0.0, 1.0)
        obstacle_present = 1.0 if self.enable_obstacle_active else 0.0
        return np.concatenate([
            self_xy, other_rel, cur_vec, nxt_vec,
            np.array([
                phase,
                float(self.completed_counts.get(aid, 0)),
                float(self.completed_counts.get(other, 0)),
                dist_cur,
                heading_align,
                obstacle_present
            ], dtype=np.float32),
            prox
        ]).astype(np.float32)

    def _build_obs(self, aid):
        vect = self._build_vect_obs(aid)
        if not self.use_camera:
            return vect
        fb = self._frame_buffers[aid]
        frame = self._capture_frame(aid)
        fb.append(frame)
        stacked = self._stacked_image(aid).astype(np.float32)
        return {"vect": vect, "img": stacked}

    def _compute_agent_reward(self, aid, vect_obs):
        reward = 0.0
        reward += self.w_time
        dist_cur = vect_obs[11]
        heading_align = vect_obs[12]
        obstacle_present = vect_obs[13]
        prox = vect_obs[14:14+self.num_prox]

        made_progress = False
        cur_idx = self._current_target_idx(aid)
        if cur_idx is not None and dist_cur > 0:
            prev = self.prev_dist[aid]
            if prev is not None:
                delta = prev - dist_cur
                made_progress = delta > self.stuck_delta
                if made_progress:
                    reward += np.clip(delta, -0.5, 0.5) * self.w_progress
                    phi_prev = -self.progress_alpha * prev
                    phi_cur  = -self.progress_alpha * dist_cur
                    reward += self.shaping_gamma * phi_cur - phi_prev
                else:
                    self.no_progress_steps[aid] += 1
                    if (self.no_progress_steps[aid] % 25 == 0) and self.verbose:
                        print(f"[Ep {self.episode_index} Step {self.current_step}] {aid} no-progress counter={self.no_progress_steps[aid]} delta={(prev - dist_cur):.4f}")
                    if self.no_progress_steps[aid] >= self.no_progress_threshold:
                        reward += self.w_no_progress
                        if self.verbose:
                            print(f"[Ep {self.episode_index} Step {self.current_step}] {aid} NO-PROGRESS PENALTY applied (steps={self.no_progress_steps[aid]}).")
                        self.no_progress_steps[aid] = 0
                if made_progress:
                    self.no_progress_steps[aid] = 0
            self.prev_dist[aid] = dist_cur

        reward += self.heading_weight * heading_align

        vx = self.controllers[aid].get_base_velocities()[0]
        if prox[3] < 0.2 and vx > 0.12:
            reward += -1.5 * vx
        elif vx > 0.10:
            reward += 0.8 * vx
        elif vx < -0.25:
            reward += -1.0 * abs(vx)

        tgt_xy = self._get_nav_target_xy(aid)
        if tgt_xy is not None and dist_cur > 0:
            self_xy = self._xy(aid)
            vec = tgt_xy - self_xy
            tgt_dir = vec / (np.linalg.norm(vec) + 1e-6)
            yaw = self.controllers[aid].get_robot_base_pose()[5]
            fwd_axis = np.array([np.cos(yaw), np.sin(yaw)])
            alignment_cos = float(np.clip(np.dot(fwd_axis, tgt_dir), -1.0, 1.0))
            reward += max(0.0, vx) * alignment_cos * self.vel_align_scale

        base_vels = self.controllers[aid].get_base_velocities()
        wz = base_vels[2] if len(base_vels) > 2 else 0.0
        if self.prev_dist[aid] is not None and self.no_progress_steps[aid] > 10 and abs(wz) > 0.6:
            reward += self.spin_penalty_scale * abs(wz)

        if obstacle_present > 0.5 and vx > 0.05 and self.completed_counts[aid] == 0 and self.current_step < 100:
            reward += 0.02

        if cur_idx is not None and 0 < dist_cur < self.near_target_radius_mult * self.success_dist:
            span = self.near_target_radius_mult * self.success_dist
            reward += ((span - dist_cur) / span) * self.near_target_bonus

        self._made_progress_flags[aid] = made_progress
        return reward

    def _collision(self):
        return self._distance(self._xy("robot_0"), self._xy("robot_1")) < self.collision_dist

    def _log_episode_header(self):
        print(f"\n=== Episode {self.episode_index} START ===")
        for aid, rname in zip(AGENTS, self.robot_names):
            start_label = self.start_dummy_chosen.get(aid, "?")
            pose = self.controllers[aid].get_robot_base_pose()
            print(f" {aid}/{rname} start_dummy={start_label} "
                  f"pose=({pose[0]:.2f},{pose[1]:.2f}) yaw={pose[5]:.2f}")

    # ---------- API ----------
    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.episode_index += 1
        self._made_progress_flags = {aid: False for aid in AGENTS}
        self._dual_stagnation_counter = 0
        self.sim.reset(); self.sim.start()
        if self.use_camera and not self.capture_enabled:
            for vs in getattr(self, "_deferred_vs_handles", []):
                try:
                    vs.set_explicit_handling(True)
                except Exception:
                    pass
        time.sleep(0.05)

        self.enable_obstacle_active = False
        if self.enable_obstacle_cfg and self._obstacle_pose_orig is not None:
            if random.random() < self.obstacle_prob:
                self.controllers[AGENTS[0]].set_object_pose(self.obstacle_name, self._obstacle_pose_orig.copy())
                self.enable_obstacle_active = True
            else:
                hidden_pose = self._obstacle_pose_orig.copy()
                hidden_pose[2] -= self.hide_offset
                self.controllers[AGENTS[0]].set_object_pose(self.obstacle_name, hidden_pose)

        starts = random.sample(self.start_names, 2) if len(self.start_names) >= 2 else self.start_names[:2]
        for aid, sname in zip(AGENTS, starts):
            self.start_dummy_chosen[aid] = sname  # record chosen start
            pose = self.controllers[aid].get_object_pose(sname)
            if pose is not None:
                self.safe_set_pose(self.controllers[aid].robot_name_in_scene, pose[:7])

        self.completed_counts = {aid: 0 for aid in AGENTS}
        self.prev_dist = {aid: None for aid in AGENTS}
        self.no_progress_steps = {aid: 0 for aid in AGENTS}
        self.last_step_rewards = {aid: 0.0 for aid in AGENTS}
        self.second_phase_chosen = False
        self.phase2_decider = None
        self._remaining_after_initial = self._pick_initial_targets()
        self.target_completion_step = [-1]*4
        self.target_completed_by = [None]*4

        for aid in AGENTS:
            idx = self.first_phase_assign[aid]
            self._teleport_nav_target(aid, idx)
            self.nav_hits[aid] = 0
            self.prev_dist[aid] = None
            self.no_progress_steps[aid] = 0

        if self.use_camera:
            for aid in AGENTS:
                fb = self._frame_buffers[aid]; fb.clear()
                first = self._capture_frame(aid)
                for _ in range(self.frame_stack):
                    fb.append(first.copy())

        self._log_episode_header()
        print(f" Phase1 assignments: robot_0->{self.target_names[self.first_phase_assign['robot_0']]}, "
              f"robot_1->{self.target_names[self.first_phase_assign['robot_1']]} (remaining: "
              f"{[self.target_names[i] for i in self._remaining_after_initial]})")
        if self.enable_obstacle_cfg:
            print(f" Obstacle {self.obstacle_name}: {'ACTIVE' if self.enable_obstacle_active else 'HIDDEN'}")
        return {aid: self._build_obs(aid) for aid in AGENTS}, {}

    def step(self, action_dict: Dict[str, np.ndarray]):
        self.current_step += 1
        self._made_progress_flags = {aid: False for aid in AGENTS}
        for aid, act in action_dict.items():
            a = np.clip(act, self.action_space.low, self.action_space.high)
            self.controllers[aid].set_base_target_velocities(
                [float(a[0]) * self._forward_speed_scale, 0.0],
                float(a[1]) * self._yaw_speed_scale
            )
        self.sim.step()

        vect_obs_cache = {aid: self._build_vect_obs(aid) for aid in AGENTS}
        rewards = {}

        for aid in AGENTS:
            vect = vect_obs_cache[aid]
            dist_cur = vect[11]
            rewards[aid] = self._compute_agent_reward(aid, vect)

            cur_idx = self._current_target_idx(aid)
            if cur_idx is not None and dist_cur < self.success_dist:
                # Only award once per target index
                if self.target_completed_by[cur_idx] is None:
                    rewards[aid] += self.w_completion
                    self.target_completed_by[cur_idx] = aid
                    self.target_completion_step[cur_idx] = self.current_step
                    self.nav_hits[aid] += 1
                    self.completed_counts[aid] += 1

                    if self.verbose:
                        print(f"[Ep {self.episode_index} Step {self.current_step}] {aid} HIT target {self.target_names[cur_idx]} (hits={self.nav_hits[aid]})")

                    # Phase transition: choose second targets once both at 1
                    if self.completed_counts[aid] == 1:
                        # If phase2 not yet chosen, decide NOW
                        if not self.second_phase_chosen:
                            self._decide_second_phase(aid, self._remaining_after_initial)
                        nxt_idx = self._current_target_idx(aid)
                        if nxt_idx is not None:
                            self._teleport_nav_target(aid, nxt_idx)
                            self.prev_dist[aid] = None
                            self.no_progress_steps[aid] = 0
                    elif self.completed_counts[aid] == 2:
                        self.nav_current_indices[aid] = None
                        self.prev_dist[aid] = None
                        self.no_progress_steps[aid] = 0
                        rewards[aid] += self.w_final_bonus

        # Stagnation (after progress flags updated)
        if not any(self._made_progress_flags.values()):
            self._dual_stagnation_counter += 1
        else:
            self._dual_stagnation_counter = 0
        early_stagnant_done = (
            self.enable_dual_stagnation_terminate and
            self._dual_stagnation_counter >= self.dual_stagnation_limit
        )
        if early_stagnant_done and self.verbose:
            print(f"[Ep {self.episode_index} Step {self.current_step}] Early termination: dual stagnation trigger.")

        # Collision
        if self._collision():
            for aid in AGENTS:
                rewards[aid] += self.w_collision
            if self.verbose:
                print(f"[Ep {self.episode_index} Step {self.current_step}] Collision penalized.")

        for aid in AGENTS:
            self.last_step_rewards[aid] = rewards[aid]

        obs = {aid: self._build_obs(aid) for aid in AGENTS}

        # Termination / truncation bookkeeping
        terminateds = {aid: self.completed_counts[aid] == 2 for aid in AGENTS}
        # Early stagnation counts as truncation (unless already terminated)
        truncateds = {
            aid: ((self.current_step >= self.max_steps) or early_stagnant_done) and not terminateds[aid]
            for aid in AGENTS
        }
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = (not terminateds["__all__"]) and any(truncateds.values())

        episode_end = terminateds["__all__"] or truncateds["__all__"]

        if (self.current_step % self.log_every == 0) or episode_end:
            d0 = vect_obs_cache["robot_0"][11]; d1 = vect_obs_cache["robot_1"][11]
            print(f"[Ep {self.episode_index} Step {self.current_step}] "
                  f"r0 Dist={d0:.2f} R={self.last_step_rewards['robot_0']:.3f} Hits={self.nav_hits['robot_0']} | "
                  f"r1 Dist={d1:.2f} R={self.last_step_rewards['robot_1']:.3f} Hits={self.nav_hits['robot_1']}")

        full_success = all(self.completed_counts[a] == 2 for a in AGENTS)
        infos = {
            aid: {
                "nav_hits": self.nav_hits[aid],
                "completed": self.completed_counts[aid],
                "dual_stagnation_steps": self._dual_stagnation_counter,
                "early_stagnation": early_stagnant_done,
                "start_dummy": self.start_dummy_chosen.get(aid, ""),
                "phase2_started": self.second_phase_chosen,
                "full_success": full_success
            } for aid in AGENTS
        }
        # Provide single-agent success alias on first agent (used by generic tooling)
        if episode_end:
            infos["robot_0"]["success"] = full_success
        return obs, rewards, terminateds, truncateds, infos