import time
import os
import csv
import json
from datetime import datetime
from typing import Any, Dict, Optional, List, Callable

import numpy as np
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from src.core.logger import setup_logger

try:
    from tqdm import trange
    USE_TQDM = True
except ImportError:
    USE_TQDM = False


class UnifiedMetricsCallback(DefaultCallbacks):
    """
    Unified metrics aggregation:
      Single-agent: expects info.get('success') boolean (e.g. NavigationEnv).
      Multi-agent: per-agent infos, may include:
        full_success, phase2_started, nav_hits, dual_stagnation_steps.
    Produces custom_metrics with *_mean suffix via RLlib aggregation.
    Backward-compatible aliases: MultiAgentMetricsCallback, SuccessCallback.
    """
    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        # RLlib EpisodeV2 may not expose `last_infos`
        infos = getattr(episode, "last_infos", None)
        if infos is None:
            infos = getattr(episode, "_agent_to_last_info", {}) or {}
            if not infos:
                try:
                    single = episode.last_info_for()
                    if single:
                        infos = {"agent_0": single}
                except Exception:
                    infos = {}
        ud = episode.user_data.setdefault("ma", {})
        ud.setdefault("completed", {})
        ud.setdefault("nav_hits", {})
        st_max = ud.get("dual_stagnation_max", 0)
        for aid, inf in infos.items():
            if not isinstance(inf, dict):
                continue
            if "completed" in inf:
                ud["completed"][aid] = int(inf["completed"])
            if "nav_hits" in inf:
                ud["nav_hits"][aid] = int(inf["nav_hits"])
            st_max = max(st_max, int(inf.get("dual_stagnation_steps", 0)))
            if inf.get("phase2_started"):
                ud["phase2_started"] = True
            if inf.get("full_success"):
                ud["full_success"] = True
            if inf.get("success"):
                ud["any_success"] = True
        ud["dual_stagnation_max"] = st_max

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Prefer accumulated user_data; fallback to last_infos
        ud = episode.user_data.get("ma", {})
        # RLlib EpisodeV2-safe retrieval
        infos = getattr(episode, "last_infos", None)
        if infos is None:
            infos = getattr(episode, "_agent_to_last_info", {}) or {}
        # Fallback single-agent
        if not infos:
            single = episode.last_info_for()
            if single:
                infos = {"agent_0": single}

        # Accumulators
        any_success = False
        full_success = False
        phase2_started = False
        dual_stagnation_max = 0
        nav_hits_vals: List[float] = []
        nav_hits_r0 = None
        nav_hits_r1 = None
        total_completed_hits = 0
        have_multi_progress = False

        # Phase-1 scaling hints (defaults if env does not supply)
        max_hits_per_agent = 2
        max_total_hits = None
        agent_ids = list(episode.get_agents()) if hasattr(episode, "get_agents") else list(infos.keys())

        # Pass 1: read per-agent infos
        for aid, inf in infos.items():
            if not isinstance(inf, dict):
                continue
            # Single-agent success
            if inf.get("success"):
                any_success = True
            # Multi-agent fields
            if inf.get("full_success"):
                full_success = True
                any_success = True
            if inf.get("phase2_started"):
                phase2_started = True
            dual_stagnation_max = max(dual_stagnation_max, int(inf.get("dual_stagnation_steps", 0)))
            if "nav_hits" in inf:
                nav_hits_vals.append(float(inf["nav_hits"]))
                if aid == "robot_0":
                    nav_hits_r0 = inf["nav_hits"]
                if aid == "robot_1":
                    nav_hits_r1 = inf["nav_hits"]
            if "completed" in inf:
                have_multi_progress = True
                try:
                    c = int(inf["completed"])
                except Exception:
                    c = 0
                total_completed_hits += max(0, min(c, max_hits_per_agent))
            # Phase-1 scaling hints (optional from env)
            if "max_hits_per_agent" in inf:
                try:
                    max_hits_per_agent = int(inf["max_hits_per_agent"])
                except Exception:
                    pass
            if "max_total_hits" in inf:
                try:
                    max_total_hits = int(inf["max_total_hits"])
                except Exception:
                    pass

        # Pass 2: merge accumulated per-step user_data
        if "completed" in ud:
            have_multi_progress = True
            for c in ud["completed"].values():
                total_completed_hits += max(0, min(int(c), max_hits_per_agent))
        if "nav_hits" in ud:
            vals = list(ud["nav_hits"].values())
            if vals:
                nav_hits_vals.append(float(np.mean(vals)))
            nav_hits_r0 = ud["nav_hits"].get("robot_0", nav_hits_r0)
            nav_hits_r1 = ud["nav_hits"].get("robot_1", nav_hits_r1)
        dual_stagnation_max = max(dual_stagnation_max, int(ud.get("dual_stagnation_max", 0)))
        phase2_started = phase2_started or bool(ud.get("phase2_started"))
        full_success = full_success or bool(ud.get("full_success"))
        any_success = any_success or bool(ud.get("any_success"))

        # Success metric:
        # - If we have multi-agent progress, scale by configured totals:
        #   denom = max_total_hits (if provided) else max_hits_per_agent * num_agents
        # Fallback: read env state directly if RLlib infos were empty (still zeros)
        if not have_multi_progress:
            try:
                envs = base_env.get_sub_environments() if hasattr(base_env, "get_sub_environments") else [base_env]
                env = envs[0]
                cc = getattr(env, "completed_counts", {})
                nh = getattr(env, "nav_hits", {})
                ds = int(getattr(env, "dual_stagnation_step", 0))
                # Record as numeric metrics (not strings)
                episode.custom_metrics["nav_hits_r0"] = float(nh.get("robot_0", 0.0))
                episode.custom_metrics["nav_hits_r1"] = float(nh.get("robot_1", 0.0))
                episode.custom_metrics["nav_hits_mean"] = float(np.mean(list(nh.values()))) if nh else 0.0
                episode.custom_metrics["dual_stagnation_steps"] = ds

                total_hits = int(sum(max(0, int(v)) for v in cc.values())) if isinstance(cc, dict) else 0
                num_agents = max(1, len(cc)) if isinstance(cc, dict) else max(1, len(episode.agent_rewards) or [0])
                # success over full 2-phase (denom = 2 hits per agent)
                episode.custom_metrics["success_qtr"] = float(np.clip(total_hits / float(2 * num_agents), 0.0, 1.0))
                # phase-1 success (denom = 1 hit per agent)
                episode.custom_metrics["success_phase1"] = float(np.clip(total_hits / float(num_agents), 0.0, 1.0))
                # coarse success (any hit)
                episode.custom_metrics["success"] = 1.0 if total_hits > 0 else 0.0
                episode.custom_metrics["full_success"] = 1.0 if total_hits >= (2 * num_agents) else 0.0
            except Exception:
                # keep metrics at 0 if env fields are missing
                pass

        if have_multi_progress:
            denom = max_total_hits if (isinstance(max_total_hits, int) and max_total_hits > 0) \
                    else (max_hits_per_agent * max(1, len(agent_ids)))
            success_val = (float(total_completed_hits) / float(denom)) if denom > 0 else 0.0
            episode.custom_metrics["success"] = float(np.clip(success_val, 0.0, 1.0))
        else:
            episode.custom_metrics["success"] = 1.0 if any_success else 0.0

        # Additional, explicit phase-1 metrics:
        # - success_phase1: scale by 2 (2 robots x 1 phase) -> shows 0.5 when only one robot hits.
        # - success_qtr: scale by 4 (2 robots x 2 phases) -> shows 0.25/0.5 style even if episode ends early.
        denom_phase1 = max(1, len(agent_ids))       # 2 robots -> 2
        denom_phase2 = 2 * denom_phase1              # 2 robots * 2 phases
        if have_multi_progress:
            episode.custom_metrics["success_phase1"] = float(
                np.clip((float(total_completed_hits) / float(denom_phase1)), 0.0, 1.0)
            )
            episode.custom_metrics["success_qtr"] = float(
                np.clip((float(total_completed_hits) / float(denom_phase2)), 0.0, 1.0)
            )
        else:
            base = 1.0 if any_success else 0.0
            episode.custom_metrics["success_phase1"] = base
            episode.custom_metrics["success_qtr"] = base

        episode.custom_metrics["full_success"] = 1.0 if full_success else 0.0
        episode.custom_metrics["phase2_started"] = 1.0 if phase2_started else 0.0
        episode.custom_metrics["nav_hits_mean"] = float(np.mean(nav_hits_vals)) if nav_hits_vals else 0.0
        episode.custom_metrics["nav_hits_r0"] = float(nav_hits_r0) if nav_hits_r0 is not None else 0.0
        episode.custom_metrics["nav_hits_r1"] = float(nav_hits_r1) if nav_hits_r1 is not None else 0.0
        episode.custom_metrics["dual_stagnation_steps"] = int(dual_stagnation_max)

MultiAgentMetricsCallback = UnifiedMetricsCallback
SuccessCallback = UnifiedMetricsCallback

class RLTrainer:
    def __init__(
        self,
        env_class,
        env_config: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs",
        log_name: str = "rl_training.log",
        num_rollout_workers: Optional[int] = None,
        train_batch_size: int = 8192,
        sgd_minibatch_size: int = 256,
        lr: float = 3e-4,
        entropy_coeff_start: float = 0.02,
        entropy_coeff_min: float = 0.005,
        entropy_decay: float = 0.995,
        curriculum_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
        checkpoint_dir: str = "checkpoints",
        batch_mode: str = "complete_episodes",
        rollout_fragment_length: int = 128
    ):
        self.env_class = env_class
        self._wall_time_total = 0.0
        self._last_iter_time_s = 0.0
        self.env_config = env_config or {}
        self._last_result: Optional[Dict[str, Any]] = None
        self.curriculum_fn = curriculum_fn
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = setup_logger("RLTrainerLogger",
                                   os.path.join(log_dir, log_name),
                                   console_output=True)

        self.metrics_csv_path = os.path.join(log_dir, "training_metrics.csv")
        self.metrics_jsonl_path = os.path.join(log_dir, "training_metrics.jsonl")
        self._init_metrics_files()
        self._prev_timesteps_total: Optional[int] = None

        if num_rollout_workers is None:
            num_rollout_workers = 0
            num_envs_per_worker = 1
        else:
            num_envs_per_worker = 1

        self.entropy_coeff = entropy_coeff_start
        self.entropy_coeff_min = entropy_coeff_min
        self.entropy_decay = entropy_decay

        base_cfg = (
            PPOConfig()
            .environment(
                env=self.env_class,
                env_config=self.env_config,
                disable_env_checking=True
            )
            .framework("torch")
            .callbacks(UnifiedMetricsCallback)
            .rollouts(
                num_rollout_workers=num_rollout_workers,
                num_envs_per_worker=num_envs_per_worker,
                rollout_fragment_length=rollout_fragment_length,
                batch_mode=batch_mode
            )
        )

        vision_model_cfg = {
            "conv_filters": [
                [32, [8, 8], 4],
                [64, [4, 4], 2],
                [64, [3, 3], 1],
            ],
            "conv_activation": "relu",
            "fcnet_hiddens": [256, 128],
            "fcnet_activation": "tanh",
            "vf_share_layers": True
        }

        is_multi = bool(self.env_config.get("multi_agent", False))
        policies_spec = self.env_config.get("policies_spec")  # Optional external specification

        cfg = base_cfg

        if is_multi:
            # Shared or multiple policies
            if not policies_spec:
                policies_spec = {
                    "shared_policy": (None, None, None, {})
                }
                def policy_mapping_fn(agent_id, *a, **k):
                    return "shared_policy"
                policies_to_train = ["shared_policy"]
            else:
                def policy_mapping_fn(agent_id, *a, **k):
                    for pname, pd in policies_spec.items():
                        if "agents" in pd and agent_id in pd["agents"]:
                            return pname
                    return list(policies_spec.keys())[0]
                transformed = {}
                for pname in policies_spec.keys():
                    transformed[pname] = (None, None, None, {})
                policies_spec = transformed
                policies_to_train = list(policies_spec.keys())

            cfg = cfg.multi_agent(
                policies=policies_spec,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=policies_to_train
            )

        cfg = cfg.training(
            model=vision_model_cfg,
            lr=lr,
            gamma=0.99,
            lambda_=0.97,
            clip_param=0.2,
            vf_clip_param=2.0,
            vf_loss_coeff=1.5,
            grad_clip=0.5,
            sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=10,
            train_batch_size=train_batch_size,
            kl_coeff=0.01,
            kl_target=0.02,
            entropy_coeff=self.entropy_coeff,
        ).resources(num_gpus=0)

        self.config: PPOConfig = cfg
        self.config.observation_filter = "NoFilter"
        self.config.clip_actions = True
        self.config.compress_observations = True
        self.trainer = PPOTrainer(config=self.config)
        self.logger.info(
            f"Initialized PPO (workers={num_rollout_workers}, train_batch={self.config.train_batch_size}, "
            f"frag_len={rollout_fragment_length}, batch_mode={batch_mode}, multi_agent={is_multi})"
        )

    def _get_rss_mb(self):
        try:
            import psutil, os
            return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        except Exception:
            return -1.0

    # ---------- Metrics ----------
    @property
    def _csv_header(self) -> List[str]:
        return [
            "training_iteration",
            "timesteps_total",
            "timesteps_this_iter",
            "episodes_total",
            "episodes_this_iter",
            "episode_len_mean",
            "episode_reward_mean",
            "episode_reward_min",
            "episode_reward_max",
            "success_mean",
            "success_phase1_mean",
            "success_qtr_mean",
            "full_success_mean",
            "phase2_started_mean",
            "nav_hits_mean_mean",
            "nav_hits_r0_mean",
            "nav_hits_r1_mean",
            "dual_stagnation_steps_mean",
            "entropy",
            "policy_loss",
            "vf_loss",
            "vf_explained_var",
            "kl",
            "cur_kl_coeff",
            "entropy_coeff_used",
            "iter_time_s",
            "wall_time_s_total"
        ]

    def _init_metrics_files(self):
        need_header = True
        if os.path.exists(self.metrics_csv_path):
            # Validate existing header length; if mismatch, archive and recreate
            try:
                with open(self.metrics_csv_path, "r") as f:
                    first = f.readline().strip().split(",")
                if len(first) == len(self._csv_header):
                    need_header = False
                else:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    bak = self.metrics_csv_path + f".{ts}.bak"
                    os.rename(self.metrics_csv_path, bak)
                    print(f"[Metrics] Archived old CSV with incompatible header to {bak}")
            except Exception:
                # If read fails, we will rewrite
                pass
        if need_header:
            with open(self.metrics_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._csv_header)

    def _extract_learner_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        learner = result.get("info", {}).get("learner", {}) or {}
        for key in ["shared_policy", "default_policy"]:
            if key in learner:
                return learner[key].get("learner_stats", {}) or {}
        if learner:
            first = next(iter(learner.values()))
            return first.get("learner_stats", {}) or {}
        return {}

    def _append_metrics(self, result: Dict[str, Any]):
        ls = self._extract_learner_stats(result)
        cms = result.get("custom_metrics", {}) or {}
        success_rate = cms.get("success_mean", None)
        full_success_rate = cms.get("full_success_mean", None)

        # Compute timesteps_this_iter if missing
        ts_total = result.get("timesteps_total") or 0
        if self._prev_timesteps_total is not None:
            ts_this = ts_total - self._prev_timesteps_total
        else:
            ts_this = result.get("timesteps_this_iter") or 0
        self._prev_timesteps_total = ts_total

        def g(key, default=0.0):
            val = cms.get(key)
            return default if val is None else val

        row = [
            result.get("training_iteration") or 0,
            ts_total,
            ts_this,
            result.get("episodes_total") or 0,
            result.get("episodes_this_iter") or 0,
            result.get("episode_len_mean") or 0.0,
            result.get("episode_reward_mean") or 0.0,
            result.get("episode_reward_min") or 0.0,
            result.get("episode_reward_max") or 0.0,
            g("success_mean", 0.0),
            g("success_phase1_mean", 0.0),
            g("success_qtr_mean", 0.0),
            g("full_success_mean", 0.0),
            g("phase2_started_mean", 0.0),
            g("nav_hits_mean_mean", 0.0),
            g("nav_hits_r0_mean", 0.0),
            g("nav_hits_r1_mean", 0.0),
            g("dual_stagnation_steps_mean", 0.0),
            ls.get("entropy") or 0.0,
            ls.get("policy_loss") or 0.0,
            ls.get("vf_loss") or 0.0,
            ls.get("vf_explained_var") or 0.0,
            ls.get("kl") or 0.0,
            ls.get("cur_kl_coeff") or 0.0,
            self.entropy_coeff,
            self._last_iter_time_s or 0.0,
            self._wall_time_total or 0.0
        ]
        try:
            with open(self.metrics_csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            self.logger.error(f"Failed writing CSV metrics: {e}")
        try:
            out = dict(result)
            out["_iter_time_s"] = self._last_iter_time_s
            out["_wall_time_s_total"] = self._wall_time_total
            with open(self.metrics_jsonl_path, "a") as f:
                f.write(json.dumps(out, default=str) + "\n")
        except Exception as e:
            self.logger.error(f"Failed writing JSONL metrics: {e}")

    # ---------- Logging helpers ----------
    def _safe_fmt(self, val, fmt=".2f"):
        return f"{val:{fmt}}" if isinstance(val, (int, float)) else str(val)

    def _format_iter_log(self, global_iter: int, result: Dict[str, Any], success_rate_display: Any) -> str:
        ls = self._extract_learner_stats(result)
        return (
            f"Iter {global_iter} "
            f"reward_mean={self._safe_fmt(result.get('episode_reward_mean'))} "
            f"succ={self._safe_fmt(success_rate_display)} "
            f"r_min={self._safe_fmt(result.get('episode_reward_min'))} "
            f"r_max={self._safe_fmt(result.get('episode_reward_max'))} "
            f"ep_len={self._safe_fmt(result.get('episode_len_mean'))} "
            f"ts_total={result.get('timesteps_total')} "
            f"kl={self._safe_fmt(ls.get('kl'))} "
            f"kl_coeff={self._safe_fmt(ls.get('cur_kl_coeff'))} "
            f"pol_loss={self._safe_fmt(ls.get('policy_loss'))} "
            f"vf_loss={self._safe_fmt(ls.get('vf_loss'))} "
            f"vf_ev={self._safe_fmt(ls.get('vf_explained_var'))} "
            f"entropy={self._safe_fmt(ls.get('entropy'))} "
            f"entropy_coeff={self._safe_fmt(self.entropy_coeff)}"
        )

    # ---------- Checkpointing ----------
    def save_checkpoint(self) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt_path = self.trainer.save(self.checkpoint_dir)
        try:
            if isinstance(ckpt_path, str) and self._last_result:
                with open(os.path.join(ckpt_path, "result.json"), "w") as f:
                    json.dump(self._last_result, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Could not write result.json: {e}")
        self.logger.info(f"Checkpoint saved: {ckpt_path}")
        return str(ckpt_path)

    def restore(self, checkpoint_path: str):
        self.trainer.restore(checkpoint_path)
        self.logger.info(f"Restored from checkpoint: {checkpoint_path}")

    # ---------- Scheduling / curricula ----------
    def _decay_entropy(self):
        if self.entropy_coeff > self.entropy_coeff_min:
            self.entropy_coeff = max(self.entropy_coeff_min,
                                     self.entropy_coeff * self.entropy_decay)
            try:
                self.trainer.config["entropy_coeff"] = self.entropy_coeff
            except Exception:
                pass
            try:
                self.trainer.workers.foreach_policy(
                    lambda p, pid=None: setattr(p, "entropy_coeff", self.entropy_coeff)
                )
            except Exception:
                pass
            try:
                for _, policy in self.trainer.workers.local_worker().policy_map.items():
                    policy.config["entropy_coeff"] = self.entropy_coeff
            except Exception:
                pass

    def _apply_curriculum(self, global_iter: int):
        if not self.curriculum_fn:
            return
        updates = self.curriculum_fn(global_iter)
        if not updates:
            return
        def _upd(env):
            # Support field updates and on-demand camera enabling
            for k, v in updates.items():
                if k == "enable_camera_capture" and v:
                    try:
                        env.enable_camera_capture()
                    except Exception:
                        pass
                elif hasattr(env, k):
                    setattr(env, k, v)
        self.trainer.workers.foreach_env(_upd)

    # ---------- Public API ----------
    def train(
        self,
        iterations: int = 1,
        checkpoint_every_global: int = 40,
        batch_log: int = 10
    ):
        """
        Run exactly `iterations` RLlib iterations (each = one trainer.train()).
        Progress bar counts remaining iterations in THIS call.
        Global PPO iteration comes from result['training_iteration'] (continues from resume).
        """
        log = self.logger
        if iterations <= 0:
            return self._last_result

        log.info(f"Starting training block: remaining_iterations={iterations} @ {datetime.now().isoformat()}")
        try:
            self._apply_curriculum(0)
        except Exception:
            pass
        iterator = trange(iterations, desc="Remaining", leave=True) if USE_TQDM and iterations > 1 else range(iterations)

        success_hist: List[float] = []
        reward_hist: List[float] = []

        last_result = None
        for _local in iterator:
            iter_t0 = time.perf_counter()
            result = self.trainer.train()
            global_iter = result.get("training_iteration")

            if global_iter % 1 == 0:
                self.logger.info(f"[Memory] Iter {global_iter} RSS_MB={self._get_rss_mb():.1f}")

            self._last_result = result
            self._last_iter_time_s = time.perf_counter() - iter_t0
            self._wall_time_total += self._last_iter_time_s

            # Use success_mean (phase-1 aware) if present; else fallback to full_success_mean
            cm = result.get("custom_metrics", {}) or {}
            success_rate = (
                cm.get("success_phase1_mean")
                if cm.get("success_phase1_mean") is not None else
                (cm.get("success_qtr_mean")
                 if cm.get("success_qtr_mean") is not None else
                 (cm.get("success_mean") if cm.get("success_mean") is not None else cm.get("full_success_mean")))
            )
            success_disp = success_rate if success_rate is not None else "N/A"

            log_line = self._format_iter_log(global_iter, result, success_disp)
            print(log_line)
            log.info(log_line)

            if isinstance(success_rate, (int, float)):
                success_hist.append(success_rate)
            reward_hist.append(result.get("episode_reward_mean", 0.0))

            if iterations > 1 and (global_iter % batch_log == 0):
                batch_succ = np.mean(success_hist[-batch_log:]) if success_hist else float("nan")
                batch_rew = np.mean(reward_hist[-batch_log:]) if reward_hist else float("nan")
                msg = f"[Rolling {batch_log}] avg_success={batch_succ:.3f} avg_reward={batch_rew:.2f} (global_iter={global_iter})"
                print(msg); log.info(msg)

            self._decay_entropy()
            self._append_metrics(result)
            self._apply_curriculum(global_iter)

            if checkpoint_every_global and (global_iter % checkpoint_every_global == 0):
                self.save_checkpoint()

        if last_result:
            last_global = last_result.get("training_iteration")
            if checkpoint_every_global and last_global % checkpoint_every_global != 0:
                self.save_checkpoint()

        return last_result

    def evaluate(self, episodes: int = 10):
        self.logger.info(f"Evaluation start ({episodes} episodes param)")
        results = self.trainer.evaluate()
        eval_reward = results.get("evaluation", {}).get("episode_reward_mean", "N/A")
        eval_success = results.get("evaluation", {}).get("success_rate", "N/A")
        msg = f"Evaluation: mean_reward={eval_reward} success_rate={eval_success}"
        print(msg)
        self.logger.info(msg)
        return results