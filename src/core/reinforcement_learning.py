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


class MultiAgentMetricsCallback(DefaultCallbacks):
    """
    Aggregates multi-agent episode metrics into custom_metrics.
    Expects per-agent infos to optionally include:
      - full_success (bool)
      - phase2_started (bool)
      - nav_hits (int)
      - dual_stagnation_steps (int)
      - success (bool) (will be injected if full_success present)
    Falls back gracefully if keys absent (single-agent envs still work).
    """
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # RLlib EpisodeV2: multi-agent infos accessible via episode.last_infos (dict)
        infos = {}
        try:
            infos = episode.last_infos or {}
        except AttributeError:
            # Fallback single agent
            single = episode.last_info_for()
            if single:
                infos = {"agent": single}

        full_success = False
        phase2_started = False
        dual_stagnation = 0
        hits_all = []
        hits_r0 = None
        hits_r1 = None

        for aid, inf in infos.items():
            if not isinstance(inf, dict):
                continue
            if inf.get("full_success"):
                full_success = True
            if inf.get("phase2_started"):
                phase2_started = True
            # Keep max dual stagnation counter seen
            dual_stagnation = max(dual_stagnation, inf.get("dual_stagnation_steps", 0))
            if "nav_hits" in inf:
                hits_all.append(inf["nav_hits"])
            if aid == "robot_0":
                hits_r0 = inf.get("nav_hits")
            if aid == "robot_1":
                hits_r1 = inf.get("nav_hits")

        # Scalar metrics
        episode.custom_metrics["success"] = 1.0 if full_success else 0.0
        episode.custom_metrics["full_success"] = 1.0 if full_success else 0.0
        episode.custom_metrics["phase2_started"] = 1.0 if phase2_started else 0.0
        if hits_all:
            episode.custom_metrics["nav_hits_mean"] = float(np.mean(hits_all))
        if hits_r0 is not None:
            episode.custom_metrics["nav_hits_r0"] = hits_r0
        if hits_r1 is not None:
            episode.custom_metrics["nav_hits_r1"] = hits_r1
        episode.custom_metrics["dual_stagnation_steps"] = dual_stagnation


class RLTrainer:
    def __init__(
        self,
        env_class,
        env_config: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs",
        log_name: str = "rl_training.log",
        num_rollout_workers: Optional[int] = None,
        train_batch_size: int = 6000,
        sgd_minibatch_size: int = 512,
        lr: float = 5e-5,
        entropy_coeff_start: float = 0.02,
        entropy_coeff_min: float = 0.005,
        entropy_decay: float = 0.995,
        curriculum_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
        checkpoint_dir: str = "checkpoints"
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
            .callbacks(MultiAgentMetricsCallback)
            .rollouts(
                num_rollout_workers=num_rollout_workers,
                num_envs_per_worker=num_envs_per_worker,
                rollout_fragment_length=64,
                batch_mode="truncate_episodes"
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
            vf_clip_param=5.0,
            vf_loss_coeff=1.5,
            grad_clip=1.0,
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
        self.config.train_batch_size = min(self.config.train_batch_size, 3072)  # cap
        self.config.compress_observations = True
        self.trainer = PPOTrainer(config=self.config)
        self.logger.info(
            f"Initialized PPO (workers={num_rollout_workers}, train_batch={train_batch_size}, multi_agent={is_multi})"
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
        # Extended header (add new metrics at the end but before time fields for clarity)
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
        return result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}) or {}

    def _append_metrics(self, result: Dict[str, Any]):
        ls = self._extract_learner_stats(result)
        cm = result.get("custom_metrics", {}) or {}

        def g(key):
            return cm.get(key)

        row = [
            result.get("training_iteration"),
            result.get("timesteps_total"),
            result.get("timesteps_this_iter"),
            result.get("episodes_total"),
            result.get("episodes_this_iter"),
            result.get("episode_len_mean"),
            result.get("episode_reward_mean"),
            result.get("episode_reward_min"),
            result.get("episode_reward_max"),
            g("success_mean"),
            g("full_success_mean"),
            g("phase2_started_mean"),
            g("nav_hits_mean_mean"),
            g("nav_hits_r0_mean"),
            g("nav_hits_r1_mean"),
            g("dual_stagnation_steps_mean"),
            ls.get("entropy"),
            ls.get("policy_loss"),
            ls.get("vf_loss"),
            ls.get("vf_explained_var"),
            ls.get("kl"),
            ls.get("cur_kl_coeff"),
            self.entropy_coeff,
            self._last_iter_time_s,
            self._wall_time_total
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
        if self._last_result:
            try:
                with open(os.path.join(ckpt_path, "result.json"), "w") as f:
                    json.dump(self._last_result, f, indent=2, default=str)
            except Exception as e:
                self.logger.warning(f"Could not write result.json: {e}")
        self.logger.info(f"Checkpoint saved: {ckpt_path}")
        return ckpt_path

    def restore(self, checkpoint_path: str):
        self.trainer.restore(checkpoint_path)
        self.logger.info(f"Restored from checkpoint: {checkpoint_path}")

    # ---------- Scheduling / curricula ----------
    def _decay_entropy(self):
        if self.entropy_coeff > self.entropy_coeff_min:
            self.entropy_coeff = max(self.entropy_coeff_min,
                                     self.entropy_coeff * self.entropy_decay)
            for _, policy in self.trainer.workers.local_worker().policy_map.items():
                policy.config["entropy_coeff"] = self.entropy_coeff

    def _apply_curriculum(self, local_iter: int):
        if not self.curriculum_fn:
            return
        updates = self.curriculum_fn(local_iter)
        if not updates:
            return
        def _upd(env):
            for k, v in updates.items():
                if hasattr(env, k):
                    setattr(env, k, v)
        self.trainer.workers.foreach_env(_upd)

    # ---------- Public API ----------
    def train(
        self,
        iterations: int = 1,
        checkpoint_every_global: int = 10,
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
        iterator = trange(iterations, desc="Remaining", leave=True) if USE_TQDM and iterations > 1 else range(iterations)

        success_hist: List[float] = []
        reward_hist: List[float] = []

        last_result = None
        for _local in iterator:
            iter_t0 = time.perf_counter()
            self._apply_curriculum(_local)
            result = self.trainer.train()
            global_iter = result.get("training_iteration")

            if global_iter % 1 == 0:
                self.logger.info(f"[Memory] Iter {global_iter} RSS_MB={self._get_rss_mb():.1f}")

            self._last_result = result
            self._last_iter_time_s = time.perf_counter() - iter_t0
            self._wall_time_total += self._last_iter_time_s

            # Use full_success_mean if present else success_mean as display
            success_rate = (
                result.get("custom_metrics", {}).get("full_success_mean",
                    result.get("custom_metrics", {}).get("success_mean"))
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