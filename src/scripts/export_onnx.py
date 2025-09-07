import argparse, os, sys, json
import numpy as np

import torch
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.algorithms.ppo import PPOConfig
import yaml
import functools
from torch.onnx import TrainingMode
import warnings
import logging

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", message="Constant folding - Only steps=1 can be constant folded*", category=UserWarning)
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean might cause the trace to be incorrect*", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", message="Using len to get tensor shape might cause the trace to be incorrect*", category=torch.jit.TracerWarning)
logging.getLogger("ray").setLevel(logging.ERROR)

from run_rl_task import build_env_config, promote_nested_keys, import_env_class


def safe_onnx_export(model, args, f, input_names, output_names, opset_version, dynamic_axes):
    # Always use stable API and set EVAL mode explicitly
    return torch.onnx.export(
        model,
        args,
        f,
        export_params=True,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        training=TrainingMode.EVAL,
    )


def _list_policies(ckpt_dir: str):
    pdir = os.path.join(ckpt_dir, "policies")
    if not os.path.isdir(pdir):
        return []
    return sorted([d for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d))])


def _load_saved_cfg(ckpt_dir: str):
    path = os.path.join(ckpt_dir, "rllib_checkpoint.json")
    try:
        with open(path, "r") as f:
            data = json.load(f) or {}
        return data.get("config") or {}
    except Exception:
        return {}

def _dist_class_name(policy) -> str:
    d = getattr(policy, "dist_class", None)
    if d is None:
        return "None"
    try:
        if isinstance(d, functools.partial):
            return getattr(d.func, "__name__", str(d.func))
        return getattr(d, "__name__", type(d).__name__)
    except Exception:
        return str(d)

def _build_trainer_from_checkpoint(checkpoint_dir: str, env_class, yaml_env_config: dict) -> PPOTrainer:
    # Best path: construct exactly as saved.
    if hasattr(PPOTrainer, "from_checkpoint"):
        try:
            return PPOTrainer.from_checkpoint(checkpoint_dir)
        except Exception:
            pass

    # Fallback: recreate PPOConfig from saved config, override env binding and worker counts only.
    saved_cfg = _load_saved_cfg(checkpoint_dir)
    cfg = PPOConfig().from_dict(saved_cfg) if saved_cfg else PPOConfig()

    env_cfg = (saved_cfg.get("env_config") if saved_cfg else None) or yaml_env_config
    cfg = (
        cfg
        .environment(env=env_class, env_config=env_cfg, disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=0, create_env_on_local_worker=True)
        .resources(num_gpus=0)
        .evaluation(evaluation_num_workers=0, evaluation_interval=None, evaluation_parallel_to_training=False)
    )

    trainer = PPOTrainer(config=cfg)
    trainer.restore(checkpoint_dir)
    return trainer


def export_policies_to_onnx(checkpoint_dir, env_class, yaml_env_config, out_dir, policy_name=None, opset=12):
    os.makedirs(out_dir, exist_ok=True)

    # Restore algorithm with matching architecture
    trainer = _build_trainer_from_checkpoint(checkpoint_dir, env_class, yaml_env_config)

    try:
        # Prefer policy list from checkpoint dir; else enumerate restored map.
        pol_ids = _list_policies(checkpoint_dir) or list(trainer.workers.local_worker().policy_map.keys())
        for pname, policy in trainer.workers.local_worker().policy_map.items():
            if policy_name and pname != policy_name:
                continue
            if pol_ids and pname not in pol_ids:
                continue

            model = policy.model
            model.eval()

            obs_space = policy.observation_space
            act_space = policy.action_space
            is_discrete = isinstance(act_space, spaces.Discrete)

            # Metadata
            meta = {
                "policy_name": pname,
                "obs_type": "dict" if isinstance(obs_space, spaces.Dict) else "box",
                "obs_shapes": (
                    {k: list(v.shape) for k, v in obs_space.spaces.items()} if isinstance(obs_space, spaces.Dict)
                    else list(obs_space.shape)
                ),
                "action_space": ("Discrete", int(act_space.n)) if is_discrete else ("Box", list(act_space.shape)),
                "action_low": (None if is_discrete else act_space.low.tolist()),
                "action_high": (None if is_discrete else act_space.high.tolist()),
                "dist_class": _dist_class_name(policy),
                "model_config": getattr(policy, "config", {}).get("model", {}),
            }

            onnx_path = os.path.join(out_dir, f"{pname}.onnx")
            meta_path = os.path.join(out_dir, f"{pname}.json")

            # Deterministic export wrapper
            with torch.no_grad():
                if isinstance(obs_space, spaces.Dict):
                    class DictWrapper(torch.nn.Module):
                        def __init__(self, pol):
                            super().__init__()
                            self.pol = pol
                            self.mdl = pol.model
                            self.Dist = pol.dist_class
                        def forward(self, vect, img):
                            obs = {"vect": vect, "img": img}
                            out, _ = self.mdl({"obs": obs}, [], None)
                            dist = self.Dist(out, self.mdl)
                            action = dist.deterministic_sample()
                            policy_out = out if isinstance(self.pol.action_space, spaces.Discrete) else getattr(dist, "loc", action)
                            value = self.mdl.value_function()
                            return action, policy_out, value

                    wrapper = DictWrapper(policy)
                    vect_shape = obs_space["vect"].shape
                    img_shape = obs_space["img"].shape
                    dummy_vect = torch.zeros((1,) + vect_shape, dtype=torch.float32)
                    dummy_img = torch.zeros((1,) + img_shape, dtype=torch.float32)

                    safe_onnx_export(
                        wrapper, (dummy_vect, dummy_img), onnx_path,
                        input_names=["vect", "img"],
                        output_names=["action", "policy_out", "value"],
                        opset_version=opset,
                        dynamic_axes={
                            "vect": {0: "batch"}, "img": {0: "batch"},
                            "action": {0: "batch"}, "policy_out": {0: "batch"}, "value": {0: "batch"}
                        },
                    )
                else:
                    class FlatWrapper(torch.nn.Module):
                        def __init__(self, pol):
                            super().__init__()
                            self.pol = pol
                            self.mdl = pol.model
                            self.Dist = pol.dist_class
                        def forward(self, obs):
                            out, _ = self.mdl({"obs": obs}, [], None)
                            dist = self.Dist(out, self.mdl)
                            action = dist.deterministic_sample()
                            policy_out = out if isinstance(self.pol.action_space, spaces.Discrete) else getattr(dist, "loc", action)
                            value = self.mdl.value_function()
                            return action, policy_out, value

                    wrapper = FlatWrapper(policy)
                    shape = obs_space.shape
                    dummy = torch.zeros((1,) + shape, dtype=torch.float32)

                    safe_onnx_export(
                        wrapper, dummy, onnx_path,
                        input_names=["obs"],
                        output_names=["action", "policy_out", "value"],
                        opset_version=opset,
                        dynamic_axes={
                            "obs": {0: "batch"},
                            "action": {0: "batch"},
                            "policy_out": {0: "batch"},
                            "value": {0: "batch"}
                        },
                    )

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"Exported policy '{pname}' to {onnx_path} (action_space={meta['action_space']})")
    finally:
        try:
            trainer.stop()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--out_dir", default="onnx_export")
    ap.add_argument("--task_yaml", required=True, help="Task YAML used during training (only used to locate env_class)")
    ap.add_argument("--policy_name", help="Export only this policy (e.g., shared_policy)")
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    with open(args.task_yaml, "r") as f:
        root_cfg = yaml.safe_load(f) or {}
    promote_nested_keys(root_cfg)

    env_class_path = root_cfg.get("env_class")
    if not env_class_path:
        raise SystemExit("YAML is missing env_class.")
    env_class = import_env_class(env_class_path)

    # Only use YAML to import env_class; env_config for build fallback
    yaml_env_config = build_env_config(root_cfg, headless=True)

    export_policies_to_onnx(
        args.checkpoint_dir,
        env_class,
        yaml_env_config,
        args.out_dir,
        policy_name=args.policy_name,
        opset=args.opset,
    )