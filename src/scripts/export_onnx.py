import argparse, os, numpy as np
import sys 

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium import spaces
import yaml

# Ensure project root on sys.path (run from repo root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from run_rl_task import build_env_config, promote_nested_keys, import_env_class
import yaml 

def _dummy_obs_from_space(space):
    if isinstance(space, spaces.Dict):
        return {k: _dummy_obs_from_space(v) for k,v in space.spaces.items()}
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    raise ValueError("Unsupported space type for dummy obs export.")

def export_policies_to_onnx(checkpoint_dir, env_class, env_config, out_dir, policy_name=None, opset=12):
    os.makedirs(out_dir, exist_ok=True)

    cfg = (
        PPOConfig()
        .environment(env=env_class, env_config=env_config, disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=0, create_env_on_local_worker=False)  # evitar lanzar env
        .resources(num_gpus=0)
    )

    trainer = PPOTrainer(config=cfg)
    trainer.restore(checkpoint_dir)
    try:
        for pname, policy in trainer.workers.local_worker().policy_map.items():
            if policy_name and pname != policy_name:
                continue
            model = policy.model
            model.eval()

            obs_space = policy.observation_space
            act_space = policy.action_space

            meta = {
                "policy_name": pname,
                "obs_type": "dict" if isinstance(obs_space, spaces.Dict) else "box",
                "obs_shapes": (
                    {k: list(v.shape) for k, v in obs_space.spaces.items()} if isinstance(obs_space, spaces.Dict)
                    else list(obs_space.shape)
                ),
                "action_low": act_space.low.tolist(),
                "action_high": act_space.high.tolist(),
            }

            onnx_path = os.path.join(out_dir, f"{pname}.onnx")
            meta_path = os.path.join(out_dir, f"{pname}.json")

            with torch.no_grad():
                if isinstance(obs_space, spaces.Dict):
                    class Wrapper(torch.nn.Module):
                        def __init__(self, mdl):
                            super().__init__(); self.mdl = mdl
                        def forward(self, vect, img):
                            obs = {"vect": vect, "img": img}
                            out, _ = self.mdl({"obs": obs}, [], None)
                            return out
                    wrapper = Wrapper(model)
                    vect_shape = obs_space["vect"].shape
                    img_shape = obs_space["img"].shape
                    dummy_vect = torch.zeros((1,)+vect_shape, dtype=torch.float32)
                    dummy_img = torch.zeros((1,)+img_shape, dtype=torch.float32)
                    torch.onnx.export(
                        wrapper, (dummy_vect, dummy_img), onnx_path,
                        input_names=["vect","img"], output_names=["logits"],
                        opset_version=opset,
                        dynamic_axes={"vect": {0: "batch"}, "img": {0: "batch"}, "logits": {0: "batch"}},
                    )
                else:
                    class FlatWrapper(torch.nn.Module):
                        def __init__(self, mdl): super().__init__(); self.mdl = mdl
                        def forward(self, obs):
                            out,_ = self.mdl({"obs": obs}, [], None)
                            return out
                    wrapper = FlatWrapper(model)
                    shape = obs_space.shape
                    dummy = torch.zeros((1,)+shape, dtype=torch.float32)
                    torch.onnx.export(
                        wrapper, dummy, onnx_path,
                        input_names=["obs"], output_names=["logits"],
                        opset_version=opset,
                        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
                    )

            # Write small metadata to help infer script
            try:
                import json
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

            print(f"Exported policy '{pname}' to {onnx_path} (act_space={act_space})")
    finally:
        try:
            trainer.stop()
        except Exception:
            pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--out_dir", default="onnx_exports")
    ap.add_argument("--task_yaml", help="YAML usado en entrenamiento (recomendado)")
    ap.add_argument("--scene_file", default="examples/navigation/scenes/empty_arena_two_robots.ttt")
    ap.add_argument("--use_camera", action="store_true")
    ap.add_argument("--policy_name", help="Exportar solo esta política (p.ej., shared_policy)")
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    env_class = None
    env_config = None
    if args.task_yaml:
        with open(args.task_yaml, "r") as f:
            root_cfg = yaml.safe_load(f) or {}
        promote_nested_keys(root_cfg)
        # Importar env_class desde YAML
        env_class_path = root_cfg.get("env_class")
        if not env_class_path:
            raise SystemExit("YAML no define env_class.")
        env_class = import_env_class(env_class_path)
        env_config = build_env_config(root_cfg, headless=True)
        if args.use_camera:
            env_config["use_camera"] = True
    else:
        # Fallback mínimo
        from examples.multirobot.envs.multirobot_env import DynamicTwoPhaseNavEnv as FallbackEnv
        env_class = FallbackEnv
        env_config = {
            "scene_file": args.scene_file,
            "robot_type": "AstiPioneerHybrid",
            "robot_names_in_scene": ["AstiPioneer1","AstiPioneer2"],
            "task_config": {},
            "use_camera": args.use_camera,
            "multi_agent": True,
            "headless": True,
        }

    export_policies_to_onnx(args.checkpoint_dir, env_class, env_config, args.out_dir, policy_name=args.policy_name, opset=args.opset)