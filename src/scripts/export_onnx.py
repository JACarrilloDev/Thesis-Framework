import argparse, os, torch, numpy as np
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.sample_batch import SampleBatch
from gymnasium import spaces

from examples.multirobot.envs.multirobot_env import DynamicTwoPhaseNavEnv

def _dummy_obs_from_space(space):
    if isinstance(space, spaces.Dict):
        return {k: _dummy_obs_from_space(v) for k,v in space.spaces.items()}
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    raise ValueError("Unsupported space type for dummy obs export.")

def export_policies_to_onnx(checkpoint_dir, env_config, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    cfg = (
        PPOConfig()
        .environment(env=DynamicTwoPhaseNavEnv, env_config=env_config, disable_env_checking=True)
        .framework("torch")
    )

    # If multi-agent policies_spec already in env_config, RLlib reconstructs them
    trainer = PPOTrainer(config=cfg)
    trainer.restore(checkpoint_dir)

    for policy_name, policy in trainer.workers.local_worker().policy_map.items():
        model = policy.model
        model.eval()

        obs_space = policy.observation_space
        act_space = policy.action_space

        dummy_obs = _dummy_obs_from_space(obs_space)

        # RLlib ModelV2 forward expects SampleBatch or tensors; we export underlying Torch module.
        # Many RLlib models wrap a custom submodule: model.base_model OR model._logits_branch.
        # Safer: trace policy.compute_single_action path: build a tensorized batch.
        # Simpler: flatten dict into expected input ordering using model._dict_to_flattened_obs if present.
        if isinstance(obs_space, spaces.Dict):
            # Build flattened obs tensor via model's preprocess_observation (calls vision stack etc.)
            # For export, we create a wrapper module to accept two tensors (vect,img)
            class Wrapper(torch.nn.Module):
                def __init__(self, mdl, keys):
                    super().__init__()
                    self.mdl = mdl
                    self.keys = keys
                def forward(self, vect, img):
                    obs = {"vect": vect, "img": img}
                    # RLlib catalog models usually process inside forward()
                    out, _ = self.mdl({"obs": obs}, [], None)
                    return out
            wrapper = Wrapper(model, ["vect","img"])
            # Shapes
            vect_shape = obs_space["vect"].shape
            img_shape = obs_space["img"].shape
            dummy_vect = torch.zeros((1,)+vect_shape, dtype=torch.float32)
            dummy_img = torch.zeros((1,)+img_shape, dtype=torch.float32)
            onnx_path = os.path.join(out_dir, f"{policy_name}.onnx")
            torch.onnx.export(
                wrapper,
                (dummy_vect, dummy_img),
                onnx_path,
                input_names=["vect","img"],
                output_names=["logits"],
                opset_version=11
            )
        else:
            # Flat Box
            shape = obs_space.shape
            dummy = torch.zeros((1,)+shape, dtype=torch.float32)
            onnx_path = os.path.join(out_dir, f"{policy_name}.onnx")
            # Directly export model forward
            class FlatWrapper(torch.nn.Module):
                def __init__(self, mdl): super().__init__(); self.mdl = mdl
                def forward(self, obs):
                    out,_ = self.mdl({"obs": obs}, [], None)
                    return out
            wrapper = FlatWrapper(model)
            torch.onnx.export(
                wrapper,
                dummy,
                onnx_path,
                input_names=["obs"],
                output_names=["logits"],
                opset_version=11
            )
        print(f"Exported policy '{policy_name}' to {onnx_path} (act_space={act_space})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--out_dir", default="onnx_exports")
    ap.add_argument("--scene_file", default="examples/navigation/scenes/empty_arena_two_robots.ttt")
    ap.add_argument("--use_camera", action="store_true")
    args = ap.parse_args()

    env_config = {
        "scene_file": args.scene_file,
        "robot_type": "AstiPioneerHybrid",
        "robot_names_in_scene": ["AstiPioneer1","AstiPioneer2"],
        "task_config": {},  # minimal for shape; real export ideally uses same task YAML
        "use_camera": args.use_camera,
        "multi_agent": True
    }

    export_policies_to_onnx(args.checkpoint_dir, env_config, args.out_dir)