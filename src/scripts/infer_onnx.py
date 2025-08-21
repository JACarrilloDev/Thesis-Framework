import argparse, os, numpy as np, onnxruntime as ort
from examples.multirobot.envs.multirobot_env import DynamicTwoPhaseNavEnv

def load_sessions(onnx_dir):
    sessions = {}
    for f in os.listdir(onnx_dir):
        if f.endswith(".onnx"):
            name = os.path.splitext(f)[0]
            sessions[name] = ort.InferenceSession(os.path.join(onnx_dir, f))
    return sessions

def select_policy(session_map, agent_id):
    # If only one (shared policy) return it
    if "shared_policy" in session_map:
        return session_map["shared_policy"]
    # Else map robot_0 -> first, robot_1 -> second by sorted name
    keys = sorted(session_map.keys())
    idx = 0 if agent_id == "robot_0" else min(1, len(keys)-1)
    return session_map[keys[idx]]

def dict_obs_to_inputs(obs, sess):
    # Expect input names either ['vect','img'] or ['obs']
    inputs = {}
    names = [i.name for i in sess.get_inputs()]
    if "obs" in names:  # flat
        inputs["obs"] = obs.astype(np.float32)[None, ...]
    else:
        # vect/img
        inputs["vect"] = obs["vect"].astype(np.float32)[None, ...]
        inputs["img"] = obs["img"].astype(np.float32)[None, ...]
    return inputs

def logits_to_action(logits, action_space):
    # PPO policy usually has logits for discrete; here continuous -> model exported logits head; you may need policy logic.
    # Simplest: tanh of logits scaled to action space range.
    out = np.tanh(logits)
    low, high = action_space.low, action_space.high
    return low + (out[0] + 1.0) * 0.5 * (high - low)

def run_inference(onnx_dir, env_config, episodes):
    env = DynamicTwoPhaseNavEnv(env_config)
    sessions = load_sessions(onnx_dir)

    for ep in range(episodes):
        obs_dict, _ = env.reset()
        done = False
        total_rew = {aid:0.0 for aid in obs_dict.keys()}
        steps = 0
        while not done:
            act_dict = {}
            for aid, obs in obs_dict.items():
                sess = select_policy(sessions, aid)
                inputs = dict_obs_to_inputs(obs, sess)
                logits = sess.run(None, inputs)[0]
                act = logits_to_action(logits, env.action_space)
                act_dict[aid] = act
            obs_dict, rew_dict, done_dict, info_dict = env.step(act_dict)
            for aid, r in rew_dict.items():
                total_rew[aid] += r
            done = done_dict["__all__"]
            steps += 1
        print(f"Episode {ep+1} steps={steps} rewards={total_rew} success={info_dict['robot_0'].get('success', False)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--scene_file", default="examples/navigation/scenes/empty_arena_two_robots.ttt")
    ap.add_argument("--use_camera", action="store_true")
    args = ap.parse_args()

    env_config = {
        "scene_file": args.scene_file,
        "robot_type": "AstiPioneerHybrid",
        "robot_names_in_scene": ["AstiPioneer1","AstiPioneer2"],
        "task_config": {},
        "use_camera": args.use_camera,
        "multi_agent": True
    }
    run_inference(args.onnx_dir, env_config, args.episodes)