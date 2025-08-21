import gymnasium as gym

class BaseEnv(gym.Env):
    def __init__(self, env_config):
        super().__init__()
        self.env_config = env_config
        # Common setup (simulation, robot, task, etc.)

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def _compute_reward(self, obs):
        raise NotImplementedError