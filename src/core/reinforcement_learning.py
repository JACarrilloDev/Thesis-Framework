import ray
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.algorithms.ppo import PPOConfig
from datetime import datetime
import os

from src.core.logger import setup_logger

try:
    from tqdm import trange
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

class RLTrainer:
    def __init__(self, env_class, env_config=None, log_dir="logs", log_name="rl_training.log"):
        self.env_class = env_class
        self.env_config = env_config or {}
        worker_env_config = {
            **self.env_config,
            "headless": True  # Workers run headless
        }

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        self.logger = setup_logger("RLTrainerLogger", os.path.join(log_dir, log_name), console_output=True)
        self.config = (
            PPOConfig()
            .environment(env=self.env_class, env_config=self.env_config, disable_env_checking=True)
            .framework("torch")
            .rollouts(
                num_rollout_workers=1,
                num_envs_per_worker=1,
                rollout_fragment_length=200,  # Shorter fragments for faster updates
                batch_mode="complete_episodes"
            )
            .training(
                model={
                    # "conv_filters": [
                    #     [16, [8, 8], 4],
                    #     [32, [4, 4], 2],
                    #     [64, [3, 3], 1],
                    # ],
                    "fcnet_hiddens": [128, 128],
                    "fcnet_activation": "tanh",
                    "vf_share_layers": False,
                },
            lr=5e-5,  # Slightly higher learning rate
            gamma=0.99,
            entropy_coeff=0.02,
            clip_param=0.2,
            vf_clip_param=1.0,  # Reduced from 10.0
            grad_clip=1.0,
            sgd_minibatch_size=64,
            num_sgd_iter=4,
            train_batch_size=1000,  # Increased slightly
            lambda_=0.95,
            # lr_schedule=[[0, 1e-4], [1000000, 5e-5]]  # Learning rate decay
        )
            .resources(num_cpus_per_worker=1,num_gpus=0)
        )
        self.trainer = PPOTrainer(config=self.config)

    def train(self, iterations=300, checkpoint_every=10):
        """Train with consistent checkpointing"""
        log = self.logger
        log.info(f"Starting RL training for {iterations} iterations at {datetime.now().isoformat()}")
        iter_range = trange(iterations, desc="Training iterations") if USE_TQDM else range(iterations)
        for i in iter_range:
            result = self.trainer.train()
            log_msg = (
                f"Iteration {i}: "
                f"mean_reward={result.get('episode_reward_mean', 'N/A'):.2f}, "
                f"max_reward={result.get('episode_reward_max', 'N/A')}, "
                f"min_reward={result.get('episode_reward_min', 'N/A')}, "
                f"episodes_this_iter={result.get('episodes_this_iter', 'N/A')}, "
                f"timesteps_total={result.get('timesteps_total', 'N/A')}, "
                f"policy_loss={result.get('info', {}).get('learner', {}).get('default_policy', {}).get('policy_loss', 'N/A')}, "
                f"vf_loss={result.get('info', {}).get('learner', {}).get('default_policy', {}).get('vf_loss', 'N/A')}, "
                f"entropy={result.get('info', {}).get('learner', {}).get('default_policy', {}).get('entropy', 'N/A')}"
            )
            print(log_msg)
            log.info(log_msg)
            if i % checkpoint_every == 0 or i == iterations - 1:
                checkpoint_path = self.trainer.save("checkpoints/")
                log.info(f"Checkpoint saved at iteration {i}: {checkpoint_path}")

    def evaluate(self, episodes=10):
        log = self.logger
        log.info(f"Starting evaluation for {episodes} episodes at {datetime.now().isoformat()}")
        results = self.trainer.evaluate()
        log.info(f"Evaluation results: {results}")
        print(f"Evaluation results: {results}")
        return results

    def restore(self, checkpoint_path):
        """Restore the underlying RLlib trainer from a checkpoint directory."""
        self.trainer.restore(checkpoint_path)