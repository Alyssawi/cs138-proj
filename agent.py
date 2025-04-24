import multiprocessing as mp
import os
import sys

import ale_py
import gymnasium as gym
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium.wrappers import AtariPreprocessing, ReshapeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from curriculum_learning_env import CurriculumLearningEnv

LOG_DIR = os.environ["LOGS_DIR"]
CHECKPOINT_DIR = (
    os.environ["CHECKPOINTS_DIR"] if "CHECKPOINTS_DIR" in os.environ else None
)


gym.register_envs(ale_py)


class SyncGlobalStepCallback(BaseCallback):
    def __init__(self, global_step, verbose=0):
        super().__init__(verbose)
        self.global_step = global_step

    def _on_step(self) -> bool:
        with self.global_step.get_lock():
            self.global_step.value = self.model.num_timesteps
        return True


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        print(observation_space.shape)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)


def make_env(
    global_step=None,
    render_mode: str | None = None,
    noise: bool = False,
    softmax: bool = False,
):
    base_env = gym.make(
        "ALE/Galaxian-v5",
        difficulty=0,
        frameskip=1,
        render_mode=render_mode,
    )
    base_env = AtariPreprocessing(
        base_env,
        frame_skip=4,
        screen_size=84,
        scale_obs=True,
        terminal_on_life_loss=False,
    )
    base_env = ReshapeObservation(base_env, (1, 84, 84))
    env = CurriculumLearningEnv(
        "ALE/Galaxian-v5",
        frameskip=1,
        noise=noise,
        softmax=softmax,
        global_step=global_step,
    )
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True,
        terminal_on_life_loss=False,
    )
    env = ReshapeObservation(env, (1, 84, 84))

    return base_env, env


def train(curriculum_learning: bool, noise: bool, softmax: bool):
    global_step = mp.Value("i", 0)
    test_env = make_vec_env(lambda: make_env()[0])
    train_env = make_vec_env(
        lambda: make_env(noise=noise, softmax=softmax, global_step=global_step)[
            curriculum_learning
        ],
        n_envs=4,
    )

    if CHECKPOINT_DIR:
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, save_path=CHECKPOINT_DIR
        )
    else:
        checkpoint_callback = None

    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path=CHECKPOINT_DIR,
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    sync_callback = SyncGlobalStepCallback(global_step)

    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs,
    )

    model.learn(
        1_000_000,
        callback=[checkpoint_callback, eval_callback, sync_callback]
        if checkpoint_callback
        else [eval_callback, sync_callback],
    )
    model.save("ppo")


def test():
    env, _ = make_env(render_mode="human")

    model = PPO.load("./checkpoints/best_model")

    obs, _ = env.reset()
    done = False
    total_reward = 0
    timestep = 0
    while not done:
        timestep += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

        total_reward = (0 if done else total_reward) + reward

        print(timestep, total_reward)


if __name__ == "__main__":
    mode = sys.argv[1]
    env = len(sys.argv) > 2 and sys.argv[2]

    if mode not in ["train", "test"]:
        print("invalid mode: ", mode)
        exit()

    if env and env not in ["curriculum"]:
        print("invalid env")
        exit()

    if mode == "test":
        test()
    else:
        noise = "--noise" in sys.argv
        softmax = "--softmax" in sys.argv
        train(env == "curriculum", noise=noise, softmax=softmax)
