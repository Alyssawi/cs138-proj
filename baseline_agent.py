import random
from typing import Any

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium.spaces import Box
from gymnasium.wrappers import AtariPreprocessing, ReshapeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from curriculum_learning_env import CurriculumLearningEnv

LOG_DIR = "./logs/"
CHECKPOINT_DIR = "./checkpoints/"


gym.register_envs(ale_py)


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


# class NewAxisWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)

# def observation(self, observation: Any) -> Any:
#     obs = super().observation(observation)
#     print("AAAAA")
#     return obs[None, ...]


def make_env():
    base_env = gym.make(
        "ALE/Galaxian-v5",
        difficulty=0,
        frameskip=1,
        render_mode=None,
    )
    base_env = AtariPreprocessing(
        base_env,
        frame_skip=4,
        screen_size=84,
        scale_obs=True,
        terminal_on_life_loss=False,
    )
    base_env = ReshapeObservation(base_env, (1, 84, 84))
    env = CurriculumLearningEnv("ALE/Galaxian-v5", frameskip=1, render_mode=None)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True,
        terminal_on_life_loss=False,
    )
    env = ReshapeObservation(base_env, (1, 84, 84))

    return base_env, env


# 4 instances of game
# vec_env = make_vec_env(make_env, n_envs=4)


def train():
    base_env, env = make_env()
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=CHECKPOINT_DIR)
    eval_callback = EvalCallback(
        base_env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path=CHECKPOINT_DIR,
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs,
    )

    model.learn(1_000_000, callback=[checkpoint_callback, eval_callback])
    model.save("ppo")


def test():
    model = PPO.load("./checkpoints/best_model")

    obs = vec_env.reset()
    done = False
    total_reward = [0, 0, 0, 0]
    timestep = 0
    while not done:
        timestep += 1
        # action = [random.randint(0, 5) for _ in range(4)]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_vec, _ = vec_env.step(action)
        done = done_vec.any()

        cv2.imshow("obs", obs[3])
        cv2.waitKey(100)

        # vec_env.render("human")

        total_reward = np.select(done_vec, np.zeros(4), total_reward + reward)

        print(done_vec)
        print(timestep, total_reward)


train()
