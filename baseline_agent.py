import numpy as np
import random
import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import AtariPreprocessing
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

LOG_DIR = "./logs/"
CHECKPOINT_DIR = "./checkpoints/"


gym.register_envs(ale_py)

def make_env():
    env = gym.make("ALE/Galaxian-v5", difficulty=0, frameskip=1, render_mode="human")
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=False)

    return env

# 4 instances of game
vec_env = make_vec_env(make_env, n_envs=4)

def train():
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=CHECKPOINT_DIR)
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path=CHECKPOINT_DIR,
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)


    model.learn(1000000000, callback=[checkpoint_callback, eval_callback])
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
        # vec_env.render("human")

        total_reward = np.select(done_vec, np.zeros(4), total_reward + reward)

        print(done_vec)
        print(timestep, total_reward)


test()
