from collections import defaultdict
from typing import Literal

import gymnasium as gym
import numpy as np
from ale_py.env import AtariEnv
from gymnasium import Env


def softmax(x):
    x_max = np.amax(x, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, keepdims=True)


class CurriculumLearningEnv(Env):
    def __init__(
        self,
        game: str,
        obs_type: Literal["rgb", "grayscale", "ram"] = "rgb",
        frameskip: tuple[int, int] | int = 4,
        render_mode: Literal["human", "rgb_array"] | None = None,
        softmax: bool = False,
        noise: bool = False,
    ):
        self._game = gym.make(
            game, obs_type=obs_type, frameskip=frameskip, render_mode=render_mode
        )

        self._frameskip = frameskip

        self.total_steps = 0
        self.step_count = defaultdict(lambda: 0)
        self.states_seen = {}

        self.observation_space = self._game.observation_space
        self.action_space = self._game.action_space
        self.metadata = self._game.metadata

        self.ale = self._game.unwrapped.ale

        self.softmax = softmax
        self.noise = noise

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.current_step = 0

        if self.total_steps == 0 or np.random.random() < 0.1:
            return self._game.reset(seed=seed, options=options)

        counts = np.array(list(self.step_count.values()))

        if self.softmax:
            p = softmax(-counts)
        else:
            base_probs = 1 / counts
            p = np.where(
                np.sum(base_probs) == 0,
                np.full_like(base_probs, 1) / len(base_probs),
                base_probs / np.sum(base_probs),
            )

        if self.noise:
            noise = np.random.dirichlet(np.full_like(p, 0.3))
            p = 0.75 * p + 0.25 * noise

        replay_index = np.random.choice(
            list(self.step_count.keys()),
            p=p,
        )

        state, obs, info = self.states_seen[replay_index]

        AtariEnv.restore_state(self._game.unwrapped, state)

        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self._game.step(action)

        self.step_count[self.current_step] += 1
        self.states_seen[self.current_step] = (
            AtariEnv.clone_state(self._game.unwrapped),
            obs,
            info,
        )
        self.current_step += 1
        self.total_steps += 1

        return obs, reward, term, trunc, info

    def render(self, *args, **kwargs):
        self._game.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        self._game.render(*args, **kwargs)

    def get_action_meanings(self, *args, **kwargs):
        return self._game.unwrapped.get_action_meanings(*args, **kwargs)
