from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation


class GridWorld(gym.Env):

    def __init__(self, size: int = 5):
        self.size = size
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size-1, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, size-1, shape=(2,), dtype=int),
        })
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance" : np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size-1)

        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorld
)

env = gym.make("gymnasium_env/GridWorld-v0")
env.observation_space

wrapped_env = FlattenObservation(env)
wrapped_env.observation_space
