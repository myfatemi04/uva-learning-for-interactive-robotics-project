from collections import deque

import gymnasium as gym
import numpy as np
import torch


class PixelWrapper(gym.Wrapper):
    """
    Wrapper for pixel observations. Compatible with DMControl environments.
    """

    def __init__(self, env: gym.Env, num_frames: int, render_size: int):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_frames * 3, render_size, render_size),
            dtype=np.uint8,
        )
        self._frames = deque([], maxlen=num_frames)
        self._render_size = render_size
        self.max_episode_steps = env.max_episode_steps  # type: ignore

    def _get_obs(self) -> torch.Tensor:
        # Render as rgb_array
        frame: np.ndarray = self.env.render()  # type: ignore
        frame = frame.transpose(2, 0, 1)
        self._frames.append(frame)
        return torch.from_numpy(np.concatenate(self._frames))

    def reset(self, seed=None, options=None) -> tuple[torch.Tensor, dict]:
        assert seed is None, options is None
        self.env.reset()
        self._frames.clear()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        _original_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(), reward, terminated, truncated, info
