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
        self.action_space = env.action_space
        self._frames: deque[torch.Tensor] = deque([], maxlen=num_frames)
        self.num_frames = num_frames
        self._render_size = render_size

    def _render_as_tensor(self) -> torch.Tensor:
        # Render as rgb_array.
        frame: np.ndarray = self.env.render()  # type: ignore
        frame = frame.transpose(2, 0, 1).copy()
        return torch.from_numpy(frame)

    def reset(self, seed=None, options=None) -> tuple[torch.Tensor, dict]:
        self.env.reset(seed=seed, options=options)

        # Populate the buffer of frames.
        frame = self._render_as_tensor()
        for _ in range(self.num_frames):
            self._frames.append(frame)

        # Concatenate the most recent frames to create an observation.
        buf = torch.cat([*self._frames], dim=0)
        return buf, {}

    def step(self, action):
        _original_obs, reward, terminated, truncated, info = self.env.step(action)

        # Render the environment and add to frame buffer.
        frame = self._render_as_tensor()
        self._frames.append(frame)

        # Concatenate the most recent frames to create an observation.
        buf = torch.cat([*self._frames], dim=0)
        return buf, reward, terminated, truncated, info
