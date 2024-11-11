from typing import Union

import gymnasium as gym
import torch
from tensordict import TensorDict


Tensorish = Union[torch.Tensor, TensorDict]


def _try_f32_tensor(x) -> torch.Tensor:
    x = torch.from_numpy(x)
    if x.dtype == torch.float64:
        x = x.float()
    return x


def _obs_to_tensor(obs) -> Tensorish:
    if isinstance(obs, dict):
        return TensorDict({k: _try_f32_tensor(v) for k, v in obs.items()})
    else:
        return _try_f32_tensor(obs)


class Tensorize(gym.Wrapper):
    """
    Wrapper for converting numpy arrays to torch tensors.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._wrapped_vectorized = env.__class__.__name__ == "Vectorized"

    def reset(self, **kwargs) -> tuple[Tensorish, dict]:
        obs, info = self.env.reset(**kwargs)
        return _obs_to_tensor(obs), info

    def step(self, action: torch.Tensor, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(
            action.numpy(), **(kwargs if self._wrapped_vectorized else {})
        )

        info = {key: torch.tensor(value) for (key, value) in info.items()}
        if "success" not in info.keys():
            if type(terminated) is not bool:
                info["success"] = torch.zeros(len(terminated))
            else:
                info["success"] = torch.tensor(0)

        return (
            _obs_to_tensor(obs),
            torch.tensor(reward, dtype=torch.float32),
            terminated,
            truncated,
            info,
        )
