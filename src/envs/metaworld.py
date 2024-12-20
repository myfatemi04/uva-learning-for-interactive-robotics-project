import numpy as np
import gymnasium as gym
from envs.wrappers.time_limit import TimeLimit

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaWorldWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        cfg,
        render_size: int,
    ):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.camera_name = "corner2"
        self.render_size = render_size
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env._freeze_rand_vec = False

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = obs.astype(np.float32)
        self.env.step(np.zeros(self.env.action_space.shape))
        return obs, info

    def step(self, action):
        reward = 0
        for _ in range(2):
            obs, r, terminated, truncated, info = self.env.step(action.copy())
            reward += r
        obs = obs.astype(np.float32)
        return obs, reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render(
            resolution=(self.render_size, self.render_size),
            camera_name=self.camera_name,
        ).copy()


def make_env(cfg):
    """
    Make Meta-World environment.
    """
    env_id = cfg.task.split("-", 1)[-1] + "-v2-goal-observable"

    if (
        not cfg.task.startswith("mw-")
        or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    ):
        raise ValueError("Unknown task:", cfg.task)
    # assert cfg.obs == "state", "This task only supports state observations."
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.seed)
    env = MetaWorldWrapper(env, cfg, render_size=cfg.render_size)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
