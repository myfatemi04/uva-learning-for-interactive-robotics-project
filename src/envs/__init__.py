from typing import Callable
import warnings
from copy import deepcopy
from functools import partial

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import Tensorize


def missing_dependencies(task):
    raise ValueError(
        f"Missing dependencies for task {task}; install dependencies to use this environment."
    )


try:
    from envs.dmcontrol import make_env as make_dm_control_env
except:
    make_dm_control_env = missing_dependencies
try:
    from envs.maniskill import make_env as make_maniskill_env
except:
    make_maniskill_env = missing_dependencies
try:
    from envs.metaworld import make_env as make_metaworld_env
except:
    make_metaworld_env = missing_dependencies
try:
    from envs.myosuite import make_env as make_myosuite_env
except:
    make_myosuite_env = missing_dependencies


def is_dm_control_env(task: str):
    from dm_control.suite import TASKS_BY_DOMAIN

    domain, task = task.replace("-", "_").split("_", 1)

    return domain in TASKS_BY_DOMAIN and task in TASKS_BY_DOMAIN[domain]


def is_maniskill_env(task: str):
    from envs.maniskill import MANISKILL_TASKS

    return task in MANISKILL_TASKS


def is_metaworld_env(task: str):
    return task.startswith("mw-")


def is_myosuite_env(task: str):
    return task.startswith("myo-")


warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_multitask_env(cfg):
    """
    Make a multi-task environment for TD-MPC2 experiments.
    """
    print("Creating multi-task environment with tasks:", cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError("Unknown task:", task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env


def make_env(cfg):
    """
    Make an environment for TD-MPC2 experiments.
    """
    gym.logger.min_level = 40
    if cfg.multitask:
        env = make_multitask_env(cfg)

    else:
        if is_dm_control_env(cfg.task):
            make_env = make_dm_control_env
            max_steps = 500
        elif is_metaworld_env(cfg.task):
            make_env = make_metaworld_env
            max_steps = 100
        elif is_maniskill_env(cfg.task):
            make_env = make_maniskill_env
            max_steps = 100
        elif is_myosuite_env(cfg.task):
            make_env = make_myosuite_env
            max_steps = 100
        else:
            raise ValueError(f"Unknown task: {cfg.task}")

        try:
            env = make_env(cfg)
            print("make_env env action space shape:", env.action_space.shape)
        except ValueError:
            raise ValueError(
                f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.'
            )

        if cfg.get("obs", "state") == "rgb":
            make_env_ = lambda: PixelWrapper(
                make_env(cfg), num_frames=3, render_size=64
            )
        else:
            make_env_ = lambda: make_env(cfg)

        env = Tensorize(SyncVectorEnv([make_env_ for _ in range(cfg.num_envs)]))

    # Remove the "vectorization" dimension for the observation space by doing ".shape[1:]".
    if isinstance(env.observation_space, gym.spaces.Dict):
        cfg.obs_shape = {k: v.shape[1:] for k, v in env.observation_space.spaces.items()}
    elif isinstance(env.observation_space, gym.spaces.Box):
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape[1:]}
    else:
        raise NotImplementedError("Unknown observation space:", env.observation_space)

    assert env.action_space.shape is not None

    # Also remove the "vectorization" dimension for the action space by selecting index 1 instead of 0.
    cfg.action_dim = env.action_space.shape[1]
    cfg.episode_length = max_steps
    cfg.seed_steps = max(1000, 5 * cfg.episode_length) * cfg.num_envs
    return env
