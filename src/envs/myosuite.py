import numpy as np
import gymnasium as gym
from envs.wrappers.time_limit import TimeLimit


MYOSUITE_TASKS = {
    "myo-reach": "myoHandReachFixed-v0",
    "myo-reach-hard": "myoHandReachRandom-v0",
    "myo-pose": "myoHandPoseFixed-v0",
    "myo-pose-hard": "myoHandPoseRandom-v0",
    "myo-obj-hold": "myoHandObjHoldFixed-v0",
    "myo-obj-hold-hard": "myoHandObjHoldRandom-v0",
    "myo-key-turn": "myoHandKeyTurnFixed-v0",
    "myo-key-turn-hard": "myoHandKeyTurnRandom-v0",
    "myo-pen-twirl": "myoHandPenTwirlFixed-v0",
    "myo-pen-twirl-hard": "myoHandPenTwirlRandom-v0",
}


class MyoSuiteWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, cfg, render_size: int):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.camera_id = "hand_side_inter"
        self.render_size = render_size

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        info["success"] = info["solved"]
        return obs, reward, False, False, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.sim.renderer.render_offscreen(
            width=self.render_size,
            height=self.render_size,
            camera_id=self.camera_id,
        ).copy()


def make_env(cfg, render_size: int = 384):
    """
    Make Myosuite environment.
    """
    if not cfg.task in MYOSUITE_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "state", "This task only supports state observations."
    import myosuite

    env = gym.make(MYOSUITE_TASKS[cfg.task])
    env = MyoSuiteWrapper(env, cfg, render_size)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
