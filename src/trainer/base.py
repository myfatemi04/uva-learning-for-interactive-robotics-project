import gymnasium as gym

from common.buffer import Buffer
from common.logger import Logger
from tdmpc2 import TDMPC2


class Trainer:
    """Base trainer class for TD-MPC2."""

    def __init__(
        self,
        cfg,
        env: gym.vector.VectorEnv,
        agent: TDMPC2,
        buffer: Buffer,
        logger: Logger,
    ):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        print("Architecture:", self.agent.model)
        print("Learnable parameters: {:,}".format(self.agent.model.total_params))

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        raise NotImplementedError

    def train(self):
        """Train a TD-MPC2 agent."""
        raise NotImplementedError
