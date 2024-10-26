import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
from .rollout import Rollout


class Actor(nn.Module):
    def __init__(self, action_space_dim: int):
        super().__init__()

        self.action_space_dim = action_space_dim

    def forward(self):
        return Normal(
            torch.zeros(self.action_space_dim),
            torch.ones(self.action_space_dim),
        )


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        # This is the representation we want to learn.
        self.encoder = object()

        # These are the ways we use the representation.
        self.actor = object()
        self.critic = object()
        self.predictive_model = object()

        self.actor_coef = 1
        self.critic_coef = 1
        self.dynamics_coef = 1

    def _critic_loss(self, rollouts: Rollout):
        # TD(lambda)
        return 0.0

    def _policy_loss(self, rollouts: Rollout):
        # PPO
        return 0.0

    def _dynamics_loss(self, rollouts: Rollout):
        # CPC
        return 0.0

    def loss(self, rollouts: Rollout):
        return (
            self.critic_coef * self.critic_loss(rollouts)
            + self.actor_coef * self.policy_loss(rollouts)
            + self.dynamics_coef * self.dynamics_loss(rollouts)
        )

    def encode(self, obs):
        return self.encoder(obs)

    def act(self, obs) -> Normal:
        return self.actor(self.encoder(obs))


@torch.no_grad()
def collect_rollouts(env: gym.Env, agent: Agent, steps=100) -> Rollout:
    obs = env.reset()
    states = []
    actions = []
    rewards = []
    logprobs = []
    dones = []
    value_estimates = []
    infos = []
    next_value_estimates = []
    for _ in range(steps):
        action = agent.act(obs)
        states.append(obs)
        value_estimates.append(agent.critic(obs))
        actions.append(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        # TODO: Implement "next_states" logic correctly here for "done" environments.
        rewards.append(reward)
        dones.append(terminated | truncated)
        next_value_estimates.append(agent.critic(obs))

    return Rollout.from_raw_states(
        states,
        actions,
        rewards,
        logprobs,
        dones,
        value_estimates,
        infos,
        next_value_estimates,
    )


def train_policy():
    agent = Agent()
    optim = torch.optim.Adam(agent.parameters())
    env = gym.make("Ant-v0")

    for batch in range(batches):
        rollouts = collect_rollouts(env, agent)

        optim.zero_grad()
        agent.loss(rollouts).backward()
        optim.step()
