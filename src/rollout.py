import torch
from dataclasses import dataclass

BLANK_TENSOR = torch.tensor(0)

@dataclass
class Rollout:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    logprobs: torch.Tensor
    dones: (
        torch.Tensor
    )  # dones[t] indicates that after taking action[t], a "done" signal was received.
    infos: list[dict]
    returns: torch.Tensor
    value_estimates: torch.Tensor
    advantages: torch.Tensor

    # next_states: torch.Tensor
    next_value_estimates: torch.Tensor

    def flatten(self):
        return Rollout(
            # Remove the "final" state.
            self.states.reshape((-1, self.states.shape[-1])),
            self.actions.reshape((-1, self.actions.shape[-1])),
            self.rewards.reshape((-1,)),
            self.logprobs.reshape((-1,)),
            # note: "dones" and "infos" lose their meaning when flattened.
            BLANK_TENSOR,  # self.dones.reshape((-1,)),
            [],  # self.infos,
            self.returns.reshape((-1,)),
            self.value_estimates.reshape((-1,)),
            self.advantages.reshape((-1,)),
            # self.next_states.reshape((-1, self.next_states.shape[-1])),
            self.next_value_estimates.reshape((-1,)),
        )

    # Performs generalized advantage estimation, calculating returns and advantages.
    @staticmethod
    def from_raw_states(
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        logprobs: torch.Tensor,
        dones: torch.Tensor,
        value_estimates: torch.Tensor,
        infos: list[dict],
        # next_states: torch.Tensor,
        next_value_estimates: torch.Tensor,
        gamma: float,
        lambda_: float,
        normalize_advantages: bool,
        finite_horizon_gae: bool,
    ):
        advantages = torch.zeros(states.shape[:-1], device=device)
        gae = 0
        seqlen = actions.shape[0]

        if finite_horizon_gae:
            raise NotImplementedError("Finite horizon GAE has been removed.")

        for t in reversed(range(seqlen)):
            delta = rewards[t] + gamma * next_value_estimates[t] - value_estimates[t]

            # If action t received a "done" signal, then future rewards should not percolate beyond
            # episode boundaries.
            gae = gamma * lambda_ * gae * (~dones[t]) + delta
            advantages[t] = gae

        returns = advantages + value_estimates
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return Rollout(
            states,
            actions,
            rewards,
            logprobs,
            dones,
            infos,
            returns,
            value_estimates,
            advantages,
            # next_states,
            next_value_estimates,
        )

    def __getitem__(self, slice):
        return Rollout(
            self.states[slice],
            self.actions[slice],
            self.rewards[slice],
            self.logprobs[slice],
            BLANK_TENSOR,  # self.dones[slice],
            [],  # self.infos[slice],
            self.returns[slice],
            self.value_estimates[slice],
            self.advantages[slice],
            # self.next_states[slice],
            self.next_value_estimates[slice],
        )

    def __len__(self):
        return self.actions.shape[0]