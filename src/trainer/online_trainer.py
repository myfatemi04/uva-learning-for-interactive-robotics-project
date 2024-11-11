from time import time
from typing import Union

import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()
        self._tds: list[TensorDict] = []

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards = []
        for i in range(self.cfg.eval_episodes // self.cfg.num_envs):
            obs = self.env.reset()
            terminated = torch.zeros(self.cfg.num_envs, dtype=torch.bool)
            truncated = torch.zeros(self.cfg.num_envs, dtype=torch.bool)
            ep_reward = 0
            t = 0

            if self.cfg.save_video:
                assert self.logger.video
                self.logger.video.init(self.env, enabled=(i == 0))

            while not (terminated | truncated).any():
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    assert self.logger.video
                    self.logger.video.record(self.env)

            assert (
                terminated | truncated
            ).all(), "Vectorized environments must reset all environments at once."

            ep_rewards.append(ep_reward)
            if self.cfg.save_video:
                assert self.logger.video
                self.logger.video.save(self._step)

        return dict(
            episode_reward=torch.cat(ep_rewards).mean(),
            episode_success=info["success"].mean(),
        )

    def to_td(self, obs: Union[TensorDict, torch.Tensor], action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            assert self.env.action_space.shape is not None
            action = torch.full(self.env.action_space.shape, float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan")).repeat(self.cfg.num_envs)
        td = TensorDict(
            {
                "obs": obs,  # type: ignore
                "action": action.unsqueeze(0),
                "reward": reward.unsqueeze(0),
            },
            batch_size=(1, self.cfg.num_envs),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        terminated = torch.tensor(True)
        truncated = torch.tensor(True)
        eval_next = True
        last_eval_step = 0

        env_step_duration = 0
        env_steps = 0
        agent_update_duration = 0
        agent_update_steps = 0
        act_duration = 0
        act_steps = 0
        has_content = False

        save_every = self.cfg.num_envs * 4000

        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            _done = terminated | truncated
            if _done.any():
                assert (
                    _done.all()
                ), "Vectorized environments must reset all environments at once."
                if eval_next:
                    self.logger.log({**self.eval(), **self.common_metrics()}, "eval")
                    eval_next = False

                if self._step > 0:
                    tds: TensorDict = torch.cat(self._tds)  # type: ignore
                    self.logger.log(
                        {
                            "episode_reward": tds["reward"].nansum(0).mean(),
                            "episode_success": info["success"].nanmean(),
                            **self.common_metrics(),
                        },
                        "eval",
                    )
                    self._ep_idx = self.buffer.add(tds)
                    has_content = True

                obs, info = self.env.reset()
                self._tds = [self.to_td(obs)]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                act_start = time()
                action = self.agent.act(obs, t0=len(self._tds) == 1)
                act_end = time()
                act_duration += act_end - act_start
                act_steps += self.cfg.num_envs
            else:
                action = torch.tensor(self.env.action_space.sample())

            step_start = time()
            obs, reward, terminated, truncated, info = self.env.step(action)
            step_end = time()
            env_steps += self.cfg.num_envs
            env_step_duration += step_end - step_start

            self._tds.append(self.to_td(obs, action, reward))

            # Update agent
            if self._step >= self.cfg.seed_steps and has_content:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps // self.cfg.steps_per_update
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = max(1, self.cfg.num_envs // self.cfg.steps_per_update)

                num_updates = 1
                update_start = time()
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                update_end = time()

                if self._step > self.cfg.seed_steps:
                    agent_update_duration += update_end - update_start
                    agent_update_steps += num_updates

                if self._step % 10 == 0:
                    self.logger.log(
                        {**_train_metrics, **self.common_metrics()}, "train"
                    )
                    print(
                        f"env.step rate: {env_step_duration/env_steps:.6f}s, agent.update rate: {agent_update_duration/(agent_update_steps+1e-8):.6f}s, agent.act rate: {act_duration/(act_steps+1e-8):.6f}"
                    )

            self._step += self.cfg.num_envs

            if self._step > 0 and self._step % save_every == 0:
                self.agent.save(self.logger.model_dir / f"step_{self._step}.pt")

        self.logger.finish(self.agent)
