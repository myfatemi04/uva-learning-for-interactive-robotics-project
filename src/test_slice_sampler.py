import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

steps_per_update = 2
num_envs = 4
batch_size = 8
buffer_size = 1000
episode_length = 17
sampler = SliceSampler(
    num_slices=batch_size // 4,
    end_key=None,
    traj_key="episode",
    truncated_key=None,
    strict_length=True,
)
storage = LazyTensorStorage(max_size=buffer_size)
buffer = ReplayBuffer(
    storage=storage,
    sampler=sampler,
    pin_memory=True,
    prefetch=num_envs // steps_per_update,
    batch_size=batch_size,
)

### Add some dummy content to the buffer.
sample_tensordict = TensorDict(
    {
        "obs": torch.randn(episode_length * num_envs, 224, 224, 3),
        "action": torch.randn(episode_length * num_envs, 1),
        "reward": torch.randn(episode_length * num_envs, 1),
        "episode": torch.zeros(episode_length * num_envs),
        # "done": torch.zeros(episode_length, num_envs, 1),
    },
    batch_size=episode_length * num_envs,
)

sample_tensordict["episode"][:] = (
    torch.arange(0, num_envs).unsqueeze(-1).repeat(1, episode_length).reshape(-1)
)

print(sample_tensordict)
print(sample_tensordict["episode"])

buffer.extend(sample_tensordict)

sample = buffer.sample()

print(sample)
print(sample["episode"].reshape(2, 4))

# Conclusion: You must add episodes to the buffer contiguously.
# You must specify the "episode" parameter.
# When you sample, you will get segments in sequence. You can reshape them into
# (batch_size, segment_length)
