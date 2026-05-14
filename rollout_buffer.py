"""Rollout buffer for PPO training with MiPN_Negotiator masks."""

import numpy as np
import torch
from typing import NamedTuple, Optional

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_nvecs: torch.Tensor


def get_obs_shape(observation_space):
    if isinstance(observation_space, tuple):
        return observation_space
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    if isinstance(observation_space, spaces.Discrete):
        return (1,)
    if isinstance(observation_space, spaces.MultiDiscrete):
        return (int(len(observation_space.nvec)),)
    if isinstance(observation_space, spaces.MultiBinary):
        return observation_space.shape
    if isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}
    raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_action_shape(action_space) -> int:
    if hasattr(action_space, "nvec"):
        return int(len(action_space.nvec))
    if isinstance(action_space, spaces.Discrete):
        return 1
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    if isinstance(action_space, spaces.MultiBinary):
        return int(action_space.n)
    raise NotImplementedError(f"{action_space} action space is not supported")


class RolloutBuffer:
    """Rollout buffer for PPO trajectories."""

    def __init__(
        self,
        buffer_size,
        n_envs,
        obs_space,
        action_space,
        device,
        gamma=0.99,
        gae_lambda=1,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.observation_space = obs_space
        self.obs_dim = get_obs_shape(obs_space)
        self.action_dim = get_action_shape(action_space)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.int64)
        self.action_nvecs = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.generator_ready = False

    def empty_cache(self):
        for name in (
            "observations",
            "actions",
            "action_nvecs",
            "rewards",
            "returns",
            "episode_starts",
            "values",
            "log_probs",
            "advantages",
        ):
            if hasattr(self, name):
                delattr(self, name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def add(self, obs, action, reward, episode_start, value, log_prob, action_nvecs: Optional[np.ndarray] = None):
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_dim))

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        if action_nvecs is None:
            action_nvecs = np.zeros((self.n_envs, self.action_dim), dtype=np.int64)
        self.action_nvecs[self.pos] = action_nvecs
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = np.asarray(episode_start, dtype=np.float32)
        self.values[self.pos] = value.cpu().flatten()
        self.log_probs[self.pos] = log_prob.cpu()
        self.pos += 1

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray):
        last_values = last_values.cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    @staticmethod
    def swap_and_flatten(arr):
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def get(self, batch_size):
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            self.observations = self.swap_and_flatten(self.observations)
            self.actions = self.swap_and_flatten(self.actions)
            self.action_nvecs = self.swap_and_flatten(self.action_nvecs)
            self.values = self.swap_and_flatten(self.values)
            self.log_probs = self.swap_and_flatten(self.log_probs)
            self.advantages = self.swap_and_flatten(self.advantages)
            self.returns = self.swap_and_flatten(self.returns)
            self.generator_ready = True

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def to_torch(self, array):
        return torch.as_tensor(array, device=self.device)

    def _get_samples(self, batch_inds):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.action_nvecs[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
