import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

try:
    from gymnasium import spaces
except ImportError:  # stable-baselines3 1.x uses gym
    from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def has_nvec(action_space) -> bool:
    return hasattr(action_space, "nvec")


class MiPNFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for padded MiPN bid-history observations."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations.float())


class MiPNPolicy(nn.Module):
    """
    Multi-Issue Policy Network.

    The network keeps MiPN's issue-wise action factorization while supporting
    MiPN_Negotiator through padded observations/actions and per-domain action
    masks. Invalid issue heads or value choices are ignored in log-probability
    and entropy calculations.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule=None,
        net_arch: Optional[Dict[str, Any]] = None,
        activation_fn=nn.ReLU,
        features_dim: int = 64,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        if net_arch is None:
            net_arch = {"pi": [64, 64], "vf": [64, 64]}
        if not has_nvec(action_space):
            raise ValueError(f"MiPNPolicy only supports MultiDiscrete-like action spaces, got {type(action_space)}")

        self.features_dim = features_dim
        self.device = device
        self.max_action_nvec = np.asarray(action_space.nvec, dtype=np.int64)
        self.max_issue_count = len(self.max_action_nvec) - 1

        if lr_schedule is None:
            lr_schedule = lambda progress: 3e-4

        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self._build_mlp_extractor()

    def _build_mlp_extractor(self) -> None:
        self.features_extractor = MiPNFeaturesExtractor(
            self.observation_space,
            features_dim=self.features_dim,
        )

        pi_layers = []
        input_dim = self.features_dim
        for hidden_dim in self.net_arch["pi"]:
            pi_layers.append(nn.Linear(input_dim, hidden_dim))
            pi_layers.append(self.activation_fn())
            input_dim = hidden_dim
        self.policy_net = nn.Sequential(*pi_layers)
        pi_last_dim = input_dim

        vf_layers = []
        input_dim = self.features_dim
        for hidden_dim in self.net_arch["vf"]:
            vf_layers.append(nn.Linear(input_dim, hidden_dim))
            vf_layers.append(self.activation_fn())
            input_dim = hidden_dim
        self.value_net = nn.Sequential(*vf_layers)
        vf_last_dim = input_dim

        self.issue_heads = nn.ModuleList(
            [nn.Linear(pi_last_dim, int(n_actions)) for n_actions in self.max_action_nvec[:-1]]
        )
        self.accept_head = nn.Linear(pi_last_dim, int(self.max_action_nvec[-1]))
        self.value_head = nn.Linear(vf_last_dim, 1)

    def _default_action_nvecs(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(self.max_action_nvec, device=device, dtype=torch.long).repeat(batch_size, 1)

    @staticmethod
    def _masked_logits(logits: torch.Tensor, n_actions: torch.Tensor) -> torch.Tensor:
        max_actions = logits.shape[-1]
        valid_counts = n_actions.clamp(min=1, max=max_actions).long()
        cols = torch.arange(max_actions, device=logits.device).unsqueeze(0)
        invalid = cols >= valid_counts.unsqueeze(1)
        return logits.masked_fill(invalid, torch.finfo(logits.dtype).min)

    def _forward_features(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features_extractor(obs)
        return self.policy_net(features), self.value_net(features)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_nvecs: Optional[torch.Tensor] = None,
        force_reject_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        obs = obs.float()
        batch_size = obs.shape[0]
        if action_nvecs is None:
            action_nvecs = self._default_action_nvecs(batch_size, obs.device)
        else:
            action_nvecs = action_nvecs.to(obs.device).long()

        pi_features, vf_features = self._forward_features(obs)
        actions = []
        log_probs = []

        for issue_idx, head in enumerate(self.issue_heads):
            logits = self._masked_logits(head(pi_features), action_nvecs[:, issue_idx])
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            valid_head = action_nvecs[:, issue_idx] > 0
            action = torch.where(valid_head, action, torch.zeros_like(action))
            actions.append(action)
            log_probs.append(torch.where(valid_head, dist.log_prob(action), torch.zeros_like(action, dtype=obs.dtype)))

        accept_logits = self._masked_logits(self.accept_head(pi_features), action_nvecs[:, -1])
        accept_dist = torch.distributions.Categorical(logits=accept_logits)
        accept_action = accept_dist.probs.argmax(dim=-1) if deterministic else accept_dist.sample()
        if force_reject_mask is not None:
            force_reject_mask = force_reject_mask.to(obs.device).bool()
            accept_action = torch.where(force_reject_mask, torch.ones_like(accept_action), accept_action)
        actions.append(accept_action)
        log_probs.append(accept_dist.log_prob(accept_action))

        values = self.value_head(vf_features)
        return torch.stack(actions, dim=-1), values, torch.stack(log_probs, dim=-1).sum(dim=-1)

    def sample(self, obs, is_first_offer=None, action_nvecs: Optional[torch.Tensor] = None):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        force_reject_mask = None
        if is_first_offer is not None:
            force_reject_mask = torch.as_tensor(is_first_offer, device=obs.device, dtype=torch.bool)
        actions, values, log_probs = self.forward(
            obs,
            deterministic=False,
            action_nvecs=action_nvecs,
            force_reject_mask=force_reject_mask,
        )
        return actions, values, log_probs

    def predict_values(self, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        _, vf_features = self._forward_features(x.float())
        return self.value_head(vf_features)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def predict(self, observation, state=None, deterministic=False, is_first_offer=None, action_nvecs=None):
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                observation = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            actions, _, _ = self.forward(observation, deterministic=deterministic, action_nvecs=action_nvecs)
            if is_first_offer is not None and any(is_first_offer):
                first_mask = torch.as_tensor(is_first_offer, device=actions.device, dtype=torch.bool)
                actions[first_mask, -1] = 1
        actions_np = actions.cpu().numpy()
        return actions_np[0] if actions_np.shape[0] == 1 else actions_np, state

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_nvecs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        obs = obs.float()
        actions = actions.long()
        batch_size = obs.shape[0]
        if action_nvecs is None:
            action_nvecs = self._default_action_nvecs(batch_size, obs.device)
        else:
            action_nvecs = action_nvecs.to(obs.device).long()

        pi_features, vf_features = self._forward_features(obs)
        log_probs = []
        entropies = []

        for issue_idx, head in enumerate(self.issue_heads):
            valid_head = action_nvecs[:, issue_idx] > 0
            logits = self._masked_logits(head(pi_features), action_nvecs[:, issue_idx])
            dist = torch.distributions.Categorical(logits=logits)
            action = actions[:, issue_idx].clamp(min=0, max=logits.shape[-1] - 1)
            log_probs.append(torch.where(valid_head, dist.log_prob(action), torch.zeros_like(action, dtype=obs.dtype)))
            entropies.append(torch.where(valid_head, dist.entropy(), torch.zeros_like(action, dtype=obs.dtype)))

        accept_logits = self._masked_logits(self.accept_head(pi_features), action_nvecs[:, -1])
        accept_dist = torch.distributions.Categorical(logits=accept_logits)
        accept_action = actions[:, -1].clamp(min=0, max=accept_logits.shape[-1] - 1)
        log_probs.append(accept_dist.log_prob(accept_action))
        entropies.append(accept_dist.entropy())

        values = self.value_head(vf_features)
        return values, torch.stack(log_probs, dim=-1).sum(dim=-1), torch.stack(entropies, dim=-1).sum(dim=-1)

    def get_distribution(self, obs: torch.Tensor, action_nvecs: Optional[torch.Tensor] = None):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        pi_features, _ = self._forward_features(obs.float())
        batch_size = obs.shape[0]
        if action_nvecs is None:
            action_nvecs = self._default_action_nvecs(batch_size, obs.device)
        else:
            action_nvecs = action_nvecs.to(obs.device).long()

        distributions = []
        for issue_idx, head in enumerate(self.issue_heads):
            distributions.append(torch.distributions.Categorical(
                logits=self._masked_logits(head(pi_features), action_nvecs[:, issue_idx])
            ))
        distributions.append(torch.distributions.Categorical(
            logits=self._masked_logits(self.accept_head(pi_features), action_nvecs[:, -1])
        ))
        return distributions
