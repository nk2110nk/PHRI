import random
import bisect
import os
import numpy as np
import torch

from negmas.sao import SAONegotiator
from typing import Optional
from negmas.common import *
from negmas.outcomes import Outcome, ResponseType
from sao.opponent_model import *
from envs.observer import *
from stable_baselines3 import PPO
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces
from policy import MiPNPolicy

PADDED_TIME_LAST_OBS_LAYOUT = 'padded_time_last'


class RLNegotiator(SAONegotiator):
    def __init__(self, name='RLAgent', **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_outcomes = None
        self.next_bid = None
        self.last_bid = None

    def on_ufun_changed(self):
        super().on_ufun_changed()
        self.next_bid = None
        self.last_bid = None
        self.n_outcomes = len(self._ami.discrete_outcomes())

    @property
    def all_bids(self):
        return self._ami.discrete_outcomes()

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        self.last_bid = self.next_bid
        return self.next_bid

    def set_next_bid(self, next_bid) -> None:
        self.next_bid = next_bid


class TestRLNegotiator(RLNegotiator):
    def __init__(self, domain, path, deterministic=True, mode='issue', opponents=None, **kwargs):
        super().__init__(**kwargs)
        self.is_scratch_model = self._is_scratch_checkpoint(path)
        self.mode = mode    # issue, venas
        self.deterministic = deterministic
        self.opponents = opponents or []
        self.observer = OnehotObserve2nT(domain, 6)
        self.domain = domain
        self.actions = None
        self.states = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.is_scratch_model:
            self.model, self.obs_dim, self.action_nvec, self.obs_layout = self._load_scratch_model(path)
        else:
            self.model = PPO.load(path)
            self.obs_dim = None
            self.action_nvec = None
            self.obs_layout = None

    @staticmethod
    def _is_scratch_checkpoint(path):
        if os.path.isdir(path):
            path = os.path.join(path, "checkpoint.pt")
        return str(path).endswith(".pt")

    def _load_scratch_model(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, "checkpoint.pt")
        checkpoint = torch.load(path, map_location=self.device)
        obs_shape = tuple(checkpoint["obs_space_shape"])
        action_nvec = np.asarray(checkpoint["action_nvec"], dtype=np.int64)
        obs_layout = checkpoint.get("obs_layout", "legacy")
        model = MiPNPolicy(
            observation_space=spaces.Box(low=0., high=1., shape=obs_shape, dtype=np.float32),
            action_space=spaces.MultiDiscrete(action_nvec),
            net_arch={"pi": [64, 64], "vf": [64, 64]},
            features_dim=64,
        )
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.device = self.device
        model.eval()
        return model, obs_shape[0], action_nvec, obs_layout

    def _observe(self, state):
        observation = self.observer(None, self.opponents) if state is None else self.observer(state.__dict__, self.opponents)
        if not self.is_scratch_model:
            return observation
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation.shape[0] > self.obs_dim:
            raise ValueError(f"Observation dim {observation.shape[0]} exceeds checkpoint dim {self.obs_dim}")
        padded = np.zeros((self.obs_dim,), dtype=np.float32)
        if self.obs_layout == PADDED_TIME_LAST_OBS_LAYOUT:
            raw_bid_dim = observation.shape[0] - 1
            target_bid_dim = self.obs_dim - 1
            if raw_bid_dim > target_bid_dim:
                raise ValueError(f"Observation dim {observation.shape[0]} exceeds checkpoint dim {self.obs_dim}")
            padded[:raw_bid_dim] = observation[:raw_bid_dim]
            padded[-1] = observation[-1]
        else:
            padded[:observation.shape[0]] = observation
        return padded

    def _domain_action_nvec(self):
        domain_nvec = np.asarray([len(issue.values) for issue in self.domain] + [2], dtype=np.int64)
        if not self.is_scratch_model:
            return domain_nvec
        if len(domain_nvec) > len(self.action_nvec):
            raise ValueError(
                f"Checkpoint has {len(self.action_nvec) - 1} issue heads but domain requires {len(domain_nvec) - 1}"
            )
        general_nvec = np.zeros_like(self.action_nvec)
        general_nvec[:len(domain_nvec) - 1] = domain_nvec[:-1]
        general_nvec[-1] = domain_nvec[-1]
        return general_nvec

    def _predict_action(self, observation, is_first_offer=False):
        if not self.is_scratch_model:
            return self.model.predict(observation, state=self.states, deterministic=self.deterministic)
        action_nvecs = torch.as_tensor(self._domain_action_nvec()[None, :], device=self.device, dtype=torch.long)
        actions, self.states = self.model.predict(
            observation[None, :],
            state=self.states,
            deterministic=self.deterministic,
            is_first_offer=[is_first_offer],
            action_nvecs=action_nvecs,
        )
        actions = np.asarray(actions).reshape(-1)
        issue_count = len(self.domain)
        compact = np.zeros(issue_count + 1, dtype=np.int64)
        compact[:issue_count] = actions[:issue_count]
        compact[-1] = actions[-1]
        return compact, self.states

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        # 初手AC用
        # if self.mode == 'issue':
        #     if self.actions[-1] == 0 and len(self.actions) != len(self.domain):
        #         return ResponseType.END_NEGOTIATION
        # elif self.mode == 'venas':
        #     if self.actions == self.n_outcomes:
        #         return ResponseType.END_NEGOTIATION

        observation = self._observe(state)
        self.actions, self.states = self._predict_action(observation)
        if self.mode == 'issue':
            if self.actions[-1] == 0 and len(self.actions) != len(self.domain):
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER
        elif self.mode == 'venas':
            if self.actions == self.n_outcomes:
                return ResponseType.ACCEPT_OFFER
            else:
                return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.actions is None:
            observation = self._observe(None)
            self.actions, self.states = self._predict_action(observation, is_first_offer=True)
        if self.mode == 'issue':
            if self.actions[-1] == 0 and len(self.actions) != len(self.domain):
                return None
            return {i.name: i.values[v] for i, v in zip(self.domain, self.actions)}
        elif self.mode == 'venas':
            if self.actions == self.n_outcomes:
                return None
            return self.all_bids[self.actions]


class RandomNegotiator(RLNegotiator):
    def __init__(self, name='Random', **kwargs):
        super().__init__(name=name, **kwargs)
        self.action = -1

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        self.action = random.randrange(self.n_outcomes + 1)
        if self.action == self.n_outcomes:
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def reset(self):
        super().reset()

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.action == -1:
            self.action = random.randrange(self.n_outcomes + 1)
        if self.action == self.n_outcomes:
            # TODO: 初ターンAcceptへの対処
            return None
        return self.all_bids[self.action]


class RLBOANegotiator(RLNegotiator):
    def __init__(self, n_ranges=10, **kwargs):
        super().__init__(**kwargs)
        self.ordered_outcomes = None
        self.ordered_utils = None
        self.n_outcomes = None
        self.target = n_ranges - 1
        self.update_threshold = 1.1
        self.n_ranges = n_ranges
        self.range_index = None
        self.om = None   # if self._utility_function is None else NoModel()

    def on_ufun_changed(self):
        super().on_ufun_changed()
        self.target = self.n_ranges - 1
        self.om = HardHeadedFrequencyModel(self._utility_function)
        # self.om = NoModel()
        outcomes = self._ami.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._utility_function(outcome), outcome) for outcome in outcomes],
            key=lambda x: float(x[0]) if x[0] is not None else float("-inf"),
            # reverse=True,
        )
        self.ordered_utils = np.array([u for (u, _) in self.ordered_outcomes])  # [::-1]])
        self.n_outcomes = len(self.ordered_utils)
        # 範囲のインデックスを取得
        step = np.linspace(self.ordered_utils[0], self.ordered_utils[-1], self.n_ranges+1)
        self.range_index = [0]
        for th in step[1:-1]:
            idx = bisect.bisect(self.ordered_utils, th)
            self.range_index.append(idx)
        self.range_index.append(len(self.ordered_utils))

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        # return ResponseType.REJECT_OFFER
        if state['relative_time'] < self.update_threshold:
            self.om.update(offer, state['relative_time'])
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        offered_util = self._utility_function(offer)
        if offered_util is None:
            return ResponseType.REJECT_OFFER
        my_util = self._utility_function(self.propose(state))
        if offered_util >= my_util and (self.reserved_value is None or offered_util > self.reserved_value):
            return ResponseType.ACCEPT_OFFER
        if self.reserved_value is not None and my_util < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if type(self.om) == NoModel:
            return random.choice(self.get_bid_range())[1]
        else:
            return self.get_best_bid(self.get_bid_range())

    def get_best_bid(self, bids):
        best_util = -1
        _, best_bid = bids[0]
        all_zero = True
        for (u, o) in bids:
            evaluation = self.om(o)
            if evaluation > 0.0001:
                all_zero = False
            if evaluation > best_util:
                best_bid = o
                best_util = evaluation
        if all_zero:
            _, best_bid = random.choice(bids)
        return best_bid

    def get_bid_range(self):
        return self.ordered_outcomes[self.range_index[self.target]:self.range_index[self.target+1]]

    def set_target(self, target) -> None:
        self.target = target
