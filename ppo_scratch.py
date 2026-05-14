from collections import deque
import distutils.version
import os
from typing import List, Tuple, Union, Optional
try:
    import gymnasium as gym
    from gymnasium import register, spaces
except ImportError:
    import gym
    from gym import spaces
    from gym.envs.registration import register
from stable_baselines3.common.vec_env import (
    VecEnv,
    DummyVecEnv,
)
try:
    from stable_baselines3.common.vec_env.patch_gym import _patch_env
except ImportError:
    def _patch_env(env):
        return env
from stable_baselines3.common.monitor import Monitor

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from policy import MiPNPolicy
from rollout_buffer import RolloutBuffer

 # 変更箇所
import random
import itertools

ENV_NAME = 'IssueActionEnv-{}-v0'
DEFAULT_GENERAL_DOMAIN = 'EnergySmall_A'
DEFAULT_OBS_LAYOUT = 'padded_time_last'

def register_neg_env(issue):
    env_name = ENV_NAME.format(issue)
    registry = getattr(gym.envs, "registry", {})
    registered = env_name in registry.env_specs if hasattr(registry, "env_specs") else env_name in registry
    if registered:
        return env_name
    register(
        id=env_name,
        entry_point='envs.env:IssueActionEnv',
        kwargs={'domain': issue, 'is_first': True},
    )
    return env_name

def make_vec_env(
    env_id,
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class = None,
    env_kwargs = None,
    vec_env_cls = None,
    vec_env_kwargs = None,
    monitor_kwargs = None,
    wrapper_kwargs = None,
) -> VecEnv:
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int):
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    if seed is not None:
        vec_env.seed(seed)
    return vec_env

class PPO():
    def __init__(
        self,
        issue: Optional[Union[str, List[str]]] = None,
        agents: Optional[Union[str, List[str]]] = ['Boulware', 'Boulware'],
        n_envs: int = 4,
        learning_rate: float = 3e-4,
        n_rollout_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        obs_space: Optional[Tuple[int, ...]] = None,
        action_space: Optional[Tuple[int, ...]] = None,
        device: torch.device = "cuda:1",
        model: Optional[nn.Module] = None,
        random_train: bool = False,
        general_domain: Optional[str] = DEFAULT_GENERAL_DOMAIN,
        obs_layout: str = DEFAULT_OBS_LAYOUT,
    ) -> None:
        
        self.issue = issue
        self.agents = agents
        self.n_envs = n_envs
        self.general_domain = general_domain
        self.obs_layout = obs_layout
        # 複数環境リスト作成
        self.env_list: list[VecEnv] = self.make_env_list()
        self.env = self.env_list[0]

        self.n_timesteps = 0
        self.learning_rate = learning_rate
        self.n_rollout_steps = n_rollout_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        general_obs_space, general_action_space = self.make_general_spaces()
        self.obs_space = obs_space if obs_space is not None else general_obs_space
        self.action_space = action_space if action_space is not None else general_action_space


        self.device = self.get_device(device)
        # モデル定義
        if model is None:
            # MiPN Policy を使用する場合の学習率スケジュール
            def lr_schedule(progress_remaining: float) -> float:
                return self.learning_rate * progress_remaining
            
            self.model = MiPNPolicy(
                observation_space=self.obs_space,
                action_space=self.action_space,
                lr_schedule=lr_schedule,
                net_arch={"pi": [64, 64], "vf": [64, 64]},
                features_dim=64,
            )
        else:
            self.model = model

        self.model.to(self.device)
        if hasattr(self.model, "device"):
            self.model.device = self.device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5)

        self.episode_frame_numbers = None
        self.episode_rewards = None
        self.vec_env_reward = None
        self.global_step = 0
        self.rollout_buffer_list = [RolloutBuffer(
            buffer_size=self.n_rollout_steps,
            n_envs=self.n_envs,
            obs_space=self.obs_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        ) for e in self.env_list]
        self.ep_info_buffer = deque(maxlen=100)

        self._last_obs = None
        self._last_episode_starts = None
        self._logger = None
        self.save_log = True
        # self.save_log = False
        self.random_train = random_train
    
    def make_env_list(self):
        if isinstance(self.issue, str):
            self.issue = [self.issue]        
        env_list = []
        
        for i in self.issue:
            env_name = register_neg_env(i)
            env = make_vec_env(env_name, n_envs=self.n_envs)
            env_list.append(env)
        return env_list

    @staticmethod
    def _get_nvec(action_space) -> np.ndarray:
        if not hasattr(action_space, "nvec"):
            raise ValueError(f"MiPN_Negotiator requires MultiDiscrete-like action spaces, got {type(action_space)}")
        return np.asarray(action_space.nvec, dtype=np.int64)

    @staticmethod
    def _obs_dim(observation_space) -> int:
        return int(np.prod(observation_space.shape))

    def make_general_spaces(self):
        general_env = None
        space_envs = list(self.env_list)
        try:
            if self.general_domain and self.general_domain not in self.issue:
                env_name = register_neg_env(self.general_domain)
                general_env = make_vec_env(env_name, n_envs=1)
                space_envs.append(general_env)

            max_obs_dim = max(self._obs_dim(env.observation_space) for env in space_envs)
            env_nvecs = [self._get_nvec(env.action_space) for env in space_envs]
            max_issue_count = max(len(nvec) - 1 for nvec in env_nvecs)
            max_issue_nvec = []
            for issue_idx in range(max_issue_count):
                max_issue_nvec.append(max(int(nvec[issue_idx]) for nvec in env_nvecs if issue_idx < len(nvec) - 1))
            accept_n = max(int(nvec[-1]) for nvec in env_nvecs)
            obs_space = spaces.Box(low=0., high=1., shape=(max_obs_dim,), dtype=np.float32)
            action_space = spaces.MultiDiscrete(max_issue_nvec + [accept_n])
            return obs_space, action_space
        finally:
            if general_env is not None:
                general_env.close()

    def _pad_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        obs = obs.reshape(obs.shape[0], -1)
        target_dim = self.obs_space.shape[0]
        if self.obs_layout == 'legacy':
            if obs.shape[1] == target_dim:
                return obs
            if obs.shape[1] > target_dim:
                raise ValueError(f"Observation dim {obs.shape[1]} exceeds MiPN_Negotiator dim {target_dim}")
            padded = np.zeros((obs.shape[0], target_dim), dtype=np.float32)
            padded[:, :obs.shape[1]] = obs
            return padded

        # OnehotObserve2nT appends relative_time as the final raw feature.
        # Keep that time feature at the final padded index for every domain.
        raw_bid_dim = obs.shape[1] - 1
        target_bid_dim = target_dim - 1
        if raw_bid_dim > target_bid_dim:
            raise ValueError(f"Observation dim {obs.shape[1]} exceeds MiPN_Negotiator dim {target_dim}")
        padded = np.zeros((obs.shape[0], target_dim), dtype=np.float32)
        padded[:, :raw_bid_dim] = obs[:, :raw_bid_dim]
        padded[:, -1] = obs[:, -1]
        return padded

    def _general_action_nvec(self, env: VecEnv) -> np.ndarray:
        env_nvec = self._get_nvec(env.action_space)
        general_nvec = np.zeros(len(self.action_space.nvec), dtype=np.int64)
        issue_count = len(env_nvec) - 1
        general_nvec[:issue_count] = env_nvec[:-1]
        general_nvec[-1] = env_nvec[-1]
        return general_nvec

    def _action_nvec_batch(self, env: VecEnv) -> np.ndarray:
        return np.repeat(self._general_action_nvec(env)[None, :], self.n_envs, axis=0)

    def _compact_actions(self, actions: np.ndarray, env: VecEnv) -> np.ndarray:
        env_nvec = self._get_nvec(env.action_space)
        issue_count = len(env_nvec) - 1
        compact = np.zeros((actions.shape[0], len(env_nvec)), dtype=np.int64)
        compact[:, :issue_count] = actions[:, :issue_count]
        compact[:, -1] = actions[:, -1]
        return compact

    @staticmethod
    def _unwrap_env(env):
        while hasattr(env, "env"):
            env = env.env
        return env

    def _set_vec_options(self, env: VecEnv, options: dict):
        if hasattr(env, "set_options"):
            env.set_options(options)
            return
        for sub_env in getattr(env, "envs", []):
            raw_env = self._unwrap_env(sub_env)
            for key, value in options.items():
                setattr(raw_env, key, value)
    
    def on_rollout_start(self):
        self.episode_frame_numbers.clear()
        self.episode_rewards.clear()
    
    def collect_rollouts(
        self, 
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        agent_id: List[int], # 変更箇所 #この部分の型変更にうまく対応できるようにする
    ):
        self.model.eval()
        n_steps = 0
        rollout_buffer.reset()
        self.on_rollout_start()
        is_first_offer = [False] * self.env.num_envs
        action_nvecs_np = self._action_nvec_batch(self.env)
        action_nvecs = torch.as_tensor(action_nvecs_np, device=self.device, dtype=torch.long)

        with tqdm(total=n_rollout_steps) as pbar:
            while n_steps < n_rollout_steps:
                with torch.no_grad():
                    for i in range(self.env.num_envs):
                        raw_env = self._unwrap_env(self.env.envs[i])
                        if getattr(raw_env, "state", None) is None:
                            is_first_offer[i] = True
                    # 行動選択
                    actions, values, log_probs = self.model.sample(self._last_obs, is_first_offer, action_nvecs=action_nvecs)
                    is_first_offer = [False] * self.env.num_envs
                actions = actions.cpu().numpy()
                if isinstance(self.action_space, spaces.Box):
                    actions = np.clip(actions, self.action_space.low, self.action_space.high)
                # 1ステップ進める
                env_actions = self._compact_actions(actions, self.env)
                new_obs, rewards, dones, infos = self.env.step(env_actions)
                new_obs = self._pad_obs(new_obs)

                self.n_timesteps += self.n_envs
                for idx, info in enumerate(infos):
                    maybe_ep_info = info.get("episode")
                    if maybe_ep_info is not None:
                        self.ep_info_buffer.extend([maybe_ep_info])
                n_steps += 1

                # 1エピソード終了
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = torch.as_tensor(
                            self._pad_obs(infos[idx]["terminal_observation"]),
                            device=self.device,
                            dtype=torch.float32,
                        )
                        with torch.no_grad():
                            terminal_value = self.model.predict_values(terminal_obs)  # type: ignore[arg-type]
                        rewards[idx] += self.gamma * terminal_value.item()

                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    action_nvecs_np,
                )
                self._last_obs = new_obs
                self._last_episode_starts = dones
                pbar.update(1)
        
        with torch.no_grad():
            values = self.model.predict_values(new_obs).to(self.device)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # Logger


    def train(
        self,
        total_timesteps:int = 1_000_000,
        save_path = None,
    ):
        # torch.autograd.set_detect_anomaly(True)
        self.episode_frame_numbers = []
        self.episode_rewards = []
        self.vec_env_reward = [0 for _ in range(self.n_envs)]
        # self._last_obs = self.env.reset()
        self._last_episode_starts = torch.ones((self.n_envs,), dtype=bool)
        self._logger = SummaryWriter(log_dir=save_path)
        self.save_path = save_path

        self.n_timesteps = 0
        self.global_step = 0
        self.stop_training = False

        iteration = 0
        n_iteration = total_timesteps // (self.n_envs*self.n_rollout_steps) + 1
        idxes_i = np.array([_ for _ in range(len(self.issue))]*n_iteration)
        np.random.shuffle(idxes_i)
        idxes_a = np.array([_ for _ in range(len(self.agents))]*n_iteration)
        idxes_b = np.array([_ for _ in range(len(self.agents))]*n_iteration) # 変更箇所 
        np.random.shuffle(idxes_a)
        np.random.shuffle(idxes_b) # 変更箇所
        with tqdm(total=total_timesteps) as pbar:
            while self.n_timesteps< total_timesteps:
                # 環境の更新
                if self.stop_training:
                    return self

                # ランダム選択方式
                if self.random_train:
                    agent_id = [idxes_a[iteration], idxes_b[iteration]] # 変更箇所
                    self.env = self.env_list[idxes_i[iteration]]
                    self.rollout_buffer = self.rollout_buffer_list[idxes_i[iteration]]
                    self._set_vec_options(self.env, {"opponent": [self.agents[agent_id[0]], self.agents[agent_id[1]]]}) # 変更箇所
                    self._last_obs = self._pad_obs(self.env.reset())
                    
                # 総当たり方式
                else:
                    pairs = list(itertools.combinations_with_replacement(range(len(self.agents)), 2)) # 変更箇所
                    agent_id = pairs[iteration%len(pairs)]  # 変更箇所
                    self.env = self.env_list[iteration%len(self.issue)]
                    self.rollout_buffer = self.rollout_buffer_list[iteration%len(self.issue)]
                    self._set_vec_options(self.env, {"opponent": [self.agents[agent_id[0]], self.agents[agent_id[1]]]}) # 変更箇所
                    
                    self._last_obs = self._pad_obs(self.env.reset())

                # ロールアウト（シミュレーション）実行
                self.collect_rollouts(self.rollout_buffer, n_rollout_steps=self.n_rollout_steps, agent_id=agent_id)

                pbar.update(self.n_envs*self.n_rollout_steps)
                iteration += 1

                # 収集したデータから勾配更新
                self.train_loop()
                self.rollout_buffer.empty_cache()

                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'obs_space_shape': self.obs_space.shape,
                    'action_nvec': np.asarray(self.action_space.nvec, dtype=np.int64),
                    'issues': self.issue,
                    'agents': self.agents,
                    'general_domain': self.general_domain,
                    'obs_layout': self.obs_layout,
                }
                torch.save(checkpoint, save_path+'/checkpoint.pt'.format())
        return self
    
    def train_loop(self):
        self.model.train()
        for epoch in tqdm(range(self.n_epochs),total=self.n_epochs):
            rollouts = self.rollout_buffer.get(self.batch_size)
            for rollout_data in rollouts:
                actions = rollout_data.actions
                values, log_prob, entropy = self.model.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_nvecs=rollout_data.action_nvecs,
                )
                values = values.flatten()

                # Normalize advantages
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # ratio between old and new policy, should be one at the 1st iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # Clipped Surrogate Objective
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values)

                # Entropy loss favor exploration
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Logs
                if self.save_log:
                    self._logger.add_scalar("train/policy_loss", policy_loss.item(), self.global_step)
                    self._logger.add_scalar("train/value_loss", value_loss.item(), self.global_step)
                    self._logger.add_scalar("train/entropy_loss", entropy_loss.item(), self.global_step)
                    self._logger.add_scalar("train/loss", loss.item(), self.global_step)
                    self._logger.add_scalar("train/values", torch.mean(values).item(), self.global_step)
                    self._logger.add_scalar("train/log_prob", torch.mean(log_prob).item(), self.global_step)
                    if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                        self._logger.add_scalar("rollout/ep_rew_mean", float(np.mean([ep_info["r"] for ep_info in self.ep_info_buffer])), self.global_step)
                        self._logger.add_scalar("rollout/ep_len_mean", float(np.mean([ep_info["l"] for ep_info in self.ep_info_buffer])), self.global_step)

                self.global_step += 1
    
    
    @staticmethod
    def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
        """
        Retrieve PyTorch device.
        It checks that the requested device is available first.
        For now, it supports only cpu and cuda.
        By default, it tries to use the gpu.

        :param device: One for 'auto', 'cuda', 'cpu'
        :return: Supported Pytorch device
        """
        # Cuda by default
        if device == "auto":
            device = "cuda"
        # Force conversion to torch.device
        device = torch.device(device)

        # Cuda not available
        if device.type == torch.device("cuda").type and not torch.cuda.is_available():
            return torch.device("cpu")

        return device
    
    def predict(self, observation, state, episode_start=None, deterministic=False):
        if np.all(observation == np.zeros(observation.shape)):
            is_first_offer = [True]
        else:
            is_first_offer = [False]
        observation = self._pad_obs(observation)
        action_nvecs = torch.as_tensor(self._action_nvec_batch(self.env)[:1], device=self.device, dtype=torch.long)
        actions, state = self.model.predict(
            observation,
            state,
            deterministic,
            is_first_offer=is_first_offer,
            action_nvecs=action_nvecs,
        )
        return self._compact_actions(np.asarray(actions).reshape(1, -1), self.env)[0], state
