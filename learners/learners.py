from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
import gymnasium.spaces as spaces

from algorithms.MAPPO import MAPPO
from envs import Environment
from models.policies import BasePolicy
from algorithms.RLAlgorithm import RLAlgorithm
from utils.buffers import RolloutBuffer
from utils.loggers import MetricsStore

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from utils.schedulers import LearningRateScheduler, LinearDecaySchedule, ExponentialDecaySchedule, EntropyScheduler


def _create_schedule_from_config(config):
    """Crea schedule da configurazione."""
    scheduler_type = config.get('type', 'linear')
    initial_value = config.get('initial_value', 3e-4)
    final_value = config.get('final_value', 1e-6)
    total_steps = config.get('total_steps', 1000)

    if scheduler_type == 'linear':
        return LinearDecaySchedule(initial_value, final_value, total_steps)
    elif scheduler_type == 'exp':
        decay_rate = config.get('decay_rate', None)
        return ExponentialDecaySchedule(initial_value, final_value, total_steps, decay_rate)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class BasicLearner:

    def __init__(self,
                 env: Environment,
                 algorithm: RLAlgorithm,
                 policies: dict[str, BasePolicy],
                 agent_policy_mapping: dict,
                 action_dim: int,
                 obs_dim: int,
                 rollout_len:int,
                 data_batch_size: int,
                 X_train: Optional[np.ndarray]=None,
                 y_train: Optional[np.ndarray]=None,
                 val_loader: Optional[torch.utils.data.DataLoader]=None,
                 test_loader: Optional[torch.utils.data.DataLoader]=None,
                 gamma: float=0.99,
                 gae_lambda: float=0.95,
                 critic_obs_dim: Optional[int]=None,
                 lr_scheduler_config: Optional[Union[dict[str, Any], dict[str, dict[str, Any]]]]=None,
                 verbose: bool=False,
                 device: str='cpu',
                 tensorboard_log_dir: Optional[Union[str, Path]]=None,
                 log_interval: int=10,
                 ) -> None:

        self.algorithm = algorithm
        self.policies = policies
        self.agent_policy_mapping = agent_policy_mapping
        self.env = env
        self.vectorized_env = hasattr(env, "n_envs")
        self.n_envs = int(getattr(env, "n_envs", 1))
        self.n_agents = int(getattr(env, "n_agents", len(agent_policy_mapping)))
        self.rollouts = rollout_len
        self.action_dim = action_dim
        self.batch_size = data_batch_size
        self.rollout_batch_size = self.n_envs
        self.X_train = X_train
        self.y_train = y_train
        self.device = device
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.verbose = verbose
        self.log_interval = max(1, int(log_interval))
        self.global_step = 0
        if tensorboard_log_dir is not None and SummaryWriter is None:
            raise ImportError("TensorBoard logging requires the tensorboard package")
        self.writer = SummaryWriter(str(tensorboard_log_dir)) if tensorboard_log_dir is not None else None

        assert rollout_len >= self.n_envs, "rollout_len must be at least the number of parallel environments"

        self.policy_agent_mapping = {pid: [] for pid in policies.keys()}

        for agent_id, policy_id in agent_policy_mapping.items():
            self.policy_agent_mapping[policy_id].append(agent_id)

        self.history_buffers = [
            RolloutBuffer(rollout_len, self.n_envs, action_dim, obs_dim, gae_lambda=gae_lambda, gamma=gamma,
                          shared_space=critic_obs_dim)
            for _ in range(self.n_agents)
        ]

        self.all_schedulers = []

        if lr_scheduler_config is not None:
            if lr_scheduler_config.keys() != self.policies.keys():
                assert set(lr_scheduler_config.keys()).isdisjoint(self.policies.keys()), ("You must specify a global "
                                                                                          "configuration for lr scheduler, "
                                                                                          "or a configuration for each of the policy.")
                config = {pid: lr_scheduler_config for pid in self.policies.keys()}
            else:
                config = lr_scheduler_config
        else:
            config = lr_scheduler_config

        self.lr_schedulers = self._create_lr_schedulers(config)
        self.lr_history = {policy_id: {} for policy_id in self.policies.keys()}

        self.metrics = MetricsStore()
        self.metrics.set_metadata("gamma", gamma)
        self.metrics.set_metadata("gae_lambda", gae_lambda)
        self.metrics.set_metadata("n_envs", self.n_envs)
        self.metrics.set_metadata("rollout_len", rollout_len)
        self.metrics.set_metadata("tensorboard_log_dir", str(tensorboard_log_dir) if tensorboard_log_dir else None)

        self.best_policies = {}

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _to_scalar(self, value):
        value = np.asarray(value)
        if value.size == 0:
            return None
        return float(value.mean())

    def _tb_scalar(self, tag: str, value, step: int):
        if self.writer is None:
            return
        scalar = self._to_scalar(value)
        if scalar is not None and np.isfinite(scalar):
            self.writer.add_scalar(tag, scalar, step)

    def _tb_text(self, tag: str, value: str, step: int=0):
        if self.writer is not None:
            self.writer.add_text(tag, value, step)

    def _log_training_stats(self, episode: int, policy_stats: dict[str, dict[str, float]]):
        for policy_id, stats in policy_stats.items():
            for key, value in stats.items():
                self._tb_scalar(f"train/{policy_id}/{key}", value, episode)

        for policy_id, schedulers in self.lr_schedulers.items():
            for scheduler_type, scheduler in schedulers.items():
                self._tb_scalar(f"train/{policy_id}/lr_{scheduler_type}", scheduler.get_lr(), episode)

        if hasattr(self, "algorithm") and hasattr(self.algorithm, "c2"):
            self._tb_scalar("train/entropy_coeff", self.algorithm.c2, episode)

        self._tb_scalar("train/global_step", self.global_step, episode)

        if episode % self.log_interval == 0:
            summary = []
            for policy_id, stats in policy_stats.items():
                avg_return = stats.get("avg_return")
                loss = stats.get("loss")
                critic_loss = stats.get("critic_loss")
                if avg_return is not None:
                    summary.append(f"{policy_id}: return={avg_return:.3f}")
                if loss is not None:
                    summary.append(f"loss={loss:.3f}")
                if critic_loss is not None:
                    summary.append(f"critic={critic_loss:.3f}")

            val_acc = self.metrics.last("val_acc")
            val_part = f", val_acc={val_acc:.3f}" if val_acc is not None else ""
            print(f"Episode {episode}: " + "; ".join(summary) + val_part)

    def _as_env_batch(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        if value.ndim == 2:
            return value[:, np.newaxis, :]
        return value

    def _as_agent_env_done(self, dones) -> np.ndarray:
        dones = np.asarray(dones, dtype=bool)

        if dones.ndim == 0:
            return np.full((self.n_agents, self.n_envs), dones, dtype=bool)

        if dones.ndim == 1:
            if self.vectorized_env and dones.shape[0] == self.n_envs:
                return np.repeat(dones[np.newaxis, :], self.n_agents, axis=0)
            if dones.shape[0] == self.n_agents:
                return dones[:, np.newaxis]
            if dones.shape[0] == self.n_envs:
                return np.repeat(dones[np.newaxis, :], self.n_agents, axis=0)

        return dones

    def _shared_critic_obs(self, infos, agent_ids):
        shared_obs = np.asarray(infos["shared_obs"])

        if shared_obs.ndim == 2:
            if shared_obs.shape[0] == self.n_agents:
                shared_obs = shared_obs[:, np.newaxis, :]
            elif shared_obs.shape[0] == self.n_envs:
                shared_obs = np.repeat(shared_obs[np.newaxis, :, :], self.n_agents, axis=0)

        return torch.from_numpy(shared_obs[agent_ids]).float().to(self.device)

    def _policy_logprob(self, policy: BasePolicy, log_prob: torch.Tensor) -> torch.Tensor:
        if isinstance(policy.action_space, spaces.Box) and log_prob.dim() > 0 and log_prob.shape[-1] != 1:
            return log_prob.sum(dim=-1, keepdim=True)
        if log_prob.dim() == 0:
            return log_prob.reshape(1, 1)
        if log_prob.dim() >= 3 and log_prob.shape[-1] == 1:
            return log_prob
        return log_prob.unsqueeze(-1)

    def _step_actions(self, actions: np.ndarray) -> np.ndarray:
        if self.vectorized_env:
            return actions
        return actions[:, 0, :]

    def _env_correct_count(self, infos, valid_envs: int) -> int:
        correct = np.asarray(infos["__all__"]["correct"], dtype=bool)
        if correct.ndim == 0:
            return int(correct.item())
        return int(correct.reshape(-1)[:valid_envs].sum())

    def _set_env_mode(self, training: bool):
        if hasattr(self.env, "_workers"):
            return
        if hasattr(self.env, "encoder") and hasattr(self.env, "decoder"):
            # The environment models are frozen preprocessing/classification transforms.
            # Keep BatchNorm/Dropout deterministic even when the RL policies are training.
            self.env.encoder.eval()
            self.env.decoder.eval()

    def _freeze_env_models(self):
        if hasattr(self.env, "_workers"):
            return
        if not (hasattr(self.env, "encoder") and hasattr(self.env, "decoder")):
            return

        for param in self.env.encoder.parameters():
            param.requires_grad = False
        for param in self.env.decoder.parameters():
            param.requires_grad = False


    def _create_lr_schedulers(self, config):
        """
        Crea scheduler per learning rate per ogni policy.
        """
        if config is None:
            return {}

        lr_schedulers = {}

        for policy_id, policy_net in self.policies.items():
            policy_schedulers = {}
            pol_config = config.get(policy_id, {})

            # Controlla se c'è configurazione separata per actor/critic
            if 'actor' in pol_config or 'critic' in pol_config:
                # Configurazione separata
                if 'actor' in pol_config:
                    actor_config = pol_config['actor']
                    actor_schedule = _create_schedule_from_config(actor_config)
                    policy_schedulers['actor'] = LearningRateScheduler(
                        policy_net.actor_optimizer, actor_schedule
                    )

                if 'critic' in pol_config:
                    critic_config = pol_config['critic']
                    critic_schedule = _create_schedule_from_config(critic_config)
                    policy_schedulers['critic'] = LearningRateScheduler(
                        policy_net.critic_optimizer, critic_schedule
                    )
            else:
                # Configurazione unificata (applica a tutti gli ottimizzatori)
                schedule = _create_schedule_from_config(pol_config)

                if hasattr(policy_net, 'actor_optimizer'):
                    policy_schedulers['actor'] = LearningRateScheduler(
                        policy_net.actor_optimizer, schedule
                    )

                if hasattr(policy_net, 'critic_optimizer'):
                    policy_schedulers['critic'] = LearningRateScheduler(
                        policy_net.critic_optimizer, schedule
                    )

                # Fallback per ottimizzatore unificato
                if hasattr(policy_net, 'optimizer') and not policy_schedulers:
                    policy_schedulers['unified'] = LearningRateScheduler(
                        policy_net.optimizer, schedule
                    )

            lr_schedulers[policy_id] = policy_schedulers

        return lr_schedulers

    def _step_lr_schedulers(self):
        """
        Esegue step per learning rate schedulers.
        """
        for policy_id, policy_schedulers in self.lr_schedulers.items():
            for scheduler_type, scheduler in policy_schedulers.items():
                new_value = scheduler.step()
                if scheduler_type not in self.lr_history[policy_id].keys():
                    self.lr_history[policy_id][scheduler_type] = []
                self.lr_history[policy_id][scheduler_type].append(new_value)

    def _step_schedulers(self):
        self._step_lr_schedulers()

    def get_current_learning_rates(self):
        """Restituisce i learning rate correnti per tutte le policy."""
        current_lrs = {}
        for policy_id, policy_schedulers in self.lr_schedulers.items():
            current_lrs[policy_id] = {}
            for scheduler_type, scheduler in policy_schedulers.items():
                current_lrs[policy_id][scheduler_type] = scheduler.get_lr()
        return current_lrs

    def train(self, epochs: int=5000):

        best_val_acc = 0

        self._set_env_mode(training=True)
        self._freeze_env_models()

        for policy in self.policies.values():
            policy.set_training_mode(True)
        self.env.set_deterministic(False)

        self._tb_text("run/metadata", str(self.metrics.metadata))

        for episode in range(epochs):

            self.__collect_trajectories()
            episode_idx = episode + 1
            policy_stats = {}

            for pid, policy in self.policies.items():
                agent_ids = self.policy_agent_mapping[pid]
                buffers = [b for i, b in enumerate(self.history_buffers) if i in agent_ids]
                with torch.no_grad():
                    next_value = policy.get_state_value(self._next_critic_obs[agent_ids]).detach().cpu().numpy()
                avg_returns, metrics = self.algorithm.update(pid, policy, buffers, next_value)
                for aid in agent_ids:
                    self.history_buffers[aid].clear()

                if metrics is not None:
                    policy_stats[pid] = {key: self._to_scalar(value) for key, value in metrics.items()}
                    policy_stats[pid]["avg_return"] = self._to_scalar(avg_returns)
                    agent_id_strs = [f"agent_{aid}" for aid in agent_ids]
                    self.metrics.log_for_agents(agent_id_strs, metrics)
                    self.metrics.log_for_agents(
                        agent_id_strs,
                        {"avg_return": avg_returns.mean().item()}
                    )

            self._step_schedulers()
            self.global_step += self.rollouts

            if (episode + 1) % 20 == 0:
                if self.val_loader is not None:
                    self.evaluate()
                    current_val_acc = self.metrics.last("val_acc")

                    if current_val_acc >= best_val_acc:
                        best_val_acc = current_val_acc
                        for key, value in self.policies.items():
                            self.best_policies[key] = {k: v.clone() for k, v in value.state_dict().items()}

            self._log_training_stats(episode_idx, policy_stats)

        self.evaluate(test=True)
        self.close()


    def __collect_trajectories(self):

        count = 0

        while count < self.rollouts:

            obs, infos = self.env.reset()
            obs = self._as_env_batch(obs)
            dones = np.zeros((self.n_agents, self.n_envs), dtype=bool)

            while not np.all(dones) and count < self.rollouts:

                actions_array = np.zeros((self.n_agents, self.n_envs, self.action_dim), dtype=np.float32)

                for policy_id, policy in self.policies.items():
                    agent_ids = self.policy_agent_mapping[policy_id]
                    obs_tensor = torch.from_numpy(obs[agent_ids]).float().to(self.device)
                    with torch.no_grad():
                        action_tensor, log_prob = policy(obs_tensor)
                        log_prob = self._policy_logprob(policy, log_prob)

                        if 'shared_obs' in infos:
                            critic_input = self._shared_critic_obs(infos, agent_ids)
                        else:
                            critic_input = obs_tensor

                        value = policy.get_state_value(critic_input)

                    for i, agent_id in enumerate(agent_ids):
                        self.history_buffers[agent_id].add(
                            action=action_tensor[i].cpu().numpy(),
                            state=obs_tensor[i].cpu().numpy(),
                            logprob=log_prob[i].cpu().numpy(),
                            crit_state=critic_input[i].cpu().numpy(),
                            state_value=value[i].cpu().numpy(),
                            perform_step=False,
                        )
                    actions_array[agent_ids, :] = action_tensor.cpu().numpy()

                next_obs, rewards, dones, infos = self.env.step(self._step_actions(actions_array))
                next_obs = self._as_env_batch(next_obs)

                dones = self._as_agent_env_done(dones['__all__'])

                for aid in range(self.n_agents):
                    self.history_buffers[aid].add(
                        rewards=rewards['agent_'+str(aid)],
                        is_terminals=dones[aid]
                    )

                with torch.no_grad():
                    if 'shared_obs' in infos:
                        next_critic_obs = np.asarray(infos['shared_obs'])
                        if next_critic_obs.ndim == 2:
                            if next_critic_obs.shape[0] == self.n_agents:
                                next_critic_obs = next_critic_obs[:, np.newaxis, :]
                            elif next_critic_obs.shape[0] == self.n_envs:
                                next_critic_obs = np.repeat(next_critic_obs[np.newaxis, :, :], self.n_agents, axis=0)
                        self._next_critic_obs = torch.from_numpy(next_critic_obs).float().to(self.device)
                    else:
                        self._next_critic_obs = torch.from_numpy(next_obs).float().to(self.device)

                obs = next_obs

                count += self.n_envs


    def evaluate(self, test=False):

        if not test:
            loader = self.val_loader
        else:
            loader = self.test_loader

        correct = 0
        total = 0
        total_steps = 0
        max_steps = self.env.n_iters

        self.env.set_deterministic()
        self._set_env_mode(training=False)

        for policy in self.policies.values():
            policy.set_training_mode(False)

        all_actions = np.zeros((max_steps, self.env.n_agents, len(loader.dataset), self.env.mol_types), dtype=np.float32)

        with torch.no_grad():
            sample_ptr = 0
            for batch_idx, (X_batch, y_batch) in enumerate(loader):
                X_batch = X_batch.detach().cpu().numpy()
                y_batch = y_batch.detach().cpu().numpy()
                batch_size = X_batch.shape[0]

                for offset in range(0, batch_size, self.n_envs):
                    valid_envs = min(self.n_envs, batch_size - offset)
                    start = sample_ptr
                    end = sample_ptr + valid_envs
                    X_eval = X_batch[offset:offset + valid_envs]
                    y_eval = y_batch[offset:offset + valid_envs]

                    if self.vectorized_env and valid_envs < self.n_envs:
                        pad = self.n_envs - valid_envs
                        X_eval = np.concatenate([X_eval, np.repeat(X_eval[-1:], pad, axis=0)], axis=0)
                        y_eval = np.concatenate([y_eval, np.repeat(y_eval[-1:], pad, axis=0)], axis=0)

                    if not self.vectorized_env:
                        X_eval = X_eval[0]
                        y_eval = y_eval[0]

                    obs, infos = self.env.reset(X_eval, y_eval)
                    obs = self._as_env_batch(obs)

                    dones = np.zeros((self.n_agents, self.n_envs), dtype=bool)
                    step_count = 0

                    while not np.all(dones[:, :valid_envs]):
                        actions_array = np.zeros((self.n_agents, self.n_envs, self.env.mol_types), dtype=np.float32)

                        for pid, policy in self.policies.items():
                            agent_ids = self.policy_agent_mapping[pid]
                            action_tensor = policy.predict(obs[agent_ids], deterministic=True)
                            actions_array[agent_ids, :] = action_tensor

                        obs, rewards, dones, infos = self.env.step(self._step_actions(actions_array))
                        obs = self._as_env_batch(obs)
                        dones = self._as_agent_env_done(dones.get('__all__', False))
                        all_actions[step_count, :, start:end, :] = actions_array[:, :valid_envs, :]
                        step_count += 1

                    total_steps += step_count
                    sample_ptr = end

                    correct += self._env_correct_count(infos, valid_envs)
                    total += valid_envs

        self._set_env_mode(training=True)
        self.env.set_deterministic(False)
        for policy in self.policies.values():
            policy.set_training_mode(True)

        accuracy = correct / total if total > 0 else 0.0

        if not test:
            self.metrics.log("val_acc", accuracy)
            self._tb_scalar("eval/val_acc", accuracy, self.global_step)
        else:
            self.metrics.log("test_acc", accuracy)
            self.metrics.log("test_messages", all_actions)
            self._tb_scalar("eval/test_acc", accuracy, self.global_step)

    def load_best_policies(self):

        for pid, p in self.policies.items():

            p.load_state_dict(self.best_policies[pid])

        return self.best_policies


def _create_entropy_scheduler(config):

    """
    Crea scheduler per entropy coefficient.
    """
    if config is None:
        return None

    schedule = _create_schedule_from_config(config)

    return EntropyScheduler(schedule)


class MAPPOLearner(BasicLearner):

    def __init__(self,
                 env: Environment,
                 algorithm: RLAlgorithm,
                 policies: dict[str, BasePolicy],
                 agent_policy_mapping: dict,
                 action_dim: int,
                 obs_dim: int,
                 rollout_len: int,
                 data_batch_size: int,
                 X_train: Optional[np.ndarray] = None,
                 y_train: Optional[np.ndarray] = None,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 critic_obs_dim: Optional[int] = None,
                 lr_scheduler_config: Optional[Union[dict[str, Any], dict[str, dict[str, Any]]]] = None,
                 verbose: bool = False,
                 device: str = 'cpu',
                 entropy_scheduler_config: Optional[Union[dict[str, Any], dict[str, Any]]] = None,
                 tensorboard_log_dir: Optional[Union[str, Path]] = None,
                 log_interval: int = 10,
                 ):

        super(MAPPOLearner, self).__init__(env, algorithm, policies, agent_policy_mapping, action_dim, obs_dim,
                                           rollout_len, data_batch_size, X_train, y_train, val_loader,
                                           test_loader, gamma, gae_lambda, critic_obs_dim,
                                           lr_scheduler_config, verbose, device, tensorboard_log_dir,
                                           log_interval)

        assert isinstance(algorithm, MAPPO), "MAPPOLearner only support MAPPO algorithm"

        self.entropy_scheduler = _create_entropy_scheduler(entropy_scheduler_config)
        self.current_entropy_coeff = self.algorithm.c2
        self.entropy_coeff_history = []

    def _step_schedulers(self):
        super()._step_schedulers()
        # Step entropy scheduler
        if self.entropy_scheduler is not None:
            new_entropy_coeff = self.entropy_scheduler.step()
            self.entropy_coeff_history.append(new_entropy_coeff)
            self.algorithm.c2 = new_entropy_coeff
        else:
            self.entropy_coeff_history.append(self.current_entropy_coeff)
