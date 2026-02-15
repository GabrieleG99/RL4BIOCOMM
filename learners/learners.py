from typing import Optional, Union, Any

import numpy as np

from algorithms.MAPPO import MAPPO
from envs import Environment
from models.policies import BasePolicy
from algorithms.RLAlgorithm import RLAlgorithm
from utils.buffers import RolloutBuffer
from utils.loggers import MetricsStore

import torch

from utils.schedulers import LearningRateScheduler, LinearDecaySchedule, ExponentialDecaySchedule, EntropyScheduler


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
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 val_loader: Optional[torch.utils.data.DataLoader]=None,
                 test_loader: Optional[torch.utils.data.DataLoader]=None,
                 gamma: float=0.99,
                 gae_lambda: float=0.95,
                 critic_obs_dim: Optional[int]=None,
                 lr_scheduler_config: Optional[Union[dict[str, Any], dict[str, dict[str, Any]]]]=None,
                 verbose: bool=False,
                 device: str='cpu'
                 ) -> None:


        """
        agent_policy_mapping: dict[int, str] example ({
            'agent_1': 'policy_1',
            'agent_2': 'policy_2',
            'agent_3': 'policy_1',
            'agent_4': 'policy_1',
            ...
        })

        """

        assert rollout_len % data_batch_size == 0, "Batch size must be divisible by train loader batch size"
        assert rollout_len >= data_batch_size, "Batch size must be smaller than train loader batch size"

        self.algorithm = algorithm
        self.policies = policies
        self.agent_policy_mapping = agent_policy_mapping
        self.n_agents = len(agent_policy_mapping)
        self.env = env
        self.rollouts = rollout_len
        self.action_dim = action_dim
        self.batch_size = data_batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.device = device
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.verbose = verbose

        self.policy_agent_mapping = {pid: [] for pid in policies.keys()}

        for agent_id, policy_id in agent_policy_mapping.items():
            self.policy_agent_mapping[policy_id].append(agent_id)

        self.history_buffers = [
            RolloutBuffer(rollout_len, data_batch_size, env.n_iters, action_dim, obs_dim, gae_lambda=gae_lambda, gamma=gamma,
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

        self.best_policies = {}


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
                    actor_schedule = self._create_schedule_from_config(actor_config)
                    policy_schedulers['actor'] = LearningRateScheduler(
                        policy_net.actor_optimizer, actor_schedule
                    )

                if 'critic' in pol_config:
                    critic_config = pol_config['critic']
                    critic_schedule = self._create_schedule_from_config(critic_config)
                    policy_schedulers['critic'] = LearningRateScheduler(
                        policy_net.critic_optimizer, critic_schedule
                    )
            else:
                # Configurazione unificata (applica a tutti gli ottimizzatori)
                schedule = self._create_schedule_from_config(pol_config)

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

    def _create_schedule_from_config(self, config):
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

    def train(self, episodes: int=5000):

        best_val_acc = 0

        self.env.encoder.train(), self.env.decoder.train()
        for eparam, dparam in zip(self.env.encoder.parameters(), self.env.decoder.parameters()):
            eparam.requires_grad = False
            dparam.requires_grad = False

        for policy in self.policies.values():
            policy.set_training_mode(True)
        self.env.set_deterministic(False)

        for episode in range(episodes):

            self.__collect_trajectories()

            for pid, policy in self.policies.items():
                agent_ids = self.policy_agent_mapping[pid]
                buffers = [b for i, b in enumerate(self.history_buffers) if i in agent_ids]
                with torch.no_grad():
                    next_value = policy.get_state_value(self._next_critic_obs[agent_ids]).detach().cpu().numpy()
                avg_returns, metrics = self.algorithm.update(pid, policy, buffers, next_value)
                for aid in agent_ids:
                    self.history_buffers[aid].clear()

                if metrics is not None:
                    agent_id_strs = [f"agent_{aid}" for aid in agent_ids]
                    self.metrics.log_for_agents(agent_id_strs, metrics)
                    self.metrics.log_for_agents(
                        agent_id_strs,
                        {"avg_return": avg_returns.mean().item()}
                    )

            self._step_schedulers()

            if (episode + 1) % 20 == 0:
                if self.val_loader is not None:
                    self.evaluate()
                    current_val_acc = self.metrics.last("val_acc")

                    if current_val_acc >= best_val_acc:
                        best_val_acc = current_val_acc
                        for key, value in self.policies.items():
                            self.best_policies[key] = {k: v.clone() for k, v in value.state_dict().items()}

            if (episode + 1) % 100 == 0 and self.verbose:
                print(f'Episode {episode + 1} completed. Val Acc: {self.metrics.last("val_acc"): .2f}, Avg Return: {self.metrics.last("avg_return", agent_id="agent_0"): .2f}')

        self.evaluate(test=True)


    def __collect_trajectories(self):

        count = 0

        while count < self.rollouts:

            data_batch_indices = np.random.choice(len(self.X_train), self.batch_size)
            X_batch, y_batch = self.X_train[data_batch_indices], self.y_train[data_batch_indices]

            obs, infos = self.env.reset(X_batch, y_batch)
            dones = np.zeros(self.n_agents, dtype=bool)

            while not np.all(dones) and count < self.rollouts:

                actions_array = np.zeros((self.n_agents, self.batch_size, self.action_dim), dtype=np.float32)

                for policy_id, policy in self.policies.items():
                    agent_ids = self.policy_agent_mapping[policy_id]
                    obs_tensor = torch.from_numpy(obs[agent_ids]).float().to(self.env.device)
                    with torch.no_grad():
                        action_tensor, log_prob = policy(obs_tensor)

                        if 'shared_obs' in infos:
                            critic_input = torch.from_numpy(infos['shared_obs']).float().to(self.env.device)
                            critic_input = critic_input.expand(len(agent_ids), -1, -1)
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

                next_obs, rewards, dones, infos = self.env.step(actions_array)

                dones = dones['__all__']

                for aid in range(self.env.n_agents):
                    self.history_buffers[aid].add(
                        rewards=rewards['agent_'+str(aid)],
                        is_terminals=dones
                    )

                with torch.no_grad():
                    if 'shared_obs' in infos:
                        next_critic_obs = torch.from_numpy(infos['shared_obs']).float().to(self.env.device)
                        self._next_critic_obs = next_critic_obs.expand(self.n_agents, -1, -1)
                    else:
                        self._next_critic_obs = torch.from_numpy(next_obs).float().to(self.env.device)

                obs = next_obs

                count += self.batch_size


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
        self.env.encoder.eval()
        self.env.decoder.eval()

        for policy in self.policies.values():
            policy.set_training_mode(False)

        all_actions = np.zeros((max_steps, self.env.n_agents, len(loader.dataset), self.env.mol_types), dtype=np.float32)

        with torch.no_grad():
            sample_ptr = 0
            for batch_idx, (X_batch, y_batch) in enumerate(loader):
                X_batch, y_batch = X_batch.to(self.env.device), y_batch.to(self.env.device)
                batch_size = X_batch.size(0)
                start = sample_ptr
                end = sample_ptr + batch_size

                obs, infos = self.env.reset(X_batch, y_batch)

                dones = np.zeros(self.n_agents, dtype=bool)
                step_count = 0

                while not np.all(dones):
                    actions_array = np.zeros((self.env.n_agents, batch_size, self.env.mol_types), dtype=np.float32)

                    for pid, policy in self.policies.items():
                        agent_ids = self.policy_agent_mapping[pid]
                        action_tensor = policy.predict(obs[agent_ids], deterministic=True)
                        actions_array[agent_ids, :] = action_tensor

                    obs, rewards, dones, infos = self.env.step(actions_array)
                    dones = dones.get('__all__', False)
                    all_actions[step_count, :, start:end, :] = actions_array
                    step_count += 1

                total_steps += step_count
                sample_ptr = end

                batch_correct = infos['__all__']['correct'].sum().item()
                correct += batch_correct
                total += y_batch.size(0)

        self.env.encoder.train()
        self.env.decoder.train()
        self.env.set_deterministic(False)
        for policy in self.policies.values():
            policy.set_training_mode(True)

        accuracy = correct / total if total > 0 else 0.0

        if not test:
            self.metrics.log("val_acc", accuracy)
        else:
            self.metrics.log("test_acc", accuracy)
            self.metrics.log("test_messages", all_actions)


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
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 critic_obs_dim: Optional[int] = None,
                 lr_scheduler_config: Optional[Union[dict[str, Any], dict[str, dict[str, Any]]]] = None,
                 verbose: bool = False,
                 device: str = 'cpu',
                 entropy_scheduler_config: Optional[Union[dict[str, Any], dict[str, Any]]] = None,
                 ):

        super(MAPPOLearner, self).__init__(env, algorithm, policies, agent_policy_mapping, action_dim, obs_dim,
                                           rollout_len, data_batch_size, X_train, y_train, val_loader,
                                           test_loader, gamma, gae_lambda, critic_obs_dim,
                                           lr_scheduler_config, verbose, device)

        assert isinstance(algorithm, MAPPO), "MAPPOLearner only support MAPPO algorithm"

        self.entropy_scheduler = self._create_entropy_scheduler(entropy_scheduler_config)
        self.current_entropy_coeff = self.algorithm.c2
        self.entropy_coeff_history = []

    def _create_entropy_scheduler(self, config):

        """
        Crea scheduler per entropy coefficient.
        """
        if config is None:
            return None

        schedule = self._create_schedule_from_config(config)

        return EntropyScheduler(schedule)

    def _step_schedulers(self):
        super()._step_schedulers()
        # Step entropy scheduler
        if self.entropy_scheduler is not None:
            new_entropy_coeff = self.entropy_scheduler.step()
            self.entropy_coeff_history.append(new_entropy_coeff)
            self.algorithm.c2 = new_entropy_coeff
        else:
            self.entropy_coeff_history.append(self.current_entropy_coeff)


