import numpy as np
import torch
from torch.nn.functional import mse_loss

from envs import Environment
from algorithms.RLAlgorithm import RLAlgorithm
from models.policies import BasePolicy
from utils.schedulers import LinearDecaySchedule, ExponentialDecaySchedule, EntropyScheduler


class REINFORCE(RLAlgorithm):

    def __init__(self,
                 env: Environment,
                 policies: dict[str, BasePolicy],
                 agent_policy_mapping: dict,
                 action_dim: int,
                 obs_dim: int,
                 critic_obs_dim: int,
                 rollout_len: int,
                 data_batch_size: int,
                 optimizer=None,
                 gamma=0.99,
                 lambda_gae=0.95,
                 c1=1.0,
                 c2=0.01,
                 std_advantages=True,
                 use_popart: bool = False,
                 device='cpu',
                 entropy_scheduler_config: dict = None,
                 lr_scheduler_config: dict = None):
        super(REINFORCE, self).__init__(
            env, policies, agent_policy_mapping,
            rollout_len=rollout_len, data_batch_size=data_batch_size,
            action_dim=action_dim, obs_dim=obs_dim, critic_obs_dim=critic_obs_dim,
            optimizer=optimizer, gae_lambda=lambda_gae, gamma=gamma, device=device,
            lr_scheduler_config=lr_scheduler_config
        )

        self.c1 = c1  # coefficiente critic
        self.c2 = c2  # coefficiente entropia
        self.std_advantages = std_advantages
        self._use_popart = use_popart

        self.entropy_scheduler = self._create_entropy_scheduler(entropy_scheduler_config)
        self.current_entropy_coeff = self.c2
        self.entropy_coeff_history = []

    def _create_entropy_scheduler(self, config):
        if config is None:
            return None
        scheduler_type = config.get('type', 'linear')
        initial_value = config.get('initial_value', self.c2)
        final_value = config.get('final_value', 0.0001)
        total_steps = config.get('total_steps', 1000)

        if scheduler_type == 'linear':
            schedule = LinearDecaySchedule(initial_value, final_value, total_steps)
        elif scheduler_type == 'exp':
            decay_rate = config.get('decay_rate', None)
            schedule = ExponentialDecaySchedule(initial_value, final_value, total_steps, decay_rate)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        return EntropyScheduler(schedule)

    def get_current_entropy_coeff(self):
        return self.current_entropy_coeff

    def update(self):
        super().update()

        # Aggiorna coefficiente entropia
        if self.entropy_scheduler is not None:
            self.current_entropy_coeff = self.entropy_scheduler.get_value()
        else:
            self.current_entropy_coeff = self.c2

        n_policies = len(self.policies)
        policy_actor_losses = [0 for _ in range(n_policies)]
        policy_entropies = [0 for _ in range(n_policies)]
        policy_critic_losses = [0 for _ in range(n_policies)]

        for policy_id, policy_net in self.policies.items():
            policy_idx = self.policy_id_to_idx[policy_id]
            agent_indices = [idx for idx, p_id in self.agent_policy_mapping.items() if p_id == policy_id]
            if not agent_indices:
                continue

            all_returns, all_advantages = [], []
            for agent_idx in agent_indices:
                returns, advantages = self.history_buffers[agent_idx].compute_returns_and_advantages()
                all_returns.append(returns)
                all_advantages.append(advantages)

            if not all_returns:
                continue

            returns = np.stack(all_returns)
            advantages = np.stack(all_advantages)

            buffers = [self.history_buffers[idx] for idx in agent_indices]

            def stack_attr(name):
                data_list = [getattr(b, name)[np.newaxis, :, :] for b in buffers]
                return torch.from_numpy(np.concatenate(data_list, axis=0)).float().to(self.device)

            states = stack_attr("states")
            actions = stack_attr("actions")
            critic_states = stack_attr("crit_state")
            returns = torch.from_numpy(returns).float().to(self.device)
            advantages = torch.from_numpy(advantages).float().to(self.device)

            self.avg_returns[policy_idx].append(returns.mean().item())

            if advantages.shape[1] > 1 and self.std_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Valutazioni modello
            log_probs, entropies = policy_net.evaluate(states, actions)
            state_values = policy_net.evaluate_critic(critic_states)

            # Critic loss (opzionale PopArt)
            if self._use_popart:
                policy_net.critic_head.update(returns)
                norm_returns = policy_net.critic_head.normalize(returns)
                critic_loss = mse_loss(norm_returns, state_values)
            else:
                critic_loss = mse_loss(returns, state_values)

            # Actor loss (REINFORCE)
            reinforce_loss = -(log_probs * advantages).mean()
            ent_loss = -self.current_entropy_coeff * entropies.mean()
            actor_loss = reinforce_loss + ent_loss

            # Backprop actor
            policy_net.actor_optimizer.zero_grad()
            actor_loss.backward()
            policy_net.actor_optimizer.step()

            # Backprop critic
            policy_net.critic_optimizer.zero_grad()
            (critic_loss * self.c1).backward()
            policy_net.critic_optimizer.step()

            # Log metriche
            policy_actor_losses[policy_idx] = actor_loss.item()
            policy_entropies[policy_idx] = entropies.mean().item()
            policy_critic_losses[policy_idx] = critic_loss.item()

            self.agents_losses[policy_idx].append(policy_actor_losses[policy_idx])
            self.avg_agents_entropies[policy_idx].append(policy_entropies[policy_idx])
            self.critic_losses[policy_idx].append(policy_critic_losses[policy_idx])

        self._step_schedulers()
        self.clear_all_buffers()

    def _step_schedulers(self):
        self._step_lr_schedulers()
        if self.entropy_scheduler is not None:
            new_entropy_coeff = self.entropy_scheduler.step()
            self.entropy_coeff_history.append(new_entropy_coeff)
        else:
            self.entropy_coeff_history.append(self.current_entropy_coeff)

    def metrics_clear(self):
        super().metrics_clear()
        self.entropy_coeff_history = []
