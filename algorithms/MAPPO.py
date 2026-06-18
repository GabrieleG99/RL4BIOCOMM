from matplotlib import pyplot as plt
import gymnasium.spaces as spaces
import torch

from algorithms.RLAlgorithm import RLAlgorithm

from utils.utils import moving_average

import numpy as np

class MAPPO(RLAlgorithm):

    def __init__(self,
                 mini_batch_size=128,
                 v_clip=None,
                 max_grad_norm=1,
                 eps_clip=0.2,
                 c1=1,
                 c2=0.01,
                 epochs=4,
                 target_kl=0.01,
                 std_advantages=True,
                 sequence_len=None,
                 device='cpu'):

        super(MAPPO, self).__init__(
            device=device,
        )

        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.v_clip = v_clip
        self.target_kl = target_kl
        self.std_advantages = std_advantages
        self.mini_batch_size = mini_batch_size
        self.sequence_len = sequence_len

        self.avg_surr_loss = []

    @staticmethod
    def _joint_distribution_value(values: torch.Tensor, policy_net) -> torch.Tensor:
        if isinstance(policy_net.action_space, spaces.Box) and values.dim() > 0 and values.shape[-1] != 1:
            return values.sum(dim=-1, keepdim=True)
        if values.dim() == 0:
            return values.reshape(1, 1)
        if values.dim() >= 3 and values.shape[-1] == 1:
            return values
        return values.unsqueeze(-1)

    def _critic_loss(self, state_values, old_state_values, returns, policy_net):
        if policy_net.use_popart:
            policy_net.critic_head.update(returns)
            target_returns = policy_net.critic_head.normalize(returns)
            old_values = policy_net.critic_head.normalize(old_state_values)
        else:
            target_returns = returns
            old_values = old_state_values

        value_loss = (state_values - target_returns).pow(2)

        if self.v_clip is None:
            return value_loss.mean()

        clipped_values = old_values + torch.clamp(
            state_values - old_values,
            -self.v_clip,
            self.v_clip,
        )
        clipped_value_loss = (clipped_values - target_returns).pow(2)
        return torch.max(value_loss, clipped_value_loss).mean()

    def update(self, policy_id, policy_net, history_buffers, next_value):
        if getattr(policy_net, "is_recurrent", False):
            return self._update_recurrent(policy_id, policy_net, history_buffers, next_value)

        policy_loss = 0
        policy_entropy = 0
        avg_epoch_critic_loss = 0
        avg_epoch_surr_loss = 0


        all_returns, all_advantages = [], []
        for i, buffer in enumerate(history_buffers):
            returns, advantages = buffer.compute_returns_and_advantages(next_value[i])
            all_returns.append(returns)
            all_advantages.append(advantages)

        if not all_returns:
            return

        returns = np.stack(all_returns)
        advantages = np.stack(all_advantages)

        # Collect buffer data
        #agent_buffers = [self.history_buffers[idx] for idx in agent_indices]

        def reshape_and_concat(attr_name):
            data_list = [getattr(b, attr_name)[np.newaxis, :, :] for b in history_buffers]
            concatenated_data = np.concatenate(data_list, axis=0)
            return torch.from_numpy(concatenated_data).float().to(self.device)

        old_states = reshape_and_concat("states")
        old_actions = reshape_and_concat("actions")
        old_logprobs = reshape_and_concat("logprobs")
        old_critic_states = reshape_and_concat("crit_state")
        old_state_values = reshape_and_concat("state_values")
        advantages = torch.from_numpy(advantages).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)

        avg_returns = returns.mean(dim=0)

        if advantages.shape[1] > 1 and self.std_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = advantages.shape[1]

        continue_training = True
        update_count = 0

        for epoch in range(self.epochs):
            if not continue_training:
                break

            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for i in range(0, n_samples, self.mini_batch_size):
                mb_indices = indices[i:i + self.mini_batch_size]
                mb_advantages = advantages[:, mb_indices]

                state_values = policy_net.evaluate_critic(old_critic_states[:, mb_indices])
                critic_loss = self._critic_loss(
                    state_values,
                    old_state_values[:, mb_indices],
                    returns[:, mb_indices],
                    policy_net,
                )

                log_probs, entropies = policy_net.evaluate(old_states[:, mb_indices],
                                                           old_actions[:, mb_indices])
                log_probs = self._joint_distribution_value(log_probs, policy_net)
                entropies = self._joint_distribution_value(entropies, policy_net)

                ratios = torch.exp(log_probs - old_logprobs[:, mb_indices])

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages

                ppo_loss = torch.min(surr1, surr2).mean()

                # Usa il coefficiente entropy corrente dallo scheduler
                ent_loss = self.c2 * entropies.mean()
                loss = -(ppo_loss + ent_loss)

                with torch.no_grad():
                    diff_logprob = log_probs - old_logprobs[:, mb_indices]
                    approx_kl_div = torch.mean((ratios - 1) - diff_logprob).cpu().numpy()

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch} for policy {policy_id} due to max KL.")
                    continue_training = False
                    break

                policy_loss += loss.item()
                policy_entropy += entropies.mean().item()
                avg_epoch_critic_loss += critic_loss.item()
                avg_epoch_surr_loss += ppo_loss.item()
                update_count += 1

                policy_net.critic_optimizer.zero_grad()
                (critic_loss * self.c1).backward()
                torch.nn.utils.clip_grad_norm_(
                    policy_net.critic_features_extractor.parameters(), max_norm=self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(policy_net.critic_head.parameters(), max_norm=self.max_grad_norm)
                policy_net.critic_optimizer.step()

                policy_net.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy_net.act_features_extractor.parameters(), max_norm=self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(policy_net.actor_head.parameters(), max_norm=self.max_grad_norm)
                policy_net.actor_optimizer.step()

                # policy_net.optimizer.zero_grad()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=self.max_grad_norm)
                # policy_net.optimizer.step()

        if update_count > 0:
            avg_epoch_surr_loss /= update_count
            avg_epoch_critic_loss /= update_count
            policy_loss /= update_count
            policy_entropy /= update_count

            metrics = {
                "loss": policy_loss,
                "entropy": policy_entropy,
                "surr_loss": avg_epoch_surr_loss,
                "critic_loss": avg_epoch_critic_loss,
            }
            return avg_returns, metrics

        else:
            return avg_returns, None

    def _get_sequence_refs(self, history_buffers, sequence_len):
        refs = []
        for buffer_idx, buffer in enumerate(history_buffers):
            max_start = buffer.step - sequence_len + 1
            for env_id in range(buffer.n_envs):
                for start in range(max_start):
                    end = start + sequence_len
                    if sequence_len > 1 and buffer.is_terminals[env_id, start:end - 1].any():
                        continue
                    refs.append((buffer_idx, env_id, start))
        return refs

    def _recurrent_sequence_batch(self, history_buffers, sequence_refs, sequence_len):
        def stack_sequence(attr_name):
            return np.stack([
                getattr(history_buffers[buffer_idx], attr_name)[env_id, start:start + sequence_len]
                for buffer_idx, env_id, start in sequence_refs
            ])

        def stack_initial(attr_name):
            return np.stack([
                getattr(history_buffers[buffer_idx], attr_name)[env_id, start]
                for buffer_idx, env_id, start in sequence_refs
            ])

        batch = {
            "states": stack_sequence("states"),
            "actions": stack_sequence("actions"),
            "logprobs": stack_sequence("logprobs"),
            "crit_states": stack_sequence("crit_state"),
            "state_values": stack_sequence("state_values"),
            "returns": np.stack([
                history_buffers[buffer_idx].returns[env_id, start:start + sequence_len]
                for buffer_idx, env_id, start in sequence_refs
            ]),
            "advantages": np.stack([
                history_buffers[buffer_idx].advantages[env_id, start:start + sequence_len]
                for buffer_idx, env_id, start in sequence_refs
            ]),
            "actor_hidden": stack_initial("actor_hidden_states"),
            "critic_hidden": stack_initial("critic_hidden_states"),
        }
        return batch

    def _hidden_to_torch(self, hidden):
        hidden = torch.from_numpy(hidden).float().to(self.device)
        return hidden.permute(1, 0, 2).contiguous().detach()

    def _update_recurrent(self, policy_id, policy_net, history_buffers, next_value):
        policy_loss = 0
        policy_entropy = 0
        avg_epoch_critic_loss = 0
        avg_epoch_surr_loss = 0

        all_returns, all_advantages = [], []
        for i, buffer in enumerate(history_buffers):
            returns, advantages = buffer.compute_returns_and_advantages(next_value[i])
            all_returns.append(returns)
            all_advantages.append(advantages)

        if not all_returns:
            return

        min_steps = min(buffer.step for buffer in history_buffers)
        sequence_len = self.sequence_len or min_steps
        sequence_len = min(sequence_len, min_steps)
        if sequence_len <= 0:
            return

        sequence_refs = self._get_sequence_refs(history_buffers, sequence_len)
        if not sequence_refs and sequence_len > 1:
            sequence_len = 1
            sequence_refs = self._get_sequence_refs(history_buffers, sequence_len)

        if not sequence_refs:
            return torch.from_numpy(np.stack(all_returns)).float().to(self.device), None

        returns_array = np.stack(all_returns)
        avg_returns = torch.from_numpy(returns_array).float().to(self.device).mean(dim=0)

        adv_mean, adv_std = None, None
        if self.std_advantages:
            flat_advantages = np.concatenate([adv.reshape(-1) for adv in all_advantages])
            if flat_advantages.size > 1:
                adv_mean = float(flat_advantages.mean())
                adv_std = float(flat_advantages.std() + 1e-8)

        continue_training = True
        update_count = 0

        for epoch in range(self.epochs):
            if not continue_training:
                break

            indices = np.arange(len(sequence_refs))
            np.random.shuffle(indices)

            for i in range(0, len(sequence_refs), self.mini_batch_size):
                mb_refs = [sequence_refs[idx] for idx in indices[i:i + self.mini_batch_size]]
                batch = self._recurrent_sequence_batch(history_buffers, mb_refs, sequence_len)

                old_states = torch.from_numpy(batch["states"]).float().to(self.device)
                old_actions = torch.from_numpy(batch["actions"]).float().to(self.device)
                old_logprobs = torch.from_numpy(batch["logprobs"]).float().to(self.device)
                old_critic_states = torch.from_numpy(batch["crit_states"]).float().to(self.device)
                old_state_values = torch.from_numpy(batch["state_values"]).float().to(self.device)
                returns = torch.from_numpy(batch["returns"]).float().to(self.device)
                mb_advantages = torch.from_numpy(batch["advantages"]).float().to(self.device)

                if adv_mean is not None:
                    mb_advantages = (mb_advantages - adv_mean) / adv_std

                critic_hidden = self._hidden_to_torch(batch["critic_hidden"])
                state_values = policy_net.evaluate_critic(
                    old_critic_states,
                    hidden_state=critic_hidden,
                    sequence=True,
                )
                critic_loss = self._critic_loss(
                    state_values,
                    old_state_values,
                    returns,
                    policy_net,
                )

                actor_hidden = self._hidden_to_torch(batch["actor_hidden"])
                log_probs, entropies = policy_net.evaluate(
                    old_states,
                    old_actions,
                    hidden_state=actor_hidden,
                    sequence=True,
                )
                log_probs = self._joint_distribution_value(log_probs, policy_net)
                entropies = self._joint_distribution_value(entropies, policy_net)

                ratios = torch.exp(log_probs - old_logprobs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages

                ppo_loss = torch.min(surr1, surr2).mean()
                ent_loss = self.c2 * entropies.mean()
                loss = -(ppo_loss + ent_loss)

                with torch.no_grad():
                    diff_logprob = log_probs - old_logprobs
                    approx_kl_div = torch.mean((ratios - 1) - diff_logprob).cpu().numpy()

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch} for policy {policy_id} due to max KL.")
                    continue_training = False
                    break

                policy_loss += loss.item()
                policy_entropy += entropies.mean().item()
                avg_epoch_critic_loss += critic_loss.item()
                avg_epoch_surr_loss += ppo_loss.item()
                update_count += 1

                policy_net.critic_optimizer.zero_grad()
                (critic_loss * self.c1).backward()
                torch.nn.utils.clip_grad_norm_(
                    policy_net.critic_features_extractor.parameters(), max_norm=self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(policy_net.critic_head.parameters(), max_norm=self.max_grad_norm)
                policy_net.critic_optimizer.step()

                policy_net.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy_net.act_features_extractor.parameters(), max_norm=self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(policy_net.actor_head.parameters(), max_norm=self.max_grad_norm)
                policy_net.actor_optimizer.step()

        if update_count > 0:
            avg_epoch_surr_loss /= update_count
            avg_epoch_critic_loss /= update_count
            policy_loss /= update_count
            policy_entropy /= update_count

            metrics = {
                "loss": policy_loss,
                "entropy": policy_entropy,
                "surr_loss": avg_epoch_surr_loss,
                "critic_loss": avg_epoch_critic_loss,
            }
            return avg_returns, metrics

        return avg_returns, None

    def _plot(self, window_size=200):
        super()._plot(window_size)

        # Plot surrogate loss
        plt.subplot(3, 4, 8)
        for i, avg_surr_loss in enumerate(self.avg_surr_loss):
            plt.plot(moving_average(avg_surr_loss, window_size),
                     label='{}'.format(list(self.policies.keys())[i]), color=f'C{i}')
        plt.title('Average Surrogate Loss')
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot entropy coefficient
        plt.subplot(3, 4, 9)
        if self.entropy_coeff_history:
            plt.plot(self.entropy_coeff_history, 'b-', label='Entropy Coefficient')
        plt.title('Entropy Coefficient Over Time')
        plt.xlabel('Updates')
        plt.ylabel('Coefficient')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot(self, window_size=200):
        """Override plot per aggiornare layout con subplot aggiuntivi."""
        self._plot(window_size)
        plt.tight_layout()
        plt.show()

    def metrics_clear(self):
        """Override per includere anche entropy history."""
        super().metrics_clear()
        self.entropy_coeff_history = []
        self.avg_surr_loss = [[] for _ in range(len(self.policies))]

