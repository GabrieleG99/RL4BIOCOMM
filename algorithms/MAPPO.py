from matplotlib import pyplot as plt
import torch
from torch.nn.functional import mse_loss

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

        self.avg_surr_loss = []

    def update(self, policy_id, policy_net, history_buffers, next_value):

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

        for epoch in range(self.epochs):
            if not continue_training:
                break

            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for i in range(0, n_samples, self.mini_batch_size):
                mb_indices = indices[i:i + self.mini_batch_size]
                mb_advantages = advantages[:, mb_indices]

                state_values = policy_net.evaluate_critic(old_critic_states[:, mb_indices])

                if self.v_clip is not None:
                    old_v = old_state_values[:, mb_indices]
                    state_values = old_v + torch.clamp(state_values - old_v, -self.v_clip, self.v_clip)

                if policy_net.use_popart:
                    policy_net.critic_head.update(returns[:, mb_indices])
                    norm_returns = policy_net.critic_head.normalize(returns[:, mb_indices])
                    critic_loss = mse_loss(norm_returns, state_values)
                else:
                    critic_loss = mse_loss(returns[:, mb_indices], state_values)

                log_probs, entropies = policy_net.evaluate(old_states[:, mb_indices],
                                                           old_actions[:, mb_indices])

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

                policy_net.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy_net.act_features_extractor.parameters(), max_norm=self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(policy_net.actor_head.parameters(), max_norm=self.max_grad_norm)
                policy_net.actor_optimizer.step()

                policy_net.critic_optimizer.zero_grad()
                (critic_loss * self.c1).backward()
                torch.nn.utils.clip_grad_norm_(
                    policy_net.critic_features_extractor.parameters(), max_norm=self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(policy_net.critic_head.parameters(), max_norm=self.max_grad_norm)
                policy_net.critic_optimizer.step()

                # policy_net.optimizer.zero_grad()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=self.max_grad_norm)
                # policy_net.optimizer.step()

        num_updates = self.epochs * (
                    n_samples // self.mini_batch_size) if n_samples > self.mini_batch_size else self.epochs

        if num_updates > 0:
            avg_epoch_surr_loss /= num_updates
            avg_epoch_critic_loss /= num_updates
            policy_loss /= num_updates
            policy_entropy /= num_updates

            metrics = {
                "loss": policy_loss,
                "entropy": policy_entropy,
                "surr_loss": avg_epoch_surr_loss,
                "critic_loss": avg_epoch_critic_loss,
            }
            return avg_returns, metrics

        else:
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




