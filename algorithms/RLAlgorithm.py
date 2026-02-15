from abc import abstractmethod, ABC

import numpy as np

from utils.utils import moving_average
import matplotlib.pyplot as plt

class RLAlgorithm(ABC):

    def __init__(self,
                 device='cpu'):

        self.device = device
        self.avg_returns = []

    def get_random_batched_data_indexes(self, batch_size, array_len):
        batch_start = np.arange(0, array_len, batch_size)
        indices = np.arange(array_len, dtype=np.float32)
        np.random.shuffle(indices)
        batches = [indices[i:i + batch_size] for i in batch_start]
        return batches

    @abstractmethod
    def update(self, policy_id, policy_net, history_buffers, next_value):
        raise NotImplementedError("You must provide an implementation of the update method for the learning algorithm")

    def _plot(self, window_size=200):

        plt.figure(figsize=(18, 12))

        # Plot esistenti...
        plt.subplot(3, 4, 1)
        smoothed_rewards = moving_average(self.avg_rewards, window_size)
        plt.plot(smoothed_rewards, color='tab:blue')
        plt.title(f'Reward per Episode (Smoothed, window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 4, 2)
        for i, loss in enumerate(self.agents_losses):
            plt.plot(moving_average(loss, window_size),
                     color=f"C{i}", label='{}'.format(list(self.policies.keys())[i]))

        plt.title(f'Loss per Episode (Smoothed, window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 4, 3)
        for i, entropy in enumerate(self.avg_agents_entropies):
            plt.plot(moving_average(entropy, window_size),
                     color=f"C{i}", label='{}'.format(list(self.policies.keys())[i]))

        plt.title(f'Entropy per Episode (Smoothed, window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Entropy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 4, 4)
        plt.plot(self.avg_rewards, alpha=0.3, color='tab:blue', label='Raw')
        plt.plot(smoothed_rewards, color='tab:blue', label='Smoothed')
        plt.title('Raw vs Smoothed Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 4, 5)
        for i, c_loss in enumerate(self.critic_losses):
            plt.plot(moving_average(c_loss, window_size),
                     color=f"C{i}", label='{}'.format(list(self.policies.keys())[i]))

        plt.title('Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 4, 6)
        for i, a_return in enumerate(self.avg_returns):
            plt.plot(moving_average(a_return, window_size),
                     color=f"C{i}", label='{}'.format(list(self.policies.keys())[i]))

        plt.title('Average Returns')
        plt.xlabel('Episode')
        plt.ylabel('Average Return')
        plt.grid(True, alpha=0.3)

        # Nuovo plot per learning rates
        plt.subplot(3, 4, 7)
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        color_idx = 0
        for policy_id, lr_data in self.lr_history.items():
            for scheduler_type, lr_history in lr_data.items():
                if lr_history:
                    plt.plot(lr_history, color=colors[color_idx % len(colors)],
                             label=f'{policy_id} {scheduler_type}', linewidth=1)
                    color_idx += 1
        plt.title('Learning Rates Over Time')
        plt.xlabel('Updates')
        plt.ylabel('Learning Rate')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

    def plot(self, window_size=200):
        self._plot(window_size)

        plt.tight_layout()
        plt.show()




