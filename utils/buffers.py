import math

import numpy as np

class RolloutBuffer:
    def __init__(self, rollout_len, batch_size, n_iters, action_space, observation_space, gamma=0.99, gae_lambda=0.95, device='cpu', shared_space=None):

        #self.total_rollout_iters = rollout_len // (batch_size * n_iters) * n_iters
        self.total_rollout_iters = math.ceil(rollout_len / batch_size)
        #
        # self.actions = np.zeros((batch_size, self.total_rollout_iters, action_space))
        # self.states = np.zeros((batch_size, self.total_rollout_iters, observation_space))
        # self.logprobs = np.zeros((batch_size, self.total_rollout_iters, 1)) #action_space))
        # self.rewards = np.zeros((batch_size, self.total_rollout_iters,))
        # self.state_values = np.zeros((batch_size, self.total_rollout_iters, 1))
        # self.is_terminals = np.zeros((batch_size, self.total_rollout_iters,))
        #
        # if shared_space is not None:
        #     self.crit_state = np.zeros((batch_size, self.total_rollout_iters, shared_space))
        #     self.crit_space = shared_space
        # else:
        #     self.crit_state = np.zeros((batch_size, self.total_rollout_iters, observation_space))
        #     self.crit_space = observation_space

        self.actions = np.zeros((rollout_len, action_space))
        self.states = np.zeros((rollout_len, observation_space))
        self.logprobs = np.zeros((rollout_len, 1))  # action_space))
        self.rewards = np.zeros((rollout_len,))
        self.state_values = np.zeros((rollout_len, 1))
        self.is_terminals = np.zeros((rollout_len,))

        if shared_space is not None:
            self.crit_state = np.zeros((rollout_len, shared_space))
            self.crit_space = shared_space
        else:
            self.crit_state = np.zeros((rollout_len, observation_space))
            self.crit_space = observation_space

        self.gamma = gamma
        self.lambda_gae = gae_lambda

        self.rollout_len = rollout_len
        self.batch_size = batch_size
        self.action_space = action_space
        self.observation_space = observation_space
        self.step = 0

    def get_random_batched_data_indexes(self, batch_size):
        batch_start = np.arange(0, len(self.states[0]), batch_size)
        indices = np.arange(len(self.states[0]), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]

        return batches

    def add(self, perform_step: bool=True, **kwargs):

        for key, value in kwargs.items():
            if key == 'action':
                self.actions[:, self.step, :] = value
            elif key == 'state':
                self.states[:, self.step, :] = value
            elif key == 'logprob':
                self.logprobs[:, self.step, :] = value
            elif key == 'state_value':
                self.state_values[:, self.step, :] = value
            elif key == 'crit_state':
                self.crit_state[:, self.step, :] = value
            elif key == 'rewards':
                self.rewards[:, self.step] = value
            elif key == 'is_terminals':
                self.is_terminals[:, self.step] = value
            else:
                raise ValueError(f'Unknown key: {key}')
        if perform_step:
            self.step += 1

    def __flatten_rollout(self):

        self.actions = self.actions.reshape(-1, self.action_space)
        self.states = self.states.reshape(-1, self.observation_space)
        self.logprobs = self.logprobs.reshape(-1, 1) #self.action_space)
        self.rewards = self.rewards.reshape(-1,)
        self.state_values = self.state_values.reshape(-1, 1)
        self.is_terminals = self.is_terminals.reshape(-1,)
        self.crit_state = self.crit_state.reshape(-1, self.crit_space)

    def compute_returns_and_advantages(self, bootstrap_value):

        T = self.step
        advantages = np.zeros((self.batch_size, T))
        gae = 0

        # We work backwards from the end of the buffer
        for t in reversed(range(T)):
            if t == T - 1:
                next_v = bootstrap_value.flatten()
            else:
                next_v = self.state_values[:, t + 1].flatten()

            non_terminal = 1.0 - self.is_terminals[:, t]
            delta = self.rewards[:, t]+ self.gamma * next_v * non_terminal - self.state_values[:, t].flatten()
            gae = delta + self.gamma * self.lambda_gae * non_terminal * gae
            advantages[:, t] = gae

        # Now you can flatten for the PPO update
        self.__flatten_rollout()
        advantages = advantages.reshape(-1, 1)
        returns = advantages + self.state_values
        return returns, advantages


    def clear(self):
        self.actions = np.zeros((self.batch_size, self.total_rollout_iters, self.action_space))
        self.states = np.zeros((self.batch_size, self.total_rollout_iters, self.observation_space))
        self.logprobs = np.zeros((self.batch_size, self.total_rollout_iters, 1)) #self.action_space))
        self.rewards = np.zeros((self.batch_size, self.total_rollout_iters,))
        self.state_values = np.zeros((self.batch_size, self.total_rollout_iters, 1))
        self.is_terminals = np.zeros((self.batch_size, self.total_rollout_iters,))
        self.crit_state = np.zeros((self.batch_size, self.total_rollout_iters, self.crit_space))
        self.step = 0