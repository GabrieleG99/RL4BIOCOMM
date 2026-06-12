import math

import numpy as np

class RolloutBuffer:
    def __init__(self, rollout_len, n_envs, action_space, observation_space, gamma=0.99, gae_lambda=0.95, device='cpu', shared_space=None):

        #self.total_rollout_iters = rollout_len // (batch_size * n_iters) * n_iters
        self.total_rollout_iters = math.ceil(rollout_len / n_envs)

        self.actions = np.zeros((n_envs, self.total_rollout_iters, action_space), dtype=np.float32)
        self.states = np.zeros((n_envs, self.total_rollout_iters, observation_space), dtype=np.float32)
        self.logprobs = np.zeros((n_envs, self.total_rollout_iters, 1), dtype=np.float32)
        self.rewards = np.zeros((n_envs, self.total_rollout_iters), dtype=np.float32)
        self.state_values = np.zeros((n_envs, self.total_rollout_iters, 1), dtype=np.float32)
        self.is_terminals = np.zeros((n_envs, self.total_rollout_iters), dtype=bool)

        if shared_space is not None:
            self.crit_state = np.zeros((n_envs, self.total_rollout_iters, shared_space), dtype=np.float32)
            self.crit_space = shared_space
        else:
            self.crit_state = np.zeros((n_envs, self.total_rollout_iters, observation_space), dtype=np.float32)
            self.crit_space = observation_space

        self.gamma = gamma
        self.lambda_gae = gae_lambda

        self.rollout_len = rollout_len
        self.n_envs = n_envs
        self.batch_size = n_envs
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
        if self.step >= self.total_rollout_iters:
            raise IndexError("RolloutBuffer is full; call clear() before adding more transitions")

        for key, value in kwargs.items():
            if key == 'action':
                self.actions[:, self.step, :] = self._coerce_value(value, self.actions[:, self.step, :])
            elif key == 'state':
                self.states[:, self.step, :] = self._coerce_value(value, self.states[:, self.step, :])
            elif key == 'logprob':
                self.logprobs[:, self.step, :] = self._coerce_value(value, self.logprobs[:, self.step, :])
            elif key == 'state_value':
                self.state_values[:, self.step, :] = self._coerce_value(value, self.state_values[:, self.step, :])
            elif key == 'crit_state':
                self.crit_state[:, self.step, :] = self._coerce_value(value, self.crit_state[:, self.step, :])
            elif key == 'rewards':
                self.rewards[:, self.step] = self._coerce_value(value, self.rewards[:, self.step])
            elif key == 'is_terminals':
                self.is_terminals[:, self.step] = self._coerce_value(value, self.is_terminals[:, self.step])
            else:
                raise ValueError(f'Unknown key: {key}')
        if perform_step:
            self.step += 1

    def _coerce_value(self, value, target):
        value = np.asarray(value, dtype=target.dtype)

        if value.shape == target.shape:
            return value

        if value.ndim == 0:
            return np.full(target.shape, value, dtype=target.dtype)

        if target.ndim > 1 and value.shape == target.shape[1:]:
            value = value[np.newaxis, ...]

        if target.ndim > 1 and target.shape[-1] == 1 and value.shape == target.shape[:-1]:
            value = value[..., np.newaxis]

        try:
            return np.broadcast_to(value, target.shape).astype(target.dtype, copy=False)
        except ValueError as exc:
            raise ValueError(f"Cannot store value with shape {value.shape} in buffer slot {target.shape}") from exc

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
                # This is the crucial 'bootstrapping' step
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
        self.actions = np.zeros((self.n_envs, self.total_rollout_iters, self.action_space), dtype=np.float32)
        self.states = np.zeros((self.n_envs, self.total_rollout_iters, self.observation_space), dtype=np.float32)
        self.logprobs = np.zeros((self.n_envs, self.total_rollout_iters, 1), dtype=np.float32)
        self.rewards = np.zeros((self.n_envs, self.total_rollout_iters), dtype=np.float32)
        self.state_values = np.zeros((self.n_envs, self.total_rollout_iters, 1), dtype=np.float32)
        self.is_terminals = np.zeros((self.n_envs, self.total_rollout_iters), dtype=bool)
        self.crit_state = np.zeros((self.n_envs, self.total_rollout_iters, self.crit_space), dtype=np.float32)
        self.step = 0
