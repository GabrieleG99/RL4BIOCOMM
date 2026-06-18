from typing import Dict
import torch
from torch.distributions import Categorical, Distribution

from gymnasium.spaces import MultiDiscrete

class MultiCategoricalDistribution(Distribution):

    def __init__(self, logits, action_space: MultiDiscrete):
        super(MultiCategoricalDistribution, self).__init__(validate_args=False)
        self.logits = logits
        if action_space.nvec.sum() != logits.shape[-1]:
            raise ValueError(f"Logits shape {logits.shape[-1]} does not match action space dimensions {action_space.nvec.sum()}.")
        self.action_dims = action_space.nvec
        self.is_uniform = bool((self.action_dims == self.action_dims[0]).all())
        self._categorical_distributions = None
        self.distribution = None

        if self.is_uniform:
            self.n_components = len(self.action_dims)
            self.n_categories = int(self.action_dims[0])
            batched_logits = logits.reshape(*logits.shape[:-1], self.n_components, self.n_categories)
            self.distribution = Categorical(logits=batched_logits)
            return

        self._categorical_distributions = []
        current_scan = 0
        for dim in self.action_dims:
            self._categorical_distributions.append(Categorical(logits=logits[..., current_scan:current_scan+dim]))
            current_scan += dim

    @property
    def categorical_distributions(self):
        if self._categorical_distributions is None:
            self._categorical_distributions = []
            current_scan = 0
            for dim in self.action_dims:
                self._categorical_distributions.append(Categorical(logits=self.logits[..., current_scan:current_scan+dim]))
                current_scan += dim
        return self._categorical_distributions

    def sample(self):
        if self.is_uniform:
            return self.distribution.sample()
        return torch.stack([dist.sample() for dist in self.categorical_distributions], dim=-1)

    def log_prob(self, actions):
        if self.is_uniform:
            return self.distribution.log_prob(actions.long()).sum(-1, keepdim=True)
        return torch.stack([dist.log_prob(actions[..., i]) for i, dist in enumerate(self.categorical_distributions)], dim=-1).sum(-1, keepdim=True)
    
    def entropy(self):
        if self.is_uniform:
            return self.distribution.entropy().sum(-1, keepdim=True)
        return torch.stack([dist.entropy() for dist in self.categorical_distributions], dim=-1).sum(-1, keepdim=True)

    @property
    def mode(self):
        if self.is_uniform:
            return self.distribution.logits.argmax(dim=-1)
        return torch.stack([dist.mode for dist in self.categorical_distributions], dim=-1)


class DictDistribution:
    """Wrapper class to handle dict of distributions as a single distribution-like object"""

    def __init__(self, distributions_dict: Dict[str, torch.distributions.Distribution]):
        self.distributions = distributions_dict

    def sample(self) -> Dict[str, torch.Tensor]:
        """Sample from all distributions"""
        return {key: dist.sample() for key, dist in self.distributions.items()}

    def log_prob(self, actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute log probability, summing across all action components"""
        log_probs = []
        for key, dist in self.distributions.items():
            log_prob = dist.log_prob(actions[key])
            if log_prob.dim() > 1:
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                log_prob = log_prob.unsqueeze(-1)

            log_probs.append(log_prob)
        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Compute entropy, summing across all action components"""
        entropies = []
        for key, dist in self.distributions.items():
            entropy = dist.entropy()
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1, keepdim=True)
            else:
                entropy = entropy.unsqueeze(-1)
            entropies.append(entropy)
        return torch.stack(entropies, dim=-1).sum(dim=-1)

    @property
    def mode(self) -> Dict[str, torch.Tensor]:
        """Get mode from all distributions"""
        return {key: dist.mode for key, dist in self.distributions.items()}
