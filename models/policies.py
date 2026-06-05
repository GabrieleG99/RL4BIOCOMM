from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Union, Dict

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import gymnasium.spaces as spaces

from .layers import FeedForwardNN, BaseNN
from dists.distributions import MultiCategoricalDistribution, DictDistribution
from .popart import PopArt

SelfBasePolicy = TypeVar('SelfBasePolicy', bound='BasePolicy')


def get_device(device: Union[torch.device, str] = 'auto') -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


class BasePolicy(nn.Module, ABC):

    BASE_LR = 1e-3

    def __init__(
            self,
            action_space: spaces.Space,
            observation_space: spaces.Space,
            optim_class: type[optim.Optimizer]=optim.Adam,
            optim_kwargs: dict[str, Any]=None,
            features_extractor_class: nn.Module=FeedForwardNN,
            features_extractor_kwargs: dict[str, Any]= {},
            activation: str='relu',
    ):
        super(BasePolicy, self).__init__()

        self.action_space = action_space
        self.observation_space = observation_space

        self._set_input_output_sizes()

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs

        self.activation = activation

    def _set_input_output_sizes(self):

        if isinstance(self.action_space, spaces.Discrete):
            self.output_size = self.action_space.n
        elif isinstance(self.action_space, spaces.Box):
            self.output_size = self.action_space.shape[0]
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.output_size = np.sum(self.action_space.nvec)
        else:
            raise NotImplementedError(f"Action space {self.action_space} is not supported.")

        if isinstance(self.observation_space, spaces.Box):
            self.input_size = self.observation_space.shape[0]
        elif isinstance(self.observation_space, spaces.MultiDiscrete):
            self.input_size = len(self.observation_space.nvec)
        else:
            raise NotImplementedError(f"Observation space {self.observation_space} is not supported.")

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity=self.activation)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def make_features_extractor(self) -> nn.Module:
        """
        Create the feature extractor module.
        :return: The feature extractor module.
        """
        return self.features_extractor_class(input_size=self.input_size, activation=self.activation, **self.features_extractor_kwargs)
    
    def _get_prob_dis_from_act_space(self, logits):
        """
        Create the probability distribution module based on the action space.
        :return: The probability distribution module.
        """
        if isinstance(self.action_space, spaces.Discrete):
            return torch.distributions.Categorical(logits=logits)
        elif isinstance(self.action_space, spaces.Box):
            return torch.distributions.Normal(loc=logits, scale=torch.ones_like(logits))
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            return MultiCategoricalDistribution(logits, self.action_space)
        else:
            raise NotImplementedError(f"Action space {self.action_space} is not supported.")

    @property
    def device(self) -> torch.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")
    
    def _get_constructor_parameters(self) -> dict[str, Any]:
        return {
            'action_space': self.action_space,
            'observation_space': self.observation_space,
        }
    
    def save(self, path: str):
        """
        Save the policy to a file.
        :param path: The path to save the policy.
        """
        torch.save({'state_dict': self.state_dict(), 'data': self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path: str, device: Union[torch.device, str] = 'auto') -> SelfBasePolicy:

        device = get_device(device)

        saved_variables = torch.load(path, map_location=device, weights_only=False)

        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model
    
    @abstractmethod
    def _predict(
            self,
            observation: torch.Tensor,
            deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Predict the action and value for the given observation.
        :param observation: The observation tensor.
        :param deterministic: Whether to use a deterministic policy.
        :return: A tuple containing the action, value, and features (if requested).
        """
        pass

    def set_training_mode(self, mode: bool = True):

        """
        Set the training mode for the policy.
        :param mode: Whether to set the policy in training mode.
        """
        self.train(mode)

    def predict(
            self,
            observation: np.ndarray,
            deterministic: bool = False,
    ) -> np.ndarray:
        """
        Predict the action and value for the given observation.
        :param observation: The observation tensor.
        :param deterministic: Whether to use a deterministic policy.
        :return: A tuple containing the action, value, and features (if requested).
        """
        self.set_training_mode(False)

        obs_tensor = torch.from_numpy(observation).float().to(self.device)

        with torch.no_grad():
            action= self._predict(obs_tensor, deterministic)

        return action.cpu().numpy()

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""
        del progress_remaining
        return 0.0
        

class ActorCriticPolicy(BasePolicy):

    """Actor-Critic implementation following (sort of) the Stable Baselines3 API."""

    def __init__(
            self,
            action_space: spaces.Space,
            observation_space: spaces.Space,
            optim_class: type[optim.Optimizer]=optim.Adam,
            optim_kwargs: dict[str, Any]=None,
            features_extractor_class: BaseNN=None,
            features_extractor_kwargs: dict[str, Any]={},
            shared_critic_input_size: Optional[int]=None,
            net_arch: dict[str, Any]=None,
            activation: str='relu',
            use_popart: bool=False,
            device='cpu'
    ):

        if optim_kwargs is None:
            optim_kwargs = {}

        super(ActorCriticPolicy, self).__init__(
            action_space, 
            observation_space, 
            optim_class, 
            optim_kwargs,
            features_extractor_class, 
            features_extractor_kwargs, 
            activation
        )

        for elem in ['output_size', 'hidden_size']:
            if elem not in features_extractor_kwargs:
                if elem == 'output_size':
                    features_extractor_kwargs[elem] = 4
                elif elem == 'hidden_size':
                    features_extractor_kwargs[elem] = []
        
        self.use_popart = use_popart

        self.features_extractor_kwargs = features_extractor_kwargs

        if net_arch is None:
            net_arch = {'pi': [8, 8], 'vf': [8, 8]}

        self.net_arch = net_arch

        if features_extractor_class is None:
            self.features_extractor_class = FeedForwardNN
        else:
            self.features_extractor_class = features_extractor_class

        self.act_features_extractor = self.make_features_extractor()

        if shared_critic_input_size is not None:
            self.input_size = shared_critic_input_size

        self.critic_features_extractor = self.make_features_extractor()

        self._create_actor_head()

        self._create_critic_head()

        self.apply(self.init_weights)

        self.shared_optim = True

        if all(elem in optim_kwargs.keys() for elem in ['critic', 'actor']):
            self.actor_optim_kwargs = optim_kwargs['actor']
            self.critic_optim_kwargs = optim_kwargs['critic']
            self.shared_optim = False

        self._build_optimizer()

        self.to(device)

    def _create_actor_head(self):
        """Create the actor head."""
        self.actor_head = FeedForwardNN(self.act_features_extractor.output_size, self.net_arch['pi'], self.output_size, activation=self.activation)

    def _create_critic_head(self):
        """Create the critic head."""
        if self.use_popart:
            self.critic_head = PopArt(self.critic_features_extractor.output_size, 1, device=self.device)
        else:
            self.critic_head = FeedForwardNN(self.critic_features_extractor.output_size, self.net_arch['vf'], 1, activation=self.activation)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """
        Get the constructor parameters for the policy.
        :return: A dictionary containing the constructor parameters.
        """
        params = super()._get_constructor_parameters()
        params.update({
            'optim_class': self.optim_class,
            'optim_kwargs': self.optim_kwargs,
            'features_extractor_class': self.features_extractor_class,
            'features_extractor_kwargs': self.features_extractor_kwargs,
            'shared_critic_input_size': self.critic_features_extractor.input_size,
            'net_arch': self.net_arch,
            'activation': self.activation,
        })
        return params

    def _build_optimizer(self) -> None:

        if self.shared_optim:
            self.optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        else:
            self.actor_optimizer = self.optim_class([{'params': self.actor_head.parameters()},
                                                     {'params': self.act_features_extractor.parameters()}], **self.actor_optim_kwargs)
            self.critic_optimizer = self.optim_class([{'params': self.critic_head.parameters()},
                                                      {'params': self.critic_features_extractor.parameters()}], **self.critic_optim_kwargs)


    def _predict(
            self,
            observation: torch.Tensor,
            deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Predict the action and value for the given observation.
        :param observation: The observation tensor.
        :param deterministic: Whether to use a deterministic policy.
        :return: A tuple containing the action, value, and features (if requested).
        """
        features = self.act_features_extractor(observation)
        logits = self.actor_head(features)

        dist  = self._get_prob_dis_from_act_space(logits)

        if deterministic:
            action = dist.mode
        else:
            action = dist.sample()

        return action
    
    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy.
        :param observation: The observation tensor.
        :return: A tuple containing the action, value, and features (if requested).
        """
        features = self.act_features_extractor(observation)
        logits = self.actor_head(features)
        dist = self._get_prob_dis_from_act_space(logits)

        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        return actions, logprobs
    
    def evaluate_critic(self, observation: torch.Tensor) -> torch.Tensor:

        """
        Get the state value for the given observation.
        :param observation: The observation tensor.
        :return: The state value tensor.
        """
        features = self.critic_features_extractor(observation)
        return self.critic_head(features)
    
    def get_state_value(self, observation: torch.Tensor) -> torch.Tensor:

        """
        Get the state value for the given observation.
        :param observation: The observation tensor.
        :return: The state value tensor.
        """            
        features = self.critic_features_extractor(observation)

        if self.use_popart:
            values = self.critic_head(features)
            return self.critic_head.denormalize(values)
        else:
            return self.critic_head(features)

    
    def evaluate(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the policy for the given observation and action.
        :param observation: The observation tensor.
        :param action: The action tensor.
        :return: A tuple containing the log probabilities and entropies.
        """
        features = self.act_features_extractor(observation)
        logits = self.actor_head(features)

        dist = self._get_prob_dis_from_act_space(logits)

        log_probs = dist.log_prob(action)
        entropies = dist.entropy()

        return log_probs, entropies


class DictActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic Policy extension for Dict action spaces.
    Extends the base ActorCriticPolicy to handle Dict action spaces with separate heads.
    """

    def __init__(
            self,
            action_space: spaces.Dict,
            observation_space: spaces.Space,
            optim_class: type[torch.optim.Optimizer] = torch.optim.Adam,
            optim_kwargs: Dict[str, Any] = None,
            features_extractor_class: BaseNN = None,
            features_extractor_kwargs: Dict[str, Any] = {},
            shared_critic_input_size: Optional[int] = None,
            net_arch: Dict[str, Any] = None,
            activation: str = 'relu',
            device: str = 'cpu'
    ):
        if not isinstance(action_space, spaces.Dict):
            raise ValueError("DictActorCriticPolicy requires a Dict action space")

        # Initialize parent with a dummy output_size (will be overridden)
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            shared_critic_input_size=shared_critic_input_size,
            net_arch=net_arch,
            activation=activation,
            device=device
        )

    def _set_input_output_sizes(self):
        """Override to handle Dict action spaces"""
        # Handle observation space (same as parent)
        if isinstance(self.observation_space, spaces.Box):
            self.input_size = self.observation_space.shape[0]
        elif isinstance(self.observation_space, spaces.MultiDiscrete):
            self.input_size = len(self.observation_space.nvec)
        else:
            raise NotImplementedError(f"Observation space {self.observation_space} is not supported.")

        # Handle Dict action space
        if isinstance(self.action_space, spaces.Dict):
            self.output_size = {}
            for key, space in self.action_space.spaces.items():
                if isinstance(space, spaces.Discrete):
                    self.output_size[key] = space.n
                elif isinstance(space, spaces.Box):
                    self.output_size[key] = space.shape[0]
                elif isinstance(space, spaces.MultiDiscrete):
                    self.output_size[key] = np.sum(space.nvec)
                else:
                    raise NotImplementedError(f"Action subspace {space} is not supported in Dict.")
        else:
            raise ValueError("DictActorCriticPolicy requires a Dict action space")

    def _create_actor_head(self):
        """Create separate actor heads for each action component"""

        self.actor_heads = nn.ModuleDict()
        for key, size in self.output_size.items():
            self.actor_heads[key] = FeedForwardNN(
                self.act_features_extractor.output_size,
                self.net_arch['pi'],
                size,
                activation=self.activation
            )

    def _build_optimizer(self) -> None:
        """Override to handle multiple actor heads"""
        # Ensure actor heads are created
        if not hasattr(self, 'actor_heads'):
            self._create_actor_head()
            self.apply(self.init_weights)

        if self.shared_optim:
            self.optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        else:
            # Collect parameters from all actor heads
            actor_params = [{'params': self.act_features_extractor.parameters()}]
            for head in self.actor_heads.values():
                actor_params.append({'params': head.parameters()})

            self.actor_optimizer = self.optim_class(actor_params, **self.actor_optim_kwargs)
            self.critic_optimizer = self.optim_class(
                [{'params': self.critic_head.parameters()},
                 {'params': self.critic_features_extractor.parameters()}],
                **self.critic_optim_kwargs
            )

    def _get_logits(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get logits from all actor heads"""
        if not hasattr(self, 'actor_heads'):
            self._create_actor_head()
            self.apply(self.init_weights)

        features = self.act_features_extractor(observation)
        logits = {}
        for key, head in self.actor_heads.items():
            logits[key] = head(features)
        return logits

    def _get_distributions(self, logits: Dict[str, torch.Tensor]) -> Dict[str, torch.distributions.Distribution]:
        """Create distributions for each action component"""
        distributions = {}
        for key, logits_tensor in logits.items():
            space = self.action_space.spaces[key]

            if isinstance(space, spaces.Discrete):
                distributions[key] = torch.distributions.Categorical(logits=logits_tensor)
            elif isinstance(space, spaces.Box):
                distributions[key] = torch.distributions.Normal(
                    loc=logits_tensor,
                    scale=torch.ones_like(logits_tensor)
                )
            elif isinstance(space, spaces.MultiDiscrete):
                distributions[key] = MultiCategoricalDistribution(logits_tensor, space)
            else:
                raise NotImplementedError(f"Action space {space} is not supported.")

        return distributions

    def _predict(
            self,
            observation: torch.Tensor,
            deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Predict actions for Dict action space"""
        logits = self._get_logits(observation)
        distributions = self._get_distributions(logits)
        dist_wrapper = DictDistribution(distributions)

        if deterministic:
            return dist_wrapper.mode
        else:
            return dist_wrapper.sample()

    def forward(self, observation: torch.Tensor) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass returning dict of actions and combined log probabilities"""
        logits = self._get_logits(observation)
        distributions = self._get_distributions(logits)
        dist_wrapper = DictDistribution(distributions)

        actions = dist_wrapper.sample()
        logprobs = dist_wrapper.log_prob(actions)

        return actions, logprobs

    def evaluate(
            self,
            observation: torch.Tensor,
            action: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate policy for dict actions"""
        logits = self._get_logits(observation)
        distributions = self._get_distributions(logits)
        dist_wrapper = DictDistribution(distributions)

        log_probs = dist_wrapper.log_prob(action)
        entropies = dist_wrapper.entropy()

        return log_probs, entropies

    def predict(
            self,
            observation: np.ndarray,
            deterministic: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Predict and return dict of numpy arrays"""
        self.set_training_mode(False)

        obs_tensor = torch.from_numpy(observation).float().to(self.device)

        with torch.no_grad():
            actions = self._predict(obs_tensor, deterministic)

        # Convert dict of tensors to dict of numpy arrays
        return {key: val.cpu().numpy() for key, val in actions.items()}

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """Get constructor parameters including Dict-specific settings"""
        params = super()._get_constructor_parameters()
        return params

        


        


