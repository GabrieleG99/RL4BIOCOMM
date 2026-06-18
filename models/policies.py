from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Union, Dict

from stable_baselines3.common.utils import get_device

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import gymnasium.spaces as spaces

from .layers import FeedForwardNN, BaseNN, RecurrentNN
from dists.distributions import MultiCategoricalDistribution, DictDistribution
from .popart import PopArt

SelfBasePolicy = TypeVar('SelfBasePolicy', bound='BasePolicy')


class BasePolicy(nn.Module, ABC):

    BASE_LR = 1e-3

    def __init__(
            self,
            action_space: spaces.Space,
            observation_space: spaces.Space,
            optim_class: type[optim.Optimizer]=optim.Adam,
            optim_kwargs: Optional[dict[str, Any]]=None,
            features_extractor_class: BaseNN=FeedForwardNN,
            features_extractor_kwargs: Optional[dict[str, Any]]= None,
            activation: str='relu',
    ):
        super(BasePolicy, self).__init__()

        self.action_space = action_space
        self.observation_space = observation_space

        self._set_input_output_sizes()

        self.features_extractor_class = features_extractor_class
        if features_extractor_kwargs is None:
            self.features_extractor_kwargs = {}
        else:
            self.features_extractor_kwargs = features_extractor_kwargs

        self.optim_class = optim_class
        if optim_kwargs is None:
            self.optim_kwargs = {}
        else:
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
            optim_kwargs: Optional[dict[str, Any]]=None,
            features_extractor_class: BaseNN=FeedForwardNN,
            features_extractor_kwargs: Optional[dict[str, Any]]=None,
            shared_critic_input_size: Optional[int]=None,
            net_arch: Optional[dict[str, Any]]=None,
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

        if isinstance(self.act_features_extractor, RecurrentNN):
            self.is_recurrent = True
            self.actor_hidden = None
            self.critic_hidden = None
        else:
            self.is_recurrent = False

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
        self.actor_head = FeedForwardNN(self.act_features_extractor.output_size, self.output_size, self.net_arch['pi'], activation=self.activation)

    def _create_critic_head(self):
        """Create the critic head."""
        if self.use_popart:
            self.critic_head = PopArt(self.critic_features_extractor.output_size, 1, device=self.device)
        else:
            self.critic_head = FeedForwardNN(self.critic_features_extractor.output_size, 1, self.net_arch['vf'], activation=self.activation)

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

    def _zero_recurrent_hidden(self, extractor: RecurrentNN, batch_size: int) -> torch.Tensor:
        n_layers, hidden_size = extractor.hidden_state_shape
        return torch.zeros(n_layers, batch_size, hidden_size, device=self.device)

    def _get_recurrent_hidden(self, name: str, extractor: RecurrentNN, batch_size: int) -> torch.Tensor:
        hidden = getattr(self, name)
        if hidden is None or hidden.shape[1] != batch_size:
            return self._zero_recurrent_hidden(extractor, batch_size)
        return hidden.detach().clone()

    def get_actor_hidden(self, batch_size: int) -> Optional[torch.Tensor]:
        if not self.is_recurrent:
            return None
        return self._get_recurrent_hidden("actor_hidden", self.act_features_extractor, batch_size)

    def get_critic_hidden(self, batch_size: int) -> Optional[torch.Tensor]:
        if not self.is_recurrent:
            return None
        return self._get_recurrent_hidden("critic_hidden", self.critic_features_extractor, batch_size)

    def reset_recurrent_states(self, done_mask: Optional[np.ndarray | torch.Tensor] = None) -> None:
        if not self.is_recurrent:
            return

        if done_mask is None:
            self.actor_hidden = None
            self.critic_hidden = None
            return

        done_mask = torch.as_tensor(done_mask, dtype=torch.bool, device=self.device).reshape(-1)
        for name in ("actor_hidden", "critic_hidden"):
            hidden = getattr(self, name)
            if hidden is not None:
                if done_mask.numel() != hidden.shape[1]:
                    raise ValueError(
                        f"done_mask has {done_mask.numel()} entries, expected {hidden.shape[1]}"
                    )
                hidden[:, done_mask, :] = 0.0

    def _recurrent_features(
            self,
            extractor: RecurrentNN,
            observation: torch.Tensor,
            hidden_state: Optional[torch.Tensor],
            state_name: str,
            update_hidden: bool,
            sequence: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence:
            if observation.dim() != 3:
                raise ValueError("Recurrent sequence observations must have shape (batch, sequence, features)")
            recurrent_input = observation
            batch_shape = observation.shape[:1]
        else:
            if observation.dim() != 3:
                raise ValueError("Recurrent observations must have shape (agents, envs, features)")
            n_agents, n_envs, obs_dim = observation.shape
            recurrent_input = observation.reshape(n_agents * n_envs, 1, obs_dim)
            batch_shape = (n_agents, n_envs)

        batch_size = recurrent_input.shape[0]
        if hidden_state is None and update_hidden:
            hidden_state = getattr(self, state_name)
            if hidden_state is not None and hidden_state.shape[1] != batch_size:
                hidden_state = None

        features, next_hidden = extractor(recurrent_input, hidden_state)

        if update_hidden:
            setattr(self, state_name, next_hidden.detach())

        if sequence:
            return features, next_hidden

        return features.reshape(*batch_shape, -1), next_hidden

    def _actor_features(
            self,
            observation: torch.Tensor,
            hidden_state: Optional[torch.Tensor] = None,
            update_hidden: bool = False,
            sequence: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.is_recurrent:
            return self._recurrent_features(
                self.act_features_extractor,
                observation,
                hidden_state,
                "actor_hidden",
                update_hidden,
                sequence,
            )
        return self.act_features_extractor(observation), None

    def _critic_features(
            self,
            observation: torch.Tensor,
            hidden_state: Optional[torch.Tensor] = None,
            update_hidden: bool = False,
            sequence: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.is_recurrent:
            return self._recurrent_features(
                self.critic_features_extractor,
                observation,
                hidden_state,
                "critic_hidden",
                update_hidden,
                sequence,
            )
        return self.critic_features_extractor(observation), None


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
        features, _ = self._actor_features(observation, update_hidden=True)
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
        features, _ = self._actor_features(observation, update_hidden=True)
        logits = self.actor_head(features)
        dist = self._get_prob_dis_from_act_space(logits)

        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        return actions, logprobs
    
    def evaluate_critic(
            self,
            observation: torch.Tensor,
            hidden_state: Optional[torch.Tensor] = None,
            sequence: bool = False,
    ) -> torch.Tensor:

        """
        Get the state value for the given observation.
        :param observation: The observation tensor.
        :return: The state value tensor.
        """
        features, _ = self._critic_features(observation, hidden_state, sequence=sequence)
        return self.critic_head(features)
    
    def get_state_value(
            self,
            observation: torch.Tensor,
            hidden_state: Optional[torch.Tensor] = None,
            sequence: bool = False,
            update_hidden: bool = True,
    ) -> torch.Tensor:

        """
        Get the state value for the given observation.
        :param observation: The observation tensor.
        :return: The state value tensor.
        """
        features, _ = self._critic_features(
            observation,
            hidden_state,
            update_hidden=update_hidden and hidden_state is None,
            sequence=sequence,
        )

        if self.use_popart:
            values = self.critic_head(features)
            return self.critic_head.denormalize(values)
        else:
            return self.critic_head(features)

    
    def evaluate(
            self,
            observation: torch.Tensor,
            action: torch.Tensor,
            hidden_state: Optional[torch.Tensor] = None,
            sequence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the policy for the given observation and action.
        :param observation: The observation tensor.
        :param action: The action tensor.
        :return: A tuple containing the log probabilities and entropies.
        """
        features, _ = self._actor_features(observation, hidden_state, sequence=sequence)
        logits = self.actor_head(features)

        dist = self._get_prob_dis_from_act_space(logits)

        if isinstance(self.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            action = action.long()

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
                size,
                self.net_arch['pi'],
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

        


        

