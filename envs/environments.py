import gymnasium as gym
from matplotlib import pyplot as plt

from typing import Dict, Optional, Sequence, Tuple, Union
import scipy

import torch

import numpy as np

from models.Agent import Agent
from models.layers import BaseNN

IDS = {
    'agent': 'A',
    'empty': 'E',
    'speaker': 'S',
    'listener': 'L'
}

class Environment:
    def __init__(self,
                 agents: Sequence[Agent],
                 policy_action_space: Dict[str, gym.Space],
                 policy_obs_space: Dict[str, gym.Space],
                 space_shape: Tuple[int, int]=(5, 5),
                 n_iters: int=3,
                 mol_types: int=1,
                 noisy: bool =True,
                 shared_obs: bool=False,
                 sr_choice: str="furthest",
                 is_continuous: bool=False,
                 encoder: Optional[BaseNN]=None,
                 decoder: Optional[BaseNN]=None,
                 X_train: Optional[np.ndarray]=None,
                 y_train: Optional[np.ndarray]=None,
                 device: str='cpu',
                 max_step_count: int=30,
                 seed: int=42):
        """
        Initialize the multi-agent communication environment.
        
        Args:
            agents: List of agent objects
            policy_action_space: Dictionary mapping policy types to their action spaces
            policy_obs_space: Dictionary mapping policy types to their observation spaces
            space_shape: Tuple defining the grid dimensions (height, width)
            n_iters: Number of communication iterations per episode
            mol_types: Number of molecule types for communication
            sr_choice: Method for selecting sender/receiver ("furthest" or "random")
            is_continuous: Whether to use continuous or discrete space
            device: Device for tensor operations ('cpu' or 'cuda')
            max_step_count: Maximum number of steps per episode
            seed: Random seed for reproducibility
            encoder: Neural network for encoding input data
            decoder: Neural network for decoding messages
        """

        self.device = device
        self.n_iters = n_iters
        self.n_agents = len(agents)
        self.is_continuous = is_continuous
        self.agents_dis = np.full((self.n_agents, self.n_agents), np.inf, dtype=np.float32)
        self.mol_types = mol_types
        self.max_step_count = max_step_count
        self.step_count = 0
        self.iter_count = 0
        self.noisy = noisy
        self.encoder: BaseNN = encoder
        self.decoder: BaseNN = decoder
        self.X_train = X_train
        self.y_train = y_train
        self.next_obs = None
        self.space_shape = np.array(space_shape, dtype=np.int32)
        self.sender = None
        self.receiver = None
        self.sr_choice = sr_choice
        self.shared_obs = shared_obs

        self.agents = agents

        # Initialize NumPy random state
        self.__seed(seed)

        self.__base_grid = self.__create_grid()
        self.__full_obs = self.__create_grid() if not is_continuous else \
            {IDS['agent'] + str(i + 1): None for i in range(self.n_agents)}

        self.__init_obs()

        self.policy_action_space = policy_action_space
        self.policy_obs_space = policy_obs_space

        self.__create_agent_policy_mapping()

        self.label = None
        self.episode_input = None

        self.deterministic = False

    def reset_rng_state(self):
        """
        Reset the random number generator state to its initial seed value.
        
        This method restores the environment's random state to ensure reproducible
        behavior when needed.
        """
        self.rng = np.random.RandomState(self.seed_value)

    def __create_agent_policy_mapping(self):
        """
        Create mapping between agent IDs and their corresponding policy types.
        
        This method establishes a bidirectional mapping:
        - agent_policy_mapping: maps agent_id -> policy_type
        - policy_agents_mapping: maps policy_type -> list of agent_ids
        
        Policy types are assigned based on agent roles (sender, receiver, or regular agent).
        """
        self.agent_policy_mapping = {}

        for aid in range(self.n_agents):
            agent_key = f'agent_{aid}'

            if aid == self.sender:
                policy_type = 'sender'
            elif aid == self.receiver:
                policy_type = 'receiver'
            else:
                policy_type = 'agent'

            self.agent_policy_mapping[agent_key] = policy_type

        self.policy_agents_mapping = {
            'sender': [],
            'receiver': [],
            'agent': []
        }

        for agent_key, policy_type in self.agent_policy_mapping.items():
            self.policy_agents_mapping[policy_type].append(agent_key)

    def _get_policy_type_for_agent(self, agent_id: int) -> str:
        """
        Get the policy type for a specific agent.
        
        Args:
            agent_id: The ID of the agent
            
        Returns:
            The policy type string ('sender', 'receiver', or 'agent')
        """
        agent_key = f'agent_{agent_id}'
        return self.agent_policy_mapping.get(agent_key, 'agent')

    def get_agents_by_policy_type(self, policy_type: str) -> list:
        """
        Get a list of agent keys that use a specific policy type.
        
        Args:
            policy_type: The policy type to query ('sender', 'receiver', or 'agent')
            
        Returns:
            List of agent keys (e.g., ['agent_0', 'agent_1']) using the specified policy
        """
        return self.policy_agents_mapping.get(policy_type, [])

    def get_action_space_for_agent(self, agent_id: int):
        """
        Get the action space for a specific agent based on its policy type.
        
        Args:
            agent_id: The ID of the agent
            
        Returns:
            The gymnasium Space object defining the agent's action space
        """
        policy_type = self._get_policy_type_for_agent(agent_id)
        return self.policy_action_space.get(policy_type, None)

    def get_obs_space_for_agent(self, agent_id: int):
        """
        Get the observation space for a specific agent based on its policy type.
        
        Args:
            agent_id: The ID of the agent
            
        Returns:
            The gymnasium Space object defining the agent's observation space
        """
        policy_type = self._get_policy_type_for_agent(agent_id)
        return self.policy_obs_space.get(policy_type, None)

    def print_policy_mapping(self):
        """
        Print the current policy mapping for debugging purposes.
        
        This method displays:
        - Current sender and receiver agent IDs
        - Complete agent-to-policy mapping
        - Agents grouped by policy type
        """
        print("=== Agent Policy Mapping ===")
        print(f"Sender: Agent {self.sender}")
        print(f"Receiver: Agent {self.receiver}")
        print(f"\nMapping: {self.agent_policy_mapping}")
        print(f"\nBy policy type:")
        for policy_type, agents in self.policy_agents_mapping.items():
            print(f"  {policy_type}: {agents}")

    def __create_grid(self):
        """
        Create an empty grid for discrete environments.
        
        Returns:
            A 2D list representing an empty grid filled with 'empty' cell identifiers
        """
        return [[IDS['empty'] for _ in range(self.space_shape[1])]
                for _ in range(self.space_shape[0])]

    def __sample_size(self):
        """
        Sample random sizes for all agents.
        
        Returns:
            np.ndarray: Array of shape (n_agents, 2) containing random size values
        """
        return self.rng.rand(self.n_agents, 2).astype(np.float32)

    def __is_cell_vacant(self, pos):
        """
        Check if a grid cell is vacant (empty).
        
        Args:
            pos: Position coordinates (x, y) to check
            
        Returns:
            bool: True if the cell is empty, False otherwise
        """
        return self.__full_obs[pos[0]][pos[1]] == IDS['empty']

    def __sample_pos(self):
        """
        Sample random positions for all agents based on environment type.
        
        For continuous environments, positions are sampled from uniform distribution
        over the space dimensions. For discrete environments, positions are sampled
        from integer grid coordinates.
        
        Returns:
            np.ndarray: Array of shape (n_agents, 2) containing position coordinates
        """
        if self.is_continuous:
            return (self.rng.rand(self.n_agents, 2) * self.space_shape.astype(np.float32)).astype(np.float32)
        else:
            return self.rng.randint(0, self.space_shape[0], size=(self.n_agents, 2)).astype(np.int32)

    def __init_obs(self):
        """
        Initialize agent observations and positions at the start of environment setup.
        
        This method:
        1. Samples initial sizes and positions for all agents
        2. Ensures no position conflicts in discrete environments
        3. Updates agent positions and grid view
        4. Computes initial distances between agents
        5. Sets sender and receiver based on the configured strategy
        """
        init_size = self.__sample_size()
        init_pos = self.__sample_pos()

        if not self.is_continuous:
            for i, pos in enumerate(init_pos):
                while not self.__is_cell_vacant(pos):
                    new_pos = self.rng.randint(0, self.space_shape[0], size=(2,)).astype(np.int32)
                    init_pos[i] = new_pos

                self.__full_obs[pos[0]][pos[1]] = IDS['agent'] + str(i + 1)

        init_obs = np.concatenate([init_pos, init_size], axis=1)
        self.__update_agents_pos(init_obs)
        self.__update_agents_view()
        self.__compute_pos_and_dist()
        self.__set_sender_receiver()

    def __update_agents_pos(self, pos: np.ndarray):
        """
        Update agent positions based on new observation data.

        Args:
            pos: Array of shape (n_agents, 4) where each row contains
                 [x_pos, y_pos, width, height] for an agent
        """

        for i, ob in enumerate(pos):
            # Convert to NumPy if necessary
            if hasattr(self.agents[i].state, 'position'):
                self.agents[i].state.position = ob[:2].astype(np.float32)
            else:
                # Create attribute if it doesn't exist
                self.agents[i].state.position = ob[:2].astype(np.float32)

            if hasattr(self.agents[i], 'dimension'):
                self.agents[i].dimension = ob[2:].astype(np.float32)

    def __update_agents_view(self):
        """
        Update the grid visualization with current agent positions.

        For discrete environments, updates the grid cells with agent identifiers.
        For continuous environments, updates the position dictionary.
        """
        if not self.is_continuous:
            for i in range(self.n_agents):
                x, y = self.agents[i].state.position.astype(int)
                self.__full_obs[x][y] = IDS['agent'] + str(i + 1)
        else:
            for i in range(self.n_agents):
                self.__full_obs[IDS['agent'] + str(i + 1)] = self.agents[i].state.position.copy()

    def __compute_pos_and_dist(self):
        """
        Compute pairwise distances between all agents using vectorized operations.
        
        This method calculates the Euclidean distance matrix between all agent pairs
        and stores it in self.agents_dis for use in communication probability calculations.
        """
        # Collect all positions
        positions = np.array([agent.state.position for agent in self.agents], dtype=np.float32)

        # Vectorized distance calculation
        # Broadcasting: (n_agents, 1, 2) - (1, n_agents, 2) -> (n_agents, n_agents, 2)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        self.agents_dis = distances

    def plot_graph(self, ax=None):
        """
        Plot the agent graph showing positions and communication links.
        
        Args:
            ax: Matplotlib axis object. If None, creates a new subplot
            
        This method visualizes:
        - All agent positions as nodes
        - Sender agent in green
        - Receiver agent in red
        - Agent IDs as text labels
        """
        if ax is None:
            _, ax = plt.subplots()

        if self.is_continuous:
            graph = np.array(list(self.__full_obs.values()))
        else:
            # For discrete grid, find agent positions
            agent_positions = []
            for i in range(self.n_agents):
                agent_positions.append(self.agents[i].state.position)
            graph = np.array(agent_positions)

        # Plot nodes
        ax.plot(graph[:, 0], graph[:, 1], "o", label="Nodes")
        for i, (x, y) in enumerate(graph):
            ax.text(x, y, f"{i}", ha="center", va="bottom")

        # Plot sender and receiver with different colors
        sender, receiver = self.__get_furthest_agents()
        ax.plot(graph[sender, 0], graph[sender, 1], "o", color="green", label="Sender")
        ax.plot(graph[receiver, 0], graph[receiver, 1], "o", color="red", label="Receiver")

        ax.set_aspect("equal", "box")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Agent Communication Graph")

    def __message_passing(self, messages: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process inter-agent messages with distance-based probabilistic filtering.

        This function performs message passing in a multi-agent system. It excludes self-messages,
        applies a probabilistic scaling based on pairwise distances between agents, and aggregates
        messages for each receiving agent.

        Args:
            messages: A 2D array of shape (n_agents, message_dim) representing the messages sent by
                     each agent.

        Returns:
            A dictionary containing processed and aggregated messages for each receiving
            agent. The array has a shape of (n_agents, batch_size, message_dim).
        """
        
        n_agents = messages.shape[0]

        # Create masks to exclude self-messages for all agents
        # Shape: (n_agents, n_agents, 1, 1)
        agents_idx = np.arange(n_agents)
        msg_select_mask = (agents_idx[:, np.newaxis] != agents_idx[np.newaxis, :])
        msg_select_mask = msg_select_mask.reshape(n_agents, n_agents, 1)

        # Expand messages for all receiving agents
        # Shape: (n_agents, n_agents, batch_size, message_dim)
        expanded_msgs = np.expand_dims(messages, 0)
        expanded_msgs = np.repeat(expanded_msgs, n_agents, axis=0)

        if self.noisy:
            # Calculate reception probabilities for all agents
            # Shape: (n_agents, n_agents, 1)
            dists = self.agents_dis  # Shape: (n_agents, n_agents)
            probs = scipy.special.erfc(dists / 2.0)
            probs = probs[..., np.newaxis]

            # Apply probabilistic scaling to all messages
            # Shape: (n_agents, n_agents, batch_size, message_dim)
            received_msgs = self.rng.binomial(expanded_msgs.astype(int), probs).astype(messages.dtype)
        else:
            received_msgs = expanded_msgs

        # Apply mask to exclude self-messages
        masked_msgs = received_msgs * msg_select_mask

        # Aggregate messages for each receiving agent
        # Shape: (n_agents, batch_size, message_dim)
        agg_msgs = np.sum(masked_msgs, axis=1)

        return agg_msgs

    def get_agent_obs(self):
        """
        Retrieve current observations for all agents in the environment.
        
        If an agent does not have a `curr_obs` attribute in its state, a placeholder array
        of zeros is returned for that agent.

        Returns:
            dict: A dictionary mapping agent identifiers as keys (e.g., "agent_0") to their
                 respective current observations. If `curr_obs` is unavailable for an
                 agent, the value will be a zero-filled NumPy array with a length equal to
                 the number of molecular types.
        """
        obs = {}
        for aid in range(self.n_agents):
            agent_key = f"agent_{aid}"
            if hasattr(self.agents[aid].state, 'curr_obs'):
                obs[agent_key] = self.agents[aid].state.curr_obs
            else:
                obs[agent_key] = np.zeros(self.mol_types, dtype=np.float32)
        return obs

    def reset(self,
              new_input: Optional[Union[np.ndarray, torch.Tensor]]=None,
              new_label: Optional[Union[np.ndarray, torch.Tensor, int]]=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment with new input data and labels.
        
        This method initializes a new episode by:
        1. Resetting step and iteration counters
        2. Setting new input data and labels
        3. Encoding the input through the sender agent
        4. Preparing initial observations for all agents
        
        Args:
            new_input: The new input data for the environment. Can be numpy array or torch tensor.
            new_label: The new target labels for the environment. Can be numpy array or torch tensor.
            
        Returns:
            tuple: A tuple containing:
                - observations: Dictionary with initial observations for each agent
                - infos: Empty dictionary for consistency with step function interface
        """
        self.step_count = 1
        self.iter_count = 0

        if new_input is None or new_label is None:
            if self.X_train is None or self.y_train is None:
                raise ValueError("X_train and y_train must be set when reset() is called without data")
            sample = self.rng.choice(self.X_train.shape[0])
            self.label = self.y_train[sample]
            self.episode_input = self.X_train[sample]
        else:
            self.label = new_label.detach().cpu().numpy() if isinstance(new_label, torch.Tensor) else new_label
            self.episode_input = new_input.detach().cpu().numpy() if isinstance(new_input, torch.Tensor) else new_input

        new_input = self.episode_input

        with torch.no_grad():
            squeeze_sender_obs = False
            if isinstance(new_input, np.ndarray):
                new_input = torch.from_numpy(new_input).float().to(self.device)
            if new_input.dim() == 1:
                new_input = new_input.unsqueeze(0)
                squeeze_sender_obs = True
            sender_obs = self.encoder(new_input).detach().cpu().numpy()
            if squeeze_sender_obs:
                sender_obs = sender_obs.squeeze(0)

        observations = np.zeros((self.n_agents, *sender_obs.shape), dtype=np.float32)
        observations[self.sender] = sender_obs

        infos = {}

        if self.shared_obs:
            shared_obs = np.concatenate(
                [
                    observations,
                    self.agents_dis[np.triu_indices(self.n_agents, k=1)][np.newaxis, :].repeat(self.n_agents, axis=0)
                ],
                 axis=-1
            )
            infos['shared_obs'] = shared_obs

        return observations, infos

    def step(self, actions):
        """
        Execute one step in the environment with the given agent actions.
        
        Args:
            actions: Dictionary or array containing actions for each agent
            
        Returns:
            tuple: Contains observations, rewards, dones, and infos for the next state
        """
        self.step_count += 1
        self.iter_count += 1
        self._validate_step_prerequisites()
        self.__compute_pos_and_dist()
        return self._process_step(actions)

    def _validate_step_prerequisites(self):
        """
        Validate that all required conditions are met before executing a step.
        
        Raises:
            ValueError: If step_label is not set before calling step()
        """
        if self.label is None:
            raise ValueError("step_label must be set before step()")

    def _update_agent_observations(self):
        """
        Update the next observation state by retrieving current agent observations.
        """
        self.next_obs = self.get_agent_obs()

    def _process_step(self, actions: np.ndarray):
        """
        Process a single environment step including message passing and reward calculation.
        
        This method performs one round of communication (message passing) to apply diffusion
        noise to messages. The designated receiver then performs classification of the message,
        and the environment returns appropriate rewards. To account for sparse rewards, the
        environment provides small intermediate rewards for successful classification even
        during intermediate message passing iterations.
        
        Args:
            actions: Array containing actions from all agents with shape (n_agents, batch_size, action_dim)
            
        Returns:
            tuple: Contains:
                - observations: Dictionary with next observations for each agent
                - rewards: Dictionary with rewards for each agent and overall reward
                - dones: Dictionary with termination flags for each agent and overall flag
                - infos: Dictionary with additional information about the step
        """

        observations = self.__message_passing(actions)

        with torch.no_grad():
            action_np = actions[self.receiver]
            if isinstance(action_np, np.ndarray):
                action_tensor = torch.from_numpy(action_np).float()
            else:
                action_tensor = torch.tensor(action_np, dtype=torch.float32)

            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)

            logits = self.decoder(action_tensor).detach()

            if not self.deterministic:
                label_pred = torch.distributions.Categorical(logits=logits).sample()
            else:
                label_pred = torch.distributions.Categorical(logits=logits).mode

            correct = ( label_pred == self.label).detach().float().cpu().numpy()
            reward = correct.copy()
            reward[reward == 0] = -1.0
            #reward = reward * 0.1
            reward = reward.item() if reward.shape[0] == 1 else reward

        done = np.full((self.n_agents,), False)

        infos = {f"agent_{aid}": {} for aid in range(self.n_agents)}
        infos['__all__'] = {'iters_end': self.iter_count >= self.n_iters}

        # Check for end of iterations
        if self.iter_count >= self.n_iters :
            self.iter_count = 0
            reward = reward * 10  # Amplify final reward
            done[:] = True

        # Update rewards and dones for all agents
        rewards = {f"agent_{aid}": reward for aid in range(self.n_agents)}
        rewards["__all__"] = reward
        dones = {f"agent_{aid}": done for aid in range(self.n_agents)}
        dones["__all__"] = done

        infos['__all__']['correct'] = bool(correct.item()) if correct.shape[0] == 1 else correct.astype(bool)
        if self.shared_obs:
            shared_obs = np.concatenate(
                [observations,
                 self.agents_dis[np.triu_indices(self.n_agents, k=1)][np.newaxis, :].repeat(self.n_agents, axis=0)
                ],
                axis=-1
            )
            infos['shared_obs'] = shared_obs

        return observations, rewards, dones, infos

    def __get_furthest_agents(self):
        """
        Find the pair of agents that are furthest apart from each other.
        
        Returns:
            tuple: A tuple containing (sender_id, receiver_id) of the furthest agent pair
        """
        max_idx = np.unravel_index(np.argmax(self.agents_dis), self.agents_dis.shape)
        return int(max_idx[0]), int(max_idx[1])

    def __set_sender_receiver(self):
        """
        Set the sender and receiver agents based on the configured selection strategy.
        
        Supports two strategies:
        - "furthest": Select the pair of agents that are furthest apart
        - "random": Randomly select two different agents
        
        Raises:
            ValueError: If sr_choice is not "furthest" or "random"
        """
        if self.sr_choice == "furthest":
            self.sender, self.receiver = self.__get_furthest_agents()
        elif self.sr_choice == "random":
            indices = self.rng.permutation(self.n_agents)[:2]
            self.sender, self.receiver = int(indices[0]), int(indices[1])
        else:
            raise ValueError(f"Invalid sender/receiver choice. Must be 'furthest' or 'random'. Got {self.sr_choice}.")

    def set_deterministic(self, mode: bool=True):

        self.deterministic = mode

    def render(self):
        """
        Render the current state of the environment.
        
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Rendering functionality is not yet implemented")

    def close(self):
        """
        Clean up and close the environment.
        
        This method performs any necessary cleanup operations when the environment
        is no longer needed.
        """
        pass

    def __seed(self, seed):
        """
        Set the random seed for reproducible environment behavior.
        
        Args:
            seed: Integer seed value for the random number generator
            
        Returns:
            list: List containing the seed value for compatibility with gym interface
        """
        self.seed_value = seed
        self.rng = np.random.RandomState(seed)
        return [seed]
