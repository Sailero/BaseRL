import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.modules.base_network import ChkptModule
from agent.modules.feature_model import FeatureModel
from common.utils import get_conv_out_size


# Define the 1D Stochastic Actor network
class StochasticActor(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a fully connected stochastic Actor network.

        Args:
            config: Configuration object containing model parameters.
            network_type: Type of network for checkpoint management.
        """
        super().__init__(config, network_type)

        self.fc_input_dim = config.env.agent_obs_dim[0]
        self.hidden_dim = config.params["actor_hidden_dim"]
        self.output_dim = config.env.agent_action_dim

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.action_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.std_out = nn.Linear(self.hidden_dim, self.output_dim)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.action_out.weight.data.normal_(0, 0.1)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        """
        Forward pass through the actor network.

        Args:
            x: Input observation tensor.

        Returns:
            actions: Action tensor with values between -1 and 1.
            std: Standard deviation tensor for stochasticity in actions.
        """
        x = x.reshape([-1, self.fc_input_dim])  # Flatten input to 2D
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.action_out(x))
        std = F.softplus(self.std_out(x)) + 1e-3  # Add small constant to avoid zero std
        return actions, std


# Define the 1D Stochastic Critic network
class StochasticCritic(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a fully connected stochastic Critic network.

        Args:
            config: Configuration object containing model parameters.
            network_type: Type of network for checkpoint management.
        """
        super(StochasticCritic, self).__init__(config, network_type)

        self.fc_input_dim = config.env.agent_obs_dim[0]
        self.hidden_dim = config.params["critic_hidden_dim"]

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_out = nn.Linear(self.hidden_dim, 1)  # Output is Q-value

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.q_out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        """
        Forward pass through the critic network.

        Args:
            state: Observation tensor.

        Returns:
            q_value: Q-value tensor.
        """
        x = state.reshape([-1, self.fc_input_dim])  # Flatten input to 2D
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value


# Define the 2D Stochastic Actor network with CNN layers
class StochasticActor2d(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a 2D Actor network with CNN layers for stochastic actions.

        Args:
            config: Configuration object containing model parameters.
            network_type: Type of network for checkpoint management.
        """
        super().__init__(config, network_type)

        self.cnn = FeatureModel()
        conv_out_size = get_conv_out_size(config.env.agent_obs_dim, self.cnn)

        # Fully connected layers
        hidden_dim = config.params["actor_hidden_dim"]
        self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.mu = nn.Linear(int(hidden_dim / 2), config.env.agent_action_dim)
        self.std = nn.Linear(int(hidden_dim / 2), config.env.agent_action_dim)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.mu.weight.data.normal_(0, 0.1)
        self.std.weight.data.normal_(0, 0.1)

    def forward(self, state):
        """
        Forward pass through the 2D actor network.

        Args:
            state: Input state tensor with shape [batch_size, channels, height, width].

        Returns:
            mu: Mean action tensor.
            std: Standard deviation tensor for stochasticity.
        """
        state = state.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel input for CNN
        x = self.cnn(state)
        x = x.view(x.size(0), -1)  # Flatten after CNN layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu(x))
        std = F.softplus(self.std(x))
        return mu, std


# Define the 2D Stochastic Critic network with CNN layers
class StochasticCritic2d(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a 2D Critic network with CNN layers for Q-value estimation.

        Args:
            config: Configuration object containing model parameters.
            network_type: Type of network for checkpoint management.
        """
        super().__init__(config, network_type)

        self.cnn = FeatureModel()
        conv_out_size = get_conv_out_size(config.env.agent_obs_dim, self.cnn)

        # Fully connected layers
        hidden_dim = config.params["critic_hidden_dim"]
        self.fc1 = nn.Linear(conv_out_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.q_out = nn.Linear(int(hidden_dim / 2), 1)  # Output is Q-value

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.q_out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        """
        Forward pass through the 2D critic network.

        Args:
            state: Input state tensor with shape [batch_size, channels, height, width].

        Returns:
            q_value: Q-value tensor.
        """
        state = state.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel input for CNN
        x = self.cnn(state)
        x = x.view(x.size(0), -1)  # Flatten after CNN layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q_value = self.q_out(x)
        return q_value
