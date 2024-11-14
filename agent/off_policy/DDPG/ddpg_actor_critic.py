import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.modules.base_network import ChkptModule
from agent.modules.feature_model import FeatureModel
from common.utils import get_conv_out_size


# Define the 1D Actor network for DDPG
class DeterministicActor(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a fully connected Actor network for DDPG.

        Args:
            config: Configuration object containing model parameters.
            network_type: Type of network for checkpoint management.
        """
        super(DeterministicActor, self).__init__(config, network_type)

        # Dimensions for input, hidden layers, and output
        self.fc_input_dim = config.env.agent_obs_dim[0]
        self.hidden_dim = config.params["actor_hidden_dim"]
        self.output_dim = config.env.agent_action_dim

        # Define fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.action_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        """
        Forward pass through the actor network.

        Args:
            x: Input observation tensor.

        Returns:
            actions: Action tensor with values between -1 and 1.
        """
        x = x.reshape([-1, self.fc_input_dim])  # Flatten input to 2D
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.action_out(x))  # Tanh to bound actions between -1 and 1
        return actions


# Define the 1D Critic network for DDPG
class DeterministicCritic(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a fully connected Critic network for DDPG.

        Args:
            config: Configuration object containing model parameters.
            network_type: Type of network for checkpoint management.
        """
        super(DeterministicCritic, self).__init__(config, network_type)

        # Dimensions for input, hidden layers, and output
        self.obs_input_dim = config.env.agent_obs_dim[0]
        self.fc_input_dim = self.obs_input_dim + config.env.agent_action_dim
        self.hidden_dim = config.params["critic_hidden_dim"]

        # Define fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_out = nn.Linear(self.hidden_dim, 1)  # Output is Q-value

    def forward(self, state, action):
        """
        Forward pass through the critic network.

        Args:
            state: Observation tensor.
            action: Action tensor.

        Returns:
            q_value: Q-value tensor.
        """
        state = state.reshape([-1, self.obs_input_dim])  # Flatten input to 2D
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value


# Define the 2D Actor network with CNN layers for DDPG
class DeterministicActor2d(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a 2D Actor network with CNN layers for DDPG.

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
        self.output = nn.Linear(int(hidden_dim / 2), config.env.agent_action_dim)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.output.weight.data.normal_(0, 0.1)

    def forward(self, state):
        """
        Forward pass through the 2D actor network.

        Args:
            state: Input state tensor with shape [batch_size, channels, height, width].

        Returns:
            Action tensor with values between -1 and 1.
        """
        state = state.unsqueeze(1)  # Add channel dimension
        state = state.repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel input for CNN
        x = self.cnn(state)
        x = x.view(x.size(0), -1)  # Flatten after CNN layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.output(x))


# Define the 2D Critic network with CNN layers for DDPG
class DeterministicCritic2d(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a 2D Critic network with CNN layers for DDPG.

        Args:
            config: Configuration object containing model parameters.
            network_type: Type of network for checkpoint management.
        """
        super().__init__(config, network_type)

        self.cnn = FeatureModel()
        conv_out_size = get_conv_out_size(config.env.agent_obs_dim, self.cnn)

        # Fully connected layers
        hidden_dim = config.params["critic_hidden_dim"]
        self.fc1 = nn.Linear(conv_out_size + config.env.agent_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.q_out = nn.Linear(int(hidden_dim / 2), 1)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.q_out.weight.data.normal_(0, 0.1)

    def forward(self, state, action):
        """
        Forward pass through the 2D critic network.

        Args:
            state: Input state tensor with shape [batch_size, channels, height, width].
            action: Action tensor.

        Returns:
            Q-value tensor.
        """
        state = state.unsqueeze(1)  # Add channel dimension
        state = state.repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel input for CNN
        x = self.cnn(state)
        x = x.view(x.size(0), -1)  # Flatten after CNN layers
        x = torch.cat([x, action], dim=1)  # Concatenate with action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_out(x)
