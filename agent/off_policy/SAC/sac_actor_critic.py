import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.modules.base_network import ChkptModule
from agent.modules.feature_model import FeatureModel
from common.utils import get_conv_out_size


# Define the 1D Stochastic Actor network for SAC
class StochasticActor(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a fully connected stochastic Actor network for SAC.

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
            action_tanh: Action tensor with values between -1 and 1.
            log_prob: Log probability of the action.
        """
        x = x.reshape([-1, self.fc_input_dim])  # Flatten input to 2D
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.action_out(x))
        std = F.softplus(self.std_out(x)) + 1e-3  # Add small constant to avoid zero std

        # Create normal distribution and sample with reparameterization trick
        dist = torch.distributions.Normal(mu, std)
        normal_sample = dist.rsample()
        action_tanh = torch.tanh(normal_sample)

        # Calculate log probability with tanh transformation
        log_prob = dist.log_prob(normal_sample) - (1.000001 - action_tanh.pow(2)).log()
        return action_tanh, log_prob.sum(-1)


# Define the 1D Stochastic Critic network for SAC
class StochasticCritic(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a fully connected stochastic Critic network for SAC.

        Args:
            config: Configuration object containing model parameters.
            network_type: Type of network for checkpoint management.
        """
        super(StochasticCritic, self).__init__(config, network_type)

        self.fc_input_dim = config.env.agent_obs_dim[0] + config.env.agent_action_dim
        self.hidden_dim = config.params["critic_hidden_dim"]

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_out = nn.Linear(self.hidden_dim, 1)  # Output is Q-value

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.q_out.weight.data.normal_(0, 0.1)

    def forward(self, state, action):
        """
        Forward pass through the critic network.

        Args:
            state: Observation tensor.
            action: Action tensor.

        Returns:
            q_value: Q-value tensor.
        """
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value


# Define the 2D Stochastic Actor network with CNN layers for SAC
class StochasticActor2d(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a 2D Actor network with CNN layers for SAC.

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
            action_tanh: Action tensor with values between -1 and 1.
            log_prob: Log probability of the action.
        """
        state = state.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel input for CNN
        x = self.cnn(state)
        x = x.view(x.size(0), -1)  # Flatten after CNN layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        std = F.softplus(self.std(x))

        dist = torch.distributions.Normal(mu, std)
        normal_sample = dist.rsample()
        action_tanh = torch.tanh(normal_sample)

        log_prob = dist.log_prob(normal_sample) - (1.000001 - action_tanh.pow(2)).log()
        return action_tanh, log_prob.sum(-1)


# Define the 2D Stochastic Critic network with CNN layers for SAC
class StochasticCritic2d(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a 2D Critic network with CNN layers for SAC.

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
        self.q_out = nn.Linear(int(hidden_dim / 2), 1)  # Output is Q-value

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.q_out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        """
        Forward pass through the 2D critic network.

        Args:
            s: Input state tensor with shape [batch_size, channels, height, width].
            a: Action tensor.

        Returns:
            q_value: Q-value tensor.
        """
        s = s.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel input for CNN
        x = self.cnn(s)
        x = x.view(x.size(0), -1)  # Flatten after CNN layers
        x = torch.cat([x, a], dim=1)  # Concatenate with action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value
