from agent.modules.base_network import ChkptModule
import torch
import torch.nn.functional as F


class Discriminator(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a Discriminator network for 1D observation and action inputs.

        Args:
            config: Configuration object containing model parameters.
            network_type: Type of network for checkpoint management.
        """
        super(Discriminator, self).__init__(config, network_type)

        # Define fully connected layers
        self.fc1 = torch.nn.Linear(
            config.env.agent_obs_dim[0] + config.env.agent_action_dim,
            config.params["discr_hidden_dim"]
        )
        self.fc2 = torch.nn.Linear(config.params["discr_hidden_dim"], 1)

        # Initialize weights with normal distribution
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        """
        Forward pass of the discriminator with state and action inputs.

        Args:
            s: State input tensor.
            a: Action input tensor.

        Returns:
            Probability output from the discriminator.
        """
        # Flatten state and concatenate with action
        x = s.view(s.size(0), -1)
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class Discriminator2d(ChkptModule):
    def __init__(self, config, network_type):
        """
        Initialize a Discriminator network for 2D observation and action inputs.

        Args:
            config: Configuration object containing model parameters.
            network_type: Type of network for checkpoint management.
        """
        super(Discriminator2d, self).__init__(config, network_type)

        # Load feature extraction model for image-based inputs
        from agent.modules.feature_model import FeatureModel
        self.cnn = FeatureModel()

        # Calculate the output size of the convolutional layers
        from common.utils import get_conv_out_size
        conv_out_size = get_conv_out_size(config.env.agent_obs_dim, self.cnn)

        # Define fully connected layers
        self.fc1 = torch.nn.Linear(conv_out_size + config.env.agent_action_dim, config.params["discr_hidden_dim"])
        self.fc2 = torch.nn.Linear(config.params["discr_hidden_dim"], 1)

        # Initialize weights with normal distribution
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        """
        Forward pass of the 2D discriminator with state and action inputs.

        Args:
            s: State input tensor with dimensions [batch_size, channels, height, width].
            a: Action input tensor.

        Returns:
            Probability output from the discriminator.
        """
        # Add channel dimension and convert grayscale to three channels
        s = s.unsqueeze(1)
        s = s.repeat(1, 3, 1, 1)

        # Extract features with CNN
        x = self.cnn(s)

        # Flatten the output from CNN
        x = x.view(x.size(0), -1)

        # Concatenate features with action and pass through fully connected layers
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
