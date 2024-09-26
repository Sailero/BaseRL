import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.base_network import ChkptModule

# define the actor network with pooling layers
class Actor(ChkptModule):
    def __init__(self, args, network_type):
        super(Actor, self).__init__(args, network_type)

        # Conv layer parameters (use args or set manually)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)  # Max pooling layer
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer

        # Compute the size of the output from conv layers
        conv_out_size = self._get_conv_out_size(args.agent_obs_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, args.actor_hidden_dim)
        self.fc2 = nn.Linear(args.actor_hidden_dim, args.actor_hidden_dim)
        self.action_out = nn.Linear(args.actor_hidden_dim, args.agent_action_dim)
        self.std_out = nn.Linear(args.actor_hidden_dim, args.agent_action_dim)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.action_out.weight.data.normal_(0, 0.1)
        self.std_out.weight.data.normal_(0, 0.1)

    def _get_conv_out_size(self, shape):
        # Pass dummy input to get the output size of the conv layers
        shape = [1] + shape
        o = torch.zeros(1, *shape)
        o = self.pool1(self.conv1(o))
        o = self.pool2(self.conv2(o))
        o = self.pool3(self.conv3(o))
        o = self.pool4(self.conv4(o))
        return int(torch.prod(torch.tensor(o.shape[1:])))

    def forward(self, x):
        # x is expected to be [batch_size, channels, height, width]
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)  # Max pooling after activation
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # Max pooling after activation
        x = F.relu(self.conv3(x))
        x = self.pool3(x)  # Max pooling after activation
        x = F.relu(self.conv4(x))
        x = self.pool4(x)  # Max pooling after activation

        # Flatten the conv output
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, conv_out_size]

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        actions = torch.tanh(self.action_out(x))
        std = F.softplus(self.std_out(x)) + 1e-3
        return actions, std


# define the critic network with pooling layers
class Critic(ChkptModule):
    def __init__(self, args, network_type):
        super(Critic, self).__init__(args, network_type)

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer

        # Compute the output size of conv layers
        conv_out_size = self._get_conv_out_size(args.agent_obs_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, args.critic_hidden_dim)
        self.q_out = nn.Linear(args.critic_hidden_dim, 1)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.q_out.weight.data.normal_(0, 0.1)

    def _get_conv_out_size(self, shape):
        # Pass dummy input to get the output size of the conv layers
        shape = [1] + shape
        o = torch.zeros(1, *shape)
        o = self.pool1(self.conv1(o))
        o = self.pool2(self.conv2(o))
        o = self.pool3(self.conv3(o))
        o = self.pool4(self.conv4(o))
        return int(torch.prod(torch.tensor(o.shape[1:])))

    def forward(self, state):
        # state is expected to be [batch_size, channels, height, width]
        state = state.unsqueeze(1)
        x = F.relu(self.conv1(state))
        x = self.pool1(x)  # Max pooling after activation
        x = F.relu(self.conv2(x))
        x = self.pool2(x)  # Max pooling after activation
        x = F.relu(self.conv3(x))
        x = self.pool3(x)  # Max pooling after activation
        x = F.relu(self.conv4(x))
        x = self.pool4(x)  # Max pooling after activation

        # Flatten the conv output
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, conv_out_size]

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q_value = self.q_out(x)
        return q_value
