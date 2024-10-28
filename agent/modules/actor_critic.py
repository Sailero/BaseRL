import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.modules.base_network import ChkptModule
from agent.modules.feature_model import FeatureModel
from common.utils import get_conv_out_size

# define the critic network with pooling layers
class Critic2d(ChkptModule):
    def __init__(self, args, network_type):
        super().__init__(args, network_type)

        self.cnn = FeatureModel()

        # Compute the output size of conv layers
        conv_out_size = get_conv_out_size(args.agent_obs_dim, self.cnn)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, int(args.critic_hidden_dim / 2))
        self.q_out = nn.Linear(int(args.critic_hidden_dim / 2), 1)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.q_out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        # state is expected to be [batch_size, channels, height, width]
        # 扩充维数
        state = state.unsqueeze(1)
        # 灰度图转为三通道
        state = state.repeat(1, 3, 1, 1)

        x = self.cnn(state)

        # Flatten to [batch_size, conv_out_size]
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q_value = self.q_out(x)
        return q_value


class CriticSAC2d(ChkptModule):
    def __init__(self, args, network_type):
        super().__init__(args, network_type)
        self.cnn = FeatureModel()

        # Compute the output size of conv layers
        conv_out_size = get_conv_out_size(args.agent_obs_dim, self.cnn)

        self.fc1 = nn.Linear(conv_out_size + args.agent_action_dim, args.critic_hidden_dim)
        self.fc2 = nn.Linear(args.critic_hidden_dim, int(args.critic_hidden_dim / 2))
        self.q_out = nn.Linear(int(args.critic_hidden_dim / 2), 1)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.q_out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        # s is expected to be [batch_size, channels, height, width]
        # 扩充维数
        s = s.unsqueeze(1)
        # 灰度图转为三通道
        s = s.repeat(1, 3, 1, 1)

        x = self.cnn(s)

        # Flatten to [batch_size, conv_out_size]
        x = x.view(x.size(0), -1)

        cat = torch.cat([x, a], dim=1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))

        q_value = self.q_out(x)
        return q_value
