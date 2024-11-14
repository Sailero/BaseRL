import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.modules.base_network import ChkptModule
from agent.modules.feature_model import FeatureModel
from common.utils import get_conv_out_size


# define the actor network
class DeterministicActor(ChkptModule):
    def __init__(self, config, network_type):
        super(DeterministicActor, self).__init__(config, network_type)
        # 获取 config 中的维度信息
        self.fc_input_dim = config.env.agent_obs_dim[0]
        self.hidden_dim = config.params["actor_hidden_dim"]
        self.output_dim = config.env.agent_action_dim

        # 定义 actor 的核心网络
        self.fc1 = nn.Linear(self.fc_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.action_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # 转换观测三维为二维
        x = x.reshape([-1, self.fc_input_dim])

        # 三层全连接神经网络
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.action_out(x))
        return actions


class DeterministicCritic(ChkptModule):
    def __init__(self, config, network_type):
        super(DeterministicCritic, self).__init__(config, network_type)

        # 获取 config 中的维度信息
        self.obs_input_dim = config.env.agent_obs_dim[0]
        self.fc_input_dim = self.obs_input_dim + config.env.agent_action_dim
        self.hidden_dim = config.params["critic_hidden_dim"]
        self.output_dim = 1

        # 定义 Critic 的核心网络
        self.fc1 = nn.Linear(self.fc_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_out = nn.Linear(self.hidden_dim, 1)

    def forward(self, state, action):
        # 转化观测三维为二维，并concat与action
        state = state.reshape([-1, self.obs_input_dim])
        x = torch.cat([state, action], dim=1)

        # 三层全连接神经网络
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value


# define the actor network with pooling layers
class DeterministicActor2d(ChkptModule):
    def __init__(self, config, network_type):
        super().__init__(config, network_type)

        self.cnn = FeatureModel()

        # Compute the size of the output from conv layers
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
        x = F.tanh(self.output(x))

        return x


# define the critic network with pooling layers
class DeterministicCritic2d(ChkptModule):
    def __init__(self, config, network_type):
        super().__init__(config, network_type)

        self.cnn = FeatureModel()

        # Compute the output size of conv layers
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
        # state is expected to be [batch_size, channels, height, width]
        # 扩充维数
        state = state.unsqueeze(1)

        # 灰度图转为三通道
        state = state.repeat(1, 3, 1, 1)

        x = self.cnn(state)

        # Flatten to [batch_size, conv_out_size]
        x = x.view(x.size(0), -1)

        cat = torch.cat([x, action], dim=1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))

        q_value = self.q_out(x)
        return q_value