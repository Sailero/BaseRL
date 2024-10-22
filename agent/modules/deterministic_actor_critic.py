import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.modules.base_network import ChkptModule
from agent.modules.feature_model import FeatureModel


# define the actor network
class DeterministicActor(ChkptModule):
    def __init__(self, args, network_type):
        super(DeterministicActor, self).__init__(args, network_type)
        # 获取 args 中的维度信息
        self.fc_input_dim = args.agent_obs_dim[0]
        self.hidden_dim = args.actor_hidden_dim
        self.output_dim = args.agent_action_dim

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
    def __init__(self, args, network_type):
        super(DeterministicCritic, self).__init__(args, network_type)

        # 获取 args 中的维度信息
        self.obs_input_dim = args.agent_obs_dim[0]
        self.fc_input_dim = self.obs_input_dim + args.agent_action_dim
        self.hidden_dim = args.critic_hidden_dim
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


# 定义二维情形的AC网络
def get_conv_out_size(shape, net):
    # Pass dummy input to get the output size of the conv layers
    shape = [3] + shape
    o = torch.zeros(1, *shape)
    o = net(o)
    return int(torch.prod(torch.tensor(o.shape[1:])))


class Discriminator(ChkptModule):
    def __init__(self, args, network_type):
        super(Discriminator, self).__init__(args, network_type)
        self.cnn = FeatureModel()

        # Compute the output size of conv layers
        conv_out_size = get_conv_out_size(args.agent_obs_dim, self.cnn)

        self.fc1 = torch.nn.Linear(conv_out_size + args.agent_action_dim, args.discr_hidden_dim)
        self.fc2 = torch.nn.Linear(args.discr_hidden_dim, 1)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

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
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))


# define the actor network with pooling layers
class DeterministicActor2d(ChkptModule):
    def __init__(self, args, network_type):
        super(DeterministicActor2d, self).__init__(args, network_type)

        self.cnn = FeatureModel()

        # Compute the size of the output from conv layers
        conv_out_size = get_conv_out_size(args.agent_obs_dim, self.cnn)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, args.actor_hidden_dim)
        self.fc2 = nn.Linear(args.actor_hidden_dim, int(args.actor_hidden_dim / 2))
        self.action_out = nn.Linear(int(args.actor_hidden_dim / 2), args.agent_action_dim)
        self.std_out = nn.Linear(int(args.actor_hidden_dim / 2), args.agent_action_dim)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.action_out.weight.data.normal_(0, 0.1)
        self.std_out.weight.data.normal_(0, 0.1)

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

        actions = torch.tanh(self.action_out(x))
        std = F.softplus(self.std_out(x)) + 1e-3
        return actions, std


# define the critic network with pooling layers
class DeterministicCritic2d(ChkptModule):
    def __init__(self, args, network_type):
        super(DeterministicCritic2d, self).__init__(args, network_type)

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