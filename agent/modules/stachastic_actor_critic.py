import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.modules.base_network import ChkptModule
from agent.modules.deterministic_actor_critic import DeterministicActor2d


# 定义一维情形的AC网络
class StochasticActor(ChkptModule):
    def __init__(self, args, network_type):
        super().__init__(args, network_type)
        # 获取 args 中的维度信息
        self.fc_input_dim = args.agent_obs_dim[0]
        self.hidden_dim = args.actor_hidden_dim
        self.output_dim = args.agent_action_dim

        # 定义 actor 的核心网络
        self.fc1 = nn.Linear(self.fc_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.action_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.std_out = nn.Linear(self.hidden_dim, self.output_dim)

        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.action_out.weight.data.normal_(0, 0.1)
        self.std_out.weight.data.normal_(0, 0.1)

    def _forward(self, x):
        # 转换观测三维为二维
        x = x.reshape([-1, self.fc_input_dim])

        # 三层全连接神经网络
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.action_out(x))
        std = F.softplus(self.std_out(x)) + 1e-3
        return actions, std

    def forward(self, x):
        return self._forward(x)


class StochasticCritic(ChkptModule):
    def __init__(self, args, network_type):
        super(StochasticCritic, self).__init__(args, network_type)

        # 获取 args 中的维度信息
        self.fc_input_dim = args.agent_obs_dim[0]
        self.hidden_dim = args.critic_hidden_dim
        self.output_dim = 1

        # 定义 Critic 的核心网络
        self.fc1 = nn.Linear(self.fc_input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_out = nn.Linear(self.hidden_dim, 1)

        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.q_out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        # 转化观测三维为二维，并concat与action
        x = state.reshape([-1, self.fc_input_dim])

        # 三层全连接神经网络
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value


class StochasticActor2d(DeterministicActor2d):
    def __init__(self, args, network_type):
        super().__init__(args, network_type)


class StochasticActorSAC(StochasticActor):
    def __init__(self, args, network_type):
        super().__init__(args, network_type)

    def forward(self, state):
        mu, std = self._forward(state)
        dist = torch.distributions.Normal(mu, std)
        # 重参数化采样
        normal_sample = dist.rsample()
        action_tanh = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = dist.log_prob(normal_sample)
        log_prob -= (1.000001 - action_tanh.pow(2)).log()
        return action_tanh, log_prob.sum(-1)
