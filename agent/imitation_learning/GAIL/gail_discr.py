from agent.modules.base_network import ChkptModule
import torch
import torch.nn.functional as F


class Discriminator(ChkptModule):
    def __init__(self, config, network_type):
        super(Discriminator, self).__init__(config, network_type)
        self.fc1 = torch.nn.Linear(config.env.agent_obs_dim[0] + config.env.agent_action_dim, config.params["discr_hidden_dim"])
        self.fc2 = torch.nn.Linear(config.params["discr_hidden_dim"], 1)

        # Initialize weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        # Flatten to [batch_size, conv_out_size]
        x = s.view(s.size(0), -1)

        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class Discriminator2d(ChkptModule):
    def __init__(self, config, network_type):
        super(Discriminator2d, self).__init__(config, network_type)
        from agent.modules.feature_model import FeatureModel
        self.cnn = FeatureModel()

        # Compute the output size of conv layers
        from common.utils import get_conv_out_size
        conv_out_size = get_conv_out_size(config.env.agent_obs_dim, self.cnn)

        self.fc1 = torch.nn.Linear(conv_out_size + config.env.agent_action_dim, config.params["discr_hidden_dim"])
        self.fc2 = torch.nn.Linear(config.params["discr_hidden_dim"], 1)

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
