import torch.nn as nn


class WrapperState2(nn.Module):
    """
    只是用来生成 tensorboard 的 graph
    """

    def __init__(
            self,
            net_actor,
            net_critic,
    ):
        super().__init__()

        self.net_actor = net_actor
        self.net_critic = net_critic

    def forward(self, state):
        q1 = self.net_actor(state)
        q2 = self.net_critic(state)
        return q1, q2


class WrapperStateAction2(nn.Module):
    """
    只是用来生成 tensorboard 的 graph
    """

    def __init__(
            self,
            net_actor,
            net_critic,
    ):
        super().__init__()

        self.net_actor = net_actor
        self.net_critic = net_critic

    def forward(self, state, action):
        q1 = self.net_actor(state)
        q2 = self.net_critic(state, action)
        return q1, q2


class WrapperStateAction3(nn.Module):
    """
    只是用来生成 tensorboard 的 graph
    """

    def __init__(
            self,
            net_actor,
            net_critic,
            net_discriminator,
    ):
        super().__init__()

        self.net_actor = net_actor
        self.net_critic = net_critic
        self.net_discriminator = net_discriminator

    def forward(self, state, action):
        q1 = self.net_actor(state)
        q2 = self.net_critic(state)
        q3 = self.net_discriminator(state, action)
        return q1, q2, q3
