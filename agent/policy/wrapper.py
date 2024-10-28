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
            actor_net,
            critic_net,
    ):
        super().__init__()

        self.actor_net = actor_net
        self.critic_net = critic_net

    def forward(self, state, action):
        q1 = self.actor_net(state)
        q2 = self.critic_net(state, action)
        return q1, q2


class WrapperStateAction3(nn.Module):
    """
    只是用来生成 tensorboard 的 graph
    """

    def __init__(
            self,
            actor_net,
            critic_net,
            discriminator_net,
    ):
        super().__init__()

        self.actor_net = actor_net
        self.critic_net = critic_net
        self.discriminator_net = discriminator_net

    def forward(self, state, action):
        q1 = self.actor_net(state)
        q2 = self.critic_net(state)
        q3 = self.discriminator_net(state, action)
        return q1, q2, q3


class WrapperStateAction3_1(nn.Module):
    """
    只是用来生成 tensorboard 的 graph
    """

    def __init__(
            self,
            actor_net,
            critic_net,
            discriminator_net,
    ):
        super().__init__()

        self.actor_net = actor_net
        self.critic_net = critic_net
        self.discriminator_net = discriminator_net

    def forward(self, state, action):
        q1 = self.actor_net(state)
        q2 = self.critic_net(state, action)
        q3 = self.discriminator_net(state, action)
        return q1, q2, q3
