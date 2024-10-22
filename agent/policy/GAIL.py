import torch
import torch.nn as nn
import numpy as np


class GAIL:
    name = 'GAIL'

    def __init__(self, args, agent):
        self.device = args.device
        self.agent = agent

        from agent.modules.stachastic_actor_critic import Discriminator
        self.discr_net = Discriminator(args, 'discriminator').to(self.device)
        # 可控制需要优化的参数
        self.discr_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discr_net.parameters()),
                                            lr=args.lr_discr)

        # 记录训练过程数据
        self.train_record = dict()

    def add_graph(self, obs, action, logger):
        from agent.policy.wrapper import WrapperStateAction3
        wrapper = WrapperStateAction3(self.agent.actor_network, self.agent.critic_network, self.discr_net)
        logger.add_graph(wrapper, (obs, action))

    def save_models(self):
        self.discr_net.save_checkpoint()
        self.agent.save_models()

    def load_models(self):
        self.discr_net.load_checkpoint()
        self.agent.load_models()

    def choose_action(self, observation):
        return self.agent.choose_action(observation)

    def train(self, transitions):
        agent_data, expert_data = transitions

        expert_obs = torch.tensor(expert_data['obs'], dtype=torch.float32).to(self.device)
        expert_action = torch.tensor(expert_data['action'], dtype=torch.float32).to(self.device)
        agent_obs = torch.tensor(np.array(agent_data['obs']), dtype=torch.float32).to(self.device)
        agent_actions = torch.tensor(np.array(agent_data['action']), dtype=torch.float32).to(self.device)

        expert_prob = self.discr_net(expert_obs, expert_action)
        agent_prob = self.discr_net(agent_obs, agent_actions)

        expert_loss = nn.BCELoss()(expert_prob, torch.zeros_like(expert_prob))
        self.train_record[self.name + '/expert_loss'] = expert_loss.item()
        agent_loss = nn.BCELoss()(agent_prob, torch.ones_like(agent_prob))
        self.train_record[self.name + '/agent_loss'] = agent_loss.item()
        discriminator_loss = agent_loss + expert_loss
        self.train_record[self.name + '/discriminator_loss'] = discriminator_loss.item()

        self.discr_optim.zero_grad()
        discriminator_loss.backward()
        self.discr_optim.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        self.train_record[self.name + '/rewards_mean'] = rewards.mean()
        new_transitions = {'obs': agent_obs,
                           'action': agent_actions,
                           'reward': rewards,
                           'next_obs': agent_data['next_obs'],
                           'done': agent_data['done']
                           }
        self.agent.train(new_transitions)

        # 合并 self.train_record 和 agent.train_record
        self.train_record.update(self.agent.train_record)
