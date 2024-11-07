import torch
import torch.nn as nn
import numpy as np
from agent.imitation_learning.im_replay_buffer import ImBuffer
from agent.imitation_learning.im_config import GAIL_CONFIG

class GAIL:
    def __init__(self, args):
        self.device = args.device
        self.im_buffer = ImBuffer(args)
        if args.policy_type == "GAIL_PPO":
            from agent.imitation_learning.GAIL.gail_ppo import GailPPO
            self.agent = GailPPO(args)
        elif args.policy_type == "GAIL_SAC":
            from agent.imitation_learning.GAIL.gail_sac_2q import GailSAC
            self.agent = GailSAC(args)
        else:
            raise ValueError(f"{args.policy_type} is not supported.")

        if len(args.agent_obs_dim) == 1:
            from agent.imitation_learning.GAIL.gail_discr import Discriminator
        else:
            from agent.imitation_learning.GAIL.gail_discr import Discriminator2d as Discriminator
        self.discr_net = Discriminator(args, 'discriminator').to(self.device)
        self.discr_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discr_net.parameters()),
                                            lr=GAIL_CONFIG["lr_discr"])

        # 记录训练过程数据
        self.train_record = dict()
        self.episode_num = 0
        self.name = args.policy_type

    def add_graph(self, obs, action, logger):
        if "SAC" in self.agent.name:
            from agent.modules.wrapper import WrapperStateAction3_1 as WrapperStateAction3
        else:
            from agent.modules.wrapper import WrapperStateAction3
        wrapper = WrapperStateAction3(self.agent.actor_net, self.agent.critic_net, self.discr_net)
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
        agent_data = transitions
        expert_data = self.im_buffer.sample(im_sample_size=len(agent_data['obs']))

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

        expert_rewards = -torch.log(agent_prob).detach().cpu().numpy()
        self.train_record[self.name + '/rewards_mean'] = expert_rewards.mean()
        new_transitions = {'obs': agent_obs,
                           'action': agent_actions,
                           'expert_reward': expert_rewards,
                           'reward': agent_data['reward'],
                           'next_obs': agent_data['next_obs'],
                           'done': agent_data['done']
                           }
        self.agent.train(new_transitions)
        self.episode_num += 1

        # 合并 self.train_record 和 agent.train_record
        self.train_record.update(self.agent.train_record)
