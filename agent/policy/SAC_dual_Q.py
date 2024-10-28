from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F


class SAC:
    name = 'SAC'

    def __init__(self, args):
        self.device = args.device  # 设备信息

        # Read the training parameters from args
        self.gamma = args.gamma

        # Read sampling parameters
        self.batch_size = args.batch_size

        # Special parameters
        self.tau = args.tau
        self.action_clip = args.action_clip

        # import network
        if isinstance(args.agent_obs_dim, int):
            raise NotImplemented("not supported yet")
        else:
            from agent.modules.deterministic_actor_critic import DeterministicActorSAC2d as ActorSAC
            from agent.modules.actor_critic import CriticSAC2d as CriticSAC

        # create the network
        self.actor_net = ActorSAC(args, 'actor').to(self.device)
        # 第一个Q网络
        self.critic_1 = CriticSAC(args, 'critic_1').to(self.device)
        # 第二个Q网络
        self.critic_2 = CriticSAC(args, 'critic_2').to(self.device)
        # 第一个目标Q网络
        self.target_critic_1 = deepcopy(self.critic_1)
        # 第二个目标Q网络
        self.target_critic_2 = deepcopy(self.critic_2)

        # create the optimizer 可控制需要优化的参数
        self.actor_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_net.parameters()),
                                            lr=args.lr_actor)
        self.critic_1_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_1.parameters()),
                                               lr=args.lr_critic)
        self.critic_2_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_2.parameters()),
                                               lr=args.lr_critic)

        self.alpha_log = torch.tensor(np.log(0.01), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=args.lr_alpha)
        self.target_entropy = args.agent_action_dim

        # 记录训练过程数据
        self.train_record = dict()

    @property
    def critic_net(self):
        return self.critic_1

    def add_graph(self, obs, action, logger):
        from agent.policy.wrapper import WrapperStateAction2
        wrapper = WrapperStateAction2(self.actor_net, self.critic_1)
        logger.add_graph(wrapper, (obs, action))

    def choose_action(self, observation):
        # Choose action based on actor network
        inputs = observation.clone().detach().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor_net(inputs)

        action = action[0].cpu().detach().numpy()

        action = np.clip(action, -self.action_clip, self.action_clip)
        return action.tolist()

    def save_models(self):
        self.actor_net.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        self.actor_net.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor_net(next_states)
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value)
        entropy = self.alpha_log.exp() * log_prob.unsqueeze(1)
        next_value = next_value - entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    # update the network
    def train(self, transitions):
        # Transit tensor to gpu
        for key in transitions.keys():
            # 如果不是tensor，则转换为tensor
            if not isinstance(transitions[key], torch.Tensor):
                transitions[key] = torch.tensor(np.array(transitions[key]), dtype=torch.float32).to(self.device)
        trans_obs = transitions['obs']
        trans_action = transitions['action']
        trans_reward = transitions['reward'].unsqueeze(1)
        trans_next_obs = transitions['next_obs']
        trans_done = transitions['done'].unsqueeze(1)

        # 更新两个Q网络
        td_target = self.calc_target(trans_reward, trans_next_obs, trans_done)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(trans_obs, trans_action), td_target.detach()))
        self.train_record[self.name + '/critic_1_loss'] = critic_1_loss.item()
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(trans_obs, trans_action), td_target.detach()))
        self.train_record[self.name + '/critic_2_loss'] = critic_2_loss.item()
        self.critic_1_optim.zero_grad()
        self.critic_2_optim.zero_grad()

        critic_1_loss.backward()
        critic_2_loss.backward()

        self.critic_1_optim.step()
        self.critic_2_optim.step()

        new_actions, log_prob = self.actor_net(trans_obs)
        entropy = log_prob.unsqueeze(1)

        # 更新alpha值
        alpha_loss = torch.mean((self.target_entropy - entropy).detach() * self.alpha_log.exp())
        self.train_record[self.name + '/alpha_loss'] = alpha_loss.item()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # 更新策略网络
        q1_value = self.critic_1(trans_obs, new_actions)
        q2_value = self.critic_2(trans_obs, new_actions)
        actor_loss = -torch.mean(torch.min(q1_value, q2_value) - self.alpha_log.exp() * entropy)
        self.train_record[self.name + '/actor_loss'] = actor_loss.item()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
