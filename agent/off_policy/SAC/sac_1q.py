import numpy as np
import os
import torch
import torch.nn.functional as F
from agent.modules.wrapper import WrapperStateAction2
from agent.off_policy.offp_config import *


class SAC:
    name = 'SAC'

    def __init__(self, args):
        # 设备信息
        self.device = args.device
        self.alpha_log_save_path = os.path.join(args.save_path, 'parameters/alpha_log.pth')

        # Parameters of SAC
        self.gamma = SAC_CONFIG["gamma"]
        self.tau = SAC_CONFIG["tau"]
        self.lr_alpha = SAC_CONFIG["lr_alpha"]

        # import network
        if len(args.agent_obs_dim) == 1:
            from sac_actor_critic import StochasticActor as ActorSAC, StochasticCritic as CriticSAC
        else:
            from sac_actor_critic import StochasticActor2d as ActorSAC, StochasticCritic2d as CriticSAC

        # create the network
        self.actor_net = ActorSAC(args, 'actor').to(self.device)
        self.critic_net = CriticSAC(args, 'critic').to(self.device)
        self.critic_target = CriticSAC(args, 'critic_target').to(self.device)
        self.critic_target.load_state_dict(self.critic_net.state_dict())

        # create the optimizer 可控制需要优化的参数
        self.actor_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_net.parameters()),
                                            lr=SAC_CONFIG["lr_actor"])
        self.critic_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_net.parameters()),
                                             lr=SAC_CONFIG["lr_critic"])

        self.alpha_log = torch.tensor(np.log(0.01), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=self.lr_alpha)
        self.target_entropy = args.agent_action_dim

        # 记录训练过程数据
        self.train_record = {"SAC/reward": [], "SAC/td_target": [], "SAC/q": [], "SAC/q_pg": [], "SAC/action": []}

    def add_graph(self, obs, action, logger):
        wrapper = WrapperStateAction2(self.actor_net, self.critic_net)
        logger.add_graph(wrapper, (obs, action))

    def choose_action(self, observation):
        # Choose action based on actor network
        inputs = observation.clone().detach().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor_net(inputs)

        action = action[0].cpu().detach().numpy()

        action = np.clip(action, -1, 1)

        self.train_record["SAC/action"].append(action.tolist())
        if len(self.train_record["SAC/action"]) > 64:
            self.train_record["SAC/action"].pop(0)

        return action.tolist()

    def save_models(self):
        self.actor_net.save_checkpoint()
        self.critic_net.save_checkpoint()
        self.critic_target.save_checkpoint()
        # 打印self.alpha_log
        print("alpha_log:", self.alpha_log.item())
        # 保存self.alpha_log
        torch.save(self.alpha_log.cpu(), self.alpha_log_save_path)

    def load_models(self):
        self.actor_net.load_checkpoint()
        self.critic_net.load_checkpoint()
        self.critic_target.load_checkpoint()
        # 加载self.alpha_log
        alpha_log = torch.load(self.alpha_log_save_path, weights_only=True)
        # 打印self.alpha_log
        print("alpha_log:", alpha_log.item())
        self.alpha_log = torch.tensor(alpha_log.item(), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=self.lr_alpha)

    def calc_target(self, rewards, next_states, dones):
        """
        计算目标Q值
        """
        next_actions, next_log_prob = self.actor_net(next_states)
        next_q = self.critic_target(next_states, next_actions)
        td_target = rewards + self.gamma * (next_q - self.alpha_log.exp() * next_log_prob.unsqueeze(-1)) * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    # update the network
    def train(self, transitions):
        self.train_record[self.name + '/reward'] = transitions['reward']

        # Transit tensor to gpu
        for key in transitions.keys():
            # 如果不是tensor，则转换为tensor
            if not isinstance(transitions[key], torch.Tensor):
                transitions[key] = torch.tensor(np.array(transitions[key]), dtype=torch.float32).to(self.device)
        trans_obs = transitions['obs']
        trans_action = transitions['action']
        trans_reward = transitions['reward']
        trans_next_obs = transitions['next_obs']
        trans_done = transitions['done']

        # 更新Q网络
        td_target = self.calc_target(trans_reward, trans_next_obs, trans_done)
        self.train_record[self.name + '/td_target'] = td_target.squeeze(1).cpu().detach().numpy().tolist()

        q_values = self.critic_net(trans_obs, trans_action)
        self.train_record[self.name + '/q'] = q_values.squeeze(1).cpu().detach().numpy().tolist()

        critic_loss = torch.mean(F.mse_loss(q_values, td_target))
        self.train_record[self.name + '/critic_loss'] = critic_loss.item()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.soft_update(self.critic_net, self.critic_target)

        action_pg, log_prob = self.actor_net(trans_obs)
        log_prob = log_prob.unsqueeze(1)

        # 更新alpha值
        alpha_loss = torch.mean((self.target_entropy - log_prob).detach() * self.alpha_log)
        self.train_record[self.name + '/alpha_loss'] = alpha_loss.item()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # 更新策略网络
        q_value_pg = self.critic_target(trans_obs, action_pg)
        self.train_record[self.name + '/q_pg'] = q_value_pg.squeeze(1).cpu().detach().numpy().tolist()

        actor_loss = torch.mean(self.alpha_log.exp() * log_prob - q_value_pg)
        self.train_record[self.name + '/actor_loss'] = actor_loss.item()
        self.train_record[self.name + '/alpha'] = self.alpha_log.exp().item()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
