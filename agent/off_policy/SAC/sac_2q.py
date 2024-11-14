import numpy as np
import os
import torch
import torch.nn.functional as F


class SAC:
    name = 'SAC_Dual_Q'

    def __init__(self, config):
        # 设备信息
        self.device = config.device.device
        self.alpha_log_save_path = os.path.join(config.save_path, 'parameters/alpha_log.pth')

        # Parameters of SAC
        self.gamma = config.params["gamma"]
        self.tau = config.params["tau"]
        self.lr_alpha = config.params["lr_alpha"]
        self.max_grad_norm = config.params["max_grad_norm"]

        # import network
        if len(config.env.agent_obs_dim) == 1:
            from agent.off_policy.SAC.sac_actor_critic import StochasticActor as ActorSAC, StochasticCritic as CriticSAC
        else:
            from agent.off_policy.SAC.sac_actor_critic import StochasticActor2d as ActorSAC, \
                StochasticCritic2d as CriticSAC

        # create the network
        self.actor_net = ActorSAC(config, 'actor').to(self.device)
        self.critic_1 = CriticSAC(config, 'critic_1').to(self.device)
        self.critic_2 = CriticSAC(config, 'critic_2').to(self.device)
        self.critic_target_1 = CriticSAC(config, 'critic_target_1').to(self.device)
        self.critic_target_2 = CriticSAC(config, 'critic_target_2').to(self.device)

        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # create the optimizer 可控制需要优化的参数
        self.actor_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_net.parameters()),
                                            lr=config.params["lr_actor"])
        self.critic_1_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_net.parameters()),
                                               lr=config.params["lr_critic"])
        self.critic_2_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_2.parameters()),
                                               lr=config.params["lr_critic"])

        self.alpha_log = torch.tensor(np.log(0.01), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=config.params.lr_alpha)
        self.target_entropy = config.env.agent_action_dim

    def choose_action(self, observation):
        # Choose action based on actor network
        inputs = observation.clone().detach().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor_net(inputs)
        action = action[0].cpu().detach().numpy()
        action = np.clip(action, -1, 1)
        return action.tolist()

    def save_models(self):
        self.actor_net.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.critic_target_1.save_checkpoint()
        self.critic_target_2.save_checkpoint()
        torch.save(self.alpha_log.cpu(), self.alpha_log_save_path)

    def load_models(self):
        self.actor_net.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.critic_target_1.load_checkpoint()
        self.critic_target_2.load_checkpoint()
        alpha_log = torch.load(self.alpha_log_save_path, weights_only=True)
        self.alpha_log = torch.tensor(alpha_log.item(), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=self.lr_alpha)

    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor_net(next_states)
        q1_value = self.critic_target_1(next_states, next_actions)
        q2_value = self.critic_target_2(next_states, next_actions)
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
        trans_reward = transitions['reward']
        trans_next_obs = transitions['next_obs']
        trans_done = transitions['done']

        # 更新两个Q网络
        td_target = self.calc_target(trans_reward, trans_next_obs, trans_done)

        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(trans_obs, trans_action), td_target.detach()))

        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(trans_obs, trans_action), td_target.detach()))

        self.critic_1_optim.zero_grad()
        self.critic_2_optim.zero_grad()

        critic_1_loss.backward()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)

        self.critic_1_optim.step()
        self.critic_2_optim.step()

        new_actions, log_prob = self.actor_net(trans_obs)
        entropy = log_prob.unsqueeze(1)

        # 更新alpha值
        alpha_loss = torch.mean((self.target_entropy - entropy).detach() * self.alpha_log.exp())

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.alpha_log], self.max_grad_norm)
        self.alpha_optim.step()

        # 更新策略网络
        q1_value = self.critic_1(trans_obs, new_actions)
        q2_value = self.critic_2(trans_obs, new_actions)

        actor_loss = -torch.mean(torch.min(q1_value, q2_value) - self.alpha_log.exp() * entropy)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        self.soft_update(self.critic_1, self.critic_target_1)
        self.soft_update(self.critic_2, self.critic_target_2)
