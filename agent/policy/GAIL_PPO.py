import torch
import torch.nn as nn
import numpy as np


class GAIL_PPO:
    def __init__(self, args):
        self.device = args.device
        self.gamma = args.gamma
        self.lam = args.lam
        self.eps_clip = args.eps_clip
        self.batch_size = args.batch_size
        self.update_nums = args.update_nums
        self.max_grad_norm = args.max_grad_norm
        self.action_clip = args.action_clip
        self.name = "GAIL_PPO"

        # 网络初始化
        if len(args.agent_obs_dim) == 1:
            from agent.modules.stachastic_actor_critic import StochasticActor as Actor, StochasticCritic as Critic, Discriminator
        else:
            from agent.modules.stachastic_actor_critic import StochasticActor2d as Actor, StochasticCritic2d as Critic, Discriminator2d as Discriminator

        # Actor 和 Critic 网络
        self.actor_network = Actor(args, 'actor').to(self.device)
        self.old_actor_network = Actor(args, 'actor').to(self.device)
        self.rl_critic_network = Critic(args, 'rl_critic').to(self.device)  # RL Critic
        self.expert_critic_network = Critic(args, 'expert_critic').to(self.device)  # Expert Critic
        self.discr_net = Discriminator(args, 'discriminator').to(self.device)  # Discriminator

        # 同步 Actor 参数
        self.old_actor_network.load_state_dict(self.actor_network.state_dict())

        # 优化器
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=args.lr_actor)
        self.rl_critic_optim = torch.optim.Adam(self.rl_critic_network.parameters(), lr=args.lr_critic)
        self.expert_critic_optim = torch.optim.Adam(self.expert_critic_network.parameters(), lr=args.lr_critic)
        self.discr_optim = torch.optim.Adam(self.discr_net.parameters(), lr=args.lr_discr)

        # 训练记录
        self.train_record = dict()
        self.episode_num = 0

    def add_graph(self, obs, action, logger):
        from agent.policy.wrapper import WrapperStateAction3
        wrapper = WrapperStateAction3(self.actor_network, self.rl_critic_network, self.discr_net)
        logger.add_graph(wrapper, (obs, action))

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = np.zeros(len(values))
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - int(dones[t])) - values[t]
            advantage = delta + self.gamma * self.lam * advantage * (1 - int(dones[t]))
            advantages[t] = advantage
        return advantages

    def train(self, transitions):
        agent_data, expert_data = transitions

        agent_obs = torch.tensor(np.array(agent_data['obs']), dtype=torch.float32).to(self.device)
        agent_actions = torch.tensor(np.array(agent_data['action']), dtype=torch.float32).to(self.device)
        expert_obs = torch.tensor(np.array(expert_data['obs']), dtype=torch.float32).to(self.device)
        expert_action = torch.tensor(np.array(expert_data['action']), dtype=torch.float32).to(self.device)

        # 1. 计算 Discriminator 的输出，用于 GAIL 奖励
        agent_prob = self.discr_net(agent_obs, agent_actions)
        expert_prob = self.discr_net(expert_obs, expert_action)

        expert_loss = nn.BCELoss()(expert_prob, torch.zeros_like(expert_prob))
        self.train_record[self.name + '/expert_loss'] = expert_loss.item()
        agent_loss = nn.BCELoss()(agent_prob, torch.ones_like(agent_prob))
        self.train_record[self.name + '/agent_loss'] = agent_loss.item()
        discriminator_loss = agent_loss + expert_loss
        self.train_record[self.name + '/discriminator_loss'] = discriminator_loss.item()

        self.discr_optim.zero_grad()
        discriminator_loss.backward()
        self.discr_optim.step()

        # 2. 结合 GAIL 奖励和 RL 奖励
        expert_rewards = -torch.log(agent_prob).detach().cpu().numpy()
        ppo_rewards = np.array(agent_data['reward']).reshape(-1, 1)
        expert_rewards = (expert_rewards - expert_rewards.mean()) / (expert_rewards.std() + 1e-6)  # 标准化奖励
        ppo_rewards = (ppo_rewards - ppo_rewards.mean()) / (ppo_rewards.std() + 1e-6)  # 标准化奖励

        trans_len = len(agent_obs)
        agent_next_obs = torch.tensor(np.array(agent_data['next_obs']), dtype=torch.float32).to(self.device)
        trans_done = torch.tensor(np.array(agent_data['done']), dtype=torch.float32).to(self.device)

        # PPO模块
        self.old_actor_network.load_state_dict(self.actor_network.state_dict())

        with torch.no_grad():
            # 计算两个 Critic 网络的值
            ppo_value = self.rl_critic_network(agent_obs)
            ppo_next_value = self.rl_critic_network(agent_next_obs)
            expert_value = self.expert_critic_network(agent_obs)
            expert_next_value = self.expert_critic_network(agent_next_obs)

        # 3. 计算 GAE
        ppo_advantages = self.compute_gae(ppo_rewards, ppo_value.cpu().numpy(), ppo_next_value.cpu().numpy(), trans_done.cpu().numpy())
        expert_advantages = self.compute_gae(expert_rewards, expert_value.cpu().numpy(), expert_next_value.cpu().numpy(), trans_done.cpu().numpy())
        ppo_advantages = torch.tensor(ppo_advantages, dtype=torch.float32).reshape(ppo_value.shape).to(self.device)
        expert_advantages = torch.tensor(expert_advantages, dtype=torch.float32).reshape(expert_value.shape).to(self.device)
        ppo_returns = ppo_advantages + ppo_value  # Rt = At + V(st)
        expert_returns = expert_advantages + expert_value  # Rt = At + V(st)

        # 计算更新actor的critic系数比例，分别代表开始的回合，持续的回合以及原始奖励所占的比例
        start_episode = 0
        decay_duration = 800
        combined_ratio = np.clip((self.episode_num - start_episode) / decay_duration, 0.9, 1.)

        # 4. Actor 和 Critic 网络的更新
        for _ in range(self.update_nums):
            batch_start_points = np.arange(0, trans_len, self.batch_size)
            buffer_indices = np.arange(trans_len)
            np.random.shuffle(buffer_indices)
            batch_id_set = [buffer_indices[i: i + self.batch_size] for i in batch_start_points]
            for batch_ids in batch_id_set:
                batch_obs = agent_obs[batch_ids]
                batch_action = agent_actions[batch_ids]

                batch_ppo_advantages = ppo_advantages[batch_ids]
                batch_expert_advantages = expert_advantages[batch_ids]
                batch_combined_advantages = combined_ratio * batch_ppo_advantages + (
                            1 - combined_ratio) * batch_expert_advantages

                batch_ppo_returns = ppo_returns[batch_ids]
                batch_expert_returns = expert_returns[batch_ids]

                # Actor 更新
                with torch.no_grad():
                    old_pi_mu, old_pi_sigma = self.old_actor_network(batch_obs)
                    old_pi = torch.distributions.Normal(old_pi_mu, old_pi_sigma)
                    batch_old_log_prob = old_pi.log_prob(batch_action).sum(-1, keepdim=True)

                pi_mu, pi_std = self.actor_network(batch_obs)
                dist = torch.distributions.Normal(pi_mu, pi_std)
                batch_new_log_prob = dist.log_prob(batch_action).sum(-1, keepdim=True)

                ratio = torch.exp(batch_new_log_prob - batch_old_log_prob)
                surr1 = ratio * batch_combined_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_combined_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                # RL Critic 更新
                ppo_critic_loss = (batch_ppo_returns - self.rl_critic_network(batch_obs)).pow(2).mean()
                self.rl_critic_optim.zero_grad()
                ppo_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rl_critic_network.parameters(), self.max_grad_norm)
                self.rl_critic_optim.step()

                # Expert Critic 更新
                expert_critic_loss = (batch_expert_returns - self.expert_critic_network(batch_obs)).pow(2).mean()
                self.expert_critic_optim.zero_grad()
                expert_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.expert_critic_network.parameters(), self.max_grad_norm)
                self.expert_critic_optim.step()

        self.train_record[self.name + '/actor_loss'] = actor_loss.item()
        self.train_record[self.name + '/ppo_critic_loss'] = ppo_critic_loss.item()
        self.train_record[self.name + '/expert_critic_loss'] = expert_critic_loss.item()
        self.episode_num += 1

    def choose_action(self, observation):
        inputs = observation.clone().detach().unsqueeze(0).to(self.device)
        with torch.no_grad():
            pi_mu, pi_std = self.actor_network(inputs)
            pi_mu, pi_std = pi_mu.reshape(-1), pi_std.reshape(-1)
        dist = torch.distributions.Normal(pi_mu, pi_std)
        action = dist.sample()
        action = np.clip(action.cpu().detach().numpy(), -self.action_clip, self.action_clip)
        return action.tolist()

    def save_models(self):
        self.actor_network.save_checkpoint()
        self.rl_critic_network.save_checkpoint()
        self.expert_critic_network.save_checkpoint()
        self.discr_net.save_checkpoint()

    def load_models(self):
        self.actor_network.load_checkpoint()
        self.rl_critic_network.load_checkpoint()
        self.expert_critic_network.load_checkpoint()
        self.discr_net.load_checkpoint()
