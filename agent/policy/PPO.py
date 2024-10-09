import torch
import numpy as np

class PPO:
    def __init__(self, args):
        # Read the training parameters from args
        self.gamma = args.gamma
        self.action_dim = args.agent_action_dim
        self.max_grad_norm = args.max_grad_norm
        self.device = args.device  # 设备信息

        # Read sampling parameters
        self.batch_size = args.batch_size

        # Special parameters of PPO
        self.lam = args.lam  # GAE lambda
        self.eps_clip = args.eps_clip
        self.update_nums = args.update_nums
        self.ent_coef = args.ent_coef  # entropy coefficient
        self.value_loss_coef = args.value_loss_coef

        # import network
        if len(args.agent_obs_dim) == 2:
            from agent.modules.online_actor_critic_2d import Actor, Critic
        else:
            from agent.modules.online_actor_critic import Actor, Critic

        # create the network
        self.actor_network = Actor(args, 'actor').to(self.device)
        self.old_actor_network = Actor(args, 'actor').to(self.device)
        self.critic_network = Critic(args, 'critic').to(self.device)

        # load the parameters
        self.old_actor_network.load_state_dict(self.actor_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=args.lr_critic)

    # GAE (Generalized Advantage Estimation)
    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            # 这里next_value和values都是length长度的列表or数组。其中next_value为value[1:]，values是value[:-1]
            delta = rewards[t] + self.gamma * next_values[t] * (1 - int(dones[t])) - values[t]
            advantage = delta + self.gamma * self.lam * advantage * (1 - int(dones[t]))
            advantages.insert(0, advantage)
        return advantages

    # update the network
    def train(self, transitions):
        # Transit tensor to gpu
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(self.device)
        trans_obs = transitions['obs']
        trans_action = transitions['action']
        trans_reward = transitions['reward']
        trans_next_obs = transitions['next_obs']
        trans_done = transitions['done']
        trans_len = len(trans_obs)

        # Scale the reward
        trans_reward = (trans_reward - trans_reward.mean()) / (trans_reward.std() + 1e-6)


        # Load state dicts
        self.old_actor_network.load_state_dict(self.actor_network.state_dict())

        with torch.no_grad():
            tran_value = self.critic_network(trans_obs)
            tran_next_value = self.critic_network(trans_next_obs)

        # Calculate target values and advantages before updating
        with torch.no_grad():
            advantages = self.compute_gae(trans_reward, tran_value, tran_next_value, trans_done)
            advantages = torch.tensor(advantages, dtype=torch.float32).reshape(tran_next_value.shape).to(self.device)
            returns = advantages + tran_value  # Rt = At + V(st)

        for _ in range(self.update_nums):
            # Acquire sampling id
            batch_start_points = np.arange(0, trans_len, self.batch_size)
            buffer_indices = np.arange(trans_len)
            np.random.shuffle(buffer_indices)
            batch_id_set = [buffer_indices[i: i + self.batch_size] for i in batch_start_points]

            for batch_ids in batch_id_set:
                # Get sampling data
                batch_obs = trans_obs[batch_ids]
                batch_action = trans_action[batch_ids]
                batch_advantages = advantages[batch_ids]
                batch_returns = returns[batch_ids]

                # Calculate old log probs
                with torch.no_grad():
                    old_pi_mu, old_pi_sigma = self.old_actor_network(batch_obs)
                    old_pi = torch.distributions.Normal(old_pi_mu, old_pi_sigma)
                    batch_old_log_prob = old_pi.log_prob(batch_action).sum(-1, keepdim=True)

                # Calculate new log probs
                print(batch_obs)
                pi_mu, pi_std = self.actor_network(batch_obs)
                print(pi_mu)
                dist = torch.distributions.Normal(pi_mu, pi_std)
                batch_new_log_prob = dist.log_prob(batch_action).sum(-1, keepdim=True)

                # Update
                ratio = torch.exp(batch_new_log_prob - batch_old_log_prob)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                # Add entropy to encourage exploration
                entropy = dist.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean()  #  - self.ent_coef * entropy

                # Critic update
                gae_value = self.critic_network(batch_obs)
                critic_loss = (batch_returns - gae_value).pow(2).mean()

                # print()
                # print('actor_loss', actor_loss)
                # print('critic_loss', critic_loss)

                # Total loss
                # total_loss = actor_loss  + self.value_loss_coef * critic_loss

                # Update the network
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                # total_loss.backward()
                actor_loss.backward()
                critic_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)

                self.actor_optim.step()
                self.critic_optim.step()


    def choose_action(self, observation):
        # Choose action based on actor network
        inputs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pi_mu, pi_std = self.actor_network(inputs)
            pi_mu, pi_std = pi_mu.reshape(-1), pi_std.reshape(-1)
            # print('pi_mu', pi_mu, 'pi_std', pi_std)
        dist = torch.distributions.Normal(pi_mu, pi_std)
        action = dist.sample()
        action = action.cpu().detach().numpy()

        action = np.clip(action, -1, 1)
        return action.tolist()

    def save_models(self):
        self.actor_network.save_checkpoint()
        self.critic_network.save_checkpoint()

    def load_models(self):
        self.actor_network.load_checkpoint()
        self.critic_network.load_checkpoint()

