import torch
from modules.st_replay_buffer import Buffer
from modules.st_actor_critic import Actor, Critic


class PPO:
    def __init__(self, args):
        # Read the training parameters from args
        self.gamma = args.gamma
        self.action_dim = args.agent_action_dim
        self.max_grad_norm = args.max_grad_norm
        self.device = args.device  # 设备信息

        # Special parameters of PPO
        self.lam = args.lam  # GAE lambda
        self.eps_clip = args.eps_clip
        self.update_steps = args.update_steps
        self.ent_coef = args.ent_coef  # entropy coefficient
        self.value_loss_coef = args.value_loss_coef

        # Initialize the buffer
        self.buffer = Buffer(args)

        # create the network
        self.actor_network = Actor(args, 'actor').to(self.device)
        self.critic_network = Critic(args, 'critic').to(self.device)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=args.lr_critic)

    # GAE (Generalized Advantage Estimation)
    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value[t] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.lam * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
        return advantages

    # update the network
    def train(self):
        # 采样transitions
        transitions = self.buffer.buffer

        # 将所有的转移迁移到GPU
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(self.device)
        batch_obs = transitions['obs']
        batch_action = transitions['action']
        batch_log_prob = transitions['log_prob']
        batch_reward = transitions['reward']
        batch_next_obs = transitions['next_obs']
        batch_done = transitions['done']
        batch_value = transitions['value']

        # Calculate target values and advantages
        with torch.no_grad():
            next_value = self.critic_network(batch_next_obs)
            advantages = self.compute_gae(batch_reward, batch_value, next_value, batch_done)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            returns = advantages + batch_value

        for _ in range(self.update_steps):
            # Actor update: calculate ratio (pi / pi_old)
            pi_mu, pi_std = self.actor_network(batch_obs)
            dist = torch.distributions.Normal(pi_mu, pi_std)
            log_prob = dist.log_prob(batch_action).sum(-1)
            ratio = torch.exp(log_prob - batch_log_prob)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Add entropy to encourage exploration
            entropy = dist.entropy().mean()
            actor_loss -= self.ent_coef * entropy

            # Critic update
            q_value = self.critic_network(batch_obs)
            critic_loss = (returns - q_value).pow(2).mean()

            # Total loss
            total_loss = actor_loss  + self.value_loss_coef * critic_loss

            # Update the network
            self.actor_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
            self.critic_optim.step()

        # Clear the memory after each update
        self.buffer.initial_buffer()

    def choose_action(self, observation):
        # Choose action based on actor network
        inputs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        pi_mu, pi_std = self.actor_network(inputs)
        dist = torch.distributions.Normal(pi_mu, pi_std)
        action = dist.sample()[0].cpu().detach().numpy()

        return action.tolist()

    def save_models(self):
        self.actor_network.save_checkpoint()
        self.critic_network.save_checkpoint()

    def load_models(self):
        self.actor_network.load_checkpoint()
        self.critic_network.load_checkpoint()

