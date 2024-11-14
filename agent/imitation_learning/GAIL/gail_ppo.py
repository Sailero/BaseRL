import numpy as np
import torch

class GailPPO:
    name = 'GailPPO'

    def __init__(self, config):
        """
        Initialize the GAIL-PPO agent with the specified configuration.

        Args:
            config: Configuration object containing environment, device, and training parameters.
        """
        # Action dimension and device information
        self.action_dim = config.env.agent_action_dim
        self.device = config.device.device

        # PPO parameters
        self.batch_size = config.params["batch_size"]
        self.max_grad_norm = config.params["max_grad_norm"]
        self.gamma = config.params["gamma"]
        self.lam = config.params["lam"]
        self.eps_clip = config.params["eps_clip"]
        self.update_nums = config.params["update_nums"]
        self.ent_coef = config.params["ent_coef"]

        # Import network modules based on observation dimension
        if len(config.env.agent_obs_dim) == 1:
            from agent.on_policy.PPO.ppo_actor_critic import StochasticActor as Actor, StochasticCritic as Critic
        else:
            from agent.on_policy.PPO.ppo_actor_critic import StochasticActor2d as Actor, StochasticCritic2d as Critic

        # Initialize networks for actor, old actor (for PPO), critic, and expert critic
        self.actor_net = Actor(config, 'actor').to(self.device)
        self.old_actor_net = Actor(config, 'old_actor').to(self.device)
        self.critic_net = Critic(config, 'critic').to(self.device)
        self.expert_critic_net = Critic(config, 'expert_critic').to(self.device)

        # Synchronize the old actor network with the current actor
        self.old_actor_net.load_state_dict(self.actor_net.state_dict())

        # Optimizers for the actor and critic networks
        self.actor_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.actor_net.parameters()),
            lr=config.params["lr_actor"]
        )
        self.critic_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.critic_net.parameters()),
            lr=config.params["lr_critic"]
        )
        self.expert_critic_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.expert_critic_net.parameters()),
            lr=config.params["lr_critic"]
        )

    def compute_gae(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation (GAE) for PPO.

        Args:
            rewards: List of rewards from the environment.
            values: List of values predicted by the critic.
            next_values: List of next state values.
            dones: List of done flags indicating episode end.

        Returns:
            Computed advantages.
        """
        advantages = np.zeros(len(values))
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - int(dones[t])) - values[t]
            advantage = delta + self.gamma * self.lam * advantage * (1 - int(dones[t]))
            advantages[t] = advantage
        return advantages

    def train(self, transitions):
        """
        Train the PPO network using transitions from the environment.

        Args:
            transitions: Dictionary containing transitions data.
        """
        # Convert data to tensors if not already
        for key in transitions.keys():
            if not isinstance(transitions[key], torch.Tensor):
                transitions[key] = torch.tensor(np.array(transitions[key]), dtype=torch.float32).to(self.device)

        trans_obs = transitions['obs']
        trans_action = transitions['action']
        trans_reward = transitions['reward']
        trans_next_obs = transitions['next_obs']
        trans_done = transitions['done']
        trans_len = len(trans_obs)

        # Train the expert critic network if imitation learning is involved
        train_expert_reward = transitions["expert_reward"]
        train_expert_reward = (train_expert_reward - train_expert_reward.mean()) / (train_expert_reward.std() + 1e-6)
        with torch.no_grad():
            trans_expert_value = self.expert_critic_net(trans_obs)
            trans_expert_next_value = self.expert_critic_net(trans_next_obs)
            expert_advantages = self.compute_gae(
                train_expert_reward.cpu().numpy(),
                trans_expert_value.cpu().numpy(),
                trans_expert_next_value.cpu().numpy(),
                trans_done.cpu().numpy()
            )
            expert_advantages = torch.tensor(expert_advantages, dtype=torch.float32).reshape(
                trans_expert_value.shape).to(self.device)
            expert_returns = expert_advantages + trans_expert_value

        # Scale rewards for stability
        trans_reward = (trans_reward - trans_reward.mean()) / (trans_reward.std() + 1e-6)

        # Synchronize the old actor network with the current actor network
        self.old_actor_net.load_state_dict(self.actor_net.state_dict())

        # Calculate target values and advantages
        with torch.no_grad():
            tran_value = self.critic_net(trans_obs)
            tran_next_value = self.critic_net(trans_next_obs)
            advantages = self.compute_gae(trans_reward, tran_value, tran_next_value, trans_done)
            advantages = torch.tensor(advantages, dtype=torch.float32).reshape(tran_next_value.shape).to(self.device)
            returns = advantages + tran_value

        # Perform PPO update
        for _ in range(self.update_nums):
            batch_start_points = np.arange(0, trans_len, self.batch_size)
            buffer_indices = np.arange(trans_len)
            np.random.shuffle(buffer_indices)
            batch_id_set = [buffer_indices[i: i + self.batch_size] for i in batch_start_points]
            for batch_ids in batch_id_set:
                batch_obs = trans_obs[batch_ids]
                batch_action = trans_action[batch_ids]
                batch_returns = returns[batch_ids]
                batch_expert_advantages = expert_advantages[batch_ids]
                batch_expert_returns = expert_returns[batch_ids]

                # Update expert critic network
                expert_critic_loss = (batch_expert_returns - self.expert_critic_net(batch_obs)).pow(2).mean()
                self.expert_critic_optim.zero_grad()
                expert_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.expert_critic_net.parameters(), self.max_grad_norm)
                self.expert_critic_optim.step()

                # Compute old and new log probabilities for actions
                with torch.no_grad():
                    old_pi_mu, old_pi_sigma = self.old_actor_net(batch_obs)
                    old_pi = torch.distributions.Normal(old_pi_mu, old_pi_sigma)
                    batch_old_log_prob = old_pi.log_prob(batch_action).sum(-1, keepdim=True)

                pi_mu, pi_std = self.actor_net(batch_obs)
                dist = torch.distributions.Normal(pi_mu, pi_std)
                batch_new_log_prob = dist.log_prob(batch_action).sum(-1, keepdim=True)

                # PPO objective
                ratio = torch.exp(batch_new_log_prob - batch_old_log_prob)
                surr1 = ratio * batch_expert_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_expert_advantages
                entropy = dist.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

                # Critic loss for value function update
                gae_value = self.critic_net(batch_obs)
                critic_loss = (batch_returns - gae_value).pow(2).mean()

                # Update critic and actor networks
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optim.step()

    def choose_action(self, observation):
        """
        Choose an action based on the current actor network's policy.

        Args:
            observation: Current observation from the environment.

        Returns:
            Selected action as a list.
        """
        inputs = observation.clone().detach().unsqueeze(0).to(self.device)
        with torch.no_grad():
            pi_mu, pi_std = self.actor_net(inputs)
            pi_mu, pi_std = pi_mu.reshape(-1), pi_std.reshape(-1)
        dist = torch.distributions.Normal(pi_mu, pi_std)
        action = dist.sample().cpu().detach().numpy()
        action = np.clip(action, -1, 1)
        return action.tolist()

    def save_models(self):
        """Save checkpoints for the actor, critic, and expert critic networks."""
        self.actor_net.save_checkpoint()
        self.critic_net.save_checkpoint()
        self.expert_critic_net.save_checkpoint()

    def load_models(self):
        """Load checkpoints for the actor, critic, and expert critic networks."""
        self.actor_net.load_checkpoint()
        self.critic_net.load_checkpoint()
        self.expert_critic_net.load_checkpoint()
