import numpy as np
import torch

class PPO:
    name = 'PPO'

    def __init__(self, config):
        """
        Initialize the Proximal Policy Optimization (PPO) agent with specified configurations.

        Args:
            config: Configuration object containing model, environment, and training parameters.
        """
        self.action_dim = config.env.agent_action_dim
        self.device = config.device.device

        # PPO hyperparameters
        self.batch_size = config.params["batch_size"]
        self.max_grad_norm = config.params["max_grad_norm"]
        self.gamma = config.params["gamma"]
        self.lam = config.params["lam"]
        self.eps_clip = config.params["eps_clip"]
        self.update_nums = config.params["update_nums"]
        self.ent_coef = config.params["ent_coef"]

        # Import appropriate actor and critic networks based on observation dimension
        if len(config.env.agent_obs_dim) == 1:
            from agent.on_policy.PPO.ppo_actor_critic import StochasticActor as Actor, StochasticCritic as Critic
        else:
            from agent.on_policy.PPO.ppo_actor_critic import StochasticActor2d as Actor, StochasticCritic2d as Critic

        # Initialize actor and critic networks
        self.actor_net = Actor(config, 'actor').to(self.device)
        self.old_actor_net = Actor(config, 'actor').to(self.device)
        self.critic_net = Critic(config, 'critic').to(self.device)

        # Synchronize old actor with the current actor
        self.old_actor_net.load_state_dict(self.actor_net.state_dict())

        # Optimizers for actor and critic networks
        self.actor_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_net.parameters()),
                                            lr=config.params["lr_actor"])
        self.critic_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_net.parameters()),
                                             lr=config.params["lr_critic"])

    def compute_gae(self, rewards, values, next_values, dones):
        """
        Calculate Generalized Advantage Estimation (GAE) for advantage computation.

        Args:
            rewards: Rewards obtained from the environment.
            values: Estimated values of current states.
            next_values: Estimated values of next states.
            dones: Flags indicating episode termination.

        Returns:
            advantages: Computed advantages for each timestep.
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
        Train the PPO agent's networks using transitions from the environment.

        Args:
            transitions: Dictionary containing batch transition data.
        """
        # Move transition data to GPU if not already in tensor format
        for key in transitions.keys():
            if not isinstance(transitions[key], torch.Tensor):
                transitions[key] = torch.tensor(np.array(transitions[key]), dtype=torch.float32).to(self.device)

        trans_obs = transitions['obs']
        trans_action = transitions['action']
        trans_reward = transitions['reward']
        trans_next_obs = transitions['next_obs']
        trans_done = transitions['done']
        trans_len = len(trans_obs)

        # Normalize rewards
        trans_reward = (trans_reward - trans_reward.mean()) / (trans_reward.std() + 1e-6)

        # Update old actor with current actor parameters
        self.old_actor_net.load_state_dict(self.actor_net.state_dict())

        with torch.no_grad():
            tran_value = self.critic_net(trans_obs)
            tran_next_value = self.critic_net(trans_next_obs)

        # Calculate target values and advantages
        advantages = self.compute_gae(trans_reward, tran_value, tran_next_value, trans_done)
        advantages = torch.tensor(advantages, dtype=torch.float32).reshape(tran_next_value.shape).to(self.device)
        returns = advantages + tran_value

        for _ in range(self.update_nums):
            # Shuffle and batch data
            batch_start_points = np.arange(0, trans_len, self.batch_size)
            buffer_indices = np.arange(trans_len)
            np.random.shuffle(buffer_indices)
            batch_id_set = [buffer_indices[i: i + self.batch_size] for i in batch_start_points]

            for batch_ids in batch_id_set:
                batch_obs = trans_obs[batch_ids]
                batch_action = trans_action[batch_ids]
                batch_advantages = advantages[batch_ids]
                batch_returns = returns[batch_ids]

                # Calculate log probabilities for old policy
                with torch.no_grad():
                    old_pi_mu, old_pi_sigma = self.old_actor_net(batch_obs)
                    old_pi = torch.distributions.Normal(old_pi_mu, old_pi_sigma)
                    batch_old_log_prob = old_pi.log_prob(batch_action).sum(-1, keepdim=True)

                # Calculate log probabilities for current policy
                pi_mu, pi_std = self.actor_net(batch_obs)
                dist = torch.distributions.Normal(pi_mu, pi_std)
                batch_new_log_prob = dist.log_prob(batch_action).sum(-1, keepdim=True)

                # PPO loss with clipping
                ratio = torch.exp(batch_new_log_prob - batch_old_log_prob)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * dist.entropy().mean()

                # Critic loss
                gae_value = self.critic_net(batch_obs)
                critic_loss = (batch_returns - gae_value).pow(2).mean()

                # Update networks
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
        Choose an action based on the current policy.

        Args:
            observation: Current observation from the environment.

        Returns:
            Selected action as a list, clipped between -1 and 1.
        """
        inputs = observation.clone().detach().unsqueeze(0).to(self.device)
        with torch.no_grad():
            pi_mu, pi_std = self.actor_net(inputs)
            pi_mu, pi_std = pi_mu.reshape(-1), pi_std.reshape(-1)
        dist = torch.distributions.Normal(pi_mu, pi_std)
        action = dist.sample()
        action = np.clip(action.cpu().detach().numpy(), -1, 1)
        return action.tolist()

    def save_models(self):
        """Save checkpoints for the actor and critic networks."""
        self.actor_net.save_checkpoint()
        self.critic_net.save_checkpoint()

    def load_models(self):
        """Load checkpoints for the actor and critic networks."""
        self.actor_net.load_checkpoint()
        self.critic_net.load_checkpoint()
