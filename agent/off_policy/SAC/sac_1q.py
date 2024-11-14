import numpy as np
import os
import torch
import torch.nn.functional as F


class SAC:
    name = 'SAC'

    def __init__(self, config):
        """
        Initialize the Soft Actor-Critic (SAC) agent with specified configurations.

        Args:
            config: Configuration object containing model, environment, and training parameters.
        """
        self.device = config.device.device
        self.alpha_log_save_path = os.path.join(config.save_path, 'parameters/alpha_log.pth')

        # SAC hyperparameters
        self.gamma = config.params["gamma"]
        self.tau = config.params["tau"]
        self.lr_alpha = config.params["lr_alpha"]
        self.max_grad_norm = config.params["max_grad_norm"]

        # Import appropriate actor and critic networks based on observation dimension
        if len(config.env.agent_obs_dim) == 1:
            from agent.off_policy.SAC.sac_actor_critic import StochasticActor as ActorSAC, StochasticCritic as CriticSAC
        else:
            from agent.off_policy.SAC.sac_actor_critic import StochasticActor2d as ActorSAC, \
                StochasticCritic2d as CriticSAC

        # Initialize actor and critic networks
        self.actor_net = ActorSAC(config, 'actor').to(self.device)
        self.critic_net = CriticSAC(config, 'critic').to(self.device)
        self.critic_target = CriticSAC(config, 'critic_target').to(self.device)
        self.critic_target.load_state_dict(self.critic_net.state_dict())

        # Optimizers for actor, critic, and temperature parameter (alpha)
        self.actor_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_net.parameters()),
                                            lr=config.params["lr_actor"])
        self.critic_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_net.parameters()),
                                             lr=config.params["lr_critic"])

        # Temperature parameter for entropy regularization
        self.alpha_log = torch.tensor(np.log(0.01), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.alpha_log], lr=self.lr_alpha)
        self.target_entropy = config.env.agent_action_dim

    def choose_action(self, observation):
        """
        Select an action based on the current policy.

        Args:
            observation: Current observation from the environment.

        Returns:
            Selected action as a list, clipped between -1 and 1.
        """
        inputs = observation.clone().detach().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor_net(inputs)
        action = np.clip(action[0].cpu().detach().numpy(), -1, 1)
        return action.tolist()

    def save_models(self):
        """Save checkpoints for actor, critic, target critic networks, and alpha_log."""
        self.actor_net.save_checkpoint()
        self.critic_net.save_checkpoint()
        self.critic_target.save_checkpoint()
        torch.save(self.alpha_log.cpu(), self.alpha_log_save_path)

    def load_models(self):
        """Load checkpoints for actor, critic, target critic networks, and alpha_log."""
        self.actor_net.load_checkpoint()
        self.critic_net.load_checkpoint()
        self.critic_target.load_checkpoint()
        alpha_log = torch.load(self.alpha_log_save_path, weights_only=True)
        self.alpha_log = torch.tensor(alpha_log.item(), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.alpha_log], lr=self.lr_alpha)

    def calc_target(self, rewards, next_states, dones):
        """
        Calculate target Q-values for SAC using next states and rewards.

        Args:
            rewards: Reward tensor.
            next_states: Next state tensor.
            dones: Done flags indicating episode end.

        Returns:
            Calculated TD target values.
        """
        next_actions, next_log_prob = self.actor_net(next_states)
        next_q = self.critic_target(next_states, next_actions)
        td_target = rewards + self.gamma * (next_q - self.alpha_log.exp() * next_log_prob.unsqueeze(-1)) * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        """
        Perform soft update of target network parameters.

        Args:
            net: Primary network.
            target_net: Target network to be updated.
        """
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_((1.0 - self.tau) * param_target.data + self.tau * param.data)

    def train(self, transitions):
        """
        Train the SAC agent's networks using transitions from the environment.

        Args:
            transitions: Dictionary containing batch transition data.
        """
        # Transfer transition data to GPU
        for key in transitions.keys():
            if not isinstance(transitions[key], torch.Tensor):
                transitions[key] = torch.tensor(np.array(transitions[key]), dtype=torch.float32).to(self.device)

        trans_obs = transitions['obs']
        trans_action = transitions['action']
        trans_reward = transitions['reward']
        trans_next_obs = transitions['next_obs']
        trans_done = transitions['done']

        # Update Q-network
        td_target = self.calc_target(trans_reward, trans_next_obs, trans_done)
        q_values = self.critic_net(trans_obs, trans_action)
        critic_loss = F.mse_loss(q_values, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        # Soft update the target critic network
        self.soft_update(self.critic_net, self.critic_target)

        # Update actor network
        action_pg, log_prob = self.actor_net(trans_obs)
        log_prob = log_prob.unsqueeze(1)

        # Update alpha (temperature) to control entropy regularization
        alpha_loss = torch.mean((self.target_entropy - log_prob).detach() * self.alpha_log.exp())
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.alpha_log], self.max_grad_norm)
        self.alpha_optim.step()

        # Policy loss for actor network
        q_value_pg = self.critic_net(trans_obs, action_pg)
        actor_loss = torch.mean(self.alpha_log.exp() * log_prob - q_value_pg)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optim.step()
