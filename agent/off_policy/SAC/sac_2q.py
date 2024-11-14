import numpy as np
import os
import torch
import torch.nn.functional as F

class SAC:
    name = 'SAC_Dual_Q'

    def __init__(self, config):
        """
        Initialize the Soft Actor-Critic (SAC) agent with dual Q-networks for stability.

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
            from agent.off_policy.SAC.sac_actor_critic import StochasticActor2d as ActorSAC, StochasticCritic2d as CriticSAC

        # Initialize actor and dual critic networks
        self.actor_net = ActorSAC(config, 'actor').to(self.device)
        self.critic_1 = CriticSAC(config, 'critic_1').to(self.device)
        self.critic_2 = CriticSAC(config, 'critic_2').to(self.device)
        self.critic_target_1 = CriticSAC(config, 'critic_target_1').to(self.device)
        self.critic_target_2 = CriticSAC(config, 'critic_target_2').to(self.device)

        # Synchronize target networks with the main critic networks
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers for actor, dual critics, and temperature parameter (alpha)
        self.actor_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_net.parameters()),
                                            lr=config.params["lr_actor"])
        self.critic_1_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_1.parameters()),
                                               lr=config.params["lr_critic"])
        self.critic_2_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_2.parameters()),
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
        """Save checkpoints for actor, dual critics, target critics, and alpha_log."""
        self.actor_net.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.critic_target_1.save_checkpoint()
        self.critic_target_2.save_checkpoint()
        torch.save(self.alpha_log.cpu(), self.alpha_log_save_path)

    def load_models(self):
        """Load checkpoints for actor, dual critics, target critics, and alpha_log."""
        self.actor_net.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.critic_target_1.load_checkpoint()
        self.critic_target_2.load_checkpoint()
        alpha_log = torch.load(self.alpha_log_save_path, weights_only=True)
        self.alpha_log = torch.tensor(alpha_log.item(), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.alpha_log], lr=self.lr_alpha)

    def calc_target(self, rewards, next_states, dones):
        """
        Calculate target Q-values using next states and rewards for SAC.

        Args:
            rewards: Reward tensor.
            next_states: Next state tensor.
            dones: Done flags indicating episode end.

        Returns:
            Calculated TD target values.
        """
        next_actions, log_prob = self.actor_net(next_states)
        q1_value = self.critic_target_1(next_states, next_actions)
        q2_value = self.critic_target_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) - self.alpha_log.exp() * log_prob.unsqueeze(1)
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        """
        Perform a soft update of target network parameters.

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

        # Update Q-networks
        td_target = self.calc_target(trans_reward, trans_next_obs, trans_done)
        critic_1_loss = F.mse_loss(self.critic_1(trans_obs, trans_action), td_target.detach())
        critic_2_loss = F.mse_loss(self.critic_2(trans_obs, trans_action), td_target.detach())

        self.critic_1_optim.zero_grad()
        self.critic_2_optim.zero_grad()
        critic_1_loss.backward()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_1_optim.step()
        self.critic_2_optim.step()

        # Update actor network
        new_actions, log_prob = self.actor_net(trans_obs)
        entropy = log_prob.unsqueeze(1)

        # Update alpha (temperature) parameter
        alpha_loss = torch.mean((self.target_entropy - entropy).detach() * self.alpha_log.exp())
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.alpha_log], self.max_grad_norm)
        self.alpha_optim.step()

        # Actor policy loss
        q1_value = self.critic_1(trans_obs, new_actions)
        q2_value = self.critic_2(trans_obs, new_actions)
        actor_loss = -torch.mean(torch.min(q1_value, q2_value) - self.alpha_log.exp() * entropy)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        # Soft update target critic networks
        self.soft_update(self.critic_1, self.critic_target_1)
        self.soft_update(self.critic_2, self.critic_target_2)
