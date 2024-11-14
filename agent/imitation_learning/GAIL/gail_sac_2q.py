import numpy as np
import os
import torch
import torch.nn.functional as F


class GailSAC:
    name = 'GAIL_SAC_2q'

    def __init__(self, config):
        """
        Initialize the GAIL-SAC agent with configuration parameters.

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
        self.critic_1 = CriticSAC(config, 'critic_1').to(self.device)
        self.critic_2 = CriticSAC(config, 'critic_2').to(self.device)
        self.critic_target_1 = CriticSAC(config, 'critic_target_1').to(self.device)
        self.critic_target_2 = CriticSAC(config, 'critic_target_2').to(self.device)

        # Initialize expert critic networks for imitation learning
        self.expert_critic_1 = CriticSAC(config, 'expert_critic_1').to(self.device)
        self.expert_critic_2 = CriticSAC(config, 'expert_critic_2').to(self.device)
        self.expert_critic_target_1 = CriticSAC(config, 'expert_critic_target_1').to(self.device)
        self.expert_critic_target_2 = CriticSAC(config, 'expert_critic_target_2').to(self.device)

        # Synchronize target networks with the primary networks
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.expert_critic_target_1.load_state_dict(self.expert_critic_1.state_dict())
        self.expert_critic_target_2.load_state_dict(self.expert_critic_2.state_dict())

        # Initialize optimizers for actor, critics, and expert critics
        self.actor_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor_net.parameters()),
                                            lr=config.params["lr_actor"])
        self.critic_1_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_1.parameters()),
                                               lr=config.params["lr_critic"])
        self.critic_2_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic_2.parameters()),
                                               lr=config.params["lr_critic"])
        self.expert_critic_1_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.expert_critic_1.parameters()),
            lr=config.params["lr_critic"])
        self.expert_critic_2_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.expert_critic_2.parameters()),
            lr=config.params["lr_critic"])

        # Initialize temperature parameter (alpha) for entropy regularization
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
        """Save checkpoints for actor, critics, and expert critic networks, along with alpha_log."""
        self.actor_net.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.critic_target_1.save_checkpoint()
        self.critic_target_2.save_checkpoint()
        self.expert_critic_1.save_checkpoint()
        self.expert_critic_2.save_checkpoint()
        self.expert_critic_target_1.save_checkpoint()
        self.expert_critic_target_2.save_checkpoint()

        # Save alpha log value
        torch.save(self.alpha_log.cpu(), self.alpha_log_save_path)

    def load_models(self):
        """Load checkpoints for actor, critics, and expert critic networks, along with alpha_log."""
        self.actor_net.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.critic_target_1.load_checkpoint()
        self.critic_target_2.load_checkpoint()
        self.expert_critic_1.load_checkpoint()
        self.expert_critic_2.load_checkpoint()
        self.expert_critic_target_1.load_checkpoint()
        self.expert_critic_target_2.load_checkpoint()

        # Load alpha log value
        alpha_log = torch.load(self.alpha_log_save_path, weights_only=True)
        self.alpha_log = torch.tensor(alpha_log.item(), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.alpha_log], lr=self.lr_alpha)

    def calc_target(self, rewards, next_states, dones, critic_target_1, critic_target_2):
        """
        Calculate target Q values for SAC using next states and rewards.

        Args:
            rewards: Reward values.
            next_states: Next state inputs.
            dones: Done flags indicating episode end.
            critic_target_1: First target critic network.
            critic_target_2: Second target critic network.

        Returns:
            Calculated TD target values.
        """
        next_actions, log_prob = self.actor_net(next_states)
        q1_value = critic_target_1(next_states, next_actions)
        q2_value = critic_target_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) - self.alpha_log.exp() * log_prob.unsqueeze(1)
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        """
        Perform soft update of target network parameters.

        Args:
            net: Primary network.
            target_net: Target network to be updated.
        """
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self, transitions):
        """
        Train SAC networks using transitions from the environment.

        Args:
            transitions: Dictionary containing transition data.
        """
        # Convert transition data to tensors if necessary
        for key in transitions.keys():
            if not isinstance(transitions[key], torch.Tensor):
                transitions[key] = torch.tensor(np.array(transitions[key]), dtype=torch.float32).to(self.device)

        # Extract transition data
        trans_obs = transitions['obs']
        trans_action = transitions['action']
        trans_reward = transitions['reward']
        trans_expert_reward = transitions['expert_reward']
        trans_next_obs = transitions['next_obs']
        trans_done = transitions['done']

        # Calculate TD targets for critic and expert critic networks
        td_target = self.calc_target(trans_reward, trans_next_obs, trans_done, self.critic_target_1,
                                     self.critic_target_2)
        expert_td_target = self.calc_target(trans_expert_reward, trans_next_obs, trans_done,
                                            self.expert_critic_target_1, self.expert_critic_target_2)

        # Update critic networks
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

        # Update expert critic networks
        expert_critic_1_loss = F.mse_loss(self.expert_critic_1(trans_obs, trans_action), expert_td_target.detach())
        expert_critic_2_loss = F.mse_loss(self.expert_critic_2(trans_obs, trans_action), expert_td_target.detach())

        self.expert_critic_1_optim.zero_grad()
        self.expert_critic_2_optim.zero_grad()
        expert_critic_1_loss.backward()
        expert_critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.expert_critic_1.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.expert_critic_2.parameters(), self.max_grad_norm)
        self.expert_critic_1_optim.step()
        self.expert_critic_2_optim.step()

        # Update temperature parameter (alpha)
        new_actions, log_prob = self.actor_net(trans_obs)
        alpha_loss = (self.target_entropy - log_prob.unsqueeze(1)).detach() * self.alpha_log.exp()
        self.alpha_optim.zero_grad()
        alpha_loss.mean().backward()
        torch.nn.utils.clip_grad_norm_([self.alpha_log], self.max_grad_norm)
        self.alpha_optim.step()

        # Update actor network
        q1_value = self.expert_critic_1(trans_obs, new_actions)
        q2_value = self.expert_critic_2(trans_obs, new_actions)
        actor_loss = -torch.mean(torch.min(q1_value, q2_value) - self.alpha_log.exp() * log_prob.unsqueeze(1))

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        # Perform soft updates on all target networks
        self.soft_update(self.critic_1, self.critic_target_1)
        self.soft_update(self.critic_2, self.critic_target_2)
        self.soft_update(self.expert_critic_1, self.expert_critic_target_1)
        self.soft_update(self.expert_critic_2, self.expert_critic_target_2)
