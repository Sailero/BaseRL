import torch
import torch.nn as nn
import numpy as np
from agent.imitation_learning.im_replay_buffer import ImBuffer


class GAIL:
    def __init__(self, config):
        """
        Initialize the GAIL (Generative Adversarial Imitation Learning) class with specified configurations.

        Args:
            config: Configuration object containing model, environment, and training parameters.
        """
        # Set device for training
        self.device = config.device.device

        # Initialize imitation buffer
        self.im_buffer = ImBuffer(config)

        # Initialize the agent based on policy type
        if config.policy_type == "GAIL_PPO":
            from agent.imitation_learning.GAIL.gail_ppo import GailPPO
            self.agent = GailPPO(config)
        elif config.policy_type == "GAIL_SAC":
            from agent.imitation_learning.GAIL.gail_sac_2q import GailSAC
            self.agent = GailSAC(config)
        else:
            raise ValueError(f"{config.policy_type} is not supported.")

        # Choose discriminator type based on observation dimension
        if len(config.env.agent_obs_dim) == 1:
            from agent.imitation_learning.GAIL.gail_discr import Discriminator
        else:
            from agent.imitation_learning.GAIL.gail_discr import Discriminator2d as Discriminator
        self.discr_net = Discriminator(config, 'discriminator').to(self.device)

        # Set up optimizer for the discriminator
        self.discr_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.discr_net.parameters()),
            lr=config.params["lr_discr"]
        )

        # Track training process
        self.episode_num = 0
        self.name = config.policy_type

    def save_models(self):
        """Save checkpoints for discriminator and agent models."""
        self.discr_net.save_checkpoint()
        self.agent.save_models()

    def load_models(self):
        """Load checkpoints for discriminator and agent models."""
        self.discr_net.load_checkpoint()
        self.agent.load_models()

    def choose_action(self, observation):
        """
        Select an action based on the current policy.

        Args:
            observation: Current observation of the environment.

        Returns:
            The action selected by the agent.
        """
        return self.agent.choose_action(observation)

    def train(self, transitions):
        """
        Train the discriminator and the agent based on transition data.

        Args:
            transitions: Dictionary containing transition data from the environment.
        """
        # Gather agent and expert data
        agent_data = transitions
        expert_data = self.im_buffer.sample(im_sample_size=len(agent_data['obs']))

        # Convert data to torch tensors
        expert_obs = torch.tensor(expert_data['obs'], dtype=torch.float32).to(self.device)
        expert_action = torch.tensor(expert_data['action'], dtype=torch.float32).to(self.device)
        agent_obs = torch.tensor(np.array(agent_data['obs']), dtype=torch.float32).to(self.device)
        agent_actions = torch.tensor(np.array(agent_data['action']), dtype=torch.float32).to(self.device)

        # Compute discriminator probabilities for expert and agent
        expert_prob = self.discr_net(expert_obs, expert_action)
        agent_prob = self.discr_net(agent_obs, agent_actions)

        # Calculate discriminator loss using binary cross-entropy
        expert_loss = nn.BCELoss()(expert_prob, torch.zeros_like(expert_prob))
        agent_loss = nn.BCELoss()(agent_prob, torch.ones_like(agent_prob))
        discriminator_loss = agent_loss + expert_loss

        # Backpropagate and update discriminator
        self.discr_optim.zero_grad()
        discriminator_loss.backward()
        self.discr_optim.step()

        # Compute rewards for the agent based on discriminator output
        expert_rewards = -torch.log(agent_prob).detach().cpu().numpy()

        # Prepare new transitions for the agent's training
        new_transitions = {
            'obs': agent_obs,
            'action': agent_actions,
            'expert_reward': expert_rewards,
            'reward': agent_data['reward'],
            'next_obs': agent_data['next_obs'],
            'done': agent_data['done']
        }

        # Train the agent with updated transitions
        self.agent.train(new_transitions)
        self.episode_num += 1
