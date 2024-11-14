import torch
import numpy as np


class DDPG:
    name = "DDPG"

    def __init__(self, config):
        # Read the training parameters from args
        self.action_dim = config.env.agent_action_dim
        self.device = config.device.device

        # Parameters of DDPG
        self.noise_rate = config.params["noise_rate"] / 2 if config.task_type == "evaluate" else config.params["noise_rate"]
        self.epsilon = 0 if config.task_type == "evaluate" else config.params["epsilon"]
        self.tau = config.params["tau"]
        self.gamma = config.params["gamma"]
        self.max_grad_norm = config.params["max_grad_norm"]

        # import network
        if len(config.env.agent_obs_dim) == 1:
            from agent.off_policy.DDPG.ddpg_actor_critic import DeterministicActor as Actor, \
                DeterministicCritic as Critic
        else:
            from agent.off_policy.DDPG.ddpg_actor_critic import DeterministicActor2d as Actor, \
                DeterministicCritic2d as Critic

        # create the network
        self.actor_network = Actor(config, 'actor').to(self.device)
        self.critic_network = Critic(config, 'critic').to(self.device)

        # build up the target network
        self.actor_target_network = Actor(config, 'target_actor').to(self.device)
        self.critic_target_network = Critic(config, 'target_critic').to(self.device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=config.params["lr_actor"])
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=config.params["lr_critic"])


    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)


    # update the network
    def train(self, transitions):
        # 将所有的转移迁移到GPU
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(self.device)
        batch_reward = transitions['reward']
        batch_done = transitions['done']
        batch_obs = transitions['obs']
        batch_action = transitions['action']
        batch_next_obs = transitions['next_obs']

        # calculate the target Q value function
        batch_pred_next_action = self.actor_target_network(batch_next_obs)
        with torch.no_grad():
            q_next = self.critic_target_network(batch_next_obs, batch_pred_next_action).detach()
            target_q = (batch_reward + self.gamma * q_next * (1 - batch_done)).detach()

        # the critic loss
        q_value = self.critic_network(batch_obs, batch_action)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        batch_online_action = self.actor_network(batch_obs)
        actor_loss = - (self.critic_network(batch_obs, batch_online_action) - q_value.detach()).mean()

        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        self._soft_update_target_network()

    def choose_action(self, observation):
        # Choose action based on actor network
        if np.random.uniform() < self.epsilon:
            action = np.random.uniform(-1, 1, self.action_dim)
        else:
            # 确保观测数据也迁移到 GPU
            inputs = observation.unsqueeze(0)
            pi = self.actor_network(inputs).squeeze(0)
            action = pi.cpu().detach().numpy()

            # gaussian noise
            noise = self.noise_rate * np.random.randn(*action.shape)
            action += noise
            action = np.clip(action, -1, 1)
        return action.tolist()

    def save_models(self):
        self.actor_network.save_checkpoint()
        self.actor_target_network.save_checkpoint()
        self.critic_network.save_checkpoint()
        self.critic_target_network.save_checkpoint()

    def load_models(self):
        self.actor_network.load_checkpoint()
        self.actor_target_network.load_checkpoint()
        self.critic_network.load_checkpoint()
        self.critic_target_network.load_checkpoint()
