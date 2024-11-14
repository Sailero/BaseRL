import gym
import numpy as np


class GymEnv:
    """
    Class to create and interact with a Gym environment.
    Supports continuous control environments such as MountainCarContinuous and Pendulum.
    """

    def __init__(self, config):
        """
        Initializes the environment based on the given configuration.

        Args:
            config: Configuration object containing environment settings.
        """
        # Check if the environment is supported
        if config.env.name in ['MountainCarContinuous-v0', 'Pendulum-v1']:
            self.scenario_name = config.env.name

            # Initialize the environment for evaluation or training
            if config.task_type == 'evaluate':
                self.env = gym.make(config.env.name, render_mode='human')
            else:
                self.env = gym.make(config.env.name)

            # Get observation space dimensions
            self.agent_obs_dim = list(self.env.observation_space.shape)

            # Get action space dimensions
            self.agent_action_dim = self.env.action_space.shape[0]
            self.action_low = self.env.action_space.low  # Minimum action value
            self.action_high = self.env.action_space.high  # Maximum action value

            # Get environment's max episode length
            self.max_episode_len = self.env.spec.max_episode_steps
        else:
            raise ValueError(f"{config.env.name} not supported")

    def reset(self):
        """
        Resets the environment to an initial state.

        Returns:
            The initial observation from the environment.
        """
        obs = self.env.reset()
        return obs[0]

    def step(self, action):
        """
        Takes a step in the environment based on the provided action.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing the next observation, reward, done flag, and additional info.
        """
        # Denormalize action to match the action space range
        action = action * self.action_high

        # Take a step in the environment
        obs_, reward, done, info1, info2 = self.env.step(action)
        if self.scenario_name == "MountainCarContinuous-v0":
            reward = reward + np.sum(action) ** 2 * 0.1
        return obs_, reward, done, [info1, info2]

    def render(self):
        """
        Renders the environment for visualization.
        """
        self.env.render()


class MpeEnv:
    """
    Class to create and interact with a Multi-Agent Particle Environment (MPE).
    """

    def __init__(self, config):
        """
        Initializes the MPE environment based on the given configuration.

        Args:
            config: Configuration object containing environment settings.
        """
        if config.env.name in ['simple']:
            from env.mpe.make_env import make_env
            self.env = make_env(config.env.name)

            # Get observation space dimensions for the first agent
            self.agent_obs_dim = list(self.env.observation_space[0].shape)

            # Get action space dimensions for the first agent
            self.agent_action_dim = self.env.action_space[0].n
            self.action_high = 1
            self.action_low = -1

            # Get the max episode length of the environment
            self.max_episode_len = 25
        else:
            raise ValueError(f"{config.env.name} not supported")

    def reset(self):
        """
        Resets the environment to an initial state.

        Returns:
            The initial observation from the environment.
        """
        obs = self.env.reset()
        return obs[0]

    def step(self, action):
        """
        Takes a step in the environment based on the provided action.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing the next observation, reward, done flag, and additional info.
        """
        # Actions need to be provided for each agent, here only the first agent is considered.
        action_n = [action]

        # Take a step in the environment
        obs_, reward, done, info = self.env.step(action_n)
        return obs_[0], reward[0], done[0], info

    def render(self):
        """
        Renders the environment for visualization.
        """
        self.env.render()


class Env:
    """
    Class to create and manage different types of environments (Gym, MPE).
    """

    def __init__(self, config):
        """
        Initializes the environment based on the provided configuration.

        Args:
            config: Configuration object containing environment settings.
        """
        # Create Gym environment if the name matches
        if config.env.name in ['MountainCarContinuous-v0', 'Pendulum-v1']:
            self.env = GymEnv(config)

        # Create MPE environment if the name matches
        elif config.env.name in ['simple']:
            from env.mpe.make_env import make_env
            self.env = MpeEnv(config)

        # Raise error for unsupported environments
        else:
            raise ValueError(f"{config.env.name} not supported")

        # Get observation space dimensions
        self.agent_obs_dim = self.env.agent_obs_dim

        # Get action space dimensions
        self.agent_action_dim = self.env.agent_action_dim
        self.action_low = self.env.action_low
        self.action_high = self.env.action_high

        # Get environment's max episode length
        self.max_episode_len = self.env.max_episode_len

    def reset(self):
        """
        Resets the environment to an initial state.

        Returns:
            The initial observation from the environment.
        """
        obs = self.env.reset()
        return obs

    def step(self, action):
        """
        Takes a step in the environment based on the provided action.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing the next observation, reward, done flag, and additional info.
        """
        obs_, reward, done, info = self.env.step(action)
        return obs_, reward, done, info

    def render(self):
        """
        Renders the environment for visualization.
        """
        self.env.render()
