import numpy as np

class Buffer:
    def __init__(self, config):
        """
        Initialize a replay buffer for storing experience tuples.

        Args:
            config: Configuration object containing buffer and batch size parameters.
        """
        self.buffer = None
        self.buffer_size = int(config.params["buffer_size"])
        self.agent_obs_dim = config.env.agent_obs_dim
        self.agent_action_dim = config.env.agent_action_dim
        self.batch_size = int(config.params["batch_size"])

        # Memory management
        self.current_size = 0  # Number of stored experiences
        self.store_i = 0       # Index to store the next experience

        # Initialize buffer with empty arrays
        self.initial_buffer()

    def store_episode(self, obs, action, reward, next_obs, done):
        """
        Store a single episode (experience tuple) in the buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Boolean indicating if the episode has ended.
        """
        self.buffer['reward'][self.store_i] = reward
        self.buffer['done'][self.store_i] = done
        self.buffer['obs'][self.store_i] = obs
        self.buffer['action'][self.store_i] = action
        self.buffer['next_obs'][self.store_i] = next_obs

        # Update buffer size and storage index
        self.current_size = min(self.current_size + 1, self.buffer_size)
        self.store_i = (self.store_i + 1) % self.buffer_size

    def sample(self):
        """
        Sample a batch of experiences from the buffer.

        Returns:
            A dictionary containing sampled experience batches.
        """
        sample_buffer = {}
        batch_id = np.random.choice(self.current_size, self.batch_size, replace=False)
        for key in self.buffer.keys():
            sample_buffer[key] = self.buffer[key][batch_id]
        return sample_buffer

    def ready(self):
        """
        Check if the buffer has enough data to sample a batch.

        Returns:
            Boolean indicating if the buffer is ready for sampling.
        """
        return self.current_size >= self.batch_size

    def initial_buffer(self):
        """Initialize the buffer with empty arrays for each experience component."""
        self.buffer = {
            'reward': np.empty([self.buffer_size, 1]),
            'done': np.empty([self.buffer_size, 1], dtype=bool),
            'obs': np.empty([self.buffer_size] + self.agent_obs_dim),
            'action': np.empty([self.buffer_size, self.agent_action_dim]),
            'next_obs': np.empty([self.buffer_size] + self.agent_obs_dim),
        }

    def load_buffer(self, load_obs, load_action, load_reward, load_next_obs, load_done):
        """
        Load experiences into the buffer from provided arrays.

        Args:
            load_obs: Array of observations.
            load_action: Array of actions.
            load_reward: Array of rewards.
            load_next_obs: Array of next observations.
            load_done: Array of done flags.
        """
        self.buffer['obs'][:len(load_obs)] = load_obs
        self.buffer['action'][:len(load_action)] = load_action
        self.buffer['reward'][:len(load_reward)] = load_reward
        self.buffer['next_obs'][:len(load_next_obs)] = load_next_obs
        self.buffer['done'][:len(load_done)] = load_done

        # Update buffer size and storage index
        self.current_size = min(len(load_obs), self.buffer_size)
        self.store_i = self.current_size
