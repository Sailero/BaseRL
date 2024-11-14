class Buffer:
    def __init__(self, config):
        """
        Initialize a replay buffer for storing experience tuples.

        Args:
            config: Configuration object containing buffer size parameters.
        """
        self.buffer = None
        self.buffer_size = config.params["buffer_size"]

        # Initialize the buffer
        self.initial_buffer()

    def store_episode(self, obs_n, action_n, reward, next_obs_n, done):
        """
        Store a single episode (experience tuple) in the buffer.

        Args:
            obs_n: Current observation.
            action_n: Action taken by the agent.
            reward: Reward received after taking the action.
            next_obs_n: Next observation after taking the action.
            done: Boolean indicating if the episode has ended.
        """
        self.buffer['reward'].append(reward)
        self.buffer['done'].append(done)
        self.buffer['obs'].append(obs_n)
        self.buffer['action'].append(action_n)
        self.buffer['next_obs'].append(next_obs_n)

    def sample(self):
        """
        Sample the current buffer data. For this implementation, it returns the entire buffer.

        Returns:
            The entire buffer data.
        """
        return self.buffer

    def ready(self):
        """
        Check if the buffer has reached its predefined size and is ready for sampling.

        Returns:
            Boolean indicating if the buffer has enough data.
        """
        return len(self.buffer["done"]) >= self.buffer_size

    def initial_buffer(self):
        """Initialize the buffer with empty lists for each experience component."""
        self.buffer = {"obs": [], "action": [], "reward": [], "next_obs": [], "done": []}
