import numpy as np


class Buffer:
    def __init__(self, args):
        # Initialize the arguments parameters
        self.buffer_size = args.buffer_size
        self.agent_obs_dim = args.agent_obs_dim
        self.agent_action_dim = args.agent_action_dim
        self.batch_size = args.batch_size

        # memory management
        self.current_size = 0
        self.store_i = 0

        # Initialize the buffer
        self.initial_buffer()

    def store_episode(self, obs_n, action_n, reward, next_obs_n, done):
        self.buffer['reward'][self.store_i] = reward
        self.buffer['done'][self.store_i] = done
        self.buffer['obs'][self.store_i] = obs_n
        self.buffer['action'][self.store_i] = action_n
        self.buffer['next_obs'][self.store_i] = next_obs_n

        self.current_size = min(self.current_size + 1, self.buffer_size)
        self.store_i = (self.store_i + 1) % self.buffer_size

    def sample(self):
        sample_buffer = {}
        batch_id = np.random.choice(self.current_size, self.batch_size, replace=False)

        for key in self.buffer.keys():
            sample_buffer[key] = self.buffer[key][batch_id]
        return sample_buffer

    def ready(self):
        if self.current_size < self.batch_size:
            return False
        else:
            return True

    def initial_buffer(self):
        self.buffer = dict()
        self.buffer['reward'] = np.empty([self.buffer_size, 1])
        self.buffer['done'] = np.empty([self.buffer_size, 1], dtype=bool)
        self.buffer['obs'] = np.empty([self.buffer_size, self.agent_obs_dim])
        self.buffer['action'] = np.empty([self.buffer_size, self.agent_action_dim])
        self.buffer['next_obs'] = np.empty([self.buffer_size, self.agent_obs_dim])


