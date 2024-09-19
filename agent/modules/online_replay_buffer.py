import numpy as np


class Buffer:
    def __init__(self, args):
        # Initialize the arguments parameters
        self.buffer_size = args.st_buffer_size
        self.agent_obs_dim = args.agent_obs_dim
        self.agent_action_dim = args.agent_action_dim

        # Initialize the buffer
        self.initial_buffer()

    def store_episode(self, obs, action, reward, next_obs, done):
        self.buffer['reward'][self.store_i] = reward
        self.buffer['done'][self.store_i] = done
        self.buffer['obs'][self.store_i] = obs
        self.buffer['action'][self.store_i] = action
        self.buffer['next_obs'][self.store_i] = next_obs

        self.store_i = self.store_i + 1

    def sample(self):
        sample_buffer = {}
        batch_id = np.arange(self.store_i)
        for key in self.buffer.keys():
            sample_buffer[key] = self.buffer[key][batch_id]

        return sample_buffer


    def ready(self):
        if self.store_i < self.buffer_size:
            return False
        else:
            return True

    def initial_buffer(self):
        # memory management
        self.store_i = 0

        # Initial buffer
        self.buffer = dict()
        self.buffer['reward'] = np.zeros([self.buffer_size, 1])
        self.buffer['done'] = np.zeros([self.buffer_size, 1], dtype=bool)
        self.buffer['obs'] = np.zeros([self.buffer_size, self.agent_obs_dim])
        self.buffer['action'] = np.zeros([self.buffer_size, self.agent_action_dim])
        self.buffer['next_obs'] = np.zeros([self.buffer_size, self.agent_obs_dim])


