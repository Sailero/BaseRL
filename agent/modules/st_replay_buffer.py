import numpy as np


class Buffer:
    def __init__(self, args):
        # Initialize the arguments parameters
        self.buffer_size = args.update_steps
        self.agent_obs_dim = args.agent_obs_dim
        self.agent_action_dim = args.agent_action_dim

        # Initialize the buffer
        self.initial_buffer()

    def store_episode(self, obs, action, log_prob, reward, next_obs_n, done, value):
        self.buffer['reward'][self.store_i] = reward
        self.buffer['done'][self.store_i] = done
        self.buffer['obs'][self.store_i] = obs
        self.buffer['action'][self.store_i] = action
        self.buffer['next_obs'][self.store_i] = next_obs_n
        self.buffer['log_prob'][self.store_i] = log_prob
        self.buffer['value'][self.store_i] = value

        self.current_size = min(self.current_size + 1, self.buffer_size)
        self.store_i = (self.store_i + 1) % self.buffer_size

    def ready(self):
        if self.current_size < self.buffer_size:
            return False
        else:
            self.current_size = 0
            return True

    def initial_buffer(self):
        # memory management
        self.current_size = 0
        self.store_i = 0

        # Initial buffer
        self.buffer = dict()
        self.buffer['reward'] = np.empty([self.buffer_size, 1])
        self.buffer['done'] = np.empty([self.buffer_size, 1], dtype=bool)
        self.buffer['obs'] = np.empty([self.buffer_size, self.agent_obs_dim])
        self.buffer['action'] = np.empty([self.buffer_size, self.agent_action_dim])
        self.buffer['next_obs'] = np.empty([self.buffer_size, self.agent_obs_dim])

        self.buffer['log_prob'] = np.empty([self.buffer_size, 1])
        self.buffer['value'] = np.empty([self.buffer_size, 1])


