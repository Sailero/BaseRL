import numpy as np
from agent.off_policy.offp_config import BUFFER_CONFIG


class Buffer:
    def __init__(self, args):
        # Initialize the arguments parameters
        self.buffer = None
        self.buffer_size = int(BUFFER_CONFIG["buffer_size"])
        self.agent_obs_dim = args.agent_obs_dim
        self.agent_action_dim = args.agent_action_dim
        self.batch_size = int(BUFFER_CONFIG["batch_size"])

        # memory management
        self.current_size = 0
        self.store_i = 0

        # Initialize the buffer
        self.initial_buffer()

        self.record = {"obs": [], "next_obs": [], "action": [], "done": [], "reward": []}

    @property
    def data(self):  # 这里SAC的data是self.record
        data_buffer = {}
        for key in self.buffer.keys():
            data_buffer[key] = self.buffer[key][:self.current_size]
        return data_buffer

    def store_episode(self, obs, action, reward, next_obs, done):
        self.buffer['reward'][self.store_i] = reward
        self.buffer['done'][self.store_i] = done
        self.buffer['obs'][self.store_i] = obs
        self.buffer['action'][self.store_i] = action
        self.buffer['next_obs'][self.store_i] = next_obs

        self.current_size = min(self.current_size + 1, self.buffer_size)
        self.store_i = (self.store_i + 1) % self.buffer_size

        self.record["action"].append(action)
        self.record["reward"].append(reward)

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
        self.buffer['obs'] = np.empty([self.buffer_size] + self.agent_obs_dim)
        self.buffer['action'] = np.empty([self.buffer_size, self.agent_action_dim])
        self.buffer['next_obs'] = np.empty([self.buffer_size] + self.agent_obs_dim)

    def load_buffer(self, load_obs, load_action, load_reward, load_next_obs, load_done):
        self.buffer['obs'][:len(load_obs)] = load_obs
        self.buffer['action'][:len(load_action)] = load_action
        self.buffer['reward'][:len(load_reward)] = load_reward
        self.buffer['next_obs'][:len(load_next_obs)] = load_next_obs
        self.buffer['done'][:len(load_done)] = load_done

        self.current_size = min(len(load_obs), self.buffer_size)
        self.store_i = self.current_size
