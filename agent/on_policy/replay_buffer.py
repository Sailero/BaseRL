from agent.on_policy.onp_config import BUFFER_CONFIG


class Buffer:
    def __init__(self, args):
        # Initialize the arguments parameters
        self.buffer = None
        self.buffer_size = BUFFER_CONFIG["buffer_size"]

        # Initialize the buffer
        self.initial_buffer()

    @property
    def data(self):
        return self.buffer

    def store_episode(self, obs_n, action_n, reward, next_obs_n, done):
        self.buffer['reward'].append(reward)
        self.buffer['done'].append(done)
        self.buffer['obs'].append(obs_n)
        self.buffer['action'].append(action_n)
        self.buffer['next_obs'].append(next_obs_n)

    def sample(self):
        return self.buffer

    def ready(self):
        if len(self.buffer["done"]) < self.buffer_size:
            return False
        else:
            return True

    def initial_buffer(self):
        self.buffer = {"obs": [], "action": [], "reward": [], "next_obs": [], "done": []}

