import numpy as np


class ImBuffer:
    def __init__(self, args):
        """
        Initialize the imitation buffer with expert data.

        Args:
            args: Argument object containing path to expert data.
        """
        self.im_buffer = {}
        self.im_buffer_size = 0
        self.expert_data_path = args.expert_data_path

        # Load expert data into the buffer
        self.load_buffer()

    def sample(self, im_sample_size):
        """
        Sample a batch of data from the imitation buffer.

        Args:
            im_sample_size: Number of samples to draw from the buffer.

        Returns:
            A dictionary of sampled data with keys matching buffer keys.
        """
        sample_buffer = {}
        # Randomly sample indices for batch
        batch_id = np.random.choice(np.arange(self.im_buffer_size), size=im_sample_size, replace=True)

        # Populate sample buffer with sampled data
        for key in self.im_buffer.keys():
            sample_buffer[key] = self.im_buffer[key][batch_id]

        return sample_buffer

    def load_buffer(self):
        """
        Load expert data from files into the imitation buffer.
        """
        expert_obs = np.load(self.expert_data_path + '/expert_obs.npy')
        expert_action = np.load(self.expert_data_path + '/expert_action.npy')
        expert_reward = np.load(self.expert_data_path + '/expert_reward.npy').reshape([-1, 1])
        expert_next_obs = np.load(self.expert_data_path + '/expert_next_obs.npy')
        expert_done = np.load(self.expert_data_path + '/expert_done.npy').reshape([-1, 1])

        # Store expert data in the buffer
        self.im_buffer['obs'] = expert_obs
        self.im_buffer['action'] = expert_action
        self.im_buffer['reward'] = expert_reward
        self.im_buffer['next_obs'] = expert_next_obs
        self.im_buffer['done'] = expert_done
        self.im_buffer_size = len(expert_obs)
