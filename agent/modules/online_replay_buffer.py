import numpy as np


class OnlineBuffer:
    def __init__(self, args):
        # Initialize the arguments parameters
        self.buffer = dict()
        self.buffer_size = 0
        self.reset()

        self.imitation_learning = args.imitation_learning
        self.im_buffer = dict()
        self.im_buffer_size = 0
        self.im_sample_size = args.im_sample_size

        self.policy_type = args.policy_type

    @property
    def data(self):
        return self.buffer

    def store_episode(self, obs, action, reward, next_obs, done):
        self.buffer['obs'].append(obs)
        self.buffer['action'].append(action)
        self.buffer['reward'].append(reward)
        self.buffer['next_obs'].append(next_obs)
        self.buffer['done'].append(done)

        self.buffer_size = len(self.buffer['obs'])

    def reset(self):
        self.buffer = {"obs": [], "next_obs": [], "action": [], "done": [], "reward": []}
        self.buffer_size = 0

    def sample(self):
        # 如果是模仿学习还需要从imitation buffer中采样
        if self.imitation_learning:
            im_buffer = self.sample_im()
            if self.policy_type == "PPO":
                return im_buffer
            else:
                return self.buffer, im_buffer
        return self.buffer

    def sample_im(self):
        sample_buffer = {}
        # 随机从imitation buffer中采样
        batch_id = np.random.choice(np.arange(self.im_buffer_size), size=self.im_sample_size, replace=True)
        for key in self.im_buffer.keys():
            # 由于为了减小图片的存储空间，保存的是int8格式的图片，因此需要转换成float32格式
            if key == 'obs' or key == 'next_obs':
                sample_buffer[key] = self.im_buffer[key][batch_id].astype(np.float32) / 255.0
            else:
                sample_buffer[key] = self.im_buffer[key][batch_id]

        return sample_buffer

    @staticmethod
    def ready():
        return True

    def load_buffer(self, load_obs, load_action, load_reward, load_next_obs, load_done):
        """
        加载专家数据
        """
        self.im_buffer['obs'] = load_obs
        self.im_buffer['action'] = load_action
        self.im_buffer['reward'] = load_reward
        self.im_buffer['next_obs'] = load_next_obs
        self.im_buffer['done'] = load_done
        self.im_buffer_size = len(load_obs)
