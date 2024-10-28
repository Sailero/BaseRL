import collections
import random


class Buffer:
    def __init__(self, args, capacity=1024):
        # 队列,先进先出
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity
        self.batch_size = args.batch_size

        # 用于日志显示
        self.record = None
        self.reset()

    def reset(self):
        self.record = {"obs": [], "next_obs": [], "action": [], "done": [], "reward": []}
        pass

    @property
    def data(self):
        return self.record

    def ready(self):
        if len(self.buffer) < self.batch_size:
            return False
        else:
            return True

    def store_episode(self, obs, action, reward, next_obs, done):
        if len(self.buffer) == self.capacity:
            self.buffer.popleft()
        self.buffer.append((obs, action, reward, next_obs, done))

        self.record["action"].append(action)
        self.record["reward"].append(reward)

    def sample(self):
        # 从 buffer 中采样数据,数量为 batch_size
        transitions = random.sample(self.buffer, self.batch_size)
        buffer = {}
        buffer["obs"], buffer["action"], buffer["reward"], buffer["next_obs"], buffer["done"] = zip(*transitions)
        return buffer

    def size(self):
        return len(self.buffer)
