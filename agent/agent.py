import torch
from agent.agent_config import POLICY_MAP


def load_class(path):
    components = path.split('.')
    module = __import__('.'.join(components[:-1]), fromlist=[components[-1]])
    return getattr(module, components[-1])


class Agent:
    # 定义基础的Agent
    def __init__(self, config):
        # 定义智能体的特征
        self.policy_type = config.policy_type
        self.device = config.device.device
        self.agent_obs_dim =config.env.agent_obs_dim
        self.agent_action_dim = config.env.agent_action_dim

        # 根据策略类型加载策略和缓冲区
        policy_config = POLICY_MAP.get(self.policy_type)
        if policy_config:
            self.policy = load_class(policy_config['policy'])(config)
            self.buffer = load_class(policy_config['buffer'])(config)
            self.online_policy = policy_config['online_policy']
        else:
            raise ValueError(f"Unsupported policy type: {self.policy_type}")

    def choose_action(self, observation):
        observation = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        action = self.policy.choose_action(observation)
        return action

    def train(self):
        transitions = self.buffer.sample()
        self.policy.train(transitions)

        if self.online_policy:
            self.buffer.initial_buffer()

    def save_models(self):
        print(f'... saving agent checkpoint ...')
        self.policy.save_models()

    def load_models(self):
        print(f'... loading agent checkpoint ...')
        self.policy.load_models()
