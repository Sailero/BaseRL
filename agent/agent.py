import torch
from agent_config import POLICY_MAP


def load_class(path):
    components = path.split('.')
    module = __import__('.'.join(components[:-1]), fromlist=[components[-1]])
    return getattr(module, components[-1])


class Agent:
    # 定义基础的Agent
    def __init__(self, args):
        # 定义智能体的特征
        self.args = args
        self.policy_type = args.policy_type
        self.device = args.device

        # 第一次需要输出网络图
        self.need_add_graph = True

        # 根据策略类型加载策略和缓冲区
        config = POLICY_MAP.get(self.policy_type)
        if config:
            self.policy = load_class(config['policy'])(args)
            self.buffer = load_class(config['buffer'])(args)
            self.online_policy = config['online_policy']
        else:
            raise ValueError(f"Unsupported policy type: {self.policy_type}")

    def choose_action(self, observation):
        # 将输入放在gpu上运行
        observation = torch.as_tensor(observation, device=self.device, dtype=torch.float32)

        # 获取动作
        action = self.policy.choose_action(observation)
        return action

    def show_graph(self, logger):
        # 生成随机的torch输入，用于网络图的可视化
        shape = [1] + self.args.agent_obs_dim
        obs = torch.empty(shape).uniform_(0, 1).to(self.device)
        shape = [1] + [self.args.agent_action_dim]
        action = torch.empty(shape).uniform_(0, 1).to(self.device)
        self.policy.add_graph(obs, action, logger)

    def train(self, num, logger):
        transitions = self.buffer.sample()

        if self.need_add_graph:
            self.need_add_graph = False
            if not isinstance(self.args.agent_obs_dim, int) and self.online_policy:
                self.show_graph(logger)

        self.policy.train(transitions)

        if self.online_policy:
            self.buffer.initial_buffer()

        # self.log_training_metrics(logger, num)

    def log_training_metrics(self, logger, num):
        record = getattr(self.policy, "train_record", None)
        if record:
            for key, value in record.items():
                if isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        if isinstance(item, (list, tuple)):
                            for c, sub_item in enumerate(item):
                                logger.add_scalar(f'{key}_{c}', sub_item, num + i)
                        else:
                            logger.add_scalar(f'{key}', item, num + i)
                else:
                    logger.add_scalar(f'{key}', value, num)

    def save_models(self):
        print(f'... saving agent checkpoint ...')
        self.policy.save_models()

    def load_models(self):
        print(f'... loading agent checkpoint ...')
        self.policy.load_models()
