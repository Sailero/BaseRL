import torch


class Agent:
    # 定义基础的Agent
    def __init__(self, args):
        # 定义智能体的特征
        self.args = args
        self.policy_type = args.policy_type
        self.device = args.device
        self.imitation_learning = args.imitation_learning
        # 第一次需要输出网络图
        self.need_add_graph = True

        # 定义智能体的策略
        if self.policy_type == 'DDPG':
            from agent.policy.DDPG import DDPG
            from agent.modules.offline_replay_buffer import OfflineBuffer
            self.policy = DDPG(args)
            self.buffer = OfflineBuffer(args)
            self.online_policy = False
        elif self.policy_type == 'PPO':
            from agent.policy.PPO import PPO
            from agent.modules.online_replay_buffer import OnlineBuffer
            self.buffer = OnlineBuffer(args)
            self.policy = PPO(args)
            self.online_policy = True
        elif self.policy_type == 'GAIL_PPO':
            from agent.policy.PPO import PPO
            from agent.modules.online_replay_buffer import OnlineBuffer
            from agent.policy.GAIL import GAIL
            self.buffer = OnlineBuffer(args)
            self.policy = GAIL(args, PPO(args))
            self.online_policy = True

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
        self.policy.add_graph(obs, action, logger)    # 这里DDPG与PPO的critic输入格式不同，会报错

    def train(self, num, logger):
        transitions = self.buffer.sample()

        if self.need_add_graph:
            self.need_add_graph = False
            self.show_graph(logger)

        self.policy.train(transitions)

        # 记录log
        record = self.policy.train_record
        for v in self.policy.train_record.keys():
            logger.add_scalar(v, record[v], num)

    def save_models(self):
        print(f'... saving agent checkpoint ...')
        self.policy.save_models()

    def load_models(self):
        print(f'... loading agent checkpoint ...')
        self.policy.load_models()
