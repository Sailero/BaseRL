import torch


class Agent:
    # 定义基础的Agent
    def __init__(self, args):
        # 定义智能体的特征
        self.args = args
        self.policy_type = args.policy_type
        self.device = args.device
        self.imitation_learning = args.imitation_learning

        # 定义智能体的策略
        if self.policy_type == 'DDPG':
            from agent.policy.DDPG import DDPG
            from agent.modules.replay_buffer import Buffer
            self.policy = DDPG(args)
            self.buffer = Buffer(args)
        elif self.policy_type == 'PPO':
            from agent.policy.PPO import PPO
            from agent.modules.online_replay_buffer import Buffer
            self.buffer = Buffer(args)
            self.policy = PPO(args)
            self.update_nums = args.update_nums

    def choose_action(self, observation):
        # 将输入放在gpu上运行
        observation = torch.tensor(observation, dtype=torch.float64).to(self.device)

        # 获取动作
        action = self.policy.choose_action(observation)
        return action

    def train(self):
        transitions = self.buffer.sample()
        self.policy.train(transitions)

        if self.policy_type in ['PPO'] and not self.imitation_learning:
            self.buffer.initial_buffer()

    def save_checkpoint(self):
        print(f'... saving agent checkpoint ...')
        self.policy.save_models()

    def load_checkpoint(self):
        print(f'... loading agent checkpoint ...')
        self.policy.load_models()
