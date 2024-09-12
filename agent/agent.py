from modules.replay_buffer import Buffer
import torch


class Agent:
    # 定义基础的Agent
    def __init__(self, args):
        # 定义智能体的特征
        self.args = args
        self.policy_type = args.policy_type
        self.device = args.device


        # 定义智能体的策略
        if self.policy_type == 'DDPG':
            from policy.DDPG import DDPG
            self.policy = DDPG(args)
            self.buffer = Buffer(args)
        elif self.policy_type == 'DQN':
            from policy.DQN import DQN
            self.policy = DQN(args)
            self.buffer = Buffer(args)
        elif self.policy_type == 'PPO':
            from policy.PPO import PPO
            self.policy = PPO(args)

    def choose_action(self, observation):
        # 将输入放在gpu上运行
        observation = torch.tensor(observation, dtype=torch.float64).to(self.device)

        # 获取动作
        action = self.policy.choose_action(observation)
        return action

    def train(self):
        if self.policy_type in ['DDPG', 'DQN']:
            transitions = self.buffer.sample()
            self.policy.train(transitions)
        else:
            self.policy.train()

    def save_checkpoint(self):
        print(f'... saving agent checkpoint ...')
        self.policy.save_models()

    def load_checkpoint(self):
        print(f'... loading agent checkpoint ...')
        self.policy.load_models()

