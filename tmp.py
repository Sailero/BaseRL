import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# 设置随机种子以保证结果一致性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 定义网格环境
GRID = np.array([
    ['S', '0', '0', '1', 'G'],
    ['0', '1', '0', '0', '0'],
    ['0', '0', '0', '1', '0'],
    ['1', '0', '0', '0', '0'],
    ['0', '0', '1', '0', '0']
])

# 起点和目标点位置
START = (0, 0)
GOAL = (0, 4)

# 动作定义：上下左右
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_MAP = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}


# 定义高层策略网络
class HighLevelPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, num_subgoals):
        super(HighLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_subgoals)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.output(x), dim=-1)
        return x


# 定义低层策略网络
class LowLevelPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(LowLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.output(x), dim=-1)
        return x


# 获取可通行的网格单元
def get_free_cells(grid):
    return {(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])
            if grid[i, j] == '0' or grid[i, j] == 'S' or grid[i, j] == 'G'}


free_cells = get_free_cells(GRID)


# 根据动作选择下一步的位置
def move(position, action):
    x, y = position
    dx, dy = ACTION_MAP[action]
    new_position = (x + dx, y + dy)
    if new_position in free_cells:
        return new_position
    return position  # 如果新位置是障碍物，保持在原地


# 选择子目标
def high_level_policy_select_subgoal(state, policy_net):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    subgoal_probs = policy_net(state_tensor)
    subgoal = torch.multinomial(subgoal_probs, num_samples=1).item()
    return subgoal


# 选择低层动作
def low_level_policy_select_action(state, policy_net):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_probs = policy_net(state_tensor)
    action_idx = torch.multinomial(action_probs, num_samples=1).item()
    return ACTIONS[action_idx]


# 构建状态向量
def build_state(current_position, goal_position):
    return np.array([current_position[0], current_position[1], goal_position[0], goal_position[1]])


# 训练函数
def train_hierarchical_policy(high_level_net, low_level_net, num_episodes=1000, gamma=0.99, lr=1e-3):
    high_level_optimizer = optim.Adam(high_level_net.parameters(), lr=lr)
    low_level_optimizer = optim.Adam(low_level_net.parameters(), lr=lr)

    high_level_rewards = []
    low_level_rewards = []

    for episode in range(num_episodes):
        current_position = START
        total_high_level_reward = 0
        total_low_level_reward = 0

        # 构建高层状态
        high_state = build_state(current_position, GOAL)

        # 高层策略选择子目标（示例中使用预定义的几个子目标）
        subgoal_idx = high_level_policy_select_subgoal(high_state, high_level_net)
        subgoals = [(0, 2), (2, 2), (4, 2)]
        subgoal = subgoals[subgoal_idx]

        done = False
        while not done:
            # 构建低层状态
            low_state = build_state(current_position, subgoal)

            # 低层策略选择动作
            action = low_level_policy_select_action(low_state, low_level_net)
            next_position = move(current_position, action)

            # 奖励设计
            if next_position == subgoal:
                reward = 10  # 到达子目标
                done = True
            elif next_position == GOAL:
                reward = 50  # 到达最终目标
                done = True
            else:
                reward = -1  # 每步移动的惩罚

            total_low_level_reward += reward

            # 更新低层策略
            low_state_tensor = torch.FloatTensor(low_state).unsqueeze(0)
            action_idx = ACTIONS.index(action)
            action_probs = low_level_net(low_state_tensor)
            low_level_loss = -torch.log(action_probs[0, action_idx]) * reward
            low_level_optimizer.zero_grad()
            low_level_loss.backward()
            low_level_optimizer.step()

            current_position = next_position

            # 如果到达最终目标，则给予高层策略奖励
            if current_position == GOAL:
                total_high_level_reward = 50

        # 更新高层策略
        high_state_tensor = torch.FloatTensor(high_state).unsqueeze(0)
        subgoal_probs = high_level_net(high_state_tensor)
        high_level_loss = -torch.log(subgoal_probs[0, subgoal_idx]) * total_high_level_reward
        high_level_optimizer.zero_grad()
        high_level_loss.backward()
        high_level_optimizer.step()

        high_level_rewards.append(total_high_level_reward)
        low_level_rewards.append(total_low_level_reward)

        if episode % 100 == 0:
            print(
                f"Episode {episode}: High-Level Reward = {total_high_level_reward}, Low-Level Reward = {total_low_level_reward}")

    return high_level_rewards, low_level_rewards


# 创建高层和低层策略网络
input_size_high = 4  # (x, y, goal_x, goal_y)
hidden_size = 64
num_subgoals = 3  # 选择子目标的数量
high_level_net = HighLevelPolicy(input_size_high, hidden_size, num_subgoals)

input_size_low = 4  # (x, y, subgoal_x, subgoal_y)
num_actions = 4  # 上、下、左、右
low_level_net = LowLevelPolicy(input_size_low, hidden_size, num_actions)

# 训练策略
train_hierarchical_policy(high_level_net, low_level_net)
