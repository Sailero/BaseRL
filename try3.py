import numpy as np


# 示例输入
rewards = [1.0, 1.0, 1.0]  # 即时奖励
values = [0.5, 0.6, 0.7]   # 值函数估计
next_values = values[1:] + [0.8]  # 最后一个next_values为外部提供的
dones = [0, 0, 1]          # 表示episode结束
gamma = 0.99               # 折扣因子
lam = 0.95                 # GAE的λ

# 手动计算 GAE (以公式为基础)
# delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
delta_2 = rewards[2] + gamma * next_values[2] * (1 - dones[2]) - values[2]
delta_1 = rewards[1] + gamma * next_values[1] * (1 - dones[1]) - values[1]
delta_0 = rewards[0] + gamma * next_values[0] * (1 - dones[0]) - values[0]

# 手动递归计算 GAE
adv_2 = delta_2  # 因为 dones[2] == 1, 最后一个时间步的优势就是 delta_2
adv_1 = delta_1 + gamma * lam * adv_2 * (1 - dones[1])
adv_0 = delta_0 + gamma * lam * adv_1 * (1 - dones[0])

manual_advantages = [adv_0, adv_1, adv_2]
print(f"手动计算的优势: {manual_advantages}")

# 使用 compute_gae 进行验证
class PPO:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.lam * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
        return advantages

# 初始化 PPO 实例并计算优势
ppo = PPO(gamma=gamma, lam=lam)
computed_advantages = ppo.compute_gae(rewards, values, next_values, dones)
print(f"代码计算的优势: {computed_advantages}")

# 比较手工计算与代码输出
print(f"是否一致: {np.allclose(manual_advantages, computed_advantages)}")

