import torch
import numpy as np
from tqdm import tqdm
import os

class Runner:
    def __init__(self, env, agent, train_episodes, max_episode_len, display_episodes, data_save_path, plt_save_path, load_pre_model=False, save_last_model=True):
        self.env = env
        self.agent = agent
        self.train_episodes = train_episodes  # 训练的回合数
        self.max_episode_len = max_episode_len  # 每个回合的最大步数
        self.display_episodes = display_episodes  # 每隔多少回合显示一次训练信息
        self.data_save_path = data_save_path  # 保存数据的路径
        self.plt_save_path = plt_save_path  # 保存训练曲线的路径
        self.load_pre_model = load_pre_model  # 是否加载之前的模型
        self.save_last_model = save_last_model  # 是否在训练结束后保存模型

        # 检查保存路径
        os.makedirs(self.data_save_path, exist_ok=True)
        os.makedirs(self.plt_save_path, exist_ok=True)

    def run(self):
        # 1. 训练准备
        agent_returns = []
        game_results = []
        train_episode_step = []

        # 初始化最大奖励
        best_agent_return = -1e5
        best_episodes = 0

        # 是否加载先前训练的模型
        if self.load_pre_model:
            self.agent.load_models()

        # 2. 开始训练
        train_step = 0
        for episode in tqdm(range(self.train_episodes)):
            # 2.1 初始化环境
            obs = self.env.reset()
            agent_episode_reward = 0

            # 2.2 开始episode内的迭代
            for step in range(self.max_episode_len):
                # 2.2.1 智能体选择动作
                action = self.agent.choose_action(obs)

                # 2.2.2 智能体更新状态
                with torch.no_grad():
                    obs_, reward, done, _ = self.env.step(action)

                # 2.2.3 存储经验
                transition = {
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'next_obs': obs_,
                    'done': done,
                    'log_prob': self.agent.actor_network.log_prob(action)  # 根据需要存储log_prob
                }
                self.agent.store_transition(transition)

                # 2.2.4 更新状态
                obs = obs_
                train_step += 1

                # 2.2.5 智能体训练，每隔10步训练一次
                if len(self.agent.memory) >= 2048 and train_step % 10 == 0:
                    self.agent.train()

                # 2.2.6 记录对局奖励
                agent_episode_reward += reward

                # 2.2.7 判断回合是否结束
                if done:
                    break

            # 2.3 增加对局和奖励记录
            game_results.append(done)
            agent_returns.append(agent_episode_reward)
            train_episode_step.append(step + 1)

            # 2.4 保存模型和奖励数据
            if (episode + 1) % self.display_episodes == 0:
                avg_agent_returns = np.mean(agent_returns[-self.display_episodes:])
                if avg_agent_returns > best_agent_return:
                    self.agent.save_models()
                    best_agent_return = avg_agent_returns
                    best_episodes = episode
                    print(f"Best reward so far: {best_agent_return:.2f}, at episode {episode + 1}")

                # 保存奖励数据
                self.save_data(game_results, "game_results", ["GameResults"])
                self.save_data(agent_returns, "agent_returns", ["ReturnsForAgent"])
                self.save_data(train_episode_step, "train_episode_step", ["EachEpisodeSteps"])

                # 可视化奖励曲线
                self.plot_returns_curves(agent_returns)

                # 打印训练信息
                print()
                print(f'Episode {episode + 1}')
                print(f'Average episode steps: {np.mean(train_episode_step[-self.display_episodes:]):.1f}')
                print(f'Agent average returns: {avg_agent_returns:.1f}')

        # 3. 保存最终模型和打印信息
        if self.save_last_model:
            self.agent.save_models()

        print(f"The best reward for the agent is {round(best_agent_return, 2)} at episode {best_episodes + 1}")

    def save_data(self, data, csv_name, column_name):
        file_path = os.path.join(self.data_save_path, f"{csv_name}.csv")
        np.savetxt(file_path, np.array(data), delimiter=",", header=",".join(column_name), comments='')

    def plot_returns_curves(self, agent_returns):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(agent_returns)
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title("Training Returns")
        plt.savefig(os.path.join(self.plt_save_path, "returns_curve.png"))
        plt.close()
