from tqdm import tqdm
import numpy as np
import time
import os
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from agent.agent import Agent
from common.utils import smooth, save_data, plot_returns_curves


class Runner:
    def __init__(self, config, env):
        # 加载环境
        self.env = env
        self.device = config.device.device

        # 加载args中训练有关参数
        self.max_episode_len = config.env.max_episode_len
        self.train_episodes = config.env.train_episodes
        self.compare_path = config.save_path
        self.expert_data_path = config.expert_data_path
        config.save_path = os.path.join(config.save_path, f"{config.policy_type}")

        self.plt_save_path = os.path.join(config.save_path, 'plt_results')
        self.data_save_path = os.path.join(config.save_path, 'data_results')

        self.load_pre_model = config.env.load_pre_model
        self.save_last_model = config.env.save_last_model

        # 加载智能体
        self.agent = Agent(config)

        # 加载演示有关参数
        self.evaluate_episodes = 100
        self.display_episodes = config.env.display_episodes
        self.force_save_model = config.env.force_save_model

        # 训练相关
        # 初始化最大奖励
        self.best_agent_return = -1e5
        self.best_episodes = 0

        # 创建保存路径
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        if not os.path.exists(self.plt_save_path):
            os.makedirs(self.plt_save_path)
        if not os.path.exists(self.data_save_path):
            os.makedirs(self.data_save_path)

    def save_run_data(self, episode, agent_returns, train_episode_step):
        avg_agent_returns = np.mean(agent_returns[-self.display_episodes:])
        if avg_agent_returns > self.best_agent_return:
            self.agent.save_models()

            self.best_agent_return = avg_agent_returns
            self.best_episodes = episode
            print(
                f"The reward is the most of the history when episode is {episode + 1}, "
                f"best_agent_return is {self.best_agent_return}")
        elif self.force_save_model:
            self.agent.save_models()

        # 保存奖励数据
        save_data(self.data_save_path, agent_returns, csv_name="agent_returns", column_name=["ReturnsForAgent"])
        save_data(self.data_save_path, train_episode_step, csv_name="train_episode_step",
                  column_name=['EachEpisodeSteps'])

        # 2.5 每100个回合记录训练曲线
        plot_returns_curves(agent_returns, self.plt_save_path)

        # 2.6 打印训练信息
        print()
        print('episode', episode + 1)
        print('\naverage episode steps', np.mean(train_episode_step[-self.display_episodes:]))
        print('agent average returns {:.1f}'.format(avg_agent_returns))

    def run(self):
        # 1. 训练准备
        # 初始化奖励列表与完成列表
        agent_returns = []
        train_episode_step = []

        # 是否加载先前训练的模型
        if self.load_pre_model:
            self.agent.load_models()

        # 2. 开始训练
        for episode in tqdm(range(self.train_episodes)):
            # 2.1 初始化环境
            obs = self.env.reset()

            # 记录对局reward
            agent_episode_reward = 0

            # 2.2 开始episode内的迭代
            for step in range(self.max_episode_len):
                # 2.2.1 智能体选择动作
                action = self.agent.choose_action(obs)

                # 2.2.2 智能体更新状态
                next_obs, reward, done, info = self.env.step(action)
                if step == self.max_episode_len - 1:
                    done = True

                # 2.2.3 存储信息（根据算法需要）
                self.agent.buffer.store_episode(obs, action, reward, next_obs, done)

                # 2.2.4 更新信息
                obs = next_obs

                # 2.2.5 离线策略训练
                if not self.agent.online_policy and self.agent.buffer.ready():
                    self.agent.train()

                # 2.2.6 记录对局reward
                agent_episode_reward += reward

                # 2.2.7 判断对局是否结束
                if done:
                    break

            # 2.3 增加对局和奖励记录
            agent_returns.append(agent_episode_reward)
            train_episode_step.append(step + 1)

            if self.agent.online_policy:
                self.agent.train()

            # 2.4 当训练没有完成时，任意一个智能体奖励提高时，保存模型，同时保存奖励数据
            if (episode + 1) % self.display_episodes == 0:
                self.save_run_data(episode, agent_returns, train_episode_step)

        # 3.1 当训练完成时，保存模型
        if self.save_last_model:
            self.agent.save_models()

        # 3.2 打印重要信息
        print(
            f"The best reward of agent is {round(self.best_agent_return, 2)} when episode is {self.best_episodes + 1}")

    def evaluate(self):
        # 加载预训练的模型
        self.agent.load_models()

        # 初始化奖励
        agent_returns = []

        # 交互演示
        for episode in tqdm(range(self.evaluate_episodes)):
            # 初始化环境
            obs = self.env.reset()

            # 记录对局reward
            agent_episode_reward = 0

            # 对于episode中的步数进行迭代
            for step in range(self.max_episode_len):
                # 可视化展示
                self.env.render()
                time.sleep(0.01)

                # 智能体选择动作
                action = self.agent.choose_action(obs)

                # 更新状态
                next_obs, reward, done, info = self.env.step(action)
                if step == self.max_episode_len - 1:
                    done = True

                # 更新信息
                obs = next_obs

                # 记录对局reward
                agent_episode_reward += reward

                # 判断对局是否结束
                if done:
                    break

            # 增加对局奖励记录
            print(f"episode {episode}'s episode_reward:f{agent_episode_reward}")
            agent_returns.append(agent_episode_reward)

    def compare_models_curves(self):
        # 从所有子文件夹里获取数据
        def get_data(name):
            data = []
            folder_list = []
            for folder in os.listdir(self.compare_path):
                folder_path = os.path.join(self.compare_path, folder, 'data_results', f'{name}.csv')
                # 检查文件是否存在
                if os.path.isfile(folder_path):
                    matrix = np.array(pd.read_csv(folder_path))
                    data.append(matrix)
                    folder_list.append(folder)
            data = np.stack(data, axis=0)
            return data, folder_list

        # 获取数据
        agent_returns, _ = get_data("agent_returns")
        train_episode_step, _ = get_data("train_episode_step")

        compare_results_path = os.path.join(self.compare_path, 'compare_results')
        if not os.path.exists(compare_results_path):
            os.makedirs(compare_results_path)

        def plot_data(data, name):
            # 绘图
            plt.figure()
            alpha = [0.1, 1]
            weight = [0.8, 0.96]
            linewidth = [4, 0.5]
            colors = ['blue', 'red', 'green', 'darkred', 'orange', 'violet', 'brown', 'navy', 'teal']

            for j in range(data.shape[2]):
                for i in range(data.shape[0]):
                    plt.plot(range(len(data[i, :, j])), smooth(data[i, :, j], weight=weight[0]),
                             color=colors[i], linewidth=linewidth[0], alpha=alpha[0])
                    plt.plot(range(len(data[i, :, j])), smooth(data[i, :, j], weight=weight[1]), label=folder_list[i],
                             color=colors[i], linewidth=linewidth[1], alpha=alpha[1])
                plt.legend()
                plt.title(f'Different policies {name}')
                plt.xlabel('episode')
                plt.ylabel(f'{name}')

                # 获取图片保存地址
                path = os.path.join(compare_results_path, f'compare_{name}.png')
                plt.savefig(path, format='png')
                plt.clf()
                plt.close()
                print(f'Different policies {name} figure is saved')

        plot_data(agent_returns, 'agent_returns')
        plot_data(train_episode_step, 'train_episode_step')
