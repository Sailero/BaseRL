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
    def __init__(self, args, env, logger):
        self.logger = logger
        # 加载环境
        self.env = env
        self.device = args.device

        # 加载args中训练有关参数
        self.render = args.render
        self.max_episode_len = args.max_episode_len
        self.train_episodes = args.train_episodes
        self.compare_path = args.save_path
        self.expert_data_path = args.expert_data_path
        args.save_path = os.path.join(args.save_path, f"{args.policy_type}")

        self.plt_save_path = os.path.join(args.save_path, 'plt_results')
        self.data_save_path = os.path.join(args.save_path, 'data_results')

        self.load_pre_model = args.load_pre_model
        self.save_last_model = args.save_last_model

        # 加载智能体
        self.agent = Agent(args)

        # 加载演示有关参数
        self.is_evaluated = args.evaluate
        self.evaluate_episodes = args.evaluate_episodes
        self.display_episodes = args.display_episodes
        self.force_save_model = args.force_save_model

        # 训练相关
        # 初始化最大奖励
        self.best_agent_return = -1e5
        self.best_episodes = 0

        # 创建保存路径
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if not os.path.exists(self.plt_save_path):
            os.makedirs(self.plt_save_path)
        if not os.path.exists(self.data_save_path):
            os.makedirs(self.data_save_path)
        if not os.path.exists(self.expert_data_path):
            os.makedirs(self.expert_data_path)

    def load_expert_data(self):
        # 加载专家数据
        expert_obs = np.load(self.expert_data_path + '/expert_obs.npy')
        expert_action = np.load(self.expert_data_path + '/expert_action.npy')
        expert_reward = np.load(self.expert_data_path + '/expert_reward.npy').reshape([-1, 1])
        expert_next_obs = np.load(self.expert_data_path + '/expert_next_obs.npy')
        expert_done = np.load(self.expert_data_path + '/expert_done.npy').reshape([-1, 1])

        print("sample data nums:", len(expert_obs))
        print("obs shape", expert_obs.shape)
        print("action shape", expert_action.shape)

        self.agent.buffer.load_buffer(expert_obs, expert_action, expert_reward, expert_next_obs, expert_done)

    def imitation_learning(self):
        self.load_expert_data()

        for num in tqdm(range(self.train_episodes)):
            self.agent.train(num, self.logger)

        self.agent.save_models()

        self.evaluate()

    def save_run_data(self, episode, agent_returns, game_results, train_episode_step):
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
        save_data(self.data_save_path, game_results, csv_name="game_results", column_name=['GameResults'])
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
        game_results = []
        train_episode_step = []
        terminated = None

        # 是否加载先前训练的模型
        if self.load_pre_model:
            self.agent.load_models()

        # 2. 开始训练
        for episode in tqdm(range(self.train_episodes)):
            # 2.1 初始化环境
            obs = self.env.reset()

            # 记录对局reward
            agent_episode_reward = 0
            done = False
            step = 0
            self.agent.buffer.reset()

            # 2.2 开始episode内的迭代
            for step in range(self.max_episode_len):
                # 2.2.0 显示训练过程
                if self.render:
                    self.env.render()
                    time.sleep(0.01)

                # 2.2.1 智能体选择动作
                action = self.agent.choose_action(obs)

                # 2.2.2 智能体更新状态
                next_obs, reward, done, info = self.env.step(action)
                if 'terminated' in info:
                    terminated = info['terminated']

                if step == self.max_episode_len - 1:
                    done = True

                # 2.2.3 存储信息（根据算法需要）
                self.agent.buffer.store_episode(obs, action, reward, next_obs, done)

                # 2.2.4 更新信息
                obs = next_obs

                # 2.2.5 离线策略训练
                if not self.agent.online_policy and self.agent.buffer.ready():
                    self.agent.train(episode, self.logger)

                # 2.2.6 记录对局reward
                agent_episode_reward += reward

                # 2.2.7 判断对局是否结束
                if done:
                    break

            # 2.3 增加对局和奖励记录
            game_results.append(done)
            agent_returns.append(agent_episode_reward)
            train_episode_step.append(step + 1)

            if self.agent.online_policy:
                self.agent.train(episode, self.logger)

            if terminated is not None:
                # 记录是否成功
                self.logger.add_scalar('train/terminated', int(terminated), episode)
            self.logger.add_scalar(f'train/episode_reward', agent_episode_reward, episode)
            # 记录动作
            action_list = self.agent.buffer.data['action']
            cnt = 0
            for action in action_list:
                for i in range(len(action)):
                    self.logger.add_scalar(f'train/action_{i}', action[i], cnt)
                cnt += 1

            # 记录奖励日志
            reward_list = self.agent.buffer.data['reward']
            cnt = 0
            for r in reward_list:
                self.logger.add_scalar(f'train/reward', r, cnt)
                cnt += 1

            # 2.4 当训练没有完成时，任意一个智能体奖励提高时，保存模型，同时保存奖励数据
            if (episode + 1) % self.display_episodes == 0:
                self.save_run_data(episode, agent_returns, game_results, train_episode_step)

        # 3.1 当训练完成时，保存模型
        if self.save_last_model:
            self.agent.save_models()

        # 3.2 打印重要信息
        print(
            f"The best reward of agent is {round(self.best_agent_return, 2)} when episode is {self.best_episodes + 1}")

    def evaluate(self):
        # 加载预训练的模型
        self.agent.load_models()
        # 输出网络结构
        if not isinstance(self.env.agent_obs_dim, int):
            self.agent.show_graph(self.logger)

        # 初始化奖励
        agent_returns = []
        game_results = []
        terminated = None
        done = False
        # 记录动作
        action_list = []
        # 记录每一步的奖励值
        reward_list = []

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
                action_list.append(action)

                # 更新状态
                next_obs, reward, done, info = self.env.step(action)

                if 'terminated' in info:
                    terminated = info['terminated']

                reward_list.append(reward)

                if step == self.max_episode_len - 1:
                    done = True

                # 更新信息
                obs = next_obs

                # 记录对局reward
                agent_episode_reward += reward

                # 判断对局是否结束
                if done:
                    break

            # 增加对局记录
            game_results.append(done)

            # 记录动作日志
            cnt = 0
            for action in action_list:
                for i in range(len(action)):
                    self.logger.add_scalar(f'evaluate/action_{i}', action[i], cnt)
                cnt += 1
            action_list.clear()

            # 记录奖励日志
            cnt = 0
            for r in reward_list:
                self.logger.add_scalar(f'evaluate/reward', r, cnt)
                cnt += 1
            reward_list.clear()

            if terminated is not None:
                # 记录是否成功
                self.logger.add_scalar('evaluate/terminated', int(terminated), episode)
            self.logger.add_scalar('evaluate/episode_reward', agent_episode_reward, episode)

            # 增加对局奖励记录
            print(f"episode {episode}'s episode_reward:f{agent_episode_reward}")
            agent_returns.append(agent_episode_reward)

        # 打印相关信息
        print(f"The probability of finishing the task is {np.sum(game_results) / len(game_results) * 100}%")

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

        game_results, folder_list = get_data("game_results")
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
            colors = ['blue', 'red', 'gree', 'darkred', 'orange', 'violet', 'brown', 'navy', 'teal']

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

        plot_data(game_results, 'game_results')
        plot_data(agent_returns, 'agent_returns')
        plot_data(train_episode_step, 'train_episode_step')
