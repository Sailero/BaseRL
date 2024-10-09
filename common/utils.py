import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def smooth(data, weight=0.96):
    sm_val = data[0]
    sm_list = []
    for val in data:
        sm_val_ = sm_val * weight + (1 - weight) * val
        sm_list.append(sm_val_)
        sm_val = sm_val_
    return sm_list


def save_data(save_path, data_list, column_name, csv_name):
    df = pd.DataFrame(data_list, columns=column_name)
    path = os.path.join(save_path, f"{csv_name}.csv")
    df.to_csv(path, index=False)


def plot_returns_curves(agent_returns, plt_save_path):
    # 绘图
    plt.figure()
    plt.plot(range(len(agent_returns)), smooth(agent_returns, weight=0.8), label='origin',
             linewidth=4, alpha=0.1, c='blue')
    plt.plot(range(len(agent_returns)), smooth(agent_returns, weight=0.96), label='smooth',
             linewidth=0.5, alpha=1, c='blue')
    plt.legend()
    plt.title(f'Forklift agent train returns')
    plt.xlabel('episode')
    plt.ylabel('each episode average return')

    # 获取图片保存地址
    pic_save_path = os.path.join(plt_save_path, f'train_returns.png')
    plt.savefig(pic_save_path, format='png')
    plt.clf()
    plt.close()


def save_expert_data(expert_path, file_name, expert_data):
    # 确保expert_data为numpy数组，以便处理多维数据
    expert_data = np.array(expert_data)

    # 确保expert_path存在
    os.makedirs(expert_path, exist_ok=True)
    file_path = os.path.join(expert_path, f"{file_name}.npy")

    if os.path.exists(file_path):
        # 如果文件存在，则加载已有数据并进行拼接
        existing_data = np.load(file_path)

        # 进行拼接，假设数据在第0维进行拼接
        combined_data = np.concatenate((existing_data, expert_data), axis=0)

        # 保存拼接后的数据
        np.save(file_path, combined_data)
    else:
        # 如果文件不存在，则直接保存expert_data
        np.save(file_path, expert_data)

    print(f"Data has been saved or appended to {file_path}.")


def make_env(args):
    # 创建环境
    from env.env import Env
    env = Env(args)

    # 获取观测信息
    args.agent_obs_dim = env.agent_obs_dim

    # 获取动作信息
    args.agent_action_dim = env.agent_action_dim

    args.max_episode_len = env.max_episode_len

    # 获取训练中的保存路径
    args.save_path = os.path.join(args.save_dir, args.scenario_name)
    args.imitation_learning_path = os.path.join(args.save_dir, 'expert_data')

    # 获取训练的device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {args.device}")

    print('Agent observation dimension:', args.agent_obs_dim)
    print('Agent action dimension:', args.agent_action_dim)
    return env, args
