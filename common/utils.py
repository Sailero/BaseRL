import torch
import os
import pandas as pd
import matplotlib.pyplot as plt


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

    # 获取训练的device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {args.device}")

    print('Agent observation dimension:', args.agent_obs_dim)
    print('Agent action dimension:', args.agent_action_dim)
    return env, args
