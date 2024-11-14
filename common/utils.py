import torch
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from common.config import Config
from common.arguments import input_args_list


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
    plt.title(f'agent train returns')
    plt.xlabel('episode')
    plt.ylabel('each episode average return')

    # 获取图片保存地址
    pic_save_path = os.path.join(plt_save_path, f'train_returns.png')
    plt.savefig(pic_save_path, format='png')
    plt.clf()
    plt.close()


def save_expert_data(expert_path, file_name, expert_data):
    if len(expert_data) == 0:
        print("data is empty, skip saving.")
        return

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
        print(f"Data has been saved or appended to {file_path}. len {len(expert_data)} -> {combined_data.shape}")
    else:
        # 如果文件不存在，则直接保存expert_data
        np.save(file_path, expert_data)
        print(f"Data has been saved or appended to {file_path}. len: {len(expert_data)}")


# 定义二维情形的AC网络
def get_conv_out_size(shape, net):
    # Pass dummy input to get the output size of the conv layers
    shape = [3] + shape
    o = torch.zeros(1, *shape)
    o = net(o)
    return int(torch.prod(torch.tensor(o.shape[1:])))


def set_random_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        torch.backends.cudnn.deterministic = True  # to ensure deterministic results
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  # for hash-based functions


def load_config(args):
    config_path = os.path.join("./config", f"{args.config_file}")
    with open(config_path, 'r') as f:
        algo_config = Config(json.load(f))

    with open("./config/default_common.json", 'r') as f:
        config = Config(json.load(f))

    # 合并 PPO 参数到 common 配置中
    for key, value in algo_config.items():
        if isinstance(value, dict) and key in config:
            # 递归更新嵌套字典，确保以 algo_config 为主
            config[key].update(value)
        else:
            # 对于顶层键，直接使用 algo_config 的值
            config[key] = value

    # 只保留 name 对应的 train 配置，并赋值到 env
    config["env"] = config["train"].get(args.scenario_name, {})
    config["env"]["name"] = args.scenario_name

    # 删除原有的 train 配置
    del config["train"]

    for key, value in config["env"].items():
        if key in config["params"]:
            config["params"][key] = value

    # 从 args 覆盖 config 参数
    for item in input_args_list:
        if item in config.train:
            config.train[item] = args.__dict__[item]
        elif item in config.device:
            config.device[item] = args.__dict__[item]
        elif item in config.scenario:
            config.env[item] = args.__dict__[item]

    config.task_type = args.task_type
    config.save_dir = args.save_dir
    config.log_dir = "./logs/" + config.env.name + "/" + config.policy_type

    return config


def make_env(config):
    # 创建环境
    from env.env import Env
    env = Env(config)

    # 获取观测信息
    config.env.agent_obs_dim = env.agent_obs_dim

    # 获取动作信息
    config.env.agent_action_dim = env.agent_action_dim
    config.env.max_episode_len = env.max_episode_len

    config.imitation_learning = True if config.policy_type in ["GAIL_PPO", "GAIL_SAC"] else False

    # 获取训练中的保存路径
    config.save_path = os.path.join(config.save_dir, config.env.name)
    config.expert_data_path = os.path.join("./expert_data", f"{config.env.name}")

    print("config: ", config)

    # 获取训练的device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device.gpu_id
    config.device.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {config.device.device}")

    return env, config
