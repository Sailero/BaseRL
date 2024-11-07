import argparse


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for single agent environments")
    # 训练的环境
    parser.add_argument("--scenario-name", type=str, default="simple",
                        help="scenario name for simulation."
                             "one of [MountainCarContinuous-v0, Pendulum-v1, simple, forklift]")
    parser.add_argument("--gpu-id", type=str, default="0",
                        help="config which gpu to use")


    # 定义架构上的训练参数
    parser.add_argument("--train-episodes", type=int, default=1000,
                        help="number of time steps")
    parser.add_argument("--load-pre-model", type=bool, default=False,
                        help="whether to load the previous model")
    parser.add_argument("--policy-type", type=str, default='GAIL_PPO',
                        help="the policy type of single agent. one of PPO, DDPG, SAC, GAIL_PPO, GAIL_SAC")

    # 定义模型保存和加载的相关参数
    parser.add_argument("--save-dir", type=str, default="./model",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-last-model", type=bool, default=True,
                        help="whether to save the last model in training episodes")
    parser.add_argument("--load-dir", type=str, default=None,
                        help="directory in which training state and model should be saved")

    # 定义可视化的相关参数
    parser.add_argument("--task-type", type=str, default="train",
                        help="the task type of the model."
                             "one of the [train, evaluate, compare]")
    parser.add_argument("--display-episodes", type=int, default=100,
                        help="number of episodes for printing and plotting results")
    parser.add_argument("--force-save-model", type=bool, default=True,
                        help="force to save the model in each display episode whether the model is better or not")

    args = parser.parse_args()

    # 添加日志文件保存路径
    args.log_dir = "./logs/" + args.scenario_name + "/" + args.policy_type
    return args
