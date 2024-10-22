import argparse


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for single agent environments")
    # 训练的环境
    parser.add_argument("--scenario-name", type=str, default="simple",
                        help="scenario name for simulation."
                             "one of [MountainCarContinuous-v0, Pendulum-v1, simple, forklift]")
    parser.add_argument("--gpu-id", type=str, default="0",
                        help="config which gpu to use")
    parser.add_argument("--render", type=bool, default=False,
                        help="whether to show the GUI of the environment")

    # forklift 环境的参数
    parser.add_argument("--pallet-random", type=bool, default=True,
                        help="whether to randomly place the pallet in the environment")
    parser.add_argument("--forklift-episode-len", type=int, default=100,
                        help="forklift number of episode steps")

    # 定义架构上的训练参数
    parser.add_argument("--train-episodes", type=int, default=1000,
                        help="number of time steps")
    parser.add_argument("--load-pre-model", type=bool, default=False,
                        help="whether to load the previous model")

    # 定义训练参数
    parser.add_argument("--policy-type", type=str, default='DDPG',
                        help="the policy type of single agent. one of PPO, DDPG, GAIL_PPO")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--actor_hidden_dim", type=int, default=128,
                        help="hidden dims of actor network")
    parser.add_argument("--critic_hidden_dim", type=int, default=128,
                        help="hidden dims of critic network")
    parser.add_argument("--lr-actor", type=float, default=5e-5,
                        help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-4,
                        help="learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")

    # DDPG的独有参数
    parser.add_argument("--buffer-size", type=int, default=int(1e5),
                        help="number of transitions can be stored in buffer")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1,
                        help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="parameter for updating the target network")

    # PPO的独有参数
    parser.add_argument("--lam", type=float, default=0.9,
                        help="coef for GAE")
    parser.add_argument("--eps-clip", type=float, default=0.2,
                        help="importance ratio parameters for clipping")
    parser.add_argument("--action-clip", type=float, default=1.0,
                        help="max amplitude actions allowed")
    parser.add_argument("--update-nums", type=int, default=4,
                        help="Number of steps required for each model update")
    parser.add_argument("--ent-coef", type=float, default=0.05,
                        help="coef for entropy loss")
    parser.add_argument("--value-loss-coef", type=float, default=0.1,
                        help="coef for the critic loss of the total loss")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="max grad norm")

    # 定义模仿学习的参数
    parser.add_argument("--imitation-learning", type=bool, default=False,
                        help="whether to do imitation learning")
    parser.add_argument("--im-sample-size", type=int, default=256,
                        help="number of transitions sampled from expert data each time")

    # GAIL 的训练参数
    parser.add_argument("--discr-hidden-dim", type=int, default=128,
                        help="hidden dims of discriminator network")
    parser.add_argument("--lr-discr", type=float, default=5e-5,
                        help="learning rate of discriminator")

    # 定义模型保存和加载的相关参数
    parser.add_argument("--save-dir", type=str, default="./model",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-last-model", type=bool, default=True,
                        help="whether to save the last model in training episodes")
    parser.add_argument("--load-dir", type=str, default=None,
                        help="directory in which training state and model should be saved")

    # 定义可视化的相关参数
    parser.add_argument("--compare", type=bool, default=False,
                        help="whether to compare or not")
    parser.add_argument("--evaluate", type=bool, default=True,
                        help="whether to evaluate or not")
    parser.add_argument("--evaluate-episodes", type=int, default=100,
                        help="number of episodes for evaluating")
    parser.add_argument("--display-episodes", type=int, default=100,
                        help="number of episodes for printing and plotting results")
    parser.add_argument("--force-save-model", type=bool, default=True,
                        help="force to save the model in each display episode whether the model is better or not")

    args = parser.parse_args()

    # 添加日志文件保存路径
    args.log_dir = "./logs/" + args.scenario_name + "/" + args.policy_type
    return args
