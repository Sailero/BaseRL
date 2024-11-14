import argparse

input_args_list = []


class CustomAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        input_args_list.append(self.dest)


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for single agent environments")

    # 配置文件
    parser.add_argument("--config-file", type=str, default="default_sac_1q.json",
                        help="path to the configuration file")
    parser.add_argument("--task-type", type=str, default="train",
                        help="the task type of the model.one of the [train, evaluate, compare]")
    parser.add_argument("--save-dir", type=str, default="./model",
                        help="directory in which training state and model should be saved")

    # 训练环境
    parser.add_argument("--scenario-name", action=CustomAction, type=str, default="forklift_1d",
                        help="scenario name for simulation."
                             "one of [MountainCarContinuous-v0, Pendulum-v1, simple, forklift, forklift_1d]")
    parser.add_argument("--gpu-id", action=CustomAction, type=str, default="0",
                        help="config which gpu to use")
    # 训练相关参数
    parser.add_argument("--train-episodes", action=CustomAction, type=int, default=1000,
                        help="number of time steps")
    parser.add_argument("--load-pre-model", action=CustomAction, type=bool, default=False,
                        help="whether to load the previous model")
    parser.add_argument("--force-save-model", action=CustomAction, type=bool, default=False,
                        help="force to save the model in each display episode whether the model is better or not")
    parser.add_argument("--save-last-model", action=CustomAction, type=bool, default=True,
                        help="whether to save the last model in training episodes")

    args = parser.parse_args()
    return args
