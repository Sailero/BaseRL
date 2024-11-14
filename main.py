import torch

from common.arguments import get_args
from common.utils import make_env, set_random_seed, load_config
from runner.runner import Runner


if __name__ == '__main__':
    # 获取命令行参数
    args = get_args()
    # 加载配置文件
    config = load_config(args)

    env, config = make_env(config)
    set_random_seed(16)

    runner = Runner(config, env)
    print("policy_type: ", config.policy_type)

    # Execute
    if config.task_type == "evaluate":
        print("in evaluate mode!")
        runner.evaluate()
    elif config.task_type == "compare":
        runner.compare_models_curves()
    else:
        runner.run()
        if torch.cuda.is_available():
            print(f"Training finished using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Training finished using CPU")
        # runner.compare_models_curves()
