import torch

from common.arguments import get_args
from common.utils import make_env, set_random_seed
from common.logger import Logger
from runner.runner import Runner


if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    logger = Logger(args)
    set_random_seed(16)

    runner = Runner(args, env, logger)
    print("policy_type: ", args.policy_type)

    # Execute
    if args.task_type == "evaluate":
        print("in evaluate mode!")
        runner.evaluate()
    elif args.task_type == "compare":
        runner.compare_models_curves()
    else:
        runner.run()
        if torch.cuda.is_available():
            print(f"Training finished using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Training finished using CPU")
        # runner.compare_models_curves()
