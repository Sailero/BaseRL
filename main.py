import torch

from common.arguments import get_args
from common.utils import make_env, set_random_seed
from common.logger import Logger

if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    logger = Logger(args)
    set_random_seed(16)

    # Initialize the runner
    if args.policy_type in ['GAIL_SAC', 'GAIL_PPO', 'GAIL_PPO_combined']:
        from runner.gail_runner import GAILRunner

        runner = GAILRunner(args, env, logger)
    else:
        from runner.runner import Runner

        runner = Runner(args, env, logger)
    print("policy_type: ", args.policy_type)

    # Execute
    if args.evaluate:
        print("in evaluate mode!")
        runner.evaluate()
    elif args.compare:
        runner.compare_models_curves()
    elif args.imitation_learning:
        print("in imitation learning mode!")
        runner.imitation_learning()
    else:
        runner.run()
        if torch.cuda.is_available():
            print(f"Training finished using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Training finished using CPU")
        # runner.compare_models_curves()
