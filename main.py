import torch

from common.arguments import get_args
from common.utils import make_env, set_random_seed, load_config
from runner.runner import Runner


if __name__ == '__main__':
    """
    Main entry point for running the reinforcement learning task. This script handles loading arguments, 
    setting up the environment, configuring the agent, and choosing the appropriate task type (evaluation, 
    comparison, or training).

    It performs the following steps:
    1. Loads command-line arguments.
    2. Loads configuration from a file.
    3. Creates the environment and sets the random seed for reproducibility.
    4. Initializes the runner (agent) and starts training, evaluation, or comparison.
    """
    # 1. Get command-line arguments
    args = get_args()

    # 2. Load configuration from a file
    config = load_config(args)

    # 3. Initialize environment and update the config based on environment settings
    env, config = make_env(config)

    # 4. Set the random seed for reproducibility
    set_random_seed(16)

    # 5. Create the runner (which manages the training/evaluation process)
    runner = Runner(config, env)
    print("policy_type: ", config.policy_type)

    # 6. Execute task based on the specified task type (evaluate, compare, or run training)
    if config.task_type == "evaluate":
        print("in evaluate mode!")  # Print message when in evaluation mode
        runner.evaluate()  # Evaluate the model
    elif config.task_type == "compare":
        runner.compare_models_curves()  # Compare models' performance
    else:
        runner.run()  # Run the training process

        # 7. Print device used for training (GPU or CPU)
        if torch.cuda.is_available():
            print(f"Training finished using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Training finished using CPU")
        # runner.compare_models_curves()  # Optionally compare models' performance after training (commented out)
