import argparse

# List to keep track of input argument names
input_args_list = []


class CustomAction(argparse.Action):
    """
    Custom argparse action to store argument names in input_args_list.
    Appends the argument's destination name to input_args_list.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        # Ensure we are modifying the global input_args_list
        global input_args_list
        setattr(namespace, self.dest, values)
        input_args_list.append(self.dest)
        print(f"Argument {self.dest} added to input_args_list.")


def get_args():
    """
    Parses command-line arguments for configuring reinforcement learning experiments
    in single-agent environments.

    Returns:
        args: Parsed arguments namespace with all command-line configurations.
    """
    parser = argparse.ArgumentParser(description="Reinforcement Learning experiments for single-agent environments")

    # Configuration file argument
    parser.add_argument("--config-file", type=str, default="default_sac_2q.json",
                        help="Path to the configuration file containing experiment parameters.")

    # Task type argument
    parser.add_argument("--task-type", type=str, default="compare",
                        help="Type of task to run; choose from [train, evaluate, compare].")

    # Directory to save model and training state
    parser.add_argument("--save-dir", type=str, default="./model",
                        help="Directory to save training progress and model files.")

    # Environment setup arguments
    parser.add_argument("--scenario-name", action=CustomAction, type=str, default="simple",
                        help="Name of the environment scenario; options include [MountainCarContinuous-v0, Pendulum-v1, simple].")

    # GPU configuration
    parser.add_argument("--gpu-id", action=CustomAction, type=str, default="0",
                        help="ID of the GPU to use for training (e.g., '0' for the first GPU).")

    # Training parameters
    parser.add_argument("--train-episodes", action=CustomAction, type=int, default=2000,
                        help="Number of episodes to run for training.")

    # Model loading and saving options
    parser.add_argument("--load-pre-model", action=CustomAction, type=bool, default=False,
                        help="Whether to load an existing model checkpoint before training starts.")
    parser.add_argument("--force-save-model", action=CustomAction, type=bool, default=False,
                        help="Force saving the model at each display episode, regardless of performance.")
    parser.add_argument("--save-last-model", action=CustomAction, type=bool, default=True,
                        help="Save the final model at the end of training.")

    args = parser.parse_args()

    # Optionally, print the parsed arguments
    print("Parsed Arguments:")
    for arg in input_args_list:
        print(f"{arg}: {getattr(args, arg)}")

    return args
