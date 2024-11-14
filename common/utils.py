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
    """
    Applies exponential smoothing to data to reduce noise.

    Args:
        data (list or array): List of data points to be smoothed.
        weight (float): Smoothing factor, where higher values give more weight to previous data points.

    Returns:
        list: Smoothed data, same length as the input.
    """
    sm_val = data[0]
    sm_list = []
    for val in data:
        sm_val = sm_val * weight + (1 - weight) * val
        sm_list.append(sm_val)
    return sm_list


def save_data(save_path, data_list, column_name, csv_name):
    """
    Saves a list of data to a CSV file.

    Args:
        save_path (str): Directory to save the file.
        data_list (list): List of data to be saved.
        column_name (list): List of column names.
        csv_name (str): Name of the CSV file.

    """
    df = pd.DataFrame(data_list, columns=column_name)
    path = os.path.join(save_path, f"{csv_name}.csv")
    df.to_csv(path, index=False)


def plot_returns_curves(agent_returns, plt_save_path):
    """
    Plots and saves a curve of agent returns over episodes.

    Args:
        agent_returns (list): List of returns for each episode.
        plt_save_path (str): Directory path to save the plot image.
    """
    plt.figure()

    # Plot original returns curve with slight transparency
    plt.plot(range(len(agent_returns)), smooth(agent_returns, weight=0.8), label='origin',
             linewidth=4, alpha=0.1, c='blue')

    # Plot smoothed returns curve
    plt.plot(range(len(agent_returns)), smooth(agent_returns, weight=0.96), label='smooth',
             linewidth=0.5, alpha=1, c='blue')

    # Add labels and title
    plt.legend()
    plt.title('Agent Training Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')

    # Save the plot to the specified path
    plt.savefig(os.path.join(plt_save_path, "returns_curve.png"))
    plt.close()


def save_expert_data(expert_path, file_name, expert_data):
    """
    Saves or appends expert data to a numpy file.

    Args:
        expert_path (str): Directory to save the file.
        file_name (str): Name of the file.
        expert_data (array): Expert data to save.
    """
    if len(expert_data) == 0:
        print("Data is empty, skipping save.")
        return

    # Convert expert data to numpy array if it's not already
    expert_data = np.array(expert_data)

    # Ensure the directory exists
    os.makedirs(expert_path, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(expert_path, f"{file_name}.npy")

    # If file exists, append the new data; otherwise, save the new data
    if os.path.exists(file_path):
        existing_data = np.load(file_path)
        combined_data = np.concatenate((existing_data, expert_data), axis=0)
        np.save(file_path, combined_data)
        print(f"Data saved and appended to {file_path}. Length: {combined_data.shape}")
    else:
        np.save(file_path, expert_data)
        print(f"Data saved to {file_path}. Length: {len(expert_data)}")


def get_conv_out_size(shape, net):
    """
    Calculates the output size of a convolutional network for a given input shape.

    Args:
        shape (list): Input shape.
        net: The convolutional neural network.

    Returns:
        int: Flattened output size of the network.
    """
    shape = [3] + shape  # Assuming 3 channels (RGB) for input
    o = torch.zeros(1, *shape)  # Create a dummy input tensor
    o = net(o)  # Pass it through the network
    return int(torch.prod(torch.tensor(o.shape[1:])))  # Flatten the output and return its size


def set_random_seed(seed):
    """
    Sets random seeds for reproducibility across libraries and environments.

    Args:
        seed (int): Seed value for random number generators.
    """
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch seed for CPU

    # If CUDA is available, set GPU seeds and deterministic options
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Single GPU seed
        torch.cuda.manual_seed_all(seed)  # Multi-GPU seed
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disables auto-tuning for determinism


def load_config(args):
    """
    Loads and merges configuration files with command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        config: Config object with loaded and merged configurations.
    """
    # Load the algorithm-specific configuration file
    config_path = os.path.join("./config", args.config_file)
    with open(config_path, 'r') as f:
        algo_config = Config(json.load(f))

    # Load the default common configuration
    with open("./config/default_common.json", 'r') as f:
        config = Config(json.load(f))

    # Merge algorithm-specific config into the common config
    for key, value in algo_config.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value

    # Set environment-specific configuration based on the scenario name
    config["env"] = config["train"].get(args.scenario_name, {})
    config["env"]["name"] = args.scenario_name

    # Remove the 'train' configuration after setting the environment parameters
    del config["train"]

    # Transfer environment-specific settings into the global config params
    for key, value in config["env"].items():
        if key in config["params"]:
            config["params"][key] = value

    # Override config parameters with command-line arguments
    for item in input_args_list:
        if item in config.env:
            config.env[item] = args.__dict__[item]
        elif item in config.device:
            config.device[item] = args.__dict__[item]

    # Set task type and directories for saving and logging
    config.task_type = args.task_type
    config.save_dir = args.save_dir
    config.log_dir = f"./logs/{config.env.name}/{config.policy_type}"

    return config


def make_env(config):
    """
    Creates and initializes the environment based on the provided configuration.

    Args:
        config: Configuration object containing environment and device settings.

    Returns:
        env: Initialized environment instance.
        config: Updated configuration object with environment-specific details.
    """
    # Import environment class and create environment instance
    from env.env import Env
    env = Env(config)

    # Update configuration with environment-specific information
    config.env.agent_obs_dim = env.agent_obs_dim  # Observation space dimensions
    config.env.agent_action_dim = env.agent_action_dim  # Action space dimensions
    config.env.max_episode_len = env.max_episode_len  # Maximum length of an episode

    # Set flag for imitation learning if the policy type is GAIL-based
    config.imitation_learning = config.policy_type in ["GAIL_PPO", "GAIL_SAC"]

    # Define paths for saving the model and expert data
    config.save_path = os.path.join(config.save_dir, config.env.name)
    config.expert_data_path = os.path.join("./expert_data", f"{config.env.name}")

    # Print the current configuration for debugging purposes
    print("Config:", config)

    # Set CUDA device for GPU usage (if available)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device.gpu_id
    config.device.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {config.device.device}")

    return env, config

