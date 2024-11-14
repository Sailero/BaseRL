from tqdm import tqdm
import numpy as np
import time
import os
import pandas as pd
import matplotlib

matplotlib.use('agg')  # Use 'agg' backend for matplotlib when saving plots without a display
import matplotlib.pyplot as plt

from agent.agent import Agent
from common.utils import smooth, save_data, plot_returns_curves


class Runner:
    def __init__(self, config, env):
        """
        Initializes the Runner class, which is responsible for running the agent's training and evaluation loop.

        Args:
            config: Configuration object containing training parameters and paths.
            env: Environment object for the agent to interact with.
        """
        # Load environment
        self.env = env
        self.device = config.device.device  # Device configuration (e.g., CPU or GPU)

        # Load training-related parameters from the config
        self.max_episode_len = config.env.max_episode_len  # Maximum length of an episode
        self.train_episodes = config.env.train_episodes  # Total number of training episodes
        self.compare_path = config.save_path  # Path to save models and results
        self.expert_data_path = config.expert_data_path  # Path to expert data for training or comparison
        config.save_path = os.path.join(config.save_path,
                                        f"{config.policy_type}")  # Modify save path based on policy type

        # Set paths for storing plot results and data results
        self.plt_save_path = os.path.join(config.save_path, 'plt_results')
        self.data_save_path = os.path.join(config.save_path, 'data_results')

        # Load model-related flags from config
        self.load_pre_model = config.env.load_pre_model  # Whether to load a pretrained model
        self.save_last_model = config.env.save_last_model  # Whether to save the final model after training

        # Initialize the agent
        self.agent = Agent(config)

        # Load evaluation-related parameters
        self.evaluate_episodes = 100  # Number of episodes to evaluate the agent
        self.display_episodes = config.env.display_episodes  # Number of episodes to display training progress
        self.force_save_model = config.env.force_save_model  # Whether to forcefully save the model during training

        # Training-related initialization
        self.best_agent_return = -1e5  # Initialize the best agent reward with a very low value
        self.best_episodes = 0  # Track the episode number with the best reward

        # Create directories to save models and results if they do not exist
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        if not os.path.exists(self.plt_save_path):
            os.makedirs(self.plt_save_path)
        if not os.path.exists(self.data_save_path):
            os.makedirs(self.data_save_path)

    def save_run_data(self, episode, agent_returns, train_episode_step):
        """
        Saves training data, including the agent's return and training steps, and updates the best agent reward
        if a new best is achieved.

        Args:
            episode: Current episode number.
            agent_returns: List of agent returns (rewards) for the episodes.
            train_episode_step: List of steps taken in each episode during training.
        """
        # Calculate the average agent return over the last 'display_episodes' episodes
        avg_agent_returns = np.mean(agent_returns[-self.display_episodes:])

        # If the current average return is the best so far, save the model
        if avg_agent_returns > self.best_agent_return:
            self.agent.save_models()  # Save the model when a new best reward is achieved

            # Update the best agent return and episode number
            self.best_agent_return = avg_agent_returns
            self.best_episodes = episode

            # Print the details of the new best reward
            print(
                f"The reward is the most of the history when episode is {episode + 1}, "
                f"best_agent_return is {self.best_agent_return}")
        elif self.force_save_model:
            # If forced to save, save the model even if it's not the best
            self.agent.save_models()

        # Save the agent's reward data to CSV
        save_data(self.data_save_path, agent_returns, csv_name="agent_returns", column_name=["ReturnsForAgent"])

        # Save the training episode steps data to CSV
        save_data(self.data_save_path, train_episode_step, csv_name="train_episode_step",
                  column_name=['EachEpisodeSteps'])

        # Plot training return curves every 100 episodes
        plot_returns_curves(agent_returns, self.plt_save_path)

        # Print training information for the current episode
        print()
        print(f'Episode: {episode + 1}')
        print(f'\nAverage episode steps: {np.mean(train_episode_step[-self.display_episodes:]):.1f}')
        print(f'Agent average returns: {avg_agent_returns:.1f}')


    def run(self):
        """
        The training loop for the agent, handling the environment interaction, collecting rewards,
        and updating the agent's model.

        This function runs for a specified number of training episodes and steps through each episode
        while training the agent using reinforcement learning.
        """
        # 1. Preparation for training
        # Initialize lists to store agent returns (rewards) and episode steps
        agent_returns = []
        train_episode_step = []

        # Load previously trained models if specified in the configuration
        if self.load_pre_model:
            self.agent.load_models()

        # 2. Start the training process for the specified number of episodes
        for episode in tqdm(range(self.train_episodes)):
            # 2.1 Initialize the environment for a new episode
            obs = self.env.reset()  # Reset environment to the initial state for the episode

            # Initialize reward for the current episode
            agent_episode_reward = 0

            # 2.2 Iterate through steps within the episode
            for step in range(self.max_episode_len):
                # 2.2.1 Agent chooses an action based on the current observation
                action = self.agent.choose_action(obs)

                # 2.2.2 Environment updates its state based on the agent's action
                next_obs, reward, done, info = self.env.step(action)

                # Ensure the episode ends if we reach the maximum step limit
                if step == self.max_episode_len - 1:
                    done = True

                # 2.2.3 Store the transition in the agent's buffer (depending on the algorithm's requirements)
                self.agent.buffer.store_episode(obs, action, reward, next_obs, done)

                # 2.2.4 Update the current observation to the next observation
                obs = next_obs

                # 2.2.5 If using offline training, train the agent when the buffer is ready
                if not self.agent.online_policy and self.agent.buffer.ready():
                    self.agent.train()

                # 2.2.6 Accumulate the reward for the current episode
                agent_episode_reward += reward

                # 2.2.7 Check if the episode is done (e.g., if the agent has reached a terminal state)
                if done:
                    break

            # 2.3 Record the episode reward and the number of steps for this episode
            agent_returns.append(agent_episode_reward)
            train_episode_step.append(step + 1)

            # 2.4 Train the agent if using an online policy
            if self.agent.online_policy:
                self.agent.train()

            # 2.5 Save model and training data periodically
            if (episode + 1) % self.display_episodes == 0:
                self.save_run_data(episode, agent_returns, train_episode_step)

        # 3.1 Save the model once training is complete
        if self.save_last_model:
            self.agent.save_models()  # Save the final model after training finishes

        # 3.2 Print important information about the best reward achieved during training
        print(
            f"The best reward of agent is {round(self.best_agent_return, 2)} when episode is {self.best_episodes + 1}")


    def evaluate(self):
        """
        Evaluates the agent's performance by running episodes using a pre-trained model.
        The agent interacts with the environment, and the total reward for each episode is recorded.

        This function loads the pre-trained model, runs a set number of evaluation episodes, and renders
        the environment for visual inspection.
        """
        # 1. Load the pre-trained model
        self.agent.load_models()

        # 2. Initialize a list to store agent's rewards for evaluation episodes
        agent_returns = []

        # 3. Start the evaluation process by running evaluation episodes
        for episode in tqdm(range(self.evaluate_episodes)):
            # 3.1 Initialize the environment for a new evaluation episode
            obs = self.env.reset()

            # 3.2 Initialize reward for the current episode
            agent_episode_reward = 0

            # 3.3 Iterate through steps within the episode
            for step in range(self.max_episode_len):
                # 3.3.1 Render the environment to visualize the agent's actions
                self.env.render()
                time.sleep(0.01)  # Slow down the rendering for better visualization

                # 3.3.2 Agent chooses an action based on the current observation
                action = self.agent.choose_action(obs)

                # 3.3.3 Update the environment's state based on the chosen action
                next_obs, reward, done, info = self.env.step(action)

                # Ensure the episode ends when the maximum step length is reached
                if step == self.max_episode_len - 1:
                    done = True

                # 3.3.4 Update the observation to the next state
                obs = next_obs

                # 3.3.5 Accumulate the reward for the episode
                agent_episode_reward += reward

                # 3.3.6 Check if the episode has ended (either by the environment or reaching the max steps)
                if done:
                    break

            # 3.4 Print the total reward for the current episode
            print(f"Episode {episode + 1}'s episode reward: {agent_episode_reward}")

            # 3.5 Add the episode's reward to the list of agent returns
            agent_returns.append(agent_episode_reward)

    def compare_models_curves(self):
        """
        Compares the performance of different models by plotting curves for agent returns and training steps
        from multiple runs stored in separate folders. The data is loaded, smoothed, and visualized for comparison.

        This function retrieves the data from saved results in subfolders, processes the data, and plots
        the comparison curves for agent returns and training episode steps.
        """

        # Helper function to retrieve data from saved results in subfolders
        def get_data(name):
            """
            Loads the data for a specific metric (e.g., 'agent_returns' or 'train_episode_step') from all subfolders.

            Args:
                name: The name of the data file (without extension) to load (e.g., "agent_returns").

            Returns:
                data: A numpy array containing the loaded data from all subfolders.
                folder_list: A list of folder names corresponding to each data entry.
            """
            data = []
            folder_list = []
            for folder in os.listdir(self.compare_path):
                folder_path = os.path.join(self.compare_path, folder, 'data_results', f'{name}.csv')
                # Check if the file exists
                if os.path.isfile(folder_path):
                    matrix = np.array(pd.read_csv(folder_path))
                    data.append(matrix)
                    folder_list.append(folder)
            data = np.stack(data, axis=0)  # Stack data into a 3D array (folders x episodes x values)
            return data, folder_list

        # Retrieve agent returns and training episode steps data
        agent_returns, folder_list = get_data("agent_returns")
        train_episode_step, _ = get_data("train_episode_step")

        # Create directory to save the comparison results
        compare_results_path = os.path.join(self.compare_path, 'compare_results')
        if not os.path.exists(compare_results_path):
            os.makedirs(compare_results_path)

        # Function to plot the data (returns or steps) for comparison
        def plot_data(data, name):
            """
            Plots the comparison curves for the given data (agent returns or training steps).

            Args:
                data: The data array to plot (e.g., agent returns or episode steps).
                name: The name of the metric (e.g., 'agent_returns' or 'train_episode_step') for the plot title.
            """
            plt.figure()  # Create a new figure for plotting
            alpha = [0.1, 1]  # Transparency levels for different plots
            weight = [0.8, 0.96]  # Smoothing weights
            linewidth = [4, 0.5]  # Line widths for the plots
            colors = ['blue', 'red', 'green', 'darkred', 'orange', 'violet', 'brown', 'navy',
                      'teal']  # Colors for the plots

            # Loop through the data to plot each model's results
            for j in range(data.shape[2]):  # Loop through different metrics (if applicable)
                for i in range(data.shape[0]):  # Loop through each folder (model)
                    # Plot the data with smoothing for each model
                    plt.plot(range(len(data[i, :, j])), smooth(data[i, :, j], weight=weight[0]),
                             color=colors[i], linewidth=linewidth[0], alpha=alpha[0])
                    plt.plot(range(len(data[i, :, j])), smooth(data[i, :, j], weight=weight[1]), label=folder_list[i],
                             color=colors[i], linewidth=linewidth[1], alpha=alpha[1])
                plt.legend()  # Display the legend with folder names
                plt.title(f'Different policies {name}')  # Set the title for the plot
                plt.xlabel('Episode')  # Label for the x-axis
                plt.ylabel(f"log({name}'s rewards)")  # Label for the y-axis

                # Determine the path to save the plot image
                path = os.path.join(compare_results_path, f"{name}_comparison.png")
                plt.savefig(path)  # Save the figure as a PNG file
                plt.close()  # Close the figure to avoid memory overflow

        # Plot the comparison curves for agent returns and training steps
        plot_data(agent_returns, 'agent_returns')
        plot_data(train_episode_step, 'train_episode_step')

