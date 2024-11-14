# Main Parameters

- **task-type**: Task type, options are `[train, compare, evaluate]`
  - `train`: RL training mode
  - `compare`: Comparison analysis mode. Collects training data of various algorithms and plots training curves for comparison. Ensure that all algorithms use the same `episodes`.
  - `evaluate`: Evaluation mode, performs model evaluation and visualization.

- **scenario-name**: Environment name, supported types are:
    - `MountainCarContinuous-v0`: A mountain climbing car environment, characterized by sparse rewards and initial misleading rewards.
    - `Pendulum-v1`: A swinging pendulum environment with continuous rewards, making it easy to train.
    - `simple`: Single-agent environment in MPE, simple objectives, continuous rewards, fewer steps, and easy to train.

- **train-episodes**: Number of training episodes

- **load-pre-model**: Whether to load a previously trained model. Useful for resuming training after pausing.

# Supported Algorithms

- `PPO`: Proximal Policy Optimization, suitable for on-policy training
- `DDPG`: Deep Deterministic Policy Gradient, suitable for off-policy training in continuous action spaces
- `SAC`: Soft Actor-Critic, an off-policy method balancing exploration and exploitation
- `SAC_2Q`: Soft Actor-Critic with two Critic networks to balance exploration and exploitation
- `GAIL+PPO`: Generative Adversarial Imitation Learning combined with PPO, used for imitation and reinforcement learning
- `GAIL+SAC`: Generative Adversarial Imitation Learning combined with SAC, used for imitation and reinforcement learning

# Algorithm Architecture

- **agent**
    - `modules`: Basic components of each algorithm, including `base_network`, `feature_model`, etc.
    - `on-policy`: Implementation of on-policy algorithms, currently supports `PPO`
    - `off-policy`: Implementation of off-policy algorithms, currently supports `DDPG` and `SAC`
    - `imitation-learning`: Implementation of imitation learning modules, currently includes `GAIL+PPO` and `GAIL+SAC`
    - `agent.py`: Interface file encapsulating algorithm modules, used for training and evaluation calls.

- **common**
    - `arguments`: Stores training-related parameters
    - `utils`: Utility functions used in the `runner` or main function
    - `config`: Configurations basic class for algorithms and environments

- **config**
    - `default_common.json`: Specific parameters for each environment, configuration file for basic parameters of each algorithm. The algorithm parameters in the environment have the highest priority.
    - `default_**.json`: Configuration files for other algorithms, with a higher priority than basic parameters.

- **env**
    - `mpe`: Multi-agent particle environment, including the trainable `simple` environment
    - `env.py`: Interface file encapsulating environments, used to call different simulation environments.

- **model**: Folder for storing trained models

- **runner**
    - `runner.py`: Main running module responsible for environment initialization, algorithm invocation, and training flow management.

- **main**: Main project execution file

---




