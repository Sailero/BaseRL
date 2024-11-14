import torch
from agent.agent_config import POLICY_MAP

def load_class(path):
    """
    Dynamically imports a class or function from a given module path.
    Args:
        path (str): Dot-separated module path to the class or function.
    Returns:
        The imported class or function.
    """
    components = path.split('.')
    module = __import__('.'.join(components[:-1]), fromlist=[components[-1]])
    return getattr(module, components[-1])

class Agent:
    """
    Base class for the agent, responsible for initializing policy and buffer,
    choosing actions, training, and model checkpoint management.
    """
    def __init__(self, config):
        """
        Initializes the agent with configurations.
        Args:
            config: Configuration object containing policy type, device,
                    and environment-specific observation and action dimensions.
        """
        self.policy_type = config.policy_type
        self.device = config.device.device
        self.agent_obs_dim = config.env.agent_obs_dim
        self.agent_action_dim = config.env.agent_action_dim

        # Load policy and buffer based on the policy type
        policy_config = POLICY_MAP.get(self.policy_type)
        if policy_config:
            self.policy = load_class(policy_config['policy'])(config)
            self.buffer = load_class(policy_config['buffer'])(config)
            self.online_policy = policy_config['online_policy']
        else:
            raise ValueError(f"Unsupported policy type: {self.policy_type}")

    def choose_action(self, observation):
        """
        Converts the observation to a tensor and selects an action using the policy.
        Args:
            observation: The current state observation of the agent.
        Returns:
            The selected action.
        """
        observation = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        action = self.policy.choose_action(observation)
        return action

    def train(self):
        """
        Samples transitions from the buffer and trains the policy.
        If online policy is enabled, re-initializes the buffer after training.
        """
        transitions = self.buffer.sample()
        self.policy.train(transitions)

        if self.online_policy:
            self.buffer.initial_buffer()

    def save_models(self):
        """Saves the current model checkpoints for the policy."""
        print('... saving agent checkpoint ...')
        self.policy.save_models()

    def load_models(self):
        """Loads model checkpoints for the policy."""
        print('... loading agent checkpoint ...')
        self.policy.load_models()
