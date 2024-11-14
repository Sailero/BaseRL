import os
import torch
import torch.nn as nn

class ChkptModule(nn.Module):
    def __init__(self, config, network_type):
        """
        Initialize the checkpoint module with configuration and network type.

        Args:
            config: Configuration object containing save path.
            network_type: Type of network (used to name the checkpoint file).
        """
        super(ChkptModule, self).__init__()

        # Define checkpoint file path based on network type and save path
        agent_save_path = os.path.join(config.save_path, 'parameters')
        if not os.path.exists(agent_save_path):
            os.makedirs(agent_save_path)
        self.chkpt_file = os.path.join(agent_save_path, network_type)

    def save_checkpoint(self):
        """Save the current state of the model to the checkpoint file."""
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        """Load the model state from the checkpoint file."""
        self.load_state_dict(torch.load(self.chkpt_file, weights_only=True))
