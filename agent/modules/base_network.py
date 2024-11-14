import os

import torch
import torch.nn as nn


class ChkptModule(nn.Module):
    def __init__(self, config, network_type):
        super(ChkptModule, self).__init__()

        name = network_type
        agent_save_path = os.path.join(config.save_path, 'parameters')
        if not os.path.exists(agent_save_path):
            os.makedirs(agent_save_path)
        self.chkpt_file = os.path.join(agent_save_path, name)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file, weights_only=True))
