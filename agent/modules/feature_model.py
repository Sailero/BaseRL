import torch.nn as nn


class FeatureModel(nn.Module):
    def __init__(self, name: str = 'shufflenet_v2'):
        super().__init__()

        if name == 'shufflenet_v2':
            from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
            self.model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
            self.model.fc = nn.Sequential()

        # 固定参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
