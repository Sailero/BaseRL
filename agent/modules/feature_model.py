import torch.nn as nn

class FeatureModel(nn.Module):
    def __init__(self, name: str = 'shufflenet_v2'):
        """
        Initialize the feature extraction model with a specified architecture.

        Args:
            name: Name of the model architecture to use. Default is 'shufflenet_v2'.
        """
        super().__init__()

        # Load the ShuffleNet V2 model with pretrained weights and remove the fully connected layer
        if name == 'shufflenet_v2':
            from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
            self.model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
            self.model.fc = nn.Sequential()  # Remove fully connected layer for feature extraction

        # Freeze model parameters to prevent updates during training
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the feature model.

        Args:
            x: Input tensor.

        Returns:
            Output of the model after feature extraction.
        """
        return self.model(x)
