import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for MNIST digit classification.

    Architecture:
    - Conv(1 -> 32, 3x3) + ReLU + MaxPool
    - Conv(32 -> 64, 3x3) + ReLU + MaxPool
    - Flatten
    - FC(64*7*7 -> 128) + ReLU + Dropout
    - FC(128 -> 10)
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        # Input: (batch, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two times pooling: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)   # (batch, 32, 14, 14)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)   # (batch, 64, 7, 7)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 64*7*7)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, num_classes)

        return x
