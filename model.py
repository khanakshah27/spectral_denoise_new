import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNetThreshold(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial = nn.Conv1d(1, 32, kernel_size=7, padding=3)

        self.block1 = ResidualBlock1D(32)
        self.block2 = ResidualBlock1D(32)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.initial(x))
        x = self.block1(x)
        x = self.block2(x)

        x = self.global_pool(x).squeeze(-1)
        threshold = torch.sigmoid(self.fc(x)) * 20  # Threshold range: 0–20%

        return threshold