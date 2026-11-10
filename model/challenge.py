"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Challenge CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


__all__ = ["Challenge"]


class ChallengeCNN(nn.Module):
    """Same convolutional stack as the source and target models, with a 2-class head.

    The conv layers are sized so the source model's weights can be loaded into them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 8, kernel_size=5, stride=2, padding=2)
        self.fc_1 = nn.Linear(32, 2)
        self.init_weights()

    def init_weights(self) -> None:
        torch.manual_seed(42)
        for conv in (self.conv1, self.conv2, self.conv3):
            std = sqrt(1 / (5 * 5 * conv.in_channels))
            nn.init.normal_(conv.weight, mean=0.0, std=std)
            nn.init.constant_(conv.bias, 0.0)
        fc_std = sqrt(1 / self.fc_1.in_features)
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=fc_std)
        nn.init.constant_(self.fc_1.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        return self.fc_1(x)


class Challenge(nn.Module):
    """Five source-initialized networks whose class probabilities are averaged.

    Each member keeps the source convolutional features frozen and learns only
    its linear head. Averaging the five heads is done in probability space.
    The returned tensor is the log of those probabilities, so the softmax in
    predict_challenge.py recovers the averaged probability.
    """

    def __init__(self, n_members: int = 5) -> None:
        super().__init__()
        self.members = nn.ModuleList(ChallengeCNN() for _ in range(n_members))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = torch.stack(
            [torch.softmax(member(x), dim=1) for member in self.members],
            dim=0,
        ).mean(dim=0)
        return torch.log(probabilities.clamp(min=1e-8))
