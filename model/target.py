"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import Target
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Target"]


class Target(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: 2(b) - define each layer
        self.conv1 = None
        self.pool = None
        self.conv2 = None
        self.conv3 = None
        self.fc_1 = None

        self.init_weights()
        raise NotImplementedError()

    def init_weights(self) -> None:
        """Initialize model weights."""
        torch.manual_seed(42)
        for conv in [self.conv1, self.conv2, self.conv3]:
            # TODO: 2(b) - initialize the parameters for the convolutional layers
            pass

        # TODO: 2(b) - initialize the parameters for [self.fc_1]
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        # TODO: 2(b) - , forward pass
        raise NotImplementedError()
