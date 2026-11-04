"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed


__all__ = ["Source"]


class Source(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: 3(a) - define each layer
        self.conv1 = None
        self.pool = None
        self.conv2 = None
        self.conv3 = None
        self.fc1 = None

        self.init_weights()
        raise NotImplementedError()

    def init_weights(self) -> None:
        """Initialize model weights."""
        set_random_seed()

        for conv in [self.conv1, self.conv2, self.conv3]:
            # TODO: 3(a) - initialize the parameters for the convolutional layers
            pass
        
        # TODO: 3(a) - initialize the parameters for [self.fc1]
        raise NotImplementedError() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward propagation for a batch of input examples. Pass the input array
        through layers of the model and return the output after the final layer.

        Args:
            x: array of shape (N, C, H, W) 
                N = number of samples
                C = number of channels
                H = height
                W = width

        Returns:
            z: array of shape (N, # output classes)
        """
        N, C, H, W = x.shape

        # TODO: 3(a) - forward pass
        raise NotImplementedError()
