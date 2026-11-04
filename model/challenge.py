"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Challenge CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""

import torch.nn as nn


__all__ = ["Challenge"]


class Challenge(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        # TODO: Define your model here
        raise NotImplementedError()
