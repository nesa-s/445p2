"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

LoRA (Low-Rank Adaptation) for the Target CNN
    Wraps each convolutional layer of a pretrained model with a trainable low-rank update,
    while keeping the pretrained weights frozen.
    Usage: from model.lora import add_lora
    (This file is provided in full -- you do not need to modify it.)
"""

from math import sqrt

import torch
import torch.nn as nn


__all__ = ["LoRAConv2d", "add_lora"]


class LoRAConv2d(nn.Module):
    """A frozen convolution plus a trainable low-rank update.

    Let W be the pretrained kernel of the wrapped convolution, reshaped into a
    (C_out) x (C_in * k * k) matrix. LoRA keeps W frozen and learns an update
    Delta W = B A, where A has shape (rank) x (C_in * k * k) and B has shape (C_out) x (rank):

        W' = W + (alpha / rank) * B A

    Instead of materializing B A, we compute the equivalent two-step convolution:
    A is a k x k convolution with `rank` output channels (same stride and padding as the
    original layer) and B is a 1 x 1 convolution from `rank` channels to C_out channels.
    The bias of the original layer is kept frozen.

    Initialization: A ~ N(0, 1 / (C_in * k * k)) and B = 0, so Delta W = 0 at the start
    and the model begins exactly at the pretrained function.
    """

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        assert rank >= 1, "rank must be a positive integer"
        self.base = base
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank

        self.lora_A = nn.Conv2d(
            base.in_channels, rank, base.kernel_size, base.stride, base.padding, bias=False
        )
        self.lora_B = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)

        fan_in = base.in_channels * base.kernel_size[0] * base.kernel_size[1]
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=1 / sqrt(fan_in))
        nn.init.zeros_(self.lora_B.weight)

        # the pretrained layer is frozen; only A and B are trained
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def add_lora(model: nn.Module, rank: int, alpha: float | None = None) -> nn.Module:
    """Add rank-`rank` LoRA adapters to conv1, conv2 and conv3 of a Target model (in place).

    All pretrained convolutional weights are frozen. The fully connected output layer
    (fc_1) is left unfrozen so it can be trained on the target task.
    Call this *after* loading the pretrained (source) weights into the model.
    """
    for name in ["conv1", "conv2", "conv3"]:
        setattr(model, name, LoRAConv2d(getattr(model, name), rank, alpha))
    return model
