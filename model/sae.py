"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Sparse Autoencoder
    Constructs a pytorch model for a sparse autoencoder (SAE). The SAE is trained to
    reconstruct the hidden activations of a CNN using a small number of active latents.
    Usage: from model.sae import SAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["SAE"]


class SAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int) -> None:
        """
        Args:
            d_in: dimension of the CNN activations being reconstructed
            d_hidden: number of SAE latents (usually several times larger than d_in)
        """
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in)

        # Activations are rescaled so their average L2 norm is sqrt(d_in). This keeps the
        # sparsity penalty on a similar scale no matter which CNN the activations came from.
        self.register_buffer("scale", torch.tensor(1.0))

    def fit_scale(self, x: torch.Tensor) -> None:
        """Set the input scale from a batch of activations of shape (N, d_in)."""
        self.scale.fill_(x.shape[1] ** 0.5 / x.norm(dim=1).mean().item())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map activations of shape (N, d_in) to non-negative latents of shape (N, d_hidden)."""
        return F.relu(self.encoder(x * self.scale - self.decoder.bias))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Map latents of shape (N, d_hidden) back to activations of shape (N, d_in)."""
        return self.decoder(f) / self.scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f

    def loss(self, x: torch.Tensor, l1_coeff: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (total loss, reconstruction loss, sparsity loss) for activations x.

        The sparsity penalty weights each latent by the norm of its decoder column, so the
        SAE can't shrink the penalty by making latents small and decoder columns large.
        """
        f = self.encode(x)
        x_hat = self.decoder(f)
        recon = (x_hat - x * self.scale).pow(2).sum(dim=1).mean()
        sparsity = (f * self.decoder.weight.norm(dim=0)).sum(dim=1).mean()
        return recon + l1_coeff * sparsity, recon, sparsity
