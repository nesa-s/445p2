"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Train SAE
    Train a sparse autoencoder (SAE) on the CONV1 activations of a trained CNN.
    Every spatial position of the CONV1 output (a 32x32 grid for a 64x64 image) is a
    separate 16-dimensional training example, so each SAE latent can later be mapped
    back to the small patch of the image it came from. Trains an SAE for the target model,
    then one for the source model, stopping each one early based on validation loss.
    Plots the train/val loss and average number of active latents during training, and
    saves each plot to sae_training_plot_{target,source}.png. Close the target SAE's plot
    to start training the source SAE. Run visualize_sae.py afterwards.

    Usage: python train_sae.py
"""

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import DogsDataset, get_train_val_test_datasets
from model.sae import SAE
from model.source import Source
from model.target import Target
from train_common import early_stopping, restore_checkpoint
from utils import config, set_random_seed


__all__ = ["MODELS", "get_activations", "sae_path"]


# Maps each model name to its class and checkpoint directory
MODELS = {
    "target": (Target, config("target.checkpoint")),
    "source": (Source, config("source.checkpoint")),
}


def sae_path(name: str) -> str:
    """Return the file where the SAE for model `name` is saved."""
    return os.path.join(config("sae.checkpoint"), f"{name}.pth.tar")


def get_activations(model: torch.nn.Module, X: npt.NDArray) -> torch.Tensor:
    """Return the CONV1 activations (after ReLU) of `model` on images X, shape (N, 16, 32, 32)."""
    activations = []
    hook = model.conv1.register_forward_hook(lambda module, inp, out: activations.append(F.relu(out)))
    model.eval()
    with torch.no_grad():
        model(torch.from_numpy(X).float())
    hook.remove()
    return activations[0]


def make_sae_training_plot(name: str) -> list[list[plt.Line2D]]:
    """Set up an interactive plot of the SAE loss and average active latents, one line per split."""
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{name.capitalize()} SAE Training")
    lines = []
    for ax, metric in zip(axes, ["Reconstruction Error", "Avg Active Latents"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        lines.append([
            ax.plot([], [], linestyle="--", marker="o", color="r", label="Validation")[0],
            ax.plot([], [], linestyle="--", marker="x", color="b", label="Train")[0],
        ])
        ax.legend()
    return lines


def update_sae_training_plot(lines: list[list[plt.Line2D]], stats: list[list[float]]) -> None:
    """Redraw the plot with stats, where each entry is [train loss, val loss, train active, val active]."""
    epochs = range(len(stats))
    for i, (val_line, train_line) in enumerate(lines):
        train_line.set_data(epochs, [stat[2 * i] for stat in stats])
        val_line.set_data(epochs, [stat[2 * i + 1] for stat in stats])
        ax = val_line.axes
        ax.relim()
        ax.autoscale_view()
    plt.pause(0.00001)


def evaluate(sae: SAE, x: torch.Tensor, l1_coeff: float) -> tuple[float, float, float, int]:
    """Return (loss, variance explained, avg active latents, dead latents) of `sae` on activations x."""
    with torch.no_grad():
        loss, _, _ = sae.loss(x, l1_coeff)
        x_hat, f = sae(x)
        explained = 1 - (x_hat - x).pow(2).sum() / (x - x.mean(dim=0)).pow(2).sum()
        active = (f > 0).float().sum(dim=1).mean()
        dead = ((f > 0).sum(dim=0) == 0).sum()
    return loss.item(), explained.item(), active.item(), dead.item()


def flatten(acts: torch.Tensor) -> torch.Tensor:
    """(N, C, H, W) -> (N * H * W, C): one example per spatial position."""
    return acts.permute(0, 2, 3, 1).reshape(-1, acts.shape[1])


def train(name: str, tr: DogsDataset, va: DogsDataset) -> None:
    """Load the CNN `name`, train an SAE on its CONV1 activations, and save both."""
    set_random_seed()

    num_latents = 128  # number of SAE latents
    l1_coeff = 0.5  # weight of the L1 sparsity penalty in the SAE loss
    patience = 5  # patience for early stopping

    model_class, checkpoint_dir = MODELS[name]
    print(f"Loading {name} model...")
    model, _, _ = restore_checkpoint(model_class(), checkpoint_dir, force=True)

    x_tr = flatten(get_activations(model, tr.X))
    x_va = flatten(get_activations(model, va.X))
    loader = DataLoader(TensorDataset(x_tr), batch_size=config("sae.batch_size"), shuffle=True)
    print(f"Training {name} SAE on {len(x_tr)} activation vectors of dimension {x_tr.shape[1]}")

    sae = SAE(x_tr.shape[1], num_latents)
    sae.fit_scale(x_tr)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config("sae.learning_rate"))

    # Each entry is [train loss, val loss, train active latents, val active latents]. The val
    # loss is at index 1, where early_stopping expects it.
    def epoch_stats():
        tr_loss, _, tr_active, _ = evaluate(sae, x_tr, l1_coeff)
        va_loss, _, va_active, _ = evaluate(sae, x_va, l1_coeff)
        return [tr_loss, va_loss, tr_active, va_active]

    lines = make_sae_training_plot(name)
    stats = [epoch_stats()]
    update_sae_training_plot(lines, stats)
    prev_val_loss = stats[0][1]
    best_state = deepcopy(sae.state_dict())
    curr_patience = 0
    epoch = 0
    while curr_patience < patience:
        for (batch,) in loader:
            loss, _, _ = sae.loss(batch, l1_coeff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch += 1

        stats.append(epoch_stats())
        update_sae_training_plot(lines, stats)

        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)
        if curr_patience == 0:
            best_state = deepcopy(sae.state_dict())

    # Keep the epoch with the lowest validation loss
    sae.load_state_dict(best_state)
    _, explained, active, dead = evaluate(sae, x_va, l1_coeff)
    print(
        f"Finished training {name} SAE after {epoch} epochs: val loss = {prev_val_loss:.3f}, "
        f"val variance explained = {explained:.3f}, avg active latents = {active:.1f}, "
        f"dead latents = {dead}"
    )

    # Save the CNN weights with the SAE so visualize_sae.py uses the exact same model
    os.makedirs(config("sae.checkpoint"), exist_ok=True)
    torch.save(
        {"model": name, "cnn": model.state_dict(), "sae": sae.state_dict(), "stats": stats},
        sae_path(name),
    )
    print(f"Saved {name} SAE to {sae_path(name)}")

    # Save figure and keep plot open
    plt.savefig(f"sae_training_plot_{name}.png", dpi=200)
    plt.ioff()
    plt.show()


def main() -> None:
    """Train an SAE for the target model, then one for the source model."""
    # Use the target task's images for both models, so the SAEs of the target and
    # source CNNs can be compared on the same images
    tr, va, _, _ = get_train_val_test_datasets(task="target")
    for name in MODELS:
        train(name, tr, va)


if __name__ == "__main__":
    main()
