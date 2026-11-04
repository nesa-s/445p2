"""
EECS 445 - Introduction to Machine Learning
Fall 2026  - Project 2

Generate confusion matrix graphs.
    Usage: python3 confusion_matrix.py
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset import get_train_val_test_loaders
from model.source import Source
from train_common import predictions, restore_checkpoint
from utils import config


def gen_labels(loader: DataLoader, model: torch.nn.Module) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Return true and predicted values.

    Args:
        loader: DataLoader for validation data
        model: subclass of torch.nn.Module, Model to evaluate
    """
    y_true, y_pred = [], []
    for X, y in loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true = np.append(y_true, y.numpy())
            y_pred = np.append(y_pred, predicted.numpy())
    return y_true, y_pred


def plot_conf(loader: DataLoader, model: torch.nn.Module, sem_labels: str, png_name: str) -> None:
    """
    Draw confusion matrix.

    Args:
        loader: DataLoader for validation data
        model: subclass of torch.nn.Module, Model to evaluate
        sem_labels: str, Legend values
        png_name: str, name of the file to save the confusion mtx plot
    """
    y_true, y_pred = gen_labels(loader, model)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(config("source.num_classes")))
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label("Frequency", rotation=270, labelpad=10)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, z, ha="center", va="center")
    plt.gcf().text(0.02, 0.4, sem_labels, fontsize=9)
    plt.subplots_adjust(left=0.3)
    ax.set_xlabel("Predictions")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("True Labels")
    print(f"Saving confusion matrix plot to {png_name}...")
    plt.savefig(png_name, dpi=200)


def main() -> None:
    """Create confusion matrix and save to file."""
    _, va_loader, _, get_semantic_label = get_train_val_test_loaders(task="source", batch_size=config("source.batch_size"))

    model = Source()
    print("Loading source...")
    model, _, _ = restore_checkpoint(model, config("source.checkpoint"), force=True)

    sem_labels = "\n".join(
        f"{i} - {get_semantic_label(i).replace('_', ' ').title()}" for i in range(config("source.num_classes"))
    )

    # Evaluate model
    plot_conf(va_loader, model, sem_labels, "confusion_matrix.png")


if __name__ == "__main__":
    main()
