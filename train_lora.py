"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Train LoRA
    Fine-tune the pretrained source model on the target task using LoRA adapters of rank
    1, 2 and 4 (see model/lora.py). The pretrained convolutional weights stay frozen.
    Periodically output training information, and saves model checkpoints.
    Usage: python train_lora.py
    (This file is provided in full -- you do not need to modify it.)
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import get_train_val_test_loaders
from model.lora import add_lora
from model.target import Target
from train_common import (
    count_parameters,
    early_stopping,
    evaluate_epoch,
    restore_checkpoint,
    save_checkpoint,
    train_epoch,
)
from utils import config, make_training_plot, set_random_seed


__all__ = ["train"]


def train(
    tr_loader: DataLoader,
    va_loader: DataLoader,
    te_loader: DataLoader,
    model: torch.nn.Module,
    model_name: str,
    rank: int,
) -> None:
    """
    Train a LoRA-adapted target model. Only the LoRA adapters and the fully connected
    layer are updated.

    Args:
        tr_loader: DataLoader for training data
        va_loader: DataLoader for validation data
        te_loader: DataLoader for test data
        model: the target model with LoRA adapters, already initialized from the source model
        model_name: str, checkpoint path for the model
        rank: int, the LoRA rank (only used for printing and naming the plot)
    """
    set_random_seed()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config("lora.learning_rate")
    )

    print(f"Training LoRA with rank {rank}")
    print("Number of trainable parameters:", count_parameters(model))
    model, start_epoch, stats = restore_checkpoint(model, model_name)

    axes = make_training_plot(f"LoRA Training (rank {rank})")

    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats, include_test=True
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    patience = 5
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats, include_test=True
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)

        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)
        epoch += 1

    print("Finished Training")

    # Summarize the epoch with the lowest validation loss (stats[e][1] is the val loss at epoch e)
    best = min(range(len(stats)), key=lambda e: stats[e][1])
    print(
        f"[rank {rank}] lowest validation loss at epoch {best}: "
        f"train AUROC {stats[best][5]:.4f} | val AUROC {stats[best][2]:.4f} | test AUROC {stats[best][8]:.4f} "
        f"| trainable parameters {count_parameters(model)}"
    )

    # Keep plot open
    print(f"Saving training plot to lora_training_plot_rank={rank}.png...")
    plt.savefig(f"lora_training_plot_rank={rank}.png", dpi=200)
    plt.ioff()
    plt.show()


def main() -> None:
    """
    Train LoRA models with rank 1, 2 and 4 and display training plots.

    Each model is initialized from the source model you trained in Section 3(c).
    """
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("lora.batch_size"),
    )

    pretrained = Target()
    print("Loading source...")
    pretrained, _, _ = restore_checkpoint(
        pretrained,
        config("source.checkpoint"),
        force=True,
        pretrain=True,
    )

    for rank in [1, 2, 4]:
        set_random_seed()
        model = add_lora(deepcopy(pretrained), rank)
        train(tr_loader, va_loader, te_loader, model, config("lora.checkpoint").format(rank=rank), rank)


if __name__ == "__main__":
    main()
