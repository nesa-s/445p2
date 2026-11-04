"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Train Target
    Train a convolutional neural network to classify images.
    Periodically output training information, and saves model checkpoints
    Usage: python train_target.py
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import get_train_val_test_loaders
from model.target import Target
from train_common import count_parameters, evaluate_epoch, early_stopping, restore_checkpoint, save_checkpoint, train_epoch
from utils import config, make_training_plot, set_random_seed


__all__ = ["freeze_layers", "train"]


def freeze_layers(model: torch.nn.Module, num_layers: int = 0) -> None:
    """
    This function modifies 'model' settings to stop tracking gradients on selected layers.
    The number of convolutional layers to stop tracking gradients for is defined by
    num_layers. You will need to look at PyTorch documentation to implement this function.

    Args:
        model: subclass of nn.Module
        num_layers: int, the number of conv layers to freeze
    """
    # TODO: 3(e) - modify model with the given layers frozen, e.g. if num_layers=2, freeze CONV1 and CONV2
    #       Hint: https://pytorch.org/docs/master/notes/autograd.html
    raise NotImplementedError()


def train(
    tr_loader: DataLoader,
    va_loader: DataLoader,
    te_loader: DataLoader,
    model: torch.nn.Module,
    model_name: str,
    num_layers: int = 0,
) -> None:
    """
    This function trains the target model. Only the weights of unfrozen layers of the model passed 
    into this function will be updated in training.
    
    Args:
        tr_loader: DataLoader for training data
        va_loader: DataLoader for validation data
        te_loader: DataLoader for test data
        model: subclass of torch.nn.Module, model to train on
        model_name: str, checkpoint path for the model
        num_layers: int, the number of source model layers to freeze
    """
    set_random_seed()
    
    # TODO: 3(e) - define loss function, and optimizer
    criterion = None
    optimizer = None

    print("Loading target model with", num_layers, "layers frozen")
    print("Number of trainable parameters:", count_parameters(model))
    model, start_epoch, stats = restore_checkpoint(model, model_name)

    axes = make_training_plot("Target Training")

    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: 3(e) - patience for early stopping, see appendix B 
    patience = None
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)

        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)
        epoch += 1

    print("Finished Training")

    # Keep plot open
    print(f"Saving training plot to target_training_plot_frozen_layers={num_layers}.png...")
    plt.savefig(f"target_training_plot_frozen_layers={num_layers}.png", dpi=200)
    plt.ioff()
    plt.show()


def main() -> None:
    """
    Train transfer learning model and display training plots.

    Train two different models: one with no layers frozen (fine-tune everything) and one with
    all 3 convolutional layers frozen (only train the fully connected layer).
    """
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("target.batch_size"),
    )

    freeze_none = Target()
    print("Loading source...")
    freeze_none, _, _ = restore_checkpoint(
        freeze_none,
        config("source.checkpoint"),
        force=True,
        pretrain=True,
    )

    freeze_three = deepcopy(freeze_none)
    freeze_layers(freeze_three, 3)

    train(tr_loader, va_loader, te_loader, freeze_none, config("target.frozen_checkpoint").format(layer=0), 0)
    train(tr_loader, va_loader, te_loader, freeze_three, config("target.frozen_checkpoint").format(layer=3), 3)


if __name__ == "__main__":
    main()
