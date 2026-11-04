"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Test ViT
    Test our trained ViT from train_vit.py on the heldout test data.
    Load the trained ViT model from a saved checkpoint and evaulates using
    accuracy and AUROC metrics.
    Usage: python test_vit.py
"""

import torch

from dataset import get_train_val_test_loaders
from model.vit import ViT
from train_common import evaluate_epoch, restore_checkpoint
from utils import config, make_training_plot, set_random_seed


def main():
    """Print performance metrics for model at specified epoch."""
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("vit.batch_size"),
    )

    # Model
    model = ViT(
        num_blocks=2,
        num_heads=2,
        num_hidden=16,
        num_patches=16,
    )

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Attempts to restore the latest checkpoint if exists
    print("Loading ToyVit...")
    model, start_epoch, stats = restore_checkpoint(model, config("vit.checkpoint"))

    axes = make_training_plot()

    # Evaluate the model
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
        update_plot=False,
    )


if __name__ == "__main__":
    main()
