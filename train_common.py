"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Helper file for common training functions.
"""

import itertools
import os

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax
from sklearn import metrics

import utils


__all__ = [
    "count_parameters",
    "save_checkpoint",
    "restore_checkpoint",
    "clear_checkpoint",
    "early_stopping",
    "evaluate_epoch",
    "train_epoch",
    "predictions",
]


def count_parameters(model: torch.nn.Module) -> int:
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: torch.nn.Module, epoch: int, checkpoint_dir: str, stats: list) -> None:
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir,exist_ok=True)
    torch.save(state, filename)


def restore_checkpoint(
    model: torch.nn.Module, 
    checkpoint_dir: str, 
    cuda: bool=False, 
    force: bool=False, 
    pretrain: bool=False
) -> tuple[nn.Module, int, list]:
    """
    Restore model from checkpoint if it exists.

    Args:
        model: The model to be restored.
        checkpoint_dir: Directory where checkpoint files are stored.
        cuda: Whether to load the model on GPU if available. Defaults to False.
        force: If True, force the user to choose an epoch. Defaults to False.
        pretrain: If True, allows partial loading of the model state (used for pretraining). Defaults to False.

    Returns:
        tuple: The restored model, the starting epoch, and the list of statistics.

    Description:
        This function attempts to restore a saved model from the specified `checkpoint_dir`.
        If no checkpoint is found, the function either raises an exception (if `force` is True) or returns
        the original model and starts from epoch 0. If a checkpoint is available, the user can choose which
        epoch to load from. The model's parameters, epoch number, and training statistics are restored.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            "Which epoch to load from? Choose in range [0, {}].".format(epoch),
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename, weights_only=False)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage, weights_only=False)

    try:
        start_epoch = checkpoint["epoch"]
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats


def clear_checkpoint(checkpoint_dir: str) -> None:
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def early_stopping(stats: list, curr_count_to_patience: int, prev_val_loss: float) -> tuple[int, float]:
    """Calculate new patience and validation loss.

    Increment curr_count_to_patience by one if new loss is not less than prev_val_loss
    Otherwise, update prev_val_loss with the current val loss, and reset curr_count_to_patience to 0

    Returns: new values of curr_count_to_patience and prev_val_loss
    """
    # TODO: 2(e) - implement early stopping
    raise NotImplementedError()


def evaluate_epoch(
    axes: matplotlib.axes.Axes,
    tr_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    te_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    epoch: int,
    stats: list,
    include_test: bool = False,
    update_plot: bool = True,
    multiclass: bool = False,
) -> None:
    """Evaluate the `model` on the train and validation set."""

    def _get_metrics(loader):
        y_true, y_pred, y_score = [], [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in loader:
            with torch.no_grad():
                output = model(X)
                predicted = predictions(output.data)
                y_true.append(y)
                y_pred.append(predicted)
                if not multiclass:
                    y_score.append(softmax(output.data, dim=1)[:, 1])
                else:
                    y_score.append(softmax(output.data, dim=1))
                total += y.size(0)
                correct += (predicted == y).sum().item()
                running_loss.append(criterion(output, y).item())

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_score = torch.cat(y_score)
        loss = np.mean(running_loss)
        acc = correct / total
        if not multiclass:
            auroc = metrics.roc_auc_score(y_true, y_score)
        else:
            auroc = metrics.roc_auc_score(y_true, y_score, multi_class="ovo")
        return acc, loss, auroc

    train_acc, train_loss, train_auc = _get_metrics(tr_loader)
    val_acc, val_loss, val_auc = _get_metrics(val_loader)

    stats_at_epoch = [
        val_acc,
        val_loss,
        val_auc,
        train_acc,
        train_loss,
        train_auc,
    ]
    if include_test:
        stats_at_epoch += list(_get_metrics(te_loader))

    stats.append(stats_at_epoch)
    utils.log_training(epoch, stats)
    if update_plot:
        utils.update_training_plot(axes, epoch, stats)


def train_epoch(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Train the `model` for one epoch of data from `data_loader`.

    Args:
        data_loader: DataLoader providing batches of input data and corresponding labels.
        model: The model to be trained. This is one of the model classes in the 'model' folder. 
        criterion: The loss function used to compute the model's loss.
        optimizer: The optimizer used to update the model parameters.

    Description:
        This function sets the model to training mode and use the data loader to iterate through the entire dataset.
        For each batch, it performs the following steps:
        1. Resets the gradient calculations in the optimizer.
        2. Performs a forward pass to get the model predictions.
        3. Computes the loss between predictions and true labels using the specified `criterion`.
        4. Performs a backward pass to calculate gradients.
        5. Updates the model weights using the `optimizer`.
    """
    for i, (X, y) in enumerate(data_loader):
        # TODO 2(e) - implement training steps:
        # 1. Reset optimizer gradient calculations
        # 2. Get model predictions
        # 3. Calculate loss between model prediction and true labels
        # 4. Perform backward pass
        # 5. Update model weights
        raise NotImplementedError()


def predictions(logits: torch.Tensor) -> torch.Tensor:
    """Determine predicted class index given logits.

    args: 
        logits: The model's output logits. It is a 2D tensor of shape (batch_size, num_classes). 

    Returns:
        the predicted class output that has the highest probability. This should be of size (batch_size,).
    """
    # TODO 2(c) - implement predictions
    raise NotImplementedError()
