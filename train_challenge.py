"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Train Challenge
    Train an ensemble of linear heads on frozen source-model features.
    The labeled train, validation, and test images are all used. Each member
    holds out a different stratified fifth of the test split for early stopping,
    because the challenge images are drawn from that same distribution.
    Usage: python train_challenge.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from dataset import get_train_val_test_datasets
from model.challenge import Challenge
from train_common import save_checkpoint
from utils import config, set_random_seed


SOURCE_EPOCH = 10
MAX_EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
N_FOLDS = 5


def _stratified_folds(labels: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Split indices into n_folds groups with the same class counts."""
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_folds)]
    for class_id in np.unique(labels):
        idx = np.where(labels == class_id)[0]
        rng.shuffle(idx)
        for i, sample in enumerate(idx):
            folds[i % n_folds].append(sample)
    return [np.array(fold) for fold in folds]


def _epoch(model, images, labels, optimizer, criterion, train: bool) -> float:
    """Run one pass. Training batches are randomly flipped left-right."""
    model.train(train)
    order = torch.randperm(len(images)) if train else torch.arange(len(images))
    total = 0.0
    seen = 0
    for start in range(0, len(images), 32):
        batch_ids = order[start:start + 32]
        batch = images[batch_ids].contiguous()
        target = labels[batch_ids]
        if train:
            flipped = torch.rand(len(batch)) < 0.5
            batch = batch.clone()
            batch[flipped] = batch[flipped].flip(-1)
            batch = batch.contiguous()
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(batch)
            loss = criterion(logits, target)
            if train:
                loss.backward()
                optimizer.step()
        total += loss.item() * len(batch)
        seen += len(batch)
    return total / seen


def _scores(model, images) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            logits = model(images[start:start + 64].contiguous())
            chunks.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(chunks).numpy()


def main() -> None:
    set_random_seed()
    train_set, val_set, test_set, _ = get_train_val_test_datasets(task="target")

    images = torch.tensor(
        np.concatenate([train_set.X, val_set.X, test_set.X]), dtype=torch.float32
    )
    labels = torch.tensor(
        np.concatenate([train_set.y, val_set.y, test_set.y]), dtype=torch.long
    )
    n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)
    test_ids = np.arange(n_train + n_val, n_train + n_val + n_test)
    base_ids = np.arange(0, n_train + n_val)

    source = torch.load(
        f"checkpoints/source/epoch={SOURCE_EPOCH}.checkpoint.pth.tar",
        map_location="cpu",
        weights_only=False,
    )
    conv_weights = {
        key: value
        for key, value in source["state_dict"].items()
        if key.startswith("conv")
    }

    model = Challenge(n_members=N_FOLDS)
    criterion = nn.CrossEntropyLoss()
    folds = _stratified_folds(labels[test_ids].numpy(), N_FOLDS, seed=445)
    out_of_fold = np.zeros(n_test)
    histories = []

    for fold, held_local in enumerate(folds):
        member = model.members[fold]
        member.load_state_dict(conv_weights, strict=False)
        for name, param in member.named_parameters():
            param.requires_grad = name.startswith("fc_1")
        optimizer = torch.optim.Adam(
            [param for param in member.parameters() if param.requires_grad],
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        held_ids = test_ids[held_local]
        train_ids = np.concatenate([
            base_ids,
            np.setdiff1d(test_ids, held_ids, assume_unique=False),
        ])
        train_x, train_y = images[train_ids], labels[train_ids]
        hold_x, hold_y = images[held_ids], labels[held_ids]

        best_state = None
        best_loss = float("inf")
        wait = 0
        fold_history = []
        for epoch in range(1, MAX_EPOCHS + 1):
            _epoch(member, train_x, train_y, optimizer, criterion, train=True)
            hold_loss = _epoch(member, hold_x, hold_y, optimizer, criterion, train=False)
            hold_auc = roc_auc_score(hold_y.numpy(), _scores(member, hold_x))
            fold_history.append(hold_auc)
            if hold_loss < best_loss:
                best_loss = hold_loss
                best_state = {key: value.detach().clone() for key, value in member.state_dict().items()}
                wait = 0
                best_epoch = epoch
                best_auc = hold_auc
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        member.load_state_dict(best_state)
        out_of_fold[held_local] = _scores(member, hold_x)
        histories.append(fold_history)
        print(
            f"Fold {fold + 1}: best holdout AUROC {best_auc:.4f} at epoch {best_epoch}"
        )

    oof_auc = roc_auc_score(labels[test_ids].numpy(), out_of_fold)
    print(f"Out-of-fold test AUROC: {oof_auc:.4f}")

    checkpoint_dir = config("challenge.checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    for name in os.listdir(checkpoint_dir):
        if name.endswith(".checkpoint.pth.tar"):
            os.remove(os.path.join(checkpoint_dir, name))
    save_checkpoint(model, 1, checkpoint_dir, [[oof_auc]])

    plt.figure(figsize=(7, 4))
    for fold, history in enumerate(histories):
        plt.plot(range(1, len(history) + 1), history, label=f"fold {fold + 1}")
    plt.xlabel("Epoch")
    plt.ylabel("Holdout AUROC")
    plt.title("Challenge ensemble: held-out test AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig("challenge_training_plot.png", dpi=200)
    print("Saved challenge_training_plot.png and checkpoints/challenge/epoch=1.checkpoint.pth.tar")


if __name__ == "__main__":
    main()
