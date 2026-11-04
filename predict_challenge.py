"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Predict Challenge
    Runs the challenge model inference on the test dataset and saves the
    predictions to disk
    Usage: python predict_challenge.py --uniqname=<uniqname> [--gpu]
"""

import argparse

import pandas as pd
import torch
from torch.nn.functional import softmax

from dataset import get_challenge
from model.challenge import Challenge
from train_common import restore_checkpoint
from utils import config


def predict_challenge(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module) -> torch.Tensor:
    """Runs the model inference on the test set and outputs the predictions."""
    y_score = []
    for X, y in data_loader:
        output = model(X)
        y_score.append(softmax(output.data, dim=1)[:, 1])
    return torch.cat(y_score)


def main(uniqname: str, gpu: bool) -> None:
    """Train challenge model."""
    # data loaders
    ch_loader, get_semantic_label = get_challenge(
        task="target",
        batch_size=config("challenge.batch_size"),
    )

    model = Challenge()

    # Attempts to restore the latest checkpoint if exists
    model, _, _ = restore_checkpoint(model, config("challenge.checkpoint"))

    # Evaluate model
    model_pred = predict_challenge(ch_loader, model)

    print("Saving challenge predictions...")

    pd_writer = pd.DataFrame(model_pred, columns=["predictions_gpu" if gpu else "predictions"])
    pd_writer.to_csv(f"{uniqname}.csv", index=False,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniqname", required=True, help="needed to identify your submission")
    parser.add_argument("--gpu", action="store_true", help="only pass this flag if you trained your model using a GPU")
    args = parser.parse_args()
    main(args.uniqname, args.gpu)
