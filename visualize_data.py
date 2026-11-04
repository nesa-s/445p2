"""
EECS 445 - Introduction to Machine Learning
Fall 2026  - Project 2

Visualize Dogs
    This will open up a window displaying randomly selected training
    images. The label of the image is shown. Click on the figure to
    refresh with a set of new images. You can save the images using
    the save button. Close the window to break out of the loop.

    The success of this script is a good indication that the data flow
    part of this project is running smoothly.

    Usage: python visualize_data.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio.v2 import imread

from dataset import resize, ImageStandardizer, DogsDataset
from utils import config, denormalize_image, set_random_seed


if __name__ == "__main__":
    set_random_seed()

    training_set = DogsDataset("train")
    training_set.X = resize(training_set.X)
    standardizer = ImageStandardizer()
    standardizer.fit(training_set.X)

    metadata = pd.read_csv(config("csv_file"))
    metadata = metadata[metadata["partition"] != "challenge"].reset_index(drop=True) 
    print("Click on the figure to choose new images. Close the figure to exit.")

    N = 4
    fig, axes = plt.subplots(nrows=2, ncols=N, figsize=(2 * N, 2 * 2))

    pad = 3
    axes[0, 0].annotate(
        "Original",
        xy=(0, 0.5),
        xytext=(-axes[0, 0].yaxis.labelpad - pad, 0),
        xycoords=axes[0, 0].yaxis.label,
        textcoords="offset points",
        size="large",
        ha="right",
        va="center",
        rotation="vertical",
    )
    axes[1, 0].annotate(
        "Preprocessed",
        xy=(0, 0.5),
        xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
        xycoords=axes[1, 0].yaxis.label,
        textcoords="offset points",
        size="large",
        ha="right",
        va="center",
        rotation="vertical",
    )

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    while True:
        rand_idx = np.random.choice(np.arange(len(metadata)), size=N, replace=False)
        X, y = [], []
        for idx in rand_idx:
            filename = os.path.join(config("image_path"), metadata.loc[idx, "filename"])
            X.append(imread(filename))
            y.append(metadata.loc[idx, "semantic_label"])

        for i, (xi, yi) in enumerate(zip(X, y)):
            axes[0, i].imshow(xi)
            axes[0, i].set_title(yi)

        X_ = resize(np.array(X))
        X_ = standardizer.transform(X_)
        for i, (xi, yi) in enumerate(zip(X_, y)):
            axes[1, i].imshow(denormalize_image(xi), interpolation="bicubic")

        fig = plt.gcf()
        plt.savefig("visualize_data.png")
        print("Saved figure to visualize_data.png.")

        def on_close(event):
            exit()

        fig.canvas.mpl_connect("close_event", on_close)

        if plt.waitforbuttonpress():
            break
