"""
EECS 445 - Introduction to Machine Learning
Fall 2026 - Project 2

Visualize SAE
    Opens a window showing a target task image and where the source and target models are
    active on it. Brighter regions mean stronger activation. You choose which part of the
    question you are answering:
      - m: two images with blue backgrounds of different shades. Each row is the source SAE
            latent most active on one image's background: the first two panels highlight where it
            is active on the two images, and the remaining panels show the images in the dataset
            where it is most active on average.
      - n: the same as m, but with the CONV1 filters of the source model instead of SAE latents.
      - o: the target SAE on an image, then the source SAE on the same image. Each row is one
            of the 4 latents most active on the image: the first panel highlights where in the
            image the latent is active, and the remaining panels show the images in the dataset
            where that latent is most active, which helps you figure out what the latent
            detects. Clicking alternates target, source, then target on a new image, and so on.
            The first image is a training collie.
    Close the window to exit. Each figure is saved to visualize_sae_part_<part>.png.

    Run train_sae.py first.

    Usage: python visualize_sae.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import softmax

from dataset import DogsDataset, get_train_val_test_datasets
from model.sae import SAE
from train_sae import MODELS, get_activations, sae_path
from utils import set_random_seed


NUM_LATENTS = 4  # number of latents (rows) to show for each image
NUM_EXAMPLES = 5  # number of top-activating dataset images to show for each latent


# Parts m and n: images with a dark blue and a light blue background
BACKGROUND_IMAGES = [125, 523]
BLUE_MARGIN = 40  # a pixel is part of the background if its blue value exceeds its red value by this much

SAE_COLORS = {"target": "tab:red", "source": "tab:green"}  # title color


def choose_part() -> str:
    """Ask the user which part of the question they are answering."""
    print("Which part of the question are you answering? Choose from m, n, o.")
    print(">> ", end="")
    part = input().strip()
    if part not in ("m", "n", "o"):
        raise Exception("Invalid part")
    return part


def show(ax, image, latent_map=None, vmax=1.0, title=""):
    """Draw `image` on `ax`, optionally overlaid with a (32, 32) latent activation map."""
    ax.clear()
    ax.imshow(image)
    if latent_map is not None:
        H, W = image.shape[:2]
        vmax = max(vmax, 1e-8)  # dead latents (never active) have vmax 0
        ax.imshow(
            latent_map,
            cmap="inferno",
            alpha=0.8 * np.clip(latent_map / vmax, 0, 1),  # inactive regions stay transparent
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
            extent=(-0.5, W - 0.5, H - 0.5, -0.5),
        )
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def load_sae(name, X):
    """Load the CNN and SAE for model `name` and compute their latents and predictions on images X."""
    if not os.path.exists(sae_path(name)):
        raise Exception(f"No SAE found at {sae_path(name)}. Run train_sae.py first.")
    checkpoint = torch.load(sae_path(name), weights_only=False)

    model_class, _ = MODELS[name]
    model = model_class()
    model.load_state_dict(checkpoint["cnn"])
    model.eval()

    d_in = checkpoint["sae"]["encoder.weight"].shape[1]
    num_latents = checkpoint["sae"]["encoder.weight"].shape[0]
    sae = SAE(d_in, num_latents)
    sae.load_state_dict(checkpoint["sae"])

    # latent_maps[i, :, :, j] is where latent j is active in image i
    acts = get_activations(model, X)
    N, C, H, W = acts.shape
    with torch.no_grad():
        latent_maps = sae.encode(acts.permute(0, 2, 3, 1).reshape(-1, C)).reshape(N, H, W, -1).numpy()
        probs = softmax(model(torch.from_numpy(X).float()), dim=1).numpy()

    peak = latent_maps.max(axis=(1, 2))  # (N, num_latents) strongest activation of each latent per image
    latent_max = peak.max(axis=0)  # color scale for each latent, shared across images
    # How close each latent gets to its strongest activation over the whole dataset. Latents are
    # ranked by this instead of peak, since some latents (e.g. for colors) are much larger than
    # others and would always rank first.
    relative_peak = peak / np.maximum(latent_max, 1e-8)
    top_examples = np.argsort(-relative_peak, axis=0)[:NUM_EXAMPLES]  # (NUM_EXAMPLES, num_latents)
    return {
        "name": name,
        "acts": acts.permute(0, 2, 3, 1).numpy(),  # (N, H, W, C) raw CONV1 activations
        "latent_maps": latent_maps,
        "probs": probs,
        "latent_max": latent_max,
        "relative_peak": relative_peak,
        "top_examples": top_examples,
    }


def background_mask(image, shape):
    """Return a mask of the blue background of `image` (64, 64, 3), downsampled to `shape` (32, 32) to
    line up with the activation maps."""
    H, W = shape
    pooled = image.reshape(H, image.shape[0] // H, W, image.shape[1] // W, 3).astype(float).mean(axis=(1, 3))
    return pooled[..., 2] - pooled[..., 0] > BLUE_MARGIN


def show_background_units(sae, images, part, unit):
    """Show the SAE latent (`unit` "latent") or CONV1 filter (`unit` "filter") most active on the background of
    each image in BACKGROUND_IMAGES, on both images and on the dataset images where it is most active."""
    if unit == "latent":
        maps, vmax, name = sae["latent_maps"], sae["latent_max"], "SAE latent"
    else:
        maps, name = sae["acts"], "CONV1 filter"
        # Color scale for each filter, shared across images. The 99th percentile is used instead of the max,
        # since a few outlier pixels would otherwise wash out every typical activation.
        vmax = np.percentile(maps, 99, axis=(0, 1, 2))
    N, H, W, _ = maps.shape

    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(
        len(BACKGROUND_IMAGES) + 1,
        NUM_EXAMPLES + 3,
        width_ratios=[1, 1, 0.3] + [1] * NUM_EXAMPLES,
        left=0.16,
        right=0.98,
        top=0.8,
        bottom=0.08,
        hspace=0.15,
        wspace=0.1,
    )
    fig.suptitle(f"Part {part}: SOURCE {name}s", fontsize=18, fontweight="bold", color=SAE_COLORS["source"])
    fig.text(
        0.5,
        0.88,
        f"Where each {name} is active on the two images (left), "
        "and the dataset images where it has the highest average value (right)",
        ha="center",
        fontsize=11,
    )
    fig.text(0.5, 0.02, "Close the window to exit.", ha="center", fontsize=11, color="tab:blue")

    for k, i in enumerate(BACKGROUND_IMAGES):
        ax = fig.add_subplot(grid[0, k])
        show(ax, images[i], title=f"image {k + 1}")
        ax.title.set_fontsize(11)
    for r, i in enumerate(BACKGROUND_IMAGES, start=1):
        # Highest average activation on the background, relative to the color scale. This is chosen here
        # instead of fixed in advance, since the same latent can end up with a different index on a
        # different machine.
        background = maps[i][background_mask(images[i], (H, W))].mean(axis=0)
        j = np.argmax(background / np.maximum(vmax, 1e-8))

        map_axes = [fig.add_subplot(grid[r, k]) for k in range(len(BACKGROUND_IMAGES))]
        for ax, k in zip(map_axes, BACKGROUND_IMAGES):
            show(ax, images[k], maps[k, :, :, j], vmax[j])
        label = f"{unit} {j}\n(highest average\nvalue on image {r}'s\nbackground)"
        map_axes[0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10, labelpad=10)

        # Rank by average activation instead of peak, since these fire across whole backgrounds. The examples
        # are shown without the activation map so their colors are visible.
        score = maps[..., j].mean(axis=(1, 2))
        score[BACKGROUND_IMAGES] = -np.inf  # only show other images
        for c, e in enumerate(np.argsort(-score)[:NUM_EXAMPLES]):
            show(fig.add_subplot(grid[r, c + 3]), images[e])

    filename = f"visualize_sae_part_{part}.png"
    plt.savefig(filename)
    print(f"Saved figure to {filename}.")
    plt.show()


def main():
    set_random_seed()

    part = choose_part()
    names = ["source"] if part in ("m", "n") else ["target", "source"]

    # Standardized images go into the model; the original images are what we display
    partitions = ["train", "val", "test"]
    datasets = get_train_val_test_datasets(task="target")[:3]
    X = np.concatenate([ds.X for ds in datasets])
    images = np.concatenate([DogsDataset(p, "target").X for p in partitions])
    labels = np.concatenate([ds.y for ds in datasets])
    splits = np.concatenate([[p] * len(ds) for p, ds in zip(partitions, datasets)])
    semantic = datasets[0].get_semantic_label

    saes = {name: load_sae(name, X) for name in names}

    if part in ("m", "n"):
        show_background_units(saes["source"], images, part, "latent" if part == "m" else "filter")
        return

    def describe(sae, i):
        text = f"{splits[i]}: {semantic(labels[i])}"
        if sae["probs"].shape[1] == 2:
            text += f"\nP({semantic(1)}) = {sae['probs'][i, 1]:.2f}"
        return text

    fig = plt.figure(figsize=(2 * (NUM_EXAMPLES + 3), 2 * NUM_LATENTS + 1))
    grid = fig.add_gridspec(NUM_LATENTS, NUM_EXAMPLES + 3, width_ratios=[2, 1, 0.15] + [1] * NUM_EXAMPLES)
    image_ax = fig.add_subplot(grid[1:3, 0])
    latent_axes = [fig.add_subplot(grid[r, 1]) for r in range(NUM_LATENTS)]
    example_axes = [[fig.add_subplot(grid[r, c + 3]) for c in range(NUM_EXAMPLES)] for r in range(NUM_LATENTS)]
    fig.subplots_adjust(top=0.85)
    title = fig.suptitle("", fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.9,
        "SAE latents on this image (left) and the dataset images where each latent has the highest peak values (right)",
        ha="center",
        fontsize=11,
    )
    instructions = (
        "Click anywhere on the figure or press any key to switch between the target and source SAE. "
        "After the source SAE, a new image is shown. Close the window to exit."
    )
    fig.text(0.5, 0.02, instructions, ha="center", fontsize=11, color="tab:blue")
    filename = f"visualize_sae_part_{part}.png"

    def on_close(event):
        exit()

    fig.canvas.mpl_connect("close_event", on_close)

    def draw(sae, i, latents):
        """Show `latents` of `sae` on image i, then wait for a click or key press."""
        title.set_text(f"Part {part}: {sae['name'].upper()} SAE")
        title.set_color(SAE_COLORS[sae["name"]])
        show(image_ax, images[i], title=describe(sae, i))
        image_ax.title.set_fontsize(11)

        for r, j in enumerate(latents):
            latent_map, latent_max = sae["latent_maps"][..., j], sae["latent_max"][j]
            show(latent_axes[r], images[i], latent_map[i], latent_max, f"latent {j}")
            for c, k in enumerate(sae["top_examples"][:, j]):
                show(example_axes[r][c], images[k], latent_map[k], latent_max, "")

        plt.draw()
        plt.savefig(filename)
        print(f"Saved figure to {filename}.")
        plt.waitforbuttonpress()

    # Start on a training collie, then show random images from any split
    i = np.random.choice(np.where((splits == "train") & (labels == 1))[0])
    while True:
        for name in names:
            draw(saes[name], i, np.argsort(-saes[name]["relative_peak"][i])[:NUM_LATENTS])
        i = np.random.randint(len(X))


if __name__ == "__main__":
    main()
