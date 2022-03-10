import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tp4


def show_loss(losses, vals, tests=None, labels=None, n_epochs=100):
    if labels is None:
        labels = [None for _ in losses]
    if tests is None:
        tests = [None for _ in losses]

    cmap = plt.get_cmap("Set1")
    for i, (loss, val, test, label) in enumerate(zip(losses, vals, tests, labels)):
        plt.semilogy(
            np.linspace(1, n_epochs, len(loss)), loss, label=label, color=cmap(i)
        )
        plt.semilogy(np.linspace(1, n_epochs, len(val)), val, "--", color=cmap(i))
        if test is not None:
            plt.semilogy(np.linspace(1, n_epochs, len(test)), test, ":", color=cmap(i))

    plt.ylim((5e-2, 2e0))
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if labels is not None:
        plt.legend()
    plt.show()


def show_loss_lr(loss_hist, lr_min=1e-10, lr_max=1e0):
    lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), len(loss_hist))
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.ylim((1e-2, 1e2))
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.plot(lrs, loss_hist, label="loss")
    plt.show()


def visualize(dataset):
    indices = np.random.randint(len(dataset), size=(3,))
    ims, ys = zip(*[dataset[i] for i in indices])
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    for ax, im, y in zip(axs, ims, ys):
        im = im.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = std * im + mean
        im = np.clip(im, 0, 1)
        ax.imshow(im)
        ax.set_title(dataset.classes[y])

        ax.axis("off")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    plt.show()


def ims_and_labels(dataset, indices):
    ims, labels = zip(*[dataset[i] for i in indices])
    ims = [im.numpy().transpose(1, 2, 0) for im in ims]
    return ims, labels


def show_ims(*ims, labels=None):
    n = len(ims)
    w = int(math.sqrt(n))
    h = n // w + min(1, n % w)
    fig, axs = plt.subplots(h, w, squeeze=False, figsize=(10, 10))

    for ax in axs.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, im in enumerate(ims):
        ax = axs[i // w, i % w]
        if labels is not None:
            ax.set_title(labels[i])
        ax.imshow(im)

    plt.show()


def imbalance(net, dataset):
    # Check for class imbalance

    predictions, test_loss, test_accuracy = tp4.training.predict(
        net, dataset, batch_size=30
    )
    counts = pd.Series(predictions).value_counts() / len(predictions)
    return pd.DataFrame(
        counts.values,
        index=np.array(dataset.classes)[counts.index],
        columns=["frequency"],
    )
