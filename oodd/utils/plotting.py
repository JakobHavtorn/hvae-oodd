from typing import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def gallery(array: np.ndarray, ncols=2):
    """Transform a set of images into a single array

    Args:
        array (list, tuple or np.ndarray): Iterable of images each of shape [height, width, channels]
        ncols (int): Number of columns to use in the gallery (must divide `n_index`)

    Returns:
        np.ndarray: Gallery of the `n_index` images of shape [channels, height * nrows, width * ncols]
                    where `nrows = nindex // ncols`
    """
    if isinstance(array, (list, tuple)):
        array = np.array(array)
    if array.ndim not in [2, 3, 4]:
        raise ValueError("Input array must have at least 2 and at most 4 dimensions (Got {})".format(array.ndim))
    if array.ndim == 2:
        array = array[np.newaxis, ...]  # Create n_index dimension
    if array.ndim == 3:
        array = array[..., np.newaxis]  # Create channels dimension

    nindex, height, width, channels = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols, "nindex must be divisible by ncols"

    # want result.shape = (height * nrows, width * ncols, channels)
    result = (
        array.reshape(nrows, ncols, height, width, channels)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, channels)
    )
    return result


def plot_gallery(array: np.ndarray, ncols=2):
    fig, ax = plt.subplots()
    grid = gallery(array, ncols=ncols)
    ax.imshow(grid)
    return fig, ax
    

def plot_likelihood_distributions(
    likelihoods: Dict[str, torch.Tensor],
    xlabel="Log-likelihood lower bound",
    title="Likelihood distributions",
    grid=True,
    ax=None,
    fig=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    for source, values in likelihoods.items():
        sns.kdeplot(values, fill=True, ax=ax, label=source)

    ax.legend()
    ax.grid(grid)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return fig, ax


def plot_roc_curve(
    fpr, tpr, roc_auc, title="Receiver operating characteristic", label="", ax=None, fig=None, lw=2, grid=True, **kwargs
):
    """Plot the ROC curve of a binary classification given FPR, TPR and area under the ROC"""
    if ax is None:
        fig, ax = plt.subplots()

    label = label + f"AUROC = {roc_auc:.3f}"
    ax.plot(fpr, tpr, lw=lw, label=label, **kwargs)
    ax.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--", **kwargs)
    if grid:
        ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig, plt.gca()
