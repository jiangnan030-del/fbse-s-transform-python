"""Visualization helpers for FBSE-domain S-transform outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_time_frequency_matrix(
    time_frequency_matrix: np.ndarray,
    title: str = "FBSE-domain S transform",
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 5),
):
    """Plot the time-frequency magnitude matrix."""
    tf = np.asarray(time_frequency_matrix)
    if tf.ndim != 2:
        raise ValueError("time_frequency_matrix must be two-dimensional")

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(tf, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Bessel-frequency index")
    fig.colorbar(image, ax=ax, label="Magnitude")
    fig.tight_layout()
    return fig, ax
