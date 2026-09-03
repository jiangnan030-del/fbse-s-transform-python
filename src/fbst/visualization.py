"""Visualization helpers for FBSE-domain S-transform outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_time_frequency_matrix(
    time_frequency_matrix: np.ndarray,
    title: str = "FBSE-domain S transform",
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 5),
    ax=None,
    show_colorbar: bool = True,
):
    """Plot the time-frequency magnitude or energy matrix."""
    tf = np.asarray(time_frequency_matrix)
    if tf.ndim != 2:
        raise ValueError("time_frequency_matrix must be two-dimensional")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    image = ax.imshow(tf, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Bessel-frequency index")

    if show_colorbar:
        fig.colorbar(image, ax=ax, label="Magnitude")

    if created_fig:
        fig.tight_layout()

    return fig, ax


def plot_signal_and_time_frequency(
    signal: np.ndarray,
    time_frequency_matrix: np.ndarray,
    title: str = "FBSE-domain S transform",
    cmap: str = "viridis",
    save_path: str | None = None,
):
    """Create a two-panel figure with signal and time-frequency map."""
    signal = np.asarray(signal)
    tf = np.asarray(time_frequency_matrix)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(signal, color="black", linewidth=1.2)
    axes[0].set_title("Signal")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Amplitude")

    plot_time_frequency_matrix(
        tf,
        title=title,
        cmap=cmap,
        ax=axes[1],
        show_colorbar=True,
    )

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes
