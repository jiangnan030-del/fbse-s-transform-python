"""Visualization helpers for FBSE-domain S-transform outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def apply_publication_style() -> None:
    """Apply a clean plotting style suitable for papers and reports."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_time_frequency_matrix(
    time_frequency_matrix: np.ndarray,
    title: str = "FBSE-domain S transform",
    cmap: str = "magma",
    figsize: tuple[int, int] = (10, 5),
    ax=None,
    show_colorbar: bool = True,
    colorbar_label: str = "Energy",
):
    """Plot a time-frequency magnitude or energy matrix."""
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
    ax.set_ylabel("Frequency index")

    if show_colorbar:
        fig.colorbar(image, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)

    if created_fig:
        fig.tight_layout()

    return fig, ax


def plot_signal_and_time_frequency(
    signal: np.ndarray,
    time_frequency_matrix: np.ndarray,
    title: str = "FBSE-domain S transform",
    cmap: str = "magma",
    save_path: str | None = None,
):
    """Create a two-panel figure with signal and time-frequency map."""
    apply_publication_style()

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
        colorbar_label="Energy",
    )

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig, axes


def plot_fbse_stft_comparison(
    signal: np.ndarray,
    fbse_tf: np.ndarray,
    stft_tf: np.ndarray,
    save_path: str | None = None,
):
    """Plot a publication-style comparison between FBSE-ST and STFT."""
    apply_publication_style()

    signal = np.asarray(signal)
    fbse_tf = np.asarray(fbse_tf)
    stft_tf = np.asarray(stft_tf)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    axes[0].plot(signal, color="black", linewidth=1.2)
    axes[0].set_title("Synthetic non-stationary signal")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Amplitude")

    im1 = axes[1].imshow(fbse_tf, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("FBSE-ST energy distribution")
    axes[1].set_xlabel("Time index")
    axes[1].set_ylabel("Bessel-frequency index")
    fig.colorbar(im1, ax=axes[1], label="Energy", fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(stft_tf, aspect="auto", origin="lower", cmap="viridis")
    axes[2].set_title("STFT energy distribution")
    axes[2].set_xlabel("Frame index")
    axes[2].set_ylabel("Fourier-frequency index")
    fig.colorbar(im2, ax=axes[2], label="Energy", fraction=0.046, pad=0.04)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig, axes
