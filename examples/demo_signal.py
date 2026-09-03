"""Demo for the FBSE-domain S transform skeleton."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow running the demo directly from the repository root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from fbst.transform import fbse_s_transform
from fbst.visualization import plot_time_frequency_matrix


def synthetic_signal(n: int = 256) -> np.ndarray:
    """Create a simple non-stationary demo signal."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)

    component_1 = np.sin(2 * np.pi * (15 * t + 25 * t**2))
    component_2 = 0.6 * np.sin(2 * np.pi * 60 * t)
    burst = np.exp(-300 * (t - 0.7) ** 2) * np.sin(2 * np.pi * 90 * t)

    return component_1 + component_2 + burst


def main() -> None:
    signal = synthetic_signal()
    result = fbse_s_transform(signal, num_zeros=len(signal), sigma_scale=8.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].plot(signal, color="black", linewidth=1.2)
    axes[0].set_title("Synthetic non-stationary signal")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Amplitude")

    plt.sca(axes[1])
    plot_time_frequency_matrix(
        result["time_frequency_matrix"],
        title="FBSE-domain S transform (demo)",
    )

    plt.show()


if __name__ == "__main__":
    main()
