"""Quick demo for the FBSE-domain S transform."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fbst.transform import fbse_s_transform
from fbst.visualization import plot_signal_and_time_frequency


def synthetic_signal(n: int = 256) -> np.ndarray:
    """Create a simple non-stationary signal."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    component_1 = np.sin(2 * np.pi * (12 * t + 30 * t**2))
    component_2 = 0.5 * np.sin(2 * np.pi * 55 * t)
    burst = np.exp(-280 * (t - 0.72) ** 2) * np.sin(2 * np.pi * 90 * t)
    return component_1 + component_2 + burst


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    signal = synthetic_signal(n=256)
    result = fbse_s_transform(signal, num_zeros=256, sigma_scale=0.08)

    plot_signal_and_time_frequency(
        signal,
        result["energy_time_frequency_matrix"],
        title="FBSE-domain S transform (demo)",
        save_path=str(output_dir / "demo_time_frequency.png"),
    )

    np.savez(
        output_dir / "demo_result.npz",
        signal=signal,
        zeros=result["zeros"],
        pseudo_frequencies=result["pseudo_frequencies"],
        time_frequency_matrix=result["time_frequency_matrix"],
        energy_time_frequency_matrix=result["energy_time_frequency_matrix"],
    )

    print(f"Saved figure to: {output_dir / 'demo_time_frequency.png'}")
    print(f"Saved arrays to: {output_dir / 'demo_result.npz'}")
    plt.show()


if __name__ == "__main__":
    main()
