"""Reproducible FBSE-ST experiment script.

This script generates a synthetic non-stationary signal, computes the
FBSE-domain S transform, saves the resulting arrays, and exports a figure.
"""

from __future__ import annotations

import argparse
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


def build_signal(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    chirp_1 = np.sin(2 * np.pi * (8 * t + 36 * t**2))
    chirp_2 = 0.7 * np.sin(2 * np.pi * (20 * t + 10 * t**2))
    burst = 0.9 * np.exp(-350 * (t - 0.62) ** 2) * np.sin(2 * np.pi * 85 * t)
    return chirp_1 + chirp_2 + burst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible FBSE-ST experiment")
    parser.add_argument("--samples", type=int, default=256, help="Number of signal samples")
    parser.add_argument("--num-zeros", type=int, default=256, help="Number of Bessel zeros")
    parser.add_argument("--sigma-scale", type=float, default=0.08, help="Adaptive Gaussian width scale")
    parser.add_argument("--output-dir", type=str, default="outputs/repro_experiment", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    signal = build_signal(args.samples)
    result = fbse_s_transform(
        signal,
        num_zeros=args.num_zeros,
        sigma_scale=args.sigma_scale,
    )

    plot_signal_and_time_frequency(
        signal,
        result["energy_time_frequency_matrix"],
        title="FBSE-ST reproducible experiment",
        save_path=str(output_dir / "fbse_st_experiment.png"),
    )

    np.savez(
        output_dir / "fbse_st_experiment.npz",
        signal=signal,
        zeros=result["zeros"],
        pseudo_frequencies=result["pseudo_frequencies"],
        gaussian_widths=result["gaussian_widths"],
        coefficients=result["coefficients"],
        time_frequency_matrix=result["time_frequency_matrix"],
        energy_time_frequency_matrix=result["energy_time_frequency_matrix"],
    )

    print("Experiment completed.")
    print(f"Figure: {output_dir / 'fbse_st_experiment.png'}")
    print(f"Arrays:  {output_dir / 'fbse_st_experiment.npz'}")
    plt.show()


if __name__ == "__main__":
    main()
