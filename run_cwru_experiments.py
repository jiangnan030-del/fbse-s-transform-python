"""Run FBSE-ST experiments on bearing fault signals (truncated for performance)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fbst.metrics import compare_energy_maps
from fbst.transform import fbse_s_transform
from fbst.visualization import plot_fbse_stft_comparison


SIGNAL_LENGTH = 512   # truncate to 512 samples for fast computation
SIGNALS = [
    ("cwru_normal",          "Normal bearing"),
    ("cwru_inner_fault_007", "Inner race fault 0.007\""),
    ("cwru_outer_fault_007", "Outer race fault 0.007\""),
    ("cwru_ball_fault_007",   "Ball fault 0.007\""),
]


def compute_stft_energy(signal, nperseg=64, noverlap=56):
    _, _, zxx = stft(signal, window="hann", nperseg=nperseg, noverlap=noverlap, boundary=None)
    return np.abs(zxx) ** 2


def main():
    print("=" * 70)
    print("FBSE-ST vs STFT: Bearing Fault Signal Experiments")
    print("=" * 70)

    all_results = {}

    for stem, label in SIGNALS:
        npy_path = ROOT / "data" / f"{stem}.npy"
        if not npy_path.exists():
            print(f"\n[SKIP] {npy_path.name} not found")
            continue

        raw = np.load(npy_path)
        # Take a representative segment from the middle
        start = len(raw) // 2 - SIGNAL_LENGTH // 2
        raw_segment = raw[start:start + SIGNAL_LENGTH]

        # Preprocess: demean + normalize
        signal = raw_segment - raw_segment.mean()
        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal = signal / max_abs

        print(f"\n[{label}]")
        print(f"  Signal length: {len(signal)} samples")
        print(f"  Range:  [{signal.min():.4f}, {signal.max():.4f}]")
        print(f"  Std:    {signal.std():.4f}")

        # Run FBSE-ST
        fbse_result = fbse_s_transform(
            signal,
            num_zeros=SIGNAL_LENGTH,
            sigma_scale=0.08,
            toeplitz_mode="page_style",
            normalize_output=True,
        )

        # Run STFT
        stft_energy = compute_stft_energy(signal)

        # Compare
        metrics = compare_energy_maps(
            fbse_result["energy_time_frequency_matrix"],
            stft_energy,
            top_k_ratio=0.05,
            renyi_order=3.0,
        )

        # Save outputs
        output_dir = ROOT / "outputs" / "cwru_experiments"
        output_dir.mkdir(parents=True, exist_ok=True)

        fig_path = output_dir / f"{stem}_fbse_st_vs_stft.png"
        plot_fbse_stft_comparison(
            signal,
            fbse_result["energy_time_frequency_matrix"],
            stft_energy,
            save_path=str(fig_path),
        )

        npz_path = output_dir / f"{stem}_result.npz"
        np.savez(
            npz_path,
            signal=signal,
            raw_segment=raw_segment,
            zeros=fbse_result["zeros"],
            pseudo_frequencies=fbse_result["pseudo_frequencies"],
            fbse_energy=fbse_result["energy_time_frequency_matrix"],
            stft_energy=stft_energy,
        )

        json_path = output_dir / f"{stem}_metrics.json"
        summary = {
            "label": label,
            "signal_length": len(signal),
            "metrics": metrics,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  Figure:  {fig_path}")
        print(f"  Arrays:  {npz_path}")
        print(f"  Metrics: {json_path}")
        print(f"  FBSE-ST: energy_concentration={metrics['fbse_st']['energy_concentration_ratio']:.4f}, "
              f"renyi={metrics['fbse_st']['renyi_entropy']:.4f}, "
              f"hoyer={metrics['fbse_st']['hoyer_sparsity']:.4f}")
        print(f"  STFT:    energy_concentration={metrics['stft']['energy_concentration_ratio']:.4f}, "
              f"renyi={metrics['stft']['renyi_entropy']:.4f}, "
              f"hoyer={metrics['stft']['hoyer_sparsity']:.4f}")

        all_results[stem] = summary

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary: FBSE-ST vs STFT on Bearing Fault Signals")
    print("=" * 70)
    print(f"{'Signal':<25} {'Method':<10} {'Energy Conc':<12} {'Renyi Ent':<12} {'Hoyer Sp':<12}")
    print("-" * 70)
    for stem, info in all_results.items():
        label = info["label"][:23]
        m = info["metrics"]
        print(f"{label:<25} {'FBSE-ST':<10} {m['fbse_st']['energy_concentration_ratio']:<12.4f} "
              f"{m['fbse_st']['renyi_entropy']:<12.4f} {m['fbse_st']['hoyer_sparsity']:<12.4f}")
        print(f"{'':<25} {'STFT':<10} {m['stft']['energy_concentration_ratio']:<12.4f} "
              f"{m['stft']['renyi_entropy']:<12.4f} {m['stft']['hoyer_sparsity']:<12.4f}")
        print()

    print("All figures saved to: outputs/cwru_experiments/")
    print("Done!")


if __name__ == "__main__":
    main()
