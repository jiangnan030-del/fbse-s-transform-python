"""Extract real CWRU bearing .mat files from the zip archive and run FBSE-ST experiments."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
import sys
import io

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fbst.metrics import compare_energy_maps
from fbst.transform import fbse_s_transform
from fbst.visualization import plot_fbse_stft_comparison

DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "cwru_real"
OUTPUT.mkdir(parents=True, exist_ok=True)

ZIP_PATH = DATA / "凯斯西储大学轴承数据集(1).zip"

# Representative files to extract (12k DE, 0.007 fault size, load 0)
# Format: (zip path, label)
TARGET_FILES = {
    "Normal": "Normal Baseline Data/97_0.mat",
    "Inner Race 0.007": "12k Drive End Bearing Fault Data/0.007/0/Inner Race/105.mat",
    "Outer Race 0.007": "12k Drive End Bearing Fault Data/0.007/0/Outer Race/130_6.mat",
    "Ball 0.007": "12k Drive End Bearing Fault Data/0.007/0/Ball/118.mat",
    "Inner Race 0.014": "12k Drive End Bearing Fault Data/0.014/0/Inner Race/169.mat",
    "Outer Race 0.014": "12k Drive End Bearing Fault Data/0.014/0/Outer Race/197_6.mat",
    "Ball 0.014": "12k Drive End Bearing Fault Data/0.014/0/Ball/185.mat",
    "Inner Race 0.021": "12k Drive End Bearing Fault Data/0.021/0/Inner Race/209.mat",
    "Outer Race 0.021": "12k Drive End Bearing Fault Data/0.021/0/Outer Race/234_6.mat",
    "Ball 0.021": "12k Drive End Bearing Fault Data/0.021/0/Ball/222.mat",
}

SIGNAL_LENGTH = 512  # truncation for fast computation


def extract_signal_from_mat(mat_bytes: bytes) -> np.ndarray:
    """Extract the driver-end accelerometer signal from a CWRU .mat file."""
    mat = loadmat(mat_bytes)
    candidates = [
        k for k in mat.keys()
        if not k.startswith("__") and "DE_time" in k
    ]
    if not candidates:
        candidates = [k for k in mat.keys() if not k.startswith("__")]
    if not candidates:
        raise ValueError("No signal key found in .mat file")
    key = candidates[0]
    signal = np.asarray(mat[key], dtype=float).squeeze()
    return signal


def compute_stft_energy(signal, nperseg=64, noverlap=56):
    _, _, zxx = stft(signal, window="hann", nperseg=nperseg, noverlap=noverlap, boundary=None)
    return np.abs(zxx) ** 2


def main():
    print("=" * 70)
    print("Real CWRU Bearing Data: FBSE-ST vs STFT Experiments")
    print("=" * 70)

    # Open zip and list mat files
    zf = zipfile.ZipFile(str(ZIP_PATH))

    # Build a mapping: filename -> zip info (handle encoding)
    zip_entries = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        # Try to decode the filename properly
        try:
            name = info.filename
        except Exception:
            name = info.filename
        zip_entries[Path(name).name] = info

    all_results = []

    for label, rel_path in TARGET_FILES.items():
        target_name = Path(rel_path).name
        if target_name not in zip_entries:
            print(f"\n[SKIP] {label}: {target_name} not found in zip")
            continue

        info = zip_entries[target_name]
        try:
            mat_bytes = io.BytesIO(zf.read(info))
        except Exception as e:
            print(f"\n[ERROR] {label}: could not read {target_name}: {e}")
            continue

        try:
            raw_signal = extract_signal_from_mat(mat_bytes)
        except Exception as e:
            print(f"\n[ERROR] {label}: could not extract signal: {e}")
            continue

        # Truncate to SIGNAL_LENGTH from the middle
        start = len(raw_signal) // 2 - SIGNAL_LENGTH // 2
        if start < 0:
            start = 0
        segment = raw_signal[start:start + SIGNAL_LENGTH]
        if len(segment) < SIGNAL_LENGTH:
            segment = raw_signal[:SIGNAL_LENGTH]

        # Preprocess: demean + normalize
        signal = segment - segment.mean()
        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal = signal / max_abs

        print(f"\n[{label}]")
        print(f"  Source:      {target_name}")
        print(f"  Raw length:  {len(raw_signal):,} samples")
        print(f"  Segment:     {len(signal)} samples")
        print(f"  Range:       [{signal.min():.4f}, {signal.max():.4f}]")
        print(f"  Std:         {signal.std():.4f}")

        # Run FBSE-ST
        fbse_result = fbse_s_transform(
            signal,
            num_zeros=len(signal),
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
        safe_stem = label.replace(" ", "_").replace("\"", "")
        fig_path = OUTPUT / f"{safe_stem}_fbse_st_vs_stft.png"
        plot_fbse_stft_comparison(
            signal,
            fbse_result["energy_time_frequency_matrix"],
            stft_energy,
            save_path=str(fig_path),
        )

        npz_path = OUTPUT / f"{safe_stem}_result.npz"
        np.savez(
            npz_path,
            raw_signal=raw_signal,
            segment=signal,
            fbse_energy=fbse_result["energy_time_frequency_matrix"],
            stft_energy=stft_energy,
        )

        m = metrics
        result = {
            "label": label,
            "source_file": target_name,
            "raw_length": len(raw_signal),
            "segment_length": len(signal),
            "metrics": m,
        }
        all_results.append(result)

        json_path = OUTPUT / f"{safe_stem}_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  FBSE-ST:  energy_conc={m['fbse_st']['energy_concentration_ratio']:.4f}, "
              f"renyi={m['fbse_st']['renyi_entropy']:.4f}, "
              f"hoyer={m['fbse_st']['hoyer_sparsity']:.4f}")
        print(f"  STFT:     energy_conc={m['stft']['energy_concentration_ratio']:.4f}, "
              f"renyi={m['stft']['renyi_entropy']:.4f}, "
              f"hoyer={m['stft']['hoyer_sparsity']:.4f}")
        print(f"  Figure:   {fig_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("Summary: Real CWRU Data — FBSE-ST vs STFT")
    print("=" * 80)
    print(f"{'Signal':<22} {'Method':<8} {'Energy Conc':>12} {'Renyi Ent':>12} {'Hoyer Sp':>12}")
    print("-" * 80)
    for r in all_results:
        m = r["metrics"]
        label = r["label"][:20]
        print(f"{label:<22} {'FBSE-ST':<8} {m['fbse_st']['energy_concentration_ratio']:>12.4f} "
              f"{m['fbse_st']['renyi_entropy']:>12.4f} {m['fbse_st']['hoyer_sparsity']:>12.4f}")
        print(f"{'':<22} {'STFT':<8} {m['stft']['energy_concentration_ratio']:>12.4f} "
              f"{m['stft']['renyi_entropy']:>12.4f} {m['stft']['hoyer_sparsity']:>12.4f}")
        print()

    # Save combined metrics
    combined_path = OUTPUT / "all_metrics.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"All metrics saved to: {combined_path}")
    print("All figures saved to: outputs/cwru_real/")
    print("Done!")


if __name__ == "__main__":
    main()
