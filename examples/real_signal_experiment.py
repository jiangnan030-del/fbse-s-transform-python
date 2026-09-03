"""Run FBSE-ST on a real signal file and compare it with STFT.

Supported input formats
-----------------------
- .npy : one-dimensional NumPy array
- .npz : specify an array key or use the first array found
- .csv/.txt/.dat : one-column or multi-column text data

Examples
--------
python examples/real_signal_experiment.py --input data/signal.npy
python examples/real_signal_experiment.py --input data/signal.csv --column 0 --delimiter ,
python examples/real_signal_experiment.py --input data/signal.npz --npz-key signal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fbst.metrics import compare_energy_maps
from fbst.transform import fbse_s_transform
from fbst.visualization import plot_fbse_stft_comparison


TEXT_SUFFIXES = {".csv", ".txt", ".dat"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FBSE-ST and STFT comparison on a real signal file"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to signal file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/real_signal_experiment",
        help="Directory for figures, arrays, and metrics",
    )
    parser.add_argument(
        "--num-zeros",
        type=int,
        default=None,
        help="Number of Bessel zeros to use (default: signal length)",
    )
    parser.add_argument(
        "--sigma-scale",
        type=float,
        default=0.08,
        help="Adaptive Gaussian width scale",
    )
    parser.add_argument(
        "--column",
        type=int,
        default=0,
        help="Column index for multi-column text data",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="Delimiter for text data. Use 'whitespace' for space-separated files.",
    )
    parser.add_argument(
        "--skiprows",
        type=int,
        default=0,
        help="Rows to skip when loading text data",
    )
    parser.add_argument(
        "--npz-key",
        type=str,
        default=None,
        help="Array key to load from an .npz file",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=1.0,
        help="Sampling rate used only for reporting",
    )
    parser.add_argument(
        "--demean",
        action="store_true",
        help="Subtract the mean before analysis",
    )
    parser.add_argument(
        "--normalize-max",
        action="store_true",
        help="Normalize by maximum absolute amplitude before analysis",
    )
    parser.add_argument(
        "--stft-nperseg",
        type=int,
        default=64,
        help="STFT window length",
    )
    parser.add_argument(
        "--stft-noverlap",
        type=int,
        default=56,
        help="STFT overlap length",
    )
    return parser.parse_args()



def load_signal_from_file(
    path: Path,
    column: int = 0,
    delimiter: str = ",",
    skiprows: int = 0,
    npz_key: str | None = None,
) -> np.ndarray:
    """Load a one-dimensional signal from disk."""
    suffix = path.suffix.lower()

    if suffix == ".npy":
        signal = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        keys = list(archive.keys())
        if not keys:
            raise ValueError("The .npz file does not contain any arrays")
        key = npz_key or keys[0]
        if key not in archive:
            raise ValueError(f"Array key '{key}' not found in {path.name}")
        signal = archive[key]
    elif suffix in TEXT_SUFFIXES:
        actual_delimiter = None if delimiter.lower() == "whitespace" else delimiter
        data = np.loadtxt(path, delimiter=actual_delimiter, skiprows=skiprows)
        if data.ndim == 1:
            signal = data
        else:
            if column < 0 or column >= data.shape[1]:
                raise ValueError(
                    f"column must be in [0, {data.shape[1] - 1}] for file {path.name}"
                )
            signal = data[:, column]
    else:
        raise ValueError(
            "Unsupported file format. Use .npy, .npz, .csv, .txt, or .dat"
        )

    signal = np.asarray(signal, dtype=float).squeeze()
    if signal.ndim != 1:
        raise ValueError("Loaded signal must be one-dimensional after squeezing")
    if signal.size == 0:
        raise ValueError("Loaded signal is empty")
    return signal



def preprocess_signal(
    signal: np.ndarray,
    demean: bool = False,
    normalize_max: bool = False,
) -> np.ndarray:
    """Apply simple optional preprocessing."""
    processed = np.asarray(signal, dtype=float).copy()

    if demean:
        processed = processed - np.mean(processed)

    if normalize_max:
        max_abs = np.max(np.abs(processed))
        if max_abs > 0:
            processed = processed / max_abs

    return processed



def compute_stft_energy(
    signal: np.ndarray,
    nperseg: int = 64,
    noverlap: int = 56,
) -> np.ndarray:
    """Compute STFT energy for comparison."""
    _, _, zxx = stft(
        signal,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )
    return np.abs(zxx) ** 2



def main() -> None:
    args = parse_args()
    input_path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_signal = load_signal_from_file(
        input_path,
        column=args.column,
        delimiter=args.delimiter,
        skiprows=args.skiprows,
        npz_key=args.npz_key,
    )
    signal = preprocess_signal(
        raw_signal,
        demean=args.demean,
        normalize_max=args.normalize_max,
    )

    num_zeros = args.num_zeros or len(signal)
    fbse_result = fbse_s_transform(
        signal,
        num_zeros=num_zeros,
        sigma_scale=args.sigma_scale,
        toeplitz_mode="page_style",
        normalize_output=True,
    )
    stft_energy = compute_stft_energy(
        signal,
        nperseg=args.stft_nperseg,
        noverlap=args.stft_noverlap,
    )
    metrics = compare_energy_maps(
        fbse_result["energy_time_frequency_matrix"],
        stft_energy,
        top_k_ratio=0.05,
        renyi_order=3.0,
    )

    stem = input_path.stem
    figure_path = output_dir / f"{stem}_fbse_st_vs_stft.png"
    arrays_path = output_dir / f"{stem}_fbse_st_vs_stft.npz"
    metrics_path = output_dir / f"{stem}_comparison_metrics.json"

    plot_fbse_stft_comparison(
        signal,
        fbse_result["energy_time_frequency_matrix"],
        stft_energy,
        save_path=str(figure_path),
    )

    np.savez(
        arrays_path,
        raw_signal=raw_signal,
        processed_signal=signal,
        zeros=fbse_result["zeros"],
        pseudo_frequencies=fbse_result["pseudo_frequencies"],
        gaussian_widths=fbse_result["gaussian_widths"],
        coefficients=fbse_result["coefficients"],
        fbse_time_frequency_matrix=fbse_result["time_frequency_matrix"],
        fbse_energy_time_frequency_matrix=fbse_result["energy_time_frequency_matrix"],
        stft_energy=stft_energy,
    )

    summary = {
        "input_file": str(input_path),
        "sampling_rate": args.sampling_rate,
        "signal_length": int(len(signal)),
        "num_zeros": int(num_zeros),
        "sigma_scale": float(args.sigma_scale),
        "demean": bool(args.demean),
        "normalize_max": bool(args.normalize_max),
        "stft_nperseg": int(args.stft_nperseg),
        "stft_noverlap": int(args.stft_noverlap),
        "metrics": metrics,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Real signal experiment completed.")
    print(f"Input file:        {input_path}")
    print(f"Sampling rate:     {args.sampling_rate}")
    print(f"Signal length:     {len(signal)}")
    print(f"Comparison figure: {figure_path}")
    print(f"Saved arrays:      {arrays_path}")
    print(f"Saved metrics:     {metrics_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    plt.show()


if __name__ == "__main__":
    main()
