"""Load CWRU bearing .mat files and export 1-D vibration signals as .npy.

The CWRU dataset stores a single column vector named like ``X100_DE_time``
(driver-end accelerometer data at 100 samples per period).

Usage::

    python prepare_cwru_data.py

After running, ``data/`` will contain ``cwru_normal.npy``,
``cwru_inner_fault.npy`` etc. ready for ``real_signal_experiment.py``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    from scipy.io import loadmat
except ImportError:
    print("scipy is required:  pip install scipy")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# Mapping: .mat filename -> (variable name hint, output .npy name, description)
FILES = {
    "97_normal_0.mat": ("normal", "Normal bearing (12 kHz DE)"),
    "118_0.mat": ("inner_fault", "Inner race fault 0.007\" (12 kHz DE)"),
}


def extract_signal(mat_path: Path) -> np.ndarray:
    """Extract the driver-end acceleration signal from a CWRU .mat file."""
    mat = loadmat(str(mat_path))
    # Find the key that contains '_DE_time' (driver-end accelerometer)
    candidates = [
        k for k in mat.keys()
        if not k.startswith("__") and "DE_time" in k
    ]
    if not candidates:
        # Fall back: take the first non-meta key
        candidates = [k for k in mat.keys() if not k.startswith("__")]
    if not candidates:
        raise ValueError(f"No signal key found in {mat_path.name}")
    key = candidates[0]
    signal = np.asarray(mat[key], dtype=float).squeeze()
    if signal.ndim != 1:
        raise ValueError(
            f"Expected 1-D signal from key '{key}', got shape {signal.shape}"
        )
    return signal


def main() -> None:
    print("CWRU Bearing Dataset Preparation")
    print("=" * 60)

    for mat_name, (out_stem, desc) in FILES.items():
        mat_path = DATA / mat_name
        if not mat_path.exists():
            print(f"\n[SKIP] {mat_name} not found — please download it first.")
            continue

        signal = extract_signal(mat_path)
        out_npy = DATA / f"cwru_{out_stem}.npy"
        np.save(out_npy, signal)

        print(f"\n[{desc}]")
        print(f"  Source:   {mat_path.name}")
        print(f"  Signal:   {out_npy.name}")
        print(f"  Length:   {len(signal):,} samples")
        print(f"  Range:    [{signal.min():.4f}, {signal.max():.4f}]")
        print(f"  Mean:     {signal.mean():.6f}")
        print(f"  Std:      {signal.std():.6f}")

    print("\n" + "=" * 60)
    print("Done! You can now run:")
    print('  python examples/real_signal_experiment.py --input data/cwru_normal.npy --demean --normalize-max')
    print('  python examples/real_signal_experiment.py --input data/cwru_inner_fault.npy --demean --normalize-max')


if __name__ == "__main__":
    main()
