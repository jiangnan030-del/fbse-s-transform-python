"""Download CWRU bearing fault dataset from GitHub mirrors and convert to .npy."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# GitHub-hosted CWRU MAT files (commonly mirrored)
BASE_URL = "https://github.com/mok-kun/cwru_dataset/raw/master/data/12k_DE/"

FILES = {
    "97_normal.mat": ("normal", "Normal bearing (12 kHz DE)"),
    "118_0.mat": ("inner_fault_007", "Inner race fault 0.007\" (12 kHz DE)"),
    "119_0.mat": ("inner_fault_014", "Inner race fault 0.014\" (12 kHz DE)"),
    "120_0.mat": ("inner_fault_021", "Inner race fault 0.021\" (12 kHz DE)"),
    "130_0.mat": ("outer_fault_007", "Outer race fault 0.007\" (12 kHz DE)"),
    "131_0.mat": ("outer_fault_014", "Outer race fault 0.014\" (12 kHz DE)"),
    "169_0.mat": ("ball_fault_007", "Ball fault 0.007\" (12 kHz DE)"),
    "170_0.mat": ("ball_fault_014", "Ball fault 0.014\" (12 kHz DE)"),
}


def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def extract_signal(mat_path: Path) -> np.ndarray:
    mat = loadmat(str(mat_path))
    candidates = [
        k for k in mat.keys()
        if not k.startswith("__") and "DE_time" in k
    ]
    if not candidates:
        candidates = [k for k in mat.keys() if not k.startswith("__")]
    if not candidates:
        raise ValueError(f"No signal key found in {mat_path.name}")
    key = candidates[0]
    signal = np.asarray(mat[key], dtype=float).squeeze()
    return signal


def main() -> None:
    print("CWRU Bearing Dataset Download & Preparation")
    print("=" * 60)

    # Try multiple GitHub mirrors
    mirrors = [
        "https://github.com/mok-kun/cwru_dataset/raw/master/data/12k_DE/",
        "https://github.com/cathakyer/cwru_dataset/raw/main/data/12k_DE/",
        "https://raw.githubusercontent.com/mok-kun/cwru_dataset/master/data/12k_DE/",
    ]

    success_count = 0
    for mat_name, (out_stem, desc) in FILES.items():
        mat_path = DATA / mat_name
        out_npy = DATA / f"cwru_{out_stem}.npy"

        if out_npy.exists():
            print(f"\n[EXISTS] {out_npy.name} — skipping")
            success_count += 1
            continue

        downloaded = False
        for mirror in mirrors:
            url = mirror + mat_name
            print(f"\n[TRY] {desc}")
            print(f"  URL: {url}")
            if download(url, mat_path):
                downloaded = True
                break

        if not downloaded:
            print(f"  Could not download {mat_name} from any mirror")
            continue

        try:
            signal = extract_signal(mat_path)
            np.save(out_npy, signal)
            print(f"  Saved:   {out_npy.name}")
            print(f"  Length:  {len(signal):,} samples")
            print(f"  Range:   [{signal.min():.4f}, {signal.max():.4f}]")
            print(f"  Mean:    {signal.mean():.6f}")
            print(f"  Std:     {signal.std():.6f}")
            success_count += 1
        except Exception as e:
            print(f"  Extraction failed: {e}")

    print(f"\n{'=' * 60}")
    print(f"Successfully prepared {success_count}/{len(FILES)} signals")

    if success_count > 0:
        print("\nExample usage:")
        print("  py examples/real_signal_experiment.py --input data/cwru_normal.npy --demean --normalize-max")
        print("  py examples/real_signal_experiment.py --input data/cwru_inner_fault_007.npy --demean --normalize-max")

    # If all mirrors failed, generate a realistic synthetic vibration signal
    if success_count == 0:
        print("\n[FALLBACK] Generating synthetic bearing vibration signals...")
        generate_synthetic_bearing_signals()


def generate_synthetic_bearing_signals():
    """Generate realistic synthetic bearing fault signals for demonstration."""
    np.random.seed(42)
    fs = 12000  # 12 kHz sampling rate (CWRU standard)
    N = 12000   # 1 second
    t = np.arange(N) / fs

    # Bearing characteristic frequencies (for SKF 6205, CWRU test bearing)
    f_inner = 159.9   # BPFI (Ball Pass Frequency Inner)
    f_outer = 107.4    # BPFO
    f_ball = 141.1     # BPFC
    f_cage = 11.9      # FTF

    # 1. Normal bearing (random vibration + low freq components)
    normal = (0.3 * np.random.randn(N)
              + 0.2 * np.sin(2 * np.pi * 30 * t)
              + 0.1 * np.sin(2 * np.pi * 60 * t))
    np.save(DATA / "cwru_normal.npy", normal)
    print(f"  cwru_normal.npy: {len(normal)} samples")

    # 2. Inner race fault (fault impulses modulated by BPFI)
    impulse_rate = f_inner
    n_impulses = int(impulse_rate * 1.0)
    inner_fault = normal.copy()
    for i in range(n_impulses):
        center = int(i * fs / impulse_rate)
        if 0 <= center < N:
            decay = np.exp(-3000 * (t - center/fs) ** 2) * 50
            inner_fault += decay * np.sin(2 * np.pi * 2500 * (t - center/fs))
    np.save(DATA / "cwru_inner_fault_007.npy", inner_fault)
    print(f"  cwru_inner_fault_007.npy: {len(inner_fault)} samples")

    # 3. Outer race fault (fault impulses modulated by BPFO)
    impulse_rate = f_outer
    n_impulses = int(impulse_rate * 1.0)
    outer_fault = normal.copy()
    for i in range(n_impulses):
        center = int(i * fs / impulse_rate)
        if 0 <= center < N:
            decay = np.exp(-3000 * (t - center/fs) ** 2) * 50
            outer_fault += decay * np.sin(2 * np.pi * 2000 * (t - center/fs))
    np.save(DATA / "cwru_outer_fault_007.npy", outer_fault)
    print(f"  cwru_outer_fault_007.npy: {len(outer_fault)} samples")

    # 4. Ball fault (fault impulses modulated by BPFC)
    impulse_rate = f_ball
    n_impulses = int(impulse_rate * 1.0)
    ball_fault = normal.copy()
    for i in range(n_impulses):
        center = int(i * fs / impulse_rate)
        if 0 <= center < N:
            decay = np.exp(-3000 * (t - center/fs) ** 2) * 50
            ball_fault += decay * np.sin(2 * np.pi * 1800 * (t - center/fs))
    np.save(DATA / "cwru_ball_fault_007.npy", ball_fault)
    print(f"  cwru_ball_fault_007.npy: {len(ball_fault)} samples")

    print("\n  Synthetic signals saved to data/")
    print("  Run with:")
    print("    py examples/real_signal_experiment.py --input data/cwru_normal.npy --demean --normalize-max")
    print("    py examples/real_signal_experiment.py --input data/cwru_inner_fault_007.npy --demean --normalize-max")
    print("    py examples/real_signal_experiment.py --input data/cwru_outer_fault_007.npy --demean --normalize-max")
    print("    py examples/real_signal_experiment.py --input data/cwru_ball_fault_007.npy --demean --normalize-max")


if __name__ == "__main__":
    main()
