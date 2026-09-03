"""Generate synthetic bearing vibration signals with realistic fault characteristics.

Based on CWRU test bearing (SKF 6205) characteristic frequencies:
  - BPFI (Ball Pass Frequency Inner race): ~159.9 Hz
  - BPFO (Ball Pass Frequency Outer race): ~107.4 Hz
  - BPFC (Ball Pass Frequency Cage):       ~141.1 Hz
  - FTF  (Fundamental Train Frequency):     ~11.9 Hz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

def main() -> None:
    np.random.seed(42)
    fs = 12000   # 12 kHz sampling rate (CWRU standard)
    N = 12000    # 1 second of data
    t = np.arange(N) / fs

    # Bearing characteristic frequencies (SKF 6205)
    f_inner = 159.9   # BPFI
    f_outer = 107.4    # BPFO
    f_ball = 141.1     # BPFC

    # --- 1. Normal bearing ---
    normal = (
        0.3 * np.random.randn(N)
        + 0.2 * np.sin(2 * np.pi * 30 * t)
        + 0.1 * np.sin(2 * np.pi * 60 * t)
    )
    np.save(DATA / "cwru_normal.npy", normal)
    print(f"[1/4] cwru_normal.npy: {len(normal)} samples, std={normal.std():.4f}")

    # --- 2. Inner race fault (impulses at BPFI, carrier ~2500 Hz) ---
    inner_fault = normal.copy()
    for i in range(int(f_inner)):
        center = int(i * fs / f_inner)
        if 0 <= center < N:
            local_t = t - center / fs
            impulse = np.exp(-3000 * local_t ** 2) * 50 * np.sin(2 * np.pi * 2500 * local_t)
            inner_fault += impulse
    np.save(DATA / "cwru_inner_fault_007.npy", inner_fault)
    print(f"[2/4] cwru_inner_fault_007.npy: {len(inner_fault)} samples, std={inner_fault.std():.4f}")

    # --- 3. Outer race fault (impulses at BPFO, carrier ~2000 Hz) ---
    outer_fault = normal.copy()
    for i in range(int(f_outer)):
        center = int(i * fs / f_outer)
        if 0 <= center < N:
            local_t = t - center / fs
            impulse = np.exp(-3000 * local_t ** 2) * 50 * np.sin(2 * np.pi * 2000 * local_t)
            outer_fault += impulse
    np.save(DATA / "cwru_outer_fault_007.npy", outer_fault)
    print(f"[3/4] cwru_outer_fault_007.npy: {len(outer_fault)} samples, std={outer_fault.std():.4f}")

    # --- 4. Ball fault (impulses at BPFC, carrier ~1800 Hz) ---
    ball_fault = normal.copy()
    for i in range(int(f_ball)):
        center = int(i * fs / f_ball)
        if 0 <= center < N:
            local_t = t - center / fs
            impulse = np.exp(-3000 * local_t ** 2) * 50 * np.sin(2 * np.pi * 1800 * local_t)
            ball_fault += impulse
    np.save(DATA / "cwru_ball_fault_007.npy", ball_fault)
    print(f"[4/4] cwru_ball_fault_007.npy: {len(ball_fault)} samples, std={ball_fault.std():.4f}")

    print(f"\nAll signals saved to {DATA}/")
    print("\nRun with:")
    print("  py examples/real_signal_experiment.py --input data/cwru_normal.npy --demean --normalize-max")
    print("  py examples/real_signal_experiment.py --input data/cwru_inner_fault_007.npy --demean --normalize-max")
    print("  py examples/real_signal_experiment.py --input data/cwru_outer_fault_007.npy --demean --normalize-max")
    print("  py examples/real_signal_experiment.py --input data/cwru_ball_fault_007.npy --demean --normalize-max")


if __name__ == "__main__":
    main()
