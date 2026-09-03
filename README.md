# FBSE S-Transform Python

Python implementation of the **Fourier-Bessel domain S transform (FBSE-ST)** for energy-adaptive time-frequency analysis of non-stationary signals.

## Overview

This project is intended to reproduce and extend the method described in the current notes page: a time-frequency analysis framework that combines:

- **Fourier-Bessel series expansion (FBSE)** for compact spectral representation
- **S-transform-style adaptive Gaussian windows** for frequency-dependent resolution
- **Inverse Bessel-basis reconstruction** for obtaining a concentrated time-frequency representation

The method is especially suitable for **non-stationary signals** with transient or time-varying frequency content.

## Why this project is useful

Traditional short-time Fourier analysis faces a fixed resolution trade-off between time and frequency. The FBSE-domain S transform aims to improve energy concentration by:

- preserving better low-/high-frequency adaptive resolution
- reducing spectral leakage around instantaneous-frequency ridges
- improving interpretability for transient, impulsive, and non-stationary signals

Potential application areas include:

- mechanical vibration analysis
- speech processing
- biomedical signal analysis
- general non-stationary time-frequency analysis research

## Method pipeline

The implementation follows four main steps:

1. **Compute zeros of the zero-order Bessel function** using Newton iteration.
2. **Build the Bessel basis matrix** and compute FBSE coefficients.
3. **Construct a frequency-adaptive Gaussian window matrix** and perform coefficient-domain weighting.
4. **Reconstruct the time-frequency matrix** through inverse mapping with the Bessel basis.

## Project goals

- Implement the FBSE-domain S transform in Python
- Provide reusable functions for each stage of the algorithm
- Offer examples on synthetic non-stationary signals
- Visualize high-resolution time-frequency representations
- Create a clean baseline for future experiments and comparisons

## Getting started

### Recommended environment

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

### Install dependencies

```bash
pip install numpy scipy matplotlib
```

### Planned structure

```text
fbse-s-transform-python/
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── bessel_zeros.py
│   ├── fbse.py
│   ├── windowing.py
│   └── fbse_st.py
└── examples/
    └── demo_signal.py
```

## Minimal code idea

The current method notes already include core logic for:

- computing Bessel zeros
- constructing the Bessel basis matrix
- computing FBSE coefficients

The next implementation step is to organize these pieces into reusable Python modules and add example scripts for visualization.

## Help and maintenance

Maintainer: **jiangnan**

If you use this repository for research development, you can extend it with:

- benchmark datasets or synthetic test signals
- comparisons with STFT, synchrosqueezing, or related methods
- notebook demos and figure reproduction scripts

## License

This repository is intended to use the MIT License.
