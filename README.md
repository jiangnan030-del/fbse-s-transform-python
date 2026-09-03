# fbse-s-transform-python

Python implementation of the **Fourier-Bessel domain S transform (FBSE-ST)** for energy-adaptive time-frequency analysis of non-stationary signals.

## Overview

This repository implements a research-oriented workflow inspired by the method described on the project page:

- **Fourier-Bessel series expansion (FBSE)** for sparse coefficient representation
- **S-transform-style adaptive Gaussian weighting** in the Bessel-frequency domain
- **Inverse Bessel-basis reconstruction** for concentrated time-frequency analysis
- **STFT comparison workflow** for baseline evaluation

The current codebase is designed as a transparent **research prototype** rather than a finalized benchmark package. It is suitable for algorithm development, exploratory experiments, figure generation, and further mathematical refinement.

## Current repository capabilities

The repository currently includes:

- computation of positive zeros of the zero-order Bessel function
- construction of the Bessel basis matrix
- computation of FBSE coefficients
- coefficient-domain Toeplitz matrix construction
- frequency-adaptive Gaussian weighting in the Bessel domain
- reconstruction of complex, magnitude, and energy time-frequency matrices
- synthetic-signal demos
- **FBSE-ST vs STFT** comparison experiment
- figure export and array export workflows
- a Jupyter notebook for interactive testing
- basic unit tests

## Method pipeline

The implementation follows four main steps.

### Step 1. Compute Bessel zeros

The method first computes the positive zeros of the zero-order Bessel function \(J_0(x)\), then maps them to a monotonic pseudo-frequency axis.

### Step 2. Compute FBSE coefficients

Given a signal \(x[n]\), the code builds a Bessel basis matrix and computes its Fourier-Bessel expansion coefficients.

### Step 3. Apply adaptive Gaussian weighting

A Toeplitz matrix is constructed from the FBSE coefficient vector, then multiplied elementwise by a frequency-adaptive Gaussian window matrix. This step is intended to concentrate leaked energy around the target Bessel-frequency locations.

### Step 4. Reconstruct the time-frequency representation

The weighted coefficient-domain matrix is projected back through the Bessel basis to obtain the final complex, magnitude, and energy time-frequency distributions.

## Mathematical notes

A stricter formulation note is included here:

- `docs/method_formulation.md`

That document explains the current matrix interpretation used by the implementation, including:

- pseudo-frequency mapping
- adaptive Gaussian widths
- Toeplitz construction choice
- reconstruction step
- relation to the page-style four-step method description

## Project structure

```text
fbse-s-transform-python/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── docs/
│   └── method_formulation.md
├── src/
│   └── fbst/
│       ├── __init__.py
│       ├── bessel.py
│       ├── fbse.py
│       ├── transform.py
│       └── visualization.py
├── examples/
│   ├── demo_signal.py
│   └── reproduce_experiment.py
├── notebooks/
│   └── fbse_st_demo.ipynb
└── tests/
    ├── test_bessel.py
    └── test_transform.py
```

## Installation

### Minimal install

```bash
git clone https://github.com/jiangnan030-del/fbse-s-transform-python.git
cd fbse-s-transform-python
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Development install

```bash
pip install -r requirements-dev.txt
pytest
```

## Quick start

### Run the demo

```bash
python examples/demo_signal.py
```

This generates a synthetic non-stationary signal and computes its FBSE-ST representation.

### Run the reproducible comparison experiment

```bash
python examples/reproduce_experiment.py
```

This script will:

- generate a synthetic signal
- compute the FBSE-ST result
- compute an STFT baseline
- save a comparison figure
- save arrays to an `.npz` file

Default outputs:

- `outputs/repro_experiment/fbse_st_vs_stft.png`
- `outputs/repro_experiment/fbse_st_vs_stft.npz`

### Run the notebook

Open:

- `notebooks/fbse_st_demo.ipynb`

The notebook includes an interactive FBSE-ST and STFT comparison workflow.

## Notes on the current implementation

This repository currently implements a **research-friendly approximation** of the method description. In particular:

- the Bessel-frequency axis is represented as a pseudo-frequency axis derived from Bessel zeros
- the Gaussian weighting is implemented as an inverse-frequency adaptive window
- the Toeplitz construction includes a `page_style` mode intended to follow the wording of the page description more closely
- the code is structured to make future formula-level refinement straightforward

## Suggested next development directions

Potential next steps include:

- stricter derivation-to-code alignment for the weighting operator
- parameter studies for Gaussian width selection
- quantitative energy-concentration metrics
- comparison with SST, SET, or other sharpened time-frequency methods
- support for real experimental signals

## Maintainer

Maintained by **jiangnan**.

## License

Released under the **MIT License**.
