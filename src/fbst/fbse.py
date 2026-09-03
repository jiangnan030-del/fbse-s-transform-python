"""Fourier-Bessel series expansion helpers."""

from __future__ import annotations

import numpy as np
from scipy.special import jn


def build_bessel_basis(signal_length: int, zeros: np.ndarray) -> np.ndarray:
    """Build the Bessel basis matrix.

    Parameters
    ----------
    signal_length:
        Length of the input signal.
    zeros:
        Positive zeros of the zero-order Bessel function.

    Returns
    -------
    np.ndarray
        Basis matrix of shape ``(len(zeros), signal_length)``.
    """
    if signal_length <= 0:
        raise ValueError("signal_length must be positive")

    sample_index = np.arange(1, signal_length + 1, dtype=float)
    basis = np.zeros((len(zeros), signal_length), dtype=float)

    for m, zero in enumerate(zeros):
        basis[m, :] = jn(0, zero / signal_length * sample_index)

    return basis


def compute_fbse_coefficients(signal: np.ndarray, zeros: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute FBSE coefficients and the corresponding Bessel basis.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.
    zeros:
        Positive zeros of the zero-order Bessel function.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(coefficients, basis_matrix)``.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    n = signal.shape[0]
    nb = np.arange(1, n + 1, dtype=float)
    basis = build_bessel_basis(n, zeros)
    coefficients = np.zeros(len(zeros), dtype=complex)

    for m, zero in enumerate(zeros):
        j1_val = jn(1, zero)
        scale = 2.0 / (n**2 * j1_val**2)
        coefficients[m] = scale * np.sum(nb * signal * basis[m, :])

    return coefficients, basis
