"""Core FBSE-domain S-transform routines."""

from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz

from .bessel import compute_bessel_zeros
from .fbse import compute_fbse_coefficients


def build_frequency_adaptive_gaussian_window(zeros: np.ndarray, sigma_scale: float = 1.0) -> np.ndarray:
    """Build a frequency-adaptive Gaussian weighting matrix.

    This is a baseline implementation intended as a clean starting point for
    experimentation. You can refine the window model later to match a target
    publication or derivation more exactly.
    """
    zeros = np.asarray(zeros, dtype=float)
    if zeros.ndim != 1:
        raise ValueError("zeros must be one-dimensional")

    m = len(zeros)
    window = np.zeros((m, m), dtype=float)

    for i in range(m):
        # Lower frequencies use wider windows; higher frequencies use narrower ones.
        width = sigma_scale / max(zeros[i], 1e-12)
        delta = zeros - zeros[i]
        window[i, :] = np.exp(-(delta**2) / (2.0 * width**2))

    return window


def build_toeplitz_from_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Construct a Toeplitz matrix from FBSE coefficients."""
    coefficients = np.asarray(coefficients)
    first_col = coefficients
    first_row = np.conjugate(coefficients)
    return toeplitz(first_col, first_row)


def fbse_s_transform(
    signal: np.ndarray,
    num_zeros: int | None = None,
    sigma_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute a baseline FBSE-domain S transform.

    Parameters
    ----------
    signal:
        Input one-dimensional signal.
    num_zeros:
        Number of Bessel zeros to use. Defaults to the signal length.
    sigma_scale:
        Scaling factor controlling Gaussian window width.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing zeros, basis, coefficients, weighted spectrum,
        and the final time-frequency matrix magnitude.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    n = signal.shape[0]
    if num_zeros is None:
        num_zeros = n

    zeros = compute_bessel_zeros(num_zeros)
    coefficients, basis = compute_fbse_coefficients(signal, zeros)
    toeplitz_matrix = build_toeplitz_from_coefficients(coefficients)
    gaussian_window = build_frequency_adaptive_gaussian_window(zeros, sigma_scale=sigma_scale)

    weighted_spectrum = toeplitz_matrix * gaussian_window
    time_frequency_matrix = np.abs(weighted_spectrum @ basis)

    return {
        "zeros": zeros,
        "basis": basis,
        "coefficients": coefficients,
        "toeplitz_matrix": toeplitz_matrix,
        "gaussian_window": gaussian_window,
        "weighted_spectrum": weighted_spectrum,
        "time_frequency_matrix": time_frequency_matrix,
    }
