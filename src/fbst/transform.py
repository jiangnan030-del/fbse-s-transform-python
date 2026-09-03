"""Core FBSE-domain S-transform routines.

This module implements a research-oriented version of the Fourier-Bessel
domain S transform (FBSE-ST) following the four-step description on the
project page.

Notation
--------
Let:

- ``x[n]`` be the discrete signal, ``n = 1, ..., N``
- ``alpha_m`` be the positive zeros of ``J0``
- ``D`` be the Bessel basis matrix with entries
  ``D[m, n] = J0(alpha_m / N * n)``
- ``a`` be the FBSE coefficient vector
- ``T`` be a Toeplitz matrix built from ``a``
- ``G`` be a frequency-adaptive Gaussian weighting matrix
- ``W = T ⊙ G`` be the weighted coefficient-domain matrix
- ``Z = W D`` be the reconstructed complex time-frequency matrix

The module returns both ``|Z|`` and ``|Z|^2`` so downstream experiments can
work with either magnitude or energy representations.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz

from .bessel import compute_bessel_zeros
from .fbse import compute_fbse_coefficients


def compute_bessel_pseudo_frequencies(
    zeros: np.ndarray,
    signal_length: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Map Bessel zeros to a monotonic pseudo-frequency axis.

    The current implementation uses the ordering of Bessel zeros as a discrete
    frequency surrogate. When ``signal_length`` is provided, zeros are first
    scaled by ``1 / N``. When ``normalize`` is true, the result is normalized
    to the interval ``(0, 1]``.
    """
    zeros = np.asarray(zeros, dtype=float)
    if zeros.ndim != 1:
        raise ValueError("zeros must be one-dimensional")
    if len(zeros) == 0:
        raise ValueError("zeros must not be empty")

    pseudo_frequencies = zeros.copy()
    if signal_length is not None:
        if signal_length <= 0:
            raise ValueError("signal_length must be positive")
        pseudo_frequencies = pseudo_frequencies / signal_length

    if normalize:
        max_value = np.max(pseudo_frequencies)
        if max_value <= 0:
            raise ValueError("pseudo-frequencies must be positive")
        pseudo_frequencies = pseudo_frequencies / max_value

    return pseudo_frequencies


def compute_adaptive_window_widths(
    pseudo_frequencies: np.ndarray,
    sigma_scale: float = 0.08,
    min_width: float = 1e-3,
) -> np.ndarray:
    """Compute inverse-frequency Gaussian widths.

    The window-width model follows the S-transform intuition:

    ``sigma_m = c / max(f_m, eps)``

    so lower frequencies receive wider windows and higher frequencies receive
    narrower windows.
    """
    pseudo_frequencies = np.asarray(pseudo_frequencies, dtype=float)
    if pseudo_frequencies.ndim != 1:
        raise ValueError("pseudo_frequencies must be one-dimensional")
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive")

    widths = sigma_scale / np.maximum(pseudo_frequencies, min_width)
    return np.maximum(widths, min_width)


def build_frequency_distance_matrix(pseudo_frequencies: np.ndarray) -> np.ndarray:
    """Build pairwise Bessel-frequency distances.

    Returns the matrix ``Delta`` with entries:

    ``Delta[m, k] = f_k - f_m``
    """
    pseudo_frequencies = np.asarray(pseudo_frequencies, dtype=float)
    if pseudo_frequencies.ndim != 1:
        raise ValueError("pseudo_frequencies must be one-dimensional")
    return pseudo_frequencies[None, :] - pseudo_frequencies[:, None]


def build_frequency_adaptive_gaussian_window(
    zeros: np.ndarray,
    sigma_scale: float = 0.08,
    signal_length: int | None = None,
    normalize_rows: bool = True,
    min_width: float = 1e-3,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build the frequency-adaptive Gaussian weighting matrix.

    For each Bessel-frequency location ``f_m``, define a Gaussian row

    ``G[m, k] = 1 / (sqrt(2π) sigma_m) * exp(-0.5 * (Delta[m, k] / sigma_m)^2)``

    where ``sigma_m`` is inversely proportional to ``f_m``.

    If ``normalize_rows`` is enabled, each row is normalized to sum to 1.
    This makes the operator easier to compare across different parameter values.
    """
    pseudo_frequencies = compute_bessel_pseudo_frequencies(
        zeros,
        signal_length=signal_length,
        normalize=True,
    )
    widths = compute_adaptive_window_widths(
        pseudo_frequencies,
        sigma_scale=sigma_scale,
        min_width=min_width,
    )
    delta = build_frequency_distance_matrix(pseudo_frequencies)

    normalization = 1.0 / (np.sqrt(2.0 * np.pi) * widths[:, None])
    gaussian_window = normalization * np.exp(-0.5 * (delta / widths[:, None]) ** 2)

    if normalize_rows:
        row_sums = np.sum(gaussian_window, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, eps)
        gaussian_window = gaussian_window / row_sums

    return gaussian_window


def build_toeplitz_from_coefficients(
    coefficients: np.ndarray,
    mode: str = "page_style",
) -> np.ndarray:
    """Construct a Toeplitz matrix from FBSE coefficients.

    Supported modes
    ---------------
    ``page_style``:
        Uses the FBSE coefficient vector as the first column and
        ``[a0, 0, 0, ...]`` as the first row. This most closely follows the
        project page wording that the Toeplitz matrix is generated from the
        first coefficient column.

    ``hermitian``:
        Uses the conjugated coefficient vector as the first row, which can be
        useful for alternative complex-valued experiments.
    """
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")

    if mode == "page_style":
        first_row = np.zeros_like(coefficients)
        first_row[0] = coefficients[0]
    elif mode == "hermitian":
        first_row = np.conjugate(coefficients)
    else:
        raise ValueError("mode must be 'page_style' or 'hermitian'")

    return toeplitz(coefficients, first_row)


def apply_fbse_frequency_weighting(
    coefficients: np.ndarray,
    zeros: np.ndarray,
    signal_length: int | None = None,
    sigma_scale: float = 0.08,
    normalize_rows: bool = True,
    toeplitz_mode: str = "page_style",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply coefficient-domain Gaussian weighting.

    This computes

    ``T = Toeplitz(a)``
    ``G = adaptive Gaussian window``
    ``W = T ⊙ G``

    and returns ``(T, G, W)``.
    """
    toeplitz_matrix = build_toeplitz_from_coefficients(coefficients, mode=toeplitz_mode)
    gaussian_window = build_frequency_adaptive_gaussian_window(
        zeros,
        sigma_scale=sigma_scale,
        signal_length=signal_length,
        normalize_rows=normalize_rows,
    )
    weighted_spectrum = toeplitz_matrix * gaussian_window
    return toeplitz_matrix, gaussian_window, weighted_spectrum


def reconstruct_time_frequency(
    weighted_spectrum: np.ndarray,
    basis: np.ndarray,
    normalize_output: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct complex, magnitude, and energy time-frequency matrices.

    The current implementation uses the direct matrix map

    ``Z = W @ D``

    where ``W`` is the weighted coefficient-domain matrix and ``D`` is the
    Bessel basis matrix.

    Returns
    -------
    complex_tf:
        Complex-valued matrix ``Z``.
    magnitude_tf:
        Magnitude matrix ``|Z|``.
    energy_tf:
        Energy matrix ``|Z|^2``.
    """
    weighted_spectrum = np.asarray(weighted_spectrum)
    basis = np.asarray(basis)

    if weighted_spectrum.ndim != 2:
        raise ValueError("weighted_spectrum must be two-dimensional")
    if basis.ndim != 2:
        raise ValueError("basis must be two-dimensional")
    if weighted_spectrum.shape[1] != basis.shape[0]:
        raise ValueError("weighted_spectrum columns must match basis rows")

    complex_tf = weighted_spectrum @ basis
    magnitude_tf = np.abs(complex_tf)
    energy_tf = magnitude_tf**2

    if normalize_output:
        max_value = np.max(energy_tf)
        if max_value > 0:
            magnitude_tf = magnitude_tf / np.sqrt(max_value)
            energy_tf = energy_tf / max_value
            complex_tf = complex_tf / np.sqrt(max_value)

    return complex_tf, magnitude_tf, energy_tf


def fbse_s_transform(
    signal: np.ndarray,
    num_zeros: int | None = None,
    sigma_scale: float = 0.08,
    normalize_window_rows: bool = True,
    toeplitz_mode: str = "page_style",
    normalize_output: bool = False,
) -> dict[str, np.ndarray]:
    """Compute the FBSE-domain S transform.

    Pipeline
    --------
    1. Compute Bessel zeros.
    2. Map them to a pseudo-frequency axis.
    3. Compute FBSE coefficients and the Bessel basis matrix.
    4. Build the Toeplitz coefficient matrix.
    5. Build and apply the adaptive Gaussian weighting matrix.
    6. Reconstruct the complex, magnitude, and energy time-frequency outputs.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if signal.size == 0:
        raise ValueError("signal must not be empty")

    n = signal.shape[0]
    if num_zeros is None:
        num_zeros = n
    if num_zeros <= 0:
        raise ValueError("num_zeros must be positive")

    zeros = compute_bessel_zeros(num_zeros)
    pseudo_frequencies = compute_bessel_pseudo_frequencies(
        zeros,
        signal_length=n,
        normalize=True,
    )
    gaussian_widths = compute_adaptive_window_widths(
        pseudo_frequencies,
        sigma_scale=sigma_scale,
    )
    coefficients, basis = compute_fbse_coefficients(signal, zeros)
    toeplitz_matrix, gaussian_window, weighted_spectrum = apply_fbse_frequency_weighting(
        coefficients,
        zeros,
        signal_length=n,
        sigma_scale=sigma_scale,
        normalize_rows=normalize_window_rows,
        toeplitz_mode=toeplitz_mode,
    )
    complex_time_frequency_matrix, time_frequency_matrix, energy_time_frequency_matrix = reconstruct_time_frequency(
        weighted_spectrum,
        basis,
        normalize_output=normalize_output,
    )

    return {
        "zeros": zeros,
        "pseudo_frequencies": pseudo_frequencies,
        "gaussian_widths": gaussian_widths,
        "basis": basis,
        "coefficients": coefficients,
        "toeplitz_matrix": toeplitz_matrix,
        "gaussian_window": gaussian_window,
        "weighted_spectrum": weighted_spectrum,
        "complex_time_frequency_matrix": complex_time_frequency_matrix,
        "time_frequency_matrix": time_frequency_matrix,
        "energy_time_frequency_matrix": energy_time_frequency_matrix,
    }
