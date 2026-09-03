"""Core FBSE-domain S-transform routines.

This module provides a more structured implementation of the method described
in the project notes:

1. Compute Fourier-Bessel series expansion (FBSE) coefficients.
2. Build a coefficient-domain Toeplitz matrix.
3. Construct a frequency-adaptive Gaussian weighting matrix.
4. Reconstruct the time-frequency representation in the Bessel basis.

The implementation remains intentionally transparent and research-friendly,
while being closer to the method description than the initial baseline.
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
    """Map Bessel zeros to a monotonically increasing pseudo-frequency axis.

    Parameters
    ----------
    zeros:
        Positive zeros of the zero-order Bessel function.
    signal_length:
        Optional signal length used to scale the zero locations.
    normalize:
        If true, normalize frequencies by their maximum value for numerical
        stability in the Gaussian weighting stage.
    """
    zeros = np.asarray(zeros, dtype=float)
    if zeros.ndim != 1:
        raise ValueError("zeros must be one-dimensional")
    if len(zeros) == 0:
        raise ValueError("zeros must not be empty")

    frequencies = zeros.copy()
    if signal_length is not None:
        if signal_length <= 0:
            raise ValueError("signal_length must be positive")
        frequencies = frequencies / signal_length

    if normalize:
        max_frequency = np.max(frequencies)
        if max_frequency <= 0:
            raise ValueError("frequencies must be positive")
        frequencies = frequencies / max_frequency

    return frequencies


def build_frequency_adaptive_gaussian_window(
    zeros: np.ndarray,
    sigma_scale: float = 1.0,
    signal_length: int | None = None,
    normalize_rows: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build a frequency-adaptive Gaussian weighting matrix.

    The weighting follows the page description more closely than the original
    baseline by:

    - deriving a pseudo-frequency axis from the Bessel zeros;
    - using widths inversely proportional to the local pseudo-frequency;
    - optionally row-normalizing the weights so the weighting stage behaves
      like a localized coefficient redistribution instead of an uncontrolled
      gain change.
    """
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive")

    frequencies = compute_bessel_pseudo_frequencies(
        zeros,
        signal_length=signal_length,
        normalize=True,
    )
    m = len(frequencies)
    window = np.zeros((m, m), dtype=float)

    for i in range(m):
        local_width = sigma_scale / max(frequencies[i], eps)
        delta = frequencies - frequencies[i]
        window[i, :] = np.exp(-(delta**2) / (2.0 * local_width**2))

    if normalize_rows:
        row_sums = np.sum(window, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, eps)
        window = window / row_sums

    return window


def build_toeplitz_from_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Construct a Toeplitz matrix from FBSE coefficients.

    The first column uses the coefficient vector directly. The first row uses
    the conjugated vector so that complex-valued inputs remain well behaved.
    """
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")

    first_col = coefficients
    first_row = np.conjugate(coefficients)
    return toeplitz(first_col, first_row)


def apply_fbse_frequency_weighting(
    coefficients: np.ndarray,
    zeros: np.ndarray,
    signal_length: int | None = None,
    sigma_scale: float = 1.0,
    normalize_rows: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Gaussian weighting in the Fourier-Bessel coefficient domain.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(toeplitz_matrix, gaussian_window, weighted_spectrum)``.
    """
    toeplitz_matrix = build_toeplitz_from_coefficients(coefficients)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct complex and magnitude time-frequency matrices."""
    weighted_spectrum = np.asarray(weighted_spectrum)
    basis = np.asarray(basis)

    if weighted_spectrum.ndim != 2:
        raise ValueError("weighted_spectrum must be two-dimensional")
    if basis.ndim != 2:
        raise ValueError("basis must be two-dimensional")
    if weighted_spectrum.shape[1] != basis.shape[0]:
        raise ValueError(
            "weighted_spectrum columns must match the number of basis rows"
        )

    complex_tf = weighted_spectrum @ basis
    magnitude_tf = np.abs(complex_tf)
    return complex_tf, magnitude_tf


def fbse_s_transform(
    signal: np.ndarray,
    num_zeros: int | None = None,
    sigma_scale: float = 1.0,
    normalize_window_rows: bool = True,
) -> dict[str, np.ndarray]:
    """Compute an FBSE-domain S transform.

    Parameters
    ----------
    signal:
        Input one-dimensional signal.
    num_zeros:
        Number of Bessel zeros to use. Defaults to the signal length.
    sigma_scale:
        Scale factor for the adaptive Gaussian window width.
    normalize_window_rows:
        Whether to row-normalize the Gaussian weighting matrix.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the intermediate and final transform outputs.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    n = signal.shape[0]
    if n == 0:
        raise ValueError("signal must not be empty")

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
    coefficients, basis = compute_fbse_coefficients(signal, zeros)
    toeplitz_matrix, gaussian_window, weighted_spectrum = apply_fbse_frequency_weighting(
        coefficients,
        zeros,
        signal_length=n,
        sigma_scale=sigma_scale,
        normalize_rows=normalize_window_rows,
    )
    complex_time_frequency_matrix, time_frequency_matrix = reconstruct_time_frequency(
        weighted_spectrum,
        basis,
    )

    return {
        "zeros": zeros,
        "pseudo_frequencies": pseudo_frequencies,
        "basis": basis,
        "coefficients": coefficients,
        "toeplitz_matrix": toeplitz_matrix,
        "gaussian_window": gaussian_window,
        "weighted_spectrum": weighted_spectrum,
        "complex_time_frequency_matrix": complex_time_frequency_matrix,
        "time_frequency_matrix": time_frequency_matrix,
    }
