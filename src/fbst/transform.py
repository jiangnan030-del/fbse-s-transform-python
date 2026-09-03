"""Core FBSE-domain S-transform routines.

This module implements a research-friendly version of the Fourier-Bessel
domain S transform (FBSE-ST). The structure follows the method description
in the project notes:

1. Compute Fourier-Bessel series expansion (FBSE) coefficients.
2. Build a coefficient-domain Toeplitz matrix.
3. Construct a frequency-adaptive Gaussian weighting matrix.
4. Reconstruct the time-frequency representation through the Bessel basis.
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
    """Map Bessel zeros to a monotonic pseudo-frequency axis."""
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

    Lower pseudo-frequencies receive wider windows, while higher
    pseudo-frequencies receive narrower windows.
    """
    pseudo_frequencies = np.asarray(pseudo_frequencies, dtype=float)
    if pseudo_frequencies.ndim != 1:
        raise ValueError("pseudo_frequencies must be one-dimensional")
    if sigma_scale <= 0:
        raise ValueError("sigma_scale must be positive")

    widths = sigma_scale / np.maximum(pseudo_frequencies, min_width)
    widths = np.maximum(widths, min_width)
    return widths


def build_frequency_adaptive_gaussian_window(
    zeros: np.ndarray,
    sigma_scale: float = 0.08,
    signal_length: int | None = None,
    normalize_rows: bool = True,
    min_width: float = 1e-3,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build the frequency-adaptive Gaussian weighting matrix.

    The window is defined on the pseudo-frequency axis derived from the Bessel
    zeros. Each row is centered at one pseudo-frequency and uses an adaptive
    width inversely proportional to that pseudo-frequency.
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

    delta = pseudo_frequencies[None, :] - pseudo_frequencies[:, None]
    window = np.exp(-0.5 * (delta / widths[:, None]) ** 2)

    if normalize_rows:
        row_sums = np.sum(window, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, eps)
        window = window / row_sums

    return window


def build_toeplitz_from_coefficients(coefficients: np.ndarray) -> np.ndarray:
    """Construct a Toeplitz matrix from FBSE coefficients."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")

    return toeplitz(coefficients, np.conjugate(coefficients))


def apply_fbse_frequency_weighting(
    coefficients: np.ndarray,
    zeros: np.ndarray,
    signal_length: int | None = None,
    sigma_scale: float = 0.08,
    normalize_rows: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply coefficient-domain Gaussian weighting."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct complex, magnitude, and energy time-frequency matrices."""
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
    return complex_tf, magnitude_tf, energy_tf


def fbse_s_transform(
    signal: np.ndarray,
    num_zeros: int | None = None,
    sigma_scale: float = 0.08,
    normalize_window_rows: bool = True,
) -> dict[str, np.ndarray]:
    """Compute the FBSE-domain S transform."""
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
    )
    complex_time_frequency_matrix, time_frequency_matrix, energy_time_frequency_matrix = reconstruct_time_frequency(
        weighted_spectrum,
        basis,
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
