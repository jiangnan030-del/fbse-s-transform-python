"""FBSE-domain S transform package.

This package provides a baseline Python skeleton for Fourier-Bessel domain
S-transform experiments on non-stationary signals.
"""

from .bessel import compute_bessel_zeros
from .fbse import build_bessel_basis, compute_fbse_coefficients
from .transform import (
    build_frequency_adaptive_gaussian_window,
    build_toeplitz_from_coefficients,
    fbse_s_transform,
)
from .visualization import plot_time_frequency_matrix

__all__ = [
    "compute_bessel_zeros",
    "build_bessel_basis",
    "compute_fbse_coefficients",
    "build_frequency_adaptive_gaussian_window",
    "build_toeplitz_from_coefficients",
    "fbse_s_transform",
    "plot_time_frequency_matrix",
]
