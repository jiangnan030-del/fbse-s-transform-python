"""FBSE-domain S transform package."""

from .bessel import compute_bessel_zeros
from .fbse import build_bessel_basis, compute_fbse_coefficients
from .transform import (
    apply_fbse_frequency_weighting,
    build_frequency_adaptive_gaussian_window,
    build_toeplitz_from_coefficients,
    compute_adaptive_window_widths,
    compute_bessel_pseudo_frequencies,
    fbse_s_transform,
    reconstruct_time_frequency,
)
from .visualization import plot_signal_and_time_frequency, plot_time_frequency_matrix

__all__ = [
    "compute_bessel_zeros",
    "build_bessel_basis",
    "compute_fbse_coefficients",
    "compute_bessel_pseudo_frequencies",
    "compute_adaptive_window_widths",
    "build_frequency_adaptive_gaussian_window",
    "build_toeplitz_from_coefficients",
    "apply_fbse_frequency_weighting",
    "reconstruct_time_frequency",
    "fbse_s_transform",
    "plot_time_frequency_matrix",
    "plot_signal_and_time_frequency",
]
