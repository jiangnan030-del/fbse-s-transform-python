"""FBSE-domain S transform package."""

from .bessel import compute_bessel_zeros
from .fbse import build_bessel_basis, compute_fbse_coefficients
from .metrics import (
    compare_energy_maps,
    energy_concentration_ratio,
    hoyer_sparsity,
    normalize_energy_map,
    renyi_entropy,
    summarize_energy_map,
)
from .transform import (
    apply_fbse_frequency_weighting,
    build_frequency_adaptive_gaussian_window,
    build_frequency_distance_matrix,
    build_toeplitz_from_coefficients,
    compute_adaptive_window_widths,
    compute_bessel_pseudo_frequencies,
    fbse_s_transform,
    reconstruct_time_frequency,
)
from .visualization import (
    apply_publication_style,
    plot_fbse_stft_comparison,
    plot_signal_and_time_frequency,
    plot_time_frequency_matrix,
)

__all__ = [
    "compute_bessel_zeros",
    "build_bessel_basis",
    "compute_fbse_coefficients",
    "normalize_energy_map",
    "energy_concentration_ratio",
    "renyi_entropy",
    "hoyer_sparsity",
    "summarize_energy_map",
    "compare_energy_maps",
    "compute_bessel_pseudo_frequencies",
    "compute_adaptive_window_widths",
    "build_frequency_distance_matrix",
    "build_frequency_adaptive_gaussian_window",
    "build_toeplitz_from_coefficients",
    "apply_fbse_frequency_weighting",
    "reconstruct_time_frequency",
    "fbse_s_transform",
    "apply_publication_style",
    "plot_time_frequency_matrix",
    "plot_signal_and_time_frequency",
    "plot_fbse_stft_comparison",
]
