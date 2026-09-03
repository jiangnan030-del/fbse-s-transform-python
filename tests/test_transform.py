from __future__ import annotations

import numpy as np

from fbst.fbse import build_bessel_basis, compute_fbse_coefficients
from fbst.transform import (
    apply_fbse_frequency_weighting,
    build_frequency_adaptive_gaussian_window,
    build_toeplitz_from_coefficients,
    compute_adaptive_window_widths,
    compute_bessel_pseudo_frequencies,
    fbse_s_transform,
    reconstruct_time_frequency,
)


def test_build_bessel_basis_shape() -> None:
    zeros = np.array([2.4048255577, 5.5200781103, 8.6537279129])
    basis = build_bessel_basis(signal_length=16, zeros=zeros)
    assert basis.shape == (3, 16)


def test_compute_fbse_coefficients_shapes() -> None:
    signal = np.linspace(0.0, 1.0, 16)
    zeros = np.array([2.4048255577, 5.5200781103, 8.6537279129])
    coefficients, basis = compute_fbse_coefficients(signal, zeros)
    assert coefficients.shape == (3,)
    assert basis.shape == (3, 16)


def test_compute_bessel_pseudo_frequencies_monotonic() -> None:
    zeros = np.array([2.4048255577, 5.5200781103, 8.6537279129])
    frequencies = compute_bessel_pseudo_frequencies(zeros, signal_length=16)
    assert frequencies.shape == (3,)
    assert np.all(np.diff(frequencies) > 0)
    assert np.isclose(frequencies[-1], 1.0)


def test_adaptive_widths_decrease_with_frequency() -> None:
    pseudo_frequencies = np.array([0.2, 0.5, 1.0])
    widths = compute_adaptive_window_widths(pseudo_frequencies, sigma_scale=0.08)
    assert widths.shape == (3,)
    assert widths[0] > widths[1] > widths[2]


def test_gaussian_window_shape_and_row_normalization() -> None:
    zeros = np.array([2.4048255577, 5.5200781103, 8.6537279129])
    window = build_frequency_adaptive_gaussian_window(
        zeros,
        sigma_scale=0.08,
        signal_length=16,
        normalize_rows=True,
    )
    assert window.shape == (3, 3)
    assert np.all(window >= 0)
    assert np.allclose(window.sum(axis=1), 1.0)


def test_toeplitz_shape() -> None:
    coefficients = np.array([1 + 0j, 2 + 1j, 3 - 1j])
    matrix = build_toeplitz_from_coefficients(coefficients)
    assert matrix.shape == (3, 3)


def test_apply_fbse_frequency_weighting_shapes() -> None:
    signal = np.linspace(0.0, 1.0, 16)
    zeros = np.array([2.4048255577, 5.5200781103, 8.6537279129])
    coefficients, _ = compute_fbse_coefficients(signal, zeros)
    toeplitz_matrix, gaussian_window, weighted_spectrum = apply_fbse_frequency_weighting(
        coefficients,
        zeros,
        signal_length=16,
        sigma_scale=0.08,
        normalize_rows=True,
    )
    assert toeplitz_matrix.shape == (3, 3)
    assert gaussian_window.shape == (3, 3)
    assert weighted_spectrum.shape == (3, 3)


def test_reconstruct_time_frequency_shapes() -> None:
    signal = np.linspace(0.0, 1.0, 16)
    zeros = np.array([2.4048255577, 5.5200781103, 8.6537279129])
    coefficients, basis = compute_fbse_coefficients(signal, zeros)
    _, _, weighted_spectrum = apply_fbse_frequency_weighting(
        coefficients,
        zeros,
        signal_length=16,
        sigma_scale=0.08,
        normalize_rows=True,
    )
    complex_tf, magnitude_tf, energy_tf = reconstruct_time_frequency(weighted_spectrum, basis)
    assert complex_tf.shape == (3, 16)
    assert magnitude_tf.shape == (3, 16)
    assert energy_tf.shape == (3, 16)
    assert np.all(magnitude_tf >= 0)
    assert np.all(energy_tf >= 0)


def test_fbse_s_transform_output_shapes() -> None:
    t = np.linspace(0.0, 1.0, 32, endpoint=False)
    signal = np.sin(2 * np.pi * (5 * t + 10 * t**2))
    result = fbse_s_transform(signal, num_zeros=16, sigma_scale=0.08)
    assert result["zeros"].shape == (16,)
    assert result["pseudo_frequencies"].shape == (16,)
    assert result["gaussian_widths"].shape == (16,)
    assert result["basis"].shape == (16, 32)
    assert result["coefficients"].shape == (16,)
    assert result["toeplitz_matrix"].shape == (16, 16)
    assert result["gaussian_window"].shape == (16, 16)
    assert result["weighted_spectrum"].shape == (16, 16)
    assert result["complex_time_frequency_matrix"].shape == (16, 32)
    assert result["time_frequency_matrix"].shape == (16, 32)
    assert result["energy_time_frequency_matrix"].shape == (16, 32)
    assert np.all(result["time_frequency_matrix"] >= 0)
    assert np.all(result["energy_time_frequency_matrix"] >= 0)
