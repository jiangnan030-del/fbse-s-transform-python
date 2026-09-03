from __future__ import annotations

import numpy as np

from fbst.fbse import build_bessel_basis, compute_fbse_coefficients
from fbst.transform import (
    build_frequency_adaptive_gaussian_window,
    build_toeplitz_from_coefficients,
    fbse_s_transform,
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


def test_gaussian_window_shape() -> None:
    zeros = np.array([2.4048255577, 5.5200781103, 8.6537279129])
    window = build_frequency_adaptive_gaussian_window(zeros, sigma_scale=2.0)

    assert window.shape == (3, 3)
    assert np.all(window >= 0)


def test_toeplitz_shape() -> None:
    coefficients = np.array([1 + 0j, 2 + 1j, 3 - 1j])
    matrix = build_toeplitz_from_coefficients(coefficients)

    assert matrix.shape == (3, 3)


def test_fbse_s_transform_output_shapes() -> None:
    t = np.linspace(0.0, 1.0, 32, endpoint=False)
    signal = np.sin(2 * np.pi * (5 * t + 10 * t**2))

    result = fbse_s_transform(signal, num_zeros=16, sigma_scale=4.0)

    assert result["zeros"].shape == (16,)
    assert result["basis"].shape == (16, 32)
    assert result["coefficients"].shape == (16,)
    assert result["toeplitz_matrix"].shape == (16, 16)
    assert result["gaussian_window"].shape == (16, 16)
    assert result["weighted_spectrum"].shape == (16, 16)
    assert result["time_frequency_matrix"].shape == (16, 32)
