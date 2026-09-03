from __future__ import annotations

import numpy as np

from fbst.bessel import compute_bessel_zeros


def test_compute_bessel_zeros_returns_sorted_positive_values() -> None:
    zeros = compute_bessel_zeros(5)

    assert zeros.shape == (5,)
    assert np.all(zeros > 0)
    assert np.all(np.diff(zeros) > 0)


def test_first_bessel_zero_is_close_to_reference_value() -> None:
    zeros = compute_bessel_zeros(1)

    assert np.isclose(zeros[0], 2.4048255577, atol=1e-6)
