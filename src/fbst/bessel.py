"""Bessel-related utilities."""

from __future__ import annotations

import numpy as np
from scipy.special import jn


def compute_bessel_zeros(num_zeros: int, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """Compute the first ``num_zeros`` positive zeros of J0(x).

    Parameters
    ----------
    num_zeros:
        Number of positive zeros to compute.
    tol:
        Convergence tolerance for Newton iteration.
    max_iter:
        Maximum iteration count for each root.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_zeros,)`` containing the positive zeros.
    """
    if num_zeros <= 0:
        raise ValueError("num_zeros must be a positive integer")

    zeros = np.zeros(num_zeros, dtype=float)
    x0 = 2.0

    for i in range(num_zeros):
        for _ in range(max_iter):
            j0 = jn(0, x0)
            j1 = jn(1, x0)

            if np.isclose(j1, 0.0):
                x0 += 1e-10
                continue

            delta = -j0 / j1
            x0 -= delta

            if abs(delta) < tol:
                break

        zeros[i] = x0
        x0 += np.pi

    return zeros
