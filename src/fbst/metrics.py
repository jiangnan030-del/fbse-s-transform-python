"""Quantitative evaluation metrics for time-frequency representations."""

from __future__ import annotations

import math
import numpy as np


def _validate_energy_map(energy_map: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy_map, dtype=float)
    if energy.ndim != 2:
        raise ValueError("energy_map must be two-dimensional")
    if np.any(energy < 0):
        raise ValueError("energy_map must be nonnegative")
    return energy


def normalize_energy_map(energy_map: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Normalize a nonnegative energy map to sum to 1."""
    energy = _validate_energy_map(energy_map)
    total = float(np.sum(energy))
    if total <= eps:
        raise ValueError("energy_map sum must be positive")
    return energy / total


def energy_concentration_ratio(energy_map: np.ndarray, top_k_ratio: float = 0.05) -> float:
    """Return the fraction of total energy contained in the top-k entries.

    Larger values indicate stronger energy concentration.
    """
    if not 0 < top_k_ratio <= 1:
        raise ValueError("top_k_ratio must be in (0, 1]")

    probability = normalize_energy_map(energy_map).ravel()
    k = max(1, int(math.ceil(probability.size * top_k_ratio)))
    largest = np.partition(probability, -k)[-k:]
    return float(np.sum(largest))


def renyi_entropy(energy_map: np.ndarray, order: float = 3.0, eps: float = 1e-15) -> float:
    """Compute Rényi entropy of a normalized energy distribution.

    Lower values indicate a more concentrated representation.
    """
    if order <= 0 or math.isclose(order, 1.0):
        raise ValueError("order must be positive and not equal to 1")

    probability = np.clip(normalize_energy_map(energy_map).ravel(), eps, None)
    return float((1.0 / (1.0 - order)) * np.log(np.sum(probability**order)))


def hoyer_sparsity(energy_map: np.ndarray, eps: float = 1e-15) -> float:
    """Compute Hoyer sparsity on the normalized energy vector.

    Values closer to 1 indicate stronger sparsity.
    """
    probability = normalize_energy_map(energy_map, eps=eps).ravel()
    n = probability.size
    l1 = np.linalg.norm(probability, ord=1)
    l2 = np.linalg.norm(probability, ord=2)
    if l2 <= eps:
        raise ValueError("l2 norm must be positive")
    return float((math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1.0))


def summarize_energy_map(energy_map: np.ndarray, top_k_ratio: float = 0.05, renyi_order: float = 3.0) -> dict[str, float]:
    """Summarize a time-frequency energy map with several scalar metrics."""
    return {
        "energy_concentration_ratio": energy_concentration_ratio(energy_map, top_k_ratio=top_k_ratio),
        "renyi_entropy": renyi_entropy(energy_map, order=renyi_order),
        "hoyer_sparsity": hoyer_sparsity(energy_map),
    }


def compare_energy_maps(
    fbse_energy_map: np.ndarray,
    stft_energy_map: np.ndarray,
    top_k_ratio: float = 0.05,
    renyi_order: float = 3.0,
) -> dict[str, dict[str, float]]:
    """Compare scalar concentration metrics for FBSE-ST and STFT."""
    fbse_summary = summarize_energy_map(
        fbse_energy_map,
        top_k_ratio=top_k_ratio,
        renyi_order=renyi_order,
    )
    stft_summary = summarize_energy_map(
        stft_energy_map,
        top_k_ratio=top_k_ratio,
        renyi_order=renyi_order,
    )

    improvement = {
        "energy_concentration_ratio_delta": fbse_summary["energy_concentration_ratio"] - stft_summary["energy_concentration_ratio"],
        "renyi_entropy_delta": fbse_summary["renyi_entropy"] - stft_summary["renyi_entropy"],
        "hoyer_sparsity_delta": fbse_summary["hoyer_sparsity"] - stft_summary["hoyer_sparsity"],
    }

    return {
        "fbse_st": fbse_summary,
        "stft": stft_summary,
        "difference": improvement,
    }
