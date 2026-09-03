from __future__ import annotations

import numpy as np

from fbst.metrics import (
    compare_energy_maps,
    energy_concentration_ratio,
    hoyer_sparsity,
    normalize_energy_map,
    renyi_entropy,
    summarize_energy_map,
)


def test_normalize_energy_map_sums_to_one() -> None:
    energy = np.array([[1.0, 2.0], [3.0, 4.0]])
    normalized = normalize_energy_map(energy)
    assert np.isclose(np.sum(normalized), 1.0)


def test_energy_concentration_ratio_prefers_peaked_map() -> None:
    peaked = np.array([[10.0, 0.0], [0.0, 0.0]])
    spread = np.array([[2.5, 2.5], [2.5, 2.5]])
    assert energy_concentration_ratio(peaked, top_k_ratio=0.25) > energy_concentration_ratio(spread, top_k_ratio=0.25)


def test_renyi_entropy_prefers_concentrated_map() -> None:
    peaked = np.array([[10.0, 0.0], [0.0, 0.0]])
    spread = np.array([[2.5, 2.5], [2.5, 2.5]])
    assert renyi_entropy(peaked, order=3.0) < renyi_entropy(spread, order=3.0)


def test_hoyer_sparsity_prefers_concentrated_map() -> None:
    peaked = np.array([[10.0, 0.0], [0.0, 0.0]])
    spread = np.array([[2.5, 2.5], [2.5, 2.5]])
    assert hoyer_sparsity(peaked) > hoyer_sparsity(spread)


def test_summarize_energy_map_contains_expected_keys() -> None:
    energy = np.array([[1.0, 2.0], [3.0, 4.0]])
    summary = summarize_energy_map(energy)
    assert set(summary.keys()) == {
        "energy_concentration_ratio",
        "renyi_entropy",
        "hoyer_sparsity",
    }


def test_compare_energy_maps_returns_three_sections() -> None:
    fbse = np.array([[10.0, 0.0], [0.0, 0.0]])
    stft = np.array([[2.5, 2.5], [2.5, 2.5]])
    comparison = compare_energy_maps(fbse, stft)
    assert set(comparison.keys()) == {"fbse_st", "stft", "difference"}
    assert comparison["difference"]["energy_concentration_ratio_delta"] > 0
    assert comparison["difference"]["renyi_entropy_delta"] < 0
    assert comparison["difference"]["hoyer_sparsity_delta"] > 0
