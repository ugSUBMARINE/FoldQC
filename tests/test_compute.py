from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import compute


def _token(idx: int, *, chain_id: str = "A"):
    return types.SimpleNamespace(token_idx=idx, chain_id=chain_id)


def test_plddt_values_for_prefers_structure_and_falls_back_to_provider() -> None:
    structure = np.array([0.9, 0.8], dtype=np.float32)
    provider = np.array([0.1, 0.2], dtype=np.float32)

    values, source = compute.plddt_values_for(
        types.SimpleNamespace(structure_plddt=structure, plddt=provider)
    )
    assert values is structure
    assert source == "structure B-factors"

    values, source = compute.plddt_values_for(
        types.SimpleNamespace(structure_plddt=None, plddt=provider)
    )
    assert values is provider
    assert source == "provider pLDDT"

    values, source = compute.plddt_values_for(
        types.SimpleNamespace(structure_plddt=None, plddt=None)
    )
    assert values is None
    assert source == ""


def test_compute_plddt_uses_preferred_source_and_errors_when_missing() -> None:
    structure = np.array([0.9, 0.8], dtype=np.float32)
    provider = np.array([0.1, 0.2], dtype=np.float32)

    values = compute.compute_metric(
        "plddt",
        types.SimpleNamespace(structure_plddt=structure, plddt=provider),
        [],
    )
    np.testing.assert_array_equal(values, structure)

    with pytest.raises(compute.MissingMetricDataError):
        compute.compute_metric(
            "plddt",
            types.SimpleNamespace(structure_plddt=None, plddt=None),
            [],
        )


def test_plddt_class_routes_to_same_array_as_plddt() -> None:
    """plddt_class must be routable through compute_metric (same data as plddt)."""
    structure = np.array([0.9, 0.5, 0.3], dtype=np.float32)
    data = types.SimpleNamespace(structure_plddt=structure, plddt=None)

    plddt_result = compute.compute_metric("plddt", data, [])
    plddt_class_result = compute.compute_metric("plddt_class", data, [])
    np.testing.assert_array_equal(plddt_class_result, plddt_result)

    with pytest.raises(compute.MissingMetricDataError):
        compute.compute_metric(
            "plddt_class",
            types.SimpleNamespace(structure_plddt=None, plddt=None),
            [],
        )


def test_pae_dispatch_and_reference_requirements() -> None:
    pae = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ],
        dtype=np.float32,
    )
    data = types.SimpleNamespace(pae=pae)

    np.testing.assert_allclose(
        compute.compute_metric("pae_row_mean", data, []),
        np.array([1.0, 4.0, 7.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        compute.compute_metric("pae_col_mean", data, []),
        np.array([3.0, 4.0, 5.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        compute.compute_metric("pae_to_sel", data, [], ref_indices=[0, 2]),
        np.array([1.0, 4.0, 7.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        compute.compute_metric("pae_col_to_sel", data, [], ref_indices=[0, 2]),
        np.array([3.0, 4.0, 5.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        compute.compute_metric("pae_sym_sel", data, [], ref_indices=[1]),
        np.array([2.0, 4.0, 6.0], dtype=np.float32),
    )
    within = compute.compute_metric("pae_sym_within_sel", data, [], ref_indices=[0, 2])
    np.testing.assert_allclose(within[[0, 2]], np.array([2.0, 6.0]))
    assert np.isnan(within[1])
    contact = compute.compute_metric(
        "pae_contact",
        data,
        [],
        ref_indices=[0, 2],
        contact_indices=[1],
    )
    np.testing.assert_allclose(contact[1], 4.0)
    assert np.isnan(contact[0])
    assert np.isnan(contact[2])
    with pytest.raises(compute.MissingReferenceError):
        compute.compute_metric("pae_to_sel", data, [])
    with pytest.raises(compute.MissingReferenceError):
        compute.compute_metric("pae_contact", data, [])
    with pytest.raises(compute.MissingContactError):
        compute.compute_metric("pae_contact", data, [], ref_indices=[0])

    # PAE is all-zeros: every threshold > 0 puts all tokens in one cluster.
    labels = compute.compute_metric("pae_domain_complete", data, [], cutoff=6.5)
    assert labels.shape == (3,)
    assert len(set(labels.tolist())) == 1  # one cluster
    assert compute.pae_domain_method("pae_domain_spectral") == "spectral"
    with pytest.raises(compute.MissingCutoffError):
        compute.compute_metric("pae_domain_spectral", data, [])


def test_pde_dispatch_reference_and_contact_requirements() -> None:
    pde = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 4.0, 5.0],
            [2.0, 4.0, 0.0, 6.0],
            [3.0, 5.0, 6.0, 0.0],
        ],
        dtype=np.float32,
    )
    data = types.SimpleNamespace(pde=pde)
    token_map = [
        _token(0, chain_id="A"),
        _token(1, chain_id="A"),
        _token(2, chain_id="B"),
        _token(3, chain_id="B"),
    ]

    np.testing.assert_allclose(
        compute.compute_metric("pde_mean", data, token_map),
        np.array([1.5, 2.5, 3.0, 3.5], dtype=np.float32),
    )
    np.testing.assert_allclose(
        compute.compute_metric("pde_chain_mean", data, token_map),
        np.array([0.5, 0.5, 3.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        compute.compute_metric("pde_to_sel", data, token_map, ref_indices=[0, 2]),
        np.array([1.0, 2.5, 1.0, 4.5], dtype=np.float32),
    )
    within = compute.compute_metric(
        "pde_within_sel", data, token_map, ref_indices=[0, 2]
    )
    np.testing.assert_allclose(within[[0, 2]], np.array([1.0, 1.0]))
    assert np.isnan(within[1])
    assert np.isnan(within[3])

    contact = compute.compute_metric(
        "pde_contact",
        data,
        token_map,
        ref_indices=[0, 2],
        contact_indices=[1, 3],
    )
    assert np.isnan(contact[0])
    assert np.isnan(contact[2])
    np.testing.assert_allclose(contact[[1, 3]], np.array([2.5, 4.5]))

    with pytest.raises(compute.MissingReferenceError):
        compute.compute_metric("pde_to_sel", data, token_map)
    with pytest.raises(compute.MissingReferenceError):
        compute.compute_metric("pde_contact", data, token_map)
    with pytest.raises(compute.MissingContactError):
        compute.compute_metric("pde_contact", data, token_map, ref_indices=[0])


def test_contact_probability_dispatch_and_missing_data() -> None:
    contact_probs = np.array(
        [
            [1.0, 0.1, 0.4],
            [0.1, 1.0, 0.7],
            [0.4, 0.7, 1.0],
        ],
        dtype=np.float32,
    )
    data = types.SimpleNamespace(contact_probs=contact_probs)

    np.testing.assert_allclose(
        compute.compute_metric("contact_prob_mean", data, []),
        np.array([0.5, 0.6, 0.7], dtype=np.float32),
    )
    np.testing.assert_allclose(
        compute.compute_metric("contact_prob_to_sel", data, [], ref_indices=[0, 2]),
        np.array([np.nan, 0.4, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    with pytest.raises(compute.MissingReferenceError):
        compute.compute_metric("contact_prob_to_sel", data, [])
    with pytest.raises(compute.MissingMetricDataError):
        compute.compute_metric(
            "contact_prob_to_sel",
            types.SimpleNamespace(contact_probs=None),
            [],
            ref_indices=[0],
        )


def test_chain_iptm_dispatch_and_missing_confidence() -> None:
    token_map = [
        _token(0, chain_id="A"),
        _token(1, chain_id="A"),
        _token(2, chain_id="B"),
    ]
    data = types.SimpleNamespace(confidence={"chains_ptm": {"0": 0.8, "1": 0.6}})

    np.testing.assert_allclose(
        compute.compute_metric("chain_iptm", data, token_map),
        np.array([0.8, 0.8, 0.6], dtype=np.float32),
    )
    with pytest.raises(compute.MissingMetricDataError):
        compute.compute_metric(
            "chain_iptm", types.SimpleNamespace(confidence=None), token_map
        )


def test_unknown_metric_raises_unsupported_metric() -> None:
    with pytest.raises(compute.UnsupportedMetricError):
        compute.compute_metric("future_metric", types.SimpleNamespace(), [])

    with pytest.raises(compute.UnsupportedMetricError):
        compute.pae_domain_method("pae_row_mean")
