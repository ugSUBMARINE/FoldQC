from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import metrics


def test_metric_keys_preserve_gui_order() -> None:
    assert [spec.key for spec in metrics.METRICS] == [
        "plddt_class",
        "plddt",
        "pae_row_mean",
        "pae_col_mean",
        "pae_to_sel",
        "pae_col_to_sel",
        "pae_sym_sel",
        "pae_sym_within_sel",
        "pae_contact",
        "pae_domain_complete",
        "pae_domain_spectral",
        "pde_mean",
        "pde_chain_mean",
        "pde_to_sel",
        "pde_within_sel",
        "pde_contact",
        "contact_prob_mean",
        "contact_prob_to_sel",
        "ensemble_rmsd",
        "ensemble_plddt_mean",
        "ensemble_plddt_std",
        "chain_iptm",
    ]


def test_property_dicts_match_metric_specs() -> None:
    by_key = {prop["key"]: prop for prop in metrics.PROPERTIES}

    assert by_key["plddt_class"]["group"] == "pLDDT"
    assert by_key["plddt"]["needs_any_plddt"] is True
    assert by_key["pae_row_mean"]["tier"] == "normal"
    assert by_key["pae_contact"]["tier"] == "advanced"
    assert by_key["pae_domain_complete"]["tier"] == "experimental"
    assert by_key["pde_within_sel"]["tier"] == "advanced"
    assert by_key["chain_iptm"]["group"] == "Chain/interface"
    assert metrics.METRIC_BY_KEY["chain_iptm"].needs_confidence is True


@pytest.mark.parametrize(
    ("key", "label"),
    [
        ("pae_row_mean", "  PAE \u2014 row mean"),
        (
            "pae_col_to_sel",
            "  PAE \u2014 column mean to selection [Advanced]",
        ),
        ("pde_contact", "  PDE \u2014 contact-filtered to selection [Advanced]"),
        (
            "pae_domain_spectral",
            "  PAE \u2014 domain labels (spectral clustering) [Experimental]",
        ),
    ],
)
def test_property_combo_label_adds_tier_marker(key: str, label: str) -> None:
    assert metrics.property_combo_label(metrics.METRIC_BY_KEY[key]) == label
    assert metrics.property_combo_label(metrics.PROPERTY_BY_KEY[key]) == label


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (
            "plddt",
            {
                "load_pae": False,
                "load_pde": False,
                "load_contact_probs": False,
                "load_structure_plddt": True,
                "load_plddt": True,
            },
        ),
        (
            "pae_to_sel",
            {
                "load_pae": True,
                "load_pde": False,
                "load_contact_probs": False,
                "load_structure_plddt": False,
                "load_plddt": False,
            },
        ),
        (
            "pae_contact",
            {
                "load_pae": True,
                "load_pde": False,
                "load_contact_probs": False,
                "load_structure_plddt": False,
                "load_plddt": False,
            },
        ),
        (
            "pde_contact",
            {
                "load_pae": False,
                "load_pde": True,
                "load_contact_probs": False,
                "load_structure_plddt": False,
                "load_plddt": False,
            },
        ),
        (
            "contact_prob_to_sel",
            {
                "load_pae": False,
                "load_pde": False,
                "load_contact_probs": True,
                "load_structure_plddt": False,
                "load_plddt": False,
            },
        ),
        (
            "ensemble_plddt_mean",
            {
                "load_pae": False,
                "load_pde": False,
                "load_contact_probs": False,
                "load_structure_plddt": True,
                "load_plddt": True,
            },
        ),
    ],
)
def test_metric_load_flags(key: str, expected: dict[str, bool]) -> None:
    assert metrics.metric_load_flags(metrics.METRIC_BY_KEY[key]) == expected
    assert metrics.metric_load_flags(metrics.PROPERTY_BY_KEY[key]) == expected


def test_metric_classification_and_plot_helpers() -> None:
    assert metrics.is_domain_label_metric("pae_domain_complete") is True
    assert metrics.is_domain_label_metric("pae_row_mean") is False
    assert metrics.metric_label("plddt") == "pLDDT \u2014 continuous"
    assert metrics.metric_label("unknown_metric") == "unknown_metric"
    assert metrics.line_compute_key("plddt_class") == "plddt"
    assert metrics.line_ylabel("pae_domain_spectral") == "Domain label"
    assert metrics.line_ylabel("contact_prob_mean") == "Interaction probability"
    assert metrics.line_ylabel("chain_iptm") == "ipTM"
    assert metrics.plot_uses_reference_scope("pae_to_sel", "line") is False
    assert metrics.plot_uses_reference_scope("pae_to_sel", "distribution") is False
    assert metrics.plot_uses_reference_scope("pae_to_sel", "matrix") is True
    assert metrics.plot_uses_reference_scope("pae_col_to_sel", "line") is False
    assert metrics.plot_uses_reference_scope("pae_contact", "distribution") is False
    assert metrics.plot_uses_reference_scope("pae_sym_within_sel", "line") is True
    assert metrics.plot_uses_reference_scope("pde_within_sel", "line") is True
    assert metrics.plot_uses_reference_scope("plddt", "distribution") is True
    assert metrics.plot_uses_reference_scope("chain_iptm", "distribution") is False
    assert metrics.plot_uses_reference_scope("chain_iptm", "matrix") is False
    assert metrics.matrix_source_for_metric("pae_to_sel") == (
        "pae",
        "Predicted Aligned Error (\u00c5)",
        "PAE (\u00c5)",
    )
    assert metrics.matrix_source_for_metric("chain_iptm") == (
        "chain_iptm",
        "Pairwise chain ipTM",
        "ipTM",
    )
    assert metrics.matrix_source_for_metric("plddt") is None


def test_ensemble_aggregate_kind() -> None:
    assert metrics.ensemble_aggregate_kind("ensemble_rmsd") == "ensemble_rmsd"
    assert metrics.ensemble_aggregate_kind("ensemble_plddt_mean") == "ensemble_mean"
    assert metrics.ensemble_aggregate_kind("ensemble_plddt_std") == "ensemble_std"
    assert metrics.ensemble_aggregate_kind("future_ensemble_metric") == "ensemble_mean"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("plddt_class", ("plddt", "higher_is_better")),
        ("pae_domain_complete", ("label", "categorical_label")),
        ("pae_contact", ("angstrom", "lower_is_better")),
        ("pde_contact", ("angstrom", "lower_is_better")),
        ("contact_prob_mean", ("probability", "higher_is_better")),
        ("chain_iptm", ("iptm", "higher_is_better")),
        ("ensemble_plddt_std", ("plddt", "lower_is_better")),
        ("unknown_metric", ("", "")),
    ],
)
def test_metric_units_and_semantics(key: str, expected: tuple[str, str]) -> None:
    assert metrics.metric_units_and_semantics(key) == expected
