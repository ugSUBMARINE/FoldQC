from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import compute, metrics


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
    assert metrics.DEFAULT_METRIC_KEY == "plddt_class"
    assert metrics.METRICS.require(metrics.DEFAULT_METRIC_KEY).label.startswith("pLDDT")


def test_metric_registry_owns_behavioral_metadata() -> None:
    assert metrics.METRICS.require("plddt_class").group == "pLDDT"
    assert metrics.METRICS.require("plddt").load_capabilities == {"plddt"}
    assert metrics.METRICS.require("pae_contact").needs_contact_shell
    assert metrics.METRICS.require("pae_contact").tier == "advanced"
    assert metrics.METRICS.require("pae_domain_complete").dependency_keys == ("scipy",)
    assert metrics.METRICS.require("pae_domain_spectral").dependency_keys == (
        "scipy",
        "sklearn",
    )
    assert metrics.METRICS.require("chain_iptm").needs_confidence
    assert metrics.METRICS.find("missing") is None
    with pytest.raises(KeyError, match="Unknown metric"):
        metrics.METRICS.require("missing")


@pytest.mark.parametrize(
    ("key", "label"),
    [
        ("pae_row_mean", "  PAE — row mean"),
        ("pae_col_to_sel", "  PAE — column mean to selection [Advanced]"),
        ("pde_contact", "  PDE — contact-filtered to selection [Advanced]"),
        (
            "pae_domain_spectral",
            "  PAE — domain labels (spectral clustering) [Experimental]",
        ),
    ],
)
def test_property_combo_label_adds_tier_marker(key: str, label: str) -> None:
    assert metrics.property_combo_label(metrics.METRICS.require(key)) == label


def test_metric_plot_matrix_and_export_metadata() -> None:
    pae = metrics.METRICS.require("pae_to_sel")
    assert pae.load_capabilities == {"pae"}
    assert pae.reference_scoped_plots == {"matrix"}
    assert pae.matrix == metrics.MatrixSpec(
        "pae", "Predicted Aligned Error (Å)", "PAE (Å)"
    )
    domain = metrics.METRICS.require("pae_domain_spectral")
    assert domain.is_domain_label
    assert domain.line_ylabel == "Domain label"
    ensemble = metrics.METRICS.require("ensemble_plddt_std")
    assert ensemble.aggregate_kind == "ensemble_std"
    chain = metrics.METRICS.require("chain_iptm")
    assert (chain.value_unit, chain.value_semantics) == (
        "iptm",
        "higher_is_better",
    )


def test_plot_registry_is_ordered_and_strict() -> None:
    assert [spec.key for spec in metrics.PLOTS] == [
        "line",
        "distribution",
        "matrix",
        "pae_summary",
        "pde_summary",
        "binding_site_fingerprint",
        "ensemble_site_summary",
    ]
    assert metrics.PLOTS.require("matrix").dependency_keys == ("matplotlib",)
    assert not metrics.PLOTS.require("pae_summary").requires_metric
    assert metrics.PLOTS.find("missing") is None


def test_registry_rejects_duplicates_and_invalid_combinations() -> None:
    spec = metrics.METRICS.require("plddt")
    with pytest.raises(ValueError, match="unique"):
        metrics.MetricRegistry((spec, spec))
    invalid = metrics.MetricSpec(
        "invalid",
        "Invalid",
        "Test",
        frozenset({"plddt"}),
        "plddt",
        "higher_is_better",
        "pLDDT",
        "{target_text}",
        "Details for {target_text}.",
        needs_contact_shell=True,
    )
    with pytest.raises(ValueError, match="contact shell"):
        metrics.MetricRegistry((invalid,))
    with pytest.raises(ValueError, match="unique"):
        plot = metrics.PLOTS.require("line")
        metrics.PlotRegistry((plot, plot))


def test_registry_values_are_immutable_and_validate_cross_field_contracts() -> None:
    spec = metrics.METRICS.require("pae_to_sel")
    with pytest.raises(FrozenInstanceError):
        spec.label = "changed"
    with pytest.raises(ValueError, match="missing required fields"):
        metrics.MetricRegistry((replace(spec, preview_template="{target_text}"),))
    with pytest.raises(ValueError, match="unknown preview fields"):
        metrics.MetricRegistry(
            (replace(spec, preview_template="{target_text} {ref_sel} {unknown}"),)
        )
    with pytest.raises(ValueError, match="details are missing required fields"):
        metrics.MetricRegistry((replace(spec, details_template="{target_text}"),))
    with pytest.raises(ValueError, match="unknown details fields"):
        metrics.MetricRegistry(
            (replace(spec, details_template="{target_text} {ref_sel} {unknown}"),)
        )
    with pytest.raises(ValueError, match="not one of its data requirements"):
        metrics.MetricRegistry((replace(spec, requirements=frozenset({"pde"})),))
    with pytest.raises(ValueError, match="Unknown plot key"):
        metrics.PlotRegistry((replace(metrics.PLOTS[0], key="unknown"),))


def test_preview_templates_and_dispatch_are_complete() -> None:
    for spec in metrics.METRICS:
        assert spec.preview_template.format(
            target_text="T", ref_sel="R", cutoff="5.0 Å"
        )
        assert spec.details_template.format(
            target_text="T", ref_sel="R", cutoff="5.0 Å"
        )
    compute.validate_dispatch_registry()
