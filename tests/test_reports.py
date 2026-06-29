from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import reports  # noqa: E402


def test_numeric_statistics_basic_summary() -> None:
    lines = reports.format_numeric_statistics(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )

    assert "tokens        : 4 (finite: 4)" in lines
    assert "mean          : 2.5" in lines
    assert "median        : 2.5" in lines
    assert "max           : 4" in lines


def test_numeric_statistics_ignores_nonfinite_values() -> None:
    lines = reports.format_numeric_statistics(
        np.array([1.0, np.nan, np.inf, 3.0], dtype=np.float32)
    )

    assert "tokens        : 4 (finite: 2)" in lines
    assert "ignored       : 2" in lines
    assert "mean          : 2" in lines


def test_numeric_statistics_reports_no_finite_values() -> None:
    lines = reports.format_numeric_statistics(np.array([np.nan, np.inf]))

    assert lines == [
        "tokens        : 2 (finite: 0)",
        "ignored       : 2",
        "No finite values.",
    ]


def test_plddt_class_statistics_counts_fractional_scale() -> None:
    lines = reports.format_plddt_class_statistics(
        np.array([0.95, 0.75, 0.55, 0.20, np.nan], dtype=np.float32)
    )

    assert "  Very high (>=90) : 1 (25.0%)" in lines
    assert "  High (70-90)     : 1 (25.0%)" in lines
    assert "  Low (50-70)      : 1 (25.0%)" in lines
    assert "  Very low (<50)   : 1 (25.0%)" in lines


def test_plddt_class_statistics_counts_percent_scale() -> None:
    lines = reports.format_plddt_class_statistics(
        np.array([95.0, 75.0, 55.0, 20.0], dtype=np.float32)
    )

    assert "  Very high (>=90) : 1 (25.0%)" in lines
    assert "  High (70-90)     : 1 (25.0%)" in lines
    assert "  Low (50-70)      : 1 (25.0%)" in lines
    assert "  Very low (<50)   : 1 (25.0%)" in lines


def test_plddt_class_statistics_reports_no_finite_values() -> None:
    lines = reports.format_plddt_class_statistics(np.array([np.nan]))

    assert lines == ["pLDDT classes:", "  No finite pLDDT values."]


def test_chain_statistics_groups_values_by_chain_order() -> None:
    token_map = [
        types.SimpleNamespace(chain_id="A"),
        types.SimpleNamespace(chain_id="A"),
        types.SimpleNamespace(chain_id=""),
        types.SimpleNamespace(chain_id="B"),
    ]

    lines = reports.format_chain_statistics(
        np.array([1.0, 3.0, 5.0, 10.0], dtype=np.float32),
        token_map,
    )

    assert "By chain:" in lines
    assert (
        lines.index("Chain A") < lines.index("Chain (blank)") < lines.index("Chain B")
    )
    assert "tokens        : 2 (finite: 2)" in lines
    assert "tokens        : 1 (finite: 1)" in lines


def test_domain_label_statistics_counts_clusters() -> None:
    lines = reports.format_domain_label_statistics(
        np.array([0.0, 0.0, 1.0, np.nan, 3.0], dtype=np.float32)
    )

    assert "domain labels:" in lines
    assert "  clusters      : 3" in lines
    assert "  label 0      : 2 tokens" in lines
    assert "  label 1      : 1 tokens" in lines
    assert "  label 3      : 1 tokens" in lines


def test_domain_label_statistics_reports_no_finite_values() -> None:
    lines = reports.format_domain_label_statistics(np.array([np.nan]))

    assert lines == ["domain labels:", "  No finite domain labels."]


def test_statistics_report_single_target() -> None:
    report = reports.format_statistics_report(
        "plddt",
        "target_model_0",
        [("target_model_0", np.array([0.8, 0.9], dtype=np.float32))],
    )

    assert "pLDDT \u2014 continuous" in report
    assert "Target: target_model_0" in report
    assert "mean" in report
    assert "Overall (pooled)" not in report


def test_statistics_report_pooled_ensemble() -> None:
    report = reports.format_statistics_report(
        "ensemble_rmsd",
        "target_ensemble",
        [
            ("target_model_0", np.array([1.0, 2.0], dtype=np.float32)),
            ("target_model_1", np.array([3.0, 4.0], dtype=np.float32)),
        ],
    )

    assert "Ensemble RMSD, aligned" in report
    assert "Target: target_ensemble" in report
    assert "Overall (pooled)" in report
    assert "target_model_0" in report
    assert "target_model_1" in report
    assert "mean          : 2.5" in report


def test_statistics_report_ensemble_level_single_array() -> None:
    report = reports.format_statistics_report(
        "ensemble_rmsd",
        "target_ensemble",
        [("target_ensemble", np.array([1.0, 2.0], dtype=np.float32))],
    )

    assert "Target: target_ensemble" in report
    assert "Overall (pooled)" not in report
    assert "tokens        : 2 (finite: 2)" in report


def test_statistics_report_includes_chain_statistics() -> None:
    token_map = [
        types.SimpleNamespace(chain_id="A"),
        types.SimpleNamespace(chain_id="A"),
        types.SimpleNamespace(chain_id="B"),
    ]

    report = reports.format_statistics_report(
        "pde_chain_mean",
        "target_model_0",
        [
            (
                "target_model_0",
                np.array([1.0, 3.0, 10.0], dtype=np.float32),
                token_map,
            )
        ],
        include_chain_stats=True,
    )

    assert "PDE \u2014 within-chain mean" in report
    assert "By chain:" in report
    assert "Chain A" in report
    assert "Chain B" in report


def test_statistics_report_includes_plddt_classes() -> None:
    report = reports.format_statistics_report(
        "plddt_class",
        "target_model_0",
        [("target_model_0", np.array([0.95, 0.75, 0.55, 0.20]))],
        include_plddt_classes=True,
    )

    assert "pLDDT classes:" in report
    assert "  Very high (>=90) : 1 (25.0%)" in report


def test_statistics_report_keeps_domain_labels_member_local() -> None:
    report = reports.format_statistics_report(
        "pae_domain_complete",
        "target_ensemble",
        [
            ("target_model_0", np.array([0.0, 0.0, 1.0])),
            ("target_model_1", np.array([0.0, 1.0, 1.0])),
        ],
        include_domain_labels=True,
    )

    assert "Overall: not pooled; domain labels are member-local." in report
    assert "domain labels:" in report
    assert "  clusters      : 2" in report


def test_confidence_summary_no_data_and_structure_only() -> None:
    assert reports.format_confidence_summary(None) == "No confidence data loaded."
    assert (
        reports.format_confidence_summary(
            types.SimpleNamespace(provider="structure_only")
        )
        == "Structure-only input: pLDDT read from B-factors."
    )


def test_confidence_summary_af3_fields_and_chain_lists() -> None:
    text = reports.format_confidence_summary(
        types.SimpleNamespace(
            provider="alphafold3",
            confidence=None,
            summary_confidence={
                "ranking_score": 0.91,
                "ptm": 0.82,
                "iptm": 0.73,
                "fraction_disordered": 0.12,
                "has_clash": False,
                "chain_ptm": [0.1, 0.2],
                "chain_iptm": [0.3, 0.4],
            },
        )
    )

    assert "provider         : AlphaFold 3" in text
    assert "ranking_score    : 0.9100" in text
    assert "fraction_disord. : 0.1200" in text
    assert "has_clash        : False" in text
    assert "chain_ptm:" in text
    assert "  chain 0: 0.1000" in text
    assert "chain_iptm:" in text
    assert "  chain 1: 0.4000" in text


def test_confidence_summary_af3_server_without_confidence() -> None:
    text = reports.format_confidence_summary(
        types.SimpleNamespace(
            provider="af3_server",
            confidence=None,
            summary_confidence=None,
        )
    )

    assert "provider         : AlphaFold 3 Server" in text
    assert "No confidence data loaded." in text


def test_confidence_summary_chai_fields_and_chain_scores() -> None:
    text = reports.format_confidence_summary(
        types.SimpleNamespace(
            provider="chai1",
            confidence={
                "ranking_score": 0.91,
                "ptm": 0.82,
                "iptm": 0.73,
                "has_clash": False,
                "chains_ptm": {"0": 0.8, "1": 0.7},
            },
            summary_confidence=None,
        )
    )

    assert "provider         : Chai-1 Discovery" in text
    assert "ranking_score    : 0.9100" in text
    assert "ptm              : 0.8200" in text
    assert "iptm             : 0.7300" in text
    assert "has_clash        : False" in text
    assert "chain_ptm:" in text
    assert "  chain 1: 0.7000" in text


def test_confidence_summary_protenix_fields_and_gpde() -> None:
    text = reports.format_confidence_summary(
        types.SimpleNamespace(
            provider="protenix",
            confidence={
                "ranking_score": 0.91,
                "ptm": 0.82,
                "iptm": 0.73,
                "gpde": 0.44,
                "disorder": 0.0,
                "has_clash": False,
                "chains_ptm": {"0": 0.8, "1": 0.7},
                "chains_iptm": {"0": 0.6, "1": 0.5},
            },
            summary_confidence=None,
        )
    )

    assert "provider         : Protenix" in text
    assert "ranking_score    : 0.9100" in text
    assert "fraction_disord. : 0.0000" in text
    assert "gpde             : 0.4400" in text
    assert "chain_ptm:" in text
    assert "chain_iptm:" in text


def test_confidence_summary_boltz_fields_chain_sorting_and_affinity() -> None:
    text = reports.format_confidence_summary(
        types.SimpleNamespace(
            provider="boltz",
            confidence={
                "confidence_score": 0.67,
                "ptm": 0.96,
                "iptm": 0.95,
                "ligand_iptm": 0.94,
                "protein_iptm": 0.93,
                "complex_plddt": 0.61,
                "complex_iplddt": 0.62,
                "complex_pde": 0.43,
                "complex_ipde": 0.44,
                "chains_ptm": {"1": 0.8, "0": 0.9},
            },
            affinity={
                "affinity_pred_value": -1.234,
                "affinity_probability_binary": 0.8765,
            },
        )
    )

    assert "provider         : Boltz-2" in text
    assert "confidence_score : 0.6700" in text
    assert "protein_iptm" in text
    assert "complex_iplddt" in text
    assert "complex_ipde     : 0.4400 Å" in text
    assert "chains_ptm:" in text
    assert text.index("chain 0") < text.index("chain 1")
    assert "affinity_pred_value       : -1.234  (log₁₀[IC₅₀/μM])" in text
    assert "affinity_probability      : 0.8765" in text


def test_confidence_summary_boltz_without_confidence() -> None:
    text = reports.format_confidence_summary(
        types.SimpleNamespace(
            provider="boltz",
            confidence=None,
            summary_confidence=None,
        )
    )

    assert "provider         : Boltz-2" in text
    assert "No confidence data loaded." in text
