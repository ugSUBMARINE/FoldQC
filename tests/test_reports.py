from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import reports  # noqa: E402
from FoldQC.confidence import (  # noqa: E402
    AffinityConfidence,
    PredictionConfidence,
)
from FoldQC.loader_models import (  # noqa: E402
    ModelFiles,
    PredictionData,
    PredictionFiles,
)
from FoldQC.providers.registry import BUILTIN_PROVIDERS  # noqa: E402
from FoldQC.token_map import TokenInfo, TokenMap  # noqa: E402


def _token_map(*chain_ids: str) -> TokenMap:
    return TokenMap(
        tuple(
            TokenInfo(index, chain_id, index + 1, "ALA", False, None)
            for index, chain_id in enumerate(chain_ids)
        )
    )


def test_numeric_statistics_basic_summary() -> None:
    lines = reports.format_numeric_statistics(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )

    assert "tokens        : 4" in lines
    assert "mean          : 2.5" in lines
    assert "median        : 2.5" in lines
    assert "max           : 4" in lines


def test_numeric_statistics_ignores_nonfinite_values() -> None:
    lines = reports.format_numeric_statistics(
        np.array([1.0, np.nan, np.inf, 3.0], dtype=np.float32)
    )

    assert "tokens        : 4 (finite: 2, ignored: 2)" in lines
    assert "mean          : 2" in lines


def test_numeric_statistics_reports_no_finite_values() -> None:
    lines = reports.format_numeric_statistics(np.array([np.nan, np.inf]))

    assert lines == [
        "tokens        : 2 (finite: 0, ignored: 2)",
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
    token_map = _token_map("A", "A", "", "B")

    lines = reports.format_chain_statistics(
        np.array([1.0, 3.0, 5.0, 10.0], dtype=np.float32),
        token_map,
    )

    assert "By chain:" in lines
    assert (
        lines.index("Chain A") < lines.index("Chain (blank)") < lines.index("Chain B")
    )
    assert "tokens        : 2" in lines
    assert "tokens        : 1" in lines


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
    assert "tokens        : 2" in report


def test_statistics_report_includes_chain_statistics() -> None:
    token_map = _token_map("A", "A", "B")

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


def _confidence_data(
    provider_key: str, confidence: PredictionConfidence | None = None
) -> PredictionData:
    return PredictionData(
        name="prediction",
        rank=0,
        structure_path=Path("/tmp/model.cif"),
        provider=BUILTIN_PROVIDERS.get(provider_key).info,
        confidence=confidence,
    )


def test_confidence_summary_no_data_and_structure_only() -> None:
    assert reports.format_confidence_summary(None) == "No confidence data loaded."
    assert reports.format_confidence_summary(_confidence_data("structure_only")) == (
        "Structure-only input: pLDDT read from B-factors."
    )


def test_confidence_summary_common_fields_and_chain_arrays() -> None:
    confidence = PredictionConfidence(
        ranking_score=0.91,
        ptm=0.82,
        iptm=0.73,
        fraction_disordered=0.12,
        has_clash=False,
        chain_ptm=np.array([0.1, 0.2]),
        chain_iptm=np.array([0.3, 0.4]),
    )
    for provider_key, provider_label in (
        ("alphafold3", "AlphaFold 3"),
        ("af3_server", "AlphaFold 3 Server"),
    ):
        text = reports.format_confidence_summary(
            _confidence_data(provider_key, confidence)
        )
        assert f"provider         : {provider_label}" in text
        assert "ranking_score    : 0.9100" in text
        assert "fraction_disord. : 0.1200" in text
        assert "has_clash        : False" in text
        assert "  chain 0: 0.1000" in text
        assert "  chain 1: 0.4000" in text


def test_chai_confidence_summary_omits_unsupported_disorder_fraction() -> None:
    text = reports.format_confidence_summary(
        _confidence_data(
            "chai1",
            PredictionConfidence(
                ranking_score=0.91,
                ptm=0.82,
                iptm=0.73,
                fraction_disordered=0.12,
                has_clash=False,
            ),
        )
    )

    assert "provider         : Chai-1 Discovery" in text
    assert "ranking_score    : 0.9100" in text
    assert "fraction_disord." not in text


def test_confidence_summary_protenix_adds_gpde() -> None:
    text = reports.format_confidence_summary(
        _confidence_data(
            "protenix",
            PredictionConfidence(ranking_score=0.91, gpde=0.44),
        )
    )
    assert "provider         : Protenix" in text
    assert "gpde             : 0.4400" in text


def test_confidence_summary_boltz_schema_and_affinity() -> None:
    confidence = PredictionConfidence(
        confidence_score=0.67,
        ptm=0.96,
        iptm=0.95,
        protein_iptm=0.93,
        complex_ipde=0.44,
        chain_ptm=np.array([0.9, 0.8]),
        affinity=AffinityConfidence(predicted_value=-1.234, probability=0.8765),
    )
    for provider_key, provider_label in (
        ("boltz", "Boltz"),
        ("boltz_lab", "Boltz Lab"),
        ("boltz_api", "Boltz API"),
    ):
        text = reports.format_confidence_summary(
            _confidence_data(provider_key, confidence)
        )
        assert f"provider         : {provider_label}" in text
        assert "confidence_score : 0.6700" in text
        assert "complex_ipde     : 0.4400 Å" in text
        assert text.index("chain 0") < text.index("chain 1")
        assert "affinity_pred_value: -1.234  (log₁₀[IC₅₀/μM])" in text
        assert "affinity_probability: 0.8765" in text


def test_confidence_summary_missing_values() -> None:
    text = reports.format_confidence_summary(_confidence_data("boltz"))
    assert "provider         : Boltz" in text
    assert "No confidence data loaded." in text


def test_model_comparison_uses_provider_columns_and_formats_missing_values() -> None:
    provider = BUILTIN_PROVIDERS.get("boltz").info
    files = PredictionFiles(
        name="prediction",
        pred_dir=Path("/tmp/prediction"),
        provider=provider,
        models=[
            ModelFiles(0, Path("/tmp/model_0.cif"), "model_0", "model_0"),
            ModelFiles(1, Path("/tmp/model_1.cif"), "model_1", "model_1"),
        ],
    )

    request = reports.build_model_comparison(
        files,
        (
            PredictionConfidence(
                confidence_score=0.92,
                ptm=0.81,
                iptm=0.74,
                affinity=AffinityConfidence(
                    predicted_value=-1.2,
                    probability=0.91,
                ),
            ),
            PredictionConfidence(confidence_score=0.88, ptm=0.79),
        ),
        selected_rank=1,
    )

    assert request.selected_rank == 1
    assert [column.label for column in request.columns[:3]] == [
        "confidence_score",
        "ptm",
        "iptm",
    ]
    assert "affinity_probability" not in [column.label for column in request.columns]
    assert request.rows[0].values[:3] == ("0.9200", "0.8100", "0.7400")
    assert request.rows[1].values[:3] == ("0.8800", "0.7900", "n/a")
