from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.confidence import (
    AffinityConfidence,
    ConfidenceFieldSpec,
    ConfidenceSectionSpec,
    ConfidenceSummarySpec,
    PredictionConfidence,
    merge_prediction_confidence,
    parse_prediction_confidence,
    validate_prediction_confidence,
)
from FoldQC.provider_errors import ProviderContractError


def _parse(payload, *, affinity=None, chains=2):
    return parse_prediction_confidence(
        payload,
        chain_count=chains,
        provider="test_provider",
        model_label="rank 0",
        source=Path("/tmp/confidence.json"),
        affinity_payload=affinity,
    )


def test_parser_normalizes_aliases_chain_order_diagonal_and_affinity() -> None:
    confidence = _parse(
        {
            "aggregate_score": 0.91,
            "structure_confidence": 0.88,
            "disorder": 0.12,
            "has_inter_chain_clashes": False,
            "chains_ptm": {"1": 0.7, "0": 0.8},
            "chain_iptm": [0.6, None],
            "chain_pair_iptm": [[0.0, 0.4], [None, 0.0]],
            "unknown_provider_field": "discard me",
        },
        affinity={
            "affinity_pred_value": -1.2,
            "affinity_probability_binary": 0.75,
            "unknown": 1,
        },
    )

    assert confidence.ranking_score == 0.91
    assert confidence.confidence_score == 0.88
    assert confidence.fraction_disordered == 0.12
    assert confidence.has_clash is False
    np.testing.assert_allclose(confidence.chain_ptm, [0.8, 0.7])
    np.testing.assert_allclose(confidence.chain_iptm, [0.6, np.nan], equal_nan=True)
    np.testing.assert_allclose(
        confidence.pair_chain_iptm,
        [[0.8, 0.4], [np.nan, 0.7]],
        equal_nan=True,
    )
    assert confidence.affinity == AffinityConfidence(-1.2, 0.75)
    assert not hasattr(confidence, "unknown_provider_field")
    for array in (
        confidence.chain_ptm,
        confidence.chain_iptm,
        confidence.pair_chain_iptm,
    ):
        assert array.dtype == np.float32
        assert array.flags.c_contiguous
        assert not array.flags.writeable


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"ptm": "0.8"}, "ptm must be numeric"),
        ({"iptm": np.inf}, "iptm must be finite"),
        ({"has_clash": 1}, "has_clash must be boolean"),
        ({"chain_ptm": [0.8]}, "must contain 2 chain values"),
        ({"chains_iptm": {"2": 0.8}}, "outside 0..1"),
        ({"chain_pair_iptm": [[0.0], [0.2]]}, r"shape \(2, 2\)"),
        ({"pair_chains_iptm": {"x": {"0": 0.2}}}, "not an integer"),
    ],
)
def test_parser_rejects_malformed_recognized_fields(payload, message: str) -> None:
    with pytest.raises(ProviderContractError, match=message) as caught:
        _parse(payload)
    text = str(caught.value)
    assert "test_provider" in text
    assert "rank 0" in text
    assert "confidence.json" in text


def test_parser_rejects_non_object_sources_and_ignores_empty_unknown_payload() -> None:
    with pytest.raises(
        ProviderContractError, match="confidence data must be an object"
    ):
        _parse([])
    with pytest.raises(ProviderContractError, match="affinity data must be an object"):
        _parse({}, affinity=[])
    assert _parse({"unknown": 1}) is None


def test_typed_confidence_validates_direct_values_and_chain_shapes() -> None:
    confidence = PredictionConfidence(
        ptm=1,
        chain_ptm=np.array([0.8, np.nan]),
        pair_chain_iptm=np.eye(2),
    )
    assert confidence.ptm == 1.0
    assert validate_prediction_confidence(confidence, 2) is confidence
    with pytest.raises(ValueError, match="must be finite"):
        PredictionConfidence(ptm=np.inf)
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        validate_prediction_confidence(confidence, 3)


def test_confidence_merge_enriches_identity_and_rejects_conflicts() -> None:
    current = PredictionConfidence(ptm=0.8, chain_ptm=np.array([0.7, 0.6]))
    incoming = PredictionConfidence(iptm=0.9, chain_ptm=np.array([0.7, 0.6]))

    merged = merge_prediction_confidence(current, incoming)

    assert merged is not current
    assert merged.ptm == 0.8
    assert merged.iptm == 0.9
    assert merged.chain_ptm is current.chain_ptm
    assert merge_prediction_confidence(current, None) is current
    with pytest.raises(ValueError, match="Conflicting model confidence field: ptm"):
        merge_prediction_confidence(
            current,
            PredictionConfidence(ptm=0.7),
            context="model confidence",
        )


def test_confidence_presentation_schemas_validate_typed_attributes() -> None:
    assert ConfidenceFieldSpec("ptm", "pTM").attribute == "ptm"
    assert (
        ConfidenceFieldSpec("probability", "Affinity", source="affinity").source
        == "affinity"
    )
    with pytest.raises(ValueError, match="Unknown confidence presentation field"):
        ConfidenceFieldSpec("unknown", "Unknown")
    with pytest.raises(ValueError, match="Unknown confidence section field"):
        ConfidenceSectionSpec("pair_chain_iptm", "Pair")
    with pytest.raises(ValueError, match="cannot also define data fields"):
        ConfidenceSummarySpec(
            fields=(ConfidenceFieldSpec("ptm", "pTM"),),
            informational_text="Info",
        )
