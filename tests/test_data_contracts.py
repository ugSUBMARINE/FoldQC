from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.data_contracts import normalize_and_validate_prediction_data
from FoldQC.loader_models import ModelFiles, PredictionData
from FoldQC.model_state import ModelState
from FoldQC.structure_index import StructureIndex
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap


def _data(**values) -> PredictionData:
    fields = {
        "name": "prediction",
        "rank": 0,
        "structure_path": Path("/tmp/model.cif"),
    }
    fields.update(values)
    return PredictionData(**fields)


def _index(size: int) -> StructureIndex:
    token_map = TokenMap(
        tuple(
            TokenInfo(index, "A", ResidueId(index + 1), "ALA", False, None)
            for index in range(size)
        )
    )
    return StructureIndex(
        Path("/tmp/model.cif"),
        "cif",
        token_map,
        size,
        tuple(range(size)),
        np.zeros(size, dtype=np.float32),
    )


def test_validation_normalizes_arrays_and_preserves_embedding_dtype() -> None:
    data = _data(
        token_plddt=np.array([90.0, 0.8, np.nan], dtype=np.float64),
        token_plddt_source="provider_token",
        pae=np.asfortranarray(np.ones((3, 3), dtype=np.float64)),
        pde=np.zeros((3, 3), dtype=np.int16),
        contact_probs=np.array(
            [[1.0, 0.2, np.nan], [0.2, 1.0, 0.3], [np.nan, 0.3, 1.0]]
        ),
        embeddings_s=np.ones((3, 4), dtype=np.int16),
        embeddings_z=np.ones((3, 3, 2), dtype=np.float64),
        confidence={},
        summary_confidence={},
        affinity={},
    )

    result = normalize_and_validate_prediction_data(data, 3)

    assert result is data
    np.testing.assert_allclose(data.token_plddt[:2], [0.9, 0.8])
    for field in ("token_plddt", "pae", "pde", "contact_probs"):
        array = getattr(data, field)
        assert array.dtype == np.float32
        assert array.flags.c_contiguous
        assert not array.flags.writeable
    assert data.embeddings_s.dtype == np.int16
    assert data.embeddings_z.dtype == np.float64
    assert not data.embeddings_s.flags.writeable
    assert not data.embeddings_z.flags.writeable


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"token_plddt": [0.8]}, "token_plddt_source"),
        (
            {"token_plddt": [0.8, 0.7], "token_plddt_source": "unknown"},
            "provenance",
        ),
        (
            {"token_plddt": [[0.8, 0.7]], "token_plddt_source": "provider_token"},
            "shape",
        ),
        ({"pae": [[1.0, 2.0]]}, "shape"),
        ({"pde": [[0.0, np.inf], [1.0, 0.0]]}, "infinity"),
        ({"contact_probs": [[1.0, 1.1], [0.2, 1.0]]}, "within 0–1"),
        ({"pae": [["x", "y"], ["z", "w"]]}, "numeric"),
        ({"embeddings_s": np.ones((2, 1))}, "provided together"),
        (
            {
                "embeddings_s": np.ones((1, 2)),
                "embeddings_z": np.ones((2, 2, 1)),
            },
            "embeddings_s shape",
        ),
        ({"confidence": []}, "dictionary or None"),
    ],
)
def test_validation_rejects_malformed_contracts(values, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_and_validate_prediction_data(_data(**values), 2)


@pytest.mark.parametrize("values", [[1.2, 0.8], [-0.1, 0.8], [200.0, 80.0]])
def test_validation_rejects_invalid_normalized_plddt(values) -> None:
    with pytest.raises(ValueError, match="normalize to 0–1"):
        normalize_and_validate_prediction_data(
            _data(token_plddt=values, token_plddt_source="provider_token"), 2
        )


def test_failed_lazy_merge_does_not_mutate_state_or_version() -> None:
    data = _data(token_plddt=[0.8, 0.7], token_plddt_source="provider_token")
    state = ModelState(0, data, _index(2))
    plddt_reference = state.data.token_plddt

    with pytest.raises(ValueError, match="pae must have shape"):
        state.merge_data(_data(pae=np.ones((1, 2))))

    assert state.data.token_plddt is plddt_reference
    assert state.data.pae is None
    assert state.version == 0


def test_model_capabilities_are_immutable_and_reject_unknown_values() -> None:
    model = ModelFiles(
        rank=0,
        structure_path=Path("model.cif"),
        display_label="model_0",
        object_name="model_0",
        capabilities={"plddt", "pae"},
    )

    assert model.capabilities == frozenset({"plddt", "pae"})
    with pytest.raises(AttributeError):
        model.capabilities.add("pde")  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="Unknown model data capabilities"):
        ModelFiles(
            rank=0,
            structure_path=Path("model.cif"),
            display_label="model_0",
            object_name="model_0",
            capabilities={"unknown"},  # type: ignore[arg-type]
        )
