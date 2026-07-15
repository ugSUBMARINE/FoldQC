import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.confidence import PredictionConfidence
from FoldQC.loader_models import PredictionData
from FoldQC.model_state import ModelState
from FoldQC.providers.registry import BUILTIN_PROVIDERS
from FoldQC.structure_index import StructureIndex
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap


def _data(rank: int, **values) -> PredictionData:
    defaults = {
        "name": "prediction",
        "rank": rank,
        "structure_path": Path(f"/tmp/model_{rank}.cif"),
        "provider": BUILTIN_PROVIDERS.get("boltz").info,
        "display_label": f"model_{rank}",
    }
    defaults.update(values)
    return PredictionData(**defaults)


def _index(data: PredictionData, token_map: TokenMap | None = None) -> StructureIndex:
    if token_map is None:
        token_count = 0
        for field in ("token_plddt", "pae", "pde", "contact_probs"):
            value = getattr(data, field)
            if value is not None:
                token_count = np.asarray(value).shape[0]
                break
        token_map = TokenMap(
            tuple(
                TokenInfo(index, "A", ResidueId(index + 1), "ALA", False, None)
                for index in range(token_count)
            )
        )
    plddt = np.zeros(len(token_map), dtype=np.float32)
    plddt.setflags(write=False)
    return StructureIndex(
        Path(data.structure_path),
        "cif",
        token_map,
        len(token_map),
        tuple(range(len(token_map))),
        plddt,
    )


def test_model_state_requires_matching_data_rank() -> None:
    with pytest.raises(ValueError, match="rank 2 cannot contain rank 1"):
        data = _data(1)
        ModelState(rank=2, data=data, structure_index=_index(data))


def test_merge_is_in_place_monotonic_and_enriches_metadata() -> None:
    original_pde = np.array([[1.0]], dtype=np.float32)
    data = _data(
        2,
        pde=original_pde,
        confidence=PredictionConfidence(ptm=0.7, has_clash=True),
    )
    state = ModelState(rank=2, data=data, structure_index=_index(data))
    incoming = _data(
        2,
        token_plddt=np.array([0.8], dtype=np.float32),
        token_plddt_source="provider_token",
        pae=np.array([[2.0]], dtype=np.float32),
        pde=np.array([[9.0]], dtype=np.float32),
        confidence=PredictionConfidence(iptm=0.9),
    )

    changed = state.merge_data(incoming)

    assert changed is True
    assert state.data is data
    assert state.data.pde is not original_pde
    np.testing.assert_array_equal(state.data.pde, original_pde)
    assert state.data.pae is incoming.pae
    assert state.data.token_plddt is incoming.token_plddt
    assert state.data.token_plddt_source == "provider_token"
    assert state.data.confidence.ptm == 0.7
    assert state.data.confidence.iptm == 0.9
    assert state.data.confidence.has_clash is True
    assert state.version == 1


@pytest.mark.parametrize(
    "incoming",
    [
        _data(3),
        _data(2, provider=BUILTIN_PROVIDERS.get("alphafold3").info),
        _data(2, structure_path=Path("/tmp/other.cif")),
        _data(2, name="other"),
    ],
)
def test_invalid_merge_leaves_state_unchanged(incoming) -> None:
    data = _data(2)
    state = ModelState(rank=2, data=data, structure_index=_index(data))

    with pytest.raises(ValueError):
        state.merge_data(incoming)

    assert state.data is data
    assert state.version == 0


def test_merge_rejects_uncoupled_plddt_and_embedding_fields() -> None:
    data = _data(2)
    state = ModelState(rank=2, data=data, structure_index=_index(data))

    with pytest.raises(ValueError, match="token_plddt_source"):
        state.merge_data(_data(2, token_plddt=np.array([0.8])))
    with pytest.raises(ValueError, match="must be provided together"):
        state.merge_data(_data(2, embeddings_s=np.ones((1, 1))))

    assert state.version == 0


def test_conflicting_confidence_stages_array_additions_transactionally() -> None:
    data = _data(2, confidence=PredictionConfidence(ptm=0.8))
    token_map = TokenMap((TokenInfo(0, "A", ResidueId(1), "ALA", False, None),))
    state = ModelState(rank=2, data=data, structure_index=_index(data, token_map))
    original_confidence = state.data.confidence
    incoming = _data(
        2,
        pae=np.zeros((1, 1), dtype=np.float32),
        confidence=PredictionConfidence(ptm=0.7),
    )

    with pytest.raises(ValueError, match="Conflicting model_2 confidence field: ptm"):
        state.merge_data(incoming)

    assert state.data.pae is None
    assert state.data.confidence is original_confidence
    assert state.version == 0


def test_snapshot_restores_fields_version_and_preserves_index_in_place() -> None:
    data = _data(2)
    token_map = TokenMap((TokenInfo(0, "A", ResidueId(1), "ALA", False, None),))
    structure_index = _index(data, token_map)
    state = ModelState(rank=2, data=data, structure_index=structure_index)
    snapshot = state.snapshot()

    state.merge_data(_data(2, pae=np.ones((1, 1), dtype=np.float32)))
    state.restore(snapshot)

    assert state.data is data
    assert state.data.pae is None
    assert state.token_map is token_map
    assert state.structure_index is structure_index
    assert state.version == 0


def test_structure_index_validation_requires_canonical_identity() -> None:
    token_map = TokenMap(())
    data = _data(2)
    structure_index = _index(data, token_map)
    state = ModelState(rank=2, data=data, structure_index=structure_index)

    state.validate_structure_index(structure_index)
    with pytest.raises(ValueError, match="different StructureIndex"):
        state.validate_structure_index(_index(data, token_map))

    assert state.token_map is token_map
