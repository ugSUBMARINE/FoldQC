import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.loader_models import PredictionData
from FoldQC.model_state import ModelState
from FoldQC.token_map import TokenMap


def _data(rank: int, name: str = "model") -> PredictionData:
    return PredictionData(
        name=name,
        rank=rank,
        structure_path=Path(f"/tmp/{name}_{rank}.cif"),
    )


def test_model_state_requires_matching_data_rank() -> None:
    with pytest.raises(ValueError, match="rank 2 cannot contain rank 1"):
        ModelState(rank=2, data=_data(1), token_map=TokenMap(()))


def test_replace_data_preserves_token_map() -> None:
    original = _data(2, "original")
    replacement = _data(2, "replacement")
    token_map = TokenMap(())
    state = ModelState(rank=2, data=original, token_map=token_map)

    state.replace_data(replacement)

    assert state.data is replacement
    assert state.token_map is token_map


def test_invalid_replacement_leaves_state_unchanged() -> None:
    original = _data(2, "original")
    token_map = TokenMap(())
    state = ModelState(rank=2, data=original, token_map=token_map)

    with pytest.raises(ValueError, match="rank 2 cannot contain rank 3"):
        state.replace(_data(3), TokenMap(()))

    assert state.data is original
    assert state.token_map is token_map


def test_replace_updates_data_and_token_map_together() -> None:
    state = ModelState(rank=2, data=_data(2, "original"), token_map=TokenMap(()))
    replacement = _data(2, "replacement")
    replacement_map = TokenMap(())

    state.replace(replacement, replacement_map)

    assert state.data is replacement
    assert state.token_map is replacement_map
