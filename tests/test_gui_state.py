from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.gui_state import (  # noqa: E402
    MetricContext,
    PluginState,
    ResolvedTarget,
)
from FoldQC.loader_models import PredictionData  # noqa: E402
from FoldQC.model_state import ModelState  # noqa: E402
from FoldQC.providers.registry import BUILTIN_PROVIDERS  # noqa: E402
from FoldQC.structure_index import StructureIndex  # noqa: E402
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap  # noqa: E402


def _index(path: Path, token_map: TokenMap) -> StructureIndex:
    values = np.zeros(len(token_map), dtype=np.float32)
    values.setflags(write=False)
    return StructureIndex(
        path,
        "cif",
        token_map,
        len(token_map),
        tuple(range(len(token_map))),
        values,
    )


def test_gui_state_uses_independent_mutable_defaults() -> None:
    first = PluginState()
    second = PluginState()

    first.model_states[1] = object()

    assert second.model_states == {}


def test_metric_context_keeps_immutable_selection_provenance() -> None:
    context = MetricContext(
        reference_selection="resname LIG",
        reference_indices=(0, 2),
        contact_indices=(1, 3),
        cutoff_angstrom=7.5,
    )

    assert context.reference_indices == (0, 2)
    assert context.contact_indices == (1, 3)
    assert context.cutoff_angstrom == 7.5


def test_resolved_target_distinguishes_ensemble_group() -> None:
    token_map = TokenMap((TokenInfo(0, "A", ResidueId(1), "ALA", False, None),))
    states = tuple(
        ModelState(
            rank=rank,
            data=PredictionData(
                name="model",
                rank=rank,
                structure_path=Path(f"model_{rank}.cif"),
                provider=BUILTIN_PROVIDERS.get("boltz").info,
            ),
            structure_index=_index(Path(f"model_{rank}.cif"), token_map),
        )
        for rank in (0, 1)
    )
    target = ResolvedTarget(
        kind="ensemble_group",
        label="foldqc_ensemble",
        obj_name="foldqc_model_0",
        model_states=states,
        members=(object(), object()),
    )

    assert target.kind == "ensemble_group"
    assert len(target.members) == 2
    assert target.data is None
    assert target.token_map is token_map


def test_resolved_target_exposes_live_state_data_and_is_not_assignable() -> None:
    token_map = TokenMap((TokenInfo(0, "A", ResidueId(1), "ALA", False, None),))
    data = PredictionData(
        name="model",
        rank=0,
        structure_path=Path("model_0.cif"),
        provider=BUILTIN_PROVIDERS.get("boltz").info,
    )
    state = ModelState(0, data, _index(Path(data.structure_path), token_map))
    target = ResolvedTarget(
        kind="single",
        label="foldqc_model_0",
        obj_name="foldqc_model_0",
        model_states=(state,),
    )
    partial = PredictionData(
        name="model",
        rank=0,
        structure_path=Path("model_0.cif"),
        provider=BUILTIN_PROVIDERS.get("boltz").info,
        pae=np.ones((1, 1), dtype=np.float32),
    )

    state.merge_data(partial)

    assert target.data is data
    assert target.data.pae is partial.pae
    assert target.token_map is token_map
    with pytest.raises(FrozenInstanceError):
        target.data = partial
    with pytest.raises(FrozenInstanceError):
        target.token_map = TokenMap(())


def test_plugin_state_resolves_the_active_canonical_model() -> None:
    state = PluginState()
    data = PredictionData(
        name="model",
        rank=2,
        structure_path=Path("model.cif"),
        provider=BUILTIN_PROVIDERS.get("boltz").info,
    )
    token_map = TokenMap(())
    model_state = ModelState(
        rank=2,
        data=data,
        structure_index=_index(Path(data.structure_path), token_map),
    )
    state.model_states = {2: model_state}
    state.active_model_rank = 2

    assert state.active_model_state is model_state
    assert state.model_states == {2: model_state}
    assert state.active_model_rank == 2
