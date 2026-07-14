from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.gui_state import (  # noqa: E402
    GuiState,
    GuiStateBacked,
    MetricContext,
    ResolvedTarget,
)
from FoldQC.loader_models import PredictionData  # noqa: E402
from FoldQC.model_state import ModelState  # noqa: E402
from FoldQC.token_map import TokenMap  # noqa: E402


def test_gui_state_uses_independent_mutable_defaults() -> None:
    first = GuiState()
    second = GuiState()

    first.accepted_token_overlap_warnings.add(("model.cif", "model"))

    assert second.accepted_token_overlap_warnings == set()
    assert first.pending_session_restore is not second.pending_session_restore


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
    target = ResolvedTarget(
        kind="ensemble_group",
        label="foldqc_ensemble",
        obj_name="foldqc_model_0",
        data=None,
        token_map=[object()],
        members=[object(), object()],
    )

    assert target.kind == "ensemble_group"
    assert len(target.members) == 2


def test_state_backed_properties_share_one_state() -> None:
    class Host(GuiStateBacked):
        pass

    host = Host()
    host._state = GuiState()
    data = PredictionData(name="model", rank=2, structure_path=Path("model.cif"))
    token_map = TokenMap(())
    model_state = ModelState(rank=2, data=data, token_map=token_map)
    host._model_states = {2: model_state}
    host._active_model_rank = 2

    assert host._active_model_state is model_state
    assert host._pred_data is data
    assert host._token_map is token_map
    assert host._state.model_states == {2: model_state}
    assert host._state.active_model_rank == 2
