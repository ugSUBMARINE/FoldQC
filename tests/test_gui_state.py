from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.gui_state import (  # noqa: E402
    GuiState,
    GuiStateBacked,
    GuiWorkflowError,
    MetricContext,
    ResolvedTarget,
    WorkflowMessage,
)


def test_gui_state_uses_independent_mutable_defaults() -> None:
    first = GuiState()
    second = GuiState()

    first.accepted_token_overlap_warnings.add(("model.cif", "model"))

    assert second.accepted_token_overlap_warnings == set()
    assert first.pending_session_restore is not second.pending_session_restore


def test_metric_context_exposes_compute_arguments_as_mutable_lists() -> None:
    context = MetricContext(
        reference_selection="resname LIG",
        reference_indices=(0, 2),
        contact_indices=(1, 3),
        cutoff_angstrom=7.5,
    )

    kwargs = context.compute_kwargs()

    assert kwargs == {
        "ref_indices": [0, 2],
        "contact_indices": [1, 3],
        "cutoff": 7.5,
    }


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


def test_state_backed_compatibility_properties_share_one_state() -> None:
    class Host(GuiStateBacked):
        pass

    host = Host()
    values = np.array([0.8], dtype=np.float32)
    host._pred_data = values
    host._token_map_structure_path = Path("model.cif")

    assert host._get_gui_state().pred_data is values
    assert host._get_gui_state().token_map_structure_path == Path("model.cif")


def test_workflow_error_retains_typed_message() -> None:
    message = WorkflowMessage("warning", "Reference selection is empty.")
    error = GuiWorkflowError(message)

    assert str(error) == message.text
    assert error.message is message
