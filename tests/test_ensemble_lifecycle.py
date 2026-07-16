from __future__ import annotations

import types
from pathlib import Path

import numpy as np
from FoldQC.ensemble import PreparedEnsemble, PreparedEnsembleMember
from FoldQC.ensemble_lifecycle import (
    ALIGNMENT_CORE_SELECTION_NAME,
    EnsembleLifecycleService,
)
from FoldQC.gui_operations import GuiOperationCoordinator
from FoldQC.gui_services import ContextSelection, ObjectTokenInspection
from FoldQC.gui_state import PluginState
from FoldQC.loader_models import (
    ModelFiles,
    PredictionData,
    PredictionFiles,
    ProviderInfo,
)
from FoldQC.model_state import ModelState
from FoldQC.structure_index import StructureIndex
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap


class _Presenter:
    def __init__(self, choice: str) -> None:
        self.choice = choice
        self.notices = []

    def choose(self, _request):
        return self.choice

    def start_progress(self, _request, _on_cancel=None) -> None:
        pass

    def update_progress(self, _operation_id: str, _label: str) -> None:
        pass

    def finish_progress(self, _operation_id: str) -> None:
        pass

    def present_notice(self, notice) -> None:
        self.notices.append(notice)


class _View:
    def set_busy(self, _state) -> None:
        pass


class _Handle:
    def abandon(self) -> None:
        pass


class _Runner:
    def submit(self, request_id, _task, _progress, result, _error):
        self.request_id = request_id
        self.result = result
        return _Handle()

    def deliver(self, value: object) -> None:
        self.result(self.request_id, value)

    def dispose(self, _value: object) -> None:
        pass


class _Scheduler:
    def call_soon(self, callback) -> None:
        callback()


class _Context:
    def __init__(self) -> None:
        self.selection = ContextSelection(metric_key="plddt")

    def set_selection(self, selection: ContextSelection) -> None:
        self.selection = selection

    def refresh_objects(self, _preferred_target=None) -> None:
        pass


class _Prediction:
    def __init__(self, state: PluginState) -> None:
        self.state = state

    def capture_model_store(self):
        return None

    def commit_model_state(self, model_state: ModelState, *, activate: bool):
        assert not activate
        self.state.model_states[model_state.rank] = model_state
        return model_state

    def restore_model_store(self, _snapshot) -> None:
        pass


class _Viewer:
    def __init__(self, coordinates: dict[str, np.ndarray]) -> None:
        self.coordinates = coordinates
        self.selections = []
        self.mappings = {}
        self.transforms = []

    def name_exists(self, name: str) -> bool:
        return name in self.coordinates

    def group_members(self, _group_name: str):
        return ()

    def capture_paint_mappings(self):
        return dict(self.mappings)

    def restore_paint_mappings(self, mappings) -> None:
        self.mappings = dict(mappings)

    def load_structure_object_if_missing(self, _path, _obj_name: str) -> bool:
        return False

    def inspect_tokens(self, obj_name: str, _token_map: TokenMap):
        mapping = types.SimpleNamespace(obj_name=obj_name)
        return ObjectTokenInspection(mapping, self.coordinates[obj_name])

    def transform(self, obj_name: str, rotation, translation) -> None:
        self.transforms.append((obj_name, rotation, translation))

    def run_suspended(self, operation):
        return operation()

    def add_to_group(self, _group_name: str, _names) -> None:
        pass

    def update_token_selection(
        self, selection_name: str, token_indices, object_token_maps
    ) -> None:
        self.selections.append(
            (selection_name, tuple(token_indices), tuple(object_token_maps))
        )


def _prepared_ensemble(*, skip_alignment: bool):
    provider = ProviderInfo("test", "Test")
    models = [
        ModelFiles(
            rank,
            Path(f"/tmp/model_{rank}.cif"),
            f"model {rank}",
            f"model_{rank}",
            capabilities=frozenset({"plddt"}),
        )
        for rank in (0, 1)
    ]
    files = PredictionFiles("prediction", Path("/tmp"), provider, models=models)
    token_map = TokenMap(
        tuple(
            TokenInfo(index, "A", ResidueId(index + 1), "ALA", False, None)
            for index in range(4)
        )
    )
    members = []
    for model in models:
        plddt = np.array([0.95, 0.9, 0.85, 0.5], dtype=np.float32)
        data = PredictionData(
            "prediction",
            model.rank,
            model.structure_path,
            provider,
            token_plddt=plddt,
            token_plddt_source="provider_token",
        )
        index = StructureIndex(
            model.structure_path,
            "cif",
            token_map,
            len(token_map),
            tuple(range(len(token_map))),
            plddt,
        )
        members.append(
            PreparedEnsembleMember(
                model.display_label,
                model.object_name,
                ModelState(model.rank, data, index),
            )
        )
    prepared = PreparedEnsemble(
        files,
        "prediction_ensemble",
        tuple(members),
        skip_alignment,
        0,
        () if skip_alignment else (0, 1, 2),
        np.array([0.95, 0.9, 0.85, 0.5], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    state = PluginState(files, {0: members[0].model_state}, 0)
    return state, prepared


def _activate(prepared: PreparedEnsemble, state: PluginState, choice: str):
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
        dtype=np.float32,
    )
    viewer = _Viewer(
        {
            "model_0": reference,
            "model_1": reference + np.array([3.0, -2.0, 1.0], dtype=np.float32),
        }
    )
    presenter = _Presenter(choice)
    runner = _Runner()
    operations = GuiOperationCoordinator(presenter, _View())
    service = EnsembleLifecycleService(
        state,
        viewer,
        presenter,
        _Scheduler(),
        runner,
        operations,
        _Prediction(state),
        _Context(),
    )
    service.activate()
    runner.deliver(prepared)
    return viewer, presenter


def test_automatic_alignment_exposes_core_on_rank_zero_reference_object() -> None:
    state, prepared = _prepared_ensemble(skip_alignment=False)

    viewer, presenter = _activate(prepared, state, "align")

    name, indices, object_token_maps = viewer.selections[-1]
    assert name == ALIGNMENT_CORE_SELECTION_NAME == "foldqc_alignment_core"
    assert indices == (0, 1, 2)
    assert len(object_token_maps) == 1
    assert object_token_maps[0][0] == "model_0"
    assert object_token_maps[0][1] is prepared.members[0].token_map
    assert "foldqc_alignment_core" in presenter.notices[-1].message


def test_current_coordinates_clear_the_alignment_core_selection() -> None:
    state, prepared = _prepared_ensemble(skip_alignment=True)

    viewer, presenter = _activate(prepared, state, "current")

    assert viewer.selections[-1][0] == ALIGNMENT_CORE_SELECTION_NAME
    assert viewer.selections[-1][1] == ()
    assert "foldqc_alignment_core" not in presenter.notices[-1].message
