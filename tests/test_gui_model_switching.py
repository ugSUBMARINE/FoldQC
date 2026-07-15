"""Phase 5 service-composition tests.

The former version of this module routed private dialog methods dynamically
through a ``__new__`` harness.  These tests exercise the public typed service
operations and fake ports instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from FoldQC import ensemble
from FoldQC.analysis import (
    AnalysisRequest,
    AnalysisResolver,
    ColorOptions,
    ComputedMetric,
    DeferredAnalysisAction,
    build_data_load_plan,
)
from FoldQC.analysis_coordinator import AnalysisCoordinator
from FoldQC.context_service import ContextService
from FoldQC.data_acquisition import DataAcquisitionService
from FoldQC.gui_application import GuiApplicationServices
from FoldQC.gui_operations import GuiOperationCoordinator
from FoldQC.gui_services import (
    BusyViewState,
    ContextSelection,
    ContextViewState,
    DataAcquisitionOutcome,
    LifecycleUiUpdate,
    TargetChoice,
)
from FoldQC.gui_state import PluginState
from FoldQC.lifecycle_support import DataLoadBatchResult, InitialLoadResult
from FoldQC.loader_models import (
    ModelFiles,
    PredictionData,
    PredictionFiles,
    ProviderInfo,
)
from FoldQC.model_state import ModelState
from FoldQC.prediction_lifecycle import PredictionLifecycleService
from FoldQC.presentation import Notice
from FoldQC.structure_index import StructureIndex
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap


class FakePresenter:
    def __init__(self) -> None:
        self.notices: list[Notice] = []
        self.progress: list[tuple[str, str]] = []
        self.finished: list[str] = []
        self.plots: list[object] = []
        self.cancel_callbacks: list[object] = []
        self.closed = False

    def present_notice(self, notice: Notice) -> None:
        self.notices.append(notice)

    def choose(self, _request):
        return "yes"

    def select_item(self, request):
        return request.default_key

    def start_progress(self, request, on_cancel=None) -> None:
        self.progress.append((request.operation_id, request.label))
        self.cancel_callbacks.append(on_cancel)

    def update_progress(self, operation_id: str, label: str) -> None:
        self.progress.append((operation_id, label))

    def finish_progress(self, operation_id: str) -> None:
        self.finished.append(operation_id)

    def set_window_title(self, _title: str) -> None:
        pass

    def show_statistics(self, _text: str) -> None:
        pass

    def show_plot(self, prepared) -> None:
        self.plots.append(prepared)

    def close(self) -> None:
        self.closed = True


class FakeView:
    def __init__(self) -> None:
        self.busy: list[BusyViewState] = []
        self.context: list[ContextViewState] = []
        self.lifecycle: list[LifecycleUiUpdate] = []
        self.closed = False

    def apply_context(self, state: ContextViewState) -> None:
        self.context.append(state)

    def apply_lifecycle(self, update: LifecycleUiUpdate) -> None:
        self.lifecycle.append(update)

    def set_busy(self, state: BusyViewState) -> None:
        self.busy.append(state)

    def close(self) -> None:
        self.closed = True


class FakeHandle:
    def __init__(self) -> None:
        self.abandoned = False

    def abandon(self) -> None:
        self.abandoned = True


class DeferredRunner:
    def __init__(self) -> None:
        self.handle = FakeHandle()
        self.request_id = 0
        self.task = None
        self.on_progress = None
        self.on_result = None
        self.on_error = None
        self.disposed: list[object] = []

    def submit(self, request_id, task, on_progress, on_result, on_error) -> FakeHandle:
        self.handle = FakeHandle()
        self.request_id = request_id
        self.task = task
        self.on_progress = on_progress
        self.on_result = on_result
        self.on_error = on_error
        return self.handle

    def dispose(self, value: object) -> None:
        self.disposed.append(value)
        close = getattr(value, "close", None)
        if close is not None:
            close()

    def deliver(self, value: object) -> None:
        assert self.on_result is not None
        self.on_result(self.request_id, value)


class ImmediateScheduler:
    def call_soon(self, callback) -> None:
        callback()

    def call_later(self, _delay_ms: int, callback) -> None:
        callback()


@dataclass
class OutcomeObserver:
    outcomes: list[DataAcquisitionOutcome]

    def data_acquisition_finished(self, outcome: DataAcquisitionOutcome) -> None:
        self.outcomes.append(outcome)


def _token_map() -> TokenMap:
    return TokenMap(
        (
            TokenInfo(0, "A", ResidueId(1), "ALA", False, None),
            TokenInfo(1, "A", ResidueId(2), "GLY", False, None),
        )
    )


def _state(*, with_pae: bool = False) -> tuple[PluginState, PredictionFiles]:
    provider = ProviderInfo("test", "Test")
    model = ModelFiles(
        0,
        Path("/tmp/model_0.cif"),
        "model 0",
        "model_0",
        capabilities=frozenset({"plddt", "pae"}),
    )
    files = PredictionFiles("prediction", Path("/tmp"), provider, models=[model])
    plddt = np.array([0.8, 0.9], dtype=np.float32)
    pae = np.ones((2, 2), dtype=np.float32) if with_pae else None
    data = PredictionData(
        "prediction",
        0,
        model.structure_path,
        provider,
        display_label=model.display_label,
        token_plddt=plddt,
        token_plddt_source="provider_token",
        pae=pae,
    )
    token_map = _token_map()
    structure_values = np.array(plddt, copy=True)
    structure_values.setflags(write=False)
    index = StructureIndex(
        model.structure_path,
        "cif",
        token_map,
        2,
        (0, 1),
        structure_values,
    )
    model_state = ModelState(0, data, index)
    return PluginState(files, {0: model_state}, 0, None), files


def _lazy_action(revision: int = 0) -> DeferredAnalysisAction:
    return DeferredAnalysisAction(
        AnalysisRequest("color", "model_0", "pae_row_mean", ui_revision=revision),
        ColorOptions("viridis"),
    )


def test_operation_coordinator_owns_exclusivity_and_abandonment() -> None:
    presenter = FakePresenter()
    view = FakeView()
    coordinator = GuiOperationCoordinator(presenter, view)
    first = coordinator.begin("data", title="Loading", label="PAE")
    assert first is not None
    assert coordinator.begin("prediction", title="Loading", label="Other") is None
    handle = FakeHandle()
    assert coordinator.attach(first, handle)
    coordinator.abandon()
    assert handle.abandoned
    assert not coordinator.is_busy
    assert view.busy == [BusyViewState(True, False), BusyViewState(False, True)]


def test_data_acquisition_commits_atomically_and_reports_typed_outcome() -> None:
    state, files = _state()
    presenter = FakePresenter()
    view = FakeView()
    runner = DeferredRunner()
    operations = GuiOperationCoordinator(presenter, view)
    service = DataAcquisitionService(
        state, presenter, ImmediateScheduler(), runner, operations
    )
    action = _lazy_action()
    resolved = AnalysisResolver().resolve(action.request, state)
    plan = build_data_load_plan(resolved)
    observer = OutcomeObserver([])
    assert service.acquire(action, plan, observer)
    requirement = plan.requirements[0]
    partial = PredictionData(
        "prediction",
        0,
        Path("/tmp/model_0.cif"),
        files.provider,
        pae=np.full((2, 2), 2.0, dtype=np.float32),
    )
    runner.deliver(DataLoadBatchResult(files, ((requirement, partial),)))
    assert observer.outcomes[0].status == "ready"
    assert np.array_equal(state.model_states[0].data.pae, partial.pae)
    assert state.model_states[0].version == 1


def test_data_acquisition_rejects_invalid_merge_without_mutation() -> None:
    state, files = _state()
    presenter = FakePresenter()
    view = FakeView()
    runner = DeferredRunner()
    service = DataAcquisitionService(
        state,
        presenter,
        ImmediateScheduler(),
        runner,
        GuiOperationCoordinator(presenter, view),
    )
    action = _lazy_action()
    plan = build_data_load_plan(AnalysisResolver().resolve(action.request, state))
    observer = OutcomeObserver([])
    original = state.model_states[0].data
    original_version = state.model_states[0].version
    assert service.acquire(action, plan, observer)
    invalid = PredictionData(
        "prediction",
        0,
        Path("/tmp/model_0.cif"),
        files.provider,
        pae=np.ones((3, 3), dtype=np.float32),
    )
    runner.deliver(DataLoadBatchResult(files, ((plan.requirements[0], invalid),)))
    assert observer.outcomes[0].status == "failed"
    assert state.model_states[0].data is original
    assert state.model_states[0].version == original_version


def test_data_acquisition_cancellation_is_typed_and_silent() -> None:
    state, _files = _state()
    presenter = FakePresenter()
    view = FakeView()
    runner = DeferredRunner()
    service = DataAcquisitionService(
        state,
        presenter,
        ImmediateScheduler(),
        runner,
        GuiOperationCoordinator(presenter, view),
    )
    action = _lazy_action()
    plan = build_data_load_plan(AnalysisResolver().resolve(action.request, state))
    observer = OutcomeObserver([])
    assert service.acquire(action, plan, observer)
    callback = presenter.cancel_callbacks[-1]
    assert callable(callback)
    callback()
    assert runner.handle.abandoned
    assert observer.outcomes[0].status == "cancelled"
    assert presenter.notices == []


def test_data_acquisition_silently_marks_replaced_prediction_stale() -> None:
    state, files = _state()
    presenter = FakePresenter()
    view = FakeView()
    runner = DeferredRunner()
    service = DataAcquisitionService(
        state,
        presenter,
        ImmediateScheduler(),
        runner,
        GuiOperationCoordinator(presenter, view),
    )
    action = _lazy_action()
    plan = build_data_load_plan(AnalysisResolver().resolve(action.request, state))
    observer = OutcomeObserver([])
    assert service.acquire(action, plan, observer)
    state.pred_files = None
    runner.deliver(DataLoadBatchResult(files, ()))
    assert observer.outcomes[0].status == "stale"
    assert presenter.notices == []


class FakeDependencies:
    def __init__(self) -> None:
        self.closed = False

    def ensure(self, _keys, *, feature_label: str) -> bool:
        return bool(feature_label)

    def close(self) -> None:
        self.closed = True


class FakeSession:
    restoring = False

    def restore(self):
        raise AssertionError("composition close attempted session restoration")

    def save(self) -> None:
        pass


class LifecycleContext:
    def __init__(self) -> None:
        self.selection = ContextSelection(metric_key="plddt")

    def set_selection(self, selection: ContextSelection) -> None:
        self.selection = selection

    def refresh_objects(self, preferred_target: str | None = None) -> ContextViewState:
        target = preferred_target or "model_0"
        return ContextViewState(
            target_choices=(TargetChoice(target, "single"),),
            selected_target=target,
        )


class LifecycleViewer:
    def __init__(self, *, created: bool) -> None:
        self.created = created
        self.cleared = False

    def ensure_structure_object(self, _path, _name: str, *, zoom: bool = True) -> bool:
        return self.created

    def capture_paint_mappings(self):
        return {}

    def restore_paint_mappings(self, _mappings) -> None:
        pass

    def clear_paint_mappings(self) -> None:
        self.cleared = True

    def delete_names(self, _names) -> None:
        pass


class TargetListViewer:
    def object_names(self, additional_names=()):
        return ["model_0", *additional_names]


class RecordingAnalysisSubmission:
    def __init__(self, operations: GuiOperationCoordinator) -> None:
        self.operations = operations
        self.ui_revision = 4
        self.actions: list[DeferredAnalysisAction] = []

    def submit(self, action: DeferredAnalysisAction) -> None:
        assert not self.operations.is_busy
        self.actions.append(action)


class CountingMetricService:
    def __init__(self) -> None:
        self.context_calls = 0
        self.compute_calls = 0

    def resolve_contexts(self, resolved):
        self.context_calls += 1
        return resolved

    def compute(self, resolved):
        self.compute_calls += 1
        member = resolved.members[0]
        return (
            ComputedMetric(
                member.rank,
                member.label,
                member.obj_name,
                member.model_state,
                member.metric_context,
                member.model_state.data.token_plddt,
            ),
        )


class RecordingColoring:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _resolved, _computed, _options) -> None:
        self.calls += 1


class NoLoadData:
    def acquire(self, _action, _plan, _observer) -> bool:
        raise AssertionError("fully loaded analysis attempted lazy acquisition")


class NoopPlots:
    def prepare(self, _resolved, _computed, _options):
        raise AssertionError("color action reached plots")


class NoopExport:
    def execute(self, _resolved, _computed, _options) -> None:
        raise AssertionError("color action reached export")


class NoopContext:
    def refresh(self) -> None:
        pass


def test_analysis_coordinator_computes_each_action_once() -> None:
    state, _files = _state(with_pae=True)
    presenter = FakePresenter()
    metrics = CountingMetricService()
    coloring = RecordingColoring()
    coordinator = AnalysisCoordinator(
        state,
        presenter,
        FakeDependencies(),
        NoLoadData(),
        metrics,
        coloring,
        NoopPlots(),
        NoopExport(),
        NoopContext(),
        set(),
    )
    coordinator.submit(_lazy_action())
    assert metrics.context_calls == 1
    assert metrics.compute_calls == 1
    assert coloring.calls == 1
    assert presenter.notices == []


def test_analysis_coordinator_discards_stale_captured_ui_revision() -> None:
    state, _files = _state(with_pae=True)
    presenter = FakePresenter()
    metrics = CountingMetricService()
    coloring = RecordingColoring()
    coordinator = AnalysisCoordinator(
        state,
        presenter,
        FakeDependencies(),
        NoLoadData(),
        metrics,
        coloring,
        NoopPlots(),
        NoopExport(),
        NoopContext(),
        set(),
    )
    coordinator.invalidate_ui()
    coordinator.submit(_lazy_action(revision=0))
    assert metrics.compute_calls == 0
    assert coloring.calls == 0
    assert presenter.notices == []


def test_typed_lifecycle_and_context_updates_are_immutable_values() -> None:
    lifecycle = LifecycleUiUpdate(
        selected_rank=2,
        selected_target="model_2",
        display_path="/tmp/prediction",
    )
    context = ContextViewState(selected_rank=2, selected_target="model_2")
    assert lifecycle.selected_rank == context.selected_rank
    assert lifecycle.display_path == "/tmp/prediction"


def test_application_close_releases_adapters_and_prediction_owner_in_order() -> None:
    state, files = _state()
    events: list[str] = []
    presenter = FakePresenter()
    view = FakeView()
    dependencies = FakeDependencies()
    runner = DeferredRunner()
    presenter.close = lambda: events.append("presenter")
    view.close = lambda: events.append("view")
    dependencies.close = lambda: events.append("dependencies")
    runner.dispose = lambda value: events.append(
        "prediction" if value is files else "other"
    )
    services = GuiApplicationServices(
        state=state,
        viewer=object(),
        presenter=presenter,
        view=view,
        scheduler=ImmediateScheduler(),
        job_runner=runner,
        session=FakeSession(),
        dependencies=dependencies,
        metric_rows={},
    )
    services.close()
    assert events == ["dependencies", "prediction", "presenter", "view"]
    assert state.pred_files is None


def test_initial_prediction_colors_only_a_newly_created_viewer_object() -> None:
    for created in (True, False):
        loaded, files = _state()
        state = PluginState()
        presenter = FakePresenter()
        view = FakeView()
        runner = DeferredRunner()
        operations = GuiOperationCoordinator(presenter, view)
        analysis = RecordingAnalysisSubmission(operations)
        service = PredictionLifecycleService(
            state,
            LifecycleViewer(created=created),
            presenter,
            view,
            LifecycleContext(),
            operations,
            runner,
            FakeSession(),
            analysis,
        )
        lease = operations.begin("prediction", title="Loading", label="Prediction")
        assert lease is not None
        result = InitialLoadResult(
            files,
            loaded.model_states[0],
            Path("/tmp/prediction"),
        )

        service._on_initial_result(lease.request_id, result)

        if created:
            assert len(analysis.actions) == 1
            action = analysis.actions[0]
            assert action.request.action == "color"
            assert action.request.target_name == "model_0"
            assert action.request.metric_key == "plddt_class"
            assert action.request.ui_revision == analysis.ui_revision
        else:
            assert analysis.actions == []


def test_context_reports_ensemble_button_availability_and_tooltips() -> None:
    presenter = FakePresenter()
    view = FakeView()

    empty_context = ContextService(
        PluginState(), TargetListViewer(), presenter, view, {}
    )
    empty = empty_context.derive_view_state(target_names=[])
    assert not empty.ensemble_enabled
    assert empty.ensemble_tooltip == "Load a prediction with at least two models first."

    state, files = _state()
    context = ContextService(state, TargetListViewer(), presenter, view, {})
    single = context.derive_view_state(target_names=["model_0"])
    assert not single.ensemble_enabled
    assert single.ensemble_tooltip == "Ensemble mode requires at least two model files."

    files.models.append(
        ModelFiles(
            1,
            Path("/tmp/model_1.cif"),
            "model 1",
            "model_1",
            capabilities=frozenset({"plddt"}),
        )
    )
    available = context.derive_view_state(target_names=["model_0"])
    assert available.ensemble_enabled
    assert available.ensemble_tooltip.startswith("Load all ranked models")

    state.ensemble = ensemble.EnsembleState(
        "prediction_ensemble",
        (
            ensemble.EnsembleMember(0, "model_0"),
            ensemble.EnsembleMember(1, "model_1"),
        ),
        False,
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    loaded = context.derive_view_state(target_names=["model_0"])
    assert not loaded.ensemble_enabled
    assert loaded.ensemble_tooltip == (
        "The ensemble for this prediction is already loaded."
    )
