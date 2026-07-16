"""Prediction discovery, replacement, and ranked-model switching."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from . import metrics
from .analysis import AnalysisRequest, ColorOptions, DeferredAnalysisAction
from .context_service import ContextService
from .ensemble import EnsembleState
from .gui_services import (
    AnalysisSubmissionPort,
    ContextSelection,
    ContextViewState,
    DialogViewPort,
    JobRunner,
    LifecycleUiUpdate,
    ModelChoice,
    ObjectPaintMapping,
    OperationCoordinatorPort,
    OperationLease,
    SessionPort,
    ViewerPort,
)
from .gui_state import PluginState
from .lifecycle_support import (
    APP_TITLE,
    InitialLoadResult,
    ModelState,
    ModelStoreSnapshot,
    ModelSwitchResult,
    _discover_prediction,
    _discovery_phase,
    _load_rank_data,
    _scan_and_load_initial_prediction,
)
from .loader_models import PredictionDiscovery, PredictionFiles
from .presentation import (
    ChoiceOption,
    ChoiceRequest,
    Notice,
    PresentationPort,
    SelectionItem,
    SelectionRequest,
)
from .session import normalize_prediction_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PredictionReplacementSnapshot:
    pred_files: PredictionFiles | None
    model_store: ModelStoreSnapshot
    ensemble: EnsembleState | None
    paint_mappings: dict[tuple[str, str], ObjectPaintMapping]
    context_selection: ContextSelection
    context_state: ContextViewState
    display_path: str


class PredictionLifecycleService:
    """Own provisional archives and commit prediction/model replacements."""

    def __init__(
        self,
        state: PluginState,
        viewer: ViewerPort,
        presenter: PresentationPort,
        view: DialogViewPort,
        context: ContextService,
        operations: OperationCoordinatorPort,
        job_runner: JobRunner,
        session: SessionPort,
        analysis: AnalysisSubmissionPort,
    ) -> None:
        self._state = state
        self._viewer = viewer
        self._presenter = presenter
        self._view = view
        self._context = context
        self._operations = operations
        self._job_runner = job_runner
        self._session = session
        self._analysis = analysis
        self._preferred_rank: int | None = None
        self._preferred_target: str | None = None
        self._discovery: PredictionDiscovery | None = None
        self._display_path = ""

    @property
    def is_loading(self) -> bool:
        active = self._operations.active
        return active is not None and active.kind in {"prediction", "model_switch"}

    def capture_model_store(self) -> ModelStoreSnapshot:
        return ModelStoreSnapshot(
            self._state.active_model_rank,
            tuple(
                (rank, model_state, model_state.snapshot())
                for rank, model_state in self._state.model_states.items()
            ),
        )

    def restore_model_store(self, snapshot: ModelStoreSnapshot) -> None:
        restored: dict[int, ModelState] = {}
        for rank, model_state, state_snapshot in snapshot.entries:
            model_state.restore(state_snapshot)
            restored[rank] = model_state
        self._state.model_states = restored
        self._state.active_model_rank = snapshot.active_rank

    def commit_model_state(
        self,
        incoming: ModelState,
        *,
        reset_store: bool = False,
        activate: bool = True,
    ) -> ModelState:
        if reset_store:
            canonical = incoming
            self._state.model_states = {incoming.rank: canonical}
        else:
            existing = self._state.model_states.get(incoming.rank)
            if existing is None:
                canonical = incoming
                self._state.model_states = {
                    **self._state.model_states,
                    incoming.rank: canonical,
                }
            else:
                canonical = existing
            if canonical is not incoming:
                canonical.validate_structure_index(incoming.structure_index)
                canonical.merge_data(incoming.data)
        if activate:
            self._state.active_model_rank = incoming.rank
        return canonical

    def load_prediction(
        self,
        path: str,
        *,
        preferred_rank: int | None = None,
        preferred_target: str | None = None,
    ) -> None:
        self._start_prediction_load(
            path,
            from_history=False,
            preferred_rank=preferred_rank,
            preferred_target=preferred_target,
        )

    def load_recent_prediction(self, path: str) -> None:
        """Load one persisted MRU path, offering removal when it is stale."""
        normalized = normalize_prediction_path(path)
        if (
            self._state.pred_files is not None
            and normalized
            and os.path.normcase(normalized) == os.path.normcase(self._display_path)
        ):
            self._restore_display_path()
            return
        self._start_prediction_load(normalized, from_history=True)

    def _start_prediction_load(
        self,
        path: str,
        *,
        from_history: bool,
        preferred_rank: int | None = None,
        preferred_target: str | None = None,
    ) -> None:
        path = normalize_prediction_path(path)
        if not path:
            self._restore_display_path()
            return
        if not Path(path).exists():
            self._restore_display_path()
            if from_history:
                self._handle_missing_recent_prediction(path)
            else:
                self._presenter.present_notice(
                    Notice(
                        "prediction_path_missing",
                        f"The prediction path does not exist:\n{path}",
                        severity="error",
                        title=f"{APP_TITLE} - error",
                    )
                )
            return
        lease = self._operations.begin(
            "prediction",
            title=f"{APP_TITLE} – Loading",
            label=_discovery_phase(path),
            cancellable=True,
            on_cancel=self._cancel_pending_load,
        )
        if lease is None:
            self._restore_display_path()
            return
        self._preferred_rank = preferred_rank
        self._preferred_target = preferred_target
        handle = self._job_runner.submit(
            lease.request_id,
            lambda report: _discover_prediction(path, report),
            self._on_progress,
            self._on_discovery,
            self._on_error,
        )
        self._operations.attach(lease, handle)

    def _handle_missing_recent_prediction(self, path: str) -> None:
        selected = self._presenter.choose(
            ChoiceRequest(
                "missing_recent_prediction",
                f"{APP_TITLE} - missing recent prediction",
                "This recent prediction no longer exists:\n\n"
                f"{path}\n\nKeep it in the recent-prediction list?",
                (
                    ChoiceOption("keep", "Keep", role="reject"),
                    ChoiceOption(
                        "remove",
                        "Remove from history",
                        role="destructive",
                    ),
                ),
                default_key="keep",
            )
        )
        if selected != "remove":
            return
        try:
            recent = self._session.remove_recent_prediction(path)
            self._view.apply_lifecycle(LifecycleUiUpdate(recent_predictions=recent))
        except Exception as exc:
            logger.exception("Could not remove a recent prediction")
            self._presenter.present_notice(
                Notice(
                    "prediction_history_save_failed",
                    f"Could not update the recent-prediction list: {exc}",
                    severity="warning",
                )
            )

    def _restore_display_path(self) -> None:
        self._view.apply_lifecycle(LifecycleUiUpdate(display_path=self._display_path))

    def _cancel_pending_load(self) -> None:
        self._preferred_rank = None
        self._preferred_target = None
        self._discovery = None
        self._restore_display_path()

    def _on_progress(self, request_id: int, label: str) -> None:
        lease = self._operations.active
        if lease is not None and lease.request_id == request_id:
            self._operations.update(lease, label)

    def _on_discovery(self, request_id: int, discovery: object) -> None:
        lease = self._operations.active
        if lease is None or lease.request_id != request_id:
            self._job_runner.dispose(discovery)
            return
        if not isinstance(discovery, PredictionDiscovery):
            self._job_runner.dispose(discovery)
            self._fail(lease, "The discovery worker returned an unexpected result.")
            return
        candidates = discovery.candidates
        if not candidates:
            self._job_runner.dispose(discovery)
            self._fail(lease, "No supported prediction outputs were found.")
            return
        candidate = candidates[0]
        if len(candidates) > 1:
            selected = self._presenter.select_item(
                SelectionRequest(
                    "prediction_candidate",
                    "Select prediction",
                    "Prediction directory:",
                    tuple(
                        SelectionItem(
                            str(index),
                            item.relative_path,
                            item.provider_label,
                        )
                        for index, item in enumerate(candidates)
                    ),
                    default_key="0",
                )
            )
            if selected is None:
                self._job_runner.dispose(discovery)
                self._preferred_rank = None
                self._preferred_target = None
                self._operations.finish(lease)
                self._restore_display_path()
                return
            candidate = candidates[int(selected)]
        self._discovery = discovery
        self._operations.update(lease, f"Scanning {candidate.provider_label} output…")
        handle = self._job_runner.submit(
            request_id,
            lambda report: _scan_and_load_initial_prediction(
                discovery, candidate, self._preferred_rank, report
            ),
            self._on_progress,
            self._on_initial_result,
            self._on_error,
        )
        self._operations.attach(lease, handle)

    def _on_initial_result(self, request_id: int, value: object) -> None:
        lease = self._operations.active
        discovery = self._discovery
        self._discovery = None
        if discovery is not None:
            self._job_runner.dispose(discovery)
        if lease is None or lease.request_id != request_id:
            self._job_runner.dispose(value)
            return
        if not isinstance(value, InitialLoadResult):
            self._job_runner.dispose(value)
            self._fail(lease, "The prediction worker returned an unexpected result.")
            return
        try:
            new_files = value.pred_files
            model = new_files.model(value.rank)
            object_name = model.object_name
            new_display_path = normalize_prediction_path(value.display_path)
            snapshot = _PredictionReplacementSnapshot(
                self._state.pred_files,
                self.capture_model_store(),
                self._state.ensemble,
                self._viewer.capture_paint_mappings(),
                self._context.selection,
                self._context.derive_view_state(),
                self._display_path,
            )
        except Exception as exc:
            value.close()
            self._fail(lease, str(exc))
            return
        created = False
        preferred_target = self._preferred_target
        self._preferred_rank = None
        self._preferred_target = None
        try:
            created = self._viewer.ensure_structure_object(
                value.model_state.data.structure_path, object_name, zoom=True
            )
            self._state.pred_files = value.take_prediction_files()
            self._state.ensemble = None
            self.commit_model_state(value.model_state, reset_store=True)
            self._viewer.clear_paint_mappings()
            self._context.set_selection(
                ContextSelection(
                    target_name=preferred_target or object_name,
                    metric_key=self._context.selection.metric_key,
                    reference_selection=self._context.selection.reference_selection,
                    cutoff_text=self._context.selection.cutoff_text,
                )
            )
            context_state = self._context.refresh_objects(
                preferred_target or object_name
            )
            self._view.apply_lifecycle(
                LifecycleUiUpdate(
                    selected_rank=value.rank,
                    selected_target=context_state.selected_target,
                    display_path=new_display_path,
                    model_choices=tuple(
                        ModelChoice(item.rank, item.display_label)
                        for item in self._state.pred_files.models
                    ),
                    target_choices=context_state.target_choices,
                )
            )
            self._display_path = new_display_path
            self._operations.finish(lease)
            if created:
                self._submit_default_coloring(object_name)
            try:
                recent = self._session.record_recent_prediction(self._display_path)
                self._view.apply_lifecycle(LifecycleUiUpdate(recent_predictions=recent))
            except Exception as exc:
                logger.exception("Could not persist the recent prediction")
                self._presenter.present_notice(
                    Notice(
                        "prediction_history_save_failed",
                        f"The prediction loaded, but its history could not be saved: {exc}",
                        severity="warning",
                    )
                )
            if snapshot.pred_files is not None:
                try:
                    self._job_runner.dispose(snapshot.pred_files)
                except Exception:
                    logger.exception("Could not dispose the previous prediction")
        except Exception as exc:
            self._rollback_replacement(snapshot, value, object_name, created)
            self._fail(lease, str(exc))

    def _rollback_replacement(
        self,
        snapshot: _PredictionReplacementSnapshot,
        value: InitialLoadResult,
        object_name: str,
        created: bool,
    ) -> None:
        self._display_path = snapshot.display_path
        if self._state.pred_files is not snapshot.pred_files:
            current = self._state.pred_files
            self._state.pred_files = snapshot.pred_files
            if current is not None:
                try:
                    self._job_runner.dispose(current)
                except Exception:
                    logger.exception("Could not dispose the failed prediction")
        try:
            self.restore_model_store(snapshot.model_store)
            self._state.ensemble = snapshot.ensemble
        except Exception:
            logger.exception("Could not restore the previous prediction data")
        try:
            self._viewer.restore_paint_mappings(snapshot.paint_mappings)
            if created:
                self._viewer.delete_names((object_name,))
        except Exception:
            logger.exception("Could not restore the previous viewer state")
        try:
            self._context.set_selection(snapshot.context_selection)
            self._view.apply_context(snapshot.context_state)
        except Exception:
            logger.exception("Could not restore the previous prediction context")
        try:
            self._restore_display_path()
        except Exception:
            logger.exception("Could not restore the previous prediction path")
        try:
            value.close()
        except Exception:
            logger.exception("Could not close the failed prediction result")

    def select_model(self, rank: int) -> None:
        files = self._state.pred_files
        if files is None or rank == self._state.active_model_rank:
            return
        cached = self._state.model_states.get(rank)
        if cached is not None:
            created = self._activate_model(files, cached)
            if created:
                self._submit_default_coloring(files.model(rank).object_name)
            return
        model = files.model(rank)
        lease = self._operations.begin(
            "model_switch",
            title=f"{APP_TITLE} – Loading",
            label=f"Loading {model.display_label}…",
            cancellable=True,
        )
        if lease is None:
            return
        handle = self._job_runner.submit(
            lease.request_id,
            lambda report: _load_rank_data(files, rank, report),
            self._on_progress,
            self._on_model_result,
            self._on_error,
        )
        self._operations.attach(lease, handle)

    def _on_model_result(self, request_id: int, value: object) -> None:
        lease = self._operations.active
        if lease is None or lease.request_id != request_id:
            self._job_runner.dispose(value)
            return
        if not isinstance(value, ModelSwitchResult):
            self._fail(lease, "The model worker returned an unexpected result.")
            return
        if value.pred_files is not self._state.pred_files:
            self._job_runner.dispose(value)
            self._operations.finish(lease)
            return
        try:
            canonical = self.commit_model_state(value.model_state, activate=False)
            created = self._activate_model(value.pred_files, canonical)
            self._operations.finish(lease)
            if created:
                self._submit_default_coloring(
                    value.pred_files.model(canonical.rank).object_name
                )
        except Exception as exc:
            self._fail(lease, str(exc))

    def _activate_model(self, files: PredictionFiles, model_state: ModelState) -> bool:
        model = files.model(model_state.rank)
        created = self._viewer.ensure_structure_object(
            model_state.data.structure_path, model.object_name, zoom=True
        )
        self._state.active_model_rank = model_state.rank
        self._context.refresh_objects(model.object_name)
        return created

    def _submit_default_coloring(self, object_name: str) -> None:
        """Paint a newly created model without touching reused viewer objects."""
        try:
            self._analysis.submit(
                DeferredAnalysisAction(
                    AnalysisRequest(
                        "color",
                        object_name,
                        metrics.DEFAULT_METRIC_KEY,
                        ui_revision=self._analysis.ui_revision,
                    ),
                    ColorOptions("viridis"),
                )
            )
        except Exception:
            logger.exception("Could not apply initial pLDDT-class coloring")

    def _on_error(self, request_id: int, failure: object) -> None:
        lease = self._operations.active
        if lease is None or lease.request_id != request_id:
            return
        traceback_text = getattr(failure, "traceback_text", "")
        if traceback_text:
            logger.error("Background prediction load failed:\n%s", traceback_text)
        self._fail(lease, str(getattr(failure, "message", failure)))

    def _fail(self, lease: OperationLease, message: str) -> None:
        discovery = self._discovery
        self._discovery = None
        self._preferred_rank = None
        self._preferred_target = None
        if discovery is not None:
            self._job_runner.dispose(discovery)
        self._operations.finish(lease)
        self._restore_display_path()
        self._presenter.present_notice(
            Notice(
                "prediction_load_failed",
                message,
                severity="error",
                title=f"{APP_TITLE} - error",
            )
        )

    def close(self) -> None:
        active = self._operations.active
        if active is not None and active.kind in {"prediction", "model_switch"}:
            self._operations.abandon()
        self._preferred_rank = None
        self._preferred_target = None
        self._discovery = None
