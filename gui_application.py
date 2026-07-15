"""Composition of explicit FoldQC GUI application services."""

from __future__ import annotations

from pathlib import Path

from . import export as export_module
from . import metrics
from .analysis import (
    AnalysisRequest,
    AnalysisResolver,
    ColorOptions,
    DeferredAnalysisAction,
    ExportOptions,
    PlotOptions,
    build_data_load_plan,
)
from .context_service import ContextWorkflow
from .data_acquisition import DataAcquisitionWorkflow
from .ensemble_lifecycle import EnsembleLifecycleWorkflow
from .gui_coloring import ColoringWorkflow
from .gui_dependencies import DependencyWorkflow
from .gui_export import ExportWorkflow
from .gui_metrics import MetricWorkflow
from .gui_services import DialogViewPort, GuiScheduler, JobRunner, ViewerPort
from .gui_state import PluginState
from .plot_coordinator import PlotCoordinator
from .prediction_lifecycle import PredictionLifecycleWorkflow
from .presentation import Notice, PreparedPlot, PresentationPort
from .session import PendingSessionRestore


class _ServiceRuntime:
    """Explicit shared adapters used by application services.

    This object deliberately exposes named properties instead of dynamic
    ``__getattr__`` or runtime method rebinding. It provides only explicit
    collaborations between independently injected workflow services.
    """

    def __init__(self, application: GuiApplicationServices, dialog: object) -> None:
        self.application = application
        self.dialog = dialog

    @property
    def services(self):
        return self.application

    @property
    def state(self):
        return self.application.state

    @property
    def widgets(self):
        return self.application.view.widgets

    @property
    def _viewer(self):
        return self.application.viewer

    @property
    def _presenter(self):
        return self.application.presenter

    @property
    def _job_runner(self):
        return self.application.job_runner

    @property
    def _pred_files(self):
        return self.state.pred_files

    @_pred_files.setter
    def _pred_files(self, value) -> None:
        self.state.pred_files = value

    @property
    def _model_states(self):
        return self.state.model_states

    @_model_states.setter
    def _model_states(self, value) -> None:
        self.state.model_states = value

    @property
    def _active_model_rank(self):
        return self.state.active_model_rank

    @_active_model_rank.setter
    def _active_model_rank(self, value) -> None:
        self.state.active_model_rank = value

    @property
    def _active_model_state(self):
        return self.state.active_model_state

    @property
    def _ensemble(self):
        return self.state.ensemble

    @_ensemble.setter
    def _ensemble(self, value) -> None:
        self.state.ensemble = value

    @property
    def _paint_mappings(self):
        viewer = self.application.viewer
        return (
            self.application.analysis.paint_mappings
            if viewer is None
            else viewer.paint_mappings
        )

    @_paint_mappings.setter
    def _paint_mappings(self, value) -> None:
        viewer = self.application.viewer
        if viewer is None:
            self.application.analysis.paint_mappings = value
        else:
            viewer.paint_mappings = value

    @property
    def _accepted_token_overlap_warnings(self):
        return self.application.analysis.accepted_overlap_warnings

    @_accepted_token_overlap_warnings.setter
    def _accepted_token_overlap_warnings(self, value) -> None:
        self.application.analysis.accepted_overlap_warnings = value

    @property
    def _loading_prediction(self):
        return self.application.lifecycle.loading_prediction

    @_loading_prediction.setter
    def _loading_prediction(self, value):
        self.application.lifecycle.loading_prediction = value

    @property
    def _loading_data(self):
        return self.application.data.loading_data

    @_loading_data.setter
    def _loading_data(self, value):
        self.application.data.loading_data = value

    @property
    def _gui_job_request_id(self):
        return self.application.lifecycle.gui_job_request_id

    @_gui_job_request_id.setter
    def _gui_job_request_id(self, value):
        self.application.lifecycle.gui_job_request_id = value

    @property
    def _data_load_request_id(self):
        return self.application.data.data_load_request_id

    @_data_load_request_id.setter
    def _data_load_request_id(self, value):
        self.application.data.data_load_request_id = value

    @property
    def _active_ensemble_viewer_transaction(self):
        owner = getattr(self.application, "ensemble", None)
        if owner is None or owner is self:
            return self.__dict__.get("_active_ensemble_viewer_transaction")
        return owner.__dict__.get("_active_ensemble_viewer_transaction")

    @_active_ensemble_viewer_transaction.setter
    def _active_ensemble_viewer_transaction(self, value):
        owner = getattr(self.application, "ensemble", None)
        if owner is None or owner is self:
            self.__dict__["_active_ensemble_viewer_transaction"] = value
        else:
            owner.__dict__["_active_ensemble_viewer_transaction"] = value

    @property
    def _pending_session_restore(self):
        return self.application.lifecycle.pending_session_restore

    @_pending_session_restore.setter
    def _pending_session_restore(self, value):
        self.application.lifecycle.pending_session_restore = value

    @property
    def _restoring_settings(self):
        return self.application.lifecycle.restoring_settings

    @_restoring_settings.setter
    def _restoring_settings(self, value):
        self.application.lifecycle.restoring_settings = value

    @property
    def _prediction_load_request_id(self):
        return self.application.lifecycle.prediction_load_request_id

    @_prediction_load_request_id.setter
    def _prediction_load_request_id(self, value):
        self.application.lifecycle.prediction_load_request_id = value

    # Typed widget registry fields used by the Qt-facing workflow adapter.
    @property
    def _apply_btn(self):
        return self.widgets._apply_btn

    @property
    def _close_btn(self):
        return self.widgets._close_btn

    @property
    def _conf_browser(self):
        return self.widgets._conf_browser

    @property
    def _cutoff_edit(self):
        return self.widgets._cutoff_edit

    @property
    def _cutoff_label(self):
        return self.widgets._cutoff_label

    @property
    def _dir_btn(self):
        return self.widgets._dir_btn

    @property
    def _dir_edit(self):
        return self.widgets._dir_edit

    @property
    def _ensemble_btn(self):
        return self.widgets._ensemble_btn

    @property
    def _export_csv_btn(self):
        return self.widgets._export_csv_btn

    @property
    def _file_btn(self):
        return self.widgets._file_btn

    @property
    def _guide_btn(self):
        return self.widgets._guide_btn

    @property
    def _model_combo(self):
        return self.widgets._model_combo

    @property
    def _obj_combo(self):
        return self.widgets._obj_combo

    @property
    def _obj_refresh_btn(self):
        return self.widgets._obj_refresh_btn

    @property
    def _palette_combo(self):
        return self.widgets._palette_combo

    @property
    def _palette_reverse_chk(self):
        return self.widgets._palette_reverse_chk

    @property
    def _plot_actions(self):
        return self.widgets._plot_actions

    @property
    def _plot_btn(self):
        return self.widgets._plot_btn

    @property
    def _preview_label(self):
        return self.widgets._preview_label

    @property
    def _prop_combo(self):
        return self.widgets._prop_combo

    @property
    def _prop_combo_rows(self):
        return self.widgets._prop_combo_rows

    @property
    def _ref_edit(self):
        return self.widgets._ref_edit

    @property
    def _ref_label(self):
        return self.widgets._ref_label

    @property
    def _stats_browser(self):
        return self.widgets._stats_browser

    @property
    def _vmax_edit(self):
        return self.widgets._vmax_edit

    @property
    def _vmin_edit(self):
        return self.widgets._vmin_edit

    def setWindowTitle(self, title: str) -> None:
        self._presenter.set_window_title(title)

    # Stable view operations used during lifecycle commits.
    def _select_object(self, name: str) -> None:
        self.application.view.select_object(name)

    def _select_combo_data(self, combo, value) -> bool:
        return self.application.view.select_combo_data(combo, value)

    def _combo_contains_text(self, combo, value: str) -> bool:
        return self.application.view.combo_contains_text(combo, value)

    def _select_model_rank(self, rank: int) -> bool:
        return self.application.view.select_model_rank(rank)

    def _select_property(self, key: str) -> None:
        self.application.view.select_property(key)

    def _select_property_if_available(self, key: str) -> bool:
        return self.application.view.select_property_if_available(key)

    def _save_session_settings(self, *_args) -> None:
        self.dialog._save_session_settings(*_args)

    def _raise_after_native_dialog(self) -> None:
        self.dialog._raise_after_native_dialog()

    # Explicit cross-service collaborations formerly resolved on the dialog.
    def _get_obj_name(self):
        return self.application.coloring._get_obj_name()

    def _selected_ensemble_member(self, obj_name):
        return self.application.coloring._selected_ensemble_member(obj_name)

    def _require_active_model_state(self):
        return self.application.metric_computation._require_active_model_state()

    def _apply_plddt_class_coloring(self, *args, **kwargs):
        return self.application.coloring._apply_plddt_class_coloring(*args, **kwargs)

    def _paint_mapping_cache_key(self, *args, **kwargs):
        return self.application.metric_computation._paint_mapping_cache_key(
            *args, **kwargs
        )

    @property
    def _active_analysis_action(self):
        return self.application.analysis.active_action

    def _analysis_metric_key(self):
        return self.application.analysis.metric_key()

    def _analysis_reference_selection(self):
        return self.application.analysis.reference_selection()

    def _selected_palette(self):
        return self.application.coloring._selected_palette()

    def _get_vmin_vmax(self):
        return self.application.coloring._get_vmin_vmax()

    def _get_cutoff_threshold(self):
        return self.application.coloring._get_cutoff_threshold()

    def _resolve_plot_target(self):
        return self.application.plots._resolve_plot_target()

    def _resolve_reference_indices(self, *args, **kwargs):
        return self.application.plots._resolve_reference_indices(*args, **kwargs)

    def _canonical_state_for_ensemble_member(self, member):
        return self.application.plots._canonical_state_for_ensemble_member(member)

    def _compute_property_for(self, *args, **kwargs):
        return self.application.metric_computation._compute_property_for(
            *args, **kwargs
        )

    def _compute_property_from_context(self, *args, **kwargs):
        return self.application.metric_computation._compute_property_from_context(
            *args, **kwargs
        )

    def _compute_ensemble_property(self, key):
        return self.application.metric_computation._compute_ensemble_property(key)

    def _validate_token_count(self, *args, **kwargs):
        return self.application.metric_computation._validate_token_count(
            *args, **kwargs
        )

    def _confirm_token_overlap_for_coloring(self, *args, **kwargs):
        return self.application.metric_computation._confirm_token_overlap_for_coloring(
            *args, **kwargs
        )

    def _prepare_paint_mapping(self, *args, **kwargs):
        return self.application.metric_computation._prepare_paint_mapping(
            *args, **kwargs
        )

    def _binding_site_token_indices(self, *args, **kwargs):
        return self.application.metric_computation._binding_site_token_indices(
            *args, **kwargs
        )

    def _ensure_current_data_for_property(self, *args, **kwargs):
        return self.application.metric_computation._ensure_current_data_for_property(
            *args, **kwargs
        )

    def _ensure_member_data_for_property(self, *args, **kwargs):
        return self.application.metric_computation._ensure_member_data_for_property(
            *args, **kwargs
        )

    def _ensure_member_data_for_plot(self, *args, **kwargs):
        return self.application.metric_computation._ensure_member_data_for_plot(
            *args, **kwargs
        )

    def _metric_dependencies_available(self, spec):
        return self.application.metric_computation._metric_dependencies_available(spec)

    def _defer_action_for_data(self, *args, **kwargs):
        return self.application.data._defer_action_for_data(*args, **kwargs)

    def _ensure_dependencies(self, keys, *, feature_label: str):
        return self.application.dependencies.ensure(keys, feature_label=feature_label)

    def _viewer_operation(self, method_name, fallback, *args, **kwargs):
        """Use the injected viewer port, with a test-only functional fallback."""
        viewer = self.application.viewer
        method = None if viewer is None else getattr(viewer, method_name, None)
        return fallback(*args, **kwargs) if method is None else method(*args, **kwargs)

    def _refresh_contextual_ui(self):
        return ContextWorkflow._refresh_contextual_ui(self.application.context)

    def _update_property_availability(self):
        return ContextWorkflow._update_property_availability(self.application.context)

    def _refresh_objects(self):
        return ContextWorkflow._refresh_objects(self.application.context)

    def _update_statistics_for_single(self, *args, **kwargs):
        context = self.application.context
        override = context.__dict__.get("_update_statistics_for_single")
        if override is not None:
            return override(*args, **kwargs)
        return ContextWorkflow._update_statistics_for_single(context, *args, **kwargs)

    def _update_statistics_for_members(self, *args, **kwargs):
        context = self.application.context
        override = context.__dict__.get("_update_statistics_for_members")
        if override is not None:
            return override(*args, **kwargs)
        return ContextWorkflow._update_statistics_for_members(context, *args, **kwargs)

    def _current_target_kind(self):
        return self.application.context._current_target_kind()

    def _current_target_model_states(self):
        return self.application.context._current_target_model_states()

    def _state_supports_family(self, *args, **kwargs):
        return self.application.context._state_supports_family(*args, **kwargs)

    def _target_all_supports_family(self, *args, **kwargs):
        return self.application.context._target_all_supports_family(*args, **kwargs)

    def _target_any_supports_family(self, *args, **kwargs):
        return self.application.context._target_any_supports_family(*args, **kwargs)

    def _has_fingerprint_data(self):
        return self.application.context._has_fingerprint_data()

    def _has_matrix_data_family(self, *args, **kwargs):
        return self.application.context._has_matrix_data_family(*args, **kwargs)

    def _current_target_has_multiple_chains(self):
        return self.application.context._current_target_has_multiple_chains()

    # Lifecycle primitives shared by the independently owned workflows.
    def _gui_job_is_busy(self):
        return self.application.lifecycle._gui_job_is_busy()

    def _next_gui_job_request_id(self):
        return self.application.lifecycle._next_gui_job_request_id()

    def _prediction_load_is_active(self, request_id):
        return self.application.lifecycle._prediction_load_is_active(request_id)

    def _data_load_is_active(self, request_id):
        return self.application.lifecycle._data_load_is_active(request_id)

    def _load_progress_is_active(self, request_id):
        return self.application.lifecycle._load_progress_is_active(request_id)

    def _schedule_load_progress(self, request_id, label):
        return self.application.lifecycle._schedule_load_progress(request_id, label)

    def _hide_load_progress(self):
        return self.application.lifecycle._hide_load_progress()

    def _set_prediction_load_controls_enabled(self, enabled):
        return self.application.lifecycle._set_prediction_load_controls_enabled(enabled)

    def _update_ensemble_button_state(self):
        return self.application.lifecycle._update_ensemble_button_state()

    def _ensemble_load_is_available(self):
        return self.application.lifecycle._ensemble_load_is_available()

    def _on_data_load_progress(self, request_id, label):
        return self.application.lifecycle._on_data_load_progress(request_id, label)

    def _finish_data_load(self, request_id, *, save_session=False):
        return self.application.data._finish_data_load(
            request_id, save_session=save_session
        )

    def _capture_model_store(self):
        return self.application.lifecycle._capture_model_store()

    def _restore_model_store(self, snapshot):
        return self.application.lifecycle._restore_model_store(snapshot)

    def _commit_model_state(self, *args, **kwargs):
        return self.application.lifecycle._commit_model_state(*args, **kwargs)

    def _capture_viewer_mapping_context(self):
        return self.application.context._capture_viewer_mapping_context()

    def _restore_viewer_mapping_context(self, context):
        return self.application.context._restore_viewer_mapping_context(context)

    def _clear_viewer_mapping_cache(self):
        return self.application.context._clear_viewer_mapping_cache()

    def _update_confidence_summary(self):
        return self.application.context._update_confidence_summary()

    def _select_first_available_property(self):
        return self.application.context._select_first_available_property()

    def _update_plot_actions(self):
        return self.application.context._update_plot_actions()

    def _rollback_ensemble_viewer_transaction(self, *, refresh_gui):
        return self.application.ensemble._rollback_ensemble_viewer_transaction(
            refresh_gui=refresh_gui
        )


class MetricComputationService(MetricWorkflow, _ServiceRuntime):
    """Own viewer-context resolution and per-model metric computation."""


class ColoringCoordinator(ColoringWorkflow, _ServiceRuntime):
    """Own transactional single-model and ensemble coloring."""


class PlotService(PlotCoordinator, _ServiceRuntime):
    """Own plot coordination and presentation-ready figure preparation."""

    def _show_plot_figure(self, figure, title: str) -> None:
        self._presenter.show_plot(PreparedPlot(figure, title))


class ExportCoordinator(ExportWorkflow, _ServiceRuntime):
    """Own captured CSV export orchestration."""


class AnalysisCoordinator(_ServiceRuntime):
    """Resolve immutable actions and coordinate dedicated analysis services."""

    def __init__(self, application: GuiApplicationServices, dialog: object) -> None:
        _ServiceRuntime.__init__(self, application, dialog)
        self.resolver = AnalysisResolver()
        self.accepted_overlap_warnings: set[tuple[str, str]] = set()
        self.paint_mappings: dict[tuple[str, str], object] = {}
        self.ui_revision = 0
        self.active_action: DeferredAnalysisAction | None = None

    def bump_revision(self, *_args) -> None:
        self.ui_revision += 1

    def capture_current(
        self, action, *, export_path: str | Path | None = None
    ) -> DeferredAnalysisAction:
        active = self.active_action
        if active is not None and active.request.action == action:
            return active
        metric_combo = getattr(self.widgets, "_prop_combo", None)
        metric_key = metric_combo.currentData() if metric_combo is not None else None
        plot = metrics.PLOTS.find(action)
        if plot is not None and not plot.requires_metric:
            metric_key = None
        cutoff_edit = getattr(self.widgets, "_cutoff_edit", None)
        cutoff_text = cutoff_edit.text().strip() if cutoff_edit is not None else ""
        metric = None if metric_key is None else metrics.METRICS.find(metric_key)
        needs_cutoff = bool(
            (metric is not None and metric.needs_cutoff)
            or action in {"binding_site_fingerprint", "ensemble_site_summary"}
        )
        if needs_cutoff:
            try:
                cutoff = float(cutoff_text) if cutoff_text else 5.0
            except ValueError as exc:
                raise ValueError(
                    "Cutoff / threshold must be a positive number in Å."
                ) from exc
            if cutoff <= 0:
                raise ValueError("Cutoff / threshold must be greater than 0 Å.")
        else:
            cutoff = None
        target_combo = getattr(self.widgets, "_obj_combo", None)
        target_name = (
            target_combo.currentText().strip() if target_combo is not None else ""
        )
        if not target_name:
            try:
                target_name = self.application.plots._resolve_plot_target().label
            except Exception:
                target_name = "current"
        reference_edit = getattr(self.widgets, "_ref_edit", None)
        request = self.capture_request(
            action,
            target_name=target_name,
            metric_key=metric_key,
            reference_selection=(
                reference_edit.text().strip() if reference_edit is not None else ""
            ),
            cutoff_angstrom=cutoff,
            ui_revision=self.ui_revision,
        )
        if action == "color":
            palette, reverse = self.application.coloring._selected_palette()
            vmin, vmax = self.application.coloring._get_vmin_vmax()
            options = ColorOptions(palette, reverse, vmin, vmax)
        elif metrics.PLOTS.find(action) is not None:
            palette, reverse = self.application.coloring._selected_palette()
            vmin, vmax = self.application.coloring._get_vmin_vmax()
            options = PlotOptions(palette, reverse, vmin, vmax)
        else:
            options = None if export_path is None else ExportOptions(Path(export_path))
        return DeferredAnalysisAction(request, options)

    def resume(self, action: DeferredAnalysisAction) -> None:
        if action.request.ui_revision != self.ui_revision:
            return
        self._execute_action(action)

    def _execute_action(self, action: DeferredAnalysisAction) -> None:
        previous = self.active_action
        self.active_action = action
        try:
            if action.request.action == "color":
                self.application.coloring._apply_coloring()
            elif action.request.action == "export":
                if not isinstance(action.options, ExportOptions):
                    raise ValueError("Deferred export is missing its output path.")
                self.application.export._export_csv_to_path(action.options.path)
            else:
                self.application.plots._show_selected_plot(action.request.action)
        finally:
            self.active_action = previous

    def metric_key(self):
        action = self.active_action
        if action is not None:
            return action.request.metric_key
        combo = getattr(self, "_prop_combo", None)
        return None if combo is None else combo.currentData()

    def reference_selection(self) -> str:
        action = self.active_action
        if action is not None:
            return action.request.reference_selection
        edit = getattr(self, "_ref_edit", None)
        return "" if edit is None else edit.text().strip()

    def _analysis_cutoff(self) -> float | None:
        action = self.active_action
        if action is not None:
            return action.request.cutoff_angstrom
        return None

    def capture_request(
        self,
        action,
        *,
        target_name: str,
        metric_key: str | None,
        reference_selection: str = "",
        cutoff_angstrom: float | None = None,
        ui_revision: int = 0,
    ) -> AnalysisRequest:
        return AnalysisRequest(
            action=action,
            target_name=target_name,
            metric_key=metric_key,
            reference_selection=reference_selection,
            cutoff_angstrom=cutoff_angstrom,
            ui_revision=ui_revision,
        )

    def resolve_and_plan(self, request: AnalysisRequest):
        resolved = self.resolver.resolve(request, self.state)
        return resolved, build_data_load_plan(resolved)

    def apply_coloring(self) -> None:
        try:
            action = self.capture_current("color")
        except ValueError as exc:
            self._presenter.present_notice(Notice("color_preflight", str(exc)))
            return
        self._execute_action(action)

    def default_export_path(self) -> str | None:
        target = self.application.plots._resolve_plot_target()
        if target is None:
            return None
        data = None if target.kind == "ensemble_group" else target.model_states[0].data
        return export_module.default_csv_export_path(
            self._pred_files,
            data,
            self._prop_combo.currentData(),
            ensemble=target.kind == "ensemble_group",
        )

    def export_csv(self, path: str | Path) -> None:
        path_obj = Path(path)
        if path_obj.suffix.lower() != ".csv":
            path_obj = path_obj.with_suffix(".csv")
        try:
            action = self.capture_current("export", export_path=path_obj)
        except ValueError as exc:
            self._presenter.present_notice(Notice("export_preflight", str(exc)))
            return
        self._execute_action(action)

    def show_plot(self, plot_type: str | None = None) -> None:
        if plot_type is None:
            self.application.plots._show_selected_plot(None)
            return
        try:
            action = self.capture_current(plot_type)
        except ValueError as exc:
            self._presenter.present_notice(Notice("plot_preflight", str(exc)))
            return
        self._execute_action(action)


class PredictionLifecycleService(PredictionLifecycleWorkflow, _ServiceRuntime):
    """Own discovery, replacement, model switching, and session restoration."""

    def __init__(self, application: GuiApplicationServices, dialog: object) -> None:
        _ServiceRuntime.__init__(self, application, dialog)
        self.loading_prediction = False
        self.gui_job_request_id = 0
        self.prediction_load_request_id = 0
        self.restoring_settings = False
        self.pending_session_restore = PendingSessionRestore()
        self._active_load_handle = None
        self._model_switch_previous_store = None
        self._model_switch_previous_viewer_context = None
        self._load_progress_dialog = None
        self._progress_show_generation = 0

    @property
    def _loading_prediction(self):
        return self.loading_prediction

    @_loading_prediction.setter
    def _loading_prediction(self, value):
        self.loading_prediction = value

    @property
    def _gui_job_request_id(self):
        return self.gui_job_request_id

    @_gui_job_request_id.setter
    def _gui_job_request_id(self, value):
        self.gui_job_request_id = value

    @property
    def _prediction_load_request_id(self):
        return self.prediction_load_request_id

    @_prediction_load_request_id.setter
    def _prediction_load_request_id(self, value):
        self.prediction_load_request_id = value

    @property
    def _restoring_settings(self):
        return self.restoring_settings

    @_restoring_settings.setter
    def _restoring_settings(self, value):
        self.restoring_settings = value

    @property
    def _pending_session_restore(self):
        return self.pending_session_restore

    @_pending_session_restore.setter
    def _pending_session_restore(self, value):
        self.pending_session_restore = value

    def load_prediction(self, path: str | None = None) -> None:
        if path is not None:
            self._dir_edit.setText(path)
        self._load_prediction_dir()

    def show_ensemble(self) -> None:
        self.application.ensemble.show_ensemble()

    def close(self) -> None:
        if self._gui_job_is_busy():
            self._abandon_active_gui_job()


class DataAcquisitionService(DataAcquisitionWorkflow, _ServiceRuntime):
    """Own lazy per-rank loading and exact deferred-action resumption."""

    def __init__(self, application: GuiApplicationServices, dialog: object) -> None:
        super().__init__(application, dialog)
        self.loading_data = False
        self.data_load_request_id = 0
        self._active_deferred_analysis = None
        self._active_data_error_title = "FoldQC - error"
        self._active_load_handle = None
        self._load_progress_dialog = None
        self._progress_show_generation = 0

    @property
    def _loading_data(self):
        return self.loading_data

    @_loading_data.setter
    def _loading_data(self, value):
        self.loading_data = value

    @property
    def _data_load_request_id(self):
        return self.data_load_request_id

    @_data_load_request_id.setter
    def _data_load_request_id(self, value):
        self.data_load_request_id = value

    @property
    def _gui_job_request_id(self):
        return self.application.lifecycle.gui_job_request_id

    @_gui_job_request_id.setter
    def _gui_job_request_id(self, value):
        self.application.lifecycle.gui_job_request_id = value


class EnsembleLifecycleService(EnsembleLifecycleWorkflow, _ServiceRuntime):
    """Own preparation and transactional activation of ensembles."""

    def __init__(self, application: GuiApplicationServices, dialog: object) -> None:
        super().__init__(application, dialog)
        self._active_ensemble_viewer_transaction = None
        self._active_load_handle = None
        self._load_progress_dialog = None
        self._progress_show_generation = 0

    def show_ensemble(self) -> None:
        EnsembleLifecycleWorkflow._show_ensemble(self)


class ContextService(ContextWorkflow, _ServiceRuntime):
    """Derive and render contextual metric, plot, and target state."""

    def refresh(self):
        self._update_ensemble_button_state()
        state = self._derive_context_view_state()
        self.application.view.apply_context(state)
        return state

    def on_property_changed(self) -> None:
        ContextWorkflow._on_property_changed(self)


class QtDependencyService(DependencyWorkflow, _ServiceRuntime):
    """Concrete Qt optional-dependency adapter."""

    def __init__(self, application: GuiApplicationServices, dialog: object) -> None:
        _ServiceRuntime.__init__(self, application, dialog)

    def initialize(self) -> None:
        self._initialize_dependency_controller()

    def ensure(self, keys, *, feature_label: str) -> bool:
        return bool(self._ensure_dependencies(keys, feature_label=feature_label))


class GuiApplicationServices:
    """All explicit application services for one dialog instance."""

    def __init__(
        self,
        dialog: object,
        *,
        state: PluginState,
        viewer: ViewerPort | None,
        presenter: PresentationPort,
        view: DialogViewPort,
        scheduler: GuiScheduler,
        job_runner: JobRunner,
    ) -> None:
        self.dialog = dialog
        self.state = state
        self.viewer = viewer
        self.presenter = presenter
        self.view = view
        self.scheduler = scheduler
        self.job_runner = job_runner
        self.lifecycle = PredictionLifecycleService(self, dialog)
        self.data = DataAcquisitionService(self, dialog)
        self.ensemble = EnsembleLifecycleService(self, dialog)
        self.context = ContextService(self, dialog)
        self.metric_computation = MetricComputationService(self, dialog)
        self.coloring = ColoringCoordinator(self, dialog)
        self.plots = PlotService(self, dialog)
        self.export = ExportCoordinator(self, dialog)
        self.analysis = AnalysisCoordinator(self, dialog)
        self.dependencies = QtDependencyService(self, dialog)
