"""Composition helpers binding explicit GUI services to the Qt adapter.

The existing workflow modules remain focused by responsibility, but they are
owned service objects rather than bases of the main dialog.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType

from . import metrics
from .analysis import (
    AnalysisRequest,
    AnalysisResolver,
    DeferredAnalysisAction,
    ExportOptions,
    build_data_load_plan,
)
from .gui_coloring import ColoringWorkflow
from .gui_dependencies import DependencyWorkflow
from .gui_export import ExportWorkflow
from .gui_loading import PredictionLifecycleWorkflow
from .gui_metrics import MetricWorkflow
from .gui_plots import PlotWorkflow
from .session import PendingSessionRestore


class _BoundWorkflowService:
    """Bind one responsibility's implementation methods to the Qt adapter."""

    implementation_types: tuple[type, ...] = ()

    def __init__(self, dialog) -> None:
        self.dialog = dialog
        self._methods = {
            name: member
            for implementation in self.implementation_types
            for name, member in vars(implementation).items()
            if callable(member) and name.startswith("_") and not name.startswith("__")
        }

    def find_method(self, name: str):
        implementation = self._methods.get(name)
        if implementation is None:
            return None
        return MethodType(implementation, self.dialog)


class AnalysisCoordinator(_BoundWorkflowService):
    """Own coloring, plotting, export, and per-model metric coordination."""

    implementation_types = (
        MetricWorkflow,
        ColoringWorkflow,
        PlotWorkflow,
        ExportWorkflow,
    )

    def __init__(self, dialog) -> None:
        super().__init__(dialog)
        self.resolver = AnalysisResolver()
        self.accepted_overlap_warnings: set[tuple[str, str]] = set()
        self.paint_mappings: dict[tuple[str, str], object] = {}
        self.ui_revision = 0

    def bump_revision(self, *_args) -> None:
        self.ui_revision += 1

    def capture_current(
        self, action, *, export_path: str | Path | None = None
    ) -> DeferredAnalysisAction:
        dialog = self.dialog
        metric_combo = getattr(dialog, "_prop_combo", None)
        metric_key = metric_combo.currentData() if metric_combo is not None else None
        plot = metrics.PLOTS.find(action)
        if plot is not None and not plot.requires_metric:
            metric_key = None
        target_combo = getattr(dialog, "_obj_combo", None)
        target_name = (
            target_combo.currentText().strip() if target_combo is not None else ""
        )
        if not target_name:
            target_resolver = dialog.__dict__.get("_resolve_plot_target")
            if callable(target_resolver):
                resolved_target = target_resolver()
                target_name = getattr(resolved_target, "label", "")
        reference_edit = getattr(dialog, "_ref_edit", None)
        reference_selection = (
            reference_edit.text().strip() if reference_edit is not None else ""
        )
        cutoff = None
        cutoff_edit = getattr(dialog, "_cutoff_edit", None)
        cutoff_text = cutoff_edit.text().strip() if cutoff_edit is not None else ""
        try:
            cutoff = float(cutoff_text) if cutoff_text else 5.0
        except ValueError:
            pass
        request = self.capture_request(
            action,
            target_name=target_name or "current",
            metric_key=metric_key,
            reference_selection=reference_selection,
            cutoff_angstrom=cutoff if cutoff is not None and cutoff > 0 else None,
            ui_revision=self.ui_revision,
        )
        options = None if export_path is None else ExportOptions(Path(export_path))
        return DeferredAnalysisAction(request, options)

    def resume(self, action: DeferredAnalysisAction) -> None:
        if action.request.ui_revision != self.ui_revision:
            return
        if action.request.action == "color":
            self.apply_coloring()
            return
        if action.request.action == "export":
            if not isinstance(action.options, ExportOptions):
                raise ValueError("Deferred export is missing its output path.")
            self.find_method("_export_csv_to_path")(action.options.path)
            return
        self.show_plot(action.request.action)

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
        resolved = self.resolver.resolve(request, self.dialog.state)
        return resolved, build_data_load_plan(resolved)

    def apply_coloring(self) -> None:
        self.find_method("_apply_coloring")()

    def export_csv(self) -> None:
        self.find_method("_export_csv")()

    def show_plot(self, plot_type: str | None = None) -> None:
        self.find_method("_show_selected_plot")(plot_type)


class PredictionLifecycleCoordinator(_BoundWorkflowService):
    """Own discovery, loading, switching, lazy data, and ensemble workflows."""

    implementation_types = (PredictionLifecycleWorkflow,)

    def __init__(self, dialog) -> None:
        super().__init__(dialog)
        self.loading_prediction = False
        self.loading_data = False
        self.gui_job_request_id = 0
        self.prediction_load_request_id = 0
        self.data_load_request_id = 0
        self.restoring_settings = False
        self.pending_session_restore = PendingSessionRestore()

    def load_prediction(self) -> None:
        self.find_method("_load_prediction_dir")()

    def show_ensemble(self) -> None:
        self.find_method("_show_ensemble")()

    def close(self) -> None:
        if self.find_method("_gui_job_is_busy")():
            self.find_method("_abandon_active_gui_job")()


class QtDependencyService(_BoundWorkflowService):
    """Explicit optional-dependency service owned by the composition root."""

    implementation_types = (DependencyWorkflow,)

    def initialize(self) -> None:
        self.find_method("_initialize_dependency_controller")()

    def ensure(self, keys, *, feature_label: str) -> bool:
        return bool(
            self.find_method("_ensure_dependencies")(keys, feature_label=feature_label)
        )


class GuiApplicationServices:
    """All composed application services for one dialog instance."""

    def __init__(self, dialog) -> None:
        self.analysis = AnalysisCoordinator(dialog)
        self.lifecycle = PredictionLifecycleCoordinator(dialog)
        self.dependencies = QtDependencyService(dialog)
        self._ordered = (self.lifecycle, self.analysis, self.dependencies)

    def resolve_implementation_method(self, name: str):
        matches = [
            method for service in self._ordered if (method := service.find_method(name))
        ]
        if not matches:
            return None
        if len(matches) > 1:
            # Shared helper names must have one authoritative owner.
            owners = ", ".join(
                type(service).__name__
                for service in self._ordered
                if service.find_method(name)
            )
            raise AttributeError(f"Ambiguous GUI service method {name!r}: {owners}")
        return matches[0]
