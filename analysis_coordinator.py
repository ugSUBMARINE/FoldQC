"""One immutable analysis pipeline shared by coloring, plots, and export."""

from __future__ import annotations

from . import export as export_module
from .analysis import (
    AnalysisPreflightError,
    AnalysisRequest,
    AnalysisResolver,
    ColorOptions,
    DataLoadPlan,
    DeferredAnalysisAction,
    ExportOptions,
    PlotOptions,
    ResolvedAnalysis,
    build_data_load_plan,
)
from .context_service import ContextService
from .data_acquisition import DataAcquisitionService
from .gui_coloring import ColoringCoordinator
from .gui_export import ExportCoordinator
from .gui_metrics import MetricComputationService
from .gui_services import DataAcquisitionOutcome, DataLoadObserver, DependencyService
from .gui_state import PluginState
from .plot_coordinator import PlotCoordinator
from .presentation import Notice, PresentationPort


class AnalysisCoordinator(DataLoadObserver):
    """Resolve, acquire, revalidate, compute, and dispatch captured actions."""

    def __init__(
        self,
        state: PluginState,
        presenter: PresentationPort,
        dependencies: DependencyService,
        data: DataAcquisitionService,
        metric_computation: MetricComputationService,
        coloring: ColoringCoordinator,
        plots: PlotCoordinator,
        export: ExportCoordinator,
        context: ContextService,
        accepted_overlap_warnings: set[tuple[str, str]],
    ) -> None:
        self._state = state
        self._presenter = presenter
        self._dependencies = dependencies
        self._data = data
        self._metric_computation = metric_computation
        self._coloring = coloring
        self._plots = plots
        self._export = export
        self._context = context
        self._accepted_overlap_warnings = accepted_overlap_warnings
        self._resolver = AnalysisResolver()
        self._ui_revision: int = 0
        self._pending: DeferredAnalysisAction | None = None

    @property
    def ui_revision(self) -> int:
        return self._ui_revision

    def invalidate_ui(self, *_args: object) -> None:
        self._ui_revision += 1

    def resolve_and_plan(
        self, request: AnalysisRequest
    ) -> tuple[ResolvedAnalysis, DataLoadPlan]:
        resolved = self._resolver.resolve(request, self._state)
        return resolved, build_data_load_plan(resolved)

    def submit(self, action: DeferredAnalysisAction) -> None:
        if action.request.ui_revision != self._ui_revision:
            return
        try:
            resolved, plan = self.resolve_and_plan(action.request)
            plot = resolved.plot_spec
            feature_label = (
                resolved.metric_spec.label
                if plot is None and resolved.metric_spec is not None
                else "analysis"
                if plot is None
                else f"The {plot.label.lower()} plot"
            )
            if not self._dependencies.ensure(
                resolved.dependency_keys, feature_label=feature_label
            ):
                return
            if not plan.is_empty:
                self._pending = action
                if not self._data.acquire(action, plan, self):
                    self._pending = None
                return
            self._execute(action, resolved)
        except AnalysisPreflightError as exc:
            self._presenter.present_notice(exc.notice)
        except Exception as exc:
            self._presenter.present_notice(
                Notice(
                    "analysis_failed",
                    str(exc),
                    severity="error",
                    title="FoldQC - error",
                )
            )

    def data_acquisition_finished(self, outcome: DataAcquisitionOutcome) -> None:
        action = self._pending
        self._pending = None
        if action is None or action is not outcome.action:
            return
        if outcome.status in {"stale", "cancelled"}:
            return
        if outcome.status == "failed":
            if outcome.notice is not None:
                self._presenter.present_notice(outcome.notice)
            return
        if action.request.ui_revision != self._ui_revision:
            return
        try:
            resolved = self._resolver.resolve(action.request, self._state)
            self._context.refresh()
            self._execute(action, resolved)
        except AnalysisPreflightError as exc:
            self._presenter.present_notice(exc.notice)
        except Exception as exc:
            self._presenter.present_notice(
                Notice(
                    "analysis_resume_failed",
                    str(exc),
                    severity="error",
                    title="FoldQC - error",
                )
            )

    def _execute(
        self, action: DeferredAnalysisAction, resolved: ResolvedAnalysis
    ) -> None:
        resolved = self._metric_computation.resolve_contexts(resolved)
        computed = self._metric_computation.compute(resolved)
        options = action.options
        if action.request.action == "color":
            if not isinstance(options, ColorOptions):
                raise ValueError("Coloring requires captured color options.")
            self._coloring.execute(resolved, computed, options)
        elif action.request.action == "export":
            if not isinstance(options, ExportOptions):
                raise ValueError("Export requires a captured output path.")
            self._export.execute(resolved, computed, options)
        else:
            if not isinstance(options, PlotOptions):
                raise ValueError("Plotting requires captured plot options.")
            self._presenter.show_plot(self._plots.prepare(resolved, computed, options))

    def default_export_path(self, request: AnalysisRequest) -> str | None:
        try:
            resolved = self._resolver.resolve(request, self._state)
        except AnalysisPreflightError:
            return None
        data = (
            None
            if resolved.target.kind == "ensemble_group"
            else resolved.members[0].model_state.data
        )
        return export_module.default_csv_export_path(
            self._state.pred_files,
            data,
            request.metric_key,
            ensemble=resolved.target.kind == "ensemble_group",
        )
