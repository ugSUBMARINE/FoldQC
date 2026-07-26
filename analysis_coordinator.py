"""One immutable analysis pipeline shared by coloring, plots, and export."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from . import export as export_module
from .analysis import (
    AnalysisPreflightError,
    AnalysisRequest,
    AnalysisResolver,
    ColorOptions,
    ComputedMetric,
    DataLoadPlan,
    DeferredAnalysisAction,
    ExportOptions,
    MetricComputationJobResult,
    PlotOptions,
    ResolvedAnalysis,
    build_data_load_plan,
)
from .context_service import ContextService
from .data_acquisition import DataAcquisitionService
from .gui_coloring import ColoringCoordinator
from .gui_export import ExportCoordinator
from .gui_metrics import MetricComputationService
from .gui_services import (
    DataAcquisitionOutcome,
    DataLoadObserver,
    DependencyService,
    JobRunner,
    OperationCoordinatorPort,
    OperationLease,
)
from .gui_state import PluginState
from .plot_coordinator import PlotCoordinator
from .presentation import Notice, PresentationPort

logger = logging.getLogger(__name__)
APP_TITLE = "FoldQC"


@dataclass(frozen=True)
class _PendingMetricComputation:
    lease: OperationLease
    action: DeferredAnalysisAction
    resolved: ResolvedAnalysis


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
        job_runner: JobRunner,
        operations: OperationCoordinatorPort,
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
        self._job_runner = job_runner
        self._operations = operations
        self._resolver = AnalysisResolver()
        self._ui_revision: int = 0
        self._pending: DeferredAnalysisAction | None = None
        self._pending_computation: _PendingMetricComputation | None = None

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
        metric = resolved.metric_spec
        if metric is not None and metric.compute_in_background:
            self._start_metric_computation(action, resolved)
            return
        computed = self._metric_computation.compute(resolved)
        self._dispatch(action, resolved, computed)

    def _start_metric_computation(
        self, action: DeferredAnalysisAction, resolved: ResolvedAnalysis
    ) -> None:
        metric = resolved.metric_spec
        if metric is None:
            raise ValueError("Background metric computation requires a metric.")
        first_member = resolved.members[0]
        member_suffix = (
            f" (1/{len(resolved.members)})" if len(resolved.members) > 1 else ""
        )
        lease = self._operations.begin(
            "analysis",
            title=f"{APP_TITLE} – Analysis",
            label=(
                f"Calculating {metric.label} for {first_member.label}…{member_suffix}"
            ),
        )
        if lease is None:
            return
        pending = _PendingMetricComputation(lease, action, resolved)
        self._pending_computation = pending

        def task(
            report_progress: Callable[[str], None],
        ) -> MetricComputationJobResult:
            try:
                computed = self._metric_computation.compute(
                    resolved, report_progress=report_progress
                )
            except AnalysisPreflightError as exc:
                return MetricComputationJobResult(resolved, notice=exc.notice)
            return MetricComputationJobResult(resolved, computed=computed)

        handle = self._job_runner.submit(
            lease.request_id,
            task,
            self._on_metric_progress,
            self._on_metric_result,
            self._on_metric_error,
        )
        self._operations.attach(lease, handle)

    def _matching_computation(
        self, request_id: int
    ) -> _PendingMetricComputation | None:
        pending = self._pending_computation
        if pending is None:
            return None
        if pending.lease.request_id != request_id or not self._operations.is_current(
            pending.lease
        ):
            return None
        return pending

    def _on_metric_progress(self, request_id: int, label: str) -> None:
        pending = self._matching_computation(request_id)
        if pending is not None:
            self._operations.update(pending.lease, label)

    def _finish_metric_computation(self, pending: _PendingMetricComputation) -> None:
        if self._pending_computation is pending:
            self._pending_computation = None
        self._operations.finish(pending.lease)

    def _on_metric_result(self, request_id: int, value: object) -> None:
        pending = self._matching_computation(request_id)
        if pending is None:
            self._job_runner.dispose(value)
            return
        if not isinstance(value, MetricComputationJobResult):
            self._job_runner.dispose(value)
            self._finish_metric_computation(pending)
            self._presenter.present_notice(
                Notice(
                    "metric_result_type",
                    "The metric worker returned an unexpected result.",
                    severity="error",
                    title=f"{APP_TITLE} - error",
                )
            )
            return
        if value.resolved is not pending.resolved:
            self._job_runner.dispose(value)
            self._finish_metric_computation(pending)
            self._presenter.present_notice(
                Notice(
                    "metric_result_context",
                    "The metric worker returned a result for an unexpected analysis.",
                    severity="error",
                    title=f"{APP_TITLE} - error",
                )
            )
            return
        if not value.resolved.validate_current(
            self._state, ui_revision=self._ui_revision
        ):
            self._job_runner.dispose(value)
            self._finish_metric_computation(pending)
            return

        self._finish_metric_computation(pending)
        if value.notice is not None:
            self._presenter.present_notice(value.notice)
            return
        computed = value.computed
        if computed is None:
            self._presenter.present_notice(
                Notice(
                    "metric_result_missing",
                    "The metric worker did not return computed values.",
                    severity="error",
                    title=f"{APP_TITLE} - error",
                )
            )
            return
        try:
            self._dispatch(pending.action, pending.resolved, computed)
        except AnalysisPreflightError as exc:
            self._presenter.present_notice(exc.notice)
        except Exception as exc:
            self._presenter.present_notice(
                Notice(
                    "analysis_failed",
                    str(exc),
                    severity="error",
                    title=f"{APP_TITLE} - error",
                )
            )

    def _on_metric_error(self, request_id: int, failure: object) -> None:
        pending = self._matching_computation(request_id)
        if pending is None:
            return
        message = str(getattr(failure, "message", failure))
        traceback_text = getattr(failure, "traceback_text", "")
        if traceback_text:
            logger.error("Background metric computation failed:\n%s", traceback_text)
        self._finish_metric_computation(pending)
        self._presenter.present_notice(
            Notice(
                "metric_computation_failed",
                message,
                severity="error",
                title=f"{APP_TITLE} - error",
            )
        )

    def _dispatch(
        self,
        action: DeferredAnalysisAction,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
    ) -> None:
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

    def close(self) -> None:
        self._pending = None
        pending = self._pending_computation
        self._pending_computation = None
        if pending is not None and self._operations.is_current(pending.lease):
            self._operations.abandon()

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
