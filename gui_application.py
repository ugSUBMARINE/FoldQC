"""Constructor-only composition of FoldQC GUI application services."""

from __future__ import annotations

from .analysis_coordinator import AnalysisCoordinator
from .context_service import ContextService
from .data_acquisition import DataAcquisitionService
from .ensemble_lifecycle import EnsembleLifecycleService
from .gui_coloring import ColoringCoordinator
from .gui_export import ExportCoordinator
from .gui_metrics import MetricComputationService
from .gui_operations import GuiOperationCoordinator
from .gui_services import (
    DependencyService,
    DialogViewPort,
    GuiScheduler,
    JobRunner,
    SessionPort,
    ViewerPort,
)
from .gui_state import PluginState
from .plot_coordinator import PlotCoordinator
from .plot_preparation import PlotPreparationService
from .prediction_lifecycle import PredictionLifecycleService
from .presentation import PresentationPort
from .statistics_selection import StatisticsSelectionService


class GuiApplicationServices:
    """Typed aggregate returned to the Qt composition root.

    Individual services receive only their direct collaborators and never retain
    this aggregate.
    """

    def __init__(
        self,
        *,
        state: PluginState,
        viewer: ViewerPort,
        presenter: PresentationPort,
        view: DialogViewPort,
        scheduler: GuiScheduler,
        job_runner: JobRunner,
        session: SessionPort,
        dependencies: DependencyService,
        metric_rows: dict[str, int],
    ) -> None:
        self.state = state
        self.viewer = viewer
        self.presenter = presenter
        self.view = view
        self.scheduler = scheduler
        self.job_runner = job_runner
        self.session = session
        self.dependencies = dependencies

        self.operations = GuiOperationCoordinator(presenter, view)
        self.context = ContextService(state, viewer, presenter, view, metric_rows)
        accepted_overlap_warnings: set[tuple[str, str]] = set()
        self.metric_computation = MetricComputationService(
            state,
            viewer,
            presenter,
            accepted_overlap_warnings,
        )
        self.statistics_selection = StatisticsSelectionService(viewer, presenter, view)
        self.coloring = ColoringCoordinator(
            viewer,
            presenter,
            self.context,
            self.metric_computation,
            self.statistics_selection,
        )
        self.plot_preparation = PlotPreparationService()
        self.plots = PlotCoordinator(self.plot_preparation)
        self.export = ExportCoordinator(state, presenter)
        self.data = DataAcquisitionService(
            state,
            presenter,
            scheduler,
            job_runner,
            self.operations,
        )
        self.analysis = AnalysisCoordinator(
            state,
            presenter,
            dependencies,
            self.data,
            self.metric_computation,
            self.coloring,
            self.plots,
            self.export,
            self.context,
            accepted_overlap_warnings,
        )
        self.lifecycle = PredictionLifecycleService(
            state,
            viewer,
            presenter,
            view,
            self.context,
            self.operations,
            job_runner,
            session,
            self.analysis,
        )
        self.ensemble = EnsembleLifecycleService(
            state,
            viewer,
            presenter,
            scheduler,
            job_runner,
            self.operations,
            self.lifecycle,
            self.context,
        )

    def close(self) -> None:
        self.ensemble.close()
        self.operations.abandon()
        self.data.close()
        self.lifecycle.close()
        self.dependencies.close()
        prediction_files = self.state.pred_files
        self.state.pred_files = None
        if prediction_files is not None:
            self.job_runner.dispose(prediction_files)
        self.presenter.close()
        self.view.close()
