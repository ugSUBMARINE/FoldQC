"""Report snapshot ownership and validated interface viewer/export interactions."""

from __future__ import annotations

from dataclasses import dataclass
from weakref import ReferenceType, ref

from .analysis import ResolvedAnalysis
from .ensemble import EnsembleState
from .export import write_interface_report
from .gui_services import ObjectTokenSelection, OperationCoordinatorPort, ViewerPort
from .gui_state import PluginState
from .interface_analysis import (
    InterfaceReport,
    InterfaceReportObserver,
    InterfaceReportViewPort,
    InterfaceSelectionRequest,
)
from .loader_models import PredictionFiles
from .model_state import ModelState
from .presentation import Notice, PresentationPort

INTERFACE_SELECTION = "foldqc_interface"


@dataclass(frozen=True)
class _MemberSource:
    rank: int
    model: ReferenceType[ModelState]
    version: int
    object_name: str


@dataclass(frozen=True)
class _ReportSource:
    prediction: ReferenceType[PredictionFiles]
    ensemble: ReferenceType[EnsembleState] | None
    target_name: str
    single: bool
    members: tuple[_MemberSource, ...]


class InterfaceReportService:
    def __init__(
        self,
        state: PluginState,
        viewer: ViewerPort,
        presenter: PresentationPort,
        operations: OperationCoordinatorPort,
    ) -> None:
        self._state = state
        self._viewer = viewer
        self._presenter = presenter
        self._operations = operations
        self._report: InterfaceReport | None = None
        self._source: _ReportSource | None = None
        self._view: InterfaceReportViewPort | None = None
        self._owns_selection = False
        self._clear_pending = False
        self._target_name: str | None = None

    @property
    def report(self) -> InterfaceReport | None:
        return self._report

    def show(
        self,
        source: ResolvedAnalysis,
        report: InterfaceReport,
        observer: InterfaceReportObserver,
    ) -> None:
        canonical = {
            member.rank: member.model_state.token_map for member in source.members
        }
        if any(
            canonical.get(model.rank) is not model.token_map for model in report.models
        ):
            raise ValueError(
                "Interface report must use the resolved canonical token maps."
            )
        new_source = _ReportSource(
            ref(source.prediction_files),
            None if source.ensemble_identity is None else ref(source.ensemble_identity),
            source.request.target_name,
            source.target.kind == "single",
            tuple(
                _MemberSource(
                    m.rank, ref(m.model_state), m.expected_version, m.obj_name
                )
                for m in source.members
            ),
        )
        new_view = self._presenter.show_interface_report(report, observer)
        self._source, self._report, self._view = new_source, report, new_view
        self.refresh_current()

    def is_current(self) -> bool:
        source = self._source
        if (
            source is None
            or self._state.pred_files is None
            or self._state.pred_files is not source.prediction()
        ):
            return False
        if self._target_name is not None and self._target_name != source.target_name:
            return False
        if source.single:
            if self._state.active_model_state is not source.members[0].model():
                return False
        elif (
            source.ensemble is None
            or source.ensemble() is None
            or self._state.ensemble is not source.ensemble()
        ):
            return False
        return all(
            (model := m.model()) is not None
            and self._state.model_states.get(m.rank) is model
            and model.version == m.version
            and self._viewer.name_exists(m.object_name)
            for m in source.members
        )

    def refresh_current(self, target_name: str | None = None) -> None:
        if target_name is not None:
            self._target_name = target_name
        if self._view is not None:
            self._view.set_current(self.is_current())
            self._view.set_busy(self._operations.is_busy)

    def operation_busy_changed(self, busy: bool) -> None:
        if self._view is not None:
            self._view.set_busy(busy)
            self._view.set_current(self.is_current())
        if not busy and self._clear_pending:
            self.clear_selection()

    def select(self, request: InterfaceSelectionRequest) -> None:
        report, source = self._report, self._source
        if report is None or source is None or self._operations.is_busy:
            return
        self.refresh_current()
        model = next((m for m in report.models if m.rank == request.rank), None)
        if (
            not self.is_current()
            or model is None
            or not any(s is request.score for s in model.scores)
        ):
            self._presenter.present_notice(
                Notice(
                    "stale_interface",
                    "This interface snapshot is no longer associated with the active model. Analyze the current target again.",
                )
            )
            return
        if not self._viewer.name_exists(model.object_name):
            if self._view is not None:
                self._view.set_current(False)
            self._presenter.present_notice(
                Notice(
                    "missing_interface_object",
                    "The analyzed viewer object no longer exists.",
                )
            )
            return
        score = request.score
        indices = tuple(
            sorted(
                set(
                    (() if request.side == "b" else score.support_a)
                    + (() if request.side == "a" else score.support_b)
                )
            )
        )
        if not indices:
            self._presenter.present_notice(
                Notice("empty_interface", "This score has no selectable contributors.")
            )
            return
        lease = self._operations.begin(
            "analysis",
            title="FoldQC – Interface selection",
            label="Selecting interface contributors…",
        )
        if lease is None:
            return
        try:
            # Recheck the canonical token mapping before touching the viewer.
            self._viewer.inspect_tokens(model.object_name, model.token_map)
            self._viewer.update_object_token_selection(
                INTERFACE_SELECTION,
                (ObjectTokenSelection(model.object_name, model.token_map, indices),),
            )
            self._owns_selection = True
        except Exception as exc:
            self._presenter.present_notice(
                Notice("interface_selection_failed", str(exc), severity="error")
            )
        finally:
            self._operations.finish(lease)
            self.refresh_current()

    def clear_selection(self) -> None:
        if not self._owns_selection:
            return
        if self._operations.is_busy:
            self._clear_pending = True
            return
        lease = self._operations.begin(
            "analysis",
            title="FoldQC – Interface selection",
            label="Clearing interface selection…",
        )
        if lease is None:
            self._clear_pending = True
            return
        self._clear_pending = False
        try:
            self._viewer.delete_names((INTERFACE_SELECTION,))
            self._owns_selection = False
        except Exception as exc:
            self._presenter.present_notice(
                Notice("interface_clear_failed", str(exc), severity="error")
            )
        finally:
            self._operations.finish(lease)

    def export(self, path: str) -> None:
        report = self._report
        if report is None:
            return
        lease = self._operations.begin(
            "analysis",
            title="FoldQC – Interface export",
            label="Exporting interface report…",
        )
        if lease is None:
            return
        try:
            write_interface_report(path, report)
        except Exception as exc:
            self._presenter.present_notice(
                Notice("interface_export_failed", str(exc), severity="error")
            )
        finally:
            self._operations.finish(lease)

    def window_closed(self) -> None:
        self.clear_selection()
        self._view = None
        self._report = None
        self._source = None
