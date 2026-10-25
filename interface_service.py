"""Report snapshot ownership and validated interface viewer/export interactions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Event
from weakref import ReferenceType, ref

from .analysis import ResolvedAnalysis
from .ensemble import EnsembleState
from .export import (
    PreparedContactCsv,
    interface_contact_rows,
    prepare_interface_contact_csv,
    write_interface_report,
)
from .gui_services import (
    JobRunner,
    ObjectTokenSelection,
    OperationCoordinatorPort,
    OperationLease,
    ViewerPort,
)
from .gui_state import PluginState
from .interface_analysis import (
    InterfaceContactExportRequest,
    InterfaceModelResult,
    InterfaceProjectionSelection,
    InterfaceReport,
    InterfaceReportObserver,
    InterfaceReportViewPort,
    InterfaceSelectionRequest,
)
from .interface_compute import iter_interface_contacts
from .interface_projection import projection_score, projection_unavailable_reason
from .loader_models import PredictionFiles
from .model_state import ModelState
from .presentation import Notice, PlotSelectionPort, PresentationPort
from .structure_index import StructureIndex

INTERFACE_SELECTION = "foldqc_interface"


@dataclass(frozen=True)
class _MemberSource:
    rank: int
    model: ReferenceType[ModelState]
    version: int
    object_name: str
    index: ReferenceType[StructureIndex]


@dataclass(frozen=True)
class _ReportSource:
    prediction: ReferenceType[PredictionFiles]
    ensemble: ReferenceType[EnsembleState] | None
    target_name: str
    single: bool
    members: tuple[_MemberSource, ...]


@dataclass(frozen=True)
class _PendingContactExport:
    lease: OperationLease
    request: InterfaceContactExportRequest
    report: InterfaceReport
    index: StructureIndex
    cancelled: Event


@dataclass(frozen=True)
class _ProjectionPlotSelection:
    service: ReferenceType[InterfaceReportService]
    selection: InterfaceProjectionSelection
    indices: frozenset[int]

    def select(self, token_indices: tuple[int, ...]) -> bool:
        service = self.service()
        return service is not None and service.select_projection_tokens(
            self.selection, tuple(i for i in token_indices if i in self.indices)
        )

    def clear(self) -> bool:
        service = self.service()
        if (
            service is None
            or service.projection_model(self.selection) is None
            or service._operations.is_busy
        ):
            return False
        service.clear_selection()
        return True


class InterfaceReportService:
    def __init__(
        self,
        state: PluginState,
        viewer: ViewerPort,
        presenter: PresentationPort,
        operations: OperationCoordinatorPort,
        job_runner: JobRunner,
    ) -> None:
        self._state = state
        self._viewer = viewer
        self._presenter = presenter
        self._operations = operations
        self._job_runner = job_runner
        self._pending_contacts: _PendingContactExport | None = None
        self._report: InterfaceReport | None = None
        self._source: _ReportSource | None = None
        self._view: InterfaceReportViewPort | None = None
        self._owns_selection = False
        self._clear_pending = False
        self._target_name: str | None = None
        self._generation = 0

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
                    m.rank,
                    ref(m.model_state),
                    m.expected_version,
                    m.obj_name,
                    ref(m.model_state.structure_index),
                )
                for m in source.members
            ),
        )
        report = replace(report, generation=self._generation + 1)
        new_view = self._presenter.show_interface_report(report, observer)
        self._generation = report.generation
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
            and model.structure_index is m.index()
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
        self._generation += 1
        self._cancel_contact_export()
        self.clear_selection()
        self._view = None
        self._report = None
        self._source = None

    def projection_model(
        self, selection: InterfaceProjectionSelection
    ) -> InterfaceModelResult | None:
        report = self._report
        if (
            report is None
            or selection.generation != self._generation
            or not self.is_current()
        ):
            return None
        model = next((m for m in report.models if m.rank == selection.rank), None)
        canonical = self._state.model_states.get(selection.rank)
        if (
            model is None
            or canonical is None
            or model.token_map is not canonical.token_map
        ):
            return None
        try:
            score = projection_score(model, selection)
            if projection_unavailable_reason(score, selection.kind):
                return None
        except ValueError:
            return None
        return model

    def projection_selection_port(
        self, selection: InterfaceProjectionSelection, indices: tuple[int, ...]
    ) -> PlotSelectionPort:
        return _ProjectionPlotSelection(ref(self), selection, frozenset(indices))

    def select_projection_tokens(
        self, selection: InterfaceProjectionSelection, indices: tuple[int, ...]
    ) -> bool:
        model = self.projection_model(selection)
        if model is None or self._operations.is_busy or not indices:
            return False
        chains = (
            (selection.chain_a,) + selection.partners
            if selection.side == "both"
            else (selection.chain_a,)
            if selection.side == "a"
            else selection.partners
        )
        eligible = frozenset(model.protein_indices)
        if any(
            i not in eligible or model.token_map[i].chain_id not in chains
            for i in indices
        ):
            return False
        lease = self._operations.begin(
            "analysis",
            title="FoldQC – Interface residues",
            label="Selecting interface residues…",
        )
        if lease is None:
            return False
        try:
            self._viewer.inspect_tokens(model.object_name, model.token_map)
            if self.projection_model(selection) is not model:
                return False
            self._viewer.update_object_token_selection(
                INTERFACE_SELECTION,
                (ObjectTokenSelection(model.object_name, model.token_map, indices),),
            )
            self._owns_selection = True
            return True
        except Exception as exc:
            self._presenter.present_notice(
                Notice("interface_selection_failed", str(exc), severity="error")
            )
            return False
        finally:
            self._operations.finish(lease)

    def _contact_index(
        self, request: InterfaceContactExportRequest
    ) -> StructureIndex | None:
        report, source = self._report, self._source
        if report is None or source is None or not self.is_current():
            return None
        model = next((m for m in report.models if m.rank == request.rank), None)
        member = next((m for m in source.members if m.rank == request.rank), None)
        canonical = None if member is None else member.model()
        if (
            model is None
            or canonical is None
            or canonical.token_map is not model.token_map
            or not any(d is request.diagnostic for d in model.contacts)
            or request.diagnostic.pair_count is None
        ):
            return None
        return canonical.structure_index

    def export_contacts(self, request: InterfaceContactExportRequest) -> None:
        if self._operations.is_busy:
            return
        index = self._contact_index(request)
        report = self._report
        if index is None or report is None:
            self.refresh_current()
            self._presenter.present_notice(
                Notice(
                    "stale_interface_contacts",
                    "Contact export requires available geometry in the current snapshot. Analyze the current target again.",
                )
            )
            return
        model = next(m for m in report.models if m.rank == request.rank)
        cancelled = Event()
        lease = self._operations.begin(
            "analysis",
            title="FoldQC – Contact export",
            label="Exporting geometric contacts…",
            cancellable=True,
            on_cancel=self._cancel_contact_export,
        )
        if lease is None:
            return
        self._pending_contacts = _PendingContactExport(
            lease, request, report, index, cancelled
        )

        def task(progress: Callable[[str], None]) -> PreparedContactCsv:
            def check() -> None:
                if cancelled.is_set():
                    raise InterruptedError("Contact export cancelled.")

            progress("Writing geometric contact pairs…")
            pairs = iter_interface_contacts(index, request.diagnostic, check)
            rows = interface_contact_rows(report, model, request.diagnostic, pairs)
            return prepare_interface_contact_csv(request.path, rows, check)

        try:
            handle = self._job_runner.submit(
                lease.request_id,
                task,
                self._contact_progress,
                self._contact_result,
                self._contact_error,
            )
            self._operations.attach(lease, handle)
        except Exception as exc:
            self._contact_error(lease.request_id, exc)

    def _cancel_contact_export(self) -> None:
        pending, self._pending_contacts = self._pending_contacts, None
        if pending is not None:
            pending.cancelled.set()
            if self._operations.is_current(pending.lease):
                self._operations.abandon()

    def _contact_progress(self, request_id: int, label: str) -> None:
        pending = self._pending_contacts
        if pending is not None and pending.lease.request_id == request_id:
            self._operations.update(pending.lease, label)

    def _contact_is_current(self, pending: _PendingContactExport) -> bool:
        return (
            self._report is pending.report
            and not pending.cancelled.is_set()
            and self._contact_index(pending.request) is pending.index
        )

    def _contact_result(self, request_id: int, value: object) -> None:
        pending = self._pending_contacts
        if (
            pending is None
            or pending.lease.request_id != request_id
            or not self._operations.is_current(pending.lease)
        ):
            self._job_runner.dispose(value)
            return
        self._pending_contacts = None
        try:
            if not self._contact_is_current(pending):
                return
            if not isinstance(value, PreparedContactCsv):
                raise ValueError("Unexpected contact export result.")
            value.commit(pending.request.path)
        except Exception as exc:
            self._presenter.present_notice(
                Notice("interface_contact_export_failed", str(exc), severity="error")
            )
        finally:
            self._job_runner.dispose(value)
            self._operations.finish(pending.lease)
            self.refresh_current()

    def _contact_error(self, request_id: int, failure: object) -> None:
        pending = self._pending_contacts
        if (
            pending is None
            or pending.lease.request_id != request_id
            or not self._operations.is_current(pending.lease)
        ):
            return
        self._pending_contacts = None
        current = self._contact_is_current(pending)
        self._operations.finish(pending.lease)
        self.refresh_current()
        if current:
            self._presenter.present_notice(
                Notice(
                    "interface_contact_export_failed",
                    str(getattr(failure, "message", failure)),
                    severity="error",
                )
            )
