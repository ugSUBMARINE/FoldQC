"""Atomic lazy prediction-data acquisition for immutable analysis actions."""

from __future__ import annotations

import logging

from .analysis import DataLoadPlan, DeferredAnalysisAction
from .gui_services import (
    DataAcquisitionOutcome,
    DataAcquisitionStatus,
    DataLoadObserver,
    GuiScheduler,
    JobRunner,
    OperationCoordinatorPort,
    OperationLease,
)
from .gui_state import PluginState
from .lifecycle_support import APP_TITLE, DataLoadBatchResult, _load_data_batch
from .presentation import Notice, PresentationPort

logger = logging.getLogger(__name__)


class DataAcquisitionService:
    """Load one deduplicated plan and commit every model merge atomically."""

    def __init__(
        self,
        state: PluginState,
        presenter: PresentationPort,
        scheduler: GuiScheduler,
        job_runner: JobRunner,
        operations: OperationCoordinatorPort,
    ) -> None:
        self._state = state
        self._presenter = presenter
        self._scheduler = scheduler
        self._job_runner = job_runner
        self._operations = operations
        self._pending: (
            tuple[
                OperationLease,
                DeferredAnalysisAction,
                DataLoadPlan,
                DataLoadObserver,
            ]
            | None
        ) = None

    @property
    def is_loading(self) -> bool:
        return self._pending is not None

    def acquire(
        self,
        action: DeferredAnalysisAction,
        plan: DataLoadPlan,
        observer: DataLoadObserver,
    ) -> bool:
        if plan.is_empty:
            return False
        first = plan.requirements[0]
        arrays = " and ".join(first.phase_arrays) or "metric data"
        lease = self._operations.begin(
            "data",
            title=f"{APP_TITLE} – Loading",
            label=f"Loading {arrays} for {first.model_label}…",
            cancellable=True,
            on_cancel=self._cancel,
        )
        if lease is None:
            return False
        self._pending = (lease, action, plan, observer)
        prediction_files = plan.analysis.prediction_files
        requirements = plan.requirements
        handle = self._job_runner.submit(
            lease.request_id,
            lambda report: _load_data_batch(prediction_files, requirements, report),
            self._on_progress,
            self._on_result,
            self._on_error,
        )
        self._operations.attach(lease, handle)
        return True

    def _cancel(self) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        lease, action, _plan, observer = pending
        outcome = DataAcquisitionOutcome(lease, action, "cancelled")
        self._scheduler.call_soon(lambda: observer.data_acquisition_finished(outcome))

    def _matching_pending(
        self, request_id: int
    ) -> (
        tuple[
            OperationLease,
            DeferredAnalysisAction,
            DataLoadPlan,
            DataLoadObserver,
        ]
        | None
    ):
        pending = self._pending
        if pending is None:
            return None
        lease = pending[0]
        if lease.request_id != request_id or not self._operations.is_current(lease):
            return None
        return pending

    def _on_progress(self, request_id: int, label: str) -> None:
        pending = self._matching_pending(request_id)
        if pending is not None:
            self._operations.update(pending[0], label)

    def _result_is_current(
        self, plan: DataLoadPlan, result: DataLoadBatchResult
    ) -> bool:
        if result.pred_files is not self._state.pred_files:
            return False
        if result.pred_files is not plan.analysis.prediction_files:
            return False
        for requirement, _data in result.loaded:
            if (
                self._state.model_states.get(requirement.rank)
                is not requirement.model_state
            ):
                return False
            if requirement.model_state.version != requirement.expected_version:
                return False
            expected_ensemble = requirement.expected_ensemble
            if expected_ensemble is not None:
                if self._state.ensemble is not expected_ensemble:
                    return False
                if requirement.rank not in expected_ensemble.ranks:
                    return False
        return True

    @staticmethod
    def _validate_loaded_result(result: DataLoadBatchResult) -> None:
        fields = (
            ("pae", "pae", "PAE"),
            ("pde", "pde", "PDE"),
            ("contact_probs", "contact_probs", "interaction probabilities"),
            ("plddt", "token_plddt", "pLDDT"),
        )
        for requirement, data in result.loaded:
            missing = [
                label
                for capability, attribute, label in fields
                if capability in requirement.capabilities
                and getattr(data, attribute, None) is None
            ]
            if missing:
                raise ValueError(
                    f"{requirement.model_label} did not provide required prediction "
                    f"data: {', '.join(missing)}"
                )

    def _notify(
        self,
        pending: tuple[
            OperationLease,
            DeferredAnalysisAction,
            DataLoadPlan,
            DataLoadObserver,
        ],
        status: DataAcquisitionStatus,
        notice: Notice | None = None,
    ) -> None:
        lease, action, _plan, observer = pending
        self._pending = None
        self._operations.finish(lease)
        outcome = DataAcquisitionOutcome(lease, action, status, notice)
        self._scheduler.call_soon(lambda: observer.data_acquisition_finished(outcome))

    def _on_result(self, request_id: int, value: object) -> None:
        pending = self._matching_pending(request_id)
        if pending is None:
            self._job_runner.dispose(value)
            return
        if not isinstance(value, DataLoadBatchResult):
            self._notify(
                pending,
                "failed",
                Notice(
                    "lazy_result_type",
                    "The lazy-data worker returned an unexpected result.",
                    severity="error",
                    title=f"{APP_TITLE} - error",
                ),
            )
            return
        plan = pending[2]
        if not self._result_is_current(plan, value):
            self._job_runner.dispose(value)
            self._notify(pending, "stale")
            return
        snapshots = []
        try:
            self._validate_loaded_result(value)
            for requirement, data in value.loaded:
                requirement.model_state.validate_merge(data)
            snapshots = [
                (requirement.model_state, requirement.model_state.snapshot())
                for requirement, _data in value.loaded
            ]
            for requirement, data in value.loaded:
                requirement.model_state.merge_data(data)
        except Exception as exc:
            for model_state, snapshot in reversed(snapshots):
                model_state.restore(snapshot)
            self._notify(
                pending,
                "failed",
                Notice(
                    "lazy_merge_failed",
                    str(exc),
                    severity="error",
                    title=f"{APP_TITLE} - error",
                ),
            )
            return
        self._operations.update(pending[0], "Preparing requested action…")
        self._notify(pending, "ready")

    def _on_error(self, request_id: int, failure: object) -> None:
        pending = self._matching_pending(request_id)
        if pending is None:
            return
        message = str(getattr(failure, "message", failure))
        traceback_text = getattr(failure, "traceback_text", "")
        if traceback_text:
            logger.error("Background lazy-data load failed:\n%s", traceback_text)
        self._notify(
            pending,
            "failed",
            Notice(
                "lazy_load_failed",
                message,
                severity="error",
                title=f"{APP_TITLE} - error",
            ),
        )

    def close(self) -> None:
        pending = self._pending
        self._pending = None
        if pending is not None and self._operations.is_current(pending[0]):
            self._operations.abandon()
