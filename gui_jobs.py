"""Qt background jobs used by GUI workflow coordinators."""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .compat import QtCore

Signal = getattr(QtCore, "Signal", None)
if Signal is None:  # PyQt exposes pyqtSignal instead of Signal.
    Signal = QtCore.pyqtSignal

ProgressReporter = Callable[[str], None]
JobTask = Callable[[ProgressReporter], Any]


def _cleanup_temporary_extraction(value: object) -> None:
    """Explicitly clean archive ownership carried by a discarded job result."""
    nested = getattr(value, "pred_files", None)
    if nested is not None:
        _cleanup_temporary_extraction(nested)
    temporary = getattr(value, "_temporary_directory", None)
    cleanup = getattr(temporary, "cleanup", None)
    if callable(cleanup):
        cleanup()


@dataclass(frozen=True)
class JobFailure:
    """Serializable background-job failure details."""

    message: str
    traceback_text: str


class JobHandle:
    """Logical ownership handle for a running background job."""

    def __init__(self) -> None:
        self._abandoned = threading.Event()

    def abandon(self) -> None:
        """Suppress further progress and result delivery without terminating work."""
        self._abandoned.set()

    @property
    def is_abandoned(self) -> bool:
        return self._abandoned.is_set()


class _JobSignals(QtCore.QObject):
    progress = Signal(int, str)
    result = Signal(int, object)
    error = Signal(int, object)


class _JobRunnable(QtCore.QRunnable):
    def __init__(self, request_id: int, task: JobTask, handle: JobHandle) -> None:
        super().__init__()
        self.request_id = request_id
        self.task = task
        self.handle = handle
        self.signals = _JobSignals()

    def run(self) -> None:
        def report_phase(label: str) -> None:
            if not self.handle.is_abandoned:
                self.signals.progress.emit(self.request_id, str(label))

        try:
            result = self.task(report_phase)
        except Exception as exc:
            if not self.handle.is_abandoned:
                message = str(exc) or type(exc).__name__
                self.signals.error.emit(
                    self.request_id,
                    JobFailure(message, traceback.format_exc()),
                )
            return

        if self.handle.is_abandoned:
            # Release archive-backed result objects in the worker thread.
            _cleanup_temporary_extraction(result)
            del result
            return
        self.signals.result.emit(self.request_id, result)


class _DiscardRunnable(QtCore.QRunnable):
    def __init__(self, value: object) -> None:
        super().__init__()
        self._value = value

    def run(self) -> None:
        # Dropping the final reference here keeps temporary extraction cleanup
        # away from the Qt GUI thread.
        _cleanup_temporary_extraction(self._value)
        self._value = None


_POOL = QtCore.QThreadPool()
_POOL.setMaxThreadCount(1)


class QtJobRunner:
    """Submit one-shot callables to FoldQC's private single-threaded pool."""

    def submit(
        self,
        request_id: int,
        task: JobTask,
        on_progress: Callable[[int, str], None],
        on_result: Callable[[int, object], None],
        on_error: Callable[[int, JobFailure], None],
    ) -> JobHandle:
        handle = JobHandle()
        runnable = _JobRunnable(request_id, task, handle)
        runnable.signals.progress.connect(on_progress)
        runnable.signals.result.connect(on_result)
        runnable.signals.error.connect(on_error)
        _POOL.start(runnable)
        return handle

    def dispose(self, value: object) -> None:
        """Release an unused result on the background pool."""
        _POOL.start(_DiscardRunnable(value))
