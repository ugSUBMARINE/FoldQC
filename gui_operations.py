"""Qt-independent ownership of serialized GUI operation lifetimes."""

from __future__ import annotations

from collections.abc import Callable

from .gui_services import (
    BusyViewState,
    DialogViewPort,
    JobHandlePort,
    OperationKind,
    OperationLease,
)
from .presentation import PresentationPort, ProgressRequest


class GuiOperationCoordinator:
    """Own the one active worker/transaction lease and its progress UI."""

    def __init__(
        self,
        presenter: PresentationPort,
        view: DialogViewPort,
    ) -> None:
        self._presenter = presenter
        self._view = view
        self._generation: int = 0
        self._active: OperationLease | None = None
        self._handle: JobHandlePort | None = None

    @property
    def is_busy(self) -> bool:
        return self._active is not None

    @property
    def active(self) -> OperationLease | None:
        return self._active

    @staticmethod
    def _progress_id(lease: OperationLease) -> str:
        return f"foldqc-{lease.kind}-{lease.request_id}"

    def begin(
        self,
        kind: OperationKind,
        *,
        title: str,
        label: str,
        delay_ms: int = 300,
        cancellable: bool = False,
        on_cancel: Callable[[], None] | None = None,
    ) -> OperationLease | None:
        if self._active is not None:
            return None
        self._generation += 1
        lease = OperationLease(self._generation, kind)
        self._active = lease
        self._handle = None
        self._view.set_busy(BusyViewState(True, False))
        cancel_callback = None
        if cancellable:

            def cancel_callback() -> None:
                try:
                    if on_cancel is not None:
                        on_cancel()
                finally:
                    self.abandon()

        self._presenter.start_progress(
            ProgressRequest(
                self._progress_id(lease),
                title,
                label,
                delay_ms=delay_ms,
                cancellable=cancellable,
            ),
            cancel_callback,
        )
        return lease

    def attach(self, lease: OperationLease, handle: JobHandlePort) -> bool:
        if not self.is_current(lease):
            handle.abandon()
            return False
        self._handle = handle
        return True

    def is_current(self, lease: OperationLease) -> bool:
        return self._active == lease

    def update(self, lease: OperationLease, label: str) -> None:
        if self.is_current(lease):
            self._presenter.update_progress(self._progress_id(lease), label)

    def finish(self, lease: OperationLease) -> bool:
        if not self.is_current(lease):
            return False
        self._presenter.finish_progress(self._progress_id(lease))
        self._active = None
        self._handle = None
        self._view.set_busy(BusyViewState(False, True))
        return True

    def abandon(self) -> None:
        lease = self._active
        if lease is None:
            return
        handle = self._handle
        if handle is not None:
            handle.abandon()
        self._generation += 1
        self._presenter.finish_progress(self._progress_id(lease))
        self._active = None
        self._handle = None
        self._view.set_busy(BusyViewState(False, True))
