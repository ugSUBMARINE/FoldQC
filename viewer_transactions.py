"""Compensating viewer transactions independent of Qt and PyMOL imports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .gui_services import (
    AtomVisualSnapshot,
    ManagedColorbar,
    PaintBatchResult,
    PaintTarget,
    ViewerPort,
)


class ViewerTransactionError(RuntimeError):
    """Viewer mutation failed; rollback details are retained when applicable."""

    def __init__(
        self, message: str, *, rollback_error: Exception | None = None
    ) -> None:
        if rollback_error is not None:
            message = f"{message}\nRollback also failed: {rollback_error}"
        super().__init__(message)
        self.rollback_error = rollback_error


@dataclass(frozen=True)
class ColorbarChange:
    mode: str
    palette: str = ""
    reverse_palette: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"replace", "remove", "keep"}:
            raise ValueError(f"Unknown colorbar transaction mode: {self.mode!r}.")
        if self.mode == "replace" and not self.palette:
            raise ValueError("Replacement colorbars require a palette.")


@dataclass
class PaintTransaction:
    """Snapshot, mutate, and atomically commit one FoldQC paint operation."""

    viewer: ViewerPort
    targets: tuple[PaintTarget, ...]
    colorbar: ColorbarChange
    _snapshots: tuple[AtomVisualSnapshot, ...] = field(default=(), init=False)
    _previous_colorbar: ManagedColorbar | None = field(default=None, init=False)
    _colorbar_mutated: bool = field(default=False, init=False)
    _committed: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.targets = tuple(self.targets)
        if not self.targets:
            raise ValueError("PaintTransaction requires at least one target.")

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(target.obj_name for target in self.targets))

    def execute(
        self,
        paint: Callable[[], PaintBatchResult],
    ) -> PaintBatchResult:
        if self._closed:
            raise RuntimeError("PaintTransaction was already closed.")

        # Read-only viewer inspection happens before update suspension. PyMOL's
        # object-model access is safer outside nested command-state changes.
        self._snapshots = tuple(
            self.viewer.snapshot_atom_visuals(name) for name in self.object_names
        )
        self._previous_colorbar = self.viewer.get_managed_colorbar()

        def operation() -> PaintBatchResult:
            result = paint()
            self._apply_colorbar(result)
            return result

        try:
            result = self.viewer.run_suspended(operation)
            self._committed = True
            self._closed = True
            return result
        except Exception as exc:
            rollback_error = None
            try:
                self.rollback()
            except Exception as rollback_exc:  # pragma: no cover - combined path
                rollback_error = rollback_exc
            raise ViewerTransactionError(
                str(exc) or type(exc).__name__, rollback_error=rollback_error
            ) from exc

    def _apply_colorbar(self, result: PaintBatchResult) -> None:
        if self.colorbar.mode == "keep":
            return
        state = None
        if self.colorbar.mode == "replace":
            state = ManagedColorbar(
                self.colorbar.palette,
                self.colorbar.reverse_palette,
                result.vmin,
                result.vmax,
                self.object_names,
            )
        # Mark first: replacement may delete the old ramp before raising.
        self._colorbar_mutated = True
        self.viewer.replace_managed_colorbar(state)

    def rollback(self) -> None:
        if self._closed and self._committed:
            return
        failures: list[Exception] = []
        if self._colorbar_mutated:
            try:
                self.viewer.replace_managed_colorbar(self._previous_colorbar)
            except Exception as exc:  # pragma: no cover - defensive
                failures.append(exc)
        for snapshot in reversed(self._snapshots):
            try:
                self.viewer.restore_atom_visuals(snapshot)
            except Exception as exc:
                failures.append(exc)
        try:
            self.viewer.rebuild()
        except Exception as exc:  # pragma: no cover - defensive
            failures.append(exc)
        self._closed = True
        if failures:
            raise RuntimeError("; ".join(str(item) for item in failures))
