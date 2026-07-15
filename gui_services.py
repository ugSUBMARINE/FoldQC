"""Explicit ports shared by FoldQC GUI application services.

Protocols live here so application logic can be tested without importing Qt or
PyMOL. Concrete adapters are composed by :mod:`gui`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np

from .analysis import AnalysisProblem
from .dependencies import DependencyKey
from .mol_viewer import ObjectPaintMapping, PaintBatchResult, PaintTarget
from .token_map import TokenMap

T = TypeVar("T")


@dataclass(frozen=True)
class AtomVisualSnapshot:
    """Restorable B-factor and color state for one viewer object."""

    obj_name: str
    atom_indices: tuple[int, ...]
    b_factors: np.ndarray
    color_indices: np.ndarray

    def __post_init__(self) -> None:
        b_factors = np.ascontiguousarray(self.b_factors, dtype=np.float32)
        colors = np.ascontiguousarray(self.color_indices, dtype=np.int32)
        if b_factors.shape != (len(self.atom_indices),):
            raise ValueError("B-factor snapshot does not match atom indices.")
        if colors.shape != (len(self.atom_indices),):
            raise ValueError("Color snapshot does not match atom indices.")
        b_factors.setflags(write=False)
        colors.setflags(write=False)
        object.__setattr__(self, "b_factors", b_factors)
        object.__setattr__(self, "color_indices", colors)


@dataclass(frozen=True)
class ManagedColorbar:
    """Recreatable specification for the colorbar owned by FoldQC."""

    palette: str
    reverse_palette: bool
    vmin: float
    vmax: float
    object_names: tuple[str, ...]


@runtime_checkable
class ViewerPort(Protocol):
    def snapshot_atom_visuals(self, obj_name: str) -> AtomVisualSnapshot: ...

    def restore_atom_visuals(self, snapshot: AtomVisualSnapshot) -> None: ...

    def paint_continuous(
        self,
        targets: Sequence[PaintTarget],
        *,
        palette: str,
        reverse_palette: bool,
        vmin: float | None,
        vmax: float | None,
        rebuild: bool = False,
    ) -> PaintBatchResult: ...

    def paint_categorical(
        self, targets: Sequence[PaintTarget], *, rebuild: bool = False
    ) -> PaintBatchResult: ...

    def paint_plddt_classes(
        self, targets: Sequence[PaintTarget], *, rebuild: bool = False
    ) -> PaintBatchResult: ...

    def get_managed_colorbar(self) -> ManagedColorbar | None: ...

    def replace_managed_colorbar(self, state: ManagedColorbar | None) -> None: ...

    def rebuild(self) -> None: ...

    def run_suspended(self, operation: Callable[[], T]) -> T: ...

    def ensure_paint_mapping(
        self,
        obj_name: str,
        token_map: TokenMap,
        existing: ObjectPaintMapping | None,
    ) -> tuple[ObjectPaintMapping, bool]: ...


@runtime_checkable
class JobRunner(Protocol):
    def submit(
        self,
        request_id: int,
        task: Callable[[Callable[[str], None]], Any],
        on_progress: Callable[[int, str], None],
        on_result: Callable[[int, object], None],
        on_error: Callable[[int, object], None],
    ) -> object: ...

    def dispose(self, value: object) -> None: ...


@runtime_checkable
class PresentationPort(Protocol):
    def present_problem(self, problem: AnalysisProblem) -> None: ...

    def confirm(self, title: str, message: str) -> bool: ...

    def set_progress(self, label: str | None) -> None: ...

    def set_window_title(self, title: str) -> None: ...

    def show_statistics(self, text: str) -> None: ...

    def show_plot(self, figure: object, title: str) -> None: ...


@runtime_checkable
class DependencyService(Protocol):
    def ensure(
        self, keys: tuple[DependencyKey, ...], *, feature_label: str
    ) -> bool: ...


LifecycleOutcomeKind = Literal["committed", "cancelled", "stale"]


@dataclass(frozen=True)
class LifecycleOutcome:
    kind: LifecycleOutcomeKind
    refresh_objects: bool = False
    refresh_metrics: bool = False
    save_session: bool = False
