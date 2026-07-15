"""Explicit ports shared by FoldQC GUI application services.

Protocols live here so application logic can be tested without importing Qt or
PyMOL. Concrete adapters are composed by :mod:`gui`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np

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
    def object_names(self, additional_names: Sequence[str] = ()) -> list[str]: ...

    def ensure_structure_object(
        self, path: object, obj_name: str, *, zoom: bool = True
    ) -> bool: ...

    def load_structure_object_if_missing(self, path: object, obj_name: str) -> bool: ...

    def delete_names(self, names: Sequence[str]) -> None: ...

    def name_exists(self, name: str) -> bool: ...

    def group_members(self, group_name: str) -> tuple[str, ...]: ...

    def add_to_group(self, group_name: str, names: Sequence[str]) -> None: ...

    def remove_from_group(self, group_name: str, names: Sequence[str]) -> None: ...

    def inspect_tokens(self, obj_name: str, token_map: TokenMap) -> object: ...

    def transform(
        self, obj_name: str, rotation: np.ndarray, translation: np.ndarray
    ) -> None: ...

    def selection_token_indices(
        self, token_map: TokenMap, selection: str, *, obj_name: str
    ) -> list[int]: ...

    def tokens_within_distance(
        self,
        token_map: TokenMap,
        obj_name: str,
        reference_selection: str,
        cutoff: float,
    ) -> list[int]: ...

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
class DependencyService(Protocol):
    def ensure(
        self, keys: tuple[DependencyKey, ...], *, feature_label: str
    ) -> bool: ...


@runtime_checkable
class GuiScheduler(Protocol):
    """Minimal main-thread scheduling contract used by GUI-neutral services."""

    def call_soon(self, callback: Callable[[], None]) -> None: ...

    def call_later(self, delay_ms: int, callback: Callable[[], None]) -> None: ...


@runtime_checkable
class DialogViewPort(Protocol):
    """Deterministic widget renderer, separate from transient presentation."""

    @property
    def widgets(self) -> object: ...

    def select_combo_data(self, combo: object, value: object) -> bool: ...

    def combo_contains_text(self, combo: object, text: str) -> bool: ...

    def select_object(self, name: str) -> None: ...

    def select_model_rank(self, rank: int) -> bool: ...

    def select_property(self, key: str) -> None: ...

    def select_property_if_available(self, key: str) -> bool: ...

    def set_metric_available(self, row: int, available: bool) -> None: ...

    def metric_is_available(self, row: int) -> bool: ...

    def set_plot_availability(
        self, availability: tuple[tuple[str, bool, str], ...]
    ) -> None: ...

    def set_confidence_text(self, text: str) -> None: ...

    def set_preview_text(self, text: str) -> None: ...

    def apply_field_context(self, state: ContextViewState) -> None: ...

    def apply_context(self, state: ContextViewState) -> None: ...


@dataclass(frozen=True)
class LifecycleUiUpdate:
    """Deterministic view changes emitted after a lifecycle commit."""

    selected_rank: int | None = None
    selected_target: str | None = None
    refresh_context: bool = True
    save_session: bool = False


@dataclass(frozen=True)
class ContextViewState:
    """Complete contextual control state rendered by ``DialogViewPort``."""

    metric_availability: tuple[tuple[int, bool], ...] = ()
    plot_availability: tuple[tuple[str, bool, str], ...] = ()
    reference_label: str = "Reference:"
    reference_tooltip: str = ""
    reference_enabled: bool = True
    cutoff_label: str = "Cutoff (Å):"
    cutoff_tooltip: str = ""
    cutoff_enabled: bool = True
    confidence_text: str = ""
    preview_text: str = ""
    statistics_text: str | None = None
