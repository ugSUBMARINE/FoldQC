"""Explicit ports shared by FoldQC GUI application services.

Protocols live here so application logic can be tested without importing Qt or
PyMOL. Concrete adapters are composed by :mod:`gui`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np

from .dependencies import DependencyKey
from .token_map import TokenMap, TokenOverlapSummary

if TYPE_CHECKING:
    from .analysis import DeferredAnalysisAction
    from .presentation import Notice
    from .session import SessionState

T = TypeVar("T")


@dataclass(frozen=True)
class ObjectPaintMapping:
    """Stable mapping from one viewer object's atoms to prediction tokens."""

    obj_name: str
    atom_index_fingerprint: tuple[int, ...]
    atom_token_indices: np.ndarray
    atom_count: int
    max_atom_index: int
    overlap: TokenOverlapSummary


@dataclass(frozen=True)
class ObjectTokenInspection:
    """Paint metadata and representative coordinates from one object snapshot."""

    paint_mapping: ObjectPaintMapping
    representative_coords: np.ndarray


@dataclass(frozen=True)
class PaintTarget:
    """One viewer object and per-token array participating in a paint."""

    obj_name: str
    token_map: TokenMap
    values: np.ndarray
    mapping: ObjectPaintMapping | None = None


@dataclass(frozen=True)
class ObjectTokenSelection:
    """FoldQC-owned token indices selected independently in one viewer object."""

    obj_name: str
    token_map: TokenMap
    token_indices: tuple[int, ...]


@dataclass(frozen=True)
class StatisticsSelectionTarget:
    """One immutable metric array available for statistics thresholding."""

    obj_name: str
    token_map: TokenMap
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(np.asarray(self.values, dtype=np.float32))
        if values.shape != (len(self.token_map),):
            raise ValueError(
                f"Metric values for {self.obj_name!r} do not match its token map."
            )
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class PaintBatchResult:
    """Resolved range and mappings from a viewer paint operation."""

    vmin: float
    vmax: float
    mappings: tuple[ObjectPaintMapping, ...]


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
        self, path: str | Path, obj_name: str, *, zoom: bool = True
    ) -> bool: ...

    def load_structure_object_if_missing(
        self, path: str | Path, obj_name: str
    ) -> bool: ...

    def delete_names(self, names: Sequence[str]) -> None: ...

    def name_exists(self, name: str) -> bool: ...

    def group_members(self, group_name: str) -> tuple[str, ...]: ...

    def add_to_group(self, group_name: str, names: Sequence[str]) -> None: ...

    def remove_from_group(self, group_name: str, names: Sequence[str]) -> None: ...

    def inspect_tokens(
        self, obj_name: str, token_map: TokenMap
    ) -> ObjectTokenInspection: ...

    def transform(
        self, obj_name: str, rotation: np.ndarray, translation: np.ndarray
    ) -> None: ...

    def update_token_selection(
        self,
        selection_name: str,
        token_indices: Sequence[int],
        object_token_maps: Sequence[tuple[str, TokenMap]],
    ) -> None: ...

    def update_object_token_selection(
        self,
        selection_name: str,
        targets: Sequence[ObjectTokenSelection],
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

    def capture_paint_mappings(
        self,
    ) -> dict[tuple[str, str], ObjectPaintMapping]: ...

    def restore_paint_mappings(
        self, mappings: dict[tuple[str, str], ObjectPaintMapping]
    ) -> None: ...

    def clear_paint_mappings(self) -> None: ...


@runtime_checkable
class JobHandlePort(Protocol):
    def abandon(self) -> None: ...


@runtime_checkable
class JobRunner(Protocol):
    def submit(
        self,
        request_id: int,
        task: Callable[[Callable[[str], None]], object],
        on_progress: Callable[[int, str], None],
        on_result: Callable[[int, object], None],
        on_error: Callable[[int, object], None],
    ) -> JobHandlePort: ...

    def dispose(self, value: object) -> None: ...


@runtime_checkable
class DependencyService(Protocol):
    def ensure(
        self, keys: tuple[DependencyKey, ...], *, feature_label: str
    ) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class GuiScheduler(Protocol):
    """Minimal main-thread scheduling contract used by GUI-neutral services."""

    def call_soon(self, callback: Callable[[], None]) -> None: ...

    def call_later(self, delay_ms: int, callback: Callable[[], None]) -> None: ...


@runtime_checkable
class DialogViewPort(Protocol):
    """Deterministic widget renderer, separate from transient presentation."""

    def apply_context(self, state: ContextViewState) -> None: ...

    def apply_lifecycle(self, update: LifecycleUiUpdate) -> None: ...

    def set_busy(self, state: BusyViewState) -> None: ...

    def set_statistics_selection(self, state: StatisticsSelectionViewState) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class SessionPort(Protocol):
    def restore(self) -> SessionState: ...

    def record_recent_prediction(self, path: str | Path) -> tuple[str, ...]: ...

    def remove_recent_prediction(self, path: str | Path) -> tuple[str, ...]: ...

    def save_geometry(self) -> None: ...


@dataclass(frozen=True)
class ContextSelection:
    target_name: str = ""
    metric_key: str | None = None
    reference_selection: str = ""
    cutoff_text: str = ""


@dataclass(frozen=True)
class ModelChoice:
    rank: int
    label: str


@dataclass(frozen=True)
class TargetChoice:
    name: str
    kind: Literal["single", "ensemble_member", "ensemble_group", "viewer"]


@dataclass(frozen=True)
class BusyViewState:
    busy: bool
    prediction_controls_enabled: bool


OperationKind = Literal["prediction", "data", "ensemble", "model_switch"]


@dataclass(frozen=True)
class OperationLease:
    request_id: int
    kind: OperationKind


@runtime_checkable
class OperationCoordinatorPort(Protocol):
    @property
    def is_busy(self) -> bool: ...

    @property
    def active(self) -> OperationLease | None: ...

    def begin(
        self,
        kind: OperationKind,
        *,
        title: str,
        label: str,
        delay_ms: int = 300,
        cancellable: bool = False,
        on_cancel: Callable[[], None] | None = None,
    ) -> OperationLease | None: ...

    def attach(self, lease: OperationLease, handle: JobHandlePort) -> bool: ...

    def is_current(self, lease: OperationLease) -> bool: ...

    def update(self, lease: OperationLease, label: str) -> None: ...

    def finish(self, lease: OperationLease) -> bool: ...

    def abandon(self) -> None: ...


DataAcquisitionStatus = Literal["ready", "stale", "cancelled", "failed"]


@dataclass(frozen=True)
class DataAcquisitionOutcome:
    lease: OperationLease
    action: DeferredAnalysisAction
    status: DataAcquisitionStatus
    notice: Notice | None = None


@runtime_checkable
class DataLoadObserver(Protocol):
    def data_acquisition_finished(self, outcome: DataAcquisitionOutcome) -> None: ...


@runtime_checkable
class AnalysisSubmissionPort(Protocol):
    """Narrow action-submission boundary used by committed lifecycle work."""

    @property
    def ui_revision(self) -> int: ...

    def submit(self, action: DeferredAnalysisAction) -> None: ...


@dataclass(frozen=True)
class LifecycleUiUpdate:
    """Deterministic view changes emitted after a lifecycle commit."""

    selected_rank: int | None = None
    selected_target: str | None = None
    display_path: str | None = None
    recent_predictions: tuple[str, ...] | None = None
    refresh_context: bool = True
    model_choices: tuple[ModelChoice, ...] | None = None
    target_choices: tuple[TargetChoice, ...] | None = None


@dataclass(frozen=True)
class ContextViewState:
    """Complete contextual control state rendered by ``DialogViewPort``."""

    metric_availability: tuple[tuple[int, bool], ...] = ()
    metric_labels: tuple[tuple[int, str], ...] = ()
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
    ensemble_enabled: bool = False
    ensemble_tooltip: str = "Load a prediction with at least two models first."
    model_comparison_enabled: bool = False
    model_comparison_tooltip: str = "Load a prediction with at least two models first."
    model_choices: tuple[ModelChoice, ...] = ()
    target_choices: tuple[TargetChoice, ...] = ()
    selected_rank: int | None = None
    selected_target: str | None = None


@dataclass(frozen=True)
class StatisticsSelectionViewState:
    """Deterministic state for the statistics threshold-selection controls."""

    enabled: bool = False
    threshold: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    status_text: str = "Apply a metric coloring first."
