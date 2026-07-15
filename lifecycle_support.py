"""Shared immutable lifecycle results and background job helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ensemble
from .analysis import DataLoadRequirement, DeferredAnalysisAction
from .gui_services import ObjectPaintMapping, ObjectTokenInspection
from .loader_models import (
    DataCapability,
    PredictionCandidate,
    PredictionData,
    PredictionDiscovery,
    PredictionFiles,
)
from .model_state import ModelState, ModelStateSnapshot

# These names form the intentionally private support surface consumed by the
# four lifecycle services. Keeping the imports here avoids each service
# depending on provider/job implementation details while leaving ownership in
# the service modules themselves.
__all__ = (
    "APP_TITLE",
    "VIEWER_NAME",
    "DataCapability",
    "DataLoadBatchResult",
    "DataLoadRequirement",
    "DeferredAnalysisAction",
    "EnsembleActivationTransaction",
    "InitialLoadResult",
    "ModelState",
    "ModelStoreSnapshot",
    "ModelSwitchResult",
    "Path",
    "PredictionFiles",
    "_discover_prediction",
    "_discovery_phase",
    "_load_data_batch",
    "_load_rank_data",
    "_prepare_ensemble_job",
    "_scan_and_load_initial_prediction",
    "_session_path_for_candidate",
    "ensemble",
    "np",
)

APP_TITLE = "FoldQC"
VIEWER_NAME = "PyMOL"

_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
ProgressReporter = Callable[[str], None]


@dataclass
class InitialLoadResult:
    """Provider files and initial lazy data prepared by a background job."""

    _pred_files: PredictionFiles | None
    model_state: ModelState
    display_path: Path

    @property
    def rank(self) -> int:
        return self.model_state.rank

    @property
    def pred_files(self) -> PredictionFiles:
        if self._pred_files is None:
            raise RuntimeError("InitialLoadResult ownership was already transferred.")
        return self._pred_files

    def take_prediction_files(self) -> PredictionFiles:
        files = self.pred_files
        self._pred_files = None
        return files

    def close(self) -> None:
        if self._pred_files is not None:
            self._pred_files.close()
            self._pred_files = None


@dataclass(frozen=True)
class ModelSwitchResult:
    """One ranked model prepared without touching Qt widgets or PyMOL."""

    pred_files: PredictionFiles
    model_state: ModelState

    @property
    def rank(self) -> int:
        return self.model_state.rank


@dataclass(frozen=True)
class DataLoadBatchResult:
    """Atomically committable lazy data returned by one worker task."""

    pred_files: PredictionFiles
    loaded: tuple[tuple[DataLoadRequirement, PredictionData], ...]


@dataclass(frozen=True)
class ModelStoreSnapshot:
    """Restorable model-store membership, contents, and active rank."""

    active_rank: int | None
    entries: tuple[tuple[int, ModelState, ModelStateSnapshot], ...]


@dataclass
class EnsembleActivationTransaction:
    """Main-thread state for an incrementally committed ensemble."""

    request_id: int
    prepared: ensemble.PreparedEnsemble
    previous_target: str = ""
    created_objects: list[str] = field(default_factory=list)
    inspections: dict[int, ObjectTokenInspection] = field(default_factory=dict)
    applied_transforms: list[ensemble.AlignmentTransform] = field(default_factory=list)
    group_existed: bool = False
    previous_group_members: tuple[str, ...] = ()
    group_additions: tuple[str, ...] = ()
    previous_ensemble: ensemble.EnsembleState | None = None
    previous_model_store: ModelStoreSnapshot | None = None
    previous_viewer_context: dict[tuple[str, str], ObjectPaintMapping] | None = None


def _session_path_for_candidate(
    discovery: PredictionDiscovery, candidate: PredictionCandidate
) -> Path:
    input_path = Path(discovery.input_path)
    if input_path.is_file():
        return input_path
    return Path(candidate.path)


def _discovery_phase(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(_ARCHIVE_SUFFIXES):
        return "Extracting archive and discovering predictions…"
    if lowered.endswith((".cif", ".pdb")):
        return "Inspecting structure file…"
    return "Discovering prediction folders…"


def _discover_prediction(
    path: str, report_phase: ProgressReporter
) -> PredictionDiscovery:
    from .loader import discover_prediction_candidates

    report_phase(_discovery_phase(path))
    return discover_prediction_candidates(path)


def _scan_and_load_initial_prediction(
    discovery: PredictionDiscovery,
    candidate: PredictionCandidate,
    preferred_rank: int | None,
    report_phase: ProgressReporter,
) -> InitialLoadResult:
    from .loader import load_prediction_data, scan_prediction_candidate
    from .structure_index import StructureIndex

    display_path = _session_path_for_candidate(discovery, candidate)
    report_phase(f"Scanning {candidate.provider_label} output…")
    pred_files = scan_prediction_candidate(discovery, candidate)
    try:
        if not pred_files.models:
            raise ValueError("No ranked model files were found.")

        available_ranks = {model.rank for model in pred_files.models}
        rank = (
            preferred_rank
            if preferred_rank is not None and preferred_rank in available_ranks
            else pred_files.models[0].rank
        )
        model = pred_files.model(rank)
        report_phase(f"Indexing {model.display_label} structure…")
        structure_index = StructureIndex.from_path(
            pred_files.structure_path(model.rank)
        )
        report_phase(f"Loading {model.display_label} data…")
        pred_data = load_prediction_data(
            pred_files,
            rank,
            load_pae=False,
            load_pde=False,
            load_contact_probs=False,
            structure_index=structure_index,
        )
        return InitialLoadResult(
            pred_files,
            ModelState(
                rank=rank,
                data=pred_data,
                structure_index=structure_index,
            ),
            display_path,
        )
    except Exception:
        pred_files.close()
        raise


def _load_rank_data(
    pred_files: PredictionFiles, rank: int, report_phase: ProgressReporter
) -> ModelSwitchResult:
    from .loader import load_prediction_data
    from .structure_index import StructureIndex

    model = pred_files.model(rank)
    report_phase(f"Indexing {model.display_label} structure…")
    structure_index = StructureIndex.from_path(pred_files.structure_path(model.rank))
    report_phase(f"Loading {model.display_label} data…")
    data = load_prediction_data(
        pred_files,
        rank,
        load_pae=False,
        load_pde=False,
        load_contact_probs=False,
        structure_index=structure_index,
    )
    return ModelSwitchResult(
        pred_files,
        ModelState(rank=rank, data=data, structure_index=structure_index),
    )


def _load_data_batch(
    pred_files: PredictionFiles,
    items: tuple[DataLoadRequirement, ...],
    report_phase: ProgressReporter,
) -> DataLoadBatchResult:
    from .loader import load_prediction_data

    loaded: list[tuple[DataLoadRequirement, PredictionData]] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        arrays = " and ".join(item.phase_arrays) or "metric data"
        suffix = f" ({index}/{total})" if total > 1 else ""
        report_phase(f"Loading {arrays} for {item.model_label}{suffix}…")
        data = load_prediction_data(
            pred_files,
            item.rank,
            structure_index=item.model_state.structure_index,
            **item.load_kwargs(),
        )
        loaded.append((item, data))
    return DataLoadBatchResult(pred_files, tuple(loaded))


def _prepare_ensemble_job(
    pred_files: PredictionFiles,
    skip_alignment: bool,
    existing_states_by_rank: dict[int, ModelState],
    report_phase: ProgressReporter,
) -> ensemble.PreparedEnsemble:
    return ensemble.prepare_ensemble(
        pred_files,
        skip_alignment=skip_alignment,
        existing_states_by_rank=existing_states_by_rank,
        report_phase=report_phase,
    )
