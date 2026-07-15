"""Shared immutable lifecycle results and background job helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ensemble, gui_rules, metrics, plot_data, reports
from .analysis import DataLoadRequirement, DeferredAnalysisAction
from .loader_models import DataCapability, PredictionData, PredictionFiles
from .model_state import ModelState, ModelStateSnapshot
from .mol_viewer import (
    add_objects_to_group,
    delete_viewer_names,
    ensure_structure_object,
    get_group_members,
    get_object_list,
    get_viewer_name,
    inspect_object_tokens,
    load_structure_object_if_missing,
    rebuild,
    remove_objects_from_group,
    run_with_updates_suspended,
    transform_object,
    viewer_name_exists,
)
from .presentation import (
    ChoiceOption,
    ChoiceRequest,
    Notice,
    ProgressRequest,
    SelectionItem,
    SelectionRequest,
)

# These names form the intentionally private support surface consumed by the
# four lifecycle services. Keeping the imports here avoids each service
# depending on provider/job implementation details while leaving ownership in
# the service modules themselves.
__all__ = (
    "APP_TITLE",
    "VIEWER_NAME",
    "ChoiceOption",
    "ChoiceRequest",
    "DataCapability",
    "DataLoadBatchResult",
    "DataLoadRequirement",
    "DeferredAnalysisAction",
    "EnsembleActivationTransaction",
    "InitialLoadResult",
    "InitialPredictionSnapshot",
    "ModelState",
    "ModelStoreSnapshot",
    "ModelSwitchResult",
    "Notice",
    "Path",
    "PredictionFiles",
    "ProgressRequest",
    "SelectionItem",
    "SelectionRequest",
    "_discover_prediction",
    "_discovery_phase",
    "_load_data_batch",
    "_load_rank_data",
    "_prepare_ensemble_job",
    "_scan_and_load_initial_prediction",
    "_session_path_for_candidate",
    "add_objects_to_group",
    "delete_viewer_names",
    "ensemble",
    "ensure_structure_object",
    "get_group_members",
    "get_object_list",
    "gui_rules",
    "inspect_object_tokens",
    "load_structure_object_if_missing",
    "logger",
    "metrics",
    "np",
    "plot_data",
    "rebuild",
    "remove_objects_from_group",
    "reports",
    "run_with_updates_suspended",
    "transform_object",
    "viewer_name_exists",
)

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()
logger = logging.getLogger(__name__)

_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


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


@dataclass(frozen=True)
class InitialPredictionSnapshot:
    """GUI state restored if initial prediction activation fails."""

    pred_files: PredictionFiles | None
    model_store: ModelStoreSnapshot
    ensemble_state: ensemble.EnsembleState | None
    display_path: str
    model_items: tuple[tuple[str, object], ...]
    selected_model_rank: object | None
    viewer_context: tuple


@dataclass
class EnsembleActivationTransaction:
    """Main-thread state for an incrementally committed ensemble."""

    request_id: int
    prepared: ensemble.PreparedEnsemble
    previous_target: str = ""
    created_objects: list[str] = field(default_factory=list)
    inspections: dict[int, object] = field(default_factory=dict)
    applied_transforms: list[ensemble.AlignmentTransform] = field(default_factory=list)
    group_existed: bool = False
    previous_group_members: tuple[str, ...] = ()
    group_additions: tuple[str, ...] = ()
    previous_ensemble: ensemble.EnsembleState | None = None
    previous_model_store: ModelStoreSnapshot | None = None
    previous_viewer_context: tuple | None = None


def _session_path_for_candidate(discovery, candidate) -> Path:
    input_path = getattr(discovery, "input_path", None)
    if input_path is not None:
        input_path = Path(input_path)
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


def _discover_prediction(path: str, report_phase):
    from .loader import discover_prediction_candidates

    report_phase(_discovery_phase(path))
    return discover_prediction_candidates(path)


def _scan_and_load_initial_prediction(
    discovery,
    candidate,
    preferred_rank: int | None,
    report_phase,
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


def _load_rank_data(pred_files, rank: int, report_phase) -> ModelSwitchResult:
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


def _load_data_batch(pred_files, items: tuple[DataLoadRequirement, ...], report_phase):
    from .loader import load_prediction_data

    loaded = []
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
    pred_files,
    skip_alignment: bool,
    existing_states_by_rank: dict[int, ModelState],
    report_phase,
):
    return ensemble.prepare_ensemble(
        pred_files,
        skip_alignment=skip_alignment,
        existing_states_by_rank=existing_states_by_rank,
        report_phase=report_phase,
    )
