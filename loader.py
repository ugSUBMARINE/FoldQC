"""Stable public facade for prediction discovery and provider loading."""

from __future__ import annotations

from pathlib import Path

from .confidence import PredictionConfidence
from .loader_discovery import (
    discover_prediction_candidates,
    scan_prediction_candidate,
)
from .loader_models import (
    PredictionData,
    PredictionFiles,
)
from .providers.base import LoadOptions
from .providers.registry import BUILTIN_PROVIDERS
from .structure_index import StructureIndex


def scan_prediction_path(path: str | Path) -> PredictionFiles:
    discovery = discover_prediction_candidates(path)
    try:
        return scan_prediction_candidate(discovery, discovery.candidates[0])
    except Exception:
        discovery.close()
        raise


def scan_prediction_dir(path: str | Path) -> PredictionFiles:
    selected_dir = Path(path).expanduser().resolve()
    if not selected_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {selected_dir}")
    discovery = discover_prediction_candidates(selected_dir)
    try:
        return scan_prediction_candidate(discovery, discovery.candidates[0])
    except Exception:
        discovery.close()
        raise


def load_prediction_data(
    pred_files: PredictionFiles,
    rank: int = 0,
    load_pae: bool = True,
    load_pde: bool = True,
    load_embeddings: bool = False,
    load_token_plddt: bool = True,
    load_contact_probs: bool = False,
    structure_index: StructureIndex | None = None,
) -> PredictionData:
    try:
        model = pred_files.model(rank)
    except KeyError as exc:
        raise ValueError(
            f"Rank {rank} not found. Available ranks: "
            + str(sorted(model.rank for model in pred_files.models))
        ) from exc
    options = LoadOptions(
        load_pae=load_pae,
        load_pde=load_pde,
        load_embeddings=load_embeddings,
        load_token_plddt=load_token_plddt,
        load_contact_probs=load_contact_probs,
    )
    if structure_index is not None and not structure_index.matches_path(
        model.structure_path
    ):
        raise ValueError(
            f"StructureIndex path {structure_index.path!s} does not match "
            f"model_{rank} structure path {model.structure_path!s}."
        )
    needs_structure_index = any(
        (
            load_pae,
            load_pde,
            load_embeddings,
            load_token_plddt,
            load_contact_probs,
        )
    ) or any(
        (
            model.summary_path is not None,
            model.confidence_path is not None,
            bool(model.metadata.get("has_metrics_confidence")),
            pred_files.affinity_file is not None,
        )
    )
    if needs_structure_index and structure_index is None:
        structure_index = StructureIndex.from_path(model.structure_path)
    return BUILTIN_PROVIDERS.get(pred_files.provider.key).load(
        pred_files,
        model,
        options,
        structure_index=structure_index,
    )


def load_prediction_confidence_summaries(
    pred_files: PredictionFiles,
) -> tuple[PredictionConfidence | None, ...]:
    """Load compact scalar confidence summaries without loading model data."""
    provider = BUILTIN_PROVIDERS.get(pred_files.provider.key)
    return tuple(
        provider.load_model_confidence_summary(pred_files, model)
        for model in pred_files.models
    )
