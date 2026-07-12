"""Stable public facade for prediction discovery and provider loading."""

from __future__ import annotations

from pathlib import Path

from .loader_discovery import discover_prediction_candidates
from .loader_models import (
    PredictionData,
    PredictionFiles,
)
from .providers.base import LoadOptions
from .providers.registry import BUILTIN_PROVIDERS


def scan_prediction_path(path: str | Path) -> PredictionFiles:
    discovery = discover_prediction_candidates(path)
    return discovery.scan(discovery.candidates[0])


def scan_prediction_dir(path: str | Path) -> PredictionFiles:
    selected_dir = Path(path).expanduser().resolve()
    if not selected_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {selected_dir}")
    discovery = discover_prediction_candidates(selected_dir)
    return discovery.scan(discovery.candidates[0])


def load_prediction_data(
    pred_files: PredictionFiles,
    rank: int = 0,
    load_pae: bool = True,
    load_pde: bool = True,
    load_embeddings: bool = False,
    load_structure_plddt: bool = True,
    load_contact_probs: bool = False,
    load_plddt: bool = True,
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
        load_structure_plddt=load_structure_plddt,
        load_contact_probs=load_contact_probs,
        load_plddt=load_plddt,
    )
    return BUILTIN_PROVIDERS.get(pred_files.provider).load(pred_files, model, options)
