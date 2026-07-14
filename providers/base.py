"""Contracts shared by FoldQC's built-in prediction providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..loader_models import (
    ModelFiles,
    PredictionCandidate,
    PredictionData,
    PredictionFiles,
)
from ..loader_utils import _load_json, _normalise_confidence
from ..structure_index import StructureIndex


@dataclass(frozen=True)
class LoadOptions:
    load_pae: bool = True
    load_pde: bool = True
    load_embeddings: bool = False
    load_token_plddt: bool = True
    load_contact_probs: bool = False


class BaseProvider(ABC):
    key: str
    label: str
    supports_ensemble: bool = True

    @abstractmethod
    def detect(self, path: Path) -> bool:
        """Return whether *path* is an output handled by this provider."""

    @abstractmethod
    def scan(self, path: Path) -> PredictionFiles:
        """Discover ranked model files below an exact provider path."""

    def load(
        self,
        pred_files: PredictionFiles,
        model: ModelFiles,
        options: LoadOptions,
        *,
        structure_index: StructureIndex | None,
    ) -> PredictionData:
        data = PredictionData(
            name=pred_files.name,
            rank=model.rank,
            structure_path=model.structure_path,
            provider=pred_files.provider,
            display_label=model.display_label,
        )
        if model.summary_path is not None:
            data.summary_confidence = _load_json(model.summary_path)
            data.confidence = _normalise_confidence(data.summary_confidence)
        self.load_model_data(
            pred_files,
            model,
            data,
            options,
            structure_index=structure_index,
        )
        if options.load_token_plddt and data.token_plddt is None:
            if structure_index is None:
                raise ValueError(
                    f"No StructureIndex available for {model.structure_path.name}."
                )
            data.token_plddt = structure_index.structure_plddt
            data.token_plddt_source = "structure_b_factor"
        if data.confidence is None and data.summary_confidence is not None:
            data.confidence = _normalise_confidence(data.summary_confidence)
        return data

    def load_model_data(
        self,
        pred_files: PredictionFiles,
        model: ModelFiles,
        data: PredictionData,
        options: LoadOptions,
        *,
        structure_index: StructureIndex | None,
    ) -> None:
        """Populate provider-specific lazy data; default is structure-only."""

    def is_internal_candidate(
        self,
        candidate: PredictionCandidate,
        candidates: list[PredictionCandidate],
    ) -> bool:
        return False


def has_ancestor_candidate(
    candidate: PredictionCandidate,
    candidates: list[PredictionCandidate],
    *,
    provider: str,
) -> bool:
    for other in candidates:
        if other.path == candidate.path or other.provider != provider:
            continue
        try:
            candidate.path.relative_to(other.path)
        except ValueError:
            continue
        return True
    return False
