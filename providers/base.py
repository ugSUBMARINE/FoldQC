"""Contracts shared by FoldQC's built-in prediction providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from ..confidence import (
    ConfidenceSummarySpec,
    merge_prediction_confidence,
    parse_prediction_confidence,
)
from ..data_contracts import (
    normalize_and_validate_prediction_data,
    require_advertised_fields,
)
from ..loader_models import (
    ModelFiles,
    PredictionCandidate,
    PredictionData,
    PredictionFiles,
    ProviderInfo,
)
from ..loader_utils import _load_json
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
    confidence_summary = ConfidenceSummarySpec()

    @cached_property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            self.key,
            self.label,
            self.supports_ensemble,
            self.confidence_summary,
        )

    def prediction_files(self, *, name: str, pred_dir: Path) -> PredictionFiles:
        """Construct provider-neutral discovery data with materialized metadata."""
        return PredictionFiles(
            name=name,
            pred_dir=pred_dir,
            provider=self.info,
            input_path=pred_dir,
        )

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
        chain_count = (
            len(structure_index.token_map.chain_order) if structure_index else 0
        )
        if model.summary_path is not None:
            if structure_index is None:
                raise ValueError(
                    f"No StructureIndex available for {model.structure_path.name}."
                )
            data.confidence = parse_prediction_confidence(
                _load_json(model.summary_path),
                chain_count=chain_count,
                provider=pred_files.provider.key,
                model_label=model.display_label,
                source=model.summary_path,
            )
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
        require_advertised_fields(
            data,
            provider=pred_files.provider.key,
            model_label=model.display_label,
            requested=(
                (
                    "pae",
                    options.load_pae and model.supports("pae"),
                    model.pae_path or model.confidence_path,
                ),
                (
                    "pde",
                    options.load_pde and model.supports("pde"),
                    model.pde_path or model.confidence_path,
                ),
                (
                    "contact_probs",
                    options.load_contact_probs and model.supports("contact_probs"),
                    model.confidence_path,
                ),
            ),
        )
        token_count = len(structure_index.token_map) if structure_index else 0
        return normalize_and_validate_prediction_data(
            data, token_count, chain_count=chain_count
        )

    def merge_confidence_payload(
        self,
        data: PredictionData,
        payload: dict | None,
        *,
        model: ModelFiles,
        structure_index: StructureIndex | None,
        source: Path | str | None,
        affinity_payload: dict | None = None,
    ) -> None:
        """Normalize one provider payload and monotonically enrich confidence."""
        if structure_index is None:
            raise ValueError(
                f"No StructureIndex available for {model.structure_path.name}."
            )
        incoming = parse_prediction_confidence(
            payload,
            chain_count=len(structure_index.token_map.chain_order),
            provider=self.key,
            model_label=model.display_label,
            source=source,
            affinity_payload=affinity_payload,
        )
        data.confidence = merge_prediction_confidence(
            data.confidence,
            incoming,
            context=f"{self.key} {model.display_label} confidence",
        )

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
        if other.path == candidate.path or other.provider.key != provider:
            continue
        try:
            candidate.path.relative_to(other.path)
        except ValueError:
            continue
        return True
    return False
