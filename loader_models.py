"""Provider-neutral data models exposed by :mod:`FoldQC.loader`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .confidence import ConfidenceSummarySpec, PredictionConfidence
from .ownership import Closeable

DataCapability = Literal["plddt", "pae", "pde", "contact_probs"]
_DATA_CAPABILITIES = frozenset({"plddt", "pae", "pde", "contact_probs"})
PlddtSource = Literal[
    "structure_b_factor",
    "provider_token",
    "provider_atom_mean",
]


@dataclass(frozen=True)
class ProviderInfo:
    """Immutable provider identity materialized at discovery time."""

    key: str
    label: str
    supports_ensemble: bool = True
    confidence_summary: ConfidenceSummarySpec = ConfidenceSummarySpec()

    def __post_init__(self) -> None:
        if not self.key or not self.label:
            raise ValueError("ProviderInfo requires non-empty key and label.")
        if not isinstance(self.confidence_summary, ConfidenceSummarySpec):
            raise ValueError(
                "ProviderInfo.confidence_summary must be ConfidenceSummarySpec."
            )


@dataclass
class ModelFiles:
    rank: int
    structure_path: Path
    display_label: str
    object_name: str
    confidence_path: Path | None = None
    summary_path: Path | None = None
    plddt_path: Path | None = None
    pae_path: Path | None = None
    pde_path: Path | None = None
    capabilities: frozenset[DataCapability] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.capabilities = frozenset(self.capabilities)
        unknown = self.capabilities - _DATA_CAPABILITIES
        if unknown:
            raise ValueError(f"Unknown model data capabilities: {sorted(unknown)!r}")

    def supports(self, capability: DataCapability) -> bool:
        return capability in self.capabilities


@dataclass
class PredictionFiles:
    name: str
    pred_dir: Path
    provider: ProviderInfo
    input_path: Path | None = None
    models: list[ModelFiles] = field(default_factory=list)
    affinity_file: Path | None = None
    embeddings_file: Path | None = None
    _resource_owner: Closeable | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.provider, ProviderInfo):
            raise ValueError("PredictionFiles.provider must be ProviderInfo.")

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def provider_label(self) -> str:
        return self.provider.label

    @property
    def supports_ensemble(self) -> bool:
        return self.provider.supports_ensemble and self.n_models >= 2

    @property
    def has_affinity(self) -> bool:
        return self.affinity_file is not None

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings_file is not None

    def model(self, rank: int) -> ModelFiles:
        models = {model.rank: model for model in self.models}
        if rank not in models:
            raise KeyError(rank)
        return models[rank]

    def model_supports(self, rank: int, capability: DataCapability) -> bool:
        return self.model(rank).supports(capability)

    def any_model_supports(self, capability: DataCapability) -> bool:
        return any(model.supports(capability) for model in self.models)

    def all_models_support(self, capability: DataCapability) -> bool:
        return bool(self.models) and all(
            model.supports(capability) for model in self.models
        )

    def structure_path(self, rank: int) -> Path:
        return self.model(rank).structure_path

    def confidence_path(self, rank: int) -> Path | None:
        return self.model(rank).confidence_path

    def summary_path(self, rank: int) -> Path | None:
        return self.model(rank).summary_path

    def plddt_path(self, rank: int) -> Path | None:
        return self.model(rank).plddt_path

    def pae_path(self, rank: int) -> Path | None:
        return self.model(rank).pae_path

    def pde_path(self, rank: int) -> Path | None:
        return self.model(rank).pde_path

    def adopt_resource_owner(self, owner: Closeable | None) -> None:
        if owner is None:
            return
        if self._resource_owner is not None:
            raise RuntimeError("PredictionFiles already owns an external resource.")
        self._resource_owner = owner

    def close(self) -> None:
        owner = self._resource_owner
        self._resource_owner = None
        if owner is not None:
            owner.close()


@dataclass
class PredictionData:
    name: str
    rank: int
    structure_path: Path
    provider: ProviderInfo
    display_label: str = ""
    token_plddt: np.ndarray | None = None
    token_plddt_source: PlddtSource | None = None
    pae: np.ndarray | None = None
    pde: np.ndarray | None = None
    contact_probs: np.ndarray | None = None
    confidence: PredictionConfidence | None = None
    embeddings_s: np.ndarray | None = None
    embeddings_z: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, ProviderInfo):
            raise ValueError("PredictionData.provider must be ProviderInfo.")


@dataclass(frozen=True)
class PredictionCandidate:
    path: Path
    provider: ProviderInfo
    relative_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, ProviderInfo):
            raise ValueError("PredictionCandidate.provider must be ProviderInfo.")

    @property
    def provider_label(self) -> str:
        return self.provider.label


@dataclass
class PredictionDiscovery:
    input_path: Path
    candidates: tuple[PredictionCandidate, ...]
    _resource_owner: Closeable | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.candidates = tuple(self.candidates)
        if not all(
            isinstance(candidate, PredictionCandidate) for candidate in self.candidates
        ):
            raise ValueError(
                "PredictionDiscovery.candidates must contain PredictionCandidate values."
            )

    def take_resource_owner(self) -> Closeable | None:
        owner = self._resource_owner
        self._resource_owner = None
        return owner

    def close(self) -> None:
        owner = self.take_resource_owner()
        if owner is not None:
            owner.close()
