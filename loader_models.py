"""Provider-neutral data models exposed by :mod:`FoldQC.loader`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionFiles:
    name: str
    pred_dir: Path
    provider: str = "boltz"
    input_path: Path | None = None
    models: list[ModelFiles] = field(default_factory=list)
    affinity_file: Path | None = None
    embeddings_file: Path | None = None
    capabilities: set[str] = field(default_factory=set)
    _temporary_directory: Any | None = field(default=None, repr=False, compare=False)

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def provider_label(self) -> str:
        from .providers.registry import BUILTIN_PROVIDERS

        return BUILTIN_PROVIDERS.get(self.provider).label

    @property
    def supports_ensemble(self) -> bool:
        from .providers.registry import BUILTIN_PROVIDERS

        return (
            BUILTIN_PROVIDERS.get(self.provider).supports_ensemble
            and self.n_models >= 2
        )

    @property
    def has_pae(self) -> bool:
        return "pae" in self.capabilities

    @property
    def has_pde(self) -> bool:
        return "pde" in self.capabilities

    @property
    def has_contact_probs(self) -> bool:
        return "contact_probs" in self.capabilities

    @property
    def has_plddt(self) -> bool:
        return "plddt" in self.capabilities

    @property
    def has_structure_plddt(self) -> bool:
        return "structure_plddt" in self.capabilities

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


@dataclass
class PredictionData:
    name: str
    rank: int
    structure_path: Path
    provider: str = "boltz"
    display_label: str = ""
    structure_plddt: np.ndarray | None = None
    plddt: np.ndarray | None = None
    pae: np.ndarray | None = None
    pde: np.ndarray | None = None
    contact_probs: np.ndarray | None = None
    confidence: dict | None = None
    summary_confidence: dict | None = None
    affinity: dict | None = None
    embeddings_s: np.ndarray | None = None
    embeddings_z: np.ndarray | None = None


@dataclass(frozen=True)
class PredictionCandidate:
    path: Path
    provider: str
    provider_label: str
    relative_path: str


@dataclass
class PredictionDiscovery:
    input_path: Path
    candidates: tuple[PredictionCandidate, ...]
    _temporary_directory: Any | None = field(default=None, repr=False, compare=False)

    def scan(self, candidate: PredictionCandidate) -> PredictionFiles:
        if candidate not in self.candidates:
            raise ValueError(f"Unknown prediction candidate: {candidate.path}")
        from .loader_discovery import scan_prediction_path_exact

        files = scan_prediction_path_exact(candidate.path, input_path=self.input_path)
        if self._temporary_directory is not None:
            files._temporary_directory = self._temporary_directory
            self._temporary_directory = None
        return files
