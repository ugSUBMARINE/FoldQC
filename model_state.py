"""Canonical per-rank prediction state independent of Qt and PyMOL."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from .confidence import merge_prediction_confidence
from .data_contracts import normalize_and_validate_prediction_data
from .loader_models import PredictionData
from .structure_index import StructureIndex

_ARRAY_FIELDS = (
    "token_plddt",
    "pae",
    "pde",
    "contact_probs",
    "embeddings_s",
    "embeddings_z",
)
_IDENTITY_FIELDS = ("name", "provider", "structure_path")
_DATA_FIELDS = tuple(item.name for item in fields(PredictionData))


@dataclass(frozen=True)
class ModelStateSnapshot:
    """Shallow, identity-preserving rollback snapshot for one model state."""

    rank: int
    values: tuple[tuple[str, object], ...]
    version: int


@dataclass
class ModelState:
    """Loaded prediction data and immutable token map for one model rank."""

    rank: int
    data: PredictionData
    structure_index: StructureIndex
    _version: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_rank(self.data)
        if not self.structure_index.matches_path(self.data.structure_path):
            raise ValueError(
                f"ModelState rank {self.rank} structure path does not match its "
                f"StructureIndex: {self.data.structure_path!s} != "
                f"{self.structure_index.path!s}."
            )
        normalize_and_validate_prediction_data(
            self.data,
            len(self.structure_index.token_map),
            chain_count=len(self.structure_index.token_map.chain_order),
        )

    @property
    def version(self) -> int:
        """Monotonic version incremented after each material data merge."""
        return self._version

    @property
    def token_map(self):
        """Return the immutable token map owned by this state's structure index."""
        return self.structure_index.token_map

    def _validate_rank(self, data: PredictionData) -> None:
        if int(data.rank) != self.rank:
            raise ValueError(
                f"ModelState rank {self.rank} cannot contain rank {data.rank} data."
            )

    def validate_merge(self, incoming: PredictionData) -> None:
        """Validate a partial provider result without mutating this state."""
        self._validate_rank(incoming)
        for name in _IDENTITY_FIELDS:
            current = getattr(self.data, name, None)
            candidate = getattr(incoming, name, None)
            if current is None or candidate is None:
                continue
            if name == "structure_path":
                current = Path(current)
                candidate = Path(candidate)
            if current != candidate:
                raise ValueError(
                    f"Cannot merge model_{self.rank} data with a different {name}: "
                    f"{candidate!s} != {current!s}."
                )

        normalize_and_validate_prediction_data(
            incoming,
            len(self.structure_index.token_map),
            chain_count=len(self.structure_index.token_map.chain_order),
        )

    def validate_structure_index(self, structure_index: StructureIndex) -> None:
        """Reject a staged state that did not reuse this canonical index."""
        if structure_index is not self.structure_index:
            raise ValueError(
                f"Cannot merge model_{self.rank} data with a different StructureIndex."
            )

    def merge_data(self, incoming: PredictionData) -> bool:
        """Monotonically merge a partial provider result in place.

        Existing arrays and confidence values remain authoritative. Incoming
        confidence may only fill fields that were previously missing.
        """
        self.validate_merge(incoming)
        additions: list[tuple[str, object]] = []
        for name in _ARRAY_FIELDS:
            if getattr(self.data, name, None) is not None:
                continue
            value = getattr(incoming, name, None)
            if value is None:
                continue
            additions.append((name, value))

        merged_confidence = merge_prediction_confidence(
            self.data.confidence,
            incoming.confidence,
            context=f"model_{self.rank} confidence",
        )
        confidence_changed = merged_confidence is not self.data.confidence
        if not additions and not confidence_changed:
            return False

        for name, value in additions:
            setattr(self.data, name, value)
            if name == "token_plddt":
                self.data.token_plddt_source = incoming.token_plddt_source
        if confidence_changed:
            self.data.confidence = merged_confidence
        self._version += 1
        return True

    def snapshot(self) -> ModelStateSnapshot:
        """Capture field references needed to restore this state in place."""
        return ModelStateSnapshot(
            rank=self.rank,
            values=tuple(
                (name, getattr(self.data, name, None)) for name in _DATA_FIELDS
            ),
            version=self._version,
        )

    def restore(self, snapshot: ModelStateSnapshot) -> None:
        """Restore a snapshot without replacing the canonical data object."""
        if snapshot.rank != self.rank:
            raise ValueError(
                f"Cannot restore rank {snapshot.rank} into ModelState rank {self.rank}."
            )
        for name, value in snapshot.values:
            setattr(self.data, name, value)
        self._version = snapshot.version
