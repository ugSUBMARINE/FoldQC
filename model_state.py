"""Canonical per-rank prediction state independent of Qt and PyMOL."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from .loader_models import PredictionData
from .token_map import TokenMap

_ARRAY_FIELDS = (
    "token_plddt",
    "pae",
    "pde",
    "contact_probs",
    "embeddings_s",
    "embeddings_z",
)
_METADATA_FIELDS = ("confidence", "summary_confidence", "affinity")
_IDENTITY_FIELDS = ("name", "provider", "structure_path")
_DATA_FIELDS = tuple(item.name for item in fields(PredictionData))


@dataclass(frozen=True)
class ModelStateSnapshot:
    """Shallow, identity-preserving rollback snapshot for one model state."""

    rank: int
    values: tuple[tuple[str, object], ...]
    token_map: TokenMap
    version: int


@dataclass
class ModelState:
    """Loaded prediction data and immutable token map for one model rank."""

    rank: int
    data: PredictionData
    token_map: TokenMap
    _version: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_rank(self.data)

    @property
    def version(self) -> int:
        """Monotonic version incremented after each material data merge."""
        return self._version

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

        token_plddt = getattr(incoming, "token_plddt", None)
        token_source = getattr(incoming, "token_plddt_source", None)
        if (token_plddt is None) != (token_source is None):
            raise ValueError(
                f"Partial model_{self.rank} data must provide token pLDDT values "
                "and provenance together."
            )

        embeddings_s = getattr(incoming, "embeddings_s", None)
        embeddings_z = getattr(incoming, "embeddings_z", None)
        if (embeddings_s is None) != (embeddings_z is None):
            raise ValueError(
                f"Partial model_{self.rank} data must provide both embedding arrays."
            )

    def validate_token_map(self, token_map: TokenMap) -> None:
        """Reject a staged state whose token order differs from this state."""
        if token_map != self.token_map:
            raise ValueError(
                f"Cannot merge model_{self.rank} data with a different token map."
            )

    def merge_data(self, incoming: PredictionData) -> bool:
        """Monotonically merge a partial provider result in place.

        Existing arrays remain authoritative. Metadata dictionaries are shallowly
        enriched, with incoming values winning for matching keys.
        """
        self.validate_merge(incoming)
        changed = False

        for name in _ARRAY_FIELDS:
            if getattr(self.data, name, None) is not None:
                continue
            value = getattr(incoming, name, None)
            if value is None:
                continue
            setattr(self.data, name, value)
            changed = True
            if name == "token_plddt":
                self.data.token_plddt_source = incoming.token_plddt_source

        for name in _METADATA_FIELDS:
            incoming_value = getattr(incoming, name, None)
            if incoming_value is None:
                continue
            current_value = getattr(self.data, name, None)
            merged = (
                {**current_value, **incoming_value}
                if isinstance(current_value, dict) and isinstance(incoming_value, dict)
                else incoming_value
            )
            if current_value != merged:
                setattr(self.data, name, merged)
                changed = True

        if changed:
            self._version += 1
        return changed

    def snapshot(self) -> ModelStateSnapshot:
        """Capture field references needed to restore this state in place."""
        return ModelStateSnapshot(
            rank=self.rank,
            values=tuple(
                (name, getattr(self.data, name, None)) for name in _DATA_FIELDS
            ),
            token_map=self.token_map,
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
        self.token_map = snapshot.token_map
        self._version = snapshot.version
