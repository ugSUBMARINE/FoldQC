"""Typed state and context values shared by FoldQC GUI coordinators.

This module is deliberately independent of Qt and PyMOL.  The dialog owns one
``PluginState`` instance while GUI-side coordinators operate on the same state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .ensemble import EnsembleMember, EnsembleState
    from .loader_models import PredictionData, PredictionFiles
    from .model_state import ModelState
    from .token_map import TokenMap


@dataclass(frozen=True)
class ResolvedTarget:
    """Prediction/model context resolved from the viewer target control."""

    kind: Literal["single", "ensemble_member", "ensemble_group"]
    label: str
    obj_name: str
    model_states: tuple[ModelState, ...]
    members: tuple[EnsembleMember, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_states:
            raise ValueError("ResolvedTarget requires at least one model state.")
        if self.kind != "ensemble_group" and len(self.model_states) != 1:
            raise ValueError(f"{self.kind} targets require exactly one model state.")
        if self.kind.startswith("ensemble") and not self.members:
            raise ValueError(f"{self.kind} targets require ensemble members.")
        if self.members and len(self.members) != len(self.model_states):
            raise ValueError(
                "ResolvedTarget members must correspond one-to-one with model states."
            )

    @property
    def data(self) -> PredictionData | None:
        """Return live data for a single-state target, or None for a group."""
        if self.kind == "ensemble_group":
            return None
        return self.model_states[0].data

    @property
    def token_map(self) -> TokenMap:
        """Return the canonical token map shared by this target."""
        return self.model_states[0].token_map


@dataclass(frozen=True)
class MetricContext:
    """Viewer-derived inputs and provenance for one metric computation."""

    reference_selection: str = ""
    reference_indices: tuple[int, ...] = ()
    contact_indices: tuple[int, ...] = ()
    cutoff_angstrom: float | None = None


@dataclass
class PluginState:
    """Mutable non-widget state owned by the main FoldQC dialog."""

    pred_files: PredictionFiles | None = None
    model_states: dict[int, ModelState] = field(default_factory=dict)
    active_model_rank: int | None = None
    ensemble: EnsembleState | None = None

    @property
    def active_model_state(self) -> ModelState | None:
        rank = self.active_model_rank
        return None if rank is None else self.model_states.get(rank)
