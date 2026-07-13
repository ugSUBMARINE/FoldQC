"""Typed state and context values shared by FoldQC GUI coordinators.

This module is deliberately independent of Qt and PyMOL.  The dialog owns one
``GuiState`` instance while GUI-side coordinators operate on the same state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .session import PendingSessionRestore


@dataclass
class ResolvedTarget:
    """Prediction/model context resolved from the viewer target control."""

    kind: Literal["single", "ensemble_member", "ensemble_group"]
    label: str
    obj_name: str
    data: object | None
    token_map: object
    members: list | None = None


@dataclass(frozen=True)
class MetricContext:
    """Viewer-derived inputs and provenance for one metric computation."""

    reference_selection: str = ""
    reference_indices: tuple[int, ...] = ()
    contact_indices: tuple[int, ...] = ()
    cutoff_angstrom: float | None = None


@dataclass
class GuiState:
    """Mutable non-widget state owned by the main FoldQC dialog."""

    pred_files: object | None = None
    pred_data: object | None = None
    token_map: object | None = None
    token_map_obj: str | None = None
    token_map_structure_path: object | None = None
    paint_mappings: dict[tuple[str, str], object] = field(default_factory=dict)
    ensemble_members: list | None = None
    ensemble_group_name: str | None = None
    ensemble_aligned: bool = False
    ensemble_rmsd: np.ndarray | None = None
    ensemble_plddt_mean: np.ndarray | None = None
    ensemble_plddt_std: np.ndarray | None = None
    accepted_token_overlap_warnings: set[tuple[str, str]] = field(default_factory=set)
    loading_prediction: bool = False
    restoring_settings: bool = False
    pending_session_restore: PendingSessionRestore = field(
        default_factory=PendingSessionRestore
    )


class GuiStateBacked:
    """Properties exposing the dialog's shared state to GUI coordinators."""


def _state_property(name: str):
    def getter(self):
        return getattr(self._state, name)

    def setter(self, value) -> None:
        setattr(self._state, name, value)

    return property(getter, setter)


for _private_name, _state_name in {
    "_pred_files": "pred_files",
    "_pred_data": "pred_data",
    "_token_map": "token_map",
    "_token_map_obj": "token_map_obj",
    "_token_map_structure_path": "token_map_structure_path",
    "_paint_mappings": "paint_mappings",
    "_ensemble_members": "ensemble_members",
    "_ensemble_group_name": "ensemble_group_name",
    "_ensemble_aligned": "ensemble_aligned",
    "_ensemble_rmsd": "ensemble_rmsd",
    "_ensemble_plddt_mean": "ensemble_plddt_mean",
    "_ensemble_plddt_std": "ensemble_plddt_std",
    "_accepted_token_overlap_warnings": "accepted_token_overlap_warnings",
    "_loading_prediction": "loading_prediction",
    "_restoring_settings": "restoring_settings",
    "_pending_session_restore": "pending_session_restore",
}.items():
    setattr(GuiStateBacked, _private_name, _state_property(_state_name))

del _private_name, _state_name
