"""Typed state and workflow values shared by FoldQC GUI coordinators.

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

    def compute_kwargs(self) -> dict[str, object]:
        """Return mutable collections expected by :func:`compute_metric`."""
        return {
            "ref_indices": list(self.reference_indices),
            "contact_indices": list(self.contact_indices),
            "cutoff": self.cutoff_angstrom,
        }


@dataclass
class GuiState:
    """Mutable non-widget state owned by the main FoldQC dialog."""

    pred_files: object | None = None
    pred_data: object | None = None
    token_map: object | None = None
    token_map_obj: str | None = None
    token_map_structure_path: object | None = None
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


@dataclass(frozen=True)
class WorkflowMessage:
    """Expected user-facing outcome returned without a Qt dependency."""

    severity: Literal["information", "warning", "critical"]
    text: str
    title: str = "FoldQC"


class GuiWorkflowError(Exception):
    """Expected workflow failure for presentation by the Qt adapter."""

    def __init__(self, message: WorkflowMessage) -> None:
        super().__init__(message.text)
        self.message = message


class GuiStateBacked:
    """Compatibility properties exposing the shared state to GUI mixins.

    Lazy creation keeps focused tests that instantiate the dialog with
    ``__new__`` working while they migrate to explicit ``GuiState`` fixtures.
    """

    def _get_gui_state(self) -> GuiState:
        state = getattr(self, "_state", None)
        if state is None:
            state = GuiState()
            self._state = state
        return state


def _state_property(name: str):
    def getter(self):
        return getattr(self._get_gui_state(), name)

    def setter(self, value) -> None:
        setattr(self._get_gui_state(), name, value)

    return property(getter, setter)


for _private_name, _state_name in {
    "_pred_files": "pred_files",
    "_pred_data": "pred_data",
    "_token_map": "token_map",
    "_token_map_obj": "token_map_obj",
    "_token_map_structure_path": "token_map_structure_path",
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
