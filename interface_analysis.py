"""Immutable interface reports and interaction contracts; no Qt or viewer imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .token_map import TokenMap


@dataclass(frozen=True)
class InterfaceAnalysisOptions:
    ipsae_cutoff: int = 10

    def __post_init__(self) -> None:
        if self.ipsae_cutoff not in (10, 15):
            raise ValueError("ipSAE requires the published 10 or 15 Å preset.")


@dataclass(frozen=True)
class InterfaceScoreResult:
    metric_key: str
    chain_a: str
    partners: tuple[str, ...]
    value: float | None
    forward: float | None = None
    reverse: float | None = None
    reason: str = ""
    parameters: tuple[tuple[str, str], ...] = ()
    counts: tuple[tuple[str, int], ...] = ()
    support_a: tuple[int, ...] = ()
    support_b: tuple[int, ...] = ()
    residue_profile: tuple[tuple[int, float], ...] = ()
    interface_plddt: float | None = None
    confidence_source: str = "not used"

    @property
    def status(self) -> str:
        return "unavailable" if self.value is None else "ok"


@dataclass(frozen=True)
class InterfaceContactDiagnostics:
    """Geometry-only counts, independent of metric confidence/support masks."""

    scope: Literal["chain_pair", "chain_rest"]
    chain_a: str
    partners: tuple[str, ...]
    convention: Literal["cb_ca_gly", "ca"]
    pair_count: int | None
    residues_a: int | None
    residues_b: int | None
    reason: str = ""
    cutoff_angstrom: float = 8.0
    comparison: Literal["<="] = "<="

    @property
    def status(self) -> str:
        return "unavailable" if self.pair_count is None else "ok"


@dataclass(frozen=True)
class InterfaceContactExportRequest:
    rank: int
    diagnostic: InterfaceContactDiagnostics
    path: str


@dataclass(frozen=True)
class InterfaceModelResult:
    rank: int
    label: str
    object_name: str
    structure_path: str
    token_map: TokenMap
    chain_order: tuple[str, ...]
    scores: tuple[InterfaceScoreResult, ...]
    diagnostics: tuple[str, ...] = ()
    protein_indices: tuple[int, ...] = ()
    contacts: tuple[InterfaceContactDiagnostics, ...] = ()


@dataclass(frozen=True)
class InterfaceAggregate:
    metric_key: str
    chain_a: str
    partners: tuple[str, ...]
    # Exact residue identities distinguish incompatible members without rank matching.
    identities: tuple[tuple[str, str, str], ...]
    ranks: tuple[int, ...]
    n_models: int
    mean: float
    std: float
    minimum: float
    maximum: float
    forward: float | None = None
    reverse: float | None = None


@dataclass(frozen=True)
class InterfaceReport:
    prediction_name: str
    provider: str
    provider_label: str
    input_path: str
    target_name: str
    options: InterfaceAnalysisOptions
    models: tuple[InterfaceModelResult, ...]
    aggregates: tuple[InterfaceAggregate, ...] = ()
    generation: int = 0


@dataclass(frozen=True)
class InterfaceProjectionSelection:
    """Compact captured identity; never retains source arrays or a report."""

    generation: int
    rank: int
    metric_key: str
    chain_a: str
    partners: tuple[str, ...]
    kind: Literal["ipsae_profile", "contributor_mask"]
    side: Literal["both", "a", "b"] = "both"

    def __post_init__(self) -> None:
        if self.kind not in {"ipsae_profile", "contributor_mask"}:
            raise ValueError("Unknown interface projection.")
        if self.side not in {"both", "a", "b"}:
            raise ValueError("Unknown interface projection side.")


@dataclass(frozen=True)
class InterfaceProjectionResult:
    selection: InterfaceProjectionSelection
    token_map: TokenMap
    object_name: str
    values: np.ndarray
    title: str
    description: str

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, order="C", copy=True)
        if values.shape != (len(self.token_map),):
            raise ValueError("Projection values must match the canonical token map.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class InterfaceSelectionRequest:
    rank: int
    score: InterfaceScoreResult
    side: Literal["both", "a", "b"] = "both"


class InterfaceReportObserver(Protocol):
    """Typed user events from the child window; never lazy-load continuations."""

    def analyze_current_interfaces(self, options: InterfaceAnalysisOptions) -> None: ...
    def select_interface_contributors(
        self, request: InterfaceSelectionRequest
    ) -> None: ...
    def clear_interface_selection(self) -> None: ...
    def export_interface_report(self, path: str) -> None: ...
    def export_interface_contacts(
        self, request: InterfaceContactExportRequest
    ) -> None: ...
    def interface_window_closed(self) -> None: ...
    def project_interface_residues(
        self,
        selection: InterfaceProjectionSelection,
        action: Literal["interface_color", "interface_profile"],
    ) -> None: ...


class InterfaceReportViewPort(Protocol):
    """Deterministic rendering, distinct from presentation/window ownership."""

    def render(self, report: InterfaceReport, *, current: bool) -> None: ...
    def set_current(self, current: bool) -> None: ...
    def set_busy(self, busy: bool) -> None: ...
