"""Viewer-neutral analysis requests, target resolution, and lazy-load planning.

This module is the common contract shared by coloring, plotting, and export.
It deliberately contains no Qt or PyMOL imports.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from . import metrics
from .dependencies import DependencyKey
from .ensemble import EnsembleMember, EnsembleState
from .gui_state import MetricContext, PluginState, ResolvedTarget
from .loader_models import DataCapability, PredictionFiles
from .model_state import ModelState

AnalysisAction = Literal[
    "color",
    "export",
    "line",
    "distribution",
    "matrix",
    "pae_summary",
    "pde_summary",
    "binding_site_fingerprint",
    "ensemble_site_summary",
]
CoveragePolicy = Literal["strict", "available"]
NoticeSeverity = Literal["information", "warning", "error"]

_PLOT_ACTIONS = frozenset(spec.key for spec in metrics.PLOTS)
_PARTIAL_ACTIONS = frozenset({"binding_site_fingerprint", "ensemble_site_summary"})


@dataclass(frozen=True)
class AnalysisRequest:
    """Immutable user intent captured before any lazy or asynchronous work."""

    action: AnalysisAction
    target_name: str
    metric_key: str | None = None
    reference_selection: str = ""
    cutoff_angstrom: float | None = None
    ui_revision: int = 0

    def __post_init__(self) -> None:
        if not self.target_name.strip():
            raise ValueError("AnalysisRequest requires a viewer target.")
        if self.action in {"color", "export"} and self.metric_key is None:
            raise ValueError(f"{self.action} requires a metric key.")
        if self.action in _PLOT_ACTIONS:
            plot = metrics.PLOTS.require(self.action)
            if plot.requires_metric and self.metric_key is None:
                raise ValueError(f"Plot {self.action!r} requires a metric key.")
        if self.metric_key is not None:
            metrics.METRICS.require(self.metric_key)
        if self.cutoff_angstrom is not None:
            value = float(self.cutoff_angstrom)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("Cutoff / threshold must be greater than 0 Å.")


@dataclass(frozen=True)
class ColorOptions:
    palette_key: str
    reverse_palette: bool = False
    vmin: float | None = None
    vmax: float | None = None

    def __post_init__(self) -> None:
        for label, value in (("minimum", self.vmin), ("maximum", self.vmax)):
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"Color scale {label} must be finite or automatic.")
        if self.vmin is not None and self.vmax is not None and self.vmin > self.vmax:
            raise ValueError("Color scale minimum must not exceed its maximum.")


@dataclass(frozen=True)
class PlotOptions:
    palette_key: str
    reverse_palette: bool = False
    vmin: float | None = None
    vmax: float | None = None


@dataclass(frozen=True)
class ExportOptions:
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True)
class DeferredAnalysisAction:
    """Exact action resumed after a lazy-load commit."""

    request: AnalysisRequest
    options: ColorOptions | PlotOptions | ExportOptions | None = None


@dataclass(frozen=True)
class AnalysisProblem:
    """Typed, presentation-neutral failure from request preflight."""

    code: str
    message: str
    severity: NoticeSeverity = "warning"
    affected_models: tuple[str, ...] = ()


class AnalysisPreflightError(ValueError):
    def __init__(self, problem: AnalysisProblem) -> None:
        super().__init__(problem.message)
        self.problem = problem


@dataclass(frozen=True)
class ResolvedMemberContext:
    rank: int
    label: str
    obj_name: str
    model_state: ModelState
    expected_version: int
    metric_context: MetricContext = MetricContext()
    member: EnsembleMember | None = None


@dataclass(frozen=True)
class ResolvedAnalysis:
    """A request bound to canonical model identities and registry contracts."""

    request: AnalysisRequest
    target: ResolvedTarget
    members: tuple[ResolvedMemberContext, ...]
    metric_spec: metrics.MetricSpec | None
    plot_spec: metrics.PlotSpec | None
    required_capabilities: frozenset[DataCapability]
    dependency_keys: tuple[DependencyKey, ...]
    coverage_policy: CoveragePolicy
    prediction_files: PredictionFiles
    ensemble_identity: EnsembleState | None

    def with_member_contexts(
        self, contexts: tuple[MetricContext, ...]
    ) -> ResolvedAnalysis:
        if len(contexts) != len(self.members):
            raise ValueError("Metric contexts must correspond to resolved members.")
        return replace(
            self,
            members=tuple(
                replace(member, metric_context=context)
                for member, context in zip(self.members, contexts)
            ),
        )

    def validate_current(self, state: PluginState, *, ui_revision: int) -> bool:
        if state.pred_files is not self.prediction_files:
            return False
        if self.request.ui_revision != ui_revision:
            return False
        if self.ensemble_identity is not state.ensemble:
            return False
        return all(
            state.model_states.get(member.rank) is member.model_state
            and member.model_state.version == member.expected_version
            for member in self.members
        )


@dataclass(frozen=True)
class DataLoadRequirement:
    rank: int
    model_label: str
    capabilities: frozenset[DataCapability]
    model_state: ModelState
    expected_version: int
    expected_ensemble: EnsembleState | None = None
    phase_arrays: tuple[str, ...] = ()

    def load_kwargs(self) -> dict[str, bool]:
        return {
            "load_pae": "pae" in self.capabilities,
            "load_pde": "pde" in self.capabilities,
            "load_contact_probs": "contact_probs" in self.capabilities,
            "load_token_plddt": "plddt" in self.capabilities,
        }


@dataclass(frozen=True)
class DataLoadPlan:
    """Deduplicated lazy data needed to resume one exact analysis request."""

    analysis: ResolvedAnalysis
    requirements: tuple[DataLoadRequirement, ...]

    @property
    def is_empty(self) -> bool:
        return not self.requirements


@dataclass(frozen=True)
class ComputedMetric:
    rank: int
    label: str
    obj_name: str
    model_state: ModelState
    metric_context: MetricContext
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(np.asarray(self.values, dtype=np.float32))
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


def _loaded_capabilities(state: ModelState) -> frozenset[DataCapability]:
    data = state.data
    return frozenset(
        capability
        for capability, attribute in (
            ("plddt", "token_plddt"),
            ("pae", "pae"),
            ("pde", "pde"),
            ("contact_probs", "contact_probs"),
        )
        if getattr(data, attribute) is not None
    )


class AnalysisResolver:
    """Resolve request targets and registry requirements from canonical state."""

    def resolve(self, request: AnalysisRequest, state: PluginState) -> ResolvedAnalysis:
        files = state.pred_files
        if files is None:
            raise AnalysisPreflightError(
                AnalysisProblem("no_prediction", "No prediction output loaded.")
            )
        target = self.resolve_target(request.target_name, state)
        metric = (
            None
            if request.metric_key is None
            else metrics.METRICS.require(request.metric_key)
        )
        plot = (
            metrics.PLOTS.require(request.action)
            if request.action in _PLOT_ACTIONS
            else None
        )
        if (
            plot is not None
            and metric is not None
            and request.action not in metric.plot_modes
        ):
            raise AnalysisPreflightError(
                AnalysisProblem(
                    "unsupported_plot",
                    f"{plot.label} is not available for {metric.label}.",
                    severity="information",
                )
            )
        if metric is not None and metric.ensemble_level and state.ensemble is None:
            raise AnalysisPreflightError(
                AnalysisProblem(
                    "ensemble_required",
                    "This metric requires an active ensemble.",
                    severity="information",
                )
            )
        coverage: CoveragePolicy = (
            "available" if request.action in _PARTIAL_ACTIONS else "strict"
        )
        capabilities = frozenset() if metric is None else metric.load_capabilities
        dependencies = tuple(
            dict.fromkeys(
                (plot.dependency_keys if plot is not None else ())
                + (metric.dependency_keys if metric is not None else ())
            )
        )
        contexts = tuple(
            ResolvedMemberContext(
                rank=model_state.rank,
                label=files.model(model_state.rank).display_label,
                obj_name=(
                    target.members[index].obj_name
                    if target.members
                    else target.obj_name
                ),
                model_state=model_state,
                expected_version=model_state.version,
                member=(target.members[index] if target.members else None),
            )
            for index, model_state in enumerate(target.model_states)
        )
        return ResolvedAnalysis(
            request=request,
            target=target,
            members=contexts,
            metric_spec=metric,
            plot_spec=plot,
            required_capabilities=capabilities,
            dependency_keys=dependencies,
            coverage_policy=coverage,
            prediction_files=files,
            ensemble_identity=state.ensemble,
        )

    def resolve_target(self, target_name: str, state: PluginState) -> ResolvedTarget:
        ensemble_state = state.ensemble
        if ensemble_state is not None and target_name == ensemble_state.group_name:
            members = tuple(sorted(ensemble_state.members, key=lambda item: item.rank))
            if not members:
                raise AnalysisPreflightError(
                    AnalysisProblem(
                        "inactive_ensemble", "The ensemble target is not active."
                    )
                )
            states = self._member_states(members, state)
            return ResolvedTarget(
                "ensemble_group",
                target_name,
                members[0].obj_name,
                states,
                members,
            )
        if ensemble_state is not None:
            for member in ensemble_state.members:
                if member.obj_name == target_name:
                    states = self._member_states((member,), state)
                    return ResolvedTarget(
                        "ensemble_member",
                        member.obj_name,
                        member.obj_name,
                        states,
                        (member,),
                    )
        active = state.active_model_state
        if active is None:
            raise AnalysisPreflightError(
                AnalysisProblem("no_model", "No prediction data loaded.")
            )
        return ResolvedTarget("single", target_name, target_name, (active,))

    @staticmethod
    def _member_states(
        members: tuple[EnsembleMember, ...], state: PluginState
    ) -> tuple[ModelState, ...]:
        resolved = []
        for member in members:
            model_state = state.model_states.get(member.rank)
            if model_state is None:
                raise AnalysisPreflightError(
                    AnalysisProblem(
                        "stale_ensemble",
                        f"Ensemble model_{member.rank} is no longer present in the canonical model store.",
                    )
                )
            resolved.append(model_state)
        return tuple(resolved)


def build_data_load_plan(analysis: ResolvedAnalysis) -> DataLoadPlan:
    """Build a strict, per-rank load plan without touching provider files."""

    requirements: list[DataLoadRequirement] = []
    unavailable: list[str] = []
    for member in analysis.members:
        missing = analysis.required_capabilities - _loaded_capabilities(
            member.model_state
        )
        if not missing:
            continue
        advertised = frozenset(
            capability
            for capability in missing
            if analysis.prediction_files.model_supports(member.rank, capability)
        )
        unsupported = missing - advertised
        if unsupported and analysis.coverage_policy == "strict":
            unavailable.append(member.label)
            continue
        if advertised:
            requirements.append(
                DataLoadRequirement(
                    member.rank,
                    member.label,
                    advertised,
                    member.model_state,
                    member.expected_version,
                    analysis.ensemble_identity,
                    tuple(
                        {
                            "pae": "PAE",
                            "pde": "PDE",
                            "contact_probs": "interaction probabilities",
                            "plddt": "pLDDT",
                        }[capability]
                        for capability in sorted(advertised)
                    ),
                )
            )
    if unavailable:
        raise AnalysisPreflightError(
            AnalysisProblem(
                "missing_capability",
                "Required data are unavailable for: " + ", ".join(unavailable),
                affected_models=tuple(unavailable),
            )
        )
    return DataLoadPlan(analysis, tuple(requirements))
