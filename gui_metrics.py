"""Viewer-context resolution and immutable metric computation."""

from __future__ import annotations

import numpy as np

from . import compute
from .analysis import (
    AnalysisPreflightError,
    ComputedMetric,
    ResolvedAnalysis,
)
from .gui_services import ObjectPaintMapping, ViewerPort
from .gui_state import MetricContext, PluginState
from .presentation import ChoiceOption, ChoiceRequest, Notice, PresentationPort
from .token_map import TokenMap

APP_TITLE = "FoldQC"


class MetricComputationService:
    """Resolve viewer inputs once and compute each targeted model once."""

    def __init__(
        self,
        state: PluginState,
        viewer: ViewerPort,
        presenter: PresentationPort,
        accepted_overlap_warnings: set[tuple[str, str]],
    ) -> None:
        self._state = state
        self._viewer = viewer
        self._presenter = presenter
        self._accepted_overlap_warnings = accepted_overlap_warnings

    def resolve_contexts(self, resolved: ResolvedAnalysis) -> ResolvedAnalysis:
        metric = resolved.metric_spec
        request = resolved.request
        requires_plot_reference = request.action in {
            "binding_site_fingerprint",
            "ensemble_site_summary",
        }
        optional_plot_reference = request.action in {
            "line",
            "distribution",
            "matrix",
            "pae_summary",
            "pde_summary",
        }
        if (
            metric is None
            and not requires_plot_reference
            and not optional_plot_reference
        ):
            return resolved
        contexts: list[MetricContext] = []
        for member in resolved.members:
            token_map = member.model_state.token_map
            reference_indices: tuple[int, ...] = ()
            contact_indices: tuple[int, ...] = ()
            needs_reference = requires_plot_reference or bool(
                metric is not None and metric.needs_reference
            )
            resolve_reference = needs_reference or (
                optional_plot_reference and bool(request.reference_selection)
            )
            if resolve_reference:
                if needs_reference and not request.reference_selection:
                    raise AnalysisPreflightError(
                        Notice(
                            "reference_required",
                            "This property requires a reference selection.",
                        )
                    )
                reference_indices = tuple(
                    self._viewer.selection_token_indices(
                        token_map,
                        request.reference_selection,
                        obj_name=member.obj_name,
                    )
                )
                if not reference_indices:
                    raise AnalysisPreflightError(
                        Notice(
                            "empty_reference",
                            f"Reference selection '{request.reference_selection}' "
                            f"matched no tokens in {member.obj_name}.",
                        )
                    )
            needs_contact = requires_plot_reference or bool(
                metric is not None and metric.needs_contact_shell
            )
            if needs_contact:
                cutoff = request.cutoff_angstrom
                if cutoff is None:
                    raise AnalysisPreflightError(
                        Notice("cutoff_required", "This property requires a cutoff.")
                    )
                raw = self._viewer.tokens_within_distance(
                    token_map,
                    member.obj_name,
                    request.reference_selection,
                    cutoff,
                )
                reference = set(reference_indices)
                contact_indices = tuple(
                    index
                    for index in raw
                    if index not in reference and not token_map[index].is_hetatm
                )
                if not contact_indices:
                    raise AnalysisPreflightError(
                        Notice(
                            "empty_contact_shell",
                            "No polymer binding-site residues were found within "
                            f"{cutoff:g} Å of the reference selection.",
                        )
                    )
            contexts.append(
                MetricContext(
                    reference_selection=request.reference_selection,
                    reference_indices=reference_indices,
                    contact_indices=contact_indices,
                    cutoff_angstrom=request.cutoff_angstrom,
                )
            )
        return resolved.with_member_contexts(tuple(contexts))

    def compute(self, resolved: ResolvedAnalysis) -> tuple[ComputedMetric, ...]:
        metric = resolved.metric_spec
        if metric is None:
            return ()
        if metric.ensemble_level:
            values = self._compute_ensemble_metric(metric.key)
            member = resolved.members[0]
            return (
                ComputedMetric(
                    member.rank,
                    resolved.target.label,
                    resolved.target.obj_name,
                    member.model_state,
                    member.metric_context,
                    values,
                ),
            )
        computed: list[ComputedMetric] = []
        for member in resolved.members:
            context = member.metric_context
            try:
                values = compute.compute_metric(
                    metric.key,
                    member.model_state.data,
                    member.model_state.token_map,
                    ref_indices=list(context.reference_indices),
                    contact_indices=list(context.contact_indices),
                    cutoff=context.cutoff_angstrom,
                )
            except compute.MetricComputationError as exc:
                raise AnalysisPreflightError(
                    Notice(
                        "metric_computation_failed",
                        str(exc),
                        affected_models=(member.label,),
                    )
                ) from exc
            if len(values) != len(member.model_state.token_map):
                raise AnalysisPreflightError(
                    Notice(
                        "metric_token_count",
                        f"Token count mismatch for {member.obj_name}: metric has "
                        f"{len(values)} values but the structure has "
                        f"{len(member.model_state.token_map)} tokens.",
                        severity="error",
                        affected_models=(member.label,),
                    )
                )
            computed.append(
                ComputedMetric(
                    member.rank,
                    member.label,
                    member.obj_name,
                    member.model_state,
                    context,
                    values,
                )
            )
        return tuple(computed)

    def _compute_ensemble_metric(self, key: str) -> np.ndarray:
        ensemble_state = self._state.ensemble
        if ensemble_state is None:
            raise AnalysisPreflightError(
                Notice("ensemble_required", "No active ensemble.")
            )
        if key == "ensemble_rmsd":
            return ensemble_state.rmsd
        if key == "ensemble_plddt_mean":
            return ensemble_state.plddt_mean
        if key == "ensemble_plddt_std":
            return ensemble_state.plddt_std
        raise AnalysisPreflightError(
            Notice("unknown_ensemble_metric", f"Unknown ensemble property: {key}")
        )

    @staticmethod
    def paint_mapping_cache_key(data: object, obj_name: str) -> tuple[str, str]:
        return str(getattr(data, "structure_path", "")), str(obj_name)

    def prepare_paint_mapping(
        self,
        token_map: TokenMap,
        obj_name: str,
        data: object,
    ) -> ObjectPaintMapping:
        key = self.paint_mapping_cache_key(data, obj_name)
        mappings = self._viewer.capture_paint_mappings()
        mapping, rebuilt = self._viewer.ensure_paint_mapping(
            obj_name, token_map, mappings.get(key)
        )
        mappings[key] = mapping
        self._viewer.restore_paint_mappings(mappings)
        if rebuilt:
            self._accepted_overlap_warnings.discard(key)
        return mapping

    def confirm_token_overlap(
        self,
        token_map: TokenMap,
        obj_name: str,
        data: object,
        mapping: ObjectPaintMapping,
        *,
        threshold: float = 0.50,
    ) -> bool:
        key = self.paint_mapping_cache_key(data, obj_name)
        if key in self._accepted_overlap_warnings:
            return True
        overlap = mapping.overlap
        if overlap.target_tokens <= 0 or overlap.target_coverage >= threshold:
            return True
        choice = self._presenter.choose(
            ChoiceRequest(
                "low_token_overlap",
                APP_TITLE,
                f"The selected viewer target '{obj_name}' has low overlap with "
                "the prediction token map. Apply the coloring anyway?",
                (
                    ChoiceOption("yes", "Apply", "accept"),
                    ChoiceOption("cancel", "Cancel", "reject"),
                ),
                default_key="cancel",
            )
        )
        if choice != "yes":
            return False
        self._accepted_overlap_warnings.add(key)
        return True
