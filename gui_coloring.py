"""Transactional coloring of resolved immutable analysis results."""

from __future__ import annotations

from .analysis import ColorOptions, ComputedMetric, ResolvedAnalysis
from .context_service import ContextService
from .gui_metrics import MetricComputationService
from .gui_services import PaintTarget, StatisticsSelectionTarget, ViewerPort
from .presentation import PresentationPort
from .statistics_selection import StatisticsSelectionService
from .viewer_transactions import ColorbarChange, PaintTransaction


class ColoringCoordinator:
    """Apply already-computed values through one compensating transaction."""

    def __init__(
        self,
        viewer: ViewerPort,
        presenter: PresentationPort,
        context: ContextService,
        metrics_service: MetricComputationService,
        statistics_selection: StatisticsSelectionService,
    ) -> None:
        self._viewer = viewer
        self._presenter = presenter
        self._context = context
        self._metrics = metrics_service
        self._statistics_selection = statistics_selection

    def execute(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
        options: ColorOptions,
    ) -> None:
        spec = resolved.metric_spec
        if spec is None:
            raise ValueError("Coloring requires a metric.")
        if not computed:
            raise ValueError("Coloring produced no metric values.")

        if spec.ensemble_level:
            source = computed[0]
            entries = tuple(
                ComputedMetric(
                    member.rank,
                    member.label,
                    member.obj_name,
                    member.model_state,
                    member.metric_context,
                    source.values,
                )
                for member in resolved.members
            )
        else:
            entries = computed

        previous_mappings = self._viewer.capture_paint_mappings()
        targets: list[PaintTarget] = []
        try:
            for item in entries:
                token_map = item.model_state.token_map
                mapping = self._metrics.prepare_paint_mapping(
                    token_map, item.obj_name, item.model_state.data
                )
                if not self._metrics.confirm_token_overlap(
                    token_map,
                    item.obj_name,
                    item.model_state.data,
                    mapping,
                ):
                    self._viewer.restore_paint_mappings(previous_mappings)
                    return
                targets.append(
                    PaintTarget(item.obj_name, token_map, item.values, mapping)
                )

            kind = (
                "plddt_class"
                if spec.key == "plddt_class"
                else "categorical"
                if spec.is_domain_label
                else "continuous"
            )
            colorbar = ColorbarChange(
                "replace" if kind == "continuous" else "remove",
                palette=options.palette_key if kind == "continuous" else "",
                reverse_palette=options.reverse_palette,
            )
            transaction = PaintTransaction(self._viewer, tuple(targets), colorbar)
            if kind == "continuous":
                transaction.execute(
                    lambda: self._viewer.paint_continuous(
                        targets,
                        palette=options.palette_key,
                        reverse_palette=options.reverse_palette,
                        vmin=options.vmin,
                        vmax=options.vmax,
                        rebuild=False,
                    )
                )
            elif kind == "categorical":
                transaction.execute(
                    lambda: self._viewer.paint_categorical(targets, rebuild=False)
                )
            else:
                transaction.execute(
                    lambda: self._viewer.paint_plddt_classes(targets, rebuild=False)
                )
        except Exception:
            self._viewer.restore_paint_mappings(previous_mappings)
            raise

        label = resolved.target.label
        if len(entries) == 1 or spec.ensemble_level:
            first = entries[0]
            self._context.show_statistics_for_single(
                spec.key,
                label,
                first.values,
                include_plddt_classes=spec.key == "plddt_class",
                include_chain_stats=spec.key == "pde_chain_mean",
                include_domain_labels=spec.is_domain_label,
                token_map=first.model_state.token_map,
            )
        else:
            member_by_rank = {member.rank: member for member in resolved.target.members}
            self._context.show_statistics_for_members(
                spec.key,
                label,
                [
                    (member_by_rank[item.rank], item.values)
                    for item in entries
                    if item.rank in member_by_rank
                ],
                include_plddt_classes=spec.key == "plddt_class",
                include_chain_stats=spec.key == "pde_chain_mean",
                include_domain_labels=spec.is_domain_label,
            )
        self._statistics_selection.set_coloring_result(
            spec.key,
            tuple(
                StatisticsSelectionTarget(
                    item.obj_name,
                    item.model_state.token_map,
                    item.values,
                )
                for item in entries
            ),
        )
