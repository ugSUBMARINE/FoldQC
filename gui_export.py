"""CSV export of resolved immutable analysis results."""

from __future__ import annotations

from . import export
from .analysis import ComputedMetric, ExportOptions, ResolvedAnalysis
from .gui_state import PluginState
from .presentation import Notice, PresentationPort

APP_TITLE = "FoldQC"


class ExportCoordinator:
    def __init__(self, state: PluginState, presenter: PresentationPort) -> None:
        self._state = state
        self._presenter = presenter

    def execute(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
        options: ExportOptions,
    ) -> None:
        spec = resolved.metric_spec
        files = self._state.pred_files
        if spec is None or files is None:
            raise ValueError("CSV export requires a loaded prediction and metric.")
        if not computed:
            raise ValueError("No token rows were available for export.")

        ensemble_state = self._state.ensemble
        rows: list[dict[str, object]] = []
        entries = computed[:1] if spec.ensemble_level else computed
        for item in entries:
            include_ensemble = resolved.target.kind.startswith("ensemble")
            member_rank = None
            member_label = ""
            aggregate_kind: str = spec.aggregate_kind
            if include_ensemble and not spec.ensemble_level:
                member_rank = item.rank
                member_label = export.model_label_for_rank(
                    files, item.rank, fallback=item.label
                )
                aggregate_kind = "ensemble_member"
            context = item.metric_context
            rows.extend(
                export.build_token_rows(
                    pred_files=files,
                    data=item.model_state.data,
                    token_map=item.model_state.token_map,
                    values=item.values,
                    metric_key=spec.key,
                    reference_selection=context.reference_selection,
                    cutoff_angstrom=context.cutoff_angstrom,
                    reference_indices=context.reference_indices,
                    contact_indices=context.contact_indices,
                    include_ensemble=include_ensemble,
                    ensemble_group=(
                        "" if ensemble_state is None else ensemble_state.group_name
                    ),
                    ensemble_member_rank=member_rank,
                    ensemble_member_label=member_label,
                    ensemble_aligned=(
                        None if ensemble_state is None else ensemble_state.aligned
                    )
                    if include_ensemble
                    else None,
                    aggregate_kind=aggregate_kind,
                )
            )
        export.write_csv(options.path, rows)
        self._presenter.present_notice(
            Notice(
                "csv_exported",
                f"Exported {len(rows)} token rows to:\n{options.path}",
                severity="information",
                title=APP_TITLE,
            )
        )
