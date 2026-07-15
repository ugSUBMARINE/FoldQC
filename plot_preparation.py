"""Viewer-neutral shaping of resolved plot data."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np

from . import plot_data
from .analysis import ComputedMetric, ResolvedAnalysis, ResolvedMemberContext

LineSeries: TypeAlias = (
    tuple[str, np.ndarray, np.ndarray | None]
    | tuple[str, np.ndarray, np.ndarray | None, str]
)
MatrixData: TypeAlias = tuple[
    np.ndarray,
    list[int],
    list[int],
    str,
    str,
    list[str] | None,
    list[str] | None,
    np.ndarray | None,
]
FingerprintData: TypeAlias = dict[str, np.ndarray | None]
SiteSummaryRow: TypeAlias = tuple[ResolvedMemberContext, list[int], dict[str, float]]


class PlotPreparationService:
    """Shape immutable resolved data without presentation or widget access."""

    def line_data(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
    ) -> tuple[np.ndarray, list[LineSeries], str]:
        spec = resolved.metric_spec
        if spec is None or not computed:
            raise ValueError("Line plots require computed metric values.")
        reference = list(computed[0].metric_context.reference_indices)
        use_reference = bool(reference) and "line" in spec.reference_scoped_plots
        indices = reference if use_reference else list(range(len(computed[0].values)))
        series: list[LineSeries]
        if len(computed) == 1 or spec.ensemble_level:
            series = [(spec.label, computed[0].values[indices], None)]
        else:
            mean, std = plot_data.nan_mean_std(
                [item.values for item in computed], len(computed[0].values)
            )
            if mean is None or std is None:
                raise ValueError("No ensemble values are available for this plot.")
            series = [(f"{spec.label} mean", mean[indices], std[indices])]
        return np.asarray(indices, dtype=np.int32), series, spec.line_ylabel

    def distribution_values(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
    ) -> tuple[np.ndarray, list[int]]:
        spec = resolved.metric_spec
        if spec is None or not computed:
            raise ValueError("Distribution plots require computed metric values.")
        reference = list(computed[0].metric_context.reference_indices)
        use_reference = (
            bool(reference) and "distribution" in spec.reference_scoped_plots
        )
        indices = reference if use_reference else list(range(len(computed[0].values)))
        if len(computed) == 1 or spec.ensemble_level:
            values: np.ndarray | None = computed[0].values
        else:
            values, _std = plot_data.nan_mean_std(
                [item.values for item in computed], len(computed[0].values)
            )
            if values is None:
                raise ValueError("No ensemble values are available for this plot.")
        return np.asarray(values, dtype=np.float32)[indices], indices

    def matrix_data(self, resolved: ResolvedAnalysis) -> MatrixData:
        spec = resolved.metric_spec
        if spec is None or spec.matrix is None:
            raise ValueError("Matrix plots require a matrix-backed metric.")
        source = spec.matrix.source
        if source == "chain_iptm":
            return plot_data.chain_iptm_matrix_plot_data(
                target_kind=resolved.target.kind,
                data=resolved.members[0].model_state.data,
                token_map=resolved.target.token_map,
                title=spec.matrix.title,
                label=spec.matrix.colorbar_label,
                members=[member.model_state for member in resolved.members],
            )
        matrices = [
            np.asarray(getattr(member.model_state.data, source), dtype=np.float32)
            for member in resolved.members
        ]
        matrix = matrices[0] if len(matrices) == 1 else np.nanmean(matrices, axis=0)
        reference = list(resolved.members[0].metric_context.reference_indices)
        if spec.key in {"pae_row_mean", "pae_col_to_sel"} and reference:
            rows = reference
            columns = list(range(matrix.shape[1]))
        elif spec.key == "pae_sym_within_sel" and reference:
            rows = reference
            columns = reference
        else:
            rows = list(range(matrix.shape[0]))
            columns = reference or list(range(matrix.shape[1]))
        title = spec.matrix.title
        if len(matrices) > 1:
            title = f"{title} — ensemble mean"
        return (
            matrix[np.ix_(rows, columns)],
            rows,
            columns,
            title,
            spec.matrix.colorbar_label,
            None,
            None,
            None,
        )

    def summary_data(
        self, resolved: ResolvedAnalysis, family: Literal["pae", "pde"]
    ) -> tuple[np.ndarray, list[LineSeries], str]:
        token_map = resolved.target.token_map
        reference = list(resolved.members[0].metric_context.reference_indices)
        indices = reference or list(range(len(token_map)))
        if len(resolved.members) == 1:
            series = plot_data.summary_series_for_data(
                family,
                resolved.members[0].model_state.data,
                token_map,
            )
        else:
            series = plot_data.summary_series_for_ensemble(
                family,
                [member.model_state.data for member in resolved.members],
                token_map,
                token_maps=[
                    member.model_state.token_map for member in resolved.members
                ],
            )
        sliced: list[LineSeries] = []
        for item in series:
            label, values, std = item[0], item[1], item[2]
            shaped = (
                label,
                np.asarray(values, dtype=np.float32)[indices],
                None if std is None else np.asarray(std, dtype=np.float32)[indices],
            )
            sliced.append((*shaped, item[3]) if len(item) == 4 else shaped)
        ylabel = "PAE gap (Å)" if family == "pae" else "PDE gap (Å)"
        return np.asarray(indices, dtype=np.int32), sliced, ylabel

    def fingerprint_data(self, resolved: ResolvedAnalysis) -> FingerprintData:
        reference = list(resolved.members[0].metric_context.reference_indices)
        if len(resolved.members) == 1:
            return plot_data.fingerprint_series_for_single(
                resolved.members[0].model_state.data, reference
            )
        return plot_data.fingerprint_series_for_ensemble(
            [member.model_state.data for member in resolved.members],
            reference,
            size=len(resolved.target.token_map),
        )

    def ensemble_site_data(
        self, resolved: ResolvedAnalysis
    ) -> tuple[list[str], list[tuple[str, np.ndarray, str]], list[list[int]]]:
        rows: list[SiteSummaryRow] = []
        for member in resolved.members:
            site_indices = list(
                dict.fromkeys(
                    member.metric_context.reference_indices
                    + member.metric_context.contact_indices
                )
            )
            rows.append(
                (
                    member,
                    site_indices,
                    plot_data.site_summary_values(
                        member.model_state.data, site_indices
                    ),
                )
            )
        labels = [member.label for member, _indices, _values in rows]
        series: list[tuple[str, np.ndarray, str]] = []
        for label, key, color in (
            ("mean pLDDT", "plddt", "steelblue"),
            ("PAE mean", "pae", "tomato"),
            ("PDE mean", "pde", "goldenrod"),
        ):
            values = np.asarray([row[2][key] for row in rows], dtype=np.float32)
            if np.any(np.isfinite(values)):
                series.append((label, values, color))
        return labels, series, [row[1] for row in rows]
