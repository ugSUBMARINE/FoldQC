"""Preparation of presentation-ready plots from resolved analysis values."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

from . import metrics, plot_data
from .analysis import ComputedMetric, PlotOptions, ResolvedAnalysis
from .plot_preparation import PlotPreparationService
from .presentation import PreparedPlot
from .token_map import TokenMap

PlotPreparer: TypeAlias = Callable[
    [ResolvedAnalysis, tuple[ComputedMetric, ...], PlotOptions], PreparedPlot
]


class PlotCoordinator:
    def __init__(self, preparation: PlotPreparationService) -> None:
        self._preparation = preparation
        self._dispatch: dict[str, PlotPreparer] = {
            "line": self._line,
            "distribution": self._distribution,
            "matrix": self._matrix,
            "pae_summary": self._pae_summary,
            "pde_summary": self._pde_summary,
            "binding_site_fingerprint": self._fingerprint,
            "ensemble_site_summary": self._ensemble_site_summary,
        }
        registered = {spec.key for spec in metrics.PLOTS}
        if set(self._dispatch) != registered:
            raise RuntimeError("Plot preparation dispatch does not match PlotRegistry.")

    def prepare(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
        options: PlotOptions,
    ) -> PreparedPlot:
        action = resolved.request.action
        preparer = self._dispatch.get(action)
        if preparer is None:
            raise ValueError(f"Unknown plot action: {action}")
        return preparer(resolved, computed, options)

    @staticmethod
    def _selection_maps(
        resolved: ResolvedAnalysis,
    ) -> tuple[list[TokenMap] | None, list[str] | None]:
        if resolved.target.kind != "ensemble_group":
            return None, None
        return (
            [member.model_state.token_map for member in resolved.members],
            [member.obj_name for member in resolved.members],
        )

    def _line(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
        options: PlotOptions,
    ) -> PreparedPlot:
        from . import plots

        x_values, series, ylabel = self._preparation.line_data(resolved, computed)
        spec = resolved.metric_spec
        if spec is None:
            raise ValueError("Line plots require a metric.")
        indices = list(map(int, x_values.tolist()))
        boundaries, labels = plot_data.chain_boundaries(
            resolved.target.token_map, indices, original_x=True
        )
        title = f"{spec.label} ({resolved.target.label})"
        figure = plots.make_line_plot(
            x_values,
            series,
            title=title,
            ylabel=ylabel,
            ymin=options.vmin,
            ymax=options.vmax,
            chain_boundaries=boundaries,
            chain_labels=labels,
        )
        token_maps, object_names = self._selection_maps(resolved)
        plots.attach_viewer_selection_metadata(
            figure,
            kind="line",
            token_map=resolved.target.token_map,
            obj_name=resolved.target.obj_name,
            token_maps=token_maps,
            token_map_obj_names=object_names,
            token_indices=indices,
            x_positions=x_values.tolist(),
        )
        return PreparedPlot(figure, title)

    def _distribution(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
        options: PlotOptions,
    ) -> PreparedPlot:
        from . import plots

        values, indices = self._preparation.distribution_values(resolved, computed)
        spec = resolved.metric_spec
        if spec is None:
            raise ValueError("Distribution plots require a metric.")
        title = f"{spec.label} distribution ({resolved.target.label})"
        positions: list[float]
        widths: list[float]
        if spec.key == "plddt_class":
            labels, counts, groups, total = plot_data.plddt_class_distribution_groups(
                values, indices
            )
            figure = plots.make_plddt_class_bar_plot(
                labels, counts, total=total, title=title
            )
            positions = [float(value) for value in range(len(labels))]
            widths = [0.8] * len(labels)
        elif spec.is_domain_label:
            labels, counts, groups, colors = plot_data.domain_label_distribution_groups(
                values, indices
            )
            figure = plots.make_categorical_bar_plot(
                labels, counts, title=title, colors=colors
            )
            positions = [float(value) for value in range(len(labels))]
            widths = [0.8] * len(labels)
        else:
            edges, groups, positions, widths = plot_data.histogram_distribution_groups(
                values, indices
            )
            figure = plots.make_histogram_plot(
                values, title=title, xlabel=spec.label, bin_edges=edges
            )
        token_maps, object_names = self._selection_maps(resolved)
        plots.attach_viewer_selection_metadata(
            figure,
            kind="bars",
            token_map=resolved.target.token_map,
            obj_name=resolved.target.obj_name,
            token_maps=token_maps,
            token_map_obj_names=object_names,
            bar_token_indices=groups,
            bar_x_positions=positions,
            bar_widths=widths,
        )
        return PreparedPlot(figure, title)

    def _matrix(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
        options: PlotOptions,
    ) -> PreparedPlot:
        from . import plots

        (
            matrix,
            rows,
            columns,
            title,
            label,
            row_labels,
            column_labels,
            cell_text,
        ) = self._preparation.matrix_data(resolved)
        spec = resolved.metric_spec
        source = None if spec is None or spec.matrix is None else spec.matrix.source
        if source == "chain_iptm":
            row_boundaries: list[float] = []
            column_boundaries: list[float] = []
            xlabel, ylabel = "Chain j", "Chain i"
        else:
            row_boundaries, _ = plot_data.chain_boundaries(
                resolved.target.token_map, rows
            )
            column_boundaries, _ = plot_data.chain_boundaries(
                resolved.target.token_map, columns
            )
            xlabel, ylabel = "Scored token j", "Alignment anchor i"
        full_title = f"{title} ({resolved.target.label})"
        figure = plots.make_matrix_plot(
            matrix,
            title=full_title,
            token_map=resolved.target.token_map,
            row_indices=rows,
            col_indices=columns,
            row_labels=row_labels,
            col_labels=column_labels,
            cell_text=None if cell_text is None else cell_text.tolist(),
            row_chain_boundaries=row_boundaries,
            col_chain_boundaries=column_boundaries,
            vmin=0.0 if options.vmin is None else options.vmin,
            vmax=options.vmax,
            palette=options.palette_key,
            reverse_palette=options.reverse_palette,
            xlabel=xlabel,
            ylabel=ylabel,
            colorbar_label=label,
        )
        if source != "chain_iptm":
            token_maps, object_names = self._selection_maps(resolved)
            plots.attach_viewer_selection_metadata(
                figure,
                kind="matrix",
                token_map=resolved.target.token_map,
                obj_name=resolved.target.obj_name,
                token_maps=token_maps,
                token_map_obj_names=object_names,
                row_indices=rows,
                col_indices=columns,
            )
        return PreparedPlot(figure, full_title)

    def _summary(
        self,
        resolved: ResolvedAnalysis,
        options: PlotOptions,
        family: Literal["pae", "pde"],
    ) -> PreparedPlot:
        from . import plots

        x_values, series, ylabel = self._preparation.summary_data(resolved, family)
        indices = list(map(int, x_values.tolist()))
        boundaries, labels = plot_data.chain_boundaries(
            resolved.target.token_map, indices, original_x=True
        )
        family_label = family.upper()
        title = f"{family_label} summary ({resolved.target.label})"
        figure = plots.make_line_plot(
            x_values,
            series,
            title=title,
            ylabel=ylabel,
            ymin=options.vmin,
            ymax=options.vmax,
            chain_boundaries=boundaries,
            chain_labels=labels,
            show_legend=True,
        )
        token_maps, object_names = self._selection_maps(resolved)
        plots.attach_viewer_selection_metadata(
            figure,
            kind="line",
            token_map=resolved.target.token_map,
            obj_name=resolved.target.obj_name,
            token_maps=token_maps,
            token_map_obj_names=object_names,
            token_indices=indices,
            x_positions=x_values.tolist(),
        )
        return PreparedPlot(figure, title)

    def _pae_summary(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
        options: PlotOptions,
    ) -> PreparedPlot:
        return self._summary(resolved, options, "pae")

    def _pde_summary(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
        options: PlotOptions,
    ) -> PreparedPlot:
        return self._summary(resolved, options, "pde")

    def _fingerprint(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
        options: PlotOptions,
    ) -> PreparedPlot:
        from . import plots

        context = resolved.members[0].metric_context
        binding_indices = list(context.contact_indices)
        if not binding_indices:
            raise ValueError("No binding-site residues are available for this plot.")
        series = self._preparation.fingerprint_data(resolved)
        title = f"Binding-site confidence fingerprint ({resolved.target.label})"
        figure = plots.make_binding_site_fingerprint(
            resolved.target.token_map,
            binding_indices,
            plddt=series["plddt"],
            plddt_std=series["plddt_std"],
            pae_to_ligand=series["pae_to_ligand"],
            pae_to_ligand_std=series["pae_to_ligand_std"],
            pae_from_ligand=series["pae_from_ligand"],
            pae_from_ligand_std=series["pae_from_ligand_std"],
            pde_to_ligand=series["pde_to_ligand"],
            pde_to_ligand_std=series["pde_to_ligand_std"],
            interaction_prob_to_ligand=series["interaction_prob_to_ligand"],
            interaction_prob_to_ligand_std=series["interaction_prob_to_ligand_std"],
            title=title,
        )
        displayed = binding_indices[: plots.MAX_BINDING_SITE_RESIDUES]
        token_maps, object_names = self._selection_maps(resolved)
        plots.attach_viewer_selection_metadata(
            figure,
            kind="bars",
            token_map=resolved.target.token_map,
            obj_name=resolved.target.obj_name,
            token_maps=token_maps,
            token_map_obj_names=object_names,
            token_indices=displayed,
            x_positions=list(range(len(displayed))),
        )
        return PreparedPlot(figure, title)

    def _ensemble_site_summary(
        self,
        resolved: ResolvedAnalysis,
        computed: tuple[ComputedMetric, ...],
        options: PlotOptions,
    ) -> PreparedPlot:
        from . import plots

        labels, series, site_indices = self._preparation.ensemble_site_data(resolved)
        if not series:
            raise ValueError("No pLDDT, PAE, or PDE site-summary data are available.")
        cutoff = resolved.request.cutoff_angstrom
        if cutoff is None:
            raise ValueError("Ensemble site summary requires a cutoff.")
        title = (
            "Ensemble site summary\n"
            f"Reference: {resolved.request.reference_selection}, cutoff {cutoff:g} Å"
        )
        figure = plots.make_ensemble_site_summary_plot(labels, series, title=title)
        plots.attach_ensemble_site_summary_metadata(
            figure,
            member_obj_names=[member.obj_name for member in resolved.members],
            member_token_maps=[
                member.model_state.token_map for member in resolved.members
            ],
            site_indices=site_indices,
            selection_name="foldqc_ensemble_site",
        )
        return PreparedPlot(figure, "Ensemble site summary")
