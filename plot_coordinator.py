"""Plot request resolution, lazy coordination, and presentation dispatch."""

from __future__ import annotations

import numpy as np

from . import gui_rules, metrics, plot_data
from .gui_state import ResolvedTarget as _PlotTarget
from .loader_models import DataCapability
from .mol_viewer import get_viewer_name, selection_to_token_indices
from .plot_preparation import PlotPreparationService
from .token_map import TokenMap
from .workflow_presentation import present_error, present_information, present_warning

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()

_PLOT_PREPARERS = {
    "line": "_show_line_plot",
    "distribution": "_show_distribution_plot",
    "matrix": "_show_matrix_plot",
    "pae_summary": "_show_pae_summary_plot",
    "pde_summary": "_show_pde_summary_plot",
    "binding_site_fingerprint": "_show_binding_site_fingerprint",
    "ensemble_site_summary": "_show_ensemble_site_summary",
}
_REGISTERED_PLOT_KEYS = frozenset(spec.key for spec in metrics.PLOTS)
if frozenset(_PLOT_PREPARERS) != _REGISTERED_PLOT_KEYS:
    missing = sorted(_REGISTERED_PLOT_KEYS - _PLOT_PREPARERS.keys())
    unknown = sorted(_PLOT_PREPARERS.keys() - _REGISTERED_PLOT_KEYS)
    raise RuntimeError(
        "Plot preparation dispatch does not match PlotRegistry "
        f"(missing={missing}, unknown={unknown})."
    )


class PlotCoordinator(PlotPreparationService):
    def _resolve_plot_target(self) -> _PlotTarget | None:
        """Resolve the current viewer target into data and token-map context."""
        obj_name = self._get_obj_name()
        if obj_name is None:
            present_warning(self, APP_TITLE, f"No {VIEWER_NAME} target selected.")
            return None

        ensemble_state = getattr(self, "_ensemble", None)
        ensemble_group_name = (
            None if ensemble_state is None else ensemble_state.group_name
        )
        ensemble_members = None if ensemble_state is None else ensemble_state.members
        if obj_name == ensemble_group_name:
            members = sorted(ensemble_members or [], key=lambda member: member.rank)
            if not members:
                present_information(
                    self,
                    APP_TITLE,
                    "The ensemble target is not active.\nUse the Ensemble\u2026 button first.",
                )
                return None
            try:
                states = tuple(
                    self._canonical_state_for_ensemble_member(member)
                    for member in members
                )
            except ValueError as exc:
                present_warning(self, APP_TITLE, str(exc))
                return None
            reference = members[0]
            return _PlotTarget(
                kind="ensemble_group",
                label=obj_name,
                obj_name=reference.obj_name,
                model_states=states,
                members=tuple(members),
            )

        member = self._selected_ensemble_member(obj_name)
        if member is not None:
            try:
                state = self._canonical_state_for_ensemble_member(member)
            except ValueError as exc:
                present_warning(self, APP_TITLE, str(exc))
                return None
            return _PlotTarget(
                kind="ensemble_member",
                label=member.obj_name,
                obj_name=member.obj_name,
                model_states=(state,),
                members=(member,),
            )

        state = self._active_model_state
        if state is None:
            present_warning(self, APP_TITLE, "No prediction data loaded.")
            return None
        return _PlotTarget(
            kind="single",
            label=obj_name,
            obj_name=obj_name,
            model_states=(state,),
        )

    def _resolve_reference_indices(
        self,
        token_map: TokenMap,
        obj_name: str,
        *,
        required: bool = False,
    ) -> list[int] | None:
        """Resolve the Reference field to token indices, preserving token order."""
        ref_sel = self._analysis_reference_selection()
        if not ref_sel:
            if required:
                present_warning(
                    self,
                    APP_TITLE,
                    "This plot requires a reference selection.\n"
                    "Enter a viewer selection in the Reference field.",
                )
                return None
            return []

        indices = self._viewer_operation(
            "selection_token_indices",
            selection_to_token_indices,
            token_map,
            ref_sel,
            obj_name=obj_name,
        )
        if not indices:
            present_warning(
                self,
                APP_TITLE,
                f"Reference selection '{ref_sel}' matched no tokens in {obj_name}.",
            )
            return None
        return indices

    def _ensemble_site_summary_for_member(
        self,
        member,
        ref_sel: str,
        cutoff: float,
    ) -> dict:
        """Compute local ligand-site summary values for one ensemble member."""
        capabilities: frozenset[DataCapability] = frozenset(
            capability
            for capability in ("plddt", "pae", "pde")
            if self._member_supports_data(member, capability)
        )
        self._ensure_member_data_for_plot(member, capabilities)
        state = self._canonical_state_for_ensemble_member(member)
        ref_indices = self._viewer_operation(
            "selection_token_indices",
            selection_to_token_indices,
            state.token_map,
            ref_sel,
            obj_name=member.obj_name,
        )
        if not ref_indices:
            raise ValueError(
                f"Reference selection '{ref_sel}' matched no tokens in {member.obj_name}."
            )
        contact_indices = self._binding_site_token_indices(
            state.token_map, member.obj_name, ref_sel, ref_indices, cutoff
        )
        site_indices = list(dict.fromkeys(list(ref_indices) + list(contact_indices)))
        if not site_indices:
            raise ValueError(f"No site tokens are available for {member.obj_name}.")

        return {
            "member": member,
            "site_indices": site_indices,
            **plot_data.site_summary_values(state.data, site_indices),
        }

    def _compute_ensemble_site_summary_data(
        self,
        ref_sel: str,
        cutoff: float,
    ) -> tuple[list, list[str], list[tuple[str, np.ndarray, str]], list[list[int]]]:
        """Return ensemble members, labels, metric series, and site-token groups."""
        ensemble_state = getattr(self, "_ensemble", None)
        members = sorted(
            () if ensemble_state is None else ensemble_state.members,
            key=lambda member: member.rank,
        )
        if not members:
            raise ValueError(
                "The ensemble target is not active. Use Load Ensemble\u2026 first."
            )

        rows = [
            self._ensemble_site_summary_for_member(member, ref_sel, cutoff)
            for member in members
        ]
        labels = [f"model_{row['member'].rank}" for row in rows]
        site_indices = [row["site_indices"] for row in rows]
        metric_specs = [
            ("mean pLDDT", "plddt", "steelblue"),
            ("PAE mean", "pae", "tomato"),
            ("PDE mean", "pde", "goldenrod"),
        ]
        series: list[tuple[str, np.ndarray, str]] = []
        for label, key, color in metric_specs:
            values = np.asarray([row[key] for row in rows], dtype=np.float32)
            if np.any(np.isfinite(values)):
                series.append((label, values, color))
        return members, labels, series, site_indices

    def _show_selected_plot(self, plot_type: str | None = None) -> None:
        """Dispatch the selected plot type to its plot handler."""
        if plot_type is None and hasattr(self, "_plot_type_combo"):
            plot_type = self._plot_type_combo.currentData()
        if plot_type is None:
            present_warning(self, APP_TITLE, "No plot type selected.")
            return
        key = self._analysis_metric_key()
        state = gui_rules.plot_action_state(
            plot_type,
            key,
            self._current_target_kind(),
            bool(self._analysis_reference_selection()),
            bool(getattr(self, "_ensemble", None)),
            has_fingerprint_data=self._has_fingerprint_data(),
            has_pae_data=self._has_matrix_data_family("pae"),
            has_pde_data=self._has_matrix_data_family("pde"),
            has_multiple_chains=self._current_target_has_multiple_chains(),
        )
        if not state.enabled:
            present_information(
                self,
                APP_TITLE,
                state.reason or f"{plot_type} is not available.",
            )
            return
        plot_spec = metrics.PLOTS.require(plot_type)
        metric_spec = metrics.METRICS.find(key)
        dependency_keys = plot_spec.dependency_keys + (
            () if metric_spec is None else metric_spec.dependency_keys
        )
        if not self._ensure_dependencies(
            dependency_keys,
            feature_label=f"The {plot_spec.label.lower()} plot",
        ):
            return
        getattr(self, _PLOT_PREPARERS[plot_type])()

    def _show_pae_summary_plot(self) -> None:
        self._show_summary_plot("pae")

    def _show_pde_summary_plot(self) -> None:
        self._show_summary_plot("pde")

    def _fingerprint_capabilities(self, *, include_contact_probs: bool) -> frozenset:
        capabilities = {"plddt"}
        if self._target_any_supports_family("pae"):
            capabilities.add("pae")
        if self._target_any_supports_family("pde"):
            capabilities.add("pde")
        if include_contact_probs and self._target_any_supports_family("contact_probs"):
            capabilities.add("contact_probs")
        return frozenset(capabilities)

    def _show_line_plot(self) -> None:
        """Open a token-indexed line plot for the selected property."""
        target = self._resolve_plot_target()
        if target is None:
            return

        key = self._analysis_metric_key()
        spec = metrics.METRICS.require(key)
        if spec.is_domain_label:
            present_information(
                self,
                APP_TITLE,
                "Line plots are not available for PAE domain labels.\n"
                "Use Distribution to inspect cluster occupancy.",
            )
            return
        if spec.needs_contact_shell:
            metric_name = "PAE" if key == "pae_contact" else "PDE"
            present_information(
                self,
                APP_TITLE,
                f"Line plots are not available for {metric_name} "
                "contact-filtered values.\n"
                "Use Distribution or Matrix instead.",
            )
            return
        ref_indices = self._resolve_reference_indices(
            target.token_map, target.obj_name, required=spec.needs_reference
        )
        if ref_indices is None:
            return
        if self._defer_action_for_data(
            target,
            spec.load_capabilities,
            error_title=f"{APP_TITLE} - error",
            deferred_action=self.services.analysis.capture_current("line"),
        ):
            return

        try:
            if target.kind == "single":
                self._ensure_current_data_for_property(spec)
            from . import plots

            x_values, series, ylabel = self._compute_line_plot_data(
                key, target, ref_indices, plot_type="line"
            )
            has_finite_values = any(
                np.any(np.isfinite(np.asarray(item[1], dtype=np.float64)))
                for item in series
            )
            if not has_finite_values:
                present_warning(
                    self,
                    APP_TITLE,
                    "No finite values are available for this line plot.",
                )
                return
            indices = list(map(int, x_values.tolist()))
            boundaries, labels = plot_data.chain_boundaries(
                target.token_map, indices, original_x=True
            )
            vmin, vmax = self._get_vmin_vmax()
            title = f"{metrics.metric_label(key)} ({target.label})"
            fig = plots.make_line_plot(
                x_values,
                series,
                title=title,
                ylabel=ylabel,
                ymin=vmin,
                ymax=vmax,
                chain_boundaries=boundaries,
                chain_labels=labels,
            )
            plots.attach_viewer_selection_metadata(
                fig,
                kind="line",
                token_map=target.token_map,
                obj_name=target.obj_name,
                token_maps=self._plot_selection_token_maps(target),
                token_map_obj_names=self._plot_selection_obj_names(target),
                token_indices=indices,
                x_positions=x_values.tolist(),
            )
            self._show_plot_figure(fig, title)
        except Exception as exc:
            present_error(self, f"{APP_TITLE} - error", str(exc))

    def _show_summary_plot(self, kind: str) -> None:
        """Open a PAE/PDE intra-chain versus inter-chain summary line plot."""
        target = self._resolve_plot_target()
        if target is None:
            return

        label = "PAE" if kind == "pae" else "PDE"
        if not self._summary_plot_has_matrix_data(kind, target):
            present_information(
                self,
                APP_TITLE,
                f"{label} summary requires {label} data.",
            )
            return
        if not plot_data.has_multiple_token_chains(target.token_map):
            present_information(
                self,
                APP_TITLE,
                f"{label} summary requires a target with more than one chain.",
            )
            return

        ref_indices = self._resolve_reference_indices(
            target.token_map, target.obj_name, required=False
        )
        if ref_indices is None:
            return
        capabilities = frozenset({kind})
        if self._defer_action_for_data(
            target,
            capabilities,
            error_title=f"{APP_TITLE} - error",
            deferred_action=self.services.analysis.capture_current(f"{kind}_summary"),
        ):
            return

        try:
            from . import plots

            x_values, series, ylabel = self._compute_summary_plot_data(
                kind, target, ref_indices
            )
            has_finite_values = any(
                np.any(np.isfinite(np.asarray(item[1], dtype=np.float64)))
                for item in series
            )
            if not has_finite_values:
                present_warning(
                    self,
                    APP_TITLE,
                    "No finite values are available for this summary plot.",
                )
                return
            indices = list(map(int, x_values.tolist()))
            boundaries, labels = plot_data.chain_boundaries(
                target.token_map, indices, original_x=True
            )
            vmin, vmax = self._get_vmin_vmax()
            title = f"{label} summary ({target.label})"
            fig = plots.make_line_plot(
                x_values,
                series,
                title=title,
                ylabel=ylabel,
                ymin=vmin,
                ymax=vmax,
                chain_boundaries=boundaries,
                chain_labels=labels,
                show_legend=True,
            )
            plots.attach_viewer_selection_metadata(
                fig,
                kind="line",
                token_map=target.token_map,
                obj_name=target.obj_name,
                token_maps=self._plot_selection_token_maps(target),
                token_map_obj_names=self._plot_selection_obj_names(target),
                token_indices=indices,
                x_positions=x_values.tolist(),
            )
            self._show_plot_figure(fig, title)
        except Exception as exc:
            present_error(self, f"{APP_TITLE} - error", str(exc))

    def _show_distribution_plot(self) -> None:
        """Open a quality-class bar plot or continuous-value histogram."""
        target = self._resolve_plot_target()
        if target is None:
            return

        key = self._analysis_metric_key()
        spec = metrics.METRICS.require(key)
        if spec.is_domain_label and target.kind == "ensemble_group":
            present_information(
                self,
                APP_TITLE,
                "Distribution plots for PAE domain labels are available for "
                "single models or individual ensemble members. Cluster labels "
                "are member-local and are not pooled across an ensemble.",
            )
            return
        if key == "chain_iptm":
            present_information(
                self,
                APP_TITLE,
                "Distribution plots are not available for chain ipTM.\n"
                "Use Matrix Plot\u2026 for pairwise chain ipTM values.",
            )
            return

        ref_indices = self._resolve_reference_indices(
            target.token_map, target.obj_name, required=spec.needs_reference
        )
        if ref_indices is None:
            return
        if self._defer_action_for_data(
            target,
            spec.load_capabilities,
            error_title=f"{APP_TITLE} - error",
            deferred_action=self.services.analysis.capture_current("distribution"),
        ):
            return

        try:
            if target.kind == "single":
                self._ensure_current_data_for_property(spec)
            elif target.kind == "ensemble_member" and target.members:
                self._ensure_member_data_for_property(target.members[0], spec)

            from . import plots

            x_values, series, _ylabel = self._compute_line_plot_data(
                key, target, ref_indices, plot_type="distribution"
            )
            if not series:
                raise ValueError("No values are available for this distribution.")
            indices = list(map(int, x_values.tolist()))
            values = np.asarray(series[0][1], dtype=np.float64).ravel()
            title = f"{metrics.metric_label(key)} distribution ({target.label})"

            if key == "plddt_class":
                title = f"{metrics.metric_label(key)} distribution\n({target.label})"
                labels, counts, bar_groups, total = (
                    plot_data.plddt_class_distribution_groups(values, indices)
                )
                fig = plots.make_plddt_class_bar_plot(
                    labels,
                    counts,
                    total=total,
                    title=title,
                )
                bar_positions = list(range(len(labels)))
                bar_widths = [0.8 for _label in labels]
            elif spec.is_domain_label:
                title = f"{metrics.metric_label(key)} distribution\n({target.label})"
                labels, counts, bar_groups, colors = (
                    plot_data.domain_label_distribution_groups(values, indices)
                )
                fig = plots.make_categorical_bar_plot(
                    labels,
                    counts,
                    title=title,
                    colors=colors,
                )
                bar_positions = list(range(len(labels)))
                bar_widths = [0.8 for _label in labels]
            else:
                edges, bar_groups, bar_positions, bar_widths = (
                    plot_data.histogram_distribution_groups(values, indices)
                )
                fig = plots.make_histogram_plot(
                    values,
                    title=title,
                    xlabel=metrics.metric_label(key),
                    bin_edges=edges,
                )

            plots.attach_viewer_selection_metadata(
                fig,
                kind="bars",
                token_map=target.token_map,
                obj_name=target.obj_name,
                token_maps=self._plot_selection_token_maps(target),
                token_map_obj_names=self._plot_selection_obj_names(target),
                bar_token_indices=bar_groups,
                bar_x_positions=bar_positions,
                bar_widths=bar_widths,
            )
            self._show_plot_figure(fig, title)
        except Exception as exc:
            present_error(self, f"{APP_TITLE} - error", str(exc))

    def _show_ensemble_site_summary(self) -> None:
        """Open the ensemble ligand-site summary plot."""
        ref_sel = self._analysis_reference_selection()
        if not ref_sel:
            present_warning(
                self,
                APP_TITLE,
                "Ensemble site summary requires a reference selection.\n"
                f"Enter a ligand or other {VIEWER_NAME} selection in the Reference field.",
            )
            return
        cutoff = self._get_cutoff_threshold()
        if cutoff is None:
            return
        ensemble_state = getattr(self, "_ensemble", None)
        if ensemble_state is None:
            present_information(
                self,
                APP_TITLE,
                "The ensemble target is not active.\nUse Load Ensemble\u2026 first.",
            )
            return

        if self._pred_files is not None:
            members = sorted(ensemble_state.members, key=lambda member: member.rank)
            reference = members[0]
            try:
                target = _PlotTarget(
                    kind="ensemble_group",
                    label=ensemble_state.group_name,
                    obj_name=reference.obj_name,
                    model_states=tuple(
                        self._canonical_state_for_ensemble_member(member)
                        for member in members
                    ),
                    members=tuple(members),
                )
            except ValueError as exc:
                present_warning(self, APP_TITLE, str(exc))
                return
            if self._defer_action_for_data(
                target,
                self._fingerprint_capabilities(include_contact_probs=False),
                error_title=f"{APP_TITLE} - error",
                allow_partial=True,
                deferred_action=self.services.analysis.capture_current(
                    "ensemble_site_summary"
                ),
            ):
                return

        try:
            from . import plots

            members, labels, series, site_indices = (
                self._compute_ensemble_site_summary_data(ref_sel, cutoff)
            )
            if not series:
                present_warning(
                    self,
                    APP_TITLE,
                    "No pLDDT, PAE, or PDE data are available for the "
                    "ensemble site summary.",
                )
                return
            title = f"Ensemble site summary\nReference: {ref_sel}, cutoff {cutoff:g} Å"
            fig = plots.make_ensemble_site_summary_plot(
                labels,
                series,
                title=title,
            )
            plots.attach_ensemble_site_summary_metadata(
                fig,
                member_obj_names=[member.obj_name for member in members],
                member_token_maps=[
                    self._model_states[member.rank].token_map for member in members
                ],
                site_indices=site_indices,
                selection_name="foldqc_ensemble_site",
            )
            self._show_plot_figure(fig, "Ensemble site summary")
        except Exception as exc:
            present_error(self, f"{APP_TITLE} - error", str(exc))

    def _show_matrix_plot(self) -> None:
        """Open a PAE or PDE matrix plot for the selected target/property."""
        target = self._resolve_plot_target()
        if target is None:
            return

        key = self._analysis_metric_key()
        spec = metrics.METRICS.require(key)
        if spec.matrix is None:
            present_information(
                self,
                APP_TITLE,
                "Matrix plots are only available when Color by is a PAE, PDE, "
                "interaction probability, or chain ipTM property.",
            )
            return

        attr = spec.matrix.source
        if attr == "chain_iptm":
            ref_indices = []
        else:
            ref_indices = self._resolve_reference_indices(
                target.token_map, target.obj_name, required=False
            )
            if ref_indices is None:
                return

        capabilities = frozenset(
            {attr} if attr in {"pae", "pde", "contact_probs"} else ()
        )
        if self._defer_action_for_data(
            target,
            capabilities,
            error_title=f"{APP_TITLE} - error",
            deferred_action=self.services.analysis.capture_current("matrix"),
        ):
            return

        try:
            from . import plots

            (
                matrix,
                row_indices,
                col_indices,
                title,
                label,
                row_labels,
                col_labels,
                cell_text,
            ) = self._compute_matrix_plot_data(key, target, ref_indices)
            if attr == "chain_iptm":
                row_boundaries = []
                col_boundaries = []
                xlabel = "Chain j"
                ylabel = "Chain i"
            else:
                row_boundaries, _ = plot_data.chain_boundaries(
                    target.token_map, row_indices
                )
                col_boundaries, _ = plot_data.chain_boundaries(
                    target.token_map, col_indices
                )
                xlabel = "Scored token j"
                ylabel = "Alignment anchor i"
            vmin, vmax = self._get_vmin_vmax()
            palette, reverse_palette = self._selected_palette()
            fig = plots.make_matrix_plot(
                matrix,
                title=f"{title} ({target.label})",
                token_map=target.token_map,
                row_indices=row_indices,
                col_indices=col_indices,
                row_labels=row_labels,
                col_labels=col_labels,
                cell_text=cell_text,
                row_chain_boundaries=row_boundaries,
                col_chain_boundaries=col_boundaries,
                vmin=0.0 if vmin is None else vmin,
                vmax=vmax,
                palette=palette,
                reverse_palette=reverse_palette,
                xlabel=xlabel,
                ylabel=ylabel,
                colorbar_label=label,
            )
            if attr != "chain_iptm":
                plots.attach_viewer_selection_metadata(
                    fig,
                    kind="matrix",
                    token_map=target.token_map,
                    obj_name=target.obj_name,
                    token_maps=self._plot_selection_token_maps(target),
                    token_map_obj_names=self._plot_selection_obj_names(target),
                    row_indices=row_indices,
                    col_indices=col_indices,
                )
            self._show_plot_figure(fig, f"{title} ({target.label})")

        except Exception as exc:
            present_error(self, f"{APP_TITLE} - error", str(exc))

    def _show_binding_site_fingerprint(self) -> None:
        """Open a binding-site confidence fingerprint for the current target."""
        ref_sel = self._analysis_reference_selection()
        if not ref_sel:
            present_warning(
                self,
                APP_TITLE,
                "Fingerprint requires a reference selection.\n"
                f"Enter a ligand or other {VIEWER_NAME} selection in the Reference field.",
            )
            return

        target = self._resolve_plot_target()
        if target is None:
            return

        ref_indices = self._resolve_reference_indices(
            target.token_map, target.obj_name, required=True
        )
        if ref_indices is None:
            return
        cutoff = self._get_cutoff_threshold()
        if cutoff is None:
            return

        if self._defer_action_for_data(
            target,
            self._fingerprint_capabilities(include_contact_probs=True),
            error_title=f"{APP_TITLE} - error",
            allow_partial=True,
            deferred_action=self.services.analysis.capture_current(
                "binding_site_fingerprint"
            ),
        ):
            return

        try:
            from . import plots

            binding_indices = self._binding_site_token_indices(
                target.token_map, target.obj_name, ref_sel, ref_indices, cutoff
            )
            if not binding_indices:
                present_warning(
                    self,
                    APP_TITLE,
                    "No polymer binding-site residues were found within "
                    f"{cutoff:g} Å of the reference selection.",
                )
                return
            series = self._compute_fingerprint_data(target, ref_indices)
            if (
                series["plddt"] is None
                and series["pae_to_ligand"] is None
                and series["pae_from_ligand"] is None
                and series["pde_to_ligand"] is None
                and series["interaction_prob_to_ligand"] is None
            ):
                present_warning(
                    self,
                    APP_TITLE,
                    "No confidence data are available for the fingerprint.",
                )
                return

            if len(binding_indices) > plots.MAX_BINDING_SITE_RESIDUES:
                present_warning(
                    self,
                    APP_TITLE,
                    "The binding-site fingerprint found "
                    f"{len(binding_indices)} polymer residues within {cutoff:g} Å "
                    "of the reference selection. Only the first "
                    f"{plots.MAX_BINDING_SITE_RESIDUES} residues in structure "
                    "token order will be shown.",
                )

            title = f"Binding-site confidence fingerprint ({target.label})"
            fig = plots.make_binding_site_fingerprint(
                target.token_map,
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
            displayed_binding_indices = binding_indices[
                : plots.MAX_BINDING_SITE_RESIDUES
            ]
            plots.attach_viewer_selection_metadata(
                fig,
                kind="bars",
                token_map=target.token_map,
                obj_name=target.obj_name,
                token_maps=self._plot_selection_token_maps(target),
                token_map_obj_names=self._plot_selection_obj_names(target),
                token_indices=displayed_binding_indices,
                x_positions=list(range(len(displayed_binding_indices))),
            )
            self._show_plot_figure(fig, title)
        except Exception as exc:
            present_error(self, f"{APP_TITLE} - error", str(exc))

    def _plot_selection_token_maps(self, target: _PlotTarget) -> list | None:
        """Return all token maps that plot selections should target."""
        if target.kind == "ensemble_group":
            members = sorted(target.members or [], key=lambda member: member.rank)
            return [
                self._canonical_state_for_ensemble_member(member).token_map
                for member in members
            ]
        return None

    def _plot_selection_obj_names(self, target: _PlotTarget) -> list[str] | None:
        """Return object names corresponding to ensemble plot token maps."""
        if target.kind == "ensemble_group":
            members = sorted(target.members or [], key=lambda member: member.rank)
            names = [getattr(member, "obj_name", None) for member in members]
            if all(names):
                return [str(name) for name in names]
        return None
