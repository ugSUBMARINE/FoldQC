"""Plot target resolution, data coordination, and figure dispatch."""

from __future__ import annotations

import numpy as np

from . import gui_rules, metrics, plot_data
from .compat import QtWidgets
from .gui_state import ResolvedTarget as _PlotTarget
from .mol_viewer import get_viewer_name, selection_to_token_indices
from .token_map import TokenMap

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()


class PlotController:
    def _resolve_plot_target(self) -> _PlotTarget | None:
        """Resolve the current viewer target into data and token-map context."""
        obj_name = self._get_obj_name()
        if obj_name is None:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, f"No {VIEWER_NAME} target selected."
            )
            return None

        ensemble_group_name = getattr(self, "_ensemble_group_name", None)
        ensemble_members = getattr(self, "_ensemble_members", None)
        if obj_name == ensemble_group_name:
            members = sorted(ensemble_members or [], key=lambda member: member.rank)
            if not members:
                QtWidgets.QMessageBox.information(
                    self,
                    APP_TITLE,
                    "The ensemble target is not active.\nUse the Ensemble\u2026 button first.",
                )
                return None
            reference = members[0]
            return _PlotTarget(
                kind="ensemble_group",
                label=obj_name,
                obj_name=reference.obj_name,
                data=None,
                token_map=reference.token_map,
                members=members,
            )

        member = self._selected_ensemble_member(obj_name)
        if member is not None:
            return _PlotTarget(
                kind="ensemble_member",
                label=member.obj_name,
                obj_name=member.obj_name,
                data=member.data,
                token_map=member.token_map,
                members=[member],
            )

        if self._pred_data is None:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "No prediction data loaded.")
            return None
        self._build_token_map_if_needed(obj_name)
        return _PlotTarget(
            kind="single",
            label=obj_name,
            obj_name=obj_name,
            data=self._pred_data,
            token_map=self._token_map,
            members=None,
        )

    def _resolve_reference_indices(
        self,
        token_map: TokenMap,
        obj_name: str,
        *,
        required: bool = False,
    ) -> list[int] | None:
        """Resolve the Reference field to token indices, preserving token order."""
        ref_sel = self._ref_edit.text().strip()
        if not ref_sel:
            if required:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "This plot requires a reference selection.\n"
                    "Enter a viewer selection in the Reference field.",
                )
                return None
            return []

        indices = selection_to_token_indices(token_map, ref_sel, obj_name=obj_name)
        if not indices:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                f"Reference selection '{ref_sel}' matched no tokens in {obj_name}.",
            )
            return None
        return indices

    def _compute_line_plot_data(
        self,
        key: str,
        target: _PlotTarget,
        ref_indices: list[int],
        *,
        plot_type: str = "line",
    ) -> tuple[np.ndarray, list[tuple[str, np.ndarray, np.ndarray | None]], str]:
        """Return x values, series tuples, and y-axis label for a line plot."""
        token_map = target.token_map
        use_ref_scope = bool(ref_indices) and metrics.plot_uses_reference_scope(
            key, plot_type
        )
        indices = list(ref_indices) if use_ref_scope else list(range(len(token_map)))
        if not indices:
            raise ValueError("No tokens are available for the line plot.")

        compute_key = metrics.line_compute_key(key)
        ref_edit = getattr(self, "_ref_edit", None)
        ref_sel = None if ref_edit is None else ref_edit.text().strip() or None

        if target.kind == "ensemble_group":
            if compute_key == "ensemble_rmsd":
                values = self._compute_ensemble_property("ensemble_rmsd")
                return (
                    np.asarray(indices, dtype=np.int32),
                    [(metrics.metric_label(key), values[indices], None)],
                    metrics.line_ylabel(compute_key),
                )
            if compute_key == "ensemble_plddt_mean":
                mean = self._compute_ensemble_property("ensemble_plddt_mean")
                std = self._compute_ensemble_property("ensemble_plddt_std")
                return (
                    np.asarray(indices, dtype=np.int32),
                    [(metrics.metric_label(key), mean[indices], std[indices])],
                    metrics.line_ylabel(compute_key),
                )
            if compute_key == "ensemble_plddt_std":
                values = self._compute_ensemble_property("ensemble_plddt_std")
                return (
                    np.asarray(indices, dtype=np.int32),
                    [(metrics.metric_label(key), values[indices], None)],
                    metrics.line_ylabel(compute_key),
                )

            (
                load_pae,
                load_pde,
                load_contact_probs,
                load_token_plddt,
            ) = plot_data.line_member_load_flags(key)
            arrays = []
            for member in target.members or []:
                kwargs = dict(
                    load_pae=load_pae,
                    load_pde=load_pde,
                    load_token_plddt=load_token_plddt,
                )
                if load_contact_probs:
                    kwargs["load_contact_probs"] = True
                self._ensure_member_data_for_plot(member, **kwargs)
                values = self._compute_property_for(
                    compute_key, ref_sel, member.data, member.token_map, member.obj_name
                )
                if values is None:
                    raise ValueError("Could not compute the selected property.")
                self._validate_token_count(values, member.token_map, member.obj_name)
                arrays.append(np.asarray(values, dtype=np.float32))
            mean, std = plot_data.nan_mean_std(arrays, len(token_map))
            if mean is None:
                raise ValueError("No ensemble values are available for this plot.")
            return (
                np.asarray(indices, dtype=np.int32),
                [(f"{metrics.metric_label(key)} mean", mean[indices], std[indices])],
                metrics.line_ylabel(compute_key),
            )

        if compute_key.startswith("ensemble_"):
            values = self._compute_ensemble_property(compute_key)
        else:
            values = self._compute_property_for(
                compute_key, ref_sel, target.data, token_map, target.obj_name
            )
        if values is None:
            raise ValueError("Could not compute the selected property.")
        self._validate_token_count(values, token_map, target.label)
        return (
            np.asarray(indices, dtype=np.int32),
            [(metrics.metric_label(key), np.asarray(values)[indices], None)],
            metrics.line_ylabel(compute_key),
        )

    def _summary_plot_has_matrix_data(self, kind: str, target: _PlotTarget) -> bool:
        """Return whether the target can provide the requested summary matrix."""
        attr = "pae" if kind == "pae" else "pde"
        if self._has_matrix_data_family(attr):
            return True
        if target.kind == "ensemble_group":
            return any(
                getattr(member.data, attr, None) is not None
                for member in target.members or []
            )
        return getattr(target.data, attr, None) is not None

    def _compute_summary_plot_data(
        self,
        kind: str,
        target: _PlotTarget,
        ref_indices: list[int],
    ) -> tuple[
        np.ndarray,
        list[
            tuple[str, np.ndarray, np.ndarray | None]
            | tuple[str, np.ndarray, np.ndarray | None, str]
        ],
        str,
    ]:
        """Return x values, series tuples, and y-axis label for a summary plot."""
        if kind not in {"pae", "pde"}:
            raise ValueError(f"Unknown summary plot kind: {kind}")
        if not plot_data.has_multiple_token_chains(target.token_map):
            raise ValueError("Summary plots require a target with more than one chain.")

        indices = (
            list(ref_indices) if ref_indices else list(range(len(target.token_map)))
        )
        if not indices:
            raise ValueError("No tokens are available for the summary plot.")

        load_pae = kind == "pae"
        load_pde = kind == "pde"
        if target.kind == "ensemble_group":
            data_items = []
            token_maps = []
            for member in sorted(target.members or [], key=lambda item: item.rank):
                self._ensure_member_data_for_plot(
                    member, load_pae=load_pae, load_pde=load_pde
                )
                data_items.append(member.data)
                token_maps.append(member.token_map)
            series = plot_data.summary_series_for_ensemble(
                kind,
                data_items,
                target.token_map,
                token_maps=token_maps,
            )
        else:
            if target.kind == "ensemble_member" and target.members:
                self._ensure_member_data_for_plot(
                    target.members[0], load_pae=load_pae, load_pde=load_pde
                )
                target.data = target.members[0].data
            elif target.kind == "single" and target.data is self._pred_data:
                self._ensure_current_data_for_property(
                    {
                        "needs_pae": load_pae,
                        "needs_pde": load_pde,
                    }
                )
                target.data = self._pred_data
            series = plot_data.summary_series_for_data(
                kind, target.data, target.token_map
            )

        sliced = []
        for item in series:
            label, values, std = item[0], item[1], item[2]
            sliced_item = (
                label,
                np.asarray(values, dtype=np.float32)[indices],
                None if std is None else np.asarray(std, dtype=np.float32)[indices],
            )
            if len(item) == 4:
                sliced.append((*sliced_item, item[3]))
            else:
                sliced.append(sliced_item)
        ylabel = "PAE gap (Å)" if kind == "pae" else "PDE gap (Å)"
        return np.asarray(indices, dtype=np.int32), sliced, ylabel

    def _compute_matrix_plot_data(
        self,
        key: str,
        target: _PlotTarget,
        ref_indices: list[int],
    ) -> tuple[
        np.ndarray,
        list[int],
        list[int],
        str,
        str,
        list[str] | None,
        list[str] | None,
        np.ndarray | None,
    ]:
        """Return matrix data and display metadata for a matrix plot."""
        source = metrics.matrix_source_for_metric(key)
        if source is None:
            raise ValueError(
                "Matrix plots are only available for PAE, PDE, interaction "
                "probability, and chain ipTM properties."
            )
        attr, title, label = source
        if attr == "chain_iptm":
            data = (
                target.members[0].data
                if target.kind == "ensemble_member" and target.members
                else target.data
            )
            return plot_data.chain_iptm_matrix_plot_data(
                target_kind=target.kind,
                data=data,
                token_map=target.token_map,
                title=title,
                label=label,
                members=target.members,
            )

        load_pae = attr == "pae"
        load_pde = attr == "pde"
        load_contact_probs = attr == "contact_probs"

        if target.kind == "ensemble_group":
            matrices = []
            for member in target.members or []:
                kwargs = dict(load_pae=load_pae, load_pde=load_pde)
                if load_contact_probs:
                    kwargs["load_contact_probs"] = True
                self._ensure_member_data_for_plot(member, **kwargs)
                matrix = getattr(member.data, attr, None)
                if matrix is None:
                    raise ValueError(
                        f"{label} matrix is not available for model_{member.rank}."
                    )
                matrices.append(np.asarray(matrix, dtype=np.float32))
            matrix = np.stack(matrices, axis=0).mean(axis=0)
            title = f"{title} — ensemble mean"
        else:
            if target.kind == "ensemble_member" and target.members:
                kwargs = dict(load_pae=load_pae, load_pde=load_pde)
                if load_contact_probs:
                    kwargs["load_contact_probs"] = True
                self._ensure_member_data_for_plot(target.members[0], **kwargs)
                target.data = target.members[0].data
            elif (
                target.kind == "single"
                and target.data is getattr(self, "_pred_data", None)
                and getattr(self, "_pred_files", None) is not None
            ):
                self._ensure_current_data_for_property(
                    metrics.PROPERTY_BY_KEY.get(key, {})
                )
                target.data = self._pred_data
            matrix = getattr(target.data, attr, None)
            if matrix is None:
                raise ValueError(f"{label} matrix is not available for this model.")
            matrix = np.asarray(matrix, dtype=np.float32)

        if key == "pae_row_mean" and ref_indices:
            row_indices = list(ref_indices)
            col_indices = list(range(matrix.shape[1]))
        elif key == "pae_col_to_sel" and ref_indices:
            row_indices = list(ref_indices)
            col_indices = list(range(matrix.shape[1]))
        elif key == "pae_sym_within_sel" and ref_indices:
            row_indices = list(ref_indices)
            col_indices = list(ref_indices)
        else:
            row_indices = list(range(matrix.shape[0]))
            col_indices = (
                list(ref_indices) if ref_indices else list(range(matrix.shape[1]))
            )
        submatrix = matrix[np.ix_(row_indices, col_indices)]
        return submatrix, row_indices, col_indices, title, label, None, None, None

    def _ensemble_site_summary_for_member(
        self,
        member,
        ref_sel: str,
        cutoff: float,
    ) -> dict:
        """Compute local ligand-site summary values for one ensemble member."""
        self._ensure_member_data_for_plot(
            member,
            load_pae=getattr(self._pred_files, "has_pae", False)
            if self._pred_files
            else False,
            load_pde=getattr(self._pred_files, "has_pde", False)
            if self._pred_files
            else False,
            load_token_plddt=True,
        )
        ref_indices = selection_to_token_indices(
            member.token_map, ref_sel, obj_name=member.obj_name
        )
        if not ref_indices:
            raise ValueError(
                f"Reference selection '{ref_sel}' matched no tokens in {member.obj_name}."
            )
        contact_indices = self._binding_site_token_indices(
            member.token_map, member.obj_name, ref_sel, ref_indices, cutoff
        )
        site_indices = list(dict.fromkeys(list(ref_indices) + list(contact_indices)))
        if not site_indices:
            raise ValueError(f"No site tokens are available for {member.obj_name}.")

        return {
            "member": member,
            "site_indices": site_indices,
            **plot_data.site_summary_values(member.data, site_indices),
        }

    def _compute_ensemble_site_summary_data(
        self,
        ref_sel: str,
        cutoff: float,
    ) -> tuple[list, list[str], list[tuple[str, np.ndarray, str]], list[list[int]]]:
        """Return ensemble members, labels, metric series, and site-token groups."""
        members = sorted(self._ensemble_members or [], key=lambda member: member.rank)
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

    def _compute_fingerprint_data(
        self,
        target: _PlotTarget,
        ref_indices: list[int],
    ) -> dict[str, np.ndarray | None]:
        """Return mean/std fingerprint series for a single target or ensemble."""
        size = len(target.token_map)
        if target.kind != "ensemble_group":
            if target.kind == "ensemble_member" and target.members:
                self._ensure_member_data_for_plot(
                    target.members[0],
                    load_pae=getattr(self._pred_files, "has_pae", False)
                    if self._pred_files
                    else False,
                    load_pde=getattr(self._pred_files, "has_pde", False)
                    if self._pred_files
                    else False,
                    load_contact_probs=getattr(
                        self._pred_files, "has_contact_probs", False
                    )
                    if self._pred_files
                    else False,
                )
                target.data = target.members[0].data
            elif (
                target.kind == "single"
                and target.data is self._pred_data
                and getattr(self, "_pred_files", None) is not None
            ):
                target.data = self._pred_data
            return plot_data.fingerprint_series_for_single(target.data, ref_indices)

        data_items = []
        for member in target.members or []:
            self._ensure_member_data_for_plot(
                member,
                load_pae=getattr(self._pred_files, "has_pae", False)
                if self._pred_files
                else False,
                load_pde=getattr(self._pred_files, "has_pde", False)
                if self._pred_files
                else False,
                load_contact_probs=getattr(self._pred_files, "has_contact_probs", False)
                if self._pred_files
                else False,
            )
            data_items.append(member.data)

        return plot_data.fingerprint_series_for_ensemble(
            data_items, ref_indices, size=size
        )

    def _show_selected_plot(self, plot_type: str | None = None) -> None:
        """Dispatch the selected plot type to its plot handler."""
        if plot_type is None and hasattr(self, "_plot_type_combo"):
            plot_type = self._plot_type_combo.currentData()
        if plot_type is None:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "No plot type selected.")
            return
        key = self._prop_combo.currentData()
        state = gui_rules.plot_action_state(
            plot_type,
            key,
            self._current_target_kind(),
            bool(self._ref_edit.text().strip()),
            bool(getattr(self, "_ensemble_members", None)),
            has_fingerprint_data=self._has_fingerprint_data(),
            has_pae_data=self._has_matrix_data_family("pae"),
            has_pde_data=self._has_matrix_data_family("pde"),
            has_multiple_chains=self._current_target_has_multiple_chains(),
        )
        if not state.enabled:
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                state.reason or f"{plot_type} is not available.",
            )
            return
        dependency_features = ["plot"]
        if metrics.is_domain_label_metric(key):
            dependency_features.append(key)
        plot_labels = {key: label for label, key in metrics.PLOT_TYPES}
        plot_label = plot_labels.get(plot_type, plot_type.replace("_", " "))
        if not self._ensure_feature_dependencies(
            dependency_features, feature_label=f"The {plot_label.lower()} plot"
        ):
            return
        if plot_type == "line":
            self._show_line_plot()
        elif plot_type == "distribution":
            self._show_distribution_plot()
        elif plot_type == "matrix":
            self._show_matrix_plot()
        elif plot_type == "pae_summary":
            self._show_summary_plot("pae")
        elif plot_type == "pde_summary":
            self._show_summary_plot("pde")
        elif plot_type == "binding_site_fingerprint":
            self._show_binding_site_fingerprint()
        elif plot_type == "ensemble_site_summary":
            self._show_ensemble_site_summary()
        else:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, f"Unknown plot type: {plot_type}"
            )

    def _fingerprint_load_flags(
        self, *, include_contact_probs: bool
    ) -> dict[str, bool]:
        pred_files = getattr(self, "_pred_files", None)
        return {
            "load_pae": bool(pred_files and pred_files.has_pae),
            "load_pde": bool(pred_files and pred_files.has_pde),
            "load_contact_probs": bool(
                include_contact_probs and pred_files and pred_files.has_contact_probs
            ),
            "load_token_plddt": True,
        }

    def _show_line_plot(self) -> None:
        """Open a token-indexed line plot for the selected property."""
        target = self._resolve_plot_target()
        if target is None:
            return

        key = self._prop_combo.currentData()
        if metrics.is_domain_label_metric(key):
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Line plots are not available for PAE domain labels.\n"
                "Use Distribution to inspect cluster occupancy.",
            )
            return
        if key in metrics.CONTACT_FILTERED_METRICS:
            metric_name = "PAE" if key == "pae_contact" else "PDE"
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                f"Line plots are not available for {metric_name} "
                "contact-filtered values.\n"
                "Use Distribution or Matrix instead.",
            )
            return
        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        ref_indices = self._resolve_reference_indices(
            target.token_map, target.obj_name, required=prop.get("needs_ref", False)
        )
        if ref_indices is None:
            return
        if self._defer_action_for_data(
            target,
            metrics.metric_load_flags(prop),
            self._show_line_plot,
            error_title=f"{APP_TITLE} - error",
        ):
            return

        try:
            if target.kind == "single":
                self._ensure_current_data_for_property(prop)
                target.data = self._pred_data
            from . import plots

            x_values, series, ylabel = self._compute_line_plot_data(
                key, target, ref_indices, plot_type="line"
            )
            has_finite_values = any(
                np.any(np.isfinite(np.asarray(item[1], dtype=np.float64)))
                for item in series
            )
            if not has_finite_values:
                QtWidgets.QMessageBox.warning(
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
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _show_summary_plot(self, kind: str) -> None:
        """Open a PAE/PDE intra-chain versus inter-chain summary line plot."""
        target = self._resolve_plot_target()
        if target is None:
            return

        label = "PAE" if kind == "pae" else "PDE"
        if not self._summary_plot_has_matrix_data(kind, target):
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                f"{label} summary requires {label} data.",
            )
            return
        if not plot_data.has_multiple_token_chains(target.token_map):
            QtWidgets.QMessageBox.information(
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
        flags = {
            "load_pae": kind == "pae",
            "load_pde": kind == "pde",
        }
        if self._defer_action_for_data(
            target,
            flags,
            lambda: self._show_summary_plot(kind),
            error_title=f"{APP_TITLE} - error",
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
                QtWidgets.QMessageBox.warning(
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
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _show_distribution_plot(self) -> None:
        """Open a quality-class bar plot or continuous-value histogram."""
        target = self._resolve_plot_target()
        if target is None:
            return

        key = self._prop_combo.currentData()
        if metrics.is_domain_label_metric(key) and target.kind == "ensemble_group":
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Distribution plots for PAE domain labels are available for "
                "single models or individual ensemble members. Cluster labels "
                "are member-local and are not pooled across an ensemble.",
            )
            return
        if key == "chain_iptm":
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Distribution plots are not available for chain ipTM.\n"
                "Use Matrix Plot\u2026 for pairwise chain ipTM values.",
            )
            return

        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        ref_indices = self._resolve_reference_indices(
            target.token_map, target.obj_name, required=prop.get("needs_ref", False)
        )
        if ref_indices is None:
            return
        if self._defer_action_for_data(
            target,
            metrics.metric_load_flags(prop),
            self._show_distribution_plot,
            error_title=f"{APP_TITLE} - error",
        ):
            return

        try:
            if target.kind == "single":
                self._ensure_current_data_for_property(prop)
                target.data = self._pred_data
            elif target.kind == "ensemble_member" and target.members:
                self._ensure_member_data_for_property(target.members[0], prop)
                target.data = target.members[0].data

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
            elif metrics.is_domain_label_metric(key):
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
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _show_ensemble_site_summary(self) -> None:
        """Open the ensemble ligand-site summary plot."""
        ref_sel = self._ref_edit.text().strip()
        if not ref_sel:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "Ensemble site summary requires a reference selection.\n"
                f"Enter a ligand or other {VIEWER_NAME} selection in the Reference field.",
            )
            return
        cutoff = self._get_cutoff_threshold()
        if cutoff is None:
            return
        if not self._ensemble_members:
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "The ensemble target is not active.\nUse Load Ensemble\u2026 first.",
            )
            return

        if self._pred_files is not None:
            members = sorted(self._ensemble_members, key=lambda member: member.rank)
            reference = members[0]
            target = _PlotTarget(
                kind="ensemble_group",
                label=self._ensemble_group_name or "ensemble",
                obj_name=reference.obj_name,
                data=None,
                token_map=reference.token_map,
                members=members,
            )
            if self._defer_action_for_data(
                target,
                self._fingerprint_load_flags(include_contact_probs=False),
                self._show_ensemble_site_summary,
                error_title=f"{APP_TITLE} - error",
            ):
                return

        try:
            from . import plots

            members, labels, series, site_indices = (
                self._compute_ensemble_site_summary_data(ref_sel, cutoff)
            )
            if not series:
                QtWidgets.QMessageBox.warning(
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
                members=members,
                site_indices=site_indices,
                selection_name="foldqc_ensemble_site",
            )
            self._show_plot_figure(fig, "Ensemble site summary")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _show_matrix_plot(self) -> None:
        """Open a PAE or PDE matrix plot for the selected target/property."""
        target = self._resolve_plot_target()
        if target is None:
            return

        key = self._prop_combo.currentData()
        source = metrics.matrix_source_for_metric(key)
        if source is None:
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Matrix plots are only available when Color by is a PAE, PDE, "
                "interaction probability, or chain ipTM property.",
            )
            return

        attr, _, _ = source
        if attr == "chain_iptm":
            ref_indices = []
        else:
            ref_indices = self._resolve_reference_indices(
                target.token_map, target.obj_name, required=False
            )
            if ref_indices is None:
                return

        flags = {
            "load_pae": attr == "pae",
            "load_pde": attr == "pde",
            "load_contact_probs": attr == "contact_probs",
        }
        if self._defer_action_for_data(
            target,
            flags,
            self._show_matrix_plot,
            error_title=f"{APP_TITLE} - error",
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
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _show_binding_site_fingerprint(self) -> None:
        """Open a binding-site confidence fingerprint for the current target."""
        ref_sel = self._ref_edit.text().strip()
        if not ref_sel:
            QtWidgets.QMessageBox.warning(
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
            self._fingerprint_load_flags(include_contact_probs=True),
            self._show_binding_site_fingerprint,
            error_title=f"{APP_TITLE} - error",
        ):
            return

        try:
            from . import plots

            binding_indices = self._binding_site_token_indices(
                target.token_map, target.obj_name, ref_sel, ref_indices, cutoff
            )
            if not binding_indices:
                QtWidgets.QMessageBox.warning(
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
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "No confidence data are available for the fingerprint.",
                )
                return

            if len(binding_indices) > plots.MAX_BINDING_SITE_RESIDUES:
                QtWidgets.QMessageBox.warning(
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
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _plot_selection_token_maps(self, target: _PlotTarget) -> list | None:
        """Return all token maps that plot selections should target."""
        if target.kind == "ensemble_group":
            members = sorted(target.members or [], key=lambda member: member.rank)
            return [member.token_map for member in members]
        return None

    def _plot_selection_obj_names(self, target: _PlotTarget) -> list[str] | None:
        """Return object names corresponding to ensemble plot token maps."""
        if target.kind == "ensemble_group":
            members = sorted(target.members or [], key=lambda member: member.rank)
            names = [getattr(member, "obj_name", None) for member in members]
            if all(names):
                return [str(name) for name in names]
        return None

    def _show_plot_figure(self, fig, title: str) -> None:
        """Show *fig* in an embedded Qt plot window, falling back externally."""
        from . import plots

        try:
            from . import plot_viewer

            def forget_window(dialog) -> None:
                try:
                    self._plot_windows.remove(dialog)
                except ValueError:
                    pass

            if not hasattr(self, "_plot_windows"):
                self._plot_windows = []
            dialog = plot_viewer.show_figure(
                fig, title=title, parent=self, on_close=forget_window
            )
            self._plot_windows.append(dialog)
        except Exception as qt_exc:
            try:
                plots.save_and_show(fig)
            except Exception as external_exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    f"{APP_TITLE} - plot error",
                    (
                        "Could not show the plot in Qt or with the external "
                        "image viewer.\n\n"
                        f"Qt error: {qt_exc}\n\n"
                        f"External viewer error: {external_exc}"
                    ),
                )
