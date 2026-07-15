"""Pure and viewer-neutral preparation of registered plot data."""

from __future__ import annotations

import numpy as np

from . import metrics, plot_data
from .gui_state import ResolvedTarget as _PlotTarget
from .loader_models import DataCapability


class PlotPreparationService:
    def _canonical_state_for_ensemble_member(self, member):
        ensemble_state = getattr(self, "_ensemble", None)
        state = self._model_states.get(member.rank)
        if (
            ensemble_state is None
            or member not in ensemble_state.members
            or state is None
        ):
            raise ValueError(
                f"Ensemble model_{member.rank} is no longer present in the "
                "canonical model store. Reload the ensemble."
            )
        return state

    def _member_supports_data(self, member, family: str) -> bool:
        state = self._canonical_state_for_ensemble_member(member)
        data_attr = "token_plddt" if family == "plddt" else family
        if getattr(state.data, data_attr, None) is not None:
            return True
        model_getter = getattr(self._pred_files, "model", None)
        return bool(
            callable(model_getter) and model_getter(member.rank).supports(family)
        )

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
        spec = metrics.METRICS.require(key)
        use_ref_scope = bool(ref_indices) and plot_type in spec.reference_scoped_plots
        indices = list(ref_indices) if use_ref_scope else list(range(len(token_map)))
        if not indices:
            raise ValueError("No tokens are available for the line plot.")

        compute_key = key
        ref_sel = self._analysis_reference_selection() or None

        if target.kind == "ensemble_group":
            if compute_key == "ensemble_rmsd":
                values = self._compute_ensemble_property("ensemble_rmsd")
                return (
                    np.asarray(indices, dtype=np.int32),
                    [(metrics.metric_label(key), values[indices], None)],
                    spec.line_ylabel,
                )
            if compute_key == "ensemble_plddt_mean":
                mean = self._compute_ensemble_property("ensemble_plddt_mean")
                std = self._compute_ensemble_property("ensemble_plddt_std")
                return (
                    np.asarray(indices, dtype=np.int32),
                    [(metrics.metric_label(key), mean[indices], std[indices])],
                    spec.line_ylabel,
                )
            if compute_key == "ensemble_plddt_std":
                values = self._compute_ensemble_property("ensemble_plddt_std")
                return (
                    np.asarray(indices, dtype=np.int32),
                    [(metrics.metric_label(key), values[indices], None)],
                    spec.line_ylabel,
                )

            arrays = []
            for member in target.members or []:
                self._ensure_member_data_for_property(member, spec)
                state = self._canonical_state_for_ensemble_member(member)
                values = self._compute_property_for(
                    compute_key,
                    ref_sel,
                    state.data,
                    state.token_map,
                    member.obj_name,
                )
                if values is None:
                    raise ValueError("Could not compute the selected property.")
                self._validate_token_count(values, state.token_map, member.obj_name)
                arrays.append(np.asarray(values, dtype=np.float32))
            mean, std = plot_data.nan_mean_std(arrays, len(token_map))
            if mean is None:
                raise ValueError("No ensemble values are available for this plot.")
            return (
                np.asarray(indices, dtype=np.int32),
                [(f"{metrics.metric_label(key)} mean", mean[indices], std[indices])],
                spec.line_ylabel,
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
            spec.line_ylabel,
        )

    def _summary_plot_has_matrix_data(self, kind: str, target: _PlotTarget) -> bool:
        """Return whether the target can provide the requested summary matrix."""
        attr = "pae" if kind == "pae" else "pde"
        if self._has_matrix_data_family(attr):
            return True
        if target.kind == "ensemble_group":
            return any(
                getattr(
                    self._canonical_state_for_ensemble_member(member).data,
                    attr,
                    None,
                )
                is not None
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

        capabilities: frozenset[DataCapability] = frozenset({kind})
        if target.kind == "ensemble_group":
            data_items = []
            token_maps = []
            for member in sorted(target.members or [], key=lambda item: item.rank):
                self._ensure_member_data_for_plot(member, capabilities)
                state = self._canonical_state_for_ensemble_member(member)
                data_items.append(state.data)
                token_maps.append(state.token_map)
            series = plot_data.summary_series_for_ensemble(
                kind,
                data_items,
                target.token_map,
                token_maps=token_maps,
            )
        else:
            if target.kind == "ensemble_member" and target.members:
                self._ensure_member_data_for_plot(target.members[0], capabilities)
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
        spec = metrics.METRICS.require(key)
        if spec.matrix is None:
            raise ValueError(
                "Matrix plots are only available for PAE, PDE, interaction "
                "probability, and chain ipTM properties."
            )
        attr = spec.matrix.source
        title = spec.matrix.title
        label = spec.matrix.colorbar_label
        if attr == "chain_iptm":
            return plot_data.chain_iptm_matrix_plot_data(
                target_kind=target.kind,
                data=target.model_states[0].data,
                token_map=target.token_map,
                title=title,
                label=label,
                members=list(target.model_states),
            )

        matrix_capabilities: frozenset[DataCapability] = frozenset({attr})

        if target.kind == "ensemble_group":
            matrices = []
            for member in target.members or []:
                self._ensure_member_data_for_plot(member, matrix_capabilities)
                state = self._canonical_state_for_ensemble_member(member)
                matrix = getattr(state.data, attr, None)
                if matrix is None:
                    raise ValueError(
                        f"{label} matrix is not available for model_{member.rank}."
                    )
                matrices.append(np.asarray(matrix, dtype=np.float32))
            matrix = np.stack(matrices, axis=0).mean(axis=0)
            title = f"{title} — ensemble mean"
        else:
            if target.kind == "ensemble_member" and target.members:
                self._ensure_member_data_for_plot(
                    target.members[0], matrix_capabilities
                )
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

    def _compute_fingerprint_data(
        self,
        target: _PlotTarget,
        ref_indices: list[int],
    ) -> dict[str, np.ndarray | None]:
        """Return mean/std fingerprint series for a single target or ensemble."""
        size = len(target.token_map)
        if target.kind != "ensemble_group":
            if target.kind == "ensemble_member" and target.members:
                member = target.members[0]
                capabilities: frozenset[DataCapability] = frozenset(
                    capability
                    for capability in ("pae", "pde", "contact_probs")
                    if self._member_supports_data(member, capability)
                )
                self._ensure_member_data_for_plot(member, capabilities)
            return plot_data.fingerprint_series_for_single(target.data, ref_indices)

        data_items = []
        for member in target.members or []:
            capabilities: frozenset[DataCapability] = frozenset(
                capability
                for capability in ("pae", "pde", "contact_probs")
                if self._member_supports_data(member, capability)
            )
            self._ensure_member_data_for_plot(member, capabilities)
            state = self._canonical_state_for_ensemble_member(member)
            data_items.append(state.data)

        return plot_data.fingerprint_series_for_ensemble(
            data_items, ref_indices, size=size
        )
