"""CSV export orchestration for GUI-resolved targets."""

from __future__ import annotations

from pathlib import Path

from . import export, metrics
from .compat import QtWidgets
from .gui_state import MetricContext
from .gui_state import ResolvedTarget as _PlotTarget
from .token_map import TokenMap

APP_TITLE = "FoldQC"


class ExportController:
    def _export_csv(self) -> None:
        """Export token-level CSV rows for the current metric and target."""
        target = self._resolve_plot_target()
        if target is None:
            return
        pred_data = (
            None if target.kind == "ensemble_group" else target.model_states[0].data
        )
        default_path = export.default_csv_export_path(
            getattr(self, "_pred_files", None),
            pred_data,
            self._prop_combo.currentData(),
            ensemble=target.kind == "ensemble_group",
        )
        result = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export token metric CSV",
            default_path,
            "CSV files (*.csv);;All files (*)",
        )
        path = result[0] if isinstance(result, tuple) else result
        if not path:
            self._raise_after_native_dialog()
            return
        path_obj = Path(path)
        if path_obj.suffix.lower() != ".csv":
            path_obj = path_obj.with_suffix(".csv")
        self._export_csv_to_path(path_obj)

    def _export_csv_to_path(self, path: str | Path) -> None:
        """Build and write CSV rows, reporting GUI errors consistently."""
        if self._pred_files is not None:
            key = self._prop_combo.currentData()
            spec = metrics.METRICS.find(key)
            target = self._resolve_plot_target() if spec is not None else None
            if target is not None and not spec.ensemble_level:
                if self._defer_action_for_data(
                    target,
                    spec.load_capabilities,
                    lambda: self._export_csv_to_path(path),
                    error_title=f"{APP_TITLE} - export error",
                ):
                    return
        try:
            rows = self._build_csv_export_rows()
            if rows is None:
                return
            if not rows:
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, "No token rows were available for export."
                )
                return
            from .export import write_csv

            write_csv(path, rows)
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                f"Exported {len(rows)} token rows to:\n{path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, f"{APP_TITLE} - export error", str(exc)
            )

    def _build_csv_export_rows(self) -> list[dict[str, object]] | None:
        """Return CSV rows for the current metric/target, or None when cancelled."""
        if self._pred_files is None:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, "No prediction output loaded."
            )
            return None
        key = self._prop_combo.currentData()
        spec = metrics.METRICS.find(key)
        if spec is None:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, "Select a Color by metric before exporting."
            )
            return None
        target = self._resolve_plot_target()
        if target is None:
            return None

        if target.kind == "ensemble_group":
            return self._csv_rows_for_ensemble_group(key, spec, target)
        return self._csv_rows_for_single_target(key, spec, target)

    def _csv_rows_for_single_target(
        self,
        key: str,
        spec: metrics.MetricSpec,
        target: _PlotTarget,
    ) -> list[dict[str, object]] | None:
        """Build CSV rows for a single model or one ensemble member."""
        member = (
            (target.members or [None])[0] if target.kind == "ensemble_member" else None
        )
        include_ensemble = target.kind == "ensemble_member"

        if spec.ensemble_level:
            values = self._compute_ensemble_property(key)
            data = (
                target.data
                if target.data is not None
                else getattr(member, "data", None)
            )
            aggregate_kind = spec.aggregate_kind
        else:
            if target.kind == "ensemble_member" and member is not None:
                self._ensure_member_data_for_property(member, spec)
            else:
                self._ensure_current_data_for_property(spec)
            context = self._csv_metric_context(
                key, spec, target.token_map, target.obj_name
            )
            if context is None:
                return None
            values = self._compute_property_from_context(
                key,
                target.data,
                target.token_map,
                context,
            )
            if values is None:
                return None
            aggregate_kind = "ensemble_member" if include_ensemble else "single_model"

        self._validate_token_count(values, target.token_map, target.label)
        if spec.ensemble_level:
            context = self._csv_metric_context(
                key, spec, target.token_map, target.obj_name
            )
            if context is None:
                return None
        return self._csv_rows_from_values(
            key,
            target.data if target.data is not None else data,
            target.token_map,
            values,
            context,
            include_ensemble=include_ensemble,
            member=member,
            aggregate_kind=aggregate_kind,
        )

    def _csv_rows_for_ensemble_group(
        self,
        key: str,
        spec: metrics.MetricSpec,
        target: _PlotTarget,
    ) -> list[dict[str, object]] | None:
        """Build CSV rows for the active ensemble group target."""
        members = sorted(target.members or [], key=lambda member: member.rank)
        if not members:
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "The ensemble target is not active.\nUse the Ensemble\u2026 button first.",
            )
            return None

        if spec.ensemble_level:
            context = self._csv_metric_context(
                key, spec, target.token_map, target.obj_name
            )
            if context is None:
                return None
            values = self._compute_ensemble_property(key)
            self._validate_token_count(values, target.token_map, target.label)
            first_state = self._canonical_state_for_ensemble_member(members[0])
            return self._csv_rows_from_values(
                key,
                first_state.data,
                target.token_map,
                values,
                context,
                include_ensemble=True,
                member=None,
                aggregate_kind=spec.aggregate_kind,
            )

        rows: list[dict[str, object]] = []
        for member in members:
            self._ensure_member_data_for_property(member, spec)
            state = self._canonical_state_for_ensemble_member(member)
            context = self._csv_metric_context(
                key, spec, state.token_map, member.obj_name
            )
            if context is None:
                return None
            values = self._compute_property_from_context(
                key,
                state.data,
                state.token_map,
                context,
            )
            if values is None:
                return None
            self._validate_token_count(values, state.token_map, member.obj_name)
            rows.extend(
                self._csv_rows_from_values(
                    key,
                    state.data,
                    state.token_map,
                    values,
                    context,
                    include_ensemble=True,
                    member=member,
                    aggregate_kind="ensemble_member",
                )
            )
        return rows

    def _csv_metric_context(
        self,
        key: str,
        spec: metrics.MetricSpec,
        token_map: TokenMap,
        obj_name: str,
    ) -> MetricContext | None:
        """Resolve reference/contact provenance for one export computation."""
        reference_selection = ""
        reference_indices: list[int] = []
        contact_indices: list[int] = []
        cutoff = None

        if spec.needs_reference:
            resolved = self._resolve_reference_indices(
                token_map, obj_name, required=True
            )
            if resolved is None:
                return None
            reference_indices = list(resolved)
            reference_selection = self._ref_edit.text().strip()

        if spec.needs_contact_shell:
            cutoff = self._get_cutoff_threshold()
            if cutoff is None:
                return None
            contact_indices = self._binding_site_token_indices(
                token_map,
                obj_name,
                reference_selection,
                reference_indices,
                cutoff,
            )
            if not contact_indices:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "No polymer binding-site residues were found within "
                    f"{cutoff:g} Å of the reference selection.",
                )
                return None
        elif spec.is_domain_label:
            cutoff = self._get_cutoff_threshold()
            if cutoff is None:
                return None

        return MetricContext(
            reference_selection=reference_selection,
            reference_indices=tuple(reference_indices),
            contact_indices=tuple(contact_indices),
            cutoff_angstrom=cutoff,
        )

    def _csv_rows_from_values(
        self,
        key: str,
        data,
        token_map: TokenMap,
        values,
        context: MetricContext,
        *,
        include_ensemble: bool,
        member,
        aggregate_kind: str,
    ) -> list[dict[str, object]]:
        """Delegate row assembly to the viewer-independent exporter."""
        from .export import build_token_rows

        member_rank = getattr(member, "rank", None) if member is not None else None
        member_label = ""
        if member is not None:
            member_label = export.model_label_for_rank(
                self._pred_files, member.rank, fallback=f"model_{member.rank}"
            )
        ensemble_state = getattr(self, "_ensemble", None)
        return build_token_rows(
            pred_files=self._pred_files,
            data=data,
            token_map=token_map,
            values=values,
            metric_key=key,
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
