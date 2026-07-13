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
        default_path = export.default_csv_export_path(
            getattr(self, "_pred_files", None),
            getattr(self, "_pred_data", None),
            self._prop_combo.currentData(),
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
        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        if not key or not prop:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, "Select a Color by metric before exporting."
            )
            return None
        target = self._resolve_plot_target()
        if target is None:
            return None

        if target.kind == "ensemble_group":
            return self._csv_rows_for_ensemble_group(key, prop, target)
        return self._csv_rows_for_single_target(key, prop, target)

    def _csv_rows_for_single_target(
        self,
        key: str,
        prop: dict,
        target: _PlotTarget,
    ) -> list[dict[str, object]] | None:
        """Build CSV rows for a single model or one ensemble member."""
        member = (
            (target.members or [None])[0] if target.kind == "ensemble_member" else None
        )
        include_ensemble = target.kind == "ensemble_member"

        if prop.get("ensemble_level", False):
            values = self._compute_ensemble_property(key)
            data = (
                target.data
                if target.data is not None
                else getattr(member, "data", None)
            )
            aggregate_kind = metrics.ensemble_aggregate_kind(key)
        else:
            if target.kind == "ensemble_member" and member is not None:
                self._ensure_member_data_for_property(member, prop)
                target.data = member.data
            else:
                self._ensure_current_data_for_property(prop)
                target.data = self._pred_data
            context = self._csv_metric_context(
                key, prop, target.token_map, target.obj_name
            )
            if context is None:
                return None
            compute_key = metrics.line_compute_key(key)
            values = self._compute_property_from_context(
                compute_key,
                target.data,
                target.token_map,
                context,
            )
            if values is None:
                return None
            aggregate_kind = "ensemble_member" if include_ensemble else "single_model"

        self._validate_token_count(values, target.token_map, target.label)
        if prop.get("ensemble_level", False):
            context = self._csv_metric_context(
                key, prop, target.token_map, target.obj_name
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
        prop: dict,
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

        if prop.get("ensemble_level", False):
            context = self._csv_metric_context(
                key, prop, target.token_map, target.obj_name
            )
            if context is None:
                return None
            values = self._compute_ensemble_property(key)
            self._validate_token_count(values, target.token_map, target.label)
            return self._csv_rows_from_values(
                key,
                members[0].data,
                target.token_map,
                values,
                context,
                include_ensemble=True,
                member=None,
                aggregate_kind=metrics.ensemble_aggregate_kind(key),
            )

        rows: list[dict[str, object]] = []
        compute_key = metrics.line_compute_key(key)
        for member in members:
            self._ensure_member_data_for_property(member, prop)
            context = self._csv_metric_context(
                key, prop, member.token_map, member.obj_name
            )
            if context is None:
                return None
            values = self._compute_property_from_context(
                compute_key,
                member.data,
                member.token_map,
                context,
            )
            if values is None:
                return None
            self._validate_token_count(values, member.token_map, member.obj_name)
            rows.extend(
                self._csv_rows_from_values(
                    key,
                    member.data,
                    member.token_map,
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
        prop: dict,
        token_map: TokenMap,
        obj_name: str,
    ) -> MetricContext | None:
        """Resolve reference/contact provenance for one export computation."""
        reference_selection = ""
        reference_indices: list[int] = []
        contact_indices: list[int] = []
        cutoff = None

        if prop.get("needs_ref", False):
            resolved = self._resolve_reference_indices(
                token_map, obj_name, required=True
            )
            if resolved is None:
                return None
            reference_indices = list(resolved)
            reference_selection = self._ref_edit.text().strip()

        if key in metrics.CONTACT_FILTERED_METRICS:
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
        elif metrics.is_domain_label_metric(key):
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
            ensemble_group=getattr(self, "_ensemble_group_name", "") or "",
            ensemble_member_rank=member_rank,
            ensemble_member_label=member_label,
            ensemble_aligned=getattr(self, "_ensemble_aligned", None)
            if include_ensemble
            else None,
            aggregate_kind=aggregate_kind,
        )
