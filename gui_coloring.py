"""Coloring workflows for single models and ensembles."""

from __future__ import annotations

import numpy as np

from . import compute, metrics
from .compat import QtWidgets
from .mol_viewer import (
    PaintTarget,
    delete_colorbar,
    get_viewer_name,
    paint_categorical_labels_batch,
    paint_categorical_labels_bulk,
    paint_plddt_class_batch,
    paint_plddt_class_coloring,
    paint_properties_bulk,
    paint_property,
    run_with_updates_suspended,
    show_colorbar,
)

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()


class ColoringController:
    def _get_obj_name(self) -> str | None:
        name = self._obj_combo.currentText().strip()
        return name if name else None

    def _selected_palette(self) -> tuple[str, bool]:
        """Return the selected palette key and reverse checkbox state."""
        return (
            str(self._palette_combo.currentData()),
            bool(self._palette_reverse_chk.isChecked()),
        )

    def _selected_ensemble_member(self, obj_name: str):
        """Return the active ensemble member matching *obj_name*, if any."""
        for member in getattr(self, "_ensemble_members", None) or []:
            if member.obj_name == obj_name:
                return member
        return None

    def _get_vmin_vmax(self) -> tuple[float | None, float | None]:
        def _parse(text: str) -> float | None:
            t = text.strip()
            if not t or t.lower() == "auto":
                return None
            try:
                return float(t)
            except ValueError:
                return None

        return _parse(self._vmin_edit.text()), _parse(self._vmax_edit.text())

    def _get_cutoff_threshold(self) -> float | None:
        """Return the user-entered positive cutoff/threshold in Å."""
        edit = getattr(self, "_cutoff_edit", None)
        text = "5.0" if edit is None else edit.text().strip()
        if not text:
            text = "5.0"
        try:
            cutoff = float(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "Cutoff / threshold must be a positive number in Å.",
            )
            return None
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "Cutoff / threshold must be greater than 0 Å.",
            )
            return None
        return cutoff

    def _apply_coloring(self) -> None:
        """Compute the selected property and paint the structure."""
        target = self._resolve_plot_target()
        if target is None:
            return
        if target.kind.startswith("ensemble"):
            self._apply_ensemble_coloring(list(target.members))
            return
        key = self._prop_combo.currentData()
        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        obj_name = target.obj_name
        if self._defer_action_for_data(
            target,
            metrics.metric_load_flags(prop),
            self._apply_coloring,
            error_title=f"{APP_TITLE} - error",
        ):
            return

        if metrics.PROPERTY_BY_KEY.get(key, {}).get("ensemble_level", False):
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "This property requires an active ensemble.\n"
                "Use the Ensemble… button, then choose the ensemble group "
                f"or one of its model objects as the {VIEWER_NAME} target.",
            )
            return

        # Class-based pLDDT coloring bypasses the B-factor/spectrum path
        if key == "plddt_class":
            try:
                self._ensure_current_data_for_property(prop)
                self._apply_plddt_class_coloring(
                    key, obj_name, model_state=target.model_states[0]
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))
            return

        palette, reverse_palette = self._selected_palette()
        vmin, vmax = self._get_vmin_vmax()
        ref_sel = self._ref_edit.text().strip() or None

        try:
            self._ensure_current_data_for_property(prop)
            state = target.model_states[0]
            data = state.data
            token_map = state.token_map
            values = self._compute_property_for(key, ref_sel, data, token_map, obj_name)
            if values is None:
                return
            self._validate_token_count(values, token_map, obj_name)
            mapping = self._prepare_paint_mapping(token_map, obj_name, data)
            if not self._confirm_token_overlap_for_coloring(
                token_map,
                obj_name,
                data,
                mapping=mapping,
            ):
                return
            if metrics.is_domain_label_metric(key):
                used_vmin, used_vmax = paint_categorical_labels_bulk(
                    obj_name,
                    token_map,
                    values,
                    mapping=mapping,
                )
                delete_colorbar()
            else:
                used_vmin, used_vmax = paint_property(
                    obj_name,
                    token_map,
                    values,
                    palette=palette,
                    reverse_palette=reverse_palette,
                    vmin=vmin,
                    vmax=vmax,
                    mapping=mapping,
                )
                show_colorbar(
                    palette,
                    reverse_palette,
                    used_vmin,
                    used_vmax,
                    object_names=[obj_name],
                )
            self.setWindowTitle(
                f"{APP_TITLE} - {key} [{used_vmin:.2f}, {used_vmax:.2f}]"
            )
            self._update_statistics_for_single(
                key,
                obj_name,
                values,
                include_chain_stats=key == "pde_chain_mean",
                include_domain_labels=metrics.is_domain_label_metric(key),
                token_map=token_map,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _apply_ensemble_coloring(self, target_members: list) -> None:
        """Apply the selected property to the chosen ensemble target."""
        key = self._prop_combo.currentData()
        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        if not target_members:
            return
        target = self._resolve_plot_target()
        if target is None:
            return
        if self._defer_action_for_data(
            target,
            metrics.metric_load_flags(prop),
            self._apply_coloring,
            error_title=f"{APP_TITLE} - error",
        ):
            return

        try:
            self._with_viewer_updates_suspended(
                lambda: self._dispatch_ensemble_coloring(key, prop, target_members)
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _dispatch_ensemble_coloring(
        self, key: str, prop: dict, target_members: list
    ) -> None:
        """Route ensemble coloring while viewer updates are suspended."""
        if prop.get("ensemble_level", False):
            self._apply_ensemble_level_property(key, target_members)
        elif key == "plddt_class":
            self._apply_ensemble_plddt_class_coloring(key, target_members)
        else:
            self._apply_individual_property_to_ensemble(key, prop, target_members)

    def _apply_ensemble_level_property(self, key: str, target_members: list) -> None:
        """Compute one ensemble-level array and paint it onto selected targets."""
        values = self._compute_ensemble_property(key)

        palette, reverse_palette = self._selected_palette()
        vmin, vmax = self._get_vmin_vmax()
        used_vmin = used_vmax = None
        member_values: list[tuple[object, np.ndarray, object]] = []
        for member in target_members:
            self._validate_token_count(values, member.token_map, member.obj_name)
            mapping = self._prepare_paint_mapping(
                member.token_map, member.obj_name, member.data
            )
            if not self._confirm_token_overlap_for_coloring(
                member.token_map,
                member.obj_name,
                member.data,
                mapping=mapping,
            ):
                return
            member_values.append((member, values, mapping))

        result = paint_properties_bulk(
            [
                PaintTarget(member.obj_name, member.token_map, item_values, mapping)
                for member, item_values, mapping in member_values
            ],
            palette=palette,
            reverse_palette=reverse_palette,
            vmin=vmin,
            vmax=vmax,
            rebuild=False,
        )
        used_vmin, used_vmax = result.vmin, result.vmax
        show_colorbar(
            palette,
            reverse_palette,
            used_vmin,
            used_vmax,
            object_names=[member.obj_name for member in target_members],
        )
        label = (
            self._ensemble_group_name
            if len(target_members) > 1
            else target_members[0].obj_name
        )
        self.setWindowTitle(
            f"{APP_TITLE} - {key} on {label} [{used_vmin:.2f}, {used_vmax:.2f}]"
        )
        self._update_statistics_for_single(key, label, values)

    def _apply_individual_property_to_ensemble(
        self, key: str, prop: dict, target_members: list
    ) -> None:
        """Compute selected per-model properties for selected ensemble targets."""
        palette, reverse_palette = self._selected_palette()
        user_vmin, user_vmax = self._get_vmin_vmax()
        ref_sel = self._ref_edit.text().strip() or None

        member_values: list[tuple[object, np.ndarray, object]] = []
        for member in target_members:
            self._ensure_member_data_for_property(member, prop)
            values = self._compute_property_for(
                key, ref_sel, member.data, member.token_map, member.obj_name
            )
            if values is None:
                return
            self._validate_token_count(values, member.token_map, member.obj_name)
            mapping = self._prepare_paint_mapping(
                member.token_map, member.obj_name, member.data
            )
            if not self._confirm_token_overlap_for_coloring(
                member.token_map,
                member.obj_name,
                member.data,
                mapping=mapping,
            ):
                return
            member_values.append((member, values, mapping))

        if metrics.is_domain_label_metric(key):
            result = paint_categorical_labels_batch(
                [
                    PaintTarget(member.obj_name, member.token_map, values, mapping)
                    for member, values, mapping in member_values
                ],
                rebuild=False,
            )
            delete_colorbar()
            shared_vmin, shared_vmax = result.vmin, result.vmax
            label = (
                self._ensemble_group_name
                if len(target_members) > 1
                else target_members[0].obj_name
            )
            self.setWindowTitle(
                f"{APP_TITLE} - {key} on {label} [{shared_vmin:.2f}, {shared_vmax:.2f}]"
            )
            self._update_statistics_for_members(
                key,
                label,
                [(member, values) for member, values, _mapping in member_values],
                include_domain_labels=True,
            )
            return

        finite = np.concatenate(
            [values[np.isfinite(values)] for _, values, _mapping in member_values]
        )
        shared_vmin = user_vmin
        shared_vmax = user_vmax
        if shared_vmin is None:
            shared_vmin = float(finite.min()) if finite.size else 0.0
        if shared_vmax is None:
            shared_vmax = float(finite.max()) if finite.size else 1.0
        if shared_vmin == shared_vmax:
            shared_vmax = shared_vmin + 1.0

        paint_properties_bulk(
            [
                PaintTarget(member.obj_name, member.token_map, values, mapping)
                for member, values, mapping in member_values
            ],
            palette=palette,
            reverse_palette=reverse_palette,
            vmin=shared_vmin,
            vmax=shared_vmax,
            rebuild=False,
        )
        show_colorbar(
            palette,
            reverse_palette,
            shared_vmin,
            shared_vmax,
            object_names=[
                member.obj_name for member, _values, _mapping in member_values
            ],
        )
        label = (
            self._ensemble_group_name
            if len(target_members) > 1
            else target_members[0].obj_name
        )
        self.setWindowTitle(
            f"{APP_TITLE} - {key} on {label} [{shared_vmin:.2f}, {shared_vmax:.2f}]"
        )
        self._update_statistics_for_members(
            key,
            label,
            [(member, values) for member, values, _mapping in member_values],
            include_chain_stats=key == "pde_chain_mean",
        )

    def _apply_ensemble_plddt_class_coloring(
        self, key: str, target_members: list
    ) -> None:
        """Apply quality-class pLDDT coloring to selected ensemble targets."""
        member_values: list[tuple[object, np.ndarray, object]] = []
        for member in target_members:
            self._ensure_member_data_for_property(
                member, metrics.PROPERTY_BY_KEY["plddt_class"]
            )
            values, _source = compute.plddt_values_for(member.data)
            if values is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    f"pLDDT data are not available for model_{member.rank}.",
                )
                return
            self._validate_token_count(values, member.token_map, member.obj_name)
            mapping = self._prepare_paint_mapping(
                member.token_map, member.obj_name, member.data
            )
            if not self._confirm_token_overlap_for_coloring(
                member.token_map,
                member.obj_name,
                member.data,
                mapping=mapping,
            ):
                return
            member_values.append((member, values, mapping))

        paint_plddt_class_batch(
            [
                PaintTarget(member.obj_name, member.token_map, values, mapping)
                for member, values, mapping in member_values
            ],
            rebuild=False,
        )
        delete_colorbar()
        label = (
            self._ensemble_group_name
            if len(target_members) > 1
            else target_members[0].obj_name
        )
        self.setWindowTitle(f"{APP_TITLE} - pLDDT quality classes on {label}")
        self._update_statistics_for_members(
            key,
            label,
            [(member, values) for member, values, _mapping in member_values],
            include_plddt_classes=True,
        )

    def _with_viewer_updates_suspended(self, func):
        """Run *func* with viewer updates suspended, then rebuild once."""
        return run_with_updates_suspended(func)

    def _apply_plddt_class_coloring(
        self, key: str, obj_name: str, *, model_state=None
    ) -> None:
        """Apply the 4-class AlphaFold pLDDT colour scheme.

        Writes preferred pLDDT values to B-factors before colouring, so previous
        plugin visualisations cannot corrupt the result.
        """
        state = model_state or self._require_active_model_state()
        data = state.data
        token_map = state.token_map
        values, _source = compute.plddt_values_for(data)
        if values is None:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "pLDDT data are not available for this model.",
            )
            return
        self._validate_token_count(values, token_map, obj_name)
        mapping = self._prepare_paint_mapping(token_map, obj_name, data)
        if not self._confirm_token_overlap_for_coloring(
            token_map,
            obj_name,
            data,
            mapping=mapping,
        ):
            return
        paint_plddt_class_coloring(
            obj_name,
            values=values,
            token_map=token_map,
            mapping=mapping,
        )

        delete_colorbar()
        self.setWindowTitle(f"{APP_TITLE} - pLDDT quality classes")
        self._update_statistics_for_single(
            key, obj_name, values, include_plddt_classes=True
        )
