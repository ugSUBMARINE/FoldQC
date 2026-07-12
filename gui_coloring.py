"""Coloring workflows for single models and ensembles."""

from __future__ import annotations

import sys

import numpy as np

from . import compute, metrics
from .compat import QtWidgets
from .mol_viewer import (
    delete_colorbar as _delete_colorbar,
)
from .mol_viewer import (
    get_viewer_name,
)
from .mol_viewer import (
    paint_categorical_labels_bulk as _paint_categorical_labels_bulk,
)
from .mol_viewer import (
    paint_plddt_class_coloring as _paint_plddt_class_coloring,
)
from .mol_viewer import (
    paint_property as _paint_property,
)
from .mol_viewer import (
    paint_property_bulk as _paint_property_bulk,
)
from .mol_viewer import (
    run_with_updates_suspended as _run_with_updates_suspended,
)
from .mol_viewer import (
    show_colorbar as _show_colorbar,
)

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()


def _viewer_call(name: str, fallback, *args, **kwargs):
    """Honor patched legacy GUI seams while controllers are migrated."""
    gui_module = sys.modules.get("FoldQC.gui")
    func = getattr(gui_module, name, fallback) if gui_module is not None else fallback
    return func(*args, **kwargs)


def delete_colorbar(*args, **kwargs):
    return _viewer_call("delete_colorbar", _delete_colorbar, *args, **kwargs)


def paint_categorical_labels_bulk(*args, **kwargs):
    return _viewer_call(
        "paint_categorical_labels_bulk",
        _paint_categorical_labels_bulk,
        *args,
        **kwargs,
    )


def paint_plddt_class_coloring(*args, **kwargs):
    return _viewer_call(
        "paint_plddt_class_coloring", _paint_plddt_class_coloring, *args, **kwargs
    )


def paint_property(*args, **kwargs):
    return _viewer_call("paint_property", _paint_property, *args, **kwargs)


def paint_property_bulk(*args, **kwargs):
    return _viewer_call("paint_property_bulk", _paint_property_bulk, *args, **kwargs)


def run_with_updates_suspended(*args, **kwargs):
    return _viewer_call(
        "run_with_updates_suspended", _run_with_updates_suspended, *args, **kwargs
    )


def show_colorbar(*args, **kwargs):
    return _viewer_call("show_colorbar", _show_colorbar, *args, **kwargs)


class ColoringController:
    def _get_obj_name(self) -> str | None:
        name = self._obj_combo.currentText().strip()
        return name if name else None

    def _selected_palette(self) -> tuple[str, bool]:
        """Return the selected palette key and reverse checkbox state."""
        combo = self._palette_combo
        try:
            key = combo.currentData()
        except AttributeError:
            key = None
        if key is None:
            key = combo.currentText()
        reverse_chk = getattr(self, "_palette_reverse_chk", None)
        reverse = bool(reverse_chk.isChecked()) if reverse_chk is not None else False
        return str(key), reverse

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

    def _get_contact_cutoff(self) -> float | None:
        """Compatibility wrapper for contact-based callers."""
        return self._get_cutoff_threshold()

    def _apply_coloring(self) -> None:
        """Compute the selected property and paint the structure."""
        obj_name = self._get_obj_name()
        if obj_name is None:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, f"No {VIEWER_NAME} target selected."
            )
            return

        if self._ensemble_members and obj_name == self._ensemble_group_name:
            self._apply_ensemble_coloring(self._ensemble_members)
            return

        member = self._selected_ensemble_member(obj_name)
        if member is not None:
            self._apply_ensemble_coloring([member])
            return

        if self._pred_data is None:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "No prediction data loaded.")
            return

        key = self._prop_combo.currentData()
        prop = metrics.PROPERTY_BY_KEY.get(key, {})
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
                self._apply_plddt_class_coloring(key, obj_name)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))
            return

        palette, reverse_palette = self._selected_palette()
        vmin, vmax = self._get_vmin_vmax()
        ref_sel = self._ref_edit.text().strip() or None

        try:
            self._ensure_current_data_for_property(prop)
            self._build_token_map_if_needed(obj_name)
            values = self._compute_property_for(
                key, ref_sel, self._pred_data, self._token_map, obj_name
            )
            if values is None:
                return
            self._validate_token_count(values, self._token_map, obj_name)
            if not self._confirm_token_overlap_for_coloring(
                self._token_map, obj_name, self._pred_data
            ):
                return
            if metrics.is_domain_label_metric(key):
                used_vmin, used_vmax = paint_categorical_labels_bulk(
                    obj_name,
                    self._token_map,
                    values,
                )
                delete_colorbar()
            else:
                used_vmin, used_vmax = paint_property(
                    obj_name,
                    self._token_map,
                    values,
                    palette=palette,
                    reverse_palette=reverse_palette,
                    vmin=vmin,
                    vmax=vmax,
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
                token_map=self._token_map,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _apply_ensemble_coloring(self, target_members: list) -> None:
        """Apply the selected property to the chosen ensemble target."""
        key = self._prop_combo.currentData()
        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        if not target_members:
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
        member_values: list[tuple[object, np.ndarray]] = []
        for member in target_members:
            self._validate_token_count(values, member.token_map, member.obj_name)
            if not self._confirm_token_overlap_for_coloring(
                member.token_map, member.obj_name, member.data
            ):
                return
            member_values.append((member, values))

        for member, values in member_values:
            used_vmin, used_vmax = paint_property_bulk(
                member.obj_name,
                member.token_map,
                values,
                palette=palette,
                reverse_palette=reverse_palette,
                vmin=vmin,
                vmax=vmax,
                rebuild=False,
            )
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

        member_values: list[tuple[object, np.ndarray]] = []
        for member in target_members:
            self._ensure_member_data_for_property(member, prop)
            values = self._compute_property_for(
                key, ref_sel, member.data, member.token_map, member.obj_name
            )
            if values is None:
                return
            self._validate_token_count(values, member.token_map, member.obj_name)
            member_values.append((member, values))

        for member, _values in member_values:
            if not self._confirm_token_overlap_for_coloring(
                member.token_map, member.obj_name, member.data
            ):
                return

        if metrics.is_domain_label_metric(key):
            used_ranges = []
            for member, values in member_values:
                used_ranges.append(
                    paint_categorical_labels_bulk(
                        member.obj_name,
                        member.token_map,
                        values,
                        rebuild=False,
                    )
                )
            delete_colorbar()
            finite_ranges = [
                (vmin, vmax)
                for vmin, vmax in used_ranges
                if np.isfinite(vmin) and np.isfinite(vmax)
            ]
            shared_vmin = min((vmin for vmin, _vmax in finite_ranges), default=0.0)
            shared_vmax = max((vmax for _vmin, vmax in finite_ranges), default=1.0)
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
                member_values,
                include_domain_labels=True,
            )
            return

        finite = np.concatenate(
            [values[np.isfinite(values)] for _, values in member_values]
        )
        shared_vmin = user_vmin
        shared_vmax = user_vmax
        if shared_vmin is None:
            shared_vmin = float(finite.min()) if finite.size else 0.0
        if shared_vmax is None:
            shared_vmax = float(finite.max()) if finite.size else 1.0
        if shared_vmin == shared_vmax:
            shared_vmax = shared_vmin + 1.0

        for member, values in member_values:
            paint_property_bulk(
                member.obj_name,
                member.token_map,
                values,
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
            object_names=[member.obj_name for member, _values in member_values],
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
            member_values,
            include_chain_stats=key == "pde_chain_mean",
        )

    def _apply_ensemble_plddt_class_coloring(
        self, key: str, target_members: list
    ) -> None:
        """Apply quality-class pLDDT coloring to selected ensemble targets."""
        member_values: list[tuple[object, np.ndarray]] = []
        for member in target_members:
            self._ensure_member_data_for_property(
                member, metrics.PROPERTY_BY_KEY["plddt_class"]
            )
            values, _source_label = compute.plddt_values_for(member.data)
            if values is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    f"pLDDT data are not available for model_{member.rank}.",
                )
                return
            self._validate_token_count(values, member.token_map, member.obj_name)
            member_values.append((member, values))

        for member, _values in member_values:
            if not self._confirm_token_overlap_for_coloring(
                member.token_map, member.obj_name, member.data
            ):
                return

        for member, values in member_values:
            paint_plddt_class_coloring(
                member.obj_name,
                values=values,
                token_map=member.token_map,
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
            key, label, member_values, include_plddt_classes=True
        )

    def _with_viewer_updates_suspended(self, func):
        """Run *func* with viewer updates suspended, then rebuild once."""
        return run_with_updates_suspended(func)

    def _apply_plddt_class_coloring(self, key: str, obj_name: str) -> None:
        """Apply the 4-class AlphaFold pLDDT colour scheme.

        Writes preferred pLDDT values to B-factors before colouring, so previous
        plugin visualisations cannot corrupt the result.
        """
        values, _source_label = compute.plddt_values_for(self._pred_data)
        if values is None:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "pLDDT data are not available for this model.",
            )
            return
        self._build_token_map_if_needed(obj_name)
        self._validate_token_count(values, self._token_map, obj_name)
        if not self._confirm_token_overlap_for_coloring(
            self._token_map, obj_name, self._pred_data
        ):
            return
        paint_plddt_class_coloring(
            obj_name,
            values=values,
            token_map=self._token_map,
        )

        delete_colorbar()
        self.setWindowTitle(f"{APP_TITLE} - pLDDT quality classes")
        self._update_statistics_for_single(
            key, obj_name, values, include_plddt_classes=True
        )
