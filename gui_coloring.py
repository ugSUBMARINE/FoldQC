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
from .viewer_transactions import ColorbarChange, PaintTransaction

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()


class ColoringWorkflow:
    def _transactional_paint(
        self,
        targets,
        *,
        kind: str,
        palette: str = "",
        reverse_palette: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """Apply one fully compensating paint operation through the viewer port."""
        viewer = getattr(self, "_viewer", None)
        if viewer is None:
            return None
        targets = tuple(targets)
        colorbar = ColorbarChange(
            "replace" if kind == "continuous" else "remove",
            palette=palette if kind == "continuous" else "",
            reverse_palette=reverse_palette,
        )
        transaction = PaintTransaction(viewer, targets, colorbar)
        if kind == "continuous":

            def paint():
                return viewer.paint_continuous(
                    targets,
                    palette=palette,
                    reverse_palette=reverse_palette,
                    vmin=vmin,
                    vmax=vmax,
                    rebuild=False,
                )
        elif kind == "categorical":

            def paint():
                return viewer.paint_categorical(targets, rebuild=False)
        elif kind == "plddt_class":

            def paint():
                return viewer.paint_plddt_classes(targets, rebuild=False)
        else:
            raise ValueError(f"Unknown paint transaction kind: {kind!r}.")
        try:
            result = transaction.execute(paint)
        except Exception:
            self._staged_paint_mappings = {}
            raise
        mappings = dict(getattr(self, "_paint_mappings", {}))
        mappings.update(getattr(self, "_staged_paint_mappings", {}))
        self._paint_mappings = mappings
        self._staged_paint_mappings = {}
        return result

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
        ensemble_state = getattr(self, "_ensemble", None)
        for member in () if ensemble_state is None else ensemble_state.members:
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
        spec = metrics.METRICS.require(key)
        obj_name = target.obj_name
        if self._defer_action_for_data(
            target,
            spec.load_capabilities,
            self._apply_coloring,
            error_title=f"{APP_TITLE} - error",
            deferred_action=self.services.analysis.capture_current("color"),
        ):
            return

        if spec.ensemble_level:
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
                self._ensure_current_data_for_property(spec)
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
            self._ensure_current_data_for_property(spec)
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
            if spec.is_domain_label:
                result = self._transactional_paint(
                    (PaintTarget(obj_name, token_map, values, mapping),),
                    kind="categorical",
                )
                if result is None:
                    used_vmin, used_vmax = paint_categorical_labels_bulk(
                        obj_name,
                        token_map,
                        values,
                        mapping=mapping,
                    )
                    delete_colorbar()
                else:
                    used_vmin, used_vmax = result.vmin, result.vmax
            else:
                result = self._transactional_paint(
                    (PaintTarget(obj_name, token_map, values, mapping),),
                    kind="continuous",
                    palette=palette,
                    reverse_palette=reverse_palette,
                    vmin=vmin,
                    vmax=vmax,
                )
                if result is None:
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
                else:
                    used_vmin, used_vmax = result.vmin, result.vmax
            self.setWindowTitle(
                f"{APP_TITLE} - {key} [{used_vmin:.2f}, {used_vmax:.2f}]"
            )
            self._update_statistics_for_single(
                key,
                obj_name,
                values,
                include_chain_stats=key == "pde_chain_mean",
                include_domain_labels=spec.is_domain_label,
                token_map=token_map,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _apply_ensemble_coloring(self, target_members: list) -> None:
        """Apply the selected property to the chosen ensemble target."""
        key = self._prop_combo.currentData()
        spec = metrics.METRICS.require(key)
        if not target_members:
            return
        target = self._resolve_plot_target()
        if target is None:
            return
        if self._defer_action_for_data(
            target,
            spec.load_capabilities,
            self._apply_coloring,
            error_title=f"{APP_TITLE} - error",
            deferred_action=self.services.analysis.capture_current("color"),
        ):
            return

        try:
            if getattr(self, "_viewer", None) is not None:
                self._dispatch_ensemble_coloring(key, spec, target_members)
            else:
                self._with_viewer_updates_suspended(
                    lambda: self._dispatch_ensemble_coloring(key, spec, target_members)
                )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _dispatch_ensemble_coloring(
        self, key: str, spec: metrics.MetricSpec, target_members: list
    ) -> None:
        """Route ensemble coloring while viewer updates are suspended."""
        if spec.ensemble_level:
            self._apply_ensemble_level_property(key, target_members)
        elif key == "plddt_class":
            self._apply_ensemble_plddt_class_coloring(key, target_members)
        else:
            self._apply_individual_property_to_ensemble(key, spec, target_members)

    def _apply_ensemble_level_property(self, key: str, target_members: list) -> None:
        """Compute one ensemble-level array and paint it onto selected targets."""
        values = self._compute_ensemble_property(key)

        palette, reverse_palette = self._selected_palette()
        vmin, vmax = self._get_vmin_vmax()
        used_vmin = used_vmax = None
        member_values: list[tuple[object, object, np.ndarray, object]] = []
        for member in target_members:
            state = self._canonical_state_for_ensemble_member(member)
            self._validate_token_count(values, state.token_map, member.obj_name)
            mapping = self._prepare_paint_mapping(
                state.token_map, member.obj_name, state.data
            )
            if not self._confirm_token_overlap_for_coloring(
                state.token_map,
                member.obj_name,
                state.data,
                mapping=mapping,
            ):
                return
            member_values.append((member, state, values, mapping))

        targets = [
            PaintTarget(member.obj_name, state.token_map, item_values, mapping)
            for member, state, item_values, mapping in member_values
        ]
        result = self._transactional_paint(
            targets,
            kind="continuous",
            palette=palette,
            reverse_palette=reverse_palette,
            vmin=vmin,
            vmax=vmax,
        )
        if result is None:
            result = paint_properties_bulk(
                targets,
                palette=palette,
                reverse_palette=reverse_palette,
                vmin=vmin,
                vmax=vmax,
                rebuild=False,
            )
        used_vmin, used_vmax = result.vmin, result.vmax
        if getattr(self, "_viewer", None) is None:
            show_colorbar(
                palette,
                reverse_palette,
                used_vmin,
                used_vmax,
                object_names=[member.obj_name for member in target_members],
            )
        ensemble_state = self._ensemble
        label = (
            ensemble_state.group_name
            if len(target_members) > 1
            else target_members[0].obj_name
        )
        self.setWindowTitle(
            f"{APP_TITLE} - {key} on {label} [{used_vmin:.2f}, {used_vmax:.2f}]"
        )
        self._update_statistics_for_single(key, label, values)

    def _apply_individual_property_to_ensemble(
        self, key: str, spec: metrics.MetricSpec, target_members: list
    ) -> None:
        """Compute selected per-model properties for selected ensemble targets."""
        palette, reverse_palette = self._selected_palette()
        user_vmin, user_vmax = self._get_vmin_vmax()
        ref_sel = self._ref_edit.text().strip() or None

        member_values: list[tuple[object, object, np.ndarray, object]] = []
        for member in target_members:
            self._ensure_member_data_for_property(member, spec)
            state = self._canonical_state_for_ensemble_member(member)
            values = self._compute_property_for(
                key, ref_sel, state.data, state.token_map, member.obj_name
            )
            if values is None:
                return
            self._validate_token_count(values, state.token_map, member.obj_name)
            mapping = self._prepare_paint_mapping(
                state.token_map, member.obj_name, state.data
            )
            if not self._confirm_token_overlap_for_coloring(
                state.token_map,
                member.obj_name,
                state.data,
                mapping=mapping,
            ):
                return
            member_values.append((member, state, values, mapping))

        if spec.is_domain_label:
            targets = [
                PaintTarget(member.obj_name, state.token_map, values, mapping)
                for member, state, values, mapping in member_values
            ]
            result = self._transactional_paint(targets, kind="categorical")
            if result is None:
                result = paint_categorical_labels_batch(targets, rebuild=False)
                delete_colorbar()
            shared_vmin, shared_vmax = result.vmin, result.vmax
            ensemble_state = self._ensemble
            label = (
                ensemble_state.group_name
                if len(target_members) > 1
                else target_members[0].obj_name
            )
            self.setWindowTitle(
                f"{APP_TITLE} - {key} on {label} [{shared_vmin:.2f}, {shared_vmax:.2f}]"
            )
            self._update_statistics_for_members(
                key,
                label,
                [
                    (member, values)
                    for member, _state, values, _mapping in member_values
                ],
                include_domain_labels=True,
            )
            return

        finite = np.concatenate(
            [
                values[np.isfinite(values)]
                for _member, _state, values, _mapping in member_values
            ]
        )
        shared_vmin = user_vmin
        shared_vmax = user_vmax
        if shared_vmin is None:
            shared_vmin = float(finite.min()) if finite.size else 0.0
        if shared_vmax is None:
            shared_vmax = float(finite.max()) if finite.size else 1.0
        if shared_vmin == shared_vmax:
            shared_vmax = shared_vmin + 1.0

        targets = [
            PaintTarget(member.obj_name, state.token_map, values, mapping)
            for member, state, values, mapping in member_values
        ]
        result = self._transactional_paint(
            targets,
            kind="continuous",
            palette=palette,
            reverse_palette=reverse_palette,
            vmin=shared_vmin,
            vmax=shared_vmax,
        )
        if result is None:
            paint_properties_bulk(
                targets,
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
                    member.obj_name
                    for member, _state, _values, _mapping in member_values
                ],
            )
        ensemble_state = self._ensemble
        label = (
            ensemble_state.group_name
            if len(target_members) > 1
            else target_members[0].obj_name
        )
        self.setWindowTitle(
            f"{APP_TITLE} - {key} on {label} [{shared_vmin:.2f}, {shared_vmax:.2f}]"
        )
        self._update_statistics_for_members(
            key,
            label,
            [(member, values) for member, _state, values, _mapping in member_values],
            include_chain_stats=key == "pde_chain_mean",
        )

    def _apply_ensemble_plddt_class_coloring(
        self, key: str, target_members: list
    ) -> None:
        """Apply quality-class pLDDT coloring to selected ensemble targets."""
        member_values: list[tuple[object, object, np.ndarray, object]] = []
        for member in target_members:
            self._ensure_member_data_for_property(
                member, metrics.METRICS.require("plddt_class")
            )
            state = self._canonical_state_for_ensemble_member(member)
            values, _source = compute.plddt_values_for(state.data)
            if values is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    f"pLDDT data are not available for model_{member.rank}.",
                )
                return
            self._validate_token_count(values, state.token_map, member.obj_name)
            mapping = self._prepare_paint_mapping(
                state.token_map, member.obj_name, state.data
            )
            if not self._confirm_token_overlap_for_coloring(
                state.token_map,
                member.obj_name,
                state.data,
                mapping=mapping,
            ):
                return
            member_values.append((member, state, values, mapping))

        targets = [
            PaintTarget(member.obj_name, state.token_map, values, mapping)
            for member, state, values, mapping in member_values
        ]
        result = self._transactional_paint(targets, kind="plddt_class")
        if result is None:
            paint_plddt_class_batch(targets, rebuild=False)
            delete_colorbar()
        ensemble_state = self._ensemble
        label = (
            ensemble_state.group_name
            if len(target_members) > 1
            else target_members[0].obj_name
        )
        self.setWindowTitle(f"{APP_TITLE} - pLDDT quality classes on {label}")
        self._update_statistics_for_members(
            key,
            label,
            [(member, values) for member, _state, values, _mapping in member_values],
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
        result = self._transactional_paint(
            (PaintTarget(obj_name, token_map, values, mapping),),
            kind="plddt_class",
        )
        if result is None:
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
