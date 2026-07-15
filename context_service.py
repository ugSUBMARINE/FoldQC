"""Context application workflow."""

from __future__ import annotations

from .gui_services import ContextViewState
from .lifecycle_support import (
    ModelState,
    get_object_list,
    gui_rules,
    metrics,
    np,
    plot_data,
    reports,
)


class ContextWorkflow:
    def _refresh_objects(self) -> None:
        """Re-populate the molecular-viewer target dropdown."""
        try:
            ensemble_state = self._ensemble
            additional = [] if ensemble_state is None else [ensemble_state.group_name]
            names = self._viewer_operation(
                "object_names",
                get_object_list,
                additional_names=additional,
            )
            names = self._ordered_target_names(names)
        except Exception:
            names = []

        self._obj_combo.blockSignals(True)
        self._obj_combo.clear()
        for n in names:
            self._obj_combo.addItem(n)
            self._style_target_combo_item(self._obj_combo.count() - 1, n)
        pending_target = getattr(self._pending_session_restore, "target_name", None)
        if pending_target and self._combo_contains_text(
            self._obj_combo, pending_target
        ):
            self._select_object(pending_target)
            if not getattr(self, "_loading_prediction", False):
                self._pending_session_restore.target_name = None
        self._obj_combo.blockSignals(False)
        self._refresh_contextual_ui()

    def _ordered_target_names(self, names: list[str]) -> list[str]:
        """Return target names in stable display order."""
        ensemble_state = self._ensemble
        group_name = None if ensemble_state is None else ensemble_state.group_name
        members = sorted(
            () if ensemble_state is None else ensemble_state.members,
            key=lambda member: member.rank,
        )
        member_names = [member.obj_name for member in members]

        name_set = set(names)
        ordered = []
        if group_name in name_set:
            ordered.append(group_name)
        ordered.extend(name for name in member_names if name in name_set)

        handled = set(ordered)
        ordered.extend(
            sorted((name for name in names if name not in handled), key=str.casefold)
        )
        return ordered

    def _style_target_combo_item(self, row: int, name: str) -> None:
        """Visually distinguish the ensemble group in the target dropdown."""
        ensemble_state = self._ensemble
        if ensemble_state is None or name != ensemble_state.group_name:
            return
        item = self._obj_combo.model().item(row)
        if item is None:
            return
        font = item.font()
        font.setBold(True)
        font.setItalic(True)
        item.setFont(font)

    def _on_property_changed(self) -> None:
        """Refresh controls whose meaning depends on the selected property."""
        self._ref_label.setVisible(True)
        self._ref_edit.setVisible(True)
        self._refresh_contextual_ui()

    def _update_confidence_summary(self) -> None:
        """Fill the confidence text browser from loaded data."""
        state = self._active_model_state
        self.application.view.set_confidence_text(
            reports.format_confidence_summary(None if state is None else state.data)
        )

    def _update_property_availability(self) -> None:
        """Grey out combo items whose required data is not available."""
        for row, available in self._metric_availability():
            self.application.view.set_metric_available(row, available)

    def _metric_availability(self) -> tuple[tuple[int, bool], ...]:
        state = self._active_model_state
        if state is None or self._pred_files is None:
            return ()
        has_pae = self._target_all_supports_family("pae")
        has_pde = self._target_all_supports_family("pde")
        has_contact_probs = self._target_all_supports_family("contact_probs")
        has_plddt = self._target_all_supports_family("plddt")
        target_states = self._current_target_model_states()
        has_confidence = bool(target_states) and all(
            target_state.data.confidence is not None for target_state in target_states
        )
        has_chain_iptm = self._has_chain_iptm_metric_data()
        has_ensemble = self._ensemble is not None

        family_available = {
            "pae": has_pae,
            "pde": has_pde,
            "plddt": has_plddt,
            "contact_probs": has_contact_probs,
            "confidence": has_confidence,
        }
        availability = []
        for spec in metrics.METRICS:
            combo_row = self._property_combo_row(spec.key)
            if combo_row is None:
                continue
            available = True
            if any(not family_available[item] for item in spec.requirements):
                available = False
            if spec.key == "chain_iptm" and not has_chain_iptm:
                available = False
            if spec.ensemble_level and not has_ensemble:
                available = False
            availability.append((combo_row, available))
        return tuple(availability)

    def _has_chain_iptm_metric_data(self) -> bool:
        """Return whether loaded confidence has data for the Chain ipTM metric."""
        states = self._current_target_model_states()
        return bool(states) and all(
            state.data.confidence is not None and state.data.confidence.has_chain_iptm
            for state in states
        )

    def _select_first_available_property(self) -> None:
        """Move the property combo away from a disabled item after loading."""
        current = self._prop_combo.currentIndex()
        if current >= 0 and self.application.view.metric_is_available(current):
            return
        for spec in metrics.METRICS:
            row = self._property_combo_row(spec.key)
            if row is None:
                continue
            if self.application.view.metric_is_available(row):
                self._prop_combo.setCurrentIndex(row)
                return

    def _clear_viewer_mapping_cache(self) -> None:
        """Drop object-specific mapping state after changing prediction context."""
        self._paint_mappings = {}
        self._accepted_token_overlap_warnings = set()

    def _capture_viewer_mapping_context(self) -> tuple:
        """Snapshot object mappings for transactional model switching."""
        return (
            dict(self._paint_mappings),
            set(self._accepted_token_overlap_warnings),
        )

    def _restore_viewer_mapping_context(self, context: tuple) -> None:
        """Restore object mappings after a failed model switch."""
        (
            paint_mappings,
            accepted_warnings,
        ) = context
        self._paint_mappings = paint_mappings
        self._accepted_token_overlap_warnings = accepted_warnings

    def _property_combo_row(self, key: str) -> int | None:
        """Return the combo row registered for a metric key."""
        return self._prop_combo_rows.get(key)

    def _current_target_kind(self) -> str:
        """Return a lightweight target kind without resolving token maps or loading data."""
        try:
            obj_name = self._get_obj_name()
        except Exception:
            obj_name = None
        if not obj_name:
            return "none"
        ensemble_state = getattr(self, "_ensemble", None)
        if ensemble_state is not None and obj_name == ensemble_state.group_name:
            return "ensemble_group"
        if self._selected_ensemble_member(obj_name) is not None:
            return "ensemble_member"
        return "single"

    def _current_target_model_states(self) -> tuple[ModelState, ...]:
        """Return canonical model states addressed by the viewer target."""
        ensemble_state = getattr(self, "_ensemble", None)
        kind = self._current_target_kind()
        if kind == "ensemble_group" and ensemble_state is not None:
            return tuple(
                state
                for member in sorted(ensemble_state.members, key=lambda item: item.rank)
                if (state := self._model_states.get(member.rank)) is not None
            )
        if kind == "ensemble_member":
            member = self._selected_ensemble_member(self._get_obj_name())
            state = None if member is None else self._model_states.get(member.rank)
            return () if state is None else (state,)
        state = self._active_model_state
        return () if state is None else (state,)

    def _state_supports_family(self, state: ModelState, family: str) -> bool:
        data_attr = "token_plddt" if family == "plddt" else family
        if getattr(state.data, data_attr, None) is not None:
            return True
        if self._pred_files is None:
            return False
        model_getter = getattr(self._pred_files, "model", None)
        if not callable(model_getter):
            return False
        return model_getter(state.rank).supports(family)

    def _target_all_supports_family(self, family: str) -> bool:
        states = self._current_target_model_states()
        return bool(states) and all(
            self._state_supports_family(state, family) for state in states
        )

    def _target_any_supports_family(self, family: str) -> bool:
        return any(
            self._state_supports_family(state, family)
            for state in self._current_target_model_states()
        )

    def _has_fingerprint_data(self) -> bool:
        """Return whether fingerprint plotting has any source family available."""
        return any(
            self._target_any_supports_family(family)
            for family in ("pae", "pde", "contact_probs", "plddt")
        )

    def _has_matrix_data_family(self, family: str) -> bool:
        """Return whether a matrix family is available from files or loaded data."""
        if family == "pae":
            return self._target_all_supports_family("pae")
        if family == "pde":
            return self._target_all_supports_family("pde")
        return False

    def _current_target_has_multiple_chains(self) -> bool:
        """Return whether the current target token map has multiple chains."""
        try:
            obj_name = self._get_obj_name()
        except Exception:
            obj_name = None
        if not obj_name:
            return False

        ensemble_state = getattr(self, "_ensemble", None)
        if ensemble_state is not None and obj_name == ensemble_state.group_name:
            members = ensemble_state.members
            if not members:
                return False
            state = self._model_states.get(members[0].rank)
            return bool(
                state is not None
                and plot_data.has_multiple_token_chains(state.token_map)
            )

        member = self._selected_ensemble_member(obj_name)
        if member is not None:
            state = self._model_states.get(member.rank)
            return bool(
                state is not None
                and plot_data.has_multiple_token_chains(state.token_map)
            )

        try:
            state = self._require_active_model_state()
        except Exception:
            return False
        return plot_data.has_multiple_token_chains(state.token_map)

    def _update_plot_actions(self) -> None:
        """Refresh plot menu action availability from current GUI state."""
        self.application.view.set_plot_availability(self._plot_availability())

    def _plot_availability(self) -> tuple[tuple[str, bool, str], ...]:
        metric_key = self._prop_combo.currentData()
        target_kind = self._current_target_kind()
        has_reference = bool(self._ref_edit.text().strip())
        has_ensemble = self._ensemble is not None
        has_fingerprint_data = self._has_fingerprint_data()
        has_pae_data = self._has_matrix_data_family("pae")
        has_pde_data = self._has_matrix_data_family("pde")
        has_multiple_chains = self._current_target_has_multiple_chains()
        availability = []
        for spec in metrics.PLOTS:
            state = gui_rules.plot_action_state(
                spec.key,
                metric_key,
                target_kind,
                has_reference,
                has_ensemble,
                has_fingerprint_data=has_fingerprint_data,
                has_pae_data=has_pae_data,
                has_pde_data=has_pde_data,
                has_multiple_chains=has_multiple_chains,
            )
            tip = state.reason or f"Show {spec.label.lower()}."
            availability.append((spec.key, state.enabled, tip))
        return tuple(availability)

    def _derive_context_view_state(self) -> ContextViewState:
        key = self._prop_combo.currentData()
        target_kind = self._current_target_kind()
        field = gui_rules.field_context(
            key,
            target_kind,
            self._ensemble is not None,
            self._has_fingerprint_data(),
        )
        ref_sel = self._ref_edit.text().strip()
        cutoff_text = self._cutoff_edit.text()
        active = self._active_model_state
        return ContextViewState(
            metric_availability=self._metric_availability(),
            plot_availability=self._plot_availability(),
            reference_label=field.ref_label,
            reference_tooltip=field.ref_tooltip,
            reference_enabled=field.ref_enabled,
            cutoff_label=field.cutoff_label,
            cutoff_tooltip=field.cutoff_tooltip,
            cutoff_enabled=field.cutoff_enabled,
            confidence_text=reports.format_confidence_summary(
                None if active is None else active.data
            ),
            preview_text=gui_rules.metric_preview_text(
                key,
                target_kind,
                ref_sel,
                cutoff_text,
                self._ensemble is not None,
            ),
        )

    def _refresh_contextual_ui(self) -> None:
        """Refresh plot actions, contextual fields, and preview text together."""
        self._update_ensemble_button_state()
        self.application.view.apply_context(self._derive_context_view_state())

    def _update_context_controls(self) -> None:
        """Apply contextual Reference and cutoff control states."""
        self.application.view.apply_field_context(self._derive_context_view_state())

    def _update_metric_preview(self) -> None:
        """Show compact practical text for the selected metric and inputs."""
        self.application.view.set_preview_text(
            self._derive_context_view_state().preview_text
        )

    def _set_statistics_text(self, text: str) -> None:
        """Update the statistics panel when it exists."""
        self._presenter.show_statistics(text)

    def _update_statistics_for_single(
        self,
        key: str,
        target_name: str,
        values: np.ndarray,
        *,
        include_plddt_classes: bool = False,
        include_chain_stats: bool = False,
        include_domain_labels: bool = False,
        token_map=None,
    ) -> None:
        """Show statistics for one successfully painted target."""
        self._set_statistics_text(
            reports.format_statistics_report(
                key,
                target_name,
                [(target_name, values, token_map)],
                include_plddt_classes=include_plddt_classes,
                include_chain_stats=include_chain_stats,
                include_domain_labels=include_domain_labels,
            )
        )

    def _update_statistics_for_members(
        self,
        key: str,
        target_label: str,
        member_values: list[tuple[object, np.ndarray]],
        *,
        include_plddt_classes: bool = False,
        include_chain_stats: bool = False,
        include_domain_labels: bool = False,
    ) -> None:
        """Show statistics for successfully painted ensemble targets."""
        entries = [
            (member.obj_name, values, getattr(member, "token_map", None))
            for member, values in member_values
        ]
        self._set_statistics_text(
            reports.format_statistics_report(
                key,
                target_label,
                entries,
                include_plddt_classes=include_plddt_classes,
                include_chain_stats=include_chain_stats,
                include_domain_labels=include_domain_labels,
            )
        )
