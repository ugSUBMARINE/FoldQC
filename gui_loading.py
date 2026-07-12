"""Prediction/model lifecycle and contextual GUI coordination."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import ensemble, gui_rules, metrics, plot_data, reports
from .compat import ItemIsEnabled, QtWidgets
from .mol_viewer import ensure_structure_object, get_object_list, get_viewer_name

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()


def _score_table_has_values(value) -> bool:
    return isinstance(value, (dict, list)) and bool(value)


def _pair_score_table_has_values(value) -> bool:
    if isinstance(value, dict):
        return any(_score_table_has_values(row) for row in value.values())
    if isinstance(value, list):
        return any(_score_table_has_values(row) for row in value)
    return False


def _confidence_has_chain_iptm_metric_data(confidence) -> bool:
    if not isinstance(confidence, dict):
        return False
    return any(
        _score_table_has_values(confidence.get(key))
        for key in ("chains_iptm", "chain_iptm", "chains_ptm")
    ) or any(
        _pair_score_table_has_values(confidence.get(key))
        for key in ("pair_chains_iptm", "chain_pair_iptm")
    )


class GuiLoadingController:
    def _load_prediction_dir(self) -> None:
        """Scan the selected path and populate the model combo."""
        from .loader import discover_prediction_candidates

        path = self._dir_edit.text().strip()
        if not path:
            return
        if getattr(self, "_loading_prediction", False):
            return

        self._loading_prediction = True
        try:
            try:
                discovery = discover_prediction_candidates(path)
                if len(discovery.candidates) == 1:
                    candidate = discovery.candidates[0]
                else:
                    candidate = self._choose_prediction_candidate(discovery.candidates)
                if candidate is None:
                    return
                self._pred_files = discovery.scan(candidate)
                self._dir_edit.setText(
                    str(self._session_path_for_loaded_candidate(discovery, candidate))
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, APP_TITLE, str(exc))
                return
            self._ensemble_members = None
            self._ensemble_group_name = None
            self._ensemble_aligned = False
            self._ensemble_rmsd = None
            self._ensemble_plddt_mean = None
            self._ensemble_plddt_std = None
            self._clear_token_map_cache()

            self._model_combo.blockSignals(True)
            self._model_combo.clear()
            for model in self._pred_files.models:
                self._model_combo.addItem(model.display_label, model.rank)
            pending_rank = getattr(self._pending_session_restore, "model_rank", None)
            if pending_rank is not None:
                self._select_model_rank(pending_rank)
            self._model_combo.blockSignals(False)

            self._refresh_objects()
            self._on_model_changed()
        finally:
            self._loading_prediction = False

    def _session_path_for_loaded_candidate(self, discovery, candidate) -> Path:
        """Return the path to show/save after loading one discovery candidate."""
        input_path = getattr(discovery, "input_path", None)
        if input_path is not None:
            input_path = Path(input_path)
            if input_path.is_file():
                return input_path
        return Path(candidate.path)

    def _choose_prediction_candidate(self, candidates):
        """Let the user pick one prediction directory from multiple candidates."""
        if not candidates:
            return None
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select prediction")
        if hasattr(dialog, "setModal"):
            dialog.setModal(True)
        if hasattr(dialog, "setMinimumWidth"):
            dialog.setMinimumWidth(520)

        layout = QtWidgets.QVBoxLayout(dialog)
        table = QtWidgets.QTableWidget(len(candidates), 2, dialog)
        table.setHorizontalHeaderLabels(["Directory", "Provider"])
        for row, candidate in enumerate(candidates):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(candidate.relative_path))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(candidate.provider_label))
        if hasattr(table, "setCurrentCell"):
            table.setCurrentCell(0, 0)
        if hasattr(table, "resizeColumnsToContents"):
            table.resizeColumnsToContents()
        header = (
            table.horizontalHeader() if hasattr(table, "horizontalHeader") else None
        )
        if header is not None and hasattr(header, "setStretchLastSection"):
            header.setStretchLastSection(True)
        layout.addWidget(table)

        button_box_cls = QtWidgets.QDialogButtonBox
        standard_button = getattr(button_box_cls, "StandardButton", button_box_cls)
        button_box = button_box_cls(standard_button.Ok | standard_button.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        exec_result = dialog.exec()
        dialog_code = getattr(QtWidgets.QDialog, "DialogCode", QtWidgets.QDialog)
        accepted = getattr(dialog_code, "Accepted", 1)
        if exec_result != accepted:
            return None
        row = table.currentRow() if hasattr(table, "currentRow") else 0
        if row < 0:
            row = 0
        return candidates[row]

    def _expected_object_name(self, rank: int) -> str:
        """Return the canonical viewer object name for one model rank."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        try:
            return self._pred_files.model(rank).object_name
        except Exception:
            return f"{self._pred_files.name}_model_{rank}"

    def _ensure_model_object(self, rank: int, *, paint: bool = True) -> str | None:
        """Load or enable the viewer object for *rank*, then select it."""
        if self._pred_files is None or not self._pred_files.models:
            return None
        obj_name = self._expected_object_name(rank)
        path = self._pred_files.structure_path(rank)
        try:
            did_load = ensure_structure_object(path, obj_name, zoom=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, f"Could not load or show {path.name}:\n{exc}"
            )
            return None

        self._refresh_objects()
        self._select_object(obj_name)
        if paint and did_load:
            try:
                self._apply_plddt_class_coloring("plddt_class", obj_name)
            except Exception:
                pass  # coloring failure must not abort model selection
        return obj_name

    def _auto_select_matching_object(self) -> None:
        """Select the first combo-box entry whose name matches the prediction."""
        if self._pred_files is None:
            return
        name = self._pred_files.name
        for i in range(self._obj_combo.count()):
            obj = self._obj_combo.itemText(i)
            if obj == name or obj.startswith(name + "_model_"):
                self._obj_combo.setCurrentIndex(i)
                return

    def _on_model_changed(self) -> None:
        """Load data for the newly selected rank and update the summary."""
        if self._pred_files is None:
            return
        rank = self._model_combo.currentData()
        if rank is None:
            return

        from .loader import load_prediction_data

        try:
            self._pred_data = load_prediction_data(
                self._pred_files,
                rank,
                load_pae=False,
                load_pde=False,
                load_contact_probs=False,
            )
            self._clear_token_map_cache()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, str(exc))
            return

        self._update_confidence_summary()
        self._update_property_availability()
        pending_metric = getattr(self._pending_session_restore, "metric_key", None)
        if pending_metric:
            if not self._select_property_if_available(pending_metric):
                self._select_first_available_property()
            self._pending_session_restore.metric_key = None
        else:
            self._select_first_available_property()
        self._ensure_model_object(rank, paint=True)
        pending_target = getattr(self._pending_session_restore, "target_name", None)
        if pending_target and self._combo_contains_text(
            self._obj_combo, pending_target
        ):
            self._select_object(pending_target)
            self._pending_session_restore.target_name = None
        self._refresh_contextual_ui()

    def _refresh_objects(self) -> None:
        """Re-populate the molecular-viewer target dropdown."""
        try:
            additional = (
                [self._ensemble_group_name] if self._ensemble_group_name else []
            )
            names = get_object_list(additional_names=additional)
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
        group_name = self._ensemble_group_name
        members = sorted(self._ensemble_members or [], key=lambda member: member.rank)
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
        if name != self._ensemble_group_name:
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
        self._conf_browser.setPlainText(
            reports.format_confidence_summary(self._pred_data)
        )

    def _update_property_availability(self) -> None:
        """Grey out combo items whose required data is not available."""
        if self._pred_data is None or self._pred_files is None:
            return
        has_pae = getattr(self._pred_files, "has_pae", False)
        has_pde = getattr(self._pred_files, "has_pde", False)
        has_contact_probs = getattr(self._pred_files, "has_contact_probs", False)
        has_plddt = (
            getattr(self._pred_files, "has_plddt", False)
            or getattr(self._pred_data, "plddt", None) is not None
        )
        has_structure_plddt = (
            getattr(self._pred_files, "has_structure_plddt", False)
            or getattr(self._pred_data, "structure_plddt", None) is not None
        )
        has_any_plddt = has_plddt or has_structure_plddt
        has_confidence = (
            getattr(self._pred_data, "confidence", None) is not None
            or getattr(self._pred_data, "summary_confidence", None) is not None
        )
        has_chain_iptm = self._has_chain_iptm_metric_data()
        has_ensemble = bool(getattr(self, "_ensemble_members", None))

        model = self._prop_combo.model()
        for row, prop in enumerate(metrics.PROPERTIES):
            combo_row = self._property_combo_row(prop["key"], row)
            available = True
            if prop["needs_pae"] and not has_pae:
                available = False
            if prop["needs_pde"] and not has_pde:
                available = False
            if prop.get("needs_plddt", False) and not has_plddt:
                available = False
            if prop.get("needs_structure_plddt", False) and not has_structure_plddt:
                available = False
            if prop.get("needs_any_plddt", False) and not has_any_plddt:
                available = False
            if prop.get("needs_contact_probs", False) and not has_contact_probs:
                available = False
            if prop.get("needs_confidence", False) and not has_confidence:
                available = False
            if prop["key"] == "chain_iptm" and not has_chain_iptm:
                available = False
            if prop.get("ensemble_level", False) and not has_ensemble:
                available = False
            item = model.item(combo_row)
            if item is not None:
                flags = item.flags()
                if available:
                    item.setFlags(flags | ItemIsEnabled)
                else:
                    item.setFlags(flags & ~ItemIsEnabled)

    def _has_chain_iptm_metric_data(self) -> bool:
        """Return whether loaded confidence has data for the Chain ipTM metric."""
        if self._pred_data is None:
            return False
        for attr in ("confidence", "summary_confidence"):
            confidence = getattr(self._pred_data, attr, None)
            if _confidence_has_chain_iptm_metric_data(confidence):
                return True
        return False

    def _select_first_available_property(self) -> None:
        """Move the property combo away from a disabled item after loading."""
        model = self._prop_combo.model()
        current = self._prop_combo.currentIndex()
        if current >= 0:
            item = model.item(current)
            if item is not None and item.flags() & ItemIsEnabled:
                return
        for prop in metrics.PROPERTIES:
            row = self._property_combo_row(prop["key"], -1)
            if row < 0:
                continue
            item = model.item(row)
            if item is not None and item.flags() & ItemIsEnabled:
                self._prop_combo.setCurrentIndex(row)
                return

    def _clear_token_map_cache(self) -> None:
        """Drop cached token-map state after changing prediction context."""
        self._token_map = None
        self._token_map_obj = None  # type: ignore[attr-defined]
        self._token_map_structure_path = None  # type: ignore[attr-defined]
        self._accepted_token_overlap_warnings = set()

    def _property_combo_row(self, key: str, fallback: int = -1) -> int:
        """Return the combo row for a property key, allowing older tests to omit maps."""
        rows = getattr(self, "_prop_combo_rows", None)
        if rows is None:
            return fallback
        return rows.get(key, fallback)

    def _current_target_kind(self) -> str:
        """Return a lightweight target kind without resolving token maps or loading data."""
        try:
            obj_name = self._get_obj_name()
        except Exception:
            obj_name = None
        if not obj_name:
            return "none"
        if obj_name == getattr(self, "_ensemble_group_name", None):
            return "ensemble_group"
        if self._selected_ensemble_member(obj_name) is not None:
            return "ensemble_member"
        return "single"

    def _has_fingerprint_data(self) -> bool:
        """Return whether fingerprint plotting has any source family available."""
        pred_files = getattr(self, "_pred_files", None)
        pred_data = getattr(self, "_pred_data", None)
        if pred_files is not None:
            if (
                getattr(pred_files, "has_pae", False)
                or getattr(pred_files, "has_pde", False)
                or getattr(pred_files, "has_contact_probs", False)
                or getattr(pred_files, "has_plddt", False)
                or getattr(pred_files, "has_structure_plddt", False)
            ):
                return True
        if pred_data is not None:
            if (
                getattr(pred_data, "pae", None) is not None
                or getattr(pred_data, "pde", None) is not None
                or getattr(pred_data, "contact_probs", None) is not None
                or getattr(pred_data, "plddt", None) is not None
                or getattr(pred_data, "structure_plddt", None) is not None
            ):
                return True
        for member in getattr(self, "_ensemble_members", None) or []:
            data = getattr(member, "data", None)
            if data is None:
                continue
            if (
                getattr(data, "pae", None) is not None
                or getattr(data, "pde", None) is not None
                or getattr(data, "contact_probs", None) is not None
                or getattr(data, "plddt", None) is not None
                or getattr(data, "structure_plddt", None) is not None
            ):
                return True
        return False

    def _has_matrix_data_family(self, family: str) -> bool:
        """Return whether a matrix family is available from files or loaded data."""
        pred_files = getattr(self, "_pred_files", None)
        pred_data = getattr(self, "_pred_data", None)
        if family == "pae":
            return bool(
                getattr(pred_files, "has_pae", False)
                or getattr(pred_data, "pae", None) is not None
            )
        if family == "pde":
            return bool(
                getattr(pred_files, "has_pde", False)
                or getattr(pred_data, "pde", None) is not None
            )
        return False

    def _current_target_has_multiple_chains(self) -> bool:
        """Return whether the current target token map has multiple chains."""
        try:
            obj_name = self._get_obj_name()
        except Exception:
            obj_name = None
        if not obj_name:
            return False

        if obj_name == getattr(self, "_ensemble_group_name", None):
            members = getattr(self, "_ensemble_members", None) or []
            if not members:
                return False
            return plot_data.has_multiple_token_chains(members[0].token_map)

        member = self._selected_ensemble_member(obj_name)
        if member is not None:
            return plot_data.has_multiple_token_chains(member.token_map)

        if getattr(self, "_pred_data", None) is None:
            return False
        try:
            self._build_token_map_if_needed(obj_name)
        except Exception:
            return False
        return plot_data.has_multiple_token_chains(self._token_map)

    def _update_plot_actions(self) -> None:
        """Refresh plot menu action availability from current GUI state."""
        actions = getattr(self, "_plot_actions", None)
        if not actions:
            return
        metric_key = self._prop_combo.currentData()
        target_kind = self._current_target_kind()
        has_reference = bool(self._ref_edit.text().strip())
        has_ensemble = bool(getattr(self, "_ensemble_members", None))
        has_fingerprint_data = self._has_fingerprint_data()
        has_pae_data = self._has_matrix_data_family("pae")
        has_pde_data = self._has_matrix_data_family("pde")
        has_multiple_chains = self._current_target_has_multiple_chains()
        for label, plot_type in metrics.PLOT_TYPES:
            action = actions.get(plot_type)
            if action is None:
                continue
            state = gui_rules.plot_action_state(
                plot_type,
                metric_key,
                target_kind,
                has_reference,
                has_ensemble,
                has_fingerprint_data=has_fingerprint_data,
                has_pae_data=has_pae_data,
                has_pde_data=has_pde_data,
                has_multiple_chains=has_multiple_chains,
            )
            action.setEnabled(state.enabled)
            tip = state.reason or f"Show {label.lower()}."
            if hasattr(action, "setToolTip"):
                action.setToolTip(tip)
            if hasattr(action, "setStatusTip"):
                action.setStatusTip(tip)

    def _refresh_contextual_ui(self) -> None:
        """Refresh plot actions, contextual fields, and preview text together."""
        self._update_plot_actions()
        self._update_context_controls()
        self._update_metric_preview()

    def _update_context_controls(self) -> None:
        """Apply contextual Reference and cutoff control states."""
        key = self._prop_combo.currentData()
        context = gui_rules.field_context(
            key,
            self._current_target_kind(),
            bool(getattr(self, "_ensemble_members", None)),
            self._has_fingerprint_data(),
        )
        self._ref_label.setText(context.ref_label)
        self._ref_label.setToolTip(context.ref_tooltip)
        self._ref_edit.setEnabled(context.ref_enabled)
        self._ref_edit.setToolTip(context.ref_tooltip)
        self._cutoff_label.setText(context.cutoff_label)
        self._cutoff_label.setToolTip(context.cutoff_tooltip)
        self._cutoff_edit.setEnabled(context.cutoff_enabled)
        self._cutoff_edit.setToolTip(context.cutoff_tooltip)

    def _update_metric_preview(self) -> None:
        """Show compact practical text for the selected metric and inputs."""
        preview = getattr(self, "_preview_label", None)
        if preview is None:
            return
        key = self._prop_combo.currentData()
        ref_sel = self._ref_edit.text().strip()
        target_kind = self._current_target_kind()
        cutoff_edit = getattr(self, "_cutoff_edit", None)
        cutoff_text = cutoff_edit.text() if cutoff_edit is not None else ""
        preview.setText(
            gui_rules.metric_preview_text(
                key,
                target_kind,
                ref_sel,
                cutoff_text,
                bool(getattr(self, "_ensemble_members", None)),
            )
        )

    def _set_statistics_text(self, text: str) -> None:
        """Update the statistics panel when it exists."""
        browser = getattr(self, "_stats_browser", None)
        if browser is not None:
            browser.setPlainText(text)

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

    def _show_ensemble(self) -> None:
        """Load, group, optionally align, and activate the ensemble."""
        if self._pred_files is None:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, "No prediction output loaded."
            )
            return
        if not self._pred_files.supports_ensemble:
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Ensemble mode requires at least two model files.",
            )
            return

        skip_alignment = self._ask_skip_ensemble_alignment()
        if skip_alignment is None:
            return

        try:
            group_name, members = ensemble.build_members(self._pred_files)
            ensemble.validate_members(members)
            metrics_result = ensemble.prepare_metrics(
                members, skip_alignment=skip_alignment
            )
            self._ensemble_members = members
            self._ensemble_group_name = group_name
            self._ensemble_aligned = metrics_result.aligned
            self._ensemble_rmsd = metrics_result.rmsd
            self._ensemble_plddt_mean = metrics_result.plddt_mean
            self._ensemble_plddt_std = metrics_result.plddt_std
            self._refresh_objects()
            if self._ensemble_group_name:
                self._select_object(self._ensemble_group_name)
            self._update_property_availability()
            self._select_property("ensemble_rmsd")

            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                f"Loaded {len(members)} ensemble models into group "
                f"'{self._ensemble_group_name}'.\n"
                f"RMSD was computed using {metrics_result.mode_label}.\n\n"
                "Use Apply Coloring to color the selected target.",
            )
            self._refresh_contextual_ui()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _ask_skip_ensemble_alignment(self) -> bool | None:
        """Return True for expert-mode no-align, False for auto-align, None on cancel."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"{APP_TITLE} Ensemble")
        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel(
            "Load all models as separate objects and group them.\n"
            "By default, models are aligned to model_0 using a high-confidence "
            "protein core."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        checkbox = QtWidgets.QCheckBox(
            f"Use current {VIEWER_NAME} coordinates; do not align"
        )
        checkbox.setToolTip(
            "Skip automatic core alignment and compute ensemble RMSD from the current "
            f"{VIEWER_NAME} object coordinates."
        )
        layout.addWidget(checkbox)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch()
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        ok_btn.setToolTip("Load the ensemble with the selected alignment option.")
        cancel_btn.setToolTip(
            "Close this dialog without loading or updating the ensemble."
        )
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        button_row.addWidget(ok_btn)
        button_row.addWidget(cancel_btn)
        layout.addLayout(button_row)

        return checkbox.isChecked() if dialog.exec() == 1 else None
