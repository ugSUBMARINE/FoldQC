"""Qt widget rendering adapter for FoldQC's application services."""

from __future__ import annotations

from .compat import ItemIsEnabled
from .gui_services import (
    PREVIEW_DETAILS_TOOLTIP,
    BusyViewState,
    ContextViewState,
    LifecycleUiUpdate,
    StatisticsSelectionViewState,
    TargetChoice,
)


class QtDialogView:
    def __init__(self, dialog, widgets) -> None:
        self.dialog = dialog
        self.widgets = widgets
        self._busy = False
        self._ensemble_enabled = False
        self._ensemble_tooltip = "Load a prediction with at least two models first."
        self._comparison_enabled = False
        self._comparison_tooltip = "Load a prediction with at least two models first."
        self._preview_details_text = ""

    @property
    def preview_details_text(self) -> str:
        return self._preview_details_text

    def select_object(self, name: str) -> None:
        combo = self.widgets._obj_combo
        for index in range(combo.count()):
            if combo.itemText(index) == name:
                combo.setCurrentIndex(index)
                return

    @staticmethod
    def select_combo_data(combo, value) -> bool:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return True
        return False

    @staticmethod
    def combo_contains_text(combo, text: str) -> bool:
        return any(combo.itemText(index) == text for index in range(combo.count()))

    def select_model_rank(self, rank: int) -> bool:
        return self.select_combo_data(self.widgets._model_combo, rank)

    def select_property(self, key: str) -> None:
        self.select_combo_data(self.widgets._prop_combo, key)

    def select_property_if_available(self, key: str) -> bool:
        row = self.widgets._prop_combo_rows.get(key)
        if row is None:
            return False
        item = self.widgets._prop_combo.model().item(row)
        if item is not None and not (item.flags() & ItemIsEnabled):
            return False
        self.widgets._prop_combo.setCurrentIndex(row)
        return True

    def set_metric_available(self, row: int, available: bool) -> None:
        item = self.widgets._prop_combo.model().item(row)
        if item is None:
            return
        flags = item.flags()
        item.setFlags(flags | ItemIsEnabled if available else flags & ~ItemIsEnabled)

    def metric_is_available(self, row: int) -> bool:
        item = self.widgets._prop_combo.model().item(row)
        return bool(item is not None and item.flags() & ItemIsEnabled)

    def set_metric_labels(self, labels: tuple[tuple[int, str], ...]) -> None:
        combo = self.widgets._prop_combo
        for row, label in labels:
            combo.setItemText(row, label)

    def set_plot_availability(
        self, availability: tuple[tuple[str, bool, str], ...]
    ) -> None:
        for key, enabled, tooltip in availability:
            action = self.widgets._plot_actions.get(key)
            if action is None:
                continue
            action.setEnabled(enabled)
            if hasattr(action, "setToolTip"):
                action.setToolTip(tooltip)
            if hasattr(action, "setStatusTip"):
                action.setStatusTip(tooltip)

    def set_confidence_text(self, text: str) -> None:
        self.widgets._conf_browser.setPlainText(text)

    def set_preview_text(self, text: str, details: str) -> None:
        self.widgets._preview_label.setText(text)
        self.widgets._preview_label.setToolTip(PREVIEW_DETAILS_TOOLTIP)
        self._preview_details_text = details
        self.widgets._preview_details_btn.setEnabled(bool(details))

    @staticmethod
    def _add_target_choice(combo, choice: TargetChoice) -> None:
        """Add one target and emphasize the active ensemble group."""
        combo.addItem(choice.name)
        if choice.kind != "ensemble_group":
            return
        item = combo.model().item(combo.count() - 1)
        if item is None:
            return
        font = item.font()
        font.setBold(True)
        font.setItalic(True)
        item.setFont(font)

    def apply_field_context(self, state: ContextViewState) -> None:
        self.widgets._ref_label.setText(state.reference_label)
        self.widgets._ref_label.setToolTip(state.reference_tooltip)
        self.widgets._ref_edit.setEnabled(state.reference_enabled)
        self.widgets._ref_edit.setToolTip(state.reference_tooltip)
        self.widgets._cutoff_label.setText(state.cutoff_label)
        self.widgets._cutoff_label.setToolTip(state.cutoff_tooltip)
        self.widgets._cutoff_spin.setEnabled(state.cutoff_enabled)
        self.widgets._cutoff_spin.setToolTip(state.cutoff_tooltip)

    def apply_context(self, state: ContextViewState) -> None:
        combo = self.widgets._model_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            for model_choice in state.model_choices:
                combo.addItem(model_choice.label, model_choice.rank)
            if state.selected_rank is not None:
                self.select_model_rank(state.selected_rank)
        finally:
            combo.blockSignals(False)
        combo = self.widgets._obj_combo
        combo.blockSignals(True)
        try:
            combo.clear()
            for target_choice in state.target_choices:
                self._add_target_choice(combo, target_choice)
            if state.selected_target:
                self.select_object(state.selected_target)
        finally:
            combo.blockSignals(False)
        self.set_metric_labels(state.metric_labels)
        for row, available in state.metric_availability:
            self.set_metric_available(row, available)
        self.set_plot_availability(state.plot_availability)
        self.apply_field_context(state)
        self.set_confidence_text(state.confidence_text)
        self.set_preview_text(state.preview_text, state.preview_details_text)
        self._ensemble_enabled = state.ensemble_enabled
        self._ensemble_tooltip = state.ensemble_tooltip
        self._render_ensemble_button()
        self._comparison_enabled = state.model_comparison_enabled
        self._comparison_tooltip = state.model_comparison_tooltip
        self._render_comparison_button()
        if state.statistics_text is not None:
            self.widgets._stats_browser.setPlainText(state.statistics_text)

    def apply_lifecycle(self, update: LifecycleUiUpdate) -> None:
        if update.recent_predictions is not None:
            self._set_recent_predictions(update.recent_predictions)
        if update.recent_afdb_accessions is not None:
            self._set_recent_afdb_accessions(update.recent_afdb_accessions)
        if update.display_path is not None:
            self.widgets._dir_edit.setText(update.display_path)
        if update.afdb_accession is not None:
            self.widgets._afdb_edit.setText(update.afdb_accession)
        if update.model_choices is not None:
            combo = self.widgets._model_combo
            combo.blockSignals(True)
            try:
                combo.clear()
                for choice in update.model_choices:
                    combo.addItem(choice.label, choice.rank)
                if update.selected_rank is not None:
                    self.select_model_rank(update.selected_rank)
            finally:
                combo.blockSignals(False)
        if update.target_choices is not None:
            combo = self.widgets._obj_combo
            combo.blockSignals(True)
            try:
                combo.clear()
                for choice in update.target_choices:
                    self._add_target_choice(combo, choice)
                if update.selected_target:
                    self.select_object(update.selected_target)
            finally:
                combo.blockSignals(False)

    def _set_recent_predictions(self, paths: tuple[str, ...]) -> None:
        combo = self.widgets._recent_combo
        edit_text = self.widgets._dir_edit.text()
        combo.blockSignals(True)
        try:
            combo.clear()
            for path in paths:
                combo.addItem(path, path)
                item = combo.model().item(combo.count() - 1)
                if item is not None and hasattr(item, "setToolTip"):
                    item.setToolTip(path)
            combo.setCurrentIndex(-1)
            combo.setEditText(edit_text)
        finally:
            combo.blockSignals(False)

    def _set_recent_afdb_accessions(self, accessions: tuple[str, ...]) -> None:
        combo = self.widgets._afdb_combo
        edit_text = self.widgets._afdb_edit.text()
        combo.blockSignals(True)
        try:
            combo.clear()
            for accession in accessions:
                combo.addItem(accession, accession)
            combo.setCurrentIndex(-1)
            combo.setEditText(edit_text)
        finally:
            combo.blockSignals(False)

    def set_busy(self, state: BusyViewState) -> None:
        self._busy = state.busy
        enabled = state.prediction_controls_enabled
        for widget in (
            self.widgets._dir_btn,
            self.widgets._file_btn,
            self.widgets._recent_combo,
            self.widgets._afdb_btn,
            self.widgets._afdb_combo,
            self.widgets._model_combo,
        ):
            widget.setEnabled(enabled)
        self._render_ensemble_button()
        self._render_comparison_button()

    def set_statistics_selection(self, state: StatisticsSelectionViewState) -> None:
        spin = self.widgets._stats_threshold_spin
        spin.blockSignals(True)
        try:
            span = max(state.maximum - state.minimum, 0.0)
            bound = max(abs(state.minimum), abs(state.maximum), 1.0) * 10.0
            spin.setRange(-bound, bound)
            spin.setSingleStep(max(span / 100.0, 0.001))
            spin.setValue(state.threshold)
            spin.setEnabled(state.enabled)
        finally:
            spin.blockSignals(False)
        self.widgets._stats_select_ge_btn.setEnabled(state.enabled)
        self.widgets._stats_select_le_btn.setEnabled(state.enabled)
        self.widgets._stats_selection_status.setText(state.status_text)

    def _render_ensemble_button(self) -> None:
        button = self.widgets._ensemble_btn
        button.setEnabled(self._ensemble_enabled and not self._busy)
        button.setToolTip(
            "Ensemble loading is unavailable while another task is running."
            if self._busy
            else self._ensemble_tooltip
        )

    def _render_comparison_button(self) -> None:
        button = self.widgets._compare_models_btn
        button.setEnabled(self._comparison_enabled and not self._busy)
        button.setToolTip(
            "Model comparison is unavailable while another task is running."
            if self._busy
            else self._comparison_tooltip
        )

    def close(self) -> None:
        """Release deterministic view resources (currently none)."""
