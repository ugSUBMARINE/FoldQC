"""Qt widget rendering adapter for FoldQC's application services."""

from __future__ import annotations

from .compat import ItemIsEnabled
from .gui_services import ContextViewState


class QtDialogView:
    def __init__(self, dialog, widgets) -> None:
        self.dialog = dialog
        self.widgets = widgets

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

    def set_preview_text(self, text: str) -> None:
        self.widgets._preview_label.setText(text)

    def apply_field_context(self, state: ContextViewState) -> None:
        self.widgets._ref_label.setText(state.reference_label)
        self.widgets._ref_label.setToolTip(state.reference_tooltip)
        self.widgets._ref_edit.setEnabled(state.reference_enabled)
        self.widgets._ref_edit.setToolTip(state.reference_tooltip)
        self.widgets._cutoff_label.setText(state.cutoff_label)
        self.widgets._cutoff_label.setToolTip(state.cutoff_tooltip)
        self.widgets._cutoff_edit.setEnabled(state.cutoff_enabled)
        self.widgets._cutoff_edit.setToolTip(state.cutoff_tooltip)

    def apply_context(self, state: ContextViewState) -> None:
        for row, available in state.metric_availability:
            self.set_metric_available(row, available)
        self.set_plot_availability(state.plot_availability)
        self.apply_field_context(state)
        self.set_confidence_text(state.confidence_text)
        self.set_preview_text(state.preview_text)
        if state.statistics_text is not None:
            self.widgets._stats_browser.setPlainText(state.statistics_text)
