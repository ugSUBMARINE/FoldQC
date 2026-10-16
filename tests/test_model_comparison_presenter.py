"""Exercise model comparison interactions without a Qt event loop or PyMOL."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from FoldQC.presentation import (
    ModelComparisonColumn,
    ModelComparisonRequest,
    ModelComparisonRow,
)


class Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in self.callbacks:
            callback(*args)


class Item:
    def __init__(self, text):
        self.text = text
        self.metadata = {}

    def setData(self, role, value):
        self.metadata[role] = value

    def data(self, role):
        return self.metadata[role]


class ComparisonUi:
    """Minimal table state; dialog.exec runs a scripted user interaction."""

    def __init__(self):
        self.interact = lambda: None
        self.tables = []
        self.widgets = SimpleNamespace(
            QDialog=self.make_dialog,
            QVBoxLayout=lambda _parent: Mock(),
            QHBoxLayout=Mock,
            QLabel=lambda _text: Mock(),
            QTableWidget=self.make_table,
            QTableWidgetItem=Item,
            QPushButton=self.make_button,
        )

    def make_dialog(self, _parent):
        self.dialog = Mock()
        self.buttons = {}
        result = 0

        def accept():
            nonlocal result
            result = 1

        def reject():
            nonlocal result
            result = 0

        def execute():
            self.interact()
            return result

        self.dialog.accept.side_effect = accept
        self.dialog.reject.side_effect = reject
        self.dialog.exec.side_effect = execute
        return self.dialog

    def make_button(self, label):
        button = Mock()
        button.clicked = Signal()
        self.buttons[label] = button
        return button

    def make_table(self, rows, columns, _parent):
        self.cells = [[None] * columns for _ in range(rows)]
        self.current_row = -1
        table = Mock()
        header = Mock()
        header.sectionClicked = Signal()
        header.length.return_value = 600
        table.horizontalHeader.return_value = header
        table.doubleClicked = Signal()
        table.setItem.side_effect = self.set_item
        table.item.side_effect = self.item
        table.currentRow.side_effect = lambda: self.current_row
        table.setCurrentCell.side_effect = self.select
        self.table = table
        self.header = header
        self.tables.append(table)
        return table

    def set_item(self, row, column, item):
        self.cells[row][column] = item

    def item(self, row, column):
        return self.cells[row][column] if row >= 0 else None

    def select(self, row, _column):
        self.current_row = row

    def labels(self):
        return [row[0].text for row in self.cells]

    def selected_label(self):
        return self.item(self.current_row, 0).text


@pytest.fixture
def comparison_ui(monkeypatch):
    ui = ComparisonUi()
    compat = ModuleType("FoldQC.compat")
    for name, value in {
        "AbstractItemViewNoEditTriggers": 0,
        "AbstractItemViewSelectRows": 1,
        "AbstractItemViewSingleSelection": 1,
        "AscendingOrder": 0,
        "DescendingOrder": 1,
        "UserRole": 256,
        "WindowCloseButtonHint": 128,
        "MessageBoxButtonRole": SimpleNamespace(),
        "MessageBoxStandardButton": SimpleNamespace(),
        "QtCore": SimpleNamespace(),
        "QtWidgets": ui.widgets,
    }.items():
        setattr(compat, name, value)
    monkeypatch.setitem(sys.modules, "FoldQC.compat", compat)
    # Load an isolated presenter so fake Qt modules cannot leak into other tests.
    spec = importlib.util.spec_from_file_location(
        "FoldQC._comparison_presenter_test",
        Path(__file__).resolve().parents[1] / "gui_presenter.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ui.presenter = module.QtPresenter(None)
    return ui


def comparison_request(selected_rank=7):
    return ModelComparisonRequest(
        "Compare",
        "Test",
        (ModelComparisonColumn("score"), ModelComparisonColumn("has_clash")),
        (
            ModelComparisonRow(7, "sample 2", ("2.0000 Å", "True"), (2.0, True)),
            ModelComparisonRow(2, "sample 10", ("10.0000 Å", "False"), (10.0, False)),
            ModelComparisonRow(9, "sample 1", ("n/a", "n/a"), (None, None)),
        ),
        selected_rank=selected_rank,
    )


def test_header_sort_direction_column_changes_and_selection(comparison_ui):
    ui = comparison_ui

    def interact():
        assert ui.labels() == ["sample 10", "sample 2", "sample 1"]
        assert ui.selected_label() == "sample 2"
        ui.header.setSortIndicator.assert_called_with(0, 0)
        ui.header.setSectionsClickable.assert_called_with(True)
        ui.header.setSortIndicatorShown.assert_called_with(True)
        ui.table.setSortingEnabled.assert_called_with(False)

        ui.header.sectionClicked.emit(1)
        assert ui.labels() == ["sample 2", "sample 10", "sample 1"]
        assert ui.selected_label() == "sample 2"
        ui.header.setSortIndicator.assert_called_with(1, 0)

        ui.header.sectionClicked.emit(1)
        assert ui.labels() == ["sample 10", "sample 2", "sample 1"]
        assert ui.selected_label() == "sample 2"
        ui.header.setSortIndicator.assert_called_with(1, 1)

        # Changing columns resets direction even after descending on another one.
        ui.header.sectionClicked.emit(2)
        assert ui.labels() == ["sample 10", "sample 2", "sample 1"]
        ui.header.setSortIndicator.assert_called_with(2, 0)
        assert ui.selected_label() == "sample 2"

        ui.header.sectionClicked.emit(0)
        assert ui.labels() == ["sample 10", "sample 2", "sample 1"]
        ui.header.sectionClicked.emit(0)
        assert ui.labels() == ["sample 1", "sample 2", "sample 10"]
        assert ui.selected_label() == "sample 2"
        ui.header.setSortIndicator.assert_called_with(0, 1)
        ui.buttons["Close"].clicked.emit()

    ui.interact = interact
    assert ui.presenter.select_comparison_model(comparison_request()) is None


@pytest.mark.parametrize("acceptance", ["button", "double_click"])
def test_acceptance_uses_metadata_of_selected_sorted_row(comparison_ui, acceptance):
    ui = comparison_ui

    def interact():
        # Select another model before sorting: preserve the current user choice.
        ui.table.setCurrentCell(0, 0)
        assert ui.selected_label() == "sample 10"
        ui.header.sectionClicked.emit(1)
        assert ui.current_row == 1
        assert ui.selected_label() == "sample 10"
        # The sorted first row is rank 7; request.rows[0] is rank 9.
        ui.table.setCurrentCell(0, 1)
        if acceptance == "button":
            ui.buttons["Use selected model"].clicked.emit()
        else:
            ui.table.doubleClicked.emit(None)

    ui.interact = interact
    # Scramble request order to ensure row-position lookup cannot pass this test.
    request = comparison_request()
    request = ModelComparisonRequest(
        request.title,
        request.provider_label,
        request.columns,
        tuple(reversed(request.rows)),
        request.selected_rank,
    )
    assert ui.presenter.select_comparison_model(request) == 7


def test_reopening_resets_sort_and_restores_active_selection(comparison_ui):
    ui = comparison_ui
    request = comparison_request()

    def first_open():
        ui.header.sectionClicked.emit(1)
        assert ui.labels()[0] == "sample 2"
        ui.buttons["Close"].clicked.emit()

    ui.interact = first_open
    assert ui.presenter.select_comparison_model(request) is None

    def reopen():
        assert ui.labels() == ["sample 10", "sample 2", "sample 1"]
        assert ui.selected_label() == "sample 2"
        ui.header.setSortIndicator.assert_called_with(0, 0)
        ui.buttons["Use selected model"].clicked.emit()

    ui.interact = reopen
    assert ui.presenter.select_comparison_model(request) == 7
    assert len(ui.tables) == 2


def test_without_active_rank_first_original_model_is_selected(comparison_ui):
    ui = comparison_ui

    def interact():
        assert ui.selected_label() == "sample 10"
        ui.buttons["Use selected model"].clicked.emit()

    ui.interact = interact
    assert ui.presenter.select_comparison_model(comparison_request(None)) == 2
