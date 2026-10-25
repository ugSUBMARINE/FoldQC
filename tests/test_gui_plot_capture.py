"""Exercise dialog action capture without importing the Qt widget tree."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from FoldQC import metrics
from FoldQC.analysis import (
    AnalysisRequest,
    ColorOptions,
    DeferredAnalysisAction,
    ExportOptions,
)


@pytest.fixture
def capture_dialog():
    # Compile the actual capture methods in isolation; the rest of the dialog
    # requires a live Qt environment. No capture behavior is duplicated here.
    source = Path(__file__).resolve().parents[1] / "gui.py"
    tree = ast.parse(source.read_text())
    dialog = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = [
        node
        for node in dialog.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_capture_action", "_optional_float"}
    ]
    dialog.bases = []
    dialog.body = methods
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ] + [dialog]
    namespace = {
        "metrics": metrics,
        "Path": Path,
        "ColorOptions": ColorOptions,
        "ExportOptions": ExportOptions,
        "DeferredAnalysisAction": DeferredAnalysisAction,
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["FoldQCPluginDialog"]()


class ColoringControl:
    def __init__(self, value):
        self.value = value
        self.reads = 0

    def text(self):
        self.reads += 1
        return self.value

    currentData = text
    isChecked = text


def coloring_widgets(minimum, maximum):
    return SimpleNamespace(
        _palette_combo=ColoringControl("white_green"),
        _palette_reverse_chk=ColoringControl(True),
        _vmin_edit=ColoringControl(minimum),
        _vmax_edit=ColoringControl(maximum),
    )


@pytest.mark.parametrize(
    "action", [spec.key for spec in metrics.PLOTS if spec.surface == "main"]
)
@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [("0.2", "0.6"), ("bad", "worse"), ("nan", "inf"), ("10", "1")],
)
def test_plot_capture_never_reads_coloring_controls(
    capture_dialog, action, minimum, maximum
):
    request = AnalysisRequest(action, "model_0", "pae_row_mean")
    capture_dialog._capture_request = lambda _action: request
    capture_dialog.widgets = coloring_widgets(minimum, maximum)

    captured = capture_dialog._capture_action(action)

    assert captured.request is request
    assert captured.options is None
    assert all(control.reads == 0 for control in vars(capture_dialog.widgets).values())


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        ("bad", "auto", "must be a number"),
        ("nan", "auto", "must be finite"),
        ("auto", "inf", "must be finite"),
        ("10", "1", "must not exceed"),
    ],
)
def test_color_capture_still_validates_bounds(
    capture_dialog, minimum, maximum, message
):
    capture_dialog._capture_request = lambda _action: AnalysisRequest(
        "color", "model_0", "plddt"
    )
    capture_dialog.widgets = coloring_widgets(minimum, maximum)
    with pytest.raises(ValueError, match=message):
        capture_dialog._capture_action("color")


def test_color_capture_keeps_palette_reversal_and_bounds(capture_dialog):
    capture_dialog._capture_request = lambda _action: AnalysisRequest(
        "color", "model_0", "plddt"
    )
    capture_dialog.widgets = coloring_widgets("0.2", "0.6")
    assert capture_dialog._capture_action("color").options == ColorOptions(
        "white_green", True, 0.2, 0.6
    )
