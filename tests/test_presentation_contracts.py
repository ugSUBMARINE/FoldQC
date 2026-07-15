from __future__ import annotations

import ast
from pathlib import Path

import pytest
from FoldQC.gui_services import ContextViewState, LifecycleUiUpdate
from FoldQC.presentation import (
    ChoiceOption,
    ChoiceRequest,
    Notice,
    PreparedPlot,
    ProgressRequest,
    SelectionItem,
    SelectionRequest,
)


def test_presentation_requests_are_immutable_and_validate_stable_keys() -> None:
    notice = Notice(
        "provider_failure", "Could not load rank 2", affected_models=("rank 2",)
    )
    assert notice.affected_models == ("rank 2",)
    with pytest.raises(ValueError, match="unique"):
        ChoiceRequest(
            "duplicate",
            "FoldQC",
            "Choose",
            (ChoiceOption("same", "First"), ChoiceOption("same", "Second")),
        )
    with pytest.raises(ValueError, match="default"):
        SelectionRequest(
            "candidate",
            "Select prediction",
            "Choose",
            (SelectionItem("0", "first"),),
            default_key="missing",
        )
    with pytest.raises(ValueError, match="negative"):
        ProgressRequest("load-1", "Loading", "Please wait", delay_ms=-1)


def test_typed_presentation_and_view_results_preserve_payload_identity() -> None:
    figure = object()
    assert PreparedPlot(figure, "PAE").figure is figure
    assert LifecycleUiUpdate(selected_rank=2, save_session=True).selected_rank == 2
    state = ContextViewState(
        metric_availability=((3, False),),
        plot_availability=(("matrix", False, "PAE is unavailable"),),
        preview_text="preview",
    )
    assert state.plot_availability[0][0] == "matrix"


def test_service_modules_have_no_direct_qt_or_pymol_imports_or_broad_ignores() -> None:
    root = Path(__file__).resolve().parents[1]
    modules = (
        "prediction_lifecycle.py",
        "data_acquisition.py",
        "ensemble_lifecycle.py",
        "context_service.py",
        "plot_coordinator.py",
        "plot_preparation.py",
    )
    for name in modules:
        source = (root / name).read_text()
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any(
            "compat" in item or item.startswith("pymol") for item in imported
        )
        assert "type: ignore" not in source


def test_dialog_and_application_have_no_dynamic_workflow_bridge() -> None:
    root = Path(__file__).resolve().parents[1]
    application_source = (root / "gui_application.py").read_text()
    source = (root / "gui.py").read_text() + application_source
    assert "MethodType" not in source
    assert "_BoundWorkflowService" not in source
    dialog_tree = ast.parse((root / "gui.py").read_text())
    dialog = next(
        node
        for node in dialog_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FoldQCPluginDialog"
    )
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
        for node in dialog.body
    )
    application_tree = ast.parse(application_source)
    analysis = next(
        node
        for node in application_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AnalysisCoordinator"
    )
    assert [ast.unparse(base) for base in analysis.bases] == ["_ServiceRuntime"]
    assert all(
        name in application_source
        for name in (
            "MetricComputationService(self, dialog)",
            "ColoringCoordinator(self, dialog)",
            "PlotService(self, dialog)",
            "ExportCoordinator(self, dialog)",
        )
    )
    assert not (root / "gui_loading.py").exists()
    assert not (root / "gui_plots.py").exists()
