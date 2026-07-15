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
        "analysis_coordinator.py",
        "prediction_lifecycle.py",
        "data_acquisition.py",
        "ensemble_lifecycle.py",
        "context_service.py",
        "gui_coloring.py",
        "gui_export.py",
        "gui_metrics.py",
        "gui_operations.py",
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
            "compat" in item or "mol_viewer" in item or item.startswith("pymol")
            for item in imported
        )
        assert "type: ignore" not in source


def test_no_workflow_fragments_or_service_locator_contract_remain() -> None:
    root = Path(__file__).resolve().parents[1]
    service_modules = (
        "analysis_coordinator.py",
        "context_service.py",
        "data_acquisition.py",
        "ensemble_lifecycle.py",
        "gui_coloring.py",
        "gui_export.py",
        "gui_metrics.py",
        "plot_coordinator.py",
        "prediction_lifecycle.py",
    )
    for name in service_modules:
        source = (root / name).read_text()
        tree = ast.parse(source)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        assert all(not node.name.endswith("Workflow") for node in classes)
        assert "GuiApplicationServices" not in source
        for node in classes:
            constructors = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            ]
            for constructor in constructors:
                parameters = {
                    item.arg
                    for item in (
                        *constructor.args.args,
                        *constructor.args.kwonlyargs,
                    )
                }
                assert not parameters & {"dialog", "widgets", "services"}

    contracts = (root / "gui_services.py").read_text()
    protocol_tree = ast.parse(contracts)
    view = next(
        node
        for node in protocol_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DialogViewPort"
    )
    assert "widgets" not in ast.unparse(view)


def test_dialog_and_application_have_no_dynamic_workflow_bridge() -> None:
    root = Path(__file__).resolve().parents[1]
    application_source = (root / "gui_application.py").read_text()
    source = (root / "gui.py").read_text() + application_source
    assert "MethodType" not in source
    assert "_BoundWorkflowService" not in source
    assert "_ServiceRuntime" not in source
    assert "Workflow" not in application_source
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
    assert [ast.unparse(base) for base in dialog.bases] == ["QtWidgets.QDialog"]
    application_tree = ast.parse(application_source)
    aggregate = next(
        node
        for node in application_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GuiApplicationServices"
    )
    constructor = next(
        node
        for node in aggregate.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    parameter_names = {argument.arg for argument in constructor.args.args}
    parameter_names.update(argument.arg for argument in constructor.args.kwonlyargs)
    assert "dialog" not in parameter_names
    assert "widgets" not in parameter_names
    assert all(
        name in application_source
        for name in (
            "MetricComputationService(",
            "ColoringCoordinator(",
            "PlotCoordinator(",
            "ExportCoordinator(",
            "DataAcquisitionService(",
            "PredictionLifecycleService(",
            "EnsembleLifecycleService(",
        )
    )
    assert not (root / "gui_loading.py").exists()
    assert not (root / "gui_plots.py").exists()


def test_plot_actions_are_parented_to_the_real_qt_menu() -> None:
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "gui_layout.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_plot_actions"
    )
    action_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "QAction"
    )
    assert len(action_call.args) == 2
    assert ast.unparse(action_call.args[1]) == "menu"
