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
    assert LifecycleUiUpdate(selected_rank=2).selected_rank == 2
    state = ContextViewState(
        metric_availability=((3, False),),
        metric_labels=((3, "  pLDDT (structure B-factors)"),),
        plot_availability=(("matrix", False, "PAE is unavailable"),),
        preview_text="preview",
    )
    assert state.metric_labels[0][1].endswith("(structure B-factors)")
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


def test_dialog_uses_content_derived_resize_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    layout_source = (root / "gui_layout.py").read_text()
    dialog_source = (root / "gui.py").read_text()

    assert "QScrollArea" not in layout_source
    assert "VerticalScrollContent" not in layout_source
    assert "root = NaturalDialogLayout(dialog)" in layout_source
    assert "root.addGrowingWidget(model_group)" in layout_source
    assert "root.addGrowingWidget(stats_group)" in layout_source
    assert "growth, remainder = divmod(surplus" in layout_source
    assert layout_source.count("_fix_height_to_content(") == 4
    assert (
        layout_source.count("setSizePolicy(SizePolicyExpanding, SizePolicyFixed)") >= 4
    )
    assert "setMinimumSize(600, 890)" not in dialog_source
    assert "self.resize(self.minimumSizeHint())" in dialog_source


def test_dialog_button_and_text_panel_geometry_is_centralized() -> None:
    source = Path(__file__).resolve().parents[1].joinpath("gui_layout.py").read_text()

    for constant in (
        "PRIMARY_BUTTON_WIDTH = 120",
        "PRIMARY_BUTTON_HEIGHT = 25",
        "STATISTICS_BUTTON_WIDTH = 80",
        "STATISTICS_BUTTON_HEIGHT = 25",
        "SMALL_BUTTON_WIDTH = 28",
        "SMALL_BUTTON_HEIGHT = 25",
        "CONFIDENCE_SUMMARY_MIN_HEIGHT = 90",
        "STATISTICS_TEXT_MIN_HEIGHT = 120",
    ):
        assert constant in source

    assert "class FixedMainButton(QtWidgets.QPushButton):" in source
    assert source.count("FixedMainButton(") == 9  # Definition plus eight buttons.
    assert source.count("setFixedSize(SMALL_BUTTON_WIDTH, SMALL_BUTTON_HEIGHT)") == 3
    assert "button.setFixedSize(STATISTICS_BUTTON_WIDTH" in source
    assert "setMaximumHeight" not in source
    assert source.count("setVerticalScrollBarPolicy(ScrollBarAsNeeded)") == 2
    for label in (
        'FixedMainButton("Folder")',
        'FixedMainButton("File")',
        'FixedMainButton("Compare")',
        'FixedMainButton("Export CSV")',
        'FixedMainButton("Load Ensemble")',
    ):
        assert label in source


def test_native_browse_paths_remain_provisional_until_lifecycle_commit() -> None:
    tree = ast.parse(Path(__file__).resolve().parents[1].joinpath("gui.py").read_text())
    dialog = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FoldQCPluginDialog"
    )
    methods = {
        node.name: node
        for node in dialog.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_browse_directory", "_browse_file"}
    }

    assert methods.keys() == {"_browse_directory", "_browse_file"}
    for method in methods.values():
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        assert not any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "setText"
            for call in calls
        )
        assert any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "load_prediction"
            and len(call.args) == 1
            and ast.unparse(call.args[0]) == "path"
            for call in calls
        )


def test_startup_restores_history_without_loading_and_path_control_is_editable() -> (
    None
):
    root = Path(__file__).resolve().parents[1]
    gui_tree = ast.parse((root / "gui.py").read_text())
    dialog = next(
        node
        for node in gui_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FoldQCPluginDialog"
    )
    restore = next(
        node
        for node in dialog.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_restore_session_settings"
    )
    assert "load_prediction" not in ast.unparse(restore)
    lifecycle_call = next(
        node
        for node in ast.walk(restore)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LifecycleUiUpdate"
    )
    keywords = {keyword.arg: keyword.value for keyword in lifecycle_call.keywords}
    assert isinstance(keywords["display_path"], ast.Constant)
    assert keywords["display_path"].value == ""
    assert "recent_predictions=state.recent_predictions" in ast.unparse(restore)

    layout = (root / "gui_layout.py").read_text()
    assert "self._recent_combo = QtWidgets.QComboBox()" in layout
    assert "self._recent_combo.setEditable(True)" in layout
    assert "self._recent_combo.setInsertPolicy(ComboBoxNoInsert)" in layout
    assert "self._recent_combo.view().setTextElideMode(ElideLeft)" in layout
    assert "RecentPredictionItemDelegate(self._recent_combo)" in layout

    populate = next(
        node
        for node in dialog.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_populate_property_combo_for"
    )
    assert "combo.setCurrentIndex(rows[metrics.DEFAULT_METRIC_KEY])" in ast.unparse(
        populate
    )


def test_dialog_title_is_constant_and_coloring_does_not_mutate_it() -> None:
    root = Path(__file__).resolve().parents[1]
    gui_source = (root / "gui.py").read_text()
    coloring_source = (root / "gui_coloring.py").read_text()
    presenter_source = (root / "gui_presenter.py").read_text()
    protocol_source = (root / "presentation.py").read_text()

    assert 'DIALOG_TITLE = f"{APP_TITLE} — Structure Prediction Quality"' in gui_source
    assert "self.setWindowTitle(DIALOG_TITLE)" in gui_source
    assert "set_window_title" not in coloring_source
    assert "set_window_title" not in presenter_source
    assert "set_window_title" not in protocol_source


def test_model_comparison_uses_selected_model_as_default_button() -> None:
    source = (
        Path(__file__).resolve().parents[1].joinpath("gui_presenter.py").read_text()
    )

    assert 'use_button = QtWidgets.QPushButton("Use selected model")' in source
    assert "use_button.setDefault(True)" in source
    assert "close_button.setAutoDefault(False)" in source


def test_model_selection_panel_stacks_controls_beside_equal_width_summary() -> None:
    source = Path(__file__).resolve().parents[1].joinpath("gui_layout.py").read_text()

    assert 'model_group = QtWidgets.QGroupBox("Model selection")' in source
    assert "model_controls_layout = QtWidgets.QVBoxLayout(model_controls)" in source
    assert 'model_label = QtWidgets.QLabel("Model:")' in source
    assert (
        "_configure_flexible_combo(self._model_combo, MODEL_SELECTION_BOX_MIN_WIDTH)"
    ) in source
    assert "self._conf_browser.setMinimumWidth(MODEL_SELECTION_BOX_MIN_WIDTH)" in source
    assert "model_layout.addWidget(model_controls, 1)" in source
    assert "model_layout.addWidget(self._conf_browser, 1)" in source
    assert "model_container" not in source


def test_palette_and_scale_range_share_one_ordered_row() -> None:
    source = Path(__file__).resolve().parents[1].joinpath("gui_layout.py").read_text()
    row_start = source.index("palette_range_row = QtWidgets.QHBoxLayout()")
    row_end = source.index(
        'prop_form.addRow("Palette/Scale range:", palette_range_row)'
    )
    row = source[row_start:row_end]

    expected = (
        "palette_range_row.addWidget(self._palette_combo, 3)",
        "palette_range_row.addWidget(reverse_label)",
        "palette_range_row.addWidget(self._palette_reverse_chk)",
        "palette_range_row.addWidget(min_label)",
        "palette_range_row.addWidget(self._vmin_edit, 1)",
        "palette_range_row.addWidget(max_label)",
        "palette_range_row.addWidget(self._vmax_edit, 1)",
    )
    positions = [row.index(statement) for statement in expected]
    assert positions == sorted(positions)


def test_cutoff_uses_a_fixed_width_numeric_spinbox() -> None:
    root = Path(__file__).resolve().parents[1]
    layout_source = (root / "gui_layout.py").read_text()
    dialog_source = (root / "gui.py").read_text()

    assert "CUTOFF_SPINBOX_WIDTH = 100" in layout_source
    assert "self._cutoff_spin = QtWidgets.QDoubleSpinBox()" in layout_source
    assert "self._cutoff_spin.setFixedWidth(CUTOFF_SPINBOX_WIDTH)" in layout_source
    assert "self.widgets._cutoff_spin.valueChanged.connect" in dialog_source
    assert "cutoff_text=str(self.widgets._cutoff_spin.value())" in dialog_source


def test_plot_button_opens_unattached_menu_from_centered_button() -> None:
    root = Path(__file__).resolve().parents[1]
    layout_source = (root / "gui_layout.py").read_text()
    dialog_source = (root / "gui.py").read_text()

    assert "self._plot_btn.setMenu" not in layout_source
    assert (
        "self.widgets._plot_btn.clicked.connect(self._show_plot_menu)" in dialog_source
    )
    assert "self.widgets._plot_menu.popup(" in dialog_source
    assert "button.mapToGlobal(button.rect().bottomLeft())" in dialog_source


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
