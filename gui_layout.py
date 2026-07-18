"""Typed widget registry and layout construction for FoldQC."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from . import metrics
from .compat import (
    ComboBoxAdjustToMinimumContentsLengthWithIcon,
    ComboBoxNoInsert,
    ElideLeft,
    FormFieldGrowthPolicy,
    QAction,
    QtCore,
    QtWidgets,
    ScrollBarAsNeeded,
    SizePolicyExpanding,
    SizePolicyFixed,
)
from .mol_viewer import get_selection_examples, get_viewer_name
from .palettes import iter_gui_palettes

VIEWER_NAME = get_viewer_name()
SELECTION_EXAMPLES = get_selection_examples()

# Main-dialog geometry tuning. These values intentionally live together so
# visual checks can refine the natural minimum without changing layout logic.
PRIMARY_BUTTON_WIDTH = 120
PRIMARY_BUTTON_HEIGHT = 25
STATISTICS_BUTTON_WIDTH = 80
STATISTICS_BUTTON_HEIGHT = 25
SMALL_BUTTON_WIDTH = 28
SMALL_BUTTON_HEIGHT = 25

CONFIDENCE_SUMMARY_MIN_HEIGHT = 90
STATISTICS_TEXT_MIN_HEIGHT = 120

# Set the minimum widths of expanding widgets to unrealistically small values;
# the min. width of the window is then determined by the width of the button row
PREDICTION_PATH_MIN_WIDTH = 100  # 380
MODEL_SELECTION_BOX_MIN_WIDTH = 100  # 250
ANALYSIS_COMBO_MIN_WIDTH = 100  # 300
REFERENCE_EDIT_MIN_WIDTH = 100  # 400
PALETTE_COMBO_MIN_WIDTH = 100  # 180
NUMERIC_EDIT_MIN_WIDTH = 70
CUTOFF_SPINBOX_WIDTH = 100
STATISTICS_TEXT_MIN_WIDTH = 100  # 360
STATISTICS_SELECTION_PANEL_WIDTH = 200

BASE_LAYOUT_SPACING = 6


class NaturalDialogLayout(QtWidgets.QVBoxLayout):
    """Allocate surplus height equally above each growing section's minimum."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._growing_widgets = []

    def addGrowingWidget(self, widget) -> None:
        self._growing_widgets.append(widget)
        self.addWidget(widget)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        if not self._growing_widgets:
            return

        content = self.contentsRect()
        items = [self.itemAt(index) for index in range(self.count())]
        minimum_heights = [item.minimumSize().height() for item in items]
        spacing_height = self.spacing() * max(0, len(items) - 1)
        surplus = max(
            0,
            content.height() - sum(minimum_heights) - spacing_height,
        )
        growth, remainder = divmod(surplus, len(self._growing_widgets))
        growing_ids = {id(widget) for widget in self._growing_widgets}

        y = content.y()
        growing_index = 0
        for item, minimum_height in zip(items, minimum_heights, strict=True):
            widget = item.widget()
            height = minimum_height
            if widget is not None and id(widget) in growing_ids:
                height += growth + int(growing_index < remainder)
                growing_index += 1
            item.setGeometry(QtCore.QRect(content.x(), y, content.width(), height))
            y += height + self.spacing()


class FixedMainButton(QtWidgets.QPushButton):
    """Primary dialog action with uniform, non-resizable geometry."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setFixedSize(PRIMARY_BUTTON_WIDTH, PRIMARY_BUTTON_HEIGHT)


def _configure_flexible_combo(combo, minimum_width: int) -> None:
    """Keep dynamic combo contents from changing the dialog minimum."""
    combo.setSizeAdjustPolicy(ComboBoxAdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(0)
    combo.setMinimumWidth(minimum_width)
    combo.setSizePolicy(SizePolicyExpanding, SizePolicyFixed)


def _configure_flexible_line_edit(line_edit, minimum_width: int) -> None:
    """Give a single-line editor a tunable minimum and all extra row width."""
    line_edit.setMinimumWidth(minimum_width)
    line_edit.setSizePolicy(SizePolicyExpanding, SizePolicyFixed)


def _fix_height_to_content(widget) -> None:
    """Lock a section to the height implied by its completed child layout."""
    widget.setFixedHeight(widget.sizeHint().height())


class RecentPredictionItemDelegate(QtWidgets.QStyledItemDelegate):
    """Elide paths against the row's real paint width on Qt5 and Qt6."""

    def paint(self, painter, option, index) -> None:
        item_option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)
        available_width = max(0, item_option.rect.width() - 12)
        item_option.text = item_option.fontMetrics.elidedText(
            str(index.data() or ""),
            ElideLeft,
            available_width,
        )
        super().paint(painter, item_option, index)


@dataclass(frozen=True)
class GuiWidgets:
    """All widgets created by the main dialog layout."""

    _apply_btn: QtWidgets.QPushButton
    _close_btn: QtWidgets.QPushButton
    _compare_models_btn: QtWidgets.QPushButton
    _conf_browser: QtWidgets.QTextBrowser
    _cutoff_spin: QtWidgets.QDoubleSpinBox
    _cutoff_label: QtWidgets.QLabel
    _dir_btn: QtWidgets.QPushButton
    _dir_edit: QtWidgets.QLineEdit
    _ensemble_btn: QtWidgets.QPushButton
    _export_csv_btn: QtWidgets.QPushButton
    _file_btn: QtWidgets.QPushButton
    _guide_btn: QtWidgets.QPushButton
    _model_combo: QtWidgets.QComboBox
    _obj_combo: QtWidgets.QComboBox
    _obj_refresh_btn: QtWidgets.QPushButton
    _palette_combo: QtWidgets.QComboBox
    _palette_reverse_chk: QtWidgets.QCheckBox
    _plot_actions: dict[metrics.PlotType, QAction]
    _plot_btn: QtWidgets.QPushButton
    _plot_menu: QtWidgets.QMenu
    _preview_caption: QtWidgets.QLabel
    _preview_details_btn: QtWidgets.QPushButton
    _preview_label: QtWidgets.QLabel
    _prop_combo: QtWidgets.QComboBox
    _prop_combo_rows: dict[str, int]
    _recent_combo: QtWidgets.QComboBox
    _ref_edit: QtWidgets.QLineEdit
    _ref_label: QtWidgets.QLabel
    _stats_browser: QtWidgets.QTextBrowser
    _stats_select_ge_btn: QtWidgets.QPushButton
    _stats_select_le_btn: QtWidgets.QPushButton
    _stats_selection_status: QtWidgets.QLabel
    _stats_threshold_spin: QtWidgets.QDoubleSpinBox
    _vmax_edit: QtWidgets.QLineEdit
    _vmin_edit: QtWidgets.QLineEdit

    @classmethod
    def capture(cls, dialog) -> GuiWidgets:
        return cls(
            _apply_btn=dialog._apply_btn,
            _close_btn=dialog._close_btn,
            _compare_models_btn=dialog._compare_models_btn,
            _conf_browser=dialog._conf_browser,
            _cutoff_spin=dialog._cutoff_spin,
            _cutoff_label=dialog._cutoff_label,
            _dir_btn=dialog._dir_btn,
            _dir_edit=dialog._dir_edit,
            _ensemble_btn=dialog._ensemble_btn,
            _export_csv_btn=dialog._export_csv_btn,
            _file_btn=dialog._file_btn,
            _guide_btn=dialog._guide_btn,
            _model_combo=dialog._model_combo,
            _obj_combo=dialog._obj_combo,
            _obj_refresh_btn=dialog._obj_refresh_btn,
            _palette_combo=dialog._palette_combo,
            _palette_reverse_chk=dialog._palette_reverse_chk,
            _plot_actions=dialog._plot_actions,
            _plot_btn=dialog._plot_btn,
            _plot_menu=dialog._plot_menu,
            _preview_caption=dialog._preview_caption,
            _preview_details_btn=dialog._preview_details_btn,
            _preview_label=dialog._preview_label,
            _prop_combo=dialog._prop_combo,
            _prop_combo_rows=dialog._prop_combo_rows,
            _recent_combo=dialog._recent_combo,
            _ref_edit=dialog._ref_edit,
            _ref_label=dialog._ref_label,
            _stats_browser=dialog._stats_browser,
            _stats_select_ge_btn=dialog._stats_select_ge_btn,
            _stats_select_le_btn=dialog._stats_select_le_btn,
            _stats_selection_status=dialog._stats_selection_status,
            _stats_threshold_spin=dialog._stats_threshold_spin,
            _vmax_edit=dialog._vmax_edit,
            _vmin_edit=dialog._vmin_edit,
        )


def build_plot_actions(menu) -> dict[metrics.PlotType, QAction]:
    """Create plot actions with a real Qt owner on both Qt5 and Qt6."""
    actions = {}
    for spec in metrics.PLOTS:
        action = QAction(spec.label, menu)
        menu.addAction(action)
        actions[spec.key] = action
    return actions


def build_dialog_ui(dialog) -> GuiWidgets:
    """Build widgets into one registry without mutating the dialog namespace."""
    self = SimpleNamespace()
    root = NaturalDialogLayout(dialog)
    root.setSpacing(BASE_LAYOUT_SPACING)

    # --- Input row ---
    dir_group = QtWidgets.QGroupBox("Prediction output or structure")
    dir_group.setSizePolicy(SizePolicyExpanding, SizePolicyFixed)
    dir_layout = QtWidgets.QHBoxLayout(dir_group)
    self._recent_combo = QtWidgets.QComboBox()
    _configure_flexible_combo(self._recent_combo, PREDICTION_PATH_MIN_WIDTH)
    self._recent_combo.setEditable(True)
    self._recent_combo.setInsertPolicy(ComboBoxNoInsert)
    self._recent_combo.setMaxVisibleItems(10)
    self._recent_combo.setCurrentIndex(-1)
    self._recent_combo.view().setTextElideMode(ElideLeft)
    self._recent_combo.setItemDelegate(RecentPredictionItemDelegate(self._recent_combo))
    self._dir_edit = self._recent_combo.lineEdit()
    if self._dir_edit is None:
        raise RuntimeError("Editable prediction history requires a line editor.")
    self._dir_edit.setPlaceholderText("Output folder, archive, .cif, or .pdb file")
    self._dir_edit.setToolTip(
        "Path to a Boltz, AlphaFold 3, AlphaFold 3 Server, or Chai-1 "
        "Discovery, or Protenix output folder, prediction archive, or single "
        "CIF/PDB structure file. Press Return to load."
    )
    self._dir_btn = FixedMainButton("Folder")
    self._dir_btn.setToolTip("Choose a prediction output folder to load.")
    self._file_btn = FixedMainButton("File")
    self._file_btn.setToolTip(
        "Choose a prediction archive or single CIF/PDB file to load."
    )
    dialog._disable_default_button(self._dir_btn)
    dialog._disable_default_button(self._file_btn)
    self._recent_combo.setToolTip(
        "Type a prediction path or choose one of the last 10 successfully loaded predictions."
    )
    dir_layout.addWidget(self._recent_combo, 1)
    dir_layout.addWidget(self._dir_btn)
    dir_layout.addWidget(self._file_btn)
    _fix_height_to_content(dir_group)
    root.addWidget(dir_group)

    # --- Model selection and confidence summary ---
    model_group = QtWidgets.QGroupBox("Model selection")
    model_group.setSizePolicy(SizePolicyExpanding, SizePolicyExpanding)
    model_group.setToolTip(
        "Select a ranked model and review its provider confidence summary."
    )
    model_layout = QtWidgets.QHBoxLayout(model_group)
    model_controls = QtWidgets.QWidget()
    model_controls.setMinimumWidth(MODEL_SELECTION_BOX_MIN_WIDTH)
    model_controls.setSizePolicy(SizePolicyExpanding, SizePolicyExpanding)
    model_controls_layout = QtWidgets.QVBoxLayout(model_controls)
    model_controls_layout.setContentsMargins(0, 0, 0, 0)
    model_label = QtWidgets.QLabel("Model:")
    model_controls_layout.addWidget(model_label)
    self._model_combo = QtWidgets.QComboBox()
    _configure_flexible_combo(self._model_combo, MODEL_SELECTION_BOX_MIN_WIDTH)
    self._model_combo.setToolTip(
        "Select the ranked model to load, summarize, and use for single-model coloring."
    )
    model_controls_layout.addWidget(self._model_combo)
    self._compare_models_btn = FixedMainButton("Compare")
    dialog._disable_default_button(self._compare_models_btn)
    self._compare_models_btn.setEnabled(False)
    self._compare_models_btn.setToolTip(
        "Compare scalar confidence summaries for every discovered rank without "
        "loading all model structures."
    )
    model_controls_layout.addWidget(self._compare_models_btn)
    model_controls_layout.addStretch(1)

    self._conf_browser = QtWidgets.QTextBrowser()
    self._conf_browser.setMinimumWidth(MODEL_SELECTION_BOX_MIN_WIDTH)
    self._conf_browser.setMinimumHeight(CONFIDENCE_SUMMARY_MIN_HEIGHT)
    self._conf_browser.setSizePolicy(SizePolicyExpanding, SizePolicyExpanding)
    self._conf_browser.setHorizontalScrollBarPolicy(ScrollBarAsNeeded)
    self._conf_browser.setVerticalScrollBarPolicy(ScrollBarAsNeeded)
    self._conf_browser.setReadOnly(True)
    self._conf_browser.setToolTip(
        "Read-only confidence metadata for the selected model, such as "
        "ranking score, chain pTM/ipTM, and affinity values when available."
    )
    model_layout.addWidget(model_controls, 1)
    model_layout.addWidget(self._conf_browser, 1)
    root.addGrowingWidget(model_group)

    # --- Property selection group ---
    prop_group = QtWidgets.QGroupBox("Analysis controls")
    prop_group.setSizePolicy(SizePolicyExpanding, SizePolicyFixed)
    prop_form = QtWidgets.QFormLayout(prop_group)
    prop_form.setFieldGrowthPolicy(FormFieldGrowthPolicy.AllNonFixedFieldsGrow)

    self._obj_combo = QtWidgets.QComboBox()
    _configure_flexible_combo(self._obj_combo, ANALYSIS_COMBO_MIN_WIDTH)
    self._obj_combo.setToolTip(
        f"{VIEWER_NAME} object, ensemble group, or ensemble member that will be colored or plotted."
    )
    self._obj_refresh_btn = QtWidgets.QPushButton("\u21ba")
    dialog._disable_default_button(self._obj_refresh_btn)
    self._obj_refresh_btn.setFixedSize(SMALL_BUTTON_WIDTH, SMALL_BUTTON_HEIGHT)
    self._obj_refresh_btn.setToolTip(
        f"Refresh the list of {VIEWER_NAME} objects and ensemble targets."
    )
    obj_row = QtWidgets.QHBoxLayout()
    obj_row.addWidget(self._obj_combo, 1)
    obj_row.addWidget(self._obj_refresh_btn)
    prop_form.addRow(f"{VIEWER_NAME} target:", obj_row)

    self._prop_combo = QtWidgets.QComboBox()
    _configure_flexible_combo(self._prop_combo, ANALYSIS_COMBO_MIN_WIDTH)
    self._prop_combo_rows: dict[str, int] = {}
    dialog._populate_property_combo_for(self._prop_combo, self._prop_combo_rows)
    self._prop_combo.setToolTip(
        "Confidence metric to write into B-factors and display on the selected target."
    )
    self._guide_btn = QtWidgets.QPushButton("?")
    dialog._disable_default_button(self._guide_btn)
    self._guide_btn.setFixedSize(SMALL_BUTTON_WIDTH, SMALL_BUTTON_HEIGHT)
    self._guide_btn.setToolTip("Open a quick guide to common FoldQC workflows.")
    metric_row = QtWidgets.QHBoxLayout()
    metric_row.addWidget(self._prop_combo, 1)
    metric_row.addWidget(self._guide_btn)
    prop_form.addRow("Color by:", metric_row)

    self._ref_label = QtWidgets.QLabel("Reference:")
    self._ref_edit = QtWidgets.QLineEdit()
    _configure_flexible_line_edit(self._ref_edit, REFERENCE_EDIT_MIN_WIDTH)
    self._ref_edit.setPlaceholderText(
        f"{VIEWER_NAME} selection, e.g. {SELECTION_EXAMPLES['general']}"
    )
    self._ref_edit.setToolTip(
        f"Optional {VIEWER_NAME} selection used by to-selection metrics, "
        "contact-filtered PAE/PDE, and binding-site fingerprints."
    )
    prop_form.addRow(self._ref_label, self._ref_edit)

    self._cutoff_spin = QtWidgets.QDoubleSpinBox()
    self._cutoff_spin.setDecimals(1)
    self._cutoff_spin.setRange(0.1, 1_000.0)
    self._cutoff_spin.setSingleStep(0.1)
    self._cutoff_spin.setValue(5.0)
    self._cutoff_spin.setKeyboardTracking(False)
    self._cutoff_spin.setFixedWidth(CUTOFF_SPINBOX_WIDTH)
    self._cutoff_spin.setToolTip(
        "Positive distance cutoff or PAE threshold in Å. Used for "
        "binding-site fingerprints, contact-filtered PAE/PDE, and PAE "
        "domain labels."
    )
    self._cutoff_label = QtWidgets.QLabel("Cutoff (Å):")
    prop_form.addRow(self._cutoff_label, self._cutoff_spin)

    self._preview_caption = QtWidgets.QLabel("Preview:")
    self._preview_label = QtWidgets.QLabel("")
    dialog._configure_preview_widgets(
        self._preview_caption,
        self._preview_label,
    )
    self._preview_label.setToolTip(
        "Compact explanation of what the selected metric means."
    )
    self._preview_details_btn = QtWidgets.QPushButton("?")
    dialog._disable_default_button(self._preview_details_btn)
    self._preview_details_btn.setFixedSize(SMALL_BUTTON_WIDTH, SMALL_BUTTON_HEIGHT)
    self._preview_details_btn.setToolTip("Show the complete preview explanation.")
    preview_row = QtWidgets.QHBoxLayout()
    preview_row.addWidget(self._preview_label, 1)
    preview_row.addWidget(self._preview_details_btn)
    prop_form.addRow(self._preview_caption, preview_row)

    self._palette_combo = QtWidgets.QComboBox()
    _configure_flexible_combo(self._palette_combo, PALETTE_COMBO_MIN_WIDTH)
    for spec in iter_gui_palettes():
        self._palette_combo.addItem(spec.label, spec.key)
    self._palette_combo.setToolTip(
        "Color palette used for continuous confidence metrics and plot heatmaps."
    )
    self._palette_reverse_chk = QtWidgets.QCheckBox()
    self._palette_reverse_chk.setAccessibleName("Reverse")
    self._palette_reverse_chk.setToolTip(
        "Reverse the selected continuous color palette."
    )
    reverse_label = QtWidgets.QLabel("rev:")
    reverse_label.setToolTip(self._palette_reverse_chk.toolTip())
    palette_range_row = QtWidgets.QHBoxLayout()
    palette_range_row.addWidget(self._palette_combo, 3)
    palette_range_row.addWidget(reverse_label)
    palette_range_row.addWidget(self._palette_reverse_chk)
    self._vmin_edit = QtWidgets.QLineEdit()
    self._vmin_edit.setPlaceholderText("auto")
    _configure_flexible_line_edit(self._vmin_edit, NUMERIC_EDIT_MIN_WIDTH)
    self._vmin_edit.setToolTip(
        "Optional lower bound for the color scale. Leave blank or use 'auto' to infer it."
    )
    self._vmax_edit = QtWidgets.QLineEdit()
    self._vmax_edit.setPlaceholderText("auto")
    _configure_flexible_line_edit(self._vmax_edit, NUMERIC_EDIT_MIN_WIDTH)
    self._vmax_edit.setToolTip(
        "Optional upper bound for the color scale. Leave blank or use 'auto' to infer it."
    )
    min_label = QtWidgets.QLabel("Min:")
    min_label.setToolTip("Lower bound for the color scale.")
    palette_range_row.addWidget(min_label)
    palette_range_row.addWidget(self._vmin_edit, 1)
    max_label = QtWidgets.QLabel("Max:")
    max_label.setToolTip("Upper bound for the color scale.")
    palette_range_row.addWidget(max_label)
    palette_range_row.addWidget(self._vmax_edit, 1)
    prop_form.addRow("Palette/Scale range:", palette_range_row)

    _fix_height_to_content(prop_group)
    root.addWidget(prop_group)

    # --- Statistics text box ---
    stats_group = QtWidgets.QGroupBox("Statistics")
    stats_group.setSizePolicy(SizePolicyExpanding, SizePolicyExpanding)
    stats_group.setToolTip("Summary statistics for the most recently applied metric.")
    stats_layout = QtWidgets.QHBoxLayout(stats_group)
    self._stats_browser = QtWidgets.QTextBrowser()
    self._stats_browser.setMinimumHeight(STATISTICS_TEXT_MIN_HEIGHT)
    self._stats_browser.setMinimumWidth(STATISTICS_TEXT_MIN_WIDTH)
    self._stats_browser.setSizePolicy(SizePolicyExpanding, SizePolicyExpanding)
    self._stats_browser.setHorizontalScrollBarPolicy(ScrollBarAsNeeded)
    self._stats_browser.setVerticalScrollBarPolicy(ScrollBarAsNeeded)
    self._stats_browser.setReadOnly(True)
    self._stats_browser.setPlainText("No property applied yet.")
    self._stats_browser.setToolTip(
        "Read-only metric statistics for the selected target after coloring "
        "or plot preparation."
    )
    stats_layout.addWidget(self._stats_browser, 1)

    selection_panel = QtWidgets.QWidget()
    selection_panel.setFixedWidth(STATISTICS_SELECTION_PANEL_WIDTH)
    selection_layout = QtWidgets.QVBoxLayout(selection_panel)
    selection_layout.setContentsMargins(6, 0, 0, 0)
    threshold_label = QtWidgets.QLabel("Threshold:")
    selection_layout.addWidget(threshold_label)
    self._stats_threshold_spin = QtWidgets.QDoubleSpinBox()
    self._stats_threshold_spin.setDecimals(3)
    self._stats_threshold_spin.setRange(-1_000_000.0, 1_000_000.0)
    self._stats_threshold_spin.setKeyboardTracking(False)
    self._stats_threshold_spin.setEnabled(False)
    self._stats_threshold_spin.setToolTip(
        "Threshold applied to FoldQC's stored metric values, not current "
        f"{VIEWER_NAME} B-factors."
    )
    selection_layout.addWidget(self._stats_threshold_spin)
    selection_buttons = QtWidgets.QHBoxLayout()
    self._stats_select_ge_btn = QtWidgets.QPushButton("Select ≥")
    self._stats_select_le_btn = QtWidgets.QPushButton("Select ≤")
    for button in (self._stats_select_ge_btn, self._stats_select_le_btn):
        dialog._disable_default_button(button)
        button.setFixedSize(STATISTICS_BUTTON_WIDTH, STATISTICS_BUTTON_HEIGHT)
        button.setEnabled(False)
        selection_buttons.addWidget(button)
    self._stats_select_ge_btn.setToolTip(
        "Create or replace a named selection for finite metric values at or above "
        "the threshold."
    )
    self._stats_select_le_btn.setToolTip(
        "Create or replace a named selection for finite metric values at or below "
        "the threshold."
    )
    selection_layout.addLayout(selection_buttons)
    self._stats_selection_status = QtWidgets.QLabel("Apply a metric coloring first.")
    self._stats_selection_status.setWordWrap(True)
    self._stats_selection_status.setToolTip(
        "The named selection remains available in the PyMOL object manager."
    )
    selection_layout.addWidget(self._stats_selection_status)
    selection_layout.addStretch()
    stats_layout.addWidget(selection_panel)
    root.addGrowingWidget(stats_group)

    # --- Button row ---
    button_container = QtWidgets.QWidget()
    button_container.setSizePolicy(SizePolicyExpanding, SizePolicyFixed)
    btn_layout = QtWidgets.QHBoxLayout(button_container)
    btn_layout.setContentsMargins(0, 0, 0, 0)
    btn_layout.setSpacing(0)
    self._apply_btn = FixedMainButton("Apply Coloring")
    self._plot_btn = FixedMainButton("Plot")
    self._export_csv_btn = FixedMainButton("Export CSV")
    self._plot_menu = QtWidgets.QMenu(self._plot_btn)
    self._plot_actions = build_plot_actions(self._plot_menu)
    self._ensemble_btn = FixedMainButton("Load Ensemble")
    self._close_btn = FixedMainButton("Close")

    self._apply_btn.setToolTip(
        f"Apply the selected coloring metric to the {VIEWER_NAME} target."
    )
    self._plot_btn.setToolTip(
        "Open an available plot for the current target and inputs."
    )
    self._export_csv_btn.setToolTip(
        "Export token-level values for the current metric and target."
    )
    self._ensemble_btn.setToolTip(
        "Load all ranked models as an ensemble and compute ensemble-level metrics."
    )
    self._close_btn.setToolTip("Close the FoldQC dialog.")

    main_buttons = (
        self._apply_btn,
        self._plot_btn,
        self._export_csv_btn,
        self._ensemble_btn,
        self._close_btn,
    )
    for index, btn in enumerate(main_buttons):
        dialog._disable_default_button(btn)
        btn_layout.addWidget(btn)
        if index < len(main_buttons) - 1:
            btn_layout.addSpacing(BASE_LAYOUT_SPACING)
            btn_layout.addStretch(1)

    _fix_height_to_content(button_container)
    root.addWidget(button_container)
    return GuiWidgets.capture(self)
