"""Typed widget registry and layout construction for FoldQC."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from . import metrics
from .compat import (
    ComboBoxNoInsert,
    ElideLeft,
    FormFieldGrowthPolicy,
    QAction,
    QtWidgets,
)
from .mol_viewer import get_selection_examples, get_viewer_name
from .palettes import iter_gui_palettes

VIEWER_NAME = get_viewer_name()
SELECTION_EXAMPLES = get_selection_examples()


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
    _cutoff_edit: QtWidgets.QLineEdit
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
    _preview_label: QtWidgets.QLabel
    _prop_combo: QtWidgets.QComboBox
    _prop_combo_rows: dict[str, int]
    _recent_combo: QtWidgets.QComboBox
    _ref_edit: QtWidgets.QLineEdit
    _ref_label: QtWidgets.QLabel
    _stats_browser: QtWidgets.QTextBrowser
    _vmax_edit: QtWidgets.QLineEdit
    _vmin_edit: QtWidgets.QLineEdit

    @classmethod
    def capture(cls, dialog) -> GuiWidgets:
        return cls(
            _apply_btn=dialog._apply_btn,
            _close_btn=dialog._close_btn,
            _compare_models_btn=dialog._compare_models_btn,
            _conf_browser=dialog._conf_browser,
            _cutoff_edit=dialog._cutoff_edit,
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
            _preview_label=dialog._preview_label,
            _prop_combo=dialog._prop_combo,
            _prop_combo_rows=dialog._prop_combo_rows,
            _recent_combo=dialog._recent_combo,
            _ref_edit=dialog._ref_edit,
            _ref_label=dialog._ref_label,
            _stats_browser=dialog._stats_browser,
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
    root = QtWidgets.QVBoxLayout(dialog)
    root.setSpacing(6)

    # --- Input row ---
    dir_group = QtWidgets.QGroupBox("Prediction output or structure")
    dir_layout = QtWidgets.QHBoxLayout(dir_group)
    self._recent_combo = QtWidgets.QComboBox()
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
    self._dir_btn = QtWidgets.QPushButton("Folder\u2026")
    self._dir_btn.setToolTip("Choose a prediction output folder to load.")
    self._file_btn = QtWidgets.QPushButton("File\u2026")
    self._file_btn.setToolTip(
        "Choose a prediction archive or single CIF/PDB file to load."
    )
    dialog._disable_default_button(self._dir_btn)
    dialog._disable_default_button(self._file_btn)
    self._recent_combo.setToolTip(
        "Type a prediction path or choose one of the last 10 successfully loaded predictions."
    )
    dir_layout.addWidget(self._recent_combo)
    dir_layout.addWidget(self._dir_btn)
    dir_layout.addWidget(self._file_btn)
    root.addWidget(dir_group)

    # --- Model selection ---
    form = QtWidgets.QFormLayout()
    form.setFieldGrowthPolicy(FormFieldGrowthPolicy.AllNonFixedFieldsGrow)

    self._model_combo = QtWidgets.QComboBox()
    self._model_combo.setToolTip(
        "Select the ranked model to load, summarize, and use for single-model coloring."
    )
    self._compare_models_btn = QtWidgets.QPushButton("Compare models…")
    dialog._disable_default_button(self._compare_models_btn)
    self._compare_models_btn.setEnabled(False)
    self._compare_models_btn.setToolTip(
        "Compare scalar confidence summaries for every discovered rank without "
        "loading all model structures."
    )
    model_row = QtWidgets.QHBoxLayout()
    model_row.addWidget(self._model_combo, 1)
    model_row.addWidget(self._compare_models_btn)
    form.addRow("Model:", model_row)

    root.addLayout(form)

    # --- Confidence summary text box ---
    conf_group = QtWidgets.QGroupBox("Confidence summary")
    conf_group.setToolTip(
        "Provider summary values loaded for the selected ranked model."
    )
    conf_layout = QtWidgets.QVBoxLayout(conf_group)
    self._conf_browser = QtWidgets.QTextBrowser()
    self._conf_browser.setMaximumHeight(150)
    self._conf_browser.setReadOnly(True)
    self._conf_browser.setToolTip(
        "Read-only confidence metadata for the selected model, such as "
        "ranking score, chain pTM/ipTM, and affinity values when available."
    )
    conf_layout.addWidget(self._conf_browser)
    root.addWidget(conf_group)

    # --- Property selection group ---
    prop_group = QtWidgets.QGroupBox("Analysis controls")
    prop_form = QtWidgets.QFormLayout(prop_group)
    prop_form.setFieldGrowthPolicy(FormFieldGrowthPolicy.AllNonFixedFieldsGrow)

    self._obj_combo = QtWidgets.QComboBox()
    self._obj_combo.setToolTip(
        f"{VIEWER_NAME} object, ensemble group, or ensemble member that will be colored or plotted."
    )
    self._obj_refresh_btn = QtWidgets.QPushButton("\u21ba")
    dialog._disable_default_button(self._obj_refresh_btn)
    self._obj_refresh_btn.setFixedWidth(28)
    self._obj_refresh_btn.setToolTip(
        f"Refresh the list of {VIEWER_NAME} objects and ensemble targets."
    )
    obj_row = QtWidgets.QHBoxLayout()
    obj_row.addWidget(self._obj_combo)
    obj_row.addWidget(self._obj_refresh_btn)
    prop_form.addRow(f"{VIEWER_NAME} target:", obj_row)

    self._prop_combo = QtWidgets.QComboBox()
    self._prop_combo_rows: dict[str, int] = {}
    dialog._populate_property_combo_for(self._prop_combo, self._prop_combo_rows)
    self._prop_combo.setToolTip(
        "Confidence metric to write into B-factors and display on the selected target."
    )
    prop_form.addRow("Color by:", self._prop_combo)

    self._ref_label = QtWidgets.QLabel("Reference:")
    self._ref_edit = QtWidgets.QLineEdit()
    self._ref_edit.setPlaceholderText(
        f"{VIEWER_NAME} selection, e.g. {SELECTION_EXAMPLES['general']}"
    )
    self._ref_edit.setToolTip(
        f"Optional {VIEWER_NAME} selection used by to-selection metrics, "
        "contact-filtered PAE/PDE, and binding-site fingerprints."
    )
    prop_form.addRow(self._ref_label, self._ref_edit)

    self._cutoff_edit = QtWidgets.QLineEdit("5.0")
    self._cutoff_edit.setFixedWidth(70)
    self._cutoff_edit.setToolTip(
        "Positive distance cutoff or PAE threshold in Å. Used for "
        "binding-site fingerprints, contact-filtered PAE/PDE, and PAE "
        "domain labels."
    )
    self._cutoff_label = QtWidgets.QLabel("Cutoff (Å):")
    prop_form.addRow(self._cutoff_label, self._cutoff_edit)

    self._preview_caption = QtWidgets.QLabel("Preview:")
    self._preview_label = QtWidgets.QLabel("")
    dialog._configure_preview_widgets(
        self._preview_caption,
        self._preview_label,
    )
    self._preview_label.setToolTip(
        "Compact summary of what the selected metric will do."
    )
    self._guide_btn = QtWidgets.QPushButton("?")
    dialog._disable_default_button(self._guide_btn)
    self._guide_btn.setFixedWidth(28)
    self._guide_btn.setToolTip("Open a quick guide to common FoldQC workflows.")
    preview_row = QtWidgets.QHBoxLayout()
    preview_row.addWidget(self._preview_label)
    preview_row.addWidget(self._guide_btn)
    prop_form.addRow(self._preview_caption, preview_row)

    self._palette_combo = QtWidgets.QComboBox()
    for spec in iter_gui_palettes():
        self._palette_combo.addItem(spec.label, spec.key)
    self._palette_combo.setToolTip(
        "Color palette used for continuous confidence metrics and plot heatmaps."
    )
    self._palette_reverse_chk = QtWidgets.QCheckBox("Reverse")
    self._palette_reverse_chk.setToolTip(
        "Reverse the selected continuous color palette."
    )
    palette_row = QtWidgets.QHBoxLayout()
    palette_row.addWidget(self._palette_combo)
    palette_row.addWidget(self._palette_reverse_chk)
    prop_form.addRow("Palette:", palette_row)

    range_row = QtWidgets.QHBoxLayout()
    self._vmin_edit = QtWidgets.QLineEdit()
    self._vmin_edit.setPlaceholderText("auto")
    self._vmin_edit.setFixedWidth(70)
    self._vmin_edit.setToolTip(
        "Optional lower bound for the color scale. Leave blank or use 'auto' to infer it."
    )
    self._vmax_edit = QtWidgets.QLineEdit()
    self._vmax_edit.setPlaceholderText("auto")
    self._vmax_edit.setFixedWidth(70)
    self._vmax_edit.setToolTip(
        "Optional upper bound for the color scale. Leave blank or use 'auto' to infer it."
    )
    min_label = QtWidgets.QLabel("Min:")
    min_label.setToolTip("Lower bound for the color scale.")
    range_row.addWidget(min_label)
    range_row.addWidget(self._vmin_edit)
    range_row.addSpacing(12)
    max_label = QtWidgets.QLabel("Max:")
    max_label.setToolTip("Upper bound for the color scale.")
    range_row.addWidget(max_label)
    range_row.addWidget(self._vmax_edit)
    range_row.addStretch()
    prop_form.addRow("Scale range:", range_row)

    root.addWidget(prop_group)

    # --- Statistics text box ---
    stats_group = QtWidgets.QGroupBox("Statistics")
    stats_group.setToolTip("Summary statistics for the most recently applied metric.")
    stats_layout = QtWidgets.QVBoxLayout(stats_group)
    self._stats_browser = QtWidgets.QTextBrowser()
    self._stats_browser.setMaximumHeight(235)
    self._stats_browser.setReadOnly(True)
    self._stats_browser.setPlainText("No property applied yet.")
    self._stats_browser.setToolTip(
        "Read-only metric statistics for the selected target after coloring "
        "or plot preparation."
    )
    stats_layout.addWidget(self._stats_browser)
    root.addWidget(stats_group)

    # --- Button row ---
    btn_layout = QtWidgets.QHBoxLayout()
    self._apply_btn = QtWidgets.QPushButton("Apply Coloring")
    self._plot_btn = QtWidgets.QPushButton("Plot")
    self._export_csv_btn = QtWidgets.QPushButton("Export CSV\u2026")
    self._plot_menu = QtWidgets.QMenu(self._plot_btn)
    self._plot_actions = build_plot_actions(self._plot_menu)
    self._plot_btn.setMenu(self._plot_menu)
    self._ensemble_btn = QtWidgets.QPushButton("Load Ensemble\u2026")
    self._close_btn = QtWidgets.QPushButton("Close")

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

    for btn in (
        self._apply_btn,
        self._plot_btn,
        self._export_csv_btn,
        self._ensemble_btn,
        self._close_btn,
    ):
        dialog._disable_default_button(btn)
        btn_layout.addWidget(btn)

    root.addLayout(btn_layout)
    return GuiWidgets.capture(self)
