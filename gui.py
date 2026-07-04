"""
GUI
===
Main Qt dialog for the FoldQC PyMOL plugin.

All Qt imports go through :mod:`compat` to handle Qt5/Qt6 differences.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import (
    compute,
    ensemble,
    export,
    gui_rules,
    metrics,
    plot_data,
    reports,
    session,
)
from .compat import (
    AlignLeft,
    AlignVCenter,
    FormFieldGrowthPolicy,
    ItemIsEnabled,
    QAction,
    QSettings,
    QtWidgets,
)
from .palettes import iter_gui_palettes

APP_TITLE = "FoldQC"
PREDICTION_FILE_FILTER = (
    "Prediction files (*.cif *.pdb *.zip *.tar *.tar.gz *.tgz);;All files (*)"
)


@dataclass
class _PlotTarget:
    """Resolved plot target from the PyMOL target combo."""

    kind: str
    label: str
    obj_name: str
    data: object | None
    token_map: object
    members: list | None = None


def _confidence_has_chain_iptm_metric_data(confidence) -> bool:
    """Return true when confidence metadata can drive the Chain ipTM metric."""
    if not isinstance(confidence, dict):
        return False
    for key in ("chains_iptm", "chain_iptm", "chains_ptm"):
        if _score_table_has_values(confidence.get(key)):
            return True
    for key in ("pair_chains_iptm", "chain_pair_iptm"):
        if _pair_score_table_has_values(confidence.get(key)):
            return True
    return False


def _score_table_has_values(value) -> bool:
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return False


def _pair_score_table_has_values(value) -> bool:
    if isinstance(value, dict):
        return any(_score_table_has_values(row) for row in value.values())
    if isinstance(value, list):
        return any(_score_table_has_values(row) for row in value)
    return False


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class FoldQCPluginDialog(QtWidgets.QDialog):
    """Main plugin window."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(APP_TITLE)
        self.setMinimumWidth(480)

        # State
        self._pred_files = None  # loader.PredictionFiles | None
        self._pred_data = None  # loader.PredictionData | None
        self._token_map = None  # list[token_map.TokenInfo] | None
        self._ensemble_members = None  # list[ensemble.EnsembleMember] | None
        self._ensemble_group_name = None  # str | None
        self._ensemble_aligned = False
        self._ensemble_rmsd = None  # np.ndarray | None
        self._ensemble_plddt_mean = None  # np.ndarray | None
        self._ensemble_plddt_std = None  # np.ndarray | None
        self._plot_windows = []  # Qt plot dialogs kept alive while visible
        self._guide_dialog = None  # Lightweight first-run guide dialog
        self._loading_prediction = False
        self._restoring_settings = False
        self._pending_session_restore = session.PendingSessionRestore()

        self._build_ui()
        self._connect_signals()
        self._restore_session_settings()
        self._on_property_changed()  # set initial reference-field visibility

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(6)

        # --- Input row ---
        dir_group = QtWidgets.QGroupBox("Prediction output or structure")
        dir_layout = QtWidgets.QHBoxLayout(dir_group)
        self._dir_edit = QtWidgets.QLineEdit()
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
        self._disable_default_button(self._dir_btn)
        self._disable_default_button(self._file_btn)
        dir_layout.addWidget(self._dir_edit)
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
        form.addRow("Model:", self._model_combo)

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
            "PyMOL object, ensemble group, or ensemble member that will be colored or plotted."
        )
        self._obj_refresh_btn = QtWidgets.QPushButton("\u21ba")
        self._disable_default_button(self._obj_refresh_btn)
        self._obj_refresh_btn.setFixedWidth(28)
        self._obj_refresh_btn.setToolTip(
            "Refresh the list of PyMOL objects and ensemble targets."
        )
        obj_row = QtWidgets.QHBoxLayout()
        obj_row.addWidget(self._obj_combo)
        obj_row.addWidget(self._obj_refresh_btn)
        prop_form.addRow("PyMOL target:", obj_row)

        self._prop_combo = QtWidgets.QComboBox()
        self._prop_combo_rows: dict[str, int] = {}
        self._populate_property_combo()
        self._prop_combo.setToolTip(
            "Confidence metric to write into B-factors and display on the selected target."
        )
        prop_form.addRow("Color by:", self._prop_combo)

        self._ref_label = QtWidgets.QLabel("Reference:")
        self._ref_edit = QtWidgets.QLineEdit()
        self._ref_edit.setPlaceholderText(
            'PyMOL selection, e.g. "chain C" or "resname LIG"'
        )
        self._ref_edit.setToolTip(
            "Optional PyMOL selection used by to-selection metrics, "
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
        self._configure_preview_widgets(
            self._preview_caption,
            self._preview_label,
        )
        self._preview_label.setToolTip(
            "Compact summary of what the selected metric will do."
        )
        self._guide_btn = QtWidgets.QPushButton("?")
        self._disable_default_button(self._guide_btn)
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
        stats_group.setToolTip(
            "Summary statistics for the most recently applied metric."
        )
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
        self._plot_actions: dict[str, object] = {}
        for label, key in metrics.PLOT_TYPES:
            action = QAction(label, self)
            action.triggered.connect(
                lambda _checked=False, plot_type=key: self._show_selected_plot(
                    plot_type
                )
            )
            self._plot_menu.addAction(action)
            self._plot_actions[key] = action
        self._plot_btn.setMenu(self._plot_menu)
        self._ensemble_btn = QtWidgets.QPushButton("Load Ensemble\u2026")
        self._close_btn = QtWidgets.QPushButton("Close")

        self._apply_btn.setToolTip(
            "Apply the selected coloring metric to the PyMOL target."
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
            self._disable_default_button(btn)
            btn_layout.addWidget(btn)

        root.addLayout(btn_layout)

    def _disable_default_button(self, btn) -> None:
        """Prevent Return/Enter from activating unrelated dialog buttons."""
        btn.setAutoDefault(False)
        btn.setDefault(False)

    def _configure_preview_widgets(self, caption, preview) -> None:
        """Apply the shared alignment and reserved height for preview text."""
        minimum_height = preview.fontMetrics().lineSpacing() * 5 + 8
        alignment = AlignLeft | AlignVCenter
        preview.setWordWrap(True)
        preview.setMinimumHeight(minimum_height)
        preview.setAlignment(alignment)
        caption.setMinimumHeight(minimum_height)
        caption.setAlignment(alignment)

    def _populate_property_combo(self) -> None:
        """Populate Color by with disabled group headers and metric rows."""
        self._prop_combo_rows = {}
        current_group = None
        for prop in metrics.PROPERTIES:
            group = prop.get("group", "Other")
            if group != current_group:
                self._prop_combo.addItem(str(group), None)
                self._disable_combo_row(self._prop_combo, self._prop_combo.count() - 1)
                current_group = group
            self._prop_combo.addItem(metrics.property_combo_label(prop), prop["key"])
            self._prop_combo_rows[prop["key"]] = self._prop_combo.count() - 1

    def _disable_combo_row(self, combo, row: int) -> None:
        """Disable one combo row when the backing model item is available."""
        item = combo.model().item(row)
        if item is not None:
            item.setFlags(item.flags() & ~ItemIsEnabled)

    def _connect_signals(self) -> None:
        self._dir_btn.clicked.connect(self._browse_directory)
        self._file_btn.clicked.connect(self._browse_file)
        self._dir_edit.returnPressed.connect(self._load_prediction_dir)
        self._dir_edit.textChanged.connect(self._save_session_settings)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._model_combo.currentIndexChanged.connect(self._save_session_settings)
        self._obj_refresh_btn.clicked.connect(self._refresh_objects)
        self._obj_combo.currentIndexChanged.connect(self._refresh_contextual_ui)
        self._obj_combo.currentIndexChanged.connect(self._save_session_settings)
        self._prop_combo.currentIndexChanged.connect(self._on_property_changed)
        self._prop_combo.currentIndexChanged.connect(self._save_session_settings)
        self._ref_edit.textChanged.connect(self._refresh_contextual_ui)
        self._ref_edit.textChanged.connect(self._save_session_settings)
        self._cutoff_edit.textChanged.connect(self._refresh_contextual_ui)
        self._cutoff_edit.textChanged.connect(self._save_session_settings)
        self._palette_combo.currentIndexChanged.connect(self._save_session_settings)
        self._palette_reverse_chk.stateChanged.connect(self._save_session_settings)
        self._vmin_edit.textChanged.connect(self._save_session_settings)
        self._vmax_edit.textChanged.connect(self._save_session_settings)
        self._guide_btn.clicked.connect(self._show_guide)
        self._apply_btn.clicked.connect(self._apply_coloring)
        self._export_csv_btn.clicked.connect(self._export_csv)
        self._ensemble_btn.clicked.connect(self._show_ensemble)
        self._close_btn.clicked.connect(self.close)

    def _guide_text(self) -> str:
        """Return compact non-computing guidance for common FoldQC workflows."""
        return (
            "FoldQC Quick Guide\n\n"
            "1. Overall local confidence\n"
            "Use: pLDDT - quality classes\n"
            "Plot: Distribution or Line\n\n"
            "2. Ligand / binding-site confidence\n"
            'Reference examples: "resname LIG", "organic"\n'
            "Use: PAE - contact-filtered to selection or "
            "PDE - contact-filtered to selection\n"
            "Plot: Binding-site fingerprint\n\n"
            "3. Chain or interface placement\n"
            'Reference example: "chain B"\n'
            "Use: PAE - symmetric mean to selection\n"
            "Alternative: Chain ipTM with Plot > Matrix\n\n"
            "4. Domain placement\n"
            "Use: PAE domain labels or PAE matrix\n"
            "Note: domain labels are categorical/experimental.\n\n"
            "5. Ensemble variability\n"
            "Use: Load Ensemble, then Ensemble RMSD or Ensemble pLDDT std."
        )

    def _show_guide(self) -> None:
        """Open or raise the lightweight FoldQC quick-guide dialog."""
        dialog = getattr(self, "_guide_dialog", None)
        if dialog is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("FoldQC Quick Guide")
            dialog.setModal(False)
            dialog.setMinimumWidth(420)

            layout = QtWidgets.QVBoxLayout(dialog)
            guide = QtWidgets.QTextBrowser()
            guide.setReadOnly(True)
            guide.setOpenExternalLinks(False)
            guide.setPlainText(self._guide_text())
            guide.setMinimumHeight(320)
            layout.addWidget(guide)

            close_btn = QtWidgets.QPushButton("Close")
            self._disable_default_button(close_btn)
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)

            dialog._foldqc_guide_text = self._guide_text()
            self._guide_dialog = dialog

        dialog.show()
        if hasattr(dialog, "raise_"):
            dialog.raise_()
        if hasattr(dialog, "activateWindow"):
            dialog.activateWindow()

    # -----------------------------------------------------------------------
    # Session settings
    # -----------------------------------------------------------------------

    def _settings(self):
        """Return the persistent settings store for FoldQC GUI state."""
        return QSettings(session.SETTINGS_ORGANIZATION, session.SETTINGS_APPLICATION)

    def _save_session_settings(self, *_args) -> None:
        """Persist lightweight GUI state for the next FoldQC dialog."""
        if getattr(self, "_restoring_settings", False):
            return
        if getattr(self, "_loading_prediction", False):
            return

        try:
            settings = self._settings()
            rank = self._model_combo.currentData()
            metric_key = self._prop_combo.currentData()
            palette_key, reverse_palette = self._selected_palette()
            geometry = None
            if hasattr(self, "saveGeometry"):
                geometry = self.saveGeometry()
            state = session.SessionState(
                path=self._dir_edit.text(),
                model_rank=rank,
                metric_key="" if metric_key is None else metric_key,
                target_name=self._obj_combo.currentText(),
                reference_text=self._ref_edit.text(),
                cutoff_text=self._cutoff_edit.text(),
                palette_key=palette_key,
                palette_reversed=reverse_palette,
                scale_min=self._vmin_edit.text(),
                scale_max=self._vmax_edit.text(),
                geometry=geometry,
            )
            session.write_session_state(settings, state)
        except Exception:
            pass

    def _restore_session_settings(self) -> None:
        """Restore saved lightweight GUI state and reload a valid last path."""
        self._restoring_settings = True
        try:
            settings = self._settings()
            state = session.read_session_state(settings)
            self._pending_session_restore = session.PendingSessionRestore(
                model_rank=state.model_rank,
                metric_key=state.metric_key or None,
                target_name=state.target_name or None,
            )

            self._dir_edit.setText(state.path)
            self._ref_edit.setText(state.reference_text)
            if state.cutoff_text:
                self._cutoff_edit.setText(state.cutoff_text)
            self._vmin_edit.setText(state.scale_min)
            self._vmax_edit.setText(state.scale_max)
            self._palette_reverse_chk.setChecked(state.palette_reversed)
            if state.palette_key:
                self._select_combo_data(self._palette_combo, state.palette_key)
            if state.metric_key:
                self._select_property_if_available(state.metric_key)

            if state.geometry and hasattr(self, "restoreGeometry"):
                try:
                    self.restoreGeometry(state.geometry)
                except Exception:
                    pass

            if state.path and Path(state.path).exists():
                self._load_prediction_dir()
        finally:
            self._restoring_settings = False

    def closeEvent(self, event) -> None:
        """Persist session state when the dialog closes."""
        self._save_session_settings()
        try:
            super().closeEvent(event)
        except AttributeError:
            pass

    # -----------------------------------------------------------------------
    # Slots
    # -----------------------------------------------------------------------

    def _browse_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select prediction output folder",
            self._dir_edit.text() or str(Path.home()),
        )
        if path:
            self._dir_edit.setText(path)
            self._load_prediction_dir()
        else:
            self._raise_after_native_dialog()

    def _browse_file(self) -> None:
        """Select a single CIF/PDB structure file or prediction archive."""
        result = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select predicted structure file or prediction archive",
            self._dir_edit.text() or str(Path.home()),
            PREDICTION_FILE_FILTER,
        )
        path = result[0] if isinstance(result, tuple) else result
        if path:
            self._dir_edit.setText(path)
            self._load_prediction_dir()
        else:
            self._raise_after_native_dialog()

    def _raise_after_native_dialog(self) -> None:
        """Bring this dialog back after a native file dialog returns focus."""
        self.raise_()
        self.activateWindow()

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
        """Return the canonical PyMOL object name for one model rank."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        try:
            return self._pred_files.model(rank).object_name
        except Exception:
            return f"{self._pred_files.name}_model_{rank}"

    def _ensure_model_object(self, rank: int, *, paint: bool = True) -> str | None:
        """Load or enable the PyMOL object for *rank*, then select it."""
        if self._pred_files is None or not self._pred_files.structure_files:
            return None
        try:
            from pymol import cmd

            current_objects = set(cmd.get_names("objects") or [])
        except Exception:
            return None

        obj_name = self._expected_object_name(rank)
        did_load = False
        if obj_name in current_objects:
            if not self._is_object_enabled(cmd, obj_name):
                try:
                    cmd.enable(obj_name)
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(
                        self, APP_TITLE, f"Could not show {obj_name}:\n{exc}"
                    )
                    return None
        else:
            path = self._pred_files.structure_path(rank)
            try:
                cmd.load(str(path), obj_name, quiet=1, zoom=1)
                did_load = True
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, f"Could not auto-load {path.name}:\n{exc}"
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

    def _is_object_enabled(self, cmd, obj_name: str) -> bool:
        """Return whether *obj_name* is currently enabled in PyMOL."""
        try:
            enabled = set(cmd.get_names("objects", enabled_only=1) or [])
        except TypeError:
            enabled = set(cmd.get_names("objects", 1) or [])
        except Exception:
            return True
        return obj_name in enabled

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
        """Re-populate the PyMOL target dropdown."""
        try:
            from pymol import cmd

            names = cmd.get_names("objects") or []
            if self._ensemble_group_name and self._ensemble_group_name not in names:
                try:
                    all_names = set(cmd.get_names("all") or [])
                except Exception:
                    all_names = set()
                if not all_names or self._ensemble_group_name in all_names:
                    names = [self._ensemble_group_name] + list(names)
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

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _get_obj_name(self) -> str | None:
        name = self._obj_combo.currentText().strip()
        return name if name else None

    def _selected_palette(self) -> tuple[str, bool]:
        """Return the selected palette key and reverse checkbox state."""
        combo = self._palette_combo
        try:
            key = combo.currentData()
        except AttributeError:
            key = None
        if key is None:
            key = combo.currentText()
        reverse_chk = getattr(self, "_palette_reverse_chk", None)
        reverse = bool(reverse_chk.isChecked()) if reverse_chk is not None else False
        return str(key), reverse

    def _selected_ensemble_member(self, obj_name: str):
        """Return the active ensemble member matching *obj_name*, if any."""
        for member in getattr(self, "_ensemble_members", None) or []:
            if member.obj_name == obj_name:
                return member
        return None

    def _get_vmin_vmax(self) -> tuple[float | None, float | None]:
        def _parse(text: str) -> float | None:
            t = text.strip()
            if not t or t.lower() == "auto":
                return None
            try:
                return float(t)
            except ValueError:
                return None

        return _parse(self._vmin_edit.text()), _parse(self._vmax_edit.text())

    def _get_cutoff_threshold(self) -> float | None:
        """Return the user-entered positive cutoff/threshold in Å."""
        edit = getattr(self, "_cutoff_edit", None)
        text = "5.0" if edit is None else edit.text().strip()
        if not text:
            text = "5.0"
        try:
            cutoff = float(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "Cutoff / threshold must be a positive number in Å.",
            )
            return None
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "Cutoff / threshold must be greater than 0 Å.",
            )
            return None
        return cutoff

    def _get_contact_cutoff(self) -> float | None:
        """Compatibility wrapper for contact-based callers."""
        return self._get_cutoff_threshold()

    def _apply_coloring(self) -> None:
        """Compute the selected property and paint the structure."""
        obj_name = self._get_obj_name()
        if obj_name is None:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "No PyMOL target selected.")
            return

        if self._ensemble_members and obj_name == self._ensemble_group_name:
            self._apply_ensemble_coloring(self._ensemble_members)
            return

        member = self._selected_ensemble_member(obj_name)
        if member is not None:
            self._apply_ensemble_coloring([member])
            return

        if self._pred_data is None:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "No prediction data loaded.")
            return

        key = self._prop_combo.currentData()
        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        if metrics.PROPERTY_BY_KEY.get(key, {}).get("ensemble_level", False):
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "This property requires an active ensemble.\n"
                "Use the Ensemble… button, then choose the ensemble group "
                "or one of its model objects as the PyMOL target.",
            )
            return

        # Class-based pLDDT coloring bypasses the B-factor/spectrum path
        if key == "plddt_class":
            try:
                self._ensure_current_data_for_property(prop)
                self._apply_plddt_class_coloring(key, obj_name)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))
            return

        palette, reverse_palette = self._selected_palette()
        vmin, vmax = self._get_vmin_vmax()
        ref_sel = self._ref_edit.text().strip() or None

        try:
            self._ensure_current_data_for_property(prop)
            self._build_token_map_if_needed(obj_name)
            values = self._compute_property_for(
                key, ref_sel, self._pred_data, self._token_map, obj_name
            )
            if values is None:
                return
            self._validate_token_count(values, self._token_map, obj_name)
            if metrics.is_domain_label_metric(key):
                from .painter import delete_colorbar, paint_categorical_labels_bulk

                used_vmin, used_vmax = paint_categorical_labels_bulk(
                    obj_name,
                    self._token_map,
                    values,
                )
                delete_colorbar()
            else:
                from .painter import paint_property, show_colorbar

                used_vmin, used_vmax = paint_property(
                    obj_name,
                    self._token_map,
                    values,
                    palette=palette,
                    reverse_palette=reverse_palette,
                    vmin=vmin,
                    vmax=vmax,
                )
                show_colorbar(
                    palette,
                    reverse_palette,
                    used_vmin,
                    used_vmax,
                    object_names=[obj_name],
                )
            self.setWindowTitle(
                f"{APP_TITLE} - {key} [{used_vmin:.2f}, {used_vmax:.2f}]"
            )
            self._update_statistics_for_single(
                key,
                obj_name,
                values,
                include_chain_stats=key == "pde_chain_mean",
                include_domain_labels=metrics.is_domain_label_metric(key),
                token_map=self._token_map,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _export_csv(self) -> None:
        """Export token-level CSV rows for the current metric and target."""
        default_path = export.default_csv_export_path(
            getattr(self, "_pred_files", None),
            getattr(self, "_pred_data", None),
            self._prop_combo.currentData(),
        )
        result = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export token metric CSV",
            default_path,
            "CSV files (*.csv);;All files (*)",
        )
        path = result[0] if isinstance(result, tuple) else result
        if not path:
            self._raise_after_native_dialog()
            return
        path_obj = Path(path)
        if path_obj.suffix.lower() != ".csv":
            path_obj = path_obj.with_suffix(".csv")
        self._export_csv_to_path(path_obj)

    def _export_csv_to_path(self, path: str | Path) -> None:
        """Build and write CSV rows, reporting GUI errors consistently."""
        try:
            rows = self._build_csv_export_rows()
            if rows is None:
                return
            if not rows:
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, "No token rows were available for export."
                )
                return
            from .export import write_csv

            write_csv(path, rows)
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                f"Exported {len(rows)} token rows to:\n{path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, f"{APP_TITLE} - export error", str(exc)
            )

    def _build_csv_export_rows(self) -> list[dict[str, object]] | None:
        """Return CSV rows for the current metric/target, or None when cancelled."""
        if self._pred_files is None:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, "No prediction output loaded."
            )
            return None
        key = self._prop_combo.currentData()
        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        if not key or not prop:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, "Select a Color by metric before exporting."
            )
            return None
        target = self._resolve_plot_target()
        if target is None:
            return None

        if target.kind == "ensemble_group":
            return self._csv_rows_for_ensemble_group(key, prop, target)
        return self._csv_rows_for_single_target(key, prop, target)

    def _csv_rows_for_single_target(
        self,
        key: str,
        prop: dict,
        target: _PlotTarget,
    ) -> list[dict[str, object]] | None:
        """Build CSV rows for a single model or one ensemble member."""
        member = (
            (target.members or [None])[0] if target.kind == "ensemble_member" else None
        )
        include_ensemble = target.kind == "ensemble_member"

        if prop.get("ensemble_level", False):
            values = self._compute_ensemble_property(key)
            data = (
                target.data
                if target.data is not None
                else getattr(member, "data", None)
            )
            aggregate_kind = metrics.ensemble_aggregate_kind(key)
        else:
            if target.kind == "ensemble_member" and member is not None:
                self._ensure_member_data_for_property(member, prop)
                target.data = member.data
            else:
                self._ensure_current_data_for_property(prop)
                target.data = self._pred_data
            context = self._csv_metric_context(
                key, prop, target.token_map, target.obj_name
            )
            if context is None:
                return None
            compute_key = metrics.line_compute_key(key)
            values = self._compute_property_from_context(
                compute_key,
                target.data,
                target.token_map,
                context,
            )
            if values is None:
                return None
            aggregate_kind = "ensemble_member" if include_ensemble else "single_model"

        self._validate_token_count(values, target.token_map, target.label)
        if prop.get("ensemble_level", False):
            context = self._csv_metric_context(
                key, prop, target.token_map, target.obj_name
            )
            if context is None:
                return None
        return self._csv_rows_from_values(
            key,
            target.data if target.data is not None else data,
            target.token_map,
            values,
            context,
            include_ensemble=include_ensemble,
            member=member,
            aggregate_kind=aggregate_kind,
        )

    def _csv_rows_for_ensemble_group(
        self,
        key: str,
        prop: dict,
        target: _PlotTarget,
    ) -> list[dict[str, object]] | None:
        """Build CSV rows for the active ensemble group target."""
        members = sorted(target.members or [], key=lambda member: member.rank)
        if not members:
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "The ensemble target is not active.\nUse the Ensemble\u2026 button first.",
            )
            return None

        if prop.get("ensemble_level", False):
            context = self._csv_metric_context(
                key, prop, target.token_map, target.obj_name
            )
            if context is None:
                return None
            values = self._compute_ensemble_property(key)
            self._validate_token_count(values, target.token_map, target.label)
            return self._csv_rows_from_values(
                key,
                members[0].data,
                target.token_map,
                values,
                context,
                include_ensemble=True,
                member=None,
                aggregate_kind=metrics.ensemble_aggregate_kind(key),
            )

        rows: list[dict[str, object]] = []
        compute_key = metrics.line_compute_key(key)
        for member in members:
            self._ensure_member_data_for_property(member, prop)
            context = self._csv_metric_context(
                key, prop, member.token_map, member.obj_name
            )
            if context is None:
                return None
            values = self._compute_property_from_context(
                compute_key,
                member.data,
                member.token_map,
                context,
            )
            if values is None:
                return None
            self._validate_token_count(values, member.token_map, member.obj_name)
            rows.extend(
                self._csv_rows_from_values(
                    key,
                    member.data,
                    member.token_map,
                    values,
                    context,
                    include_ensemble=True,
                    member=member,
                    aggregate_kind="ensemble_member",
                )
            )
        return rows

    def _csv_metric_context(
        self,
        key: str,
        prop: dict,
        token_map,
        obj_name: str,
    ) -> dict[str, object] | None:
        """Resolve reference/contact provenance for one export computation."""
        reference_selection = ""
        reference_indices: list[int] = []
        contact_indices: list[int] = []
        cutoff = None

        if prop.get("needs_ref", False):
            resolved = self._resolve_reference_indices(
                token_map, obj_name, required=True
            )
            if resolved is None:
                return None
            reference_indices = list(resolved)
            reference_selection = self._ref_edit.text().strip()

        if key in metrics.CONTACT_FILTERED_METRICS:
            cutoff = self._get_contact_cutoff()
            if cutoff is None:
                return None
            contact_indices = self._binding_site_token_indices(
                token_map,
                obj_name,
                reference_selection,
                reference_indices,
                cutoff,
            )
            if not contact_indices:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "No polymer binding-site residues were found within "
                    f"{cutoff:g} Å of the reference selection.",
                )
                return None
        elif metrics.is_domain_label_metric(key):
            cutoff = self._get_cutoff_threshold()
            if cutoff is None:
                return None

        return {
            "reference_selection": reference_selection,
            "reference_indices": reference_indices,
            "contact_indices": contact_indices,
            "cutoff_angstrom": cutoff,
        }

    def _csv_rows_from_values(
        self,
        key: str,
        data,
        token_map,
        values,
        context: dict[str, object],
        *,
        include_ensemble: bool,
        member,
        aggregate_kind: str,
    ) -> list[dict[str, object]]:
        """Delegate row assembly to the PyMOL-independent exporter."""
        from .export import build_token_rows

        member_rank = getattr(member, "rank", None) if member is not None else None
        member_label = ""
        if member is not None:
            member_label = export.model_label_for_rank(
                self._pred_files, member.rank, fallback=f"model_{member.rank}"
            )
        return build_token_rows(
            pred_files=self._pred_files,
            data=data,
            token_map=token_map,
            values=values,
            metric_key=key,
            reference_selection=str(context["reference_selection"]),
            cutoff_angstrom=context["cutoff_angstrom"],
            reference_indices=context["reference_indices"],
            contact_indices=context["contact_indices"],
            include_ensemble=include_ensemble,
            ensemble_group=getattr(self, "_ensemble_group_name", "") or "",
            ensemble_member_rank=member_rank,
            ensemble_member_label=member_label,
            ensemble_aligned=getattr(self, "_ensemble_aligned", None)
            if include_ensemble
            else None,
            aggregate_kind=aggregate_kind,
        )

    def _apply_ensemble_coloring(self, target_members: list) -> None:
        """Apply the selected property to the chosen ensemble target."""
        key = self._prop_combo.currentData()
        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        if not target_members:
            return

        try:
            self._with_pymol_updates_suspended(
                lambda: self._dispatch_ensemble_coloring(key, prop, target_members)
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _dispatch_ensemble_coloring(
        self, key: str, prop: dict, target_members: list
    ) -> None:
        """Route ensemble coloring while PyMOL updates are suspended."""
        if prop.get("ensemble_level", False):
            self._apply_ensemble_level_property(key, target_members)
        elif key == "plddt_class":
            self._apply_ensemble_plddt_class_coloring(key, target_members)
        else:
            self._apply_individual_property_to_ensemble(key, prop, target_members)

    def _apply_ensemble_level_property(self, key: str, target_members: list) -> None:
        """Compute one ensemble-level array and paint it onto selected targets."""
        values = self._compute_ensemble_property(key)

        from .painter import paint_property_bulk, show_colorbar

        palette, reverse_palette = self._selected_palette()
        vmin, vmax = self._get_vmin_vmax()
        used_vmin = used_vmax = None
        member_values: list[tuple[object, np.ndarray]] = []
        for member in target_members:
            self._validate_token_count(values, member.token_map, member.obj_name)
            used_vmin, used_vmax = paint_property_bulk(
                member.obj_name,
                member.token_map,
                values,
                palette=palette,
                reverse_palette=reverse_palette,
                vmin=vmin,
                vmax=vmax,
                rebuild=False,
            )
            member_values.append((member, values))
        show_colorbar(
            palette,
            reverse_palette,
            used_vmin,
            used_vmax,
            object_names=[member.obj_name for member in target_members],
        )
        label = (
            self._ensemble_group_name
            if len(target_members) > 1
            else target_members[0].obj_name
        )
        self.setWindowTitle(
            f"{APP_TITLE} - {key} on {label} [{used_vmin:.2f}, {used_vmax:.2f}]"
        )
        self._update_statistics_for_single(key, label, values)

    def _apply_individual_property_to_ensemble(
        self, key: str, prop: dict, target_members: list
    ) -> None:
        """Compute selected per-model properties for selected ensemble targets."""
        from .painter import (
            delete_colorbar,
            paint_categorical_labels_bulk,
            paint_property_bulk,
            show_colorbar,
        )

        palette, reverse_palette = self._selected_palette()
        user_vmin, user_vmax = self._get_vmin_vmax()
        ref_sel = self._ref_edit.text().strip() or None

        member_values: list[tuple[object, np.ndarray]] = []
        for member in target_members:
            self._ensure_member_data_for_property(member, prop)
            values = self._compute_property_for(
                key, ref_sel, member.data, member.token_map, member.obj_name
            )
            if values is None:
                return
            self._validate_token_count(values, member.token_map, member.obj_name)
            member_values.append((member, values))

        if metrics.is_domain_label_metric(key):
            used_ranges = []
            for member, values in member_values:
                used_ranges.append(
                    paint_categorical_labels_bulk(
                        member.obj_name,
                        member.token_map,
                        values,
                        rebuild=False,
                    )
                )
            delete_colorbar()
            finite_ranges = [
                (vmin, vmax)
                for vmin, vmax in used_ranges
                if np.isfinite(vmin) and np.isfinite(vmax)
            ]
            shared_vmin = min((vmin for vmin, _vmax in finite_ranges), default=0.0)
            shared_vmax = max((vmax for _vmin, vmax in finite_ranges), default=1.0)
            label = (
                self._ensemble_group_name
                if len(target_members) > 1
                else target_members[0].obj_name
            )
            self.setWindowTitle(
                f"{APP_TITLE} - {key} on {label} [{shared_vmin:.2f}, {shared_vmax:.2f}]"
            )
            self._update_statistics_for_members(
                key,
                label,
                member_values,
                include_domain_labels=True,
            )
            return

        finite = np.concatenate(
            [values[np.isfinite(values)] for _, values in member_values]
        )
        shared_vmin = user_vmin
        shared_vmax = user_vmax
        if shared_vmin is None:
            shared_vmin = float(finite.min()) if finite.size else 0.0
        if shared_vmax is None:
            shared_vmax = float(finite.max()) if finite.size else 1.0
        if shared_vmin == shared_vmax:
            shared_vmax = shared_vmin + 1.0

        for member, values in member_values:
            paint_property_bulk(
                member.obj_name,
                member.token_map,
                values,
                palette=palette,
                reverse_palette=reverse_palette,
                vmin=shared_vmin,
                vmax=shared_vmax,
                rebuild=False,
            )
        show_colorbar(
            palette,
            reverse_palette,
            shared_vmin,
            shared_vmax,
            object_names=[member.obj_name for member, _values in member_values],
        )
        label = (
            self._ensemble_group_name
            if len(target_members) > 1
            else target_members[0].obj_name
        )
        self.setWindowTitle(
            f"{APP_TITLE} - {key} on {label} [{shared_vmin:.2f}, {shared_vmax:.2f}]"
        )
        self._update_statistics_for_members(
            key,
            label,
            member_values,
            include_chain_stats=key == "pde_chain_mean",
        )

    def _apply_ensemble_plddt_class_coloring(
        self, key: str, target_members: list
    ) -> None:
        """Apply quality-class pLDDT coloring to selected ensemble targets."""
        from .painter import delete_colorbar, paint_plddt_class_coloring

        member_values: list[tuple[object, np.ndarray]] = []
        for member in target_members:
            self._ensure_member_data_for_property(
                member, metrics.PROPERTY_BY_KEY["plddt_class"]
            )
            values, _source_label = compute.plddt_values_for(member.data)
            if values is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    f"pLDDT data are not available for model_{member.rank}.",
                )
                return
            self._validate_token_count(values, member.token_map, member.obj_name)
            paint_plddt_class_coloring(
                member.obj_name,
                values=values,
                token_map=member.token_map,
                rebuild=False,
            )
            member_values.append((member, values))
        delete_colorbar()
        label = (
            self._ensemble_group_name
            if len(target_members) > 1
            else target_members[0].obj_name
        )
        self.setWindowTitle(f"{APP_TITLE} - pLDDT quality classes on {label}")
        self._update_statistics_for_members(
            key, label, member_values, include_plddt_classes=True
        )

    def _with_pymol_updates_suspended(self, func):
        """Run *func* with PyMOL viewport updates suspended, then rebuild once."""
        from pymol import cmd

        try:
            cmd.set("suspend_updates", "on")
        except Exception:
            pass
        try:
            return func()
        finally:
            try:
                cmd.set("suspend_updates", "off")
                cmd.rebuild()
            except Exception:
                pass

    def _apply_plddt_class_coloring(self, key: str, obj_name: str) -> None:
        """Apply the 4-class AlphaFold pLDDT colour scheme.

        Writes preferred pLDDT values to B-factors before colouring, so previous
        plugin visualisations cannot corrupt the result.
        """
        from .painter import delete_colorbar, paint_plddt_class_coloring

        values, _source_label = compute.plddt_values_for(self._pred_data)
        if values is None:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "pLDDT data are not available for this model.",
            )
            return
        self._build_token_map_if_needed(obj_name)
        self._validate_token_count(values, self._token_map, obj_name)
        paint_plddt_class_coloring(
            obj_name,
            values=values,
            token_map=self._token_map,
        )

        delete_colorbar()
        self.setWindowTitle(f"{APP_TITLE} - pLDDT quality classes")
        self._update_statistics_for_single(
            key, obj_name, values, include_plddt_classes=True
        )

    def _build_token_map_if_needed(self, obj_name: str) -> None:
        """(Re-)build the token map if the object changed.

        The structure path from the loaded prediction data is passed to
        ``build_token_map`` so that HETATM atom order is read from the file
        rather than from PyMOL's internal (alphabetically sorted) model.
        """
        current_obj = getattr(self, "_token_map_obj", None)
        current_path = getattr(self, "_token_map_structure_path", None)
        structure_path = (
            None if self._pred_data is None else self._pred_data.structure_path
        )
        if (
            self._token_map is None
            or current_obj != obj_name
            or current_path != structure_path
        ):
            if self._pred_data is None:
                raise ValueError("No prediction data loaded; cannot build token map.")
            from .token_map import build_token_map

            self._token_map = build_token_map(obj_name, self._pred_data.structure_path)
            self._token_map_obj = obj_name  # type: ignore[attr-defined]
            self._token_map_structure_path = self._pred_data.structure_path

    def _compute_property(self, key: str, ref_sel: str | None):
        """Dispatch to properties.py functions. Returns per-token array or None."""
        return self._compute_property_for(
            key,
            ref_sel,
            self._pred_data,
            self._token_map,
            self._obj_combo.currentText(),
        )

    def _compute_property_for(
        self,
        key: str,
        ref_sel: str | None,
        data,
        tm,
        obj_name: str,
    ):
        """Resolve GUI/PyMOL context and dispatch one per-model metric."""
        from .token_map import selection_to_token_indices

        def _need_ref():
            if not ref_sel:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "This property requires a reference selection.\n"
                    "Enter a PyMOL selection in the Reference field.",
                )
                return None
            indices = selection_to_token_indices(tm, ref_sel, obj_name=obj_name)
            if not indices:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    f"Reference selection '{ref_sel}' matched no tokens in {obj_name}.",
                )
                return None
            return indices

        ref_indices = None
        contact_indices = None
        cutoff = None

        if key == "ensemble_rmsd":
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Ensemble RMSD requires all models to be loaded.\n"
                "Use the Ensemble… button.",
            )
            return None

        if (
            key == "contact_prob_to_sel"
            and getattr(data, "contact_probs", None) is None
        ):
            return self._compute_metric_with_messages(key, data, tm)

        if key in {
            "pae_to_sel",
            "pae_col_to_sel",
            "pae_sym_sel",
            "pae_sym_within_sel",
            "pae_contact",
            "pde_to_sel",
            "pde_within_sel",
            "pde_contact",
            "contact_prob_to_sel",
        }:
            ref_indices = _need_ref()
            if ref_indices is None:
                return None

        if key in ("pae_domain_complete", "pae_domain_spectral"):
            cutoff = self._get_cutoff_threshold()
            if cutoff is None:
                return None
            method = compute.pae_domain_method(key)
            if not self._pae_domain_dependency_available(method):
                return None

        if key in metrics.CONTACT_FILTERED_METRICS:
            cutoff = self._get_contact_cutoff()
            if cutoff is None:
                return None
            contact_indices = self._binding_site_token_indices(
                tm, obj_name, ref_sel, ref_indices, cutoff
            )
            if not contact_indices:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "No polymer binding-site residues were found within "
                    f"{cutoff:g} Å of the reference selection.",
                )
                return None

        return self._compute_metric_with_messages(
            key,
            data,
            tm,
            ref_indices=ref_indices,
            contact_indices=contact_indices,
            cutoff=cutoff,
        )

    def _compute_property_from_context(
        self,
        key: str,
        data,
        tm,
        context: dict[str, object],
    ):
        """Dispatch a metric using already-resolved export context."""
        cutoff = context["cutoff_angstrom"]
        if key in ("pae_domain_complete", "pae_domain_spectral"):
            method = compute.pae_domain_method(key)
            if not self._pae_domain_dependency_available(method):
                return None
        return self._compute_metric_with_messages(
            key,
            data,
            tm,
            ref_indices=list(context["reference_indices"]),
            contact_indices=list(context["contact_indices"]),
            cutoff=cutoff,
        )

    def _compute_metric_with_messages(
        self,
        key: str,
        data,
        tm,
        *,
        ref_indices: list[int] | None = None,
        contact_indices: list[int] | None = None,
        cutoff: float | None = None,
    ):
        """Call pure compute dispatch and translate expected errors to GUI text."""
        try:
            return compute.compute_metric(
                key,
                data,
                tm,
                ref_indices=ref_indices,
                contact_indices=contact_indices,
                cutoff=cutoff,
            )
        except compute.MissingMetricDataError:
            if key == "plddt":
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, "pLDDT data are not available for this model."
                )
            elif key.startswith("contact_prob"):
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, "Interaction probability data are not available."
                )
            elif key == "chain_iptm":
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, "Confidence JSON not available."
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, "Required metric data are not available."
                )
            return None
        except compute.MissingReferenceError:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "This property requires a reference selection.\n"
                "Enter a PyMOL selection in the Reference field.",
            )
            return None
        except compute.MissingContactError:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "No polymer binding-site residues were found within "
                "the cutoff of the reference selection.",
            )
            return None
        except compute.UnsupportedMetricError:
            if key == "ensemble_rmsd":
                QtWidgets.QMessageBox.information(
                    self,
                    APP_TITLE,
                    "Ensemble RMSD requires all models to be loaded.\n"
                    "Use the Ensemble… button.",
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, f"Unknown property key: {key}"
                )
            return None
        except compute.MetricComputationError as exc:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, str(exc))
            return None

    def _pae_domain_dependency_available(self, method: str) -> bool:
        """Warn and return False when a PAE domain-label dependency is missing."""
        if method == "complete_linkage":
            if importlib.util.find_spec("scipy") is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "PAE domain labels (complete linkage) require SciPy. "
                    "Install scipy in the Python environment used by PyMOL.",
                )
                return False
            return True
        if method == "spectral":
            missing = []
            if importlib.util.find_spec("scipy") is None:
                missing.append("SciPy")
            if importlib.util.find_spec("sklearn") is None:
                missing.append("scikit-learn")
            if missing:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "PAE domain labels (spectral clustering) require "
                    f"{' and '.join(missing)}. Install the missing package(s) "
                    "in the Python environment used by PyMOL.",
                )
                return False
            return True
        return True

    def _compute_ensemble_property(self, key: str) -> np.ndarray:
        """Return an ensemble-level per-token array."""
        members = self._ensemble_members or []
        if not members:
            raise ValueError("No active ensemble.")

        if key == "ensemble_rmsd":
            if self._ensemble_rmsd is None:
                raise ValueError(
                    "Ensemble RMSD has not been computed. Use the Ensemble… button again."
                )
            return self._ensemble_rmsd

        if key in ("ensemble_plddt_mean", "ensemble_plddt_std"):
            if self._ensemble_plddt_mean is None or self._ensemble_plddt_std is None:
                raise ValueError(
                    "Ensemble pLDDT consensus has not been computed. "
                    "Use the Ensemble… button again."
                )
            return (
                self._ensemble_plddt_mean
                if key == "ensemble_plddt_mean"
                else self._ensemble_plddt_std
            )

        raise ValueError(f"Unknown ensemble property: {key}")

    def _validate_token_count(self, values, token_map, obj_name: str) -> None:
        """Raise a helpful error if a property array does not match a token map."""
        if values is None or token_map is None:
            raise ValueError("No values or token map available for coloring.")
        if len(values) != len(token_map):
            raise ValueError(
                f"Token count mismatch for {obj_name}: property has {len(values)} "
                f"values, but the loaded structure maps to {len(token_map)} tokens. "
                "Check that the PyMOL object belongs to the selected prediction model."
            )

    def _binding_site_token_indices(
        self,
        token_map,
        obj_name: str,
        ref_sel: str,
        ref_indices: list[int],
        cutoff: float,
    ) -> list[int]:
        """Return polymer tokens with any atom within *cutoff* Å of reference."""
        from .token_map import selection_to_token_indices

        scoped_ref_sel = f"(({ref_sel}) and ({obj_name}))"
        binding_sel = (
            f"byres (({obj_name}) within {cutoff:g} of {scoped_ref_sel}) "
            f"and not {scoped_ref_sel}"
        )
        raw_binding_indices = selection_to_token_indices(
            token_map, binding_sel, obj_name=obj_name
        )
        ref_set = set(ref_indices)
        return [
            idx
            for idx in raw_binding_indices
            if idx not in ref_set and not token_map[idx].is_hetatm
        ]

    def _ensure_current_data_for_property(self, prop: dict) -> None:
        """Reload current single-model data if a lazy property needs more arrays."""
        if self._pred_files is None or self._pred_data is None:
            raise ValueError("No prediction output loaded.")
        flags = metrics.metric_load_flags(prop)
        data = self._pred_data
        if prop.get("needs_any_plddt", False) and (
            getattr(data, "structure_plddt", None) is not None
            or getattr(data, "plddt", None) is not None
        ):
            flags["load_structure_plddt"] = False
            flags["load_plddt"] = False
        needs_reload = (
            (flags["load_pae"] and getattr(data, "pae", None) is None)
            or (flags["load_pde"] and getattr(data, "pde", None) is None)
            or (
                flags["load_contact_probs"]
                and getattr(data, "contact_probs", None) is None
            )
            or (
                flags["load_structure_plddt"]
                and getattr(data, "structure_plddt", None) is None
            )
            or (flags["load_plddt"] and getattr(data, "plddt", None) is None)
        )
        if needs_reload:
            self._pred_data = self._reload_prediction_data(
                data.rank,
                load_pae=flags["load_pae"] or getattr(data, "pae", None) is not None,
                load_pde=flags["load_pde"] or getattr(data, "pde", None) is not None,
                load_contact_probs=flags["load_contact_probs"]
                or getattr(data, "contact_probs", None) is not None,
                load_structure_plddt=flags["load_structure_plddt"]
                or getattr(data, "structure_plddt", None) is not None,
                load_plddt=flags["load_plddt"]
                or getattr(data, "plddt", None) is not None,
            )

    def _reload_prediction_data(self, rank: int, **flags):
        """Load one model while preserving the provider-aware loader defaults."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        from .loader import load_prediction_data

        return load_prediction_data(self._pred_files, rank, **flags)

    def _ensure_member_data_for_property(self, member, prop: dict) -> None:
        """Reload member data with large matrices only when the property needs them."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        flags = metrics.metric_load_flags(prop)
        needs_pae = flags["load_pae"]
        needs_pde = flags["load_pde"]
        needs_contact_probs = flags["load_contact_probs"]
        needs_structure_plddt = flags["load_structure_plddt"]
        needs_plddt = flags["load_plddt"]
        if prop.get("needs_any_plddt", False) and (
            getattr(member.data, "structure_plddt", None) is not None
            or getattr(member.data, "plddt", None) is not None
        ):
            needs_structure_plddt = False
            needs_plddt = False
        if (
            (needs_pae and getattr(member.data, "pae", None) is None)
            or (needs_pde and getattr(member.data, "pde", None) is None)
            or (
                needs_contact_probs
                and getattr(member.data, "contact_probs", None) is None
            )
            or (
                needs_structure_plddt
                and getattr(member.data, "structure_plddt", None) is None
            )
            or (needs_plddt and getattr(member.data, "plddt", None) is None)
        ):
            member.data = self._reload_prediction_data(
                member.rank,
                load_pae=needs_pae or getattr(member.data, "pae", None) is not None,
                load_pde=needs_pde or getattr(member.data, "pde", None) is not None,
                load_contact_probs=needs_contact_probs
                or getattr(member.data, "contact_probs", None) is not None,
                load_structure_plddt=needs_structure_plddt
                or getattr(member.data, "structure_plddt", None) is not None,
                load_plddt=needs_plddt
                or getattr(member.data, "plddt", None) is not None,
            )

    def _ensure_member_data_for_plot(
        self,
        member,
        *,
        load_pae: bool = False,
        load_pde: bool = False,
        load_contact_probs: bool = False,
        load_structure_plddt: bool = False,
        load_plddt: bool = False,
    ) -> None:
        """Reload an ensemble member while preserving already-loaded plot arrays."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        data = member.data
        if (
            load_structure_plddt
            and load_plddt
            and (
                getattr(data, "structure_plddt", None) is not None
                or getattr(data, "plddt", None) is not None
            )
        ):
            load_structure_plddt = False
            load_plddt = False
        needs_reload = (
            (load_pae and getattr(data, "pae", None) is None)
            or (load_pde and getattr(data, "pde", None) is None)
            or (load_contact_probs and getattr(data, "contact_probs", None) is None)
            or (load_structure_plddt and getattr(data, "structure_plddt", None) is None)
            or (load_plddt and getattr(data, "plddt", None) is None)
        )
        if not needs_reload:
            return

        member.data = self._reload_prediction_data(
            member.rank,
            load_pae=load_pae or getattr(data, "pae", None) is not None,
            load_pde=load_pde or getattr(data, "pde", None) is not None,
            load_contact_probs=load_contact_probs
            or getattr(data, "contact_probs", None) is not None,
            load_structure_plddt=load_structure_plddt
            or getattr(data, "structure_plddt", None) is not None,
            load_plddt=load_plddt or getattr(data, "plddt", None) is not None,
        )

    def _resolve_plot_target(self) -> _PlotTarget | None:
        """Resolve the current PyMOL target into data and token-map context."""
        obj_name = self._get_obj_name()
        if obj_name is None:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "No PyMOL target selected.")
            return None

        ensemble_group_name = getattr(self, "_ensemble_group_name", None)
        ensemble_members = getattr(self, "_ensemble_members", None)
        if obj_name == ensemble_group_name:
            members = sorted(ensemble_members or [], key=lambda member: member.rank)
            if not members:
                QtWidgets.QMessageBox.information(
                    self,
                    APP_TITLE,
                    "The ensemble target is not active.\nUse the Ensemble\u2026 button first.",
                )
                return None
            reference = members[0]
            return _PlotTarget(
                kind="ensemble_group",
                label=obj_name,
                obj_name=reference.obj_name,
                data=None,
                token_map=reference.token_map,
                members=members,
            )

        member = self._selected_ensemble_member(obj_name)
        if member is not None:
            return _PlotTarget(
                kind="ensemble_member",
                label=member.obj_name,
                obj_name=member.obj_name,
                data=member.data,
                token_map=member.token_map,
                members=[member],
            )

        if self._pred_data is None:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "No prediction data loaded.")
            return None
        self._build_token_map_if_needed(obj_name)
        return _PlotTarget(
            kind="single",
            label=obj_name,
            obj_name=obj_name,
            data=self._pred_data,
            token_map=self._token_map,
            members=None,
        )

    def _resolve_reference_indices(
        self,
        token_map,
        obj_name: str,
        *,
        required: bool = False,
    ) -> list[int] | None:
        """Resolve the Reference field to token indices, preserving token order."""
        ref_sel = self._ref_edit.text().strip()
        if not ref_sel:
            if required:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "This plot requires a reference selection.\n"
                    "Enter a PyMOL selection in the Reference field.",
                )
                return None
            return []

        from .token_map import selection_to_token_indices

        indices = selection_to_token_indices(token_map, ref_sel, obj_name=obj_name)
        if not indices:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                f"Reference selection '{ref_sel}' matched no tokens in {obj_name}.",
            )
            return None
        return indices

    def _compute_line_plot_data(
        self,
        key: str,
        target: _PlotTarget,
        ref_indices: list[int],
        *,
        plot_type: str = "line",
    ) -> tuple[np.ndarray, list[tuple[str, np.ndarray, np.ndarray | None]], str]:
        """Return x values, series tuples, and y-axis label for a line plot."""
        token_map = target.token_map
        use_ref_scope = bool(ref_indices) and metrics.plot_uses_reference_scope(
            key, plot_type
        )
        indices = list(ref_indices) if use_ref_scope else list(range(len(token_map)))
        if not indices:
            raise ValueError("No tokens are available for the line plot.")

        compute_key = metrics.line_compute_key(key)
        ref_edit = getattr(self, "_ref_edit", None)
        ref_sel = None if ref_edit is None else ref_edit.text().strip() or None

        if target.kind == "ensemble_group":
            if compute_key == "ensemble_rmsd":
                values = self._compute_ensemble_property("ensemble_rmsd")
                return (
                    np.asarray(indices, dtype=np.int32),
                    [(metrics.metric_label(key), values[indices], None)],
                    metrics.line_ylabel(compute_key),
                )
            if compute_key == "ensemble_plddt_mean":
                mean = self._compute_ensemble_property("ensemble_plddt_mean")
                std = self._compute_ensemble_property("ensemble_plddt_std")
                return (
                    np.asarray(indices, dtype=np.int32),
                    [(metrics.metric_label(key), mean[indices], std[indices])],
                    metrics.line_ylabel(compute_key),
                )
            if compute_key == "ensemble_plddt_std":
                values = self._compute_ensemble_property("ensemble_plddt_std")
                return (
                    np.asarray(indices, dtype=np.int32),
                    [(metrics.metric_label(key), values[indices], None)],
                    metrics.line_ylabel(compute_key),
                )

            (
                load_pae,
                load_pde,
                load_contact_probs,
                load_structure,
                load_plddt,
            ) = plot_data.line_member_load_flags(key)
            arrays = []
            for member in target.members or []:
                kwargs = dict(
                    load_pae=load_pae,
                    load_pde=load_pde,
                    load_structure_plddt=load_structure,
                )
                if load_contact_probs:
                    kwargs["load_contact_probs"] = True
                if load_plddt:
                    kwargs["load_plddt"] = True
                self._ensure_member_data_for_plot(member, **kwargs)
                values = self._compute_property_for(
                    compute_key, ref_sel, member.data, member.token_map, member.obj_name
                )
                if values is None:
                    raise ValueError("Could not compute the selected property.")
                self._validate_token_count(values, member.token_map, member.obj_name)
                arrays.append(np.asarray(values, dtype=np.float32))
            mean, std = plot_data.nan_mean_std(arrays, len(token_map))
            if mean is None:
                raise ValueError("No ensemble values are available for this plot.")
            return (
                np.asarray(indices, dtype=np.int32),
                [(f"{metrics.metric_label(key)} mean", mean[indices], std[indices])],
                metrics.line_ylabel(compute_key),
            )

        if compute_key.startswith("ensemble_"):
            values = self._compute_ensemble_property(compute_key)
        else:
            values = self._compute_property_for(
                compute_key, ref_sel, target.data, token_map, target.obj_name
            )
        if values is None:
            raise ValueError("Could not compute the selected property.")
        self._validate_token_count(values, token_map, target.label)
        return (
            np.asarray(indices, dtype=np.int32),
            [(metrics.metric_label(key), np.asarray(values)[indices], None)],
            metrics.line_ylabel(compute_key),
        )

    def _summary_plot_has_matrix_data(self, kind: str, target: _PlotTarget) -> bool:
        """Return whether the target can provide the requested summary matrix."""
        attr = "pae" if kind == "pae" else "pde"
        if self._has_matrix_data_family(attr):
            return True
        if target.kind == "ensemble_group":
            return any(
                getattr(member.data, attr, None) is not None
                for member in target.members or []
            )
        return getattr(target.data, attr, None) is not None

    def _compute_summary_plot_data(
        self,
        kind: str,
        target: _PlotTarget,
        ref_indices: list[int],
    ) -> tuple[
        np.ndarray,
        list[
            tuple[str, np.ndarray, np.ndarray | None]
            | tuple[str, np.ndarray, np.ndarray | None, str]
        ],
        str,
    ]:
        """Return x values, series tuples, and y-axis label for a summary plot."""
        if kind not in {"pae", "pde"}:
            raise ValueError(f"Unknown summary plot kind: {kind}")
        if not plot_data.has_multiple_token_chains(target.token_map):
            raise ValueError("Summary plots require a target with more than one chain.")

        indices = (
            list(ref_indices) if ref_indices else list(range(len(target.token_map)))
        )
        if not indices:
            raise ValueError("No tokens are available for the summary plot.")

        load_pae = kind == "pae"
        load_pde = kind == "pde"
        if target.kind == "ensemble_group":
            data_items = []
            token_maps = []
            for member in sorted(target.members or [], key=lambda item: item.rank):
                self._ensure_member_data_for_plot(
                    member, load_pae=load_pae, load_pde=load_pde
                )
                data_items.append(member.data)
                token_maps.append(member.token_map)
            series = plot_data.summary_series_for_ensemble(
                kind,
                data_items,
                target.token_map,
                token_maps=token_maps,
            )
        else:
            if target.kind == "ensemble_member" and target.members:
                self._ensure_member_data_for_plot(
                    target.members[0], load_pae=load_pae, load_pde=load_pde
                )
                target.data = target.members[0].data
            elif target.kind == "single" and target.data is self._pred_data:
                self._ensure_current_data_for_property(
                    {
                        "needs_pae": load_pae,
                        "needs_pde": load_pde,
                    }
                )
                target.data = self._pred_data
            series = plot_data.summary_series_for_data(
                kind, target.data, target.token_map
            )

        sliced = []
        for item in series:
            label, values, std = item[0], item[1], item[2]
            sliced_item = (
                label,
                np.asarray(values, dtype=np.float32)[indices],
                None if std is None else np.asarray(std, dtype=np.float32)[indices],
            )
            if len(item) == 4:
                sliced.append((*sliced_item, item[3]))
            else:
                sliced.append(sliced_item)
        ylabel = "PAE gap (Å)" if kind == "pae" else "PDE gap (Å)"
        return np.asarray(indices, dtype=np.int32), sliced, ylabel

    def _compute_matrix_plot_data(
        self,
        key: str,
        target: _PlotTarget,
        ref_indices: list[int],
    ) -> tuple[
        np.ndarray,
        list[int],
        list[int],
        str,
        str,
        list[str] | None,
        list[str] | None,
        np.ndarray | None,
    ]:
        """Return matrix data and display metadata for a matrix plot."""
        source = metrics.matrix_source_for_metric(key)
        if source is None:
            raise ValueError(
                "Matrix plots are only available for PAE, PDE, interaction "
                "probability, and chain ipTM properties."
            )
        attr, title, label = source
        if attr == "chain_iptm":
            data = (
                target.members[0].data
                if target.kind == "ensemble_member" and target.members
                else target.data
            )
            return plot_data.chain_iptm_matrix_plot_data(
                target_kind=target.kind,
                data=data,
                token_map=target.token_map,
                title=title,
                label=label,
                members=target.members,
            )

        load_pae = attr == "pae"
        load_pde = attr == "pde"
        load_contact_probs = attr == "contact_probs"

        if target.kind == "ensemble_group":
            matrices = []
            for member in target.members or []:
                kwargs = dict(load_pae=load_pae, load_pde=load_pde)
                if load_contact_probs:
                    kwargs["load_contact_probs"] = True
                self._ensure_member_data_for_plot(member, **kwargs)
                matrix = getattr(member.data, attr, None)
                if matrix is None:
                    raise ValueError(
                        f"{label} matrix is not available for model_{member.rank}."
                    )
                matrices.append(np.asarray(matrix, dtype=np.float32))
            matrix = np.stack(matrices, axis=0).mean(axis=0)
            title = f"{title} — ensemble mean"
        else:
            if target.kind == "ensemble_member" and target.members:
                kwargs = dict(load_pae=load_pae, load_pde=load_pde)
                if load_contact_probs:
                    kwargs["load_contact_probs"] = True
                self._ensure_member_data_for_plot(target.members[0], **kwargs)
                target.data = target.members[0].data
            elif (
                target.kind == "single"
                and target.data is getattr(self, "_pred_data", None)
                and getattr(self, "_pred_files", None) is not None
            ):
                self._ensure_current_data_for_property(
                    metrics.PROPERTY_BY_KEY.get(key, {})
                )
                target.data = self._pred_data
            matrix = getattr(target.data, attr, None)
            if matrix is None:
                raise ValueError(f"{label} matrix is not available for this model.")
            matrix = np.asarray(matrix, dtype=np.float32)

        if key == "pae_row_mean" and ref_indices:
            row_indices = list(ref_indices)
            col_indices = list(range(matrix.shape[1]))
        elif key == "pae_col_to_sel" and ref_indices:
            row_indices = list(ref_indices)
            col_indices = list(range(matrix.shape[1]))
        elif key == "pae_sym_within_sel" and ref_indices:
            row_indices = list(ref_indices)
            col_indices = list(ref_indices)
        else:
            row_indices = list(range(matrix.shape[0]))
            col_indices = (
                list(ref_indices) if ref_indices else list(range(matrix.shape[1]))
            )
        submatrix = matrix[np.ix_(row_indices, col_indices)]
        return submatrix, row_indices, col_indices, title, label, None, None, None

    def _ensemble_site_summary_for_member(
        self,
        member,
        ref_sel: str,
        cutoff: float,
    ) -> dict:
        """Compute local ligand-site summary values for one ensemble member."""
        from .token_map import selection_to_token_indices

        self._ensure_member_data_for_plot(
            member,
            load_pae=getattr(self._pred_files, "has_pae", False)
            if self._pred_files
            else False,
            load_pde=getattr(self._pred_files, "has_pde", False)
            if self._pred_files
            else False,
            load_structure_plddt=True,
            load_plddt=True,
        )
        ref_indices = selection_to_token_indices(
            member.token_map, ref_sel, obj_name=member.obj_name
        )
        if not ref_indices:
            raise ValueError(
                f"Reference selection '{ref_sel}' matched no tokens in {member.obj_name}."
            )
        contact_indices = self._binding_site_token_indices(
            member.token_map, member.obj_name, ref_sel, ref_indices, cutoff
        )
        site_indices = list(dict.fromkeys(list(ref_indices) + list(contact_indices)))
        if not site_indices:
            raise ValueError(f"No site tokens are available for {member.obj_name}.")

        return {
            "member": member,
            "site_indices": site_indices,
            **plot_data.site_summary_values(member.data, site_indices),
        }

    def _compute_ensemble_site_summary_data(
        self,
        ref_sel: str,
        cutoff: float,
    ) -> tuple[list, list[str], list[tuple[str, np.ndarray, str]], list[list[int]]]:
        """Return ensemble members, labels, metric series, and site-token groups."""
        members = sorted(self._ensemble_members or [], key=lambda member: member.rank)
        if not members:
            raise ValueError(
                "The ensemble target is not active. Use Load Ensemble\u2026 first."
            )

        rows = [
            self._ensemble_site_summary_for_member(member, ref_sel, cutoff)
            for member in members
        ]
        labels = [f"model_{row['member'].rank}" for row in rows]
        site_indices = [row["site_indices"] for row in rows]
        metric_specs = [
            ("mean pLDDT", "plddt", "steelblue"),
            ("PAE mean", "pae", "tomato"),
            ("PDE mean", "pde", "goldenrod"),
        ]
        series: list[tuple[str, np.ndarray, str]] = []
        for label, key, color in metric_specs:
            values = np.asarray([row[key] for row in rows], dtype=np.float32)
            if np.any(np.isfinite(values)):
                series.append((label, values, color))
        return members, labels, series, site_indices

    def _compute_fingerprint_data(
        self,
        target: _PlotTarget,
        ref_indices: list[int],
    ) -> dict[str, np.ndarray | None]:
        """Return mean/std fingerprint series for a single target or ensemble."""
        size = len(target.token_map)
        if target.kind != "ensemble_group":
            if target.kind == "ensemble_member" and target.members:
                self._ensure_member_data_for_plot(
                    target.members[0],
                    load_pae=getattr(self._pred_files, "has_pae", False)
                    if self._pred_files
                    else False,
                    load_pde=getattr(self._pred_files, "has_pde", False)
                    if self._pred_files
                    else False,
                    load_contact_probs=getattr(
                        self._pred_files, "has_contact_probs", False
                    )
                    if self._pred_files
                    else False,
                )
                target.data = target.members[0].data
            elif (
                target.kind == "single"
                and target.data is self._pred_data
                and getattr(self, "_pred_files", None) is not None
            ):
                self._pred_data = self._reload_prediction_data(
                    self._pred_data.rank,
                    load_pae=getattr(self._pred_files, "has_pae", False)
                    if self._pred_files
                    else False,
                    load_pde=getattr(self._pred_files, "has_pde", False)
                    if self._pred_files
                    else False,
                    load_contact_probs=getattr(
                        self._pred_files, "has_contact_probs", False
                    )
                    if self._pred_files
                    else False,
                    load_structure_plddt=True,
                    load_plddt=True,
                )
                target.data = self._pred_data
            return plot_data.fingerprint_series_for_single(target.data, ref_indices)

        data_items = []
        for member in target.members or []:
            self._ensure_member_data_for_plot(
                member,
                load_pae=getattr(self._pred_files, "has_pae", False)
                if self._pred_files
                else False,
                load_pde=getattr(self._pred_files, "has_pde", False)
                if self._pred_files
                else False,
                load_contact_probs=getattr(self._pred_files, "has_contact_probs", False)
                if self._pred_files
                else False,
            )
            data_items.append(member.data)

        return plot_data.fingerprint_series_for_ensemble(
            data_items, ref_indices, size=size
        )

    def _show_selected_plot(self, plot_type: str | None = None) -> None:
        """Dispatch the selected plot type to its plot handler."""
        if plot_type is None and hasattr(self, "_plot_type_combo"):
            plot_type = self._plot_type_combo.currentData()
        if plot_type is None:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "No plot type selected.")
            return
        key = self._prop_combo.currentData()
        state = gui_rules.plot_action_state(
            plot_type,
            key,
            self._current_target_kind(),
            bool(self._ref_edit.text().strip()),
            bool(getattr(self, "_ensemble_members", None)),
            has_fingerprint_data=self._has_fingerprint_data(),
            has_pae_data=self._has_matrix_data_family("pae"),
            has_pde_data=self._has_matrix_data_family("pde"),
            has_multiple_chains=self._current_target_has_multiple_chains(),
        )
        if not state.enabled:
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                state.reason or f"{plot_type} is not available.",
            )
            return
        if plot_type == "line":
            self._show_line_plot()
        elif plot_type == "distribution":
            self._show_distribution_plot()
        elif plot_type == "matrix":
            self._show_matrix_plot()
        elif plot_type == "pae_summary":
            self._show_summary_plot("pae")
        elif plot_type == "pde_summary":
            self._show_summary_plot("pde")
        elif plot_type == "binding_site_fingerprint":
            self._show_binding_site_fingerprint()
        elif plot_type == "ensemble_site_summary":
            self._show_ensemble_site_summary()
        else:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, f"Unknown plot type: {plot_type}"
            )

    def _show_line_plot(self) -> None:
        """Open a token-indexed line plot for the selected property."""
        target = self._resolve_plot_target()
        if target is None:
            return

        key = self._prop_combo.currentData()
        if metrics.is_domain_label_metric(key):
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Line plots are not available for PAE domain labels.\n"
                "Use Distribution to inspect cluster occupancy.",
            )
            return
        if key in metrics.CONTACT_FILTERED_METRICS:
            metric_name = "PAE" if key == "pae_contact" else "PDE"
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                f"Line plots are not available for {metric_name} "
                "contact-filtered values.\n"
                "Use Distribution or Matrix instead.",
            )
            return
        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        ref_indices = self._resolve_reference_indices(
            target.token_map, target.obj_name, required=prop.get("needs_ref", False)
        )
        if ref_indices is None:
            return

        try:
            if target.kind == "single":
                self._ensure_current_data_for_property(prop)
                target.data = self._pred_data
            from . import plots

            x_values, series, ylabel = self._compute_line_plot_data(
                key, target, ref_indices, plot_type="line"
            )
            has_finite_values = any(
                np.any(np.isfinite(np.asarray(item[1], dtype=np.float64)))
                for item in series
            )
            if not has_finite_values:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "No finite values are available for this line plot.",
                )
                return
            indices = list(map(int, x_values.tolist()))
            boundaries, labels = plot_data.chain_boundaries(
                target.token_map, indices, original_x=True
            )
            vmin, vmax = self._get_vmin_vmax()
            title = f"{metrics.metric_label(key)} ({target.label})"
            fig = plots.make_line_plot(
                x_values,
                series,
                title=title,
                ylabel=ylabel,
                ymin=vmin,
                ymax=vmax,
                chain_boundaries=boundaries,
                chain_labels=labels,
            )
            plots.attach_pymol_selection_metadata(
                fig,
                kind="line",
                token_map=target.token_map,
                obj_name=target.obj_name,
                token_maps=self._plot_selection_token_maps(target),
                token_map_obj_names=self._plot_selection_obj_names(target),
                token_indices=indices,
                x_positions=x_values.tolist(),
            )
            self._show_plot_figure(fig, title)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _show_summary_plot(self, kind: str) -> None:
        """Open a PAE/PDE intra-chain versus inter-chain summary line plot."""
        target = self._resolve_plot_target()
        if target is None:
            return

        label = "PAE" if kind == "pae" else "PDE"
        if not self._summary_plot_has_matrix_data(kind, target):
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                f"{label} summary requires {label} data.",
            )
            return
        if not plot_data.has_multiple_token_chains(target.token_map):
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                f"{label} summary requires a target with more than one chain.",
            )
            return

        ref_indices = self._resolve_reference_indices(
            target.token_map, target.obj_name, required=False
        )
        if ref_indices is None:
            return

        try:
            from . import plots

            x_values, series, ylabel = self._compute_summary_plot_data(
                kind, target, ref_indices
            )
            has_finite_values = any(
                np.any(np.isfinite(np.asarray(item[1], dtype=np.float64)))
                for item in series
            )
            if not has_finite_values:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "No finite values are available for this summary plot.",
                )
                return
            indices = list(map(int, x_values.tolist()))
            boundaries, labels = plot_data.chain_boundaries(
                target.token_map, indices, original_x=True
            )
            vmin, vmax = self._get_vmin_vmax()
            title = f"{label} summary ({target.label})"
            fig = plots.make_line_plot(
                x_values,
                series,
                title=title,
                ylabel=ylabel,
                ymin=vmin,
                ymax=vmax,
                chain_boundaries=boundaries,
                chain_labels=labels,
                show_legend=True,
            )
            plots.attach_pymol_selection_metadata(
                fig,
                kind="line",
                token_map=target.token_map,
                obj_name=target.obj_name,
                token_maps=self._plot_selection_token_maps(target),
                token_map_obj_names=self._plot_selection_obj_names(target),
                token_indices=indices,
                x_positions=x_values.tolist(),
            )
            self._show_plot_figure(fig, title)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _show_distribution_plot(self) -> None:
        """Open a quality-class bar plot or continuous-value histogram."""
        target = self._resolve_plot_target()
        if target is None:
            return

        key = self._prop_combo.currentData()
        if metrics.is_domain_label_metric(key) and target.kind == "ensemble_group":
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Distribution plots for PAE domain labels are available for "
                "single models or individual ensemble members. Cluster labels "
                "are member-local and are not pooled across an ensemble.",
            )
            return
        if key == "chain_iptm":
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Distribution plots are not available for chain ipTM.\n"
                "Use Matrix Plot\u2026 for pairwise chain ipTM values.",
            )
            return

        prop = metrics.PROPERTY_BY_KEY.get(key, {})
        ref_indices = self._resolve_reference_indices(
            target.token_map, target.obj_name, required=prop.get("needs_ref", False)
        )
        if ref_indices is None:
            return

        try:
            if target.kind == "single":
                self._ensure_current_data_for_property(prop)
                target.data = self._pred_data
            elif target.kind == "ensemble_member" and target.members:
                self._ensure_member_data_for_property(target.members[0], prop)
                target.data = target.members[0].data

            from . import plots

            x_values, series, _ylabel = self._compute_line_plot_data(
                key, target, ref_indices, plot_type="distribution"
            )
            if not series:
                raise ValueError("No values are available for this distribution.")
            indices = list(map(int, x_values.tolist()))
            values = np.asarray(series[0][1], dtype=np.float64).ravel()
            title = f"{metrics.metric_label(key)} distribution ({target.label})"

            if key == "plddt_class":
                title = f"{metrics.metric_label(key)} distribution\n({target.label})"
                labels, counts, bar_groups, total = (
                    plot_data.plddt_class_distribution_groups(values, indices)
                )
                fig = plots.make_plddt_class_bar_plot(
                    labels,
                    counts,
                    total=total,
                    title=title,
                )
                bar_positions = list(range(len(labels)))
                bar_widths = [0.8 for _label in labels]
            elif metrics.is_domain_label_metric(key):
                title = f"{metrics.metric_label(key)} distribution\n({target.label})"
                labels, counts, bar_groups, colors = (
                    plot_data.domain_label_distribution_groups(values, indices)
                )
                fig = plots.make_categorical_bar_plot(
                    labels,
                    counts,
                    title=title,
                    colors=colors,
                )
                bar_positions = list(range(len(labels)))
                bar_widths = [0.8 for _label in labels]
            else:
                edges, bar_groups, bar_positions, bar_widths = (
                    plot_data.histogram_distribution_groups(values, indices)
                )
                fig = plots.make_histogram_plot(
                    values,
                    title=title,
                    xlabel=metrics.metric_label(key),
                    bin_edges=edges,
                )

            plots.attach_pymol_selection_metadata(
                fig,
                kind="bars",
                token_map=target.token_map,
                obj_name=target.obj_name,
                token_maps=self._plot_selection_token_maps(target),
                token_map_obj_names=self._plot_selection_obj_names(target),
                bar_token_indices=bar_groups,
                bar_x_positions=bar_positions,
                bar_widths=bar_widths,
            )
            self._show_plot_figure(fig, title)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _show_ensemble_site_summary(self) -> None:
        """Open the ensemble ligand-site summary plot."""
        ref_sel = self._ref_edit.text().strip()
        if not ref_sel:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "Ensemble site summary requires a reference selection.\n"
                "Enter a ligand or other PyMOL selection in the Reference field.",
            )
            return
        cutoff = self._get_contact_cutoff()
        if cutoff is None:
            return
        if not self._ensemble_members:
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "The ensemble target is not active.\nUse Load Ensemble\u2026 first.",
            )
            return

        try:
            from . import plots

            members, labels, series, site_indices = (
                self._compute_ensemble_site_summary_data(ref_sel, cutoff)
            )
            if not series:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "No pLDDT, PAE, or PDE data are available for the "
                    "ensemble site summary.",
                )
                return
            title = f"Ensemble site summary\nReference: {ref_sel}, cutoff {cutoff:g} Å"
            fig = plots.make_ensemble_site_summary_plot(
                labels,
                series,
                title=title,
            )
            plots.attach_ensemble_site_summary_metadata(
                fig,
                members=members,
                site_indices=site_indices,
                selection_name="foldqc_ensemble_site",
            )
            self._show_plot_figure(fig, "Ensemble site summary")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _show_matrix_plot(self) -> None:
        """Open a PAE or PDE matrix plot for the selected target/property."""
        target = self._resolve_plot_target()
        if target is None:
            return

        key = self._prop_combo.currentData()
        source = metrics.matrix_source_for_metric(key)
        if source is None:
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Matrix plots are only available when Color by is a PAE, PDE, "
                "interaction probability, or chain ipTM property.",
            )
            return

        attr, _, _ = source
        if attr == "chain_iptm":
            ref_indices = []
        else:
            ref_indices = self._resolve_reference_indices(
                target.token_map, target.obj_name, required=False
            )
            if ref_indices is None:
                return

        try:
            from . import plots

            (
                matrix,
                row_indices,
                col_indices,
                title,
                label,
                row_labels,
                col_labels,
                cell_text,
            ) = self._compute_matrix_plot_data(key, target, ref_indices)
            if attr == "chain_iptm":
                row_boundaries = []
                col_boundaries = []
                xlabel = "Chain j"
                ylabel = "Chain i"
            else:
                row_boundaries, _ = plot_data.chain_boundaries(
                    target.token_map, row_indices
                )
                col_boundaries, _ = plot_data.chain_boundaries(
                    target.token_map, col_indices
                )
                xlabel = "Scored token j"
                ylabel = "Alignment anchor i"
            vmin, vmax = self._get_vmin_vmax()
            palette, reverse_palette = self._selected_palette()
            fig = plots.make_matrix_plot(
                matrix,
                title=f"{title} ({target.label})",
                token_map=target.token_map,
                row_indices=row_indices,
                col_indices=col_indices,
                row_labels=row_labels,
                col_labels=col_labels,
                cell_text=cell_text,
                row_chain_boundaries=row_boundaries,
                col_chain_boundaries=col_boundaries,
                vmin=0.0 if vmin is None else vmin,
                vmax=vmax,
                palette=palette,
                reverse_palette=reverse_palette,
                xlabel=xlabel,
                ylabel=ylabel,
                colorbar_label=label,
            )
            if attr != "chain_iptm":
                plots.attach_pymol_selection_metadata(
                    fig,
                    kind="matrix",
                    token_map=target.token_map,
                    obj_name=target.obj_name,
                    token_maps=self._plot_selection_token_maps(target),
                    token_map_obj_names=self._plot_selection_obj_names(target),
                    row_indices=row_indices,
                    col_indices=col_indices,
                )
            self._show_plot_figure(fig, f"{title} ({target.label})")

        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _show_heatmap(self) -> None:
        """Compatibility wrapper for older callers."""
        self._show_matrix_plot()

    def _show_binding_site_fingerprint(self) -> None:
        """Open a binding-site confidence fingerprint for the current target."""
        ref_sel = self._ref_edit.text().strip()
        if not ref_sel:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "Fingerprint requires a reference selection.\n"
                "Enter a ligand or other PyMOL selection in the Reference field.",
            )
            return

        target = self._resolve_plot_target()
        if target is None:
            return

        ref_indices = self._resolve_reference_indices(
            target.token_map, target.obj_name, required=True
        )
        if ref_indices is None:
            return
        cutoff = self._get_contact_cutoff()
        if cutoff is None:
            return

        try:
            from . import plots

            binding_indices = self._binding_site_token_indices(
                target.token_map, target.obj_name, ref_sel, ref_indices, cutoff
            )
            if not binding_indices:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "No polymer binding-site residues were found within "
                    f"{cutoff:g} Å of the reference selection.",
                )
                return
            series = self._compute_fingerprint_data(target, ref_indices)
            if (
                series["plddt"] is None
                and series["pae_to_ligand"] is None
                and series["pae_from_ligand"] is None
                and series["pde_to_ligand"] is None
                and series["interaction_prob_to_ligand"] is None
            ):
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "No confidence data are available for the fingerprint.",
                )
                return

            if len(binding_indices) > plots.MAX_BINDING_SITE_RESIDUES:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "The binding-site fingerprint found "
                    f"{len(binding_indices)} polymer residues within {cutoff:g} Å "
                    "of the reference selection. Only the first "
                    f"{plots.MAX_BINDING_SITE_RESIDUES} residues in structure "
                    "token order will be shown.",
                )

            title = f"Binding-site confidence fingerprint ({target.label})"
            fig = plots.make_binding_site_fingerprint(
                target.token_map,
                binding_indices,
                plddt=series["plddt"],
                plddt_std=series["plddt_std"],
                pae_to_ligand=series["pae_to_ligand"],
                pae_to_ligand_std=series["pae_to_ligand_std"],
                pae_from_ligand=series["pae_from_ligand"],
                pae_from_ligand_std=series["pae_from_ligand_std"],
                pde_to_ligand=series["pde_to_ligand"],
                pde_to_ligand_std=series["pde_to_ligand_std"],
                interaction_prob_to_ligand=series["interaction_prob_to_ligand"],
                interaction_prob_to_ligand_std=series["interaction_prob_to_ligand_std"],
                title=title,
            )
            displayed_binding_indices = binding_indices[
                : plots.MAX_BINDING_SITE_RESIDUES
            ]
            plots.attach_pymol_selection_metadata(
                fig,
                kind="bars",
                token_map=target.token_map,
                obj_name=target.obj_name,
                token_maps=self._plot_selection_token_maps(target),
                token_map_obj_names=self._plot_selection_obj_names(target),
                token_indices=displayed_binding_indices,
                x_positions=list(range(len(displayed_binding_indices))),
            )
            self._show_plot_figure(fig, title)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, f"{APP_TITLE} - error", str(exc))

    def _plot_selection_token_maps(self, target: _PlotTarget) -> list | None:
        """Return all token maps that plot selections should target."""
        if target.kind == "ensemble_group":
            members = sorted(target.members or [], key=lambda member: member.rank)
            return [member.token_map for member in members]
        return None

    def _plot_selection_obj_names(self, target: _PlotTarget) -> list[str] | None:
        """Return object names corresponding to ensemble plot token maps."""
        if target.kind == "ensemble_group":
            members = sorted(target.members or [], key=lambda member: member.rank)
            names = [getattr(member, "obj_name", None) for member in members]
            if all(names):
                return [str(name) for name in names]
        return None

    def _show_plot_figure(self, fig, title: str) -> None:
        """Show *fig* in an embedded Qt plot window, falling back externally."""
        from . import plots

        try:
            from . import plot_viewer

            def forget_window(dialog) -> None:
                try:
                    self._plot_windows.remove(dialog)
                except ValueError:
                    pass

            if not hasattr(self, "_plot_windows"):
                self._plot_windows = []
            dialog = plot_viewer.show_figure(
                fig, title=title, parent=self, on_close=forget_window
            )
            self._plot_windows.append(dialog)
        except Exception as qt_exc:
            try:
                plots.save_and_show(fig)
            except Exception as external_exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    f"{APP_TITLE} - plot error",
                    (
                        "Could not show the plot in Qt or with the external "
                        "image viewer.\n\n"
                        f"Qt error: {qt_exc}\n\n"
                        f"External viewer error: {external_exc}"
                    ),
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

        checkbox = QtWidgets.QCheckBox("Use current PyMOL coordinates; do not align")
        checkbox.setToolTip(
            "Skip automatic core alignment and compute ensemble RMSD from the current "
            "PyMOL object coordinates."
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

    def _select_object(self, obj_name: str) -> None:
        """Select an object by name in the object combo when present."""
        for i in range(self._obj_combo.count()):
            if self._obj_combo.itemText(i) == obj_name:
                self._obj_combo.setCurrentIndex(i)
                return

    def _combo_item_data(self, combo, row: int):
        """Return combo item data, tolerating minimal fake Qt combos in tests."""
        if hasattr(combo, "itemData"):
            return combo.itemData(row)
        data = getattr(combo, "_data", None)
        if data is not None and 0 <= row < len(data):
            return data[row]
        return None

    def _select_combo_data(self, combo, value) -> bool:
        """Select the first combo row whose item data matches *value*."""
        for i in range(combo.count()):
            if self._combo_item_data(combo, i) == value:
                combo.setCurrentIndex(i)
                return True
        return False

    def _combo_contains_text(self, combo, text: str) -> bool:
        """Return True when a combo has an item with exactly *text*."""
        for i in range(combo.count()):
            if combo.itemText(i) == text:
                return True
        return False

    def _select_model_rank(self, rank: int) -> bool:
        """Select a model rank if it is present in the model combo."""
        return self._select_combo_data(self._model_combo, rank)

    def _select_property(self, key: str) -> None:
        """Select a property combo item by internal key."""
        for i in range(self._prop_combo.count()):
            if self._combo_item_data(self._prop_combo, i) == key:
                self._prop_combo.setCurrentIndex(i)
                return

    def _select_property_if_available(self, key: str) -> bool:
        """Select a property only when the combo row exists and is enabled."""
        row = self._property_combo_row(key, -1)
        if row < 0:
            return False
        item = self._prop_combo.model().item(row)
        if item is not None and not (item.flags() & ItemIsEnabled):
            return False
        self._prop_combo.setCurrentIndex(row)
        return True
