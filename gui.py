"""
GUI
===
Main Qt dialog for the FoldQC molecular-viewer plugin.

All Qt imports go through :mod:`compat` to handle Qt5/Qt6 differences.
"""

from __future__ import annotations

from pathlib import Path

from . import metrics, session
from .compat import (
    AlignLeft,
    AlignVCenter,
    ItemIsEnabled,
    QSettings,
    QtWidgets,
)
from .gui_coloring import ColoringController
from .gui_export import ExportController
from .gui_layout import build_dialog_ui
from .gui_loading import GuiLoadingController
from .gui_metrics import MetricController
from .gui_plots import PlotController
from .gui_state import GuiState, GuiStateBacked, ResolvedTarget
from .mol_viewer import get_selection_examples, get_viewer_name

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()
SELECTION_EXAMPLES = get_selection_examples()
PREDICTION_FILE_FILTER = (
    "Prediction files (*.cif *.pdb *.zip *.tar *.tar.gz *.tgz);;All files (*)"
)


_PlotTarget = ResolvedTarget

# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class FoldQCPluginDialog(
    GuiLoadingController,
    ColoringController,
    ExportController,
    MetricController,
    PlotController,
    GuiStateBacked,
    QtWidgets.QDialog,
):
    """Main plugin window."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        job_runner=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(APP_TITLE)
        self.setMinimumWidth(480)

        # Non-widget state is shared with the GUI-side workflow coordinators.
        self._state = GuiState()
        self._plot_windows = []  # Qt plot dialogs kept alive while visible
        self._guide_dialog = None  # Lightweight first-run guide dialog
        if job_runner is None:
            from .gui_jobs import QtJobRunner

            job_runner = QtJobRunner()
        self._job_runner = job_runner
        self._active_load_handle = None
        self._active_data_continuation = None
        self._active_data_error_title = f"{APP_TITLE} - error"
        self._model_switch_previous_data = None
        self._model_switch_previous_token_context = None
        self._active_ensemble_viewer_transaction = None
        self._load_progress_dialog = None
        self._progress_show_generation = 0

        self._build_ui()
        self._connect_signals()
        self._restore_session_settings()
        self._on_property_changed()  # set initial reference-field visibility

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.widgets = build_dialog_ui(self)

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
            f"Reference examples: {SELECTION_EXAMPLES['ligand']}\n"
            "Use: PAE - contact-filtered to selection or "
            "PDE - contact-filtered to selection\n"
            "Plot: Binding-site fingerprint\n\n"
            "3. Chain or interface placement\n"
            f"Reference example: {SELECTION_EXAMPLES['chain']}\n"
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
        if self._gui_job_is_busy():
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
        if self._gui_job_is_busy():
            self._abandon_active_gui_job()
        else:
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

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _select_object(self, obj_name: str) -> None:
        """Select an object by name in the object combo when present."""
        for i in range(self._obj_combo.count()):
            if self._obj_combo.itemText(i) == obj_name:
                self._obj_combo.setCurrentIndex(i)
                return

    def _select_combo_data(self, combo, value) -> bool:
        """Select the first combo row whose item data matches *value*."""
        for i in range(combo.count()):
            if combo.itemData(i) == value:
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
            if self._prop_combo.itemData(i) == key:
                self._prop_combo.setCurrentIndex(i)
                return

    def _select_property_if_available(self, key: str) -> bool:
        """Select a property only when the combo row exists and is enabled."""
        row = self._property_combo_row(key)
        if row is None:
            return False
        item = self._prop_combo.model().item(row)
        if item is not None and not (item.flags() & ItemIsEnabled):
            return False
        self._prop_combo.setCurrentIndex(row)
        return True
