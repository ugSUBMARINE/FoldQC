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
from .gui_application import GuiApplicationServices
from .gui_layout import build_dialog_ui
from .gui_presenter import QtGuiScheduler, QtPresenter
from .gui_state import PluginState, ResolvedTarget
from .gui_view import QtDialogView
from .mol_viewer import PyMOLViewer, get_selection_examples, get_viewer_name

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


class FoldQCPluginDialog(QtWidgets.QDialog):
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

        self._build_ui()
        self._presenter = QtPresenter(self)
        self._scheduler = QtGuiScheduler()
        self._view = QtDialogView(self, self.widgets)

        # Non-widget state is shared with the injected workflow coordinators.
        self.state = PluginState()
        self._viewer = PyMOLViewer()
        self._guide_dialog = None  # Lightweight first-run guide dialog
        if job_runner is None:
            from .gui_jobs import QtJobRunner

            job_runner = QtJobRunner()
        self._job_runner = job_runner
        self.services = GuiApplicationServices(
            self,
            state=self.state,
            viewer=self._viewer,
            presenter=self._presenter,
            view=self._view,
            scheduler=self._scheduler,
            job_runner=self._job_runner,
        )
        self.services.dependencies.initialize()
        self._connect_signals()
        self._restore_session_settings()
        self.services.context.on_property_changed()

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
        self.widgets._prop_combo_rows = {}
        self._populate_property_combo_for(
            self.widgets._prop_combo, self.widgets._prop_combo_rows
        )

    def _populate_property_combo_for(self, combo, rows: dict[str, int]) -> None:
        """Populate a provided metric combo while the widget registry is built."""
        current_group = None
        for spec in metrics.METRICS:
            group = spec.group
            if group != current_group:
                combo.addItem(str(group), None)
                self._disable_combo_row(combo, combo.count() - 1)
                current_group = group
            combo.addItem(metrics.property_combo_label(spec), spec.key)
            rows[spec.key] = combo.count() - 1

    def _disable_combo_row(self, combo, row: int) -> None:
        """Disable one combo row when the backing model item is available."""
        item = combo.model().item(row)
        if item is not None:
            item.setFlags(item.flags() & ~ItemIsEnabled)

    def _connect_signals(self) -> None:
        self.widgets._dir_btn.clicked.connect(self._browse_directory)
        self.widgets._file_btn.clicked.connect(self._browse_file)
        self.widgets._dir_edit.returnPressed.connect(
            self.services.lifecycle.load_prediction
        )
        self.widgets._dir_edit.textChanged.connect(self._save_session_settings)
        self.widgets._model_combo.currentIndexChanged.connect(
            self.services.lifecycle._on_model_changed
        )
        self.widgets._model_combo.currentIndexChanged.connect(
            self._save_session_settings
        )
        self.widgets._obj_refresh_btn.clicked.connect(
            self.services.context._refresh_objects
        )
        self.widgets._obj_combo.currentIndexChanged.connect(
            self.services.context.refresh
        )
        self.widgets._obj_combo.currentIndexChanged.connect(self._save_session_settings)
        self.widgets._prop_combo.currentIndexChanged.connect(
            self.services.context.on_property_changed
        )
        self.widgets._prop_combo.currentIndexChanged.connect(
            self._save_session_settings
        )
        self.widgets._ref_edit.textChanged.connect(self.services.context.refresh)
        self.widgets._ref_edit.textChanged.connect(self._save_session_settings)
        self.widgets._ref_edit.textChanged.connect(self.services.analysis.bump_revision)
        self.widgets._cutoff_edit.textChanged.connect(self.services.context.refresh)
        self.widgets._cutoff_edit.textChanged.connect(self._save_session_settings)
        self.widgets._cutoff_edit.textChanged.connect(
            self.services.analysis.bump_revision
        )
        self.widgets._palette_combo.currentIndexChanged.connect(
            self._save_session_settings
        )
        self.widgets._palette_reverse_chk.stateChanged.connect(
            self._save_session_settings
        )
        self.widgets._vmin_edit.textChanged.connect(self._save_session_settings)
        self.widgets._vmax_edit.textChanged.connect(self._save_session_settings)
        self.widgets._model_combo.currentIndexChanged.connect(
            self.services.analysis.bump_revision
        )
        self.widgets._obj_combo.currentIndexChanged.connect(
            self.services.analysis.bump_revision
        )
        self.widgets._prop_combo.currentIndexChanged.connect(
            self.services.analysis.bump_revision
        )
        self.widgets._palette_combo.currentIndexChanged.connect(
            self.services.analysis.bump_revision
        )
        self.widgets._palette_reverse_chk.stateChanged.connect(
            self.services.analysis.bump_revision
        )
        self.widgets._vmin_edit.textChanged.connect(
            self.services.analysis.bump_revision
        )
        self.widgets._vmax_edit.textChanged.connect(
            self.services.analysis.bump_revision
        )
        self.widgets._guide_btn.clicked.connect(self._show_guide)
        self.widgets._apply_btn.clicked.connect(self.services.analysis.apply_coloring)
        self.widgets._export_csv_btn.clicked.connect(self._export_csv)
        self.widgets._ensemble_btn.clicked.connect(
            self.services.lifecycle.show_ensemble
        )
        self.widgets._close_btn.clicked.connect(self.close)

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
        if self.services.lifecycle.restoring_settings:
            return
        if self.services.lifecycle._gui_job_is_busy():
            return

        try:
            settings = self._settings()
            rank = self.widgets._model_combo.currentData()
            metric_key = self.widgets._prop_combo.currentData()
            palette_key = str(self.widgets._palette_combo.currentData())
            reverse_palette = bool(self.widgets._palette_reverse_chk.isChecked())
            geometry = None
            if hasattr(self, "saveGeometry"):
                geometry = self.saveGeometry()
            state = session.SessionState(
                path=self.widgets._dir_edit.text(),
                model_rank=rank,
                metric_key="" if metric_key is None else metric_key,
                target_name=self.widgets._obj_combo.currentText(),
                reference_text=self.widgets._ref_edit.text(),
                cutoff_text=self.widgets._cutoff_edit.text(),
                palette_key=palette_key,
                palette_reversed=reverse_palette,
                scale_min=self.widgets._vmin_edit.text(),
                scale_max=self.widgets._vmax_edit.text(),
                geometry=geometry,
            )
            session.write_session_state(settings, state)
        except Exception:
            pass

    def _restore_session_settings(self) -> None:
        """Restore saved lightweight GUI state and reload a valid last path."""
        self.services.lifecycle.restoring_settings = True
        try:
            settings = self._settings()
            state = session.read_session_state(settings)
            self.services.lifecycle.pending_session_restore = (
                session.PendingSessionRestore(
                    model_rank=state.model_rank,
                    metric_key=state.metric_key or None,
                    target_name=state.target_name or None,
                )
            )

            self.widgets._dir_edit.setText(state.path)
            self.widgets._ref_edit.setText(state.reference_text)
            if state.cutoff_text:
                self.widgets._cutoff_edit.setText(state.cutoff_text)
            self.widgets._vmin_edit.setText(state.scale_min)
            self.widgets._vmax_edit.setText(state.scale_max)
            self.widgets._palette_reverse_chk.setChecked(state.palette_reversed)
            if state.palette_key:
                self._view.select_combo_data(
                    self.widgets._palette_combo, state.palette_key
                )
            if state.metric_key:
                self._view.select_property_if_available(state.metric_key)

            if state.geometry and hasattr(self, "restoreGeometry"):
                try:
                    self.restoreGeometry(state.geometry)
                except Exception:
                    pass

            if state.path and Path(state.path).exists():
                self.services.lifecycle.load_prediction(state.path)
        finally:
            self.services.lifecycle.restoring_settings = False

    def closeEvent(self, event) -> None:
        """Persist session state when the dialog closes."""
        if self.services.dependencies._dependency_close_is_blocked(event):
            return
        if self.services.lifecycle._gui_job_is_busy():
            self.services.lifecycle._abandon_active_gui_job()
        else:
            self._save_session_settings()
        pred_files = self.state.pred_files
        self.state.pred_files = None
        if pred_files is not None:
            self._job_runner.dispose(pred_files)
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
            self.widgets._dir_edit.text() or str(Path.home()),
        )
        if path:
            self.widgets._dir_edit.setText(path)
            self.services.lifecycle.load_prediction(path)
        else:
            self._raise_after_native_dialog()

    def _browse_file(self) -> None:
        """Select a single CIF/PDB structure file or prediction archive."""
        result = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select predicted structure file or prediction archive",
            self.widgets._dir_edit.text() or str(Path.home()),
            PREDICTION_FILE_FILTER,
        )
        path = result[0] if isinstance(result, tuple) else result
        if path:
            self.widgets._dir_edit.setText(path)
            self.services.lifecycle.load_prediction(path)
        else:
            self._raise_after_native_dialog()

    def _export_csv(self) -> None:
        """Capture a native save path before submitting the export request."""
        default_path = self.services.analysis.default_export_path()
        if default_path is None:
            return
        result = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export token metric CSV",
            default_path,
            "CSV files (*.csv);;All files (*)",
        )
        path = result[0] if isinstance(result, tuple) else result
        if path:
            self.services.analysis.export_csv(path)
        else:
            self._raise_after_native_dialog()

    def _raise_after_native_dialog(self) -> None:
        """Bring this dialog back after a native file dialog returns focus."""
        self.raise_()
        self.activateWindow()
