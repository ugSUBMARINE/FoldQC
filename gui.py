"""
GUI
===
Main Qt dialog for the FoldQC molecular-viewer plugin.

All Qt imports go through :mod:`compat` to handle Qt5/Qt6 differences.
"""

from __future__ import annotations

from pathlib import Path

from . import metrics
from .analysis import (
    AnalysisAction,
    AnalysisRequest,
    ColorOptions,
    DeferredAnalysisAction,
    ExportOptions,
    PlotOptions,
)
from .compat import (
    AlignLeft,
    AlignVCenter,
    ItemIsEnabled,
    QtWidgets,
)
from .gui_application import GuiApplicationServices
from .gui_dependencies import QtDependencyService
from .gui_layout import build_dialog_ui
from .gui_presenter import QtGuiScheduler, QtPresenter
from .gui_services import ContextSelection
from .gui_session import QtSessionAdapter
from .gui_state import PluginState
from .gui_view import QtDialogView
from .mol_viewer import PyMOLViewer, get_selection_examples, get_viewer_name
from .presentation import Notice

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()
SELECTION_EXAMPLES = get_selection_examples()
PREDICTION_FILE_FILTER = (
    "Prediction files (*.cif *.pdb *.zip *.tar *.tar.gz *.tgz);;All files (*)"
)


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
        self._shutdown_complete = False
        self.setWindowTitle(APP_TITLE)
        self.setMinimumWidth(480)

        self._build_ui()
        self._presenter = QtPresenter(self)
        self._scheduler = QtGuiScheduler()
        self._view = QtDialogView(self, self.widgets)
        self._session = QtSessionAdapter(self, self.widgets)
        self._dependencies = QtDependencyService(self)

        # Non-widget state is shared with the injected workflow coordinators.
        self.state = PluginState()
        self._viewer = PyMOLViewer()
        self._guide_dialog = None  # Lightweight first-run guide dialog
        if job_runner is None:
            from .gui_jobs import QtJobRunner

            job_runner = QtJobRunner()
        self._job_runner = job_runner
        self.services = GuiApplicationServices(
            state=self.state,
            viewer=self._viewer,
            presenter=self._presenter,
            view=self._view,
            scheduler=self._scheduler,
            job_runner=self._job_runner,
            session=self._session,
            dependencies=self._dependencies,
            metric_rows=self.widgets._prop_combo_rows,
        )
        self._connect_signals()
        self._restore_session_settings()
        self.services.context.refresh(self._capture_context_selection())
        self._connect_shutdown()

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
        self.widgets._dir_edit.returnPressed.connect(self._load_entered_prediction)
        self.widgets._dir_edit.textChanged.connect(self._save_session_settings)
        self.widgets._model_combo.currentIndexChanged.connect(self._model_changed)
        self.widgets._model_combo.currentIndexChanged.connect(
            self._save_session_settings
        )
        self.widgets._obj_refresh_btn.clicked.connect(
            self.services.context.refresh_objects
        )
        self.widgets._obj_combo.currentIndexChanged.connect(self._context_changed)
        self.widgets._obj_combo.currentIndexChanged.connect(self._save_session_settings)
        self.widgets._prop_combo.currentIndexChanged.connect(self._context_changed)
        self.widgets._prop_combo.currentIndexChanged.connect(
            self._save_session_settings
        )
        self.widgets._ref_edit.textChanged.connect(self._context_changed)
        self.widgets._ref_edit.textChanged.connect(self._save_session_settings)
        self.widgets._cutoff_edit.textChanged.connect(self._context_changed)
        self.widgets._cutoff_edit.textChanged.connect(self._save_session_settings)
        self.widgets._palette_combo.currentIndexChanged.connect(
            self._save_session_settings
        )
        self.widgets._palette_reverse_chk.stateChanged.connect(
            self._save_session_settings
        )
        self.widgets._vmin_edit.textChanged.connect(self._save_session_settings)
        self.widgets._vmax_edit.textChanged.connect(self._save_session_settings)
        self.widgets._palette_combo.currentIndexChanged.connect(
            self.services.analysis.invalidate_ui
        )
        self.widgets._palette_reverse_chk.stateChanged.connect(
            self.services.analysis.invalidate_ui
        )
        self.widgets._vmin_edit.textChanged.connect(
            self.services.analysis.invalidate_ui
        )
        self.widgets._vmax_edit.textChanged.connect(
            self.services.analysis.invalidate_ui
        )
        for plot_type, action in self.widgets._plot_actions.items():
            action.triggered.connect(
                lambda _checked=False, key=plot_type: self._submit_plot(key)
            )
        self.widgets._guide_btn.clicked.connect(self._show_guide)
        self.widgets._apply_btn.clicked.connect(self._apply_coloring)
        self.widgets._export_csv_btn.clicked.connect(self._export_csv)
        self.widgets._ensemble_btn.clicked.connect(self._activate_ensemble)
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

    def _save_session_settings(self, *_args) -> None:
        if not self.services.operations.is_busy:
            try:
                self._session.save()
            except Exception:
                pass

    def _restore_session_settings(self) -> None:
        """Restore saved lightweight GUI state and reload a valid last path."""
        self._session.set_restoring(True)
        try:
            state = self._session.restore()

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
                self.services.context.set_selection(self._capture_context_selection())
                self.services.lifecycle.load_prediction(
                    state.path,
                    preferred_rank=state.model_rank,
                    preferred_target=state.target_name,
                )
        finally:
            self._session.set_restoring(False)

    def _connect_shutdown(self) -> None:
        """Release session-owned resources only when the Qt application exits."""
        application = QtWidgets.QApplication.instance()
        about_to_quit = getattr(application, "aboutToQuit", None)
        if about_to_quit is not None:
            about_to_quit.connect(self.shutdown)

    def shutdown(self) -> None:
        """Persist lightweight state and release resources at PyMOL shutdown."""
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        if not self.services.operations.is_busy:
            self._save_session_settings()
        self.services.close()

    def closeEvent(self, event) -> None:
        """Hide the dialog while retaining its in-session state and services."""
        event.ignore()
        self.hide()

    # -----------------------------------------------------------------------
    # Slots
    # -----------------------------------------------------------------------

    def _capture_context_selection(self) -> ContextSelection:
        metric = self.widgets._prop_combo.currentData()
        return ContextSelection(
            target_name=self.widgets._obj_combo.currentText().strip(),
            metric_key=None if metric is None else str(metric),
            reference_selection=self.widgets._ref_edit.text().strip(),
            cutoff_text=self.widgets._cutoff_edit.text().strip(),
        )

    @staticmethod
    def _optional_float(text: str, *, label: str) -> float | None:
        stripped = text.strip()
        if not stripped or stripped.lower() == "auto":
            return None
        try:
            return float(stripped)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number or 'auto'.") from exc

    def _capture_action(
        self,
        action: AnalysisAction,
        *,
        export_path: Path | None = None,
    ) -> DeferredAnalysisAction:
        request = self._capture_request(action)
        plot = metrics.PLOTS.find(action)
        if action == "color":
            options: ColorOptions | PlotOptions | ExportOptions = ColorOptions(
                str(self.widgets._palette_combo.currentData()),
                bool(self.widgets._palette_reverse_chk.isChecked()),
                self._optional_float(self.widgets._vmin_edit.text(), label="Minimum"),
                self._optional_float(self.widgets._vmax_edit.text(), label="Maximum"),
            )
        elif plot is not None:
            options = PlotOptions(
                str(self.widgets._palette_combo.currentData()),
                bool(self.widgets._palette_reverse_chk.isChecked()),
                self._optional_float(self.widgets._vmin_edit.text(), label="Minimum"),
                self._optional_float(self.widgets._vmax_edit.text(), label="Maximum"),
            )
        else:
            if export_path is None:
                raise ValueError("Export requires an output path.")
            options = ExportOptions(export_path)
        return DeferredAnalysisAction(request, options)

    def _capture_request(self, action: AnalysisAction) -> AnalysisRequest:
        selection = self._capture_context_selection()
        plot = metrics.PLOTS.find(action)
        metric_key = selection.metric_key
        if plot is not None and not plot.requires_metric:
            metric_key = None
        metric = None if metric_key is None else metrics.METRICS.require(metric_key)
        needs_cutoff = bool(
            (metric is not None and metric.needs_cutoff)
            or action in {"binding_site_fingerprint", "ensemble_site_summary"}
        )
        cutoff = None
        if needs_cutoff:
            cutoff = self._optional_float(
                selection.cutoff_text or "5.0", label="Cutoff"
            )
            if cutoff is None or cutoff <= 0:
                raise ValueError("Cutoff / threshold must be greater than 0 Å.")
        return AnalysisRequest(
            action,
            selection.target_name,
            metric_key,
            selection.reference_selection,
            cutoff,
            self.services.analysis.ui_revision,
        )

    def _context_changed(self, *_args) -> None:
        self.services.analysis.invalidate_ui()
        self.services.context.refresh(self._capture_context_selection())

    def _model_changed(self, *_args) -> None:
        self.services.analysis.invalidate_ui()
        self.services.context.set_selection(self._capture_context_selection())
        rank = self.widgets._model_combo.currentData()
        if rank is not None:
            self.services.lifecycle.select_model(int(rank))

    def _load_entered_prediction(self) -> None:
        self.services.lifecycle.load_prediction(self.widgets._dir_edit.text())

    def _apply_coloring(self) -> None:
        try:
            self.services.analysis.submit(self._capture_action("color"))
        except ValueError as exc:
            self._presenter.present_notice(Notice("color_preflight", str(exc)))

    def _submit_plot(self, plot_type: AnalysisAction) -> None:
        try:
            self.services.analysis.submit(self._capture_action(plot_type))
        except ValueError as exc:
            self._presenter.present_notice(Notice("plot_preflight", str(exc)))

    def _activate_ensemble(self) -> None:
        self.services.ensemble.activate(self.widgets._obj_combo.currentText().strip())

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
        try:
            request = self._capture_request("export")
        except ValueError as exc:
            self._presenter.present_notice(Notice("export_preflight", str(exc)))
            return
        default_path = self.services.analysis.default_export_path(request)
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
            self.services.analysis.submit(
                DeferredAnalysisAction(request, ExportOptions(Path(path)))
            )
        else:
            self._raise_after_native_dialog()

    def _raise_after_native_dialog(self) -> None:
        """Bring this dialog back after a native file dialog returns focus."""
        self.raise_()
        self.activateWindow()
