"""Optional-dependency prompts and asynchronous installation coordination."""

from __future__ import annotations

import importlib
import sys

from . import dependencies
from .compat import (
    MessageBoxButtonRole,
    MessageBoxIcon,
    QProcessError,
    QtCore,
    QtWidgets,
    WindowCloseButtonHint,
)

APP_TITLE = "FoldQC"


class DependencyWorkflow:
    """GUI-side optional dependency installation workflow."""

    def _initialize_dependency_controller(self) -> None:
        self._dependency_process = None
        self._dependency_process_phase = None
        self._dependency_process_error = ""
        self._dependency_install_active = False
        self._dependency_install_keys: tuple[str, ...] = ()
        self._dependency_progress_dialog = None
        self._dependency_progress_label = None
        self._dependency_progress_bar = None
        self._dependency_progress_log = None
        self._dependency_progress_details_button = None
        self._dependency_progress_close_button = None
        self._dependency_log_chunks: list[str] = []

    def _ensure_dependencies(self, dependency_keys, *, feature_label: str) -> bool:
        """Return whether explicit dependencies are present and usable now."""
        required = dependencies.required_dependency_keys(dependency_keys)
        if not required:
            return True

        missing = dependencies.missing_dependency_keys(required)
        if not missing:
            return True
        if getattr(self, "_dependency_install_active", False):
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "A dependency installation is already in progress.",
            )
            return False

        action = self._prompt_dependency_action(missing, feature_label=feature_label)
        if action == "install":
            self._start_dependency_install(missing)
        elif action == "manual":
            self._dependency_log_chunks = []
            self._show_manual_dependency_instructions(
                missing,
                reason=f"Install the missing packages before using {feature_label}.",
            )
        return False

    def _dependency_display_names(self, dependency_keys) -> str:
        names = [
            dependency.display_name
            for dependency in dependencies.dependency_specs(dependency_keys)
        ]
        if len(names) < 2:
            return names[0] if names else "The dependencies"
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{', '.join(names[:-1])}, and {names[-1]}"

    def _prompt_dependency_action(self, dependency_keys, *, feature_label: str) -> str:
        names = self._dependency_display_names(dependency_keys)
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(APP_TITLE)
        if hasattr(box, "setIcon"):
            box.setIcon(MessageBoxIcon.Information)
        box.setText(f"{feature_label} requires {names}.")
        box.setInformativeText(
            "FoldQC can install the missing package(s) into the Python environment "
            "used by PyMOL. A PyMOL restart may be necessary afterward."
        )
        install_button = box.addButton("Install", MessageBoxButtonRole.AcceptRole)
        manual_button = box.addButton(
            "Manual instructions", MessageBoxButtonRole.ActionRole
        )
        cancel_button = box.addButton("Cancel", MessageBoxButtonRole.RejectRole)
        box.setDefaultButton(install_button)
        box.setEscapeButton(cancel_button)
        box.exec()
        clicked = box.clickedButton()
        if clicked is install_button:
            return "install"
        if clicked is manual_button:
            return "manual"
        return "cancel"

    def _start_dependency_install(self, dependency_keys) -> None:
        keys = tuple(dependency_keys)
        self._dependency_install_keys = keys
        self._dependency_log_chunks = []
        if not dependencies.environment_is_writable():
            self._show_manual_dependency_instructions(
                keys,
                reason=(
                    "PyMOL's Python environment does not appear to be writable, "
                    "so FoldQC did not start pip."
                ),
            )
            return

        self._dependency_install_active = True
        self._ensure_dependency_progress_dialog()
        self._set_dependency_progress_phase(
            f"Installing {self._dependency_display_names(keys)}…"
        )
        self._start_dependency_process(
            sys.executable,
            dependencies.pip_install_args(keys),
            phase="install",
        )

    def _ensure_dependency_progress_dialog(self):
        old_dialog = getattr(self, "_dependency_progress_dialog", None)
        if old_dialog is not None:
            old_dialog.close()

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"{APP_TITLE} – Installing dependencies")
        dialog.setModal(False)
        if hasattr(dialog, "setMinimumWidth"):
            dialog.setMinimumWidth(580)
        if hasattr(dialog, "setWindowFlag"):
            dialog.setWindowFlag(WindowCloseButtonHint, False)

        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel(dialog)
        label.setWordWrap(True)
        layout.addWidget(label)

        progress = QtWidgets.QProgressBar(dialog)
        progress.setRange(0, 0)
        layout.addWidget(progress)

        details_button = QtWidgets.QPushButton("Show details", dialog)
        details_button.setAutoDefault(False)
        details_button.setDefault(False)
        layout.addWidget(details_button)

        log = QtWidgets.QPlainTextEdit(dialog)
        log.setReadOnly(True)
        log.setVisible(False)
        if hasattr(log, "setMinimumHeight"):
            log.setMinimumHeight(180)
        layout.addWidget(log)

        close_button = QtWidgets.QPushButton("Close", dialog)
        close_button.setAutoDefault(False)
        close_button.setDefault(False)
        close_button.setEnabled(False)
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        details_button.clicked.connect(self._toggle_dependency_log)
        dialog.show()
        if hasattr(dialog, "raise_"):
            dialog.raise_()

        self._dependency_progress_dialog = dialog
        self._dependency_progress_label = label
        self._dependency_progress_bar = progress
        self._dependency_progress_log = log
        self._dependency_progress_details_button = details_button
        self._dependency_progress_close_button = close_button
        return dialog

    def _toggle_dependency_log(self) -> None:
        log = getattr(self, "_dependency_progress_log", None)
        button = getattr(self, "_dependency_progress_details_button", None)
        if log is None or button is None:
            return
        visible = not log.isVisible()
        log.setVisible(visible)
        button.setText("Hide details" if visible else "Show details")

    def _set_dependency_progress_phase(self, text: str) -> None:
        label = getattr(self, "_dependency_progress_label", None)
        if label is not None:
            label.setText(text)

    def _start_dependency_process(
        self, program: str, arguments: list[str], *, phase: str
    ):
        process = QtCore.QProcess(self)
        self._dependency_process = process
        self._dependency_process_phase = phase
        self._dependency_process_error = ""
        process.readyReadStandardOutput.connect(
            lambda process=process: self._read_dependency_process_output(
                process, standard_error=False
            )
        )
        process.readyReadStandardError.connect(
            lambda process=process: self._read_dependency_process_output(
                process, standard_error=True
            )
        )
        process.errorOccurred.connect(
            lambda error, process=process: self._on_dependency_process_error(
                process, error
            )
        )
        process.finished.connect(
            lambda exit_code, _exit_status=None, process=process: (
                self._on_dependency_process_finished(process, int(exit_code))
            )
        )
        process.start(program, arguments)

    def _read_dependency_process_output(self, process, *, standard_error: bool) -> None:
        if process is not getattr(self, "_dependency_process", None):
            return
        raw = (
            process.readAllStandardError()
            if standard_error
            else process.readAllStandardOutput()
        )
        text = bytes(raw).decode("utf-8", errors="replace")
        if not text:
            return
        self._dependency_log_chunks.append(text)
        log = getattr(self, "_dependency_progress_log", None)
        if log is not None:
            log.setPlainText("".join(self._dependency_log_chunks))

    def _on_dependency_process_error(self, process, error) -> None:
        if process is not getattr(self, "_dependency_process", None):
            return
        self._dependency_process_error = process.errorString()
        failed_to_start = getattr(QProcessError, "FailedToStart", None)
        if failed_to_start is not None and error == failed_to_start:
            self._finish_dependency_failure(
                f"Could not start PyMOL's Python interpreter: {process.errorString()}"
            )

    def _on_dependency_process_finished(self, process, exit_code: int) -> None:
        if process is not getattr(self, "_dependency_process", None):
            return
        self._read_dependency_process_output(process, standard_error=False)
        self._read_dependency_process_output(process, standard_error=True)
        phase = getattr(self, "_dependency_process_phase", None)
        error_text = getattr(self, "_dependency_process_error", "")
        self._dependency_process = None
        if hasattr(process, "deleteLater"):
            process.deleteLater()

        if exit_code != 0:
            detail = (
                error_text
                or f"The {phase or 'dependency'} process exited with code {exit_code}."
            )
            self._finish_dependency_failure(detail)
            return

        if phase == "install":
            self._set_dependency_progress_phase("Verifying the installed packages…")
            QtCore.QTimer.singleShot(0, self._start_dependency_validation)
            return
        if phase == "validate":
            self._finish_dependency_success()

    def _start_dependency_validation(self) -> None:
        if not getattr(self, "_dependency_install_active", False):
            return
        self._start_dependency_process(
            sys.executable,
            dependencies.validation_args(self._dependency_install_keys),
            phase="validate",
        )

    def _finish_dependency_success(self) -> None:
        self._dependency_install_active = False
        importlib.invalidate_caches()
        self._set_dependency_progress_phase(
            "Installation verified. Retry the requested action. A PyMOL restart may "
            "be necessary if the feature is not immediately available."
        )
        progress = getattr(self, "_dependency_progress_bar", None)
        if progress is not None:
            progress.setRange(0, 1)
            progress.setValue(1)
        close_button = getattr(self, "_dependency_progress_close_button", None)
        if close_button is not None:
            close_button.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self,
            APP_TITLE,
            "The missing dependencies were installed and verified. You can retry "
            "the requested action now. A PyMOL restart may be necessary if the "
            "feature is not immediately available.",
        )

    def _finish_dependency_failure(self, reason: str) -> None:
        if not getattr(self, "_dependency_install_active", False):
            return
        keys = tuple(getattr(self, "_dependency_install_keys", ()))
        self._dependency_install_active = False
        process = self._dependency_process
        self._dependency_process = None
        if process is not None and hasattr(process, "deleteLater"):
            process.deleteLater()
        self._set_dependency_progress_phase("Dependency installation failed.")
        progress = getattr(self, "_dependency_progress_bar", None)
        if progress is not None:
            progress.setRange(0, 1)
            progress.setValue(0)
        close_button = getattr(self, "_dependency_progress_close_button", None)
        if close_button is not None:
            close_button.setEnabled(True)
        self._show_manual_dependency_instructions(keys, reason=reason)

    def _show_manual_dependency_instructions(
        self, dependency_keys, *, reason: str
    ) -> None:
        instructions = dependencies.manual_install_instructions(dependency_keys)
        log_text = "".join(getattr(self, "_dependency_log_chunks", ())).strip()
        details = instructions
        if log_text:
            details = f"{instructions}\n\nInstaller log:\n{log_text}"

        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(f"{APP_TITLE} – Manual dependency installation")
        if hasattr(box, "setIcon"):
            box.setIcon(MessageBoxIcon.Warning)
        box.setText(reason)
        box.setInformativeText(
            f"Python interpreter: {sys.executable}\nEnvironment prefix: {sys.prefix}\n\n"
            "Use one of the commands in Show Details, then retry the feature. A "
            "PyMOL restart may be necessary if it is not immediately available. "
            "If shown, the pip --user alternative is not run automatically because "
            "it can mix packages across conda environments."
        )
        box.setDetailedText(details)
        box.exec()

    def _dependency_close_is_blocked(self, event) -> bool:
        if not getattr(self, "_dependency_install_active", False):
            return False
        QtWidgets.QMessageBox.information(
            self,
            APP_TITLE,
            "Dependency installation is still running. Wait for it to finish before "
            "closing FoldQC.",
        )
        if hasattr(event, "ignore"):
            event.ignore()
        return True
