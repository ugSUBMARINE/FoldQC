"""Qt implementation of the presentation port used by GUI services."""

from __future__ import annotations

from .analysis import AnalysisProblem
from .compat import MessageBoxStandardButton, QtWidgets


class QtPresenter:
    def __init__(self, dialog) -> None:
        self.dialog = dialog

    def present_problem(self, problem: AnalysisProblem) -> None:
        title = "FoldQC" if problem.severity != "error" else "FoldQC - error"
        method = {
            "information": QtWidgets.QMessageBox.information,
            "warning": QtWidgets.QMessageBox.warning,
            "error": QtWidgets.QMessageBox.critical,
        }[problem.severity]
        method(self.dialog, title, problem.message)

    def confirm(self, title: str, message: str) -> bool:
        buttons = MessageBoxStandardButton.Yes | MessageBoxStandardButton.Cancel
        return (
            QtWidgets.QMessageBox.question(
                self.dialog,
                title,
                message,
                buttons,
                MessageBoxStandardButton.Cancel,
            )
            == MessageBoxStandardButton.Yes
        )

    def set_progress(self, label: str | None) -> None:
        progress = getattr(self.dialog, "_load_progress_dialog", None)
        if progress is not None:
            if label is None:
                progress.hide()
            else:
                progress.setLabelText(label)

    def set_window_title(self, title: str) -> None:
        self.dialog.setWindowTitle(title)

    def show_statistics(self, text: str) -> None:
        self.dialog._stats_browser.setPlainText(text)

    def show_plot(self, figure: object, title: str) -> None:
        self.dialog._show_plot_figure(figure, title)
