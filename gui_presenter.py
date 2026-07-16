"""Qt implementation of FoldQC's presentation and scheduling ports."""

from __future__ import annotations

from collections.abc import Callable

from .compat import (
    AbstractItemViewNoEditTriggers,
    AbstractItemViewSelectRows,
    AbstractItemViewSingleSelection,
    MessageBoxButtonRole,
    MessageBoxStandardButton,
    QtCore,
    QtWidgets,
    WindowCloseButtonHint,
)
from .presentation import (
    ChoiceRequest,
    ModelComparisonRequest,
    Notice,
    PreparedPlot,
    ProgressRequest,
    SelectionRequest,
)


class QtPresenter:
    def __init__(self, dialog) -> None:
        self.dialog = dialog
        self._progress: dict[str, object] = {}
        self._progress_generation: dict[str, int] = {}
        self._plot_windows: list[object] = []

    def present_notice(self, notice: Notice) -> None:
        title = notice.title
        if notice.severity == "error" and title == "FoldQC":
            title = "FoldQC - error"
        method = {
            "information": QtWidgets.QMessageBox.information,
            "warning": QtWidgets.QMessageBox.warning,
            "error": QtWidgets.QMessageBox.critical,
        }[notice.severity]
        method(self.dialog, title, notice.message)

    def choose(self, request: ChoiceRequest) -> str | None:
        if tuple(option.key for option in request.options) == ("yes", "cancel"):
            buttons = MessageBoxStandardButton.Yes | MessageBoxStandardButton.Cancel
            result = QtWidgets.QMessageBox.question(
                self.dialog,
                request.title,
                request.message,
                buttons,
                MessageBoxStandardButton.Cancel,
            )
            return "yes" if result == MessageBoxStandardButton.Yes else "cancel"
        box = QtWidgets.QMessageBox(self.dialog)
        box.setWindowTitle(request.title)
        box.setText(request.message)
        buttons = {}
        role_by_name = {
            "accept": MessageBoxButtonRole.AcceptRole,
            "reject": MessageBoxButtonRole.RejectRole,
            "destructive": getattr(
                MessageBoxButtonRole,
                "DestructiveRole",
                MessageBoxButtonRole.RejectRole,
            ),
            "help": getattr(
                MessageBoxButtonRole, "HelpRole", MessageBoxButtonRole.AcceptRole
            ),
        }
        for option in request.options:
            button = box.addButton(option.label, role_by_name[option.role])
            buttons[button] = option.key
            if option.key == request.default_key:
                box.setDefaultButton(button)
        box.exec()
        return buttons.get(box.clickedButton())

    def select_item(self, request: SelectionRequest) -> str | None:
        labels = [
            item.label if not item.description else f"{item.label} — {item.description}"
            for item in request.items
        ]
        default = 0
        if request.default_key is not None:
            default = next(
                index
                for index, item in enumerate(request.items)
                if item.key == request.default_key
            )
        label, accepted = QtWidgets.QInputDialog.getItem(
            self.dialog, request.title, request.message, labels, default, False
        )
        if not accepted:
            return None
        index = labels.index(str(label))
        return request.items[index].key

    def select_comparison_model(self, request: ModelComparisonRequest) -> int | None:
        dialog = QtWidgets.QDialog(self.dialog)
        dialog.setWindowTitle(request.title)
        dialog.setModal(True)

        layout = QtWidgets.QVBoxLayout(dialog)
        description = QtWidgets.QLabel(
            f"{request.provider_label} scalar confidence summaries. "
            "Select a row to switch to that model."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        table = QtWidgets.QTableWidget(
            len(request.rows), len(request.columns) + 1, dialog
        )
        table.setHorizontalHeaderLabels(
            ["Model", *(column.label for column in request.columns)]
        )
        table.setEditTriggers(AbstractItemViewNoEditTriggers)
        table.setSelectionBehavior(AbstractItemViewSelectRows)
        table.setSelectionMode(AbstractItemViewSingleSelection)
        table.setAlternatingRowColors(True)
        for row_index, row in enumerate(request.rows):
            table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(row.label))
            for column_index, value in enumerate(row.values, start=1):
                table.setItem(
                    row_index,
                    column_index,
                    QtWidgets.QTableWidgetItem(value),
                )
            if row.rank == request.selected_rank:
                table.setCurrentCell(row_index, 0)
        if table.currentRow() < 0:
            table.setCurrentCell(0, 0)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        use_button = QtWidgets.QPushButton("Use selected model")
        use_button.setAutoDefault(True)
        use_button.setDefault(True)
        close_button = QtWidgets.QPushButton("Close")
        close_button.setAutoDefault(False)
        buttons.addWidget(use_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        use_button.clicked.connect(dialog.accept)
        close_button.clicked.connect(dialog.reject)
        table.doubleClicked.connect(lambda _index: dialog.accept())

        header_width = table.horizontalHeader().length()
        dialog.resize(min(max(header_width + 70, 560), 1100), 300)
        accepted = bool(dialog.exec())
        row_index = table.currentRow()
        if not accepted or row_index < 0:
            return None
        return request.rows[row_index].rank

    def start_progress(
        self,
        request: ProgressRequest,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self.finish_progress(request.operation_id)
        dialog = QtWidgets.QProgressDialog(self.dialog)
        dialog.setWindowTitle(request.title)
        dialog.setLabelText(request.label)
        dialog.setModal(False)
        dialog.setRange(0, 0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        if hasattr(dialog, "setWindowFlag"):
            dialog.setWindowFlag(WindowCloseButtonHint, False)
        if not request.cancellable:
            dialog.setCancelButton(None)
        elif on_cancel is not None:
            dialog.canceled.connect(on_cancel)
        self._progress[request.operation_id] = dialog
        generation = self._progress_generation.get(request.operation_id, 0) + 1
        self._progress_generation[request.operation_id] = generation

        def show_if_current() -> None:
            if (
                self._progress.get(request.operation_id) is dialog
                and self._progress_generation.get(request.operation_id) == generation
            ):
                dialog.show()

        if request.delay_ms:
            QtCore.QTimer.singleShot(request.delay_ms, show_if_current)
        else:
            show_if_current()

    def update_progress(self, operation_id: str, label: str) -> None:
        dialog = self._progress.get(operation_id)
        if dialog is not None:
            dialog.setLabelText(label)

    def finish_progress(self, operation_id: str) -> None:
        self._progress_generation[operation_id] = (
            self._progress_generation.get(operation_id, 0) + 1
        )
        dialog = self._progress.pop(operation_id, None)
        if dialog is not None:
            dialog.hide()
            if hasattr(dialog, "deleteLater"):
                dialog.deleteLater()

    def show_statistics(self, text: str) -> None:
        self.dialog.widgets._stats_browser.setPlainText(text)

    def show_plot(self, prepared: PreparedPlot) -> None:
        from . import plots

        figure = prepared.figure
        title = prepared.title

        try:
            from . import plot_viewer

            def forget_window(dialog) -> None:
                try:
                    self._plot_windows.remove(dialog)
                except ValueError:
                    pass

            dialog = plot_viewer.show_figure(
                figure,
                title=title,
                parent=self.dialog,
                on_close=forget_window,
            )
            self._plot_windows.append(dialog)
        except Exception as qt_exc:
            try:
                plots.save_and_show(figure)
            except Exception as external_exc:
                self.present_notice(
                    Notice(
                        "plot_display_failed",
                        "Could not show the plot in Qt or with the external "
                        "image viewer.\n\n"
                        f"Qt error: {qt_exc}\n\n"
                        f"External viewer error: {external_exc}",
                        severity="error",
                        title="FoldQC - plot error",
                    )
                )

    def close(self) -> None:
        for operation_id in tuple(self._progress):
            self.finish_progress(operation_id)
        for window in tuple(self._plot_windows):
            try:
                window.close()
            except Exception:
                pass
        self._plot_windows.clear()


class QtGuiScheduler:
    """Schedule small GUI-thread continuations without leaking Qt into services."""

    def call_soon(self, callback: Callable[[], None]) -> None:
        QtCore.QTimer.singleShot(0, callback)

    def call_later(self, delay_ms: int, callback: Callable[[], None]) -> None:
        QtCore.QTimer.singleShot(int(delay_ms), callback)
