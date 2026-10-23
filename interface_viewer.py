"""Non-modal Qt interface report. Scientific calculations stay outside this module."""

from __future__ import annotations

import html
from collections.abc import Callable

from . import metrics
from .compat import (
    AbstractItemViewNoEditTriggers,
    AbstractItemViewSelectRows,
    AbstractItemViewSingleSelection,
    AlignCenter,
    AscendingOrder,
    BackgroundRole,
    DisplayRole,
    ForegroundRole,
    Horizontal,
    ItemIsEnabled,
    ItemIsSelectable,
    QtCore,
    QtGui,
    QtWidgets,
    TextAlignmentRole,
    ToolTipRole,
)
from .interface_analysis import (
    InterfaceAnalysisOptions,
    InterfaceReport,
    InterfaceReportObserver,
    InterfaceSelectionRequest,
)
from .interface_report_data import (
    InterfaceTableRow,
    heatmap_value,
    pair_metric_specs,
    report_rows,
    score_details,
)
from .palettes import VIRIDIS_STOPS


class InterfaceTableModel(QtCore.QAbstractTableModel):
    def __init__(self, rows: tuple[InterfaceTableRow, ...], scope: str, parent=None):
        super().__init__(parent)
        self.rows = list(rows)
        self.specs = tuple(s for s in metrics.METRICS if s.result_scope == scope)
        self.scope = scope
        self.headers = [
            "Model",
            "Chain A" if scope == "chain_pair" else "Scored chain",
            "Partner(s)",
        ]
        self.headers += [s.label for s in self.specs]
        self.headers += [
            "Cβ contact pairs" if scope == "chain_pair" else "Cα contact pairs"
        ]
        if scope == "chain_rest":
            self.headers += ["Interface pLDDT (0–100)"]

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def value(self, row: InterfaceTableRow, column: int):
        if column < 3:
            return (row.label, row.chain_a, ", ".join(row.partners))[column]
        if column < 3 + len(self.specs):
            score = row.score(self.specs[column - 3].key)
            return None if score is None else score.value
        score = row.score(
            "interface_pdockq"
            if self.scope == "chain_pair"
            else "interface_pdockq2_zhu"
        )
        if score is None:
            return None
        if column == len(self.headers) - 1 and self.scope == "chain_rest":
            return score.interface_plddt
        return dict(score.counts).get(
            "cb_contact_pairs" if self.scope == "chain_pair" else "ca_contact_pairs"
        )

    def data(self, index, role=DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        value = self.value(row, index.column())
        if role == DisplayRole:
            return (
                "N/A"
                if value is None
                else f"{value:.5f}"
                if isinstance(value, float)
                else str(value)
            )
        if role == ToolTipRole and 3 <= index.column() < 3 + len(self.specs):
            score = row.score(self.specs[index.column() - 3].key)
            return (
                "Unavailable"
                if score is None
                else score.reason
                or "Published score; see details for scope and implementation."
            )
        return None

    def headerData(self, section, orientation, role=DisplayRole):
        if role == DisplayRole and orientation == Horizontal:
            return self.headers[section]
        return None

    def sort(self, column, order=AscendingOrder):
        self.layoutAboutToBeChanged.emit()
        # Keep N/A after finite values in either sort direction.
        available = [r for r in self.rows if self.value(r, column) is not None]
        missing = [r for r in self.rows if self.value(r, column) is None]
        available.sort(
            key=lambda r: self.value(r, column), reverse=order != AscendingOrder
        )
        self.rows = available + missing
        self.layoutChanged.emit()


class InterfaceHeatmapModel(QtCore.QAbstractTableModel):
    def __init__(self, rows, chains, key, rank, directional, parent=None):
        super().__init__(parent)
        self.rows, self.chains, self.key = rows, chains, key
        self.rank, self.directional = rank, directional

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.chains)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.chains)

    def data(self, index, role=DisplayRole):
        if not index.isValid():
            return None
        a, b = self.chains[index.row()], self.chains[index.column()]
        value = heatmap_value(self.rows, a, b, self.key, self.rank, self.directional)
        if role == DisplayRole:
            return "—" if a == b else "N/A" if value is None else f"{value:.3f}"
        if role == TextAlignmentRole:
            return AlignCenter
        if role == ForegroundRole:
            return QtGui.QColor(
                "white" if value is not None and value < 0.55 else "black"
            )
        if role == BackgroundRole:
            if value is None:
                return QtGui.QColor("#dedede")
            position = max(0.0, min(1.0, value)) * (len(VIRIDIS_STOPS) - 1)
            lo = min(int(position), len(VIRIDIS_STOPS) - 2)
            fraction = position - lo
            rgb = [
                x * (1 - fraction) + y * fraction
                for x, y in zip(VIRIDIS_STOPS[lo], VIRIDIS_STOPS[lo + 1], strict=True)
            ]
            return QtGui.QColor.fromRgbF(*rgb)
        if role == ToolTipRole:
            return (
                f"Alignment {a} → scored {b}: {value}"
                if value is not None
                else "Undefined diagonal, unavailable data, or incompatible ensemble residue identities."
            )
        return None

    def headerData(self, section, orientation, role=DisplayRole):
        return self.chains[section] if role == DisplayRole else None

    def flags(self, index):
        return ItemIsEnabled | ItemIsSelectable


class InterfaceDialog(QtWidgets.QDialog):
    """Qt adapter implementing InterfaceReportViewPort; owned by QtPresenter."""

    def __init__(
        self,
        report: InterfaceReport,
        observer: InterfaceReportObserver,
        parent=None,
        on_close: Callable[[], None] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("FoldQC — Analyze interfaces")
        self.setModal(False)
        self.resize(1100, 720)
        self._observer, self._on_close = observer, on_close
        self._current, self._busy = True, False
        self._closed = False
        self._selected: tuple[InterfaceTableRow, str] | None = None
        layout = QtWidgets.QVBoxLayout(self)
        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        layout.addWidget(self.header)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("ipSAE preset:"))
        self.preset = QtWidgets.QComboBox()
        for cutoff in (10, 15):
            self.preset.addItem(f"PAE <{cutoff} Å", cutoff)
        controls.addWidget(self.preset)
        self.analyze = QtWidgets.QPushButton("Analyze current target")
        self.analyze.clicked.connect(
            lambda: self._observer.analyze_current_interfaces(
                InterfaceAnalysisOptions(int(self.preset.currentData()))
            )
        )
        controls.addWidget(self.analyze)
        controls.addStretch()
        layout.addLayout(controls)
        self.tabs = QtWidgets.QTabWidget()
        self.pairs = self._table()
        self.rest = self._table()
        self.tabs.addTab(self.pairs, "Chain pairs")
        self.tabs.addTab(self.rest, "Chain versus rest")
        heatmap = QtWidgets.QWidget()
        heatlayout = QtWidgets.QVBoxLayout(heatmap)
        heatcontrols = QtWidgets.QHBoxLayout()
        self.heat_metric, self.heat_model = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.directional = QtWidgets.QCheckBox("Directional values")
        for spec in pair_metric_specs():
            self.heat_metric.addItem(spec.label, spec.key)
        heatcontrols.addWidget(self.heat_metric)
        heatcontrols.addWidget(self.heat_model)
        heatcontrols.addWidget(self.directional)
        heatlayout.addLayout(heatcontrols)
        heatlayout.addWidget(
            QtWidgets.QLabel(
                "Rows: alignment chain · Columns: scored chain · Viridis scale: 0–1 · Grey: undefined/N/A"
            )
        )
        self.heat = QtWidgets.QTableView()
        self.heat.setEditTriggers(AbstractItemViewNoEditTriggers)
        heatlayout.addWidget(self.heat)
        self.tabs.addTab(heatmap, "Heatmap")
        layout.addWidget(self.tabs, 3)
        selection = QtWidgets.QHBoxLayout()
        selection.addWidget(QtWidgets.QLabel("Contributors for:"))
        self.selection_metric = QtWidgets.QComboBox()
        for spec in metrics.METRICS:
            if spec.interface is not None:
                self.selection_metric.addItem(spec.label, spec.key)
        self.side = QtWidgets.QComboBox()
        for name, key in (
            ("Both chains", "both"),
            ("A / scored chain", "a"),
            ("B / rest", "b"),
        ):
            self.side.addItem(name, key)
        self.select = QtWidgets.QPushButton("Select contributors")
        self.select.clicked.connect(self._select)
        self.clear = QtWidgets.QPushButton("Clear selection")
        self.clear.clicked.connect(self._observer.clear_interface_selection)
        for widget in (self.selection_metric, self.side, self.select, self.clear):
            selection.addWidget(widget)
        layout.addLayout(selection)
        self.details = QtWidgets.QTextBrowser()
        self.details.setOpenExternalLinks(True)
        layout.addWidget(self.details, 2)
        footer = QtWidgets.QHBoxLayout()
        self.export = QtWidgets.QPushButton("Export CSV")
        self.export.clicked.connect(self._export)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        footer.addWidget(self.export)
        footer.addStretch()
        footer.addWidget(close)
        layout.addLayout(footer)
        for button in (self.analyze, self.select, self.clear, self.export, close):
            button.setAutoDefault(False)
            button.setDefault(False)
        self.heat_metric.currentIndexChanged.connect(self._heatmap)
        self.heat_model.currentIndexChanged.connect(self._heatmap)
        self.directional.stateChanged.connect(self._heatmap)
        self.heat.clicked.connect(self._heat_clicked)
        self.selection_metric.currentIndexChanged.connect(self._details)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.render(report, current=True)

    def _table(self):
        table = QtWidgets.QTableView()
        table.setEditTriggers(AbstractItemViewNoEditTriggers)
        table.setSelectionBehavior(AbstractItemViewSelectRows)
        table.setSelectionMode(AbstractItemViewSingleSelection)
        table.setAlternatingRowColors(True)
        return table

    def render(self, report: InterfaceReport, *, current: bool) -> None:
        self.report = report
        self._selected = None
        self.preset.setCurrentIndex(0 if report.options.ipsae_cutoff == 10 else 1)
        self.pair_rows = report_rows(report, "chain_pair")
        for table, scope in ((self.pairs, "chain_pair"), (self.rest, "chain_rest")):
            model = InterfaceTableModel(report_rows(report, scope), scope, table)
            previous_model, previous_selection = table.model(), table.selectionModel()
            table.setModel(model)
            if previous_model is not None:
                previous_model.deleteLater()
            if previous_selection is not None:
                previous_selection.deleteLater()
            table.setSortingEnabled(True)
            table.resizeColumnsToContents()
            table.selectionModel().currentChanged.connect(
                lambda index, _previous, t=table: self._table_selected(t, index)
            )
        self.heat_model.blockSignals(True)
        self.heat_model.clear()
        for model in report.models:
            self.heat_model.addItem(model.label, model.rank)
        if len(report.models) > 1:
            self.heat_model.addItem("Ensemble mean", None)
        self.heat_model.blockSignals(False)
        self._heatmap()
        self.set_current(current)
        diagnostics = [line for model in report.models for line in model.diagnostics]
        self.details.setPlainText(
            "Select a table row or heatmap cell for details.\n" + "\n".join(diagnostics)
        )
        if self.pairs.model().rowCount():
            self.pairs.setCurrentIndex(self.pairs.model().index(0, 3))

    def set_current(self, current: bool) -> None:
        self._current = current
        report = self.report
        status = (
            "Current source"
            if current
            else "Snapshot — source is no longer current; viewer selection disabled"
        )
        self.header.setText(
            f"{report.prediction_name} · {report.provider_label} · {report.target_name}\n"
            f"{len(report.models)} model(s) · Original prediction coordinates · Analyzed ipSAE: PAE <{report.options.ipsae_cutoff} Å · {status}"
        )
        self._enable_selection()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.analyze.setEnabled(not busy)
        self.export.setEnabled(not busy)
        self.clear.setEnabled(not busy)
        self._enable_selection()

    def _enable_selection(self) -> None:
        selected = self._selected
        enabled = bool(
            selected
            and selected[0].rank is not None
            and self._current
            and not self._busy
        )
        score = (
            None
            if selected is None
            else selected[0].score(str(self.selection_metric.currentData()))
        )
        self.select.setEnabled(
            enabled and score is not None and bool(score.support_a or score.support_b)
        )

    def _table_selected(self, table, index) -> None:
        if not index.isValid():
            return
        model = table.model()
        row = model.rows[index.row()]
        if 3 <= index.column() < 3 + len(model.specs):
            key = model.specs[index.column() - 3].key
        else:
            key = model.specs[0].key
        self._selected = row, key
        self.selection_metric.blockSignals(True)
        self.selection_metric.clear()
        for score in row.scores:
            self.selection_metric.addItem(
                metrics.METRICS.require(score.metric_key).label, score.metric_key
            )
        self.selection_metric.setCurrentIndex(self.selection_metric.findData(key))
        self.selection_metric.blockSignals(False)
        self._details()

    def _details(self, *_args) -> None:
        if self._selected is None:
            return
        row, _key = self._selected
        key = str(self.selection_metric.currentData())
        text = score_details(self.report, row, key)
        definition = metrics.METRICS.require(key).interface
        assert definition is not None
        link = definition.citation
        citation = (
            f'<a href="{html.escape(link, quote=True)}">{html.escape(link)}</a>'
            if link.startswith("https://")
            else html.escape(link)
        )
        self.details.setHtml(
            f'<pre style="white-space: pre-wrap">{html.escape(text)}</pre><p>{citation}</p>'
        )
        self._enable_selection()

    def _tab_changed(self, index) -> None:
        if index in (0, 1):
            table = self.pairs if index == 0 else self.rest
            if table.model().rowCount():
                selected = table.currentIndex()
                self._table_selected(
                    table, selected if selected.isValid() else table.model().index(0, 3)
                )
        elif index == 2:
            self._heatmap()

    def _heatmap(self, *_args) -> None:
        if not hasattr(self, "report"):
            return
        key = str(self.heat_metric.currentData())
        directional = key in {
            "interface_ipsae",
            "interface_lis_kim",
            "interface_lis_dunbrack",
            "interface_clis",
            "interface_pdockq2_dunbrack",
            "interface_iptm",
        }
        self.directional.setEnabled(directional)
        rank = self.heat_model.currentData()
        chains = tuple(
            dict.fromkeys(
                c
                for m in self.report.models
                if rank is None or m.rank == rank
                for c in m.chain_order
            )
        )
        previous_model, previous_selection = (
            self.heat.model(),
            self.heat.selectionModel(),
        )
        self.heat.setModel(
            InterfaceHeatmapModel(
                self.pair_rows,
                chains,
                key,
                rank,
                directional and self.directional.isChecked(),
                self.heat,
            )
        )
        self.heat.resizeColumnsToContents()
        if previous_model is not None:
            previous_model.deleteLater()
        if previous_selection is not None:
            previous_selection.deleteLater()
        if self.tabs.currentIndex() == 2 and len(chains) > 1:
            self._heat_clicked(self.heat.model().index(0, 1))

    def _heat_clicked(self, index) -> None:
        model = self.heat.model()
        a, b = model.chains[index.row()], model.chains[index.column()]
        matches = [
            (i, r)
            for i, r in enumerate(self.pairs.model().rows)
            if r.rank == model.rank and {r.chain_a, *r.partners} == {a, b}
        ]
        if len(matches) != 1 or a == b:
            self._selected = None
            self.details.setPlainText(
                "Choose an individual model: this cell has no unique compatible interface result."
            )
            self._enable_selection()
            return
        column = 3 + next(
            i
            for i, spec in enumerate(self.pairs.model().specs)
            if spec.key == model.key
        )
        pair_index = self.pairs.model().index(matches[0][0], column)
        self.pairs.setCurrentIndex(pair_index)
        self._table_selected(self.pairs, pair_index)

    def _select(self) -> None:
        if self._selected is None:
            return
        row, _key = self._selected
        score = row.score(str(self.selection_metric.currentData()))
        if row.rank is not None and score is not None:
            self._observer.select_interface_contributors(
                InterfaceSelectionRequest(row.rank, score, str(self.side.currentData()))
            )

    def _export(self) -> None:
        result = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export interface report",
            "foldqc_interfaces.csv",
            "CSV files (*.csv);;All files (*)",
        )
        path = result[0] if isinstance(result, tuple) else result
        if path:
            self._observer.export_interface_report(str(path))
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        if not self._closed:
            self._closed = True
            self._observer.interface_window_closed()
            if self._on_close is not None:
                self._on_close()
        super().closeEvent(event)
        self.deleteLater()
