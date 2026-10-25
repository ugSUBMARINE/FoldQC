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
    AlignLeft,
    AscendingOrder,
    BackgroundRole,
    DisplayRole,
    ForegroundRole,
    HeaderResizeFixed,
    HeaderResizeToContents,
    Horizontal,
    ItemIsEnabled,
    ItemIsSelectable,
    LayoutUsesWidgetRect,
    QtCore,
    QtGui,
    QtWidgets,
    TextAlignmentRole,
    ToolTipRole,
)
from .gui_layout import BASE_LAYOUT_SPACING, FixedMainButton
from .interface_analysis import (
    InterfaceAnalysisOptions,
    InterfaceContactExportRequest,
    InterfaceProjectionSelection,
    InterfaceReport,
    InterfaceReportObserver,
    InterfaceSelectionRequest,
)
from .interface_report_data import (
    InterfaceTableRow,
    contact_details,
    heatmap_value,
    pair_metric_specs,
    report_metric_specs,
    report_rows,
    residue_action_reason,
    score_details,
)
from .palettes import VIRIDIS_STOPS

INTERFACE_COLUMN_WIDTH = 88
INTERFACE_DETAILS_CONTROL_WIDTH = 200
INTERFACE_DETAILS_DROPDOWN_WIDTH = 160
INTERFACE_WINDOW_WIDTH = 1120
INTERFACE_WINDOW_HEIGHT = 690


class InterfaceTableModel(QtCore.QAbstractTableModel):
    def __init__(self, rows: tuple[InterfaceTableRow, ...], scope: str, parent=None):
        super().__init__(parent)
        self.rows = list(rows)
        self.specs = report_metric_specs(scope)
        self.scope = scope
        self.headers = [
            "Model",
            "Chain A" if scope == "chain_pair" else "Chain",
            "Partner(s)",
        ]
        self.headers += [s.interface.report_label for s in self.specs]
        self.headers += ["Cβ pairs" if scope == "chain_pair" else "Cα pairs"]
        if scope == "chain_rest":
            self.headers += ["pLDDT"]

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
        if column == len(self.headers) - 1 and self.scope == "chain_rest":
            score = row.score("interface_pdockq2_zhu")
            return None if score is None else score.interface_plddt
        return None if row.contacts is None else row.contacts.pair_count

    def data(self, index, role=DisplayRole):
        if not index.isValid():
            return None
        if role == TextAlignmentRole:
            return AlignCenter
        row = self.rows[index.row()]
        value = self.value(row, index.column())
        if role == DisplayRole:
            if value is None:
                return "N/A"
            if 3 <= index.column() < 3 + len(self.specs):
                return f"{value:.3f}"
            if self.scope == "chain_rest" and index.column() == len(self.headers) - 1:
                return f"{value:.1f}"
            return str(value)
        if role == ToolTipRole and 3 <= index.column() < 3 + len(self.specs):
            score = row.score(self.specs[index.column() - 3].key)
            return (
                "Unavailable"
                if score is None
                else score.reason
                or "Published score; see details for scope and implementation."
            )
        if role == ToolTipRole and index.column() == 3 + len(self.specs):
            return contact_details(row.contacts)
        return None

    def headerData(self, section, orientation, role=DisplayRole):
        if role == DisplayRole and orientation == Horizontal:
            return self.headers[section]
        if role == ToolTipRole and orientation == Horizontal:
            if section == 3 + len(self.specs):
                return "Geometric residue pairs at distance ≤8 Å; Cβ (Cα for glycine) for chain pairs, Cα for chain versus rest. See details for unique residue counts."
            if self.scope == "chain_rest" and section == 1:
                return "Scored chain"
            if self.scope == "chain_rest" and section == len(self.headers) - 1:
                return "Interface pLDDT (0–100)"
        if (
            role == ToolTipRole
            and orientation == Horizontal
            and 3 <= section < 3 + len(self.specs)
        ):
            spec = self.specs[section - 3]
            return f"{spec.label}; summary: {spec.interface.aggregation}"
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
                "white" if value is not None and value > 0.45 else "black"
            )
        if role == BackgroundRole:
            if value is None:
                return QtGui.QColor("#dedede")
            position = (1.0 - max(0.0, min(1.0, value))) * (len(VIRIDIS_STOPS) - 1)
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
        self.resize(INTERFACE_WINDOW_WIDTH, INTERFACE_WINDOW_HEIGHT)
        self._observer, self._on_close = observer, on_close
        self._current, self._busy = True, False
        self._closed = False
        self._selected: tuple[InterfaceTableRow, str] | None = None
        self._projection_metric: str | None = None
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
        self.analyze = FixedMainButton("Analyze target")
        self.analyze.setToolTip(
            "Analyze the current target selected in the main FoldQC window using this ipSAE preset."
        )
        self.analyze.clicked.connect(
            lambda: self._observer.analyze_current_interfaces(
                InterfaceAnalysisOptions(int(self.preset.currentData()))
            )
        )
        controls.addWidget(self.analyze)
        self.export = FixedMainButton("Export CSV")
        self.export.setToolTip(
            "Export the complete displayed interface snapshot, including all models, metrics, and both interface scopes."
        )
        self.export.clicked.connect(self._export)
        controls.addWidget(self.export)
        self.contacts_export = FixedMainButton("Contacts CSV")
        self.contacts_export.clicked.connect(self._export_contacts)
        controls.addWidget(self.contacts_export)
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
            self.heat_metric.addItem(spec.interface.report_label, spec.key)
        heatcontrols.addWidget(self.heat_metric)
        heatcontrols.addWidget(self.heat_model)
        heatcontrols.addWidget(self.directional)
        heatlayout.addLayout(heatcontrols)
        heatlayout.addWidget(
            QtWidgets.QLabel(
                "Rows: alignment chain · Columns: scored chain · Reversed viridis: 0 bright → 1 dark · Grey: undefined/N/A"
            )
        )
        self.heat = QtWidgets.QTableView()
        self.heat.setEditTriggers(AbstractItemViewNoEditTriggers)
        heatlayout.addWidget(self.heat)
        self.tabs.addTab(heatmap, "Heatmap")
        layout.addWidget(self.tabs, 3)
        details_row = QtWidgets.QHBoxLayout()
        self.details_controls = QtWidgets.QWidget()
        self.details_controls.setFixedWidth(INTERFACE_DETAILS_CONTROL_WIDTH)
        selection = QtWidgets.QVBoxLayout(self.details_controls)
        selection.setContentsMargins(0, 0, 0, 0)
        selection.setSpacing(BASE_LAYOUT_SPACING)
        details_label = QtWidgets.QLabel("Details for:")
        details_label.setAlignment(AlignLeft)
        selection.addWidget(details_label)
        self.selection_metric = QtWidgets.QComboBox()
        self.selection_metric.setFixedWidth(INTERFACE_DETAILS_DROPDOWN_WIDTH)
        for spec in metrics.METRICS:
            if spec.interface is not None and spec.interface.report_label is not None:
                self.selection_metric.addItem(spec.interface.report_label, spec.key)
        selection.addWidget(self.selection_metric, 0, AlignCenter)
        selection_label = QtWidgets.QLabel("Select contributing\nresidues in:")
        selection_label.setAlignment(AlignLeft)
        selection.addWidget(selection_label)
        self.side = QtWidgets.QComboBox()
        self.side.setFixedWidth(INTERFACE_DETAILS_DROPDOWN_WIDTH)
        for name, key in (
            ("Both chains", "both"),
            ("A / scored chain", "a"),
            ("B / rest", "b"),
        ):
            self.side.addItem(name, key)
        self.select = FixedMainButton("Select")
        self.select.setToolTip(
            "Select contributor residues for the chosen metric and side in PyMOL."
        )
        self.select.clicked.connect(self._select)
        selection.addWidget(self.side, 0, AlignCenter)
        selection.addWidget(self.select, 0, AlignCenter)
        selection.addSpacing(BASE_LAYOUT_SPACING)
        projection_label = QtWidgets.QLabel("Residue view:")
        selection.addWidget(projection_label)
        self.residue_view = QtWidgets.QComboBox()
        self.residue_view.setFixedWidth(INTERFACE_DETAILS_DROPDOWN_WIDTH)
        selection.addWidget(self.residue_view, 0, AlignCenter)
        self.residue_scope = QtWidgets.QComboBox()
        self.residue_scope.setFixedWidth(INTERFACE_DETAILS_DROPDOWN_WIDTH)
        selection.addWidget(self.residue_scope, 0, AlignCenter)
        self.plot_residues = FixedMainButton("Plot residues")
        self.color_residues = FixedMainButton("Color residues")
        selection.addWidget(self.plot_residues, 0, AlignCenter)
        selection.addWidget(self.color_residues, 0, AlignCenter)
        self.plot_residues.clicked.connect(
            lambda: self._project_residues("interface_profile")
        )
        self.color_residues.clicked.connect(
            lambda: self._project_residues("interface_color")
        )
        self.residue_view.currentIndexChanged.connect(self._projection_kind_changed)
        self.residue_scope.currentIndexChanged.connect(self._enable_selection)
        selection.addStretch(1)
        details_row.addWidget(self.details_controls)
        self.details = QtWidgets.QTextBrowser()
        self.details.setOpenExternalLinks(True)
        details_row.addWidget(self.details, 1)
        layout.addLayout(details_row, 2)
        self.details_controls.ensurePolished()
        control_stack = (
            details_label,
            self.selection_metric,
            selection_label,
            self.side,
            self.select,
            projection_label,
            self.residue_view,
            self.residue_scope,
            self.plot_residues,
            self.color_residues,
        )
        # macOS otherwise subtracts button layout margins from the visible gap,
        # which can make consecutive fixed-height buttons touch or overlap.
        for widget in control_stack:
            widget.setAttribute(LayoutUsesWidgetRect, True)
        # Native macOS layout hints can be shorter than the full control bounds.
        # Include actual widget size requirements so the last button cannot clip.
        control_height = sum(
            widget.sizeHint()
            .boundedTo(widget.maximumSize())
            .expandedTo(widget.minimumSize())
            .height()
            for widget in control_stack
        )
        margins = selection.contentsMargins()
        self.details_controls.setMinimumHeight(
            control_height
            + (len(control_stack) - 1) * selection.spacing()
            + BASE_LAYOUT_SPACING  # separation before the residue-view section
            + margins.top()
            + margins.bottom()
        )
        for button in (
            self.analyze,
            self.select,
            self.export,
            self.contacts_export,
            self.plot_residues,
            self.color_residues,
        ):
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
            header = table.horizontalHeader()
            header.setStretchLastSection(False)
            header.setSectionResizeMode(HeaderResizeFixed)
            for column in range(1, model.columnCount()):
                header.resizeSection(column, INTERFACE_COLUMN_WIDTH)
            header.setSectionResizeMode(0, HeaderResizeToContents)
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
            "Matches current target"
            if current
            else "Snapshot — source is no longer current; residue actions, viewer selection and contact export disabled"
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
        self._enable_selection()

    def _enable_selection(self, *_args) -> None:
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
        diagnostic = None if selected is None else selected[0].contacts
        self.contacts_export.setEnabled(
            enabled and diagnostic is not None and diagnostic.pair_count is not None
        )
        self.contacts_export.setToolTip(
            "Export all geometric contacts for the selected model/interface. Requires a current per-model snapshot with available geometry."
            if selected is None or diagnostic is None
            else f"Export {selected[0].label}: {diagnostic.chain_a} versus {', '.join(diagnostic.partners)}. "
            + contact_details(diagnostic)
            + " Requires a current snapshot; independent of selected metric and contributor side."
        )
        reason = residue_action_reason(
            None if selected is None else selected[0],
            str(self.selection_metric.currentData()),
            str(self.residue_view.currentData()),
            current=self._current,
            busy=self._busy,
        )
        self.plot_residues.setEnabled(not reason)
        self.color_residues.setEnabled(not reason)
        self.plot_residues.setToolTip(
            reason
            or "Plot retained residue values with a fixed 0–1 scale. Click/drag selects residues while the source remains current."
        )
        legend = (
            "ipSAE: yellow 0 → purple 1; each chain uses its own alignment direction."
            if self.residue_view.currentData() == "ipsae_profile"
            else "Orange: contributor (1); light grey: eligible non-contributor (0)."
        )
        if self.selection_metric.currentData() == "interface_ilis":
            legend += (
                " iLIS shows contact-filtered cLIS support, not every input to iLIS."
            )
        self.residue_view.setToolTip(legend)
        self.color_residues.setToolTip(
            reason
            or legend
            + " Other tokens in this member object become grey70. Other objects keep their colors."
        )

    def _refresh_projection_controls(self) -> None:
        key = str(self.selection_metric.currentData())
        if key != self._projection_metric:
            self._projection_metric = key
            self.residue_view.blockSignals(True)
            self.residue_view.clear()
            for spec in metrics.INTERFACE_PROJECTIONS.values():
                if key in spec.metric_keys:
                    label = (
                        "Contact-filtered support"
                        if key == "interface_ilis"
                        else spec.label
                    )
                    self.residue_view.addItem(label, spec.key)
            self.residue_view.blockSignals(False)
        self._projection_kind_changed()

    def _projection_kind_changed(self, *_args) -> None:
        side = self.residue_scope.currentData() or "both"
        self.residue_scope.blockSignals(True)
        self.residue_scope.clear()
        if self._selected is not None and self.residue_view.currentData() is not None:
            row = self._selected[0]
            a, b = row.chain_a, ", ".join(row.partners)
            profile = self.residue_view.currentData() == "ipsae_profile"
            for label, value in (
                ("Both directions" if profile else "Both sides", "both"),
                (f"{a} → {b}" if profile else a, "a"),
                (f"{b} → {a}" if profile else b, "b"),
            ):
                self.residue_scope.addItem(label, value)
            self.residue_scope.setCurrentIndex(
                max(0, self.residue_scope.findData(side))
            )
        self.residue_scope.blockSignals(False)
        self._enable_selection()

    def _project_residues(self, action) -> None:
        button = (
            self.color_residues if action == "interface_color" else self.plot_residues
        )
        if not button.isEnabled() or self._selected is None:
            return
        row = self._selected[0]
        self._observer.project_interface_residues(
            InterfaceProjectionSelection(
                self.report.generation,
                row.rank,
                str(self.selection_metric.currentData()),
                row.chain_a,
                row.partners,
                str(self.residue_view.currentData()),
                str(self.residue_scope.currentData()),
            ),
            action,
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
            spec = metrics.METRICS.require(score.metric_key)
            self.selection_metric.addItem(spec.interface.report_label, score.metric_key)
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
        references = []
        for reference in definition.references:
            label = html.escape(reference.label)
            references.append(
                f'<li><a href="{html.escape(reference.url, quote=True)}">{label}</a></li>'
                if reference.url is not None and reference.url.startswith("https://")
                else f"<li>{label}</li>"
            )
        citation = "<b>References</b><ul>" + "".join(references) + "</ul>"
        if definition.implementation_url is not None:
            citation += (
                f'<p><a href="{html.escape(definition.implementation_url, quote=True)}">'
                "Pinned implementation</a></p>"
            )
        self.details.setHtml(
            f'<pre style="white-space: pre-wrap">{html.escape(text)}</pre>{citation}'
        )
        self._refresh_projection_controls()
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
        if model.directional and model.key == "interface_ipsae":
            self.residue_view.setCurrentIndex(
                self.residue_view.findData("ipsae_profile")
            )
            row = self._selected[0]
            self.residue_scope.setCurrentIndex(
                self.residue_scope.findData("a" if row.chain_a == a else "b")
            )

    def _select(self) -> None:
        if self._selected is None:
            return
        row, _key = self._selected
        score = row.score(str(self.selection_metric.currentData()))
        if row.rank is not None and score is not None:
            self._observer.select_interface_contributors(
                InterfaceSelectionRequest(row.rank, score, str(self.side.currentData()))
            )

    def _export_contacts(self) -> None:
        if not self.contacts_export.isEnabled() or self._selected is None:
            return
        row, _key = self._selected
        diagnostic = row.contacts
        if row.rank is None or diagnostic is None:
            return
        result = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export selected interface contacts",
            "foldqc_contacts.csv",
            "CSV files (*.csv);;All files (*)",
        )
        path = result[0] if isinstance(result, tuple) else result
        if path:
            self._observer.export_interface_contacts(
                InterfaceContactExportRequest(row.rank, diagnostic, str(path))
            )
        if not self._closed:
            self.raise_()
            self.activateWindow()

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
