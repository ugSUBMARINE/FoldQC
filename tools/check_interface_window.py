"""Real PyMOL/Qt smoke check; run with a PyMOL Python, QT_QPA_PLATFORM=offscreen.

Uses temporary settings and structures; never changes the user's PyMOL session.
Writes optional screenshots to the directory passed as the first argument.
An optional second argument selects a Qt style (e.g. macOS) for layout checks.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.compat import (  # noqa: E402
    QT_MAJOR,
    AlignCenter,
    AlignLeft,
    BackgroundRole,
    DescendingOrder,
    ForegroundRole,
    HeaderResizeFixed,
    HeaderResizeToContents,
    QtCore,
    QtWidgets,
    SettingsIniFormat,
    TextAlignmentRole,
)
from FoldQC.gui import FoldQCPluginDialog  # noqa: E402
from FoldQC.gui_layout import (  # noqa: E402
    PRIMARY_BUTTON_HEIGHT,
    PRIMARY_BUTTON_WIDTH,
    FixedMainButton,
)
from FoldQC.gui_session import QtSessionAdapter  # noqa: E402
from FoldQC.interface_analysis import InterfaceAnalysisOptions, InterfaceReport  # noqa: E402
from FoldQC.interface_compute import aggregate_interface_models, compute_interface_model  # noqa: E402
from FoldQC.interface_viewer import (  # noqa: E402
    INTERFACE_COLUMN_WIDTH,
    INTERFACE_DETAILS_CONTROL_WIDTH,
    InterfaceDialog,
    InterfaceHeatmapModel,
)
from FoldQC.mol_viewer import PyMOLViewer  # noqa: E402
from FoldQC.structure_index import StructureGeometry, StructureIndex  # noqa: E402
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap  # noqa: E402


def main() -> None:
    fixture = json.loads(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("tests/fixtures/interface_reference.json")
        .read_text()
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if len(sys.argv) > 2:
        assert app.setStyle(sys.argv[2]) is not None, (
            "Requested Qt style is unavailable."
        )
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    tm = TokenMap(
        tuple(
            TokenInfo(i, c, ResidueId(i + 1), r, False, None)
            for i, (c, r) in enumerate(
                zip(fixture["chains"], fixture["residues"], strict=True)
            )
        )
    )
    geometry = StructureGeometry(
        np.array(fixture["ca"]),
        np.array(fixture["cb"]),
        np.array(fixture["ca_plddt"]),
        np.array(fixture["cb_plddt"]),
        np.ones(4, bool),
    )
    index = StructureIndex(
        Path("/tmp/foldqc_qt_reference.cif"),
        "cif",
        tm,
        4,
        (0, 1, 2, 3),
        np.ones(4),
        geometry,
    )
    model = compute_interface_model(
        index,
        np.array(fixture["pae"]),
        None,
        InterfaceAnalysisOptions(),
        rank=0,
        label="Model 0",
        object_name="reference",
    )
    second = replace(model, rank=1, label="Model 1")
    report = InterfaceReport(
        "Reference dimer",
        "test",
        "Reference benchmark",
        str(index.path),
        "reference_ensemble",
        InterfaceAnalysisOptions(),
        (model, second),
        aggregate_interface_models((model, second)),
    )
    observer = Mock()
    window = InterfaceDialog(report, observer)
    window.show()
    app.processEvents()
    buttons = window.findChildren(QtWidgets.QPushButton)
    assert {button.text() for button in buttons} == {
        "Analyze target",
        "Export CSV",
        "Select",
    }
    assert len(buttons) == 3
    for button in buttons:
        assert isinstance(button, FixedMainButton)
        assert button.width() == PRIMARY_BUTTON_WIDTH
        assert button.height() == PRIMARY_BUTTON_HEIGHT
    assert window.analyze.y() == window.export.y()
    assert window.export.x() > window.analyze.x()
    panel = window.details_controls
    assert (
        panel.minimumWidth() == panel.maximumWidth() == INTERFACE_DETAILS_CONTROL_WIDTH
    )
    labels = panel.findChildren(QtWidgets.QLabel)
    assert [label.text() for label in labels] == [
        "Details for:",
        "Select contributing\nresidues in:",
    ]
    assert all(label.alignment() & AlignLeft for label in labels)
    for control in (window.selection_metric, window.side, window.select):
        assert abs(2 * control.x() + control.width() - panel.width()) <= 1
    stack = (labels[0], window.selection_metric, labels[1], window.side, window.select)
    original_stack = [(widget.y(), widget.height()) for widget in stack]
    original_size = window.size()
    original_details = window.details.size()
    window.resize(window.width() + 160, window.height() + 120)
    app.processEvents()
    assert panel.width() == INTERFACE_DETAILS_CONTROL_WIDTH
    assert [(widget.y(), widget.height()) for widget in stack] == original_stack
    assert window.details.width() == original_details.width() + 160
    assert window.details.height() > original_details.height()
    window.resize(original_size.width(), window.minimumSizeHint().height())
    app.processEvents()
    assert panel.height() >= panel.minimumSizeHint().height()
    assert window.select.geometry().bottom() < panel.height()
    assert window.details.height() == panel.height()
    assert [(widget.y(), widget.height()) for widget in stack] == original_stack
    window.resize(1, 1)
    app.processEvents()
    window.layout().activate()
    app.processEvents()
    assert all(panel.rect().contains(widget.geometry()) for widget in stack)
    if output is not None:
        assert window.grab().save(str(output / f"qt{QT_MAJOR}-minimum.png"))
    window.resize(original_size)
    app.processEvents()
    assert window.pairs.model().rowCount() == 3
    assert window.pairs.model().headers == [
        "Model",
        "Chain A",
        "Partner(s)",
        "ipSAE",
        "LIS",
        "cLIS",
        "iLIS",
        "pDockQ",
        "pDockQ2",
        "Pair ipTM",
        "Cβ pairs",
    ]
    assert window.heat_metric.findData("interface_lis_dunbrack") == -1
    lis_column = 3 + next(
        i
        for i, spec in enumerate(window.pairs.model().specs)
        if spec.key == "interface_lis_kim"
    )
    np.testing.assert_allclose(
        window.pairs.model().value(window.pairs.model().rows[0], lis_column),
        fixture["expected"]["lis_kim_mean"],
    )
    assert (
        window.pairs.model().data(window.pairs.model().index(0, lis_column)) == "0.486"
    )
    member_row = next(
        i for i, row in enumerate(window.pairs.model().rows) if row.rank == 0
    )
    assert window.pairs.model().data(window.pairs.model().index(member_row, 10)) == "2"
    assert window.rest.model().rowCount() == 6
    rest_row = next(
        i
        for i, row in enumerate(window.rest.model().rows)
        if row.rank == 0 and row.chain_a == "A"
    )
    assert window.rest.model().headers[3] == "pDockQ2"
    assert window.rest.model().data(window.rest.model().index(rest_row, 5)) == "76.7"
    assert window.rest.model().headers[4:] == ["Cα pairs", "pLDDT"]
    for table in (window.pairs, window.rest):
        header = table.horizontalHeader()
        assert header.sectionResizeMode(0) == HeaderResizeToContents
        for column in range(1, table.model().columnCount()):
            assert header.sectionResizeMode(column) == HeaderResizeFixed
            assert table.columnWidth(column) == INTERFACE_COLUMN_WIDTH
            assert (
                table.model().data(table.model().index(0, column), TextAlignmentRole)
                == AlignCenter
            )
    bright = InterfaceHeatmapModel(
        window.pair_rows, ("A", "B"), "interface_ipsae", 0, False
    )
    dark = InterfaceHeatmapModel(
        window.pair_rows, ("A", "B"), "interface_clis", 0, False
    )
    assert bright.data(bright.index(0, 1), ForegroundRole).name() == "#000000"
    assert dark.data(dark.index(0, 1), ForegroundRole).name() == "#ffffff"
    assert sum(bright.data(bright.index(0, 1), BackgroundRole).getRgbF()[:3]) > sum(
        dark.data(dark.index(0, 1), BackgroundRole).getRgbF()[:3]
    )
    assert window.select.isEnabled()
    assert "Original Zhu chain-versus-rest scores" not in window.details.toPlainText()
    assert "Dunbrack (2025)" in window.details.toPlainText()
    assert "10.1101/2025.02.10.637595" in window.details.toHtml()
    assert "Pinned implementation" in window.details.toPlainText()
    window.selection_metric.setCurrentIndex(
        window.selection_metric.findData("interface_pdockq")
    )
    assert "10.1038/s41467-022-28865-w" in window.details.toHtml()
    assert "10.1038/s41467-022-29480-5" in window.details.toHtml()
    assert "Bryant, Pozzati & Elofsson (2022)" in window.details.toPlainText()
    window.selection_metric.setCurrentIndex(
        window.selection_metric.findData("interface_ipsae")
    )
    window._select()
    assert observer.select_interface_contributors.call_args.args[0].side == "both"
    window.rest.model().sort(3, DescendingOrder)
    assert window.rest.model().rows[0].chain_a == "B"
    for tab, name in enumerate(("pairs", "rest", "heatmap")):
        window.tabs.setCurrentIndex(tab)
        app.processEvents()
        if output is not None:
            assert window.grab().save(str(output / f"qt{QT_MAJOR}-{name}.png"))
    window.directional.setChecked(True)
    window._heat_clicked(window.heat.model().index(0, 1))
    assert window.selection_metric.currentData() == "interface_ipsae"
    window.heat_model.setCurrentIndex(2)
    window._heat_clicked(window.heat.model().index(0, 1))
    assert not window.select.isEnabled()  # ensemble mean is never a viewer target
    window.heat_model.setCurrentIndex(0)
    window._heat_clicked(window.heat.model().index(0, 1))
    window.set_current(False)
    assert not window.select.isEnabled()
    window.set_current(True)
    window.set_busy(True)
    assert not window.select.isEnabled() and not window.analyze.isEnabled()
    assert not window.export.isEnabled()
    window.set_busy(False)
    window.preset.setCurrentIndex(1)
    window.analyze.click()
    assert observer.analyze_current_interfaces.call_args.args[0].ipsae_cutoff == 15
    window.close()
    observer.interface_window_closed.assert_called_once()

    # Test the actual main-dialog composition, worker callbacks and PyMOL selection.
    with tempfile.TemporaryDirectory(prefix="foldqc-qt-") as tmp:
        folder = Path(tmp)
        pdb = folder / "reference.pdb"
        lines = []
        serial = 0
        for i, token in enumerate(tm):
            for name, coords, confidence in (
                ("CA", geometry.ca[i], geometry.ca_plddt[i]),
                ("CB", geometry.cb[i], geometry.cb_plddt[i]),
            ):
                if token.res_name == "GLY" and name == "CB":
                    continue
                serial += 1
                x, y, z = coords
                lines.append(
                    f"ATOM  {serial:5} {name:^4} {token.res_name:3} {token.chain_id}{token.res_num:4}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{confidence:6.2f}           C\n"
                )
        pdb.write_text("".join(lines))
        settings = QtCore.QSettings(str(folder / "settings.ini"), SettingsIniFormat)

        def wait_until(predicate):
            deadline = time.monotonic() + 12
            while not predicate():
                app.processEvents()
                if time.monotonic() > deadline:
                    raise AssertionError("Qt worker did not complete")
                time.sleep(0.005)
            app.processEvents()

        with patch.object(QtSessionAdapter, "_settings", return_value=settings):
            dialog = FoldQCPluginDialog()
            dialog.show()
            dialog.services.lifecycle.load_prediction(str(pdb))
            wait_until(
                lambda: (
                    dialog.state.active_model_state is not None
                    and not dialog.services.operations.is_busy
                )
            )
            assert dialog.widgets._interfaces_btn.isEnabled()
            assert all(
                not str(dialog.widgets._prop_combo.itemData(i) or "").startswith(
                    "interface_"
                )
                for i in range(dialog.widgets._prop_combo.count())
            )
            dialog.widgets._interfaces_btn.click()
            wait_until(
                lambda: (
                    dialog.services.interfaces.report is not None
                    and not dialog.services.operations.is_busy
                )
            )
            child = dialog._presenter._interface_window
            assert child is not None
            app.processEvents()
            child.selection_metric.setCurrentIndex(
                child.selection_metric.findData("interface_pdockq")
            )
            assert child.select.isEnabled()
            child.select.click()
            viewer = PyMOLViewer()
            captured = dialog.services.interfaces.report.models[0]
            pdockq = next(
                s for s in captured.scores if s.metric_key == "interface_pdockq"
            )
            assert abs(pdockq.value - fixture["expected"]["pdockq"]) < 1e-10
            assert viewer.selection_token_indices(
                captured.token_map, "foldqc_interface", obj_name=captured.object_name
            )
            viewer.transform(
                captured.object_name, np.eye(3), np.array([20.0, 5.0, -10.0])
            )
            before = tuple(s.value for s in captured.scores)
            dialog.services.analysis.analyze_current_interfaces(
                InterfaceAnalysisOptions()
            )
            wait_until(lambda: not dialog.services.operations.is_busy)
            after = tuple(
                s.value for s in dialog.services.interfaces.report.models[0].scores
            )
            assert before == after
            child.export.setEnabled(True)
            csv = folder / "interfaces.csv"
            with patch.object(
                QtWidgets.QFileDialog,
                "getSaveFileName",
                return_value=(str(csv), "CSV files (*.csv)"),
            ):
                child.export.click()
            assert "interface_pdockq" in csv.read_text()
            assert dialog._presenter.raise_interface_report()
            child.select.click()
            assert viewer.name_exists("foldqc_interface")
            active_model = dialog.state.active_model_state
            dialog.close()
            assert not child.isVisible()
            assert not dialog.isVisible()
            assert dialog._presenter._interface_window is None
            assert not viewer.name_exists("foldqc_interface")
            assert dialog.services.interfaces.report is None
            assert dialog.state.active_model_state is active_model
            assert not dialog._shutdown_complete
            dialog.show()
            dialog.widgets._interfaces_btn.click()
            wait_until(
                lambda: (
                    dialog._presenter._interface_window is not None
                    and not dialog.services.operations.is_busy
                )
            )
            assert dialog._presenter._interface_window.isVisible()
            if output is not None:
                app.processEvents()
                assert dialog.grab().save(str(output / f"qt{QT_MAJOR}-main.png"))
            dialog.shutdown()
            dialog.close()
    print(
        f"Qt{QT_MAJOR} smoke passed: tables, heatmap, controls, main composition, workers, PyMOL contributors, export and cleanup."
    )


if __name__ == "__main__":
    main()
