"""Real PyMOL/Qt smoke check; run with a PyMOL Python, QT_QPA_PLATFORM=offscreen.

Uses temporary settings and structures; never changes the user's PyMOL session.
Writes optional screenshots to the directory passed as the first argument.
An optional second argument selects a Qt style (e.g. macOS) for layout checks.
"""

from __future__ import annotations

import csv as csv_module
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
    BASE_LAYOUT_SPACING,
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
        "Contacts CSV",
        "Select",
        "Plot residues",
        "Color residues",
    }
    assert len(buttons) == 6
    for button in buttons:
        assert isinstance(button, FixedMainButton)
        assert button.width() == PRIMARY_BUTTON_WIDTH
        assert button.height() == PRIMARY_BUTTON_HEIGHT
    assert window.analyze.y() == window.export.y()
    assert window.export.x() > window.analyze.x()
    assert window.contacts_export.y() == window.export.y()
    assert window.contacts_export.x() > window.export.x()
    panel = window.details_controls
    assert (
        panel.minimumWidth() == panel.maximumWidth() == INTERFACE_DETAILS_CONTROL_WIDTH
    )
    labels = panel.findChildren(QtWidgets.QLabel)
    assert [label.text() for label in labels] == [
        "Details for:",
        "Select contributing\nresidues in:",
        "Residue view:",
    ]
    assert all(label.alignment() & AlignLeft for label in labels)
    for control in (
        window.selection_metric,
        window.side,
        window.select,
        window.residue_view,
        window.residue_scope,
        window.plot_residues,
        window.color_residues,
    ):
        assert abs(2 * control.x() + control.width() - panel.width()) <= 1
    stack = (
        labels[0],
        window.selection_metric,
        labels[1],
        window.side,
        window.select,
        labels[2],
        window.residue_view,
        window.residue_scope,
        window.plot_residues,
        window.color_residues,
    )
    original_stack = [(widget.y(), widget.height()) for widget in stack]
    # Check actual widget bounds: native macOS layout margins can hide overlaps
    # even when QVBoxLayout.spacing() and the overall minimum height look right.
    for previous, following in zip(stack, stack[1:]):
        expected_gap = BASE_LAYOUT_SPACING * (2 if following is labels[2] else 1)
        assert following.y() - previous.geometry().bottom() - 1 == expected_gap
    original_size = window.size()
    original_details = window.details.size()
    window.resize(window.width() + 160, window.height() + 400)
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
    assert window.contacts_export.isEnabled()
    assert "Geometric contacts: 2 pairs" in window.details.toPlainText()
    captured_contact = window._selected[0].contacts

    def change_selection_in_dialog(*_args):
        window.tabs.setCurrentIndex(1)
        return "/tmp/contact-capture-only.csv", "CSV files (*.csv)"

    with patch.object(
        QtWidgets.QFileDialog, "getSaveFileName", side_effect=change_selection_in_dialog
    ):
        window.contacts_export.click()
    assert (
        observer.export_interface_contacts.call_args.args[0].diagnostic
        is captured_contact
    )
    window.tabs.setCurrentIndex(0)
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
    assert window.residue_view.currentData() == "ipsae_profile"
    assert window.plot_residues.isEnabled() and window.color_residues.isEnabled()
    observer.project_interface_residues.assert_not_called()
    window.residue_scope.setCurrentIndex(window.residue_scope.findData("b"))
    window.plot_residues.click()
    captured_projection, captured_action = (
        observer.project_interface_residues.call_args.args
    )
    assert captured_action == "interface_profile" and captured_projection.side == "b"
    assert captured_projection.metric_key == "interface_ipsae"
    window.residue_view.setCurrentIndex(
        window.residue_view.findData("contributor_mask")
    )
    window.color_residues.click()
    assert (
        observer.project_interface_residues.call_args.args[0].kind == "contributor_mask"
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
    assert window.residue_view.currentData() == "ipsae_profile"
    assert window.residue_scope.currentData() == "a"
    window.heat_model.setCurrentIndex(2)
    window._heat_clicked(window.heat.model().index(0, 1))
    assert not window.select.isEnabled()  # ensemble mean is never a viewer target
    assert not window.contacts_export.isEnabled()
    assert (
        not window.plot_residues.isEnabled() and not window.color_residues.isEnabled()
    )
    window.heat_model.setCurrentIndex(0)
    window._heat_clicked(window.heat.model().index(0, 1))
    window.set_current(False)
    assert not window.select.isEnabled()
    assert not window.contacts_export.isEnabled()
    assert window.export.isEnabled()  # snapshot CSV remains exportable
    assert (
        not window.plot_residues.isEnabled() and not window.color_residues.isEnabled()
    )
    window.set_current(True)
    window.set_busy(True)
    assert not window.select.isEnabled() and not window.analyze.isEnabled()
    assert not window.export.isEnabled()
    assert not window.contacts_export.isEnabled()
    assert (
        not window.plot_residues.isEnabled() and not window.color_residues.isEnabled()
    )
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
            assert "contact_diagnostics" in csv.read_text()
            contact_csv = folder / "contacts.csv"
            with patch.object(
                QtWidgets.QFileDialog,
                "getSaveFileName",
                return_value=(str(contact_csv), "CSV files (*.csv)"),
            ):
                assert child.contacts_export.isEnabled()
                child.contacts_export.click()
            wait_until(lambda: not dialog.services.operations.is_busy)
            with contact_csv.open() as stream:
                contact_rows = list(csv_module.DictReader(stream))
            assert len(contact_rows) == 2
            for row in contact_rows:
                a, b = int(row["token_index_a"]), int(row["token_index_b"])
                assert float(row["distance_angstrom"]) == float(
                    np.linalg.norm(geometry.cb[a] - geometry.cb[b])
                )
            assert dialog._presenter.raise_interface_report()
            # Exercise the new workflow using only in-memory canonical data.
            active = dialog.state.active_model_state
            active.merge_data(replace(active.data, pae=np.array(fixture["pae"])))
            dialog.services.analysis.analyze_current_interfaces(
                InterfaceAnalysisOptions()
            )
            wait_until(lambda: not dialog.services.operations.is_busy)
            child.selection_metric.setCurrentIndex(
                child.selection_metric.findData("interface_ipsae")
            )
            child.residue_scope.setCurrentIndex(child.residue_scope.findData("a"))
            assert child.color_residues.isEnabled()
            child.color_residues.click()
            wait_until(lambda: not dialog.services.operations.is_busy)
            visual = viewer.snapshot_atom_visuals(captured.object_name)
            mapping = viewer.inspect_tokens(
                captured.object_name, active.token_map
            ).paint_mapping
            atom_tokens = mapping.atom_token_indices[list(visual.atom_indices)]
            colored = np.array(
                [visual.b_factors[atom_tokens == i][0] for i in range(4)]
            )
            ipsae = next(
                s
                for s in dialog.services.interfaces.report.models[0].scores
                if s.metric_key == "interface_ipsae"
            )
            np.testing.assert_allclose(
                colored[:2], [dict(ipsae.residue_profile)[i] for i in (0, 1)], rtol=1e-6
            )
            assert np.all(colored[2:] < 0)  # viewer's undefined B-factor sentinel
            assert dialog.services.viewer.get_managed_colorbar().reverse_palette
            assert "ipSAE profile" in dialog.widgets._stats_browser.toPlainText()
            child.plot_residues.click()
            wait_until(lambda: not dialog.services.operations.is_busy)
            assert dialog._presenter._plot_windows
            residue_plot = dialog._presenter._plot_windows[-1]
            assert residue_plot._selection_port is not None
            assert residue_plot._figure.axes[0].get_ylim() == (0, 1)
            assert "value:" in residue_plot._figure.axes[0].format_coord(0, 0.4)
            residue_plot._apply_viewer_selection([0])
            assert viewer.selection_token_indices(
                active.token_map, "foldqc_interface", obj_name=captured.object_name
            ) == [0]
            child.residue_view.setCurrentIndex(
                child.residue_view.findData("contributor_mask")
            )
            child.selection_metric.setCurrentIndex(
                child.selection_metric.findData("interface_clis")
            )
            child.residue_scope.setCurrentIndex(child.residue_scope.findData("both"))
            # Fail after a real PyMOL paint and verify restoration of actual colors.
            old_visuals = dialog.services.viewer.snapshot_atom_visuals(
                captured.object_name
            )
            old_ramp = dialog.services.viewer.get_managed_colorbar()
            old_statistics = dialog.widgets._stats_browser.toPlainText()
            paint = dialog.services.viewer.paint_continuous

            def fail_after_paint(*args, **kwargs):
                paint(*args, **kwargs)
                raise RuntimeError("Injected residue paint failure")

            with (
                patch.object(
                    dialog.services.viewer,
                    "paint_continuous",
                    side_effect=fail_after_paint,
                ),
                patch.object(dialog._presenter, "present_notice") as notice,
            ):
                child.color_residues.click()
                wait_until(lambda: not dialog.services.operations.is_busy)
                assert notice.call_args.args[0].code == "interface_projection_failed"
            restored = dialog.services.viewer.snapshot_atom_visuals(
                captured.object_name
            )
            np.testing.assert_array_equal(restored.b_factors, old_visuals.b_factors)
            np.testing.assert_array_equal(
                restored.color_indices, old_visuals.color_indices
            )
            assert dialog.services.viewer.get_managed_colorbar() == old_ramp
            assert dialog.widgets._stats_browser.toPlainText() == old_statistics
            child.color_residues.click()
            wait_until(lambda: not dialog.services.operations.is_busy)
            assert dialog.services.viewer.get_managed_colorbar() is None
            assert "Contributors:" in dialog.widgets._stats_browser.toPlainText()
            masked = viewer.snapshot_atom_visuals(captured.object_name)
            mask_values = np.array(
                [masked.b_factors[atom_tokens == i][0] for i in range(4)]
            )
            mask_colors = np.array(
                [masked.color_indices[atom_tokens == i][0] for i in range(4)]
            )
            clis = next(
                s
                for s in dialog.services.interfaces.report.models[0].scores
                if s.metric_key == "interface_clis"
            )
            expected_mask = np.array(
                [int(i in clis.support_a + clis.support_b) for i in range(4)]
            )
            np.testing.assert_array_equal(mask_values, expected_mask)
            assert len(set(mask_colors[expected_mask == 1])) == 1
            assert len(set(mask_colors[expected_mask == 0])) == 1
            assert (
                mask_colors[expected_mask == 1][0] != mask_colors[expected_mask == 0][0]
            )
            child.plot_residues.click()
            wait_until(lambda: not dialog.services.operations.is_busy)
            mask_plot = dialog._presenter._plot_windows[-1]
            if output is not None:
                mask_plot._figure.savefig(output / f"qt{QT_MAJOR}-contributor-mask.png")
            mask_plot.close()
            # A successful refresh invalidates the old plot's select and clear events.
            dialog.services.analysis.analyze_current_interfaces(
                InterfaceAnalysisOptions(15)
            )
            wait_until(lambda: not dialog.services.operations.is_busy)
            assert not residue_plot._selection_port.select((1,))
            assert not residue_plot._selection_port.clear()
            assert viewer.selection_token_indices(
                active.token_map, "foldqc_interface", obj_name=captured.object_name
            ) == [0]
            if output is not None:
                residue_plot._figure.savefig(
                    output / f"qt{QT_MAJOR}-residue-profile.png"
                )
            residue_plot.close()
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
        f"Qt{QT_MAJOR} smoke passed: tables, heatmap, controls, workers, residue profiles/masks, transactional coloring, guarded plot selection, export and cleanup."
    )


if __name__ == "__main__":
    main()
