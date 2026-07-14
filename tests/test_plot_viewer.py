from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _install_fake_pymol() -> None:
    if "pymol" in sys.modules and "pymol.Qt" in sys.modules:
        return

    flag = types.SimpleNamespace(
        AlignLeft=1,
        AlignRight=2,
        AlignCenter=4,
        AlignTop=8,
        AlignBottom=16,
        AlignVCenter=32,
        ItemIsEnabled=64,
        WindowCloseButtonHint=128,
    )
    orientation = types.SimpleNamespace(Horizontal=1, Vertical=2)
    scroll_policy = types.SimpleNamespace(ScrollBarAlwaysOff=1, ScrollBarAsNeeded=2)
    qt_core = types.SimpleNamespace(
        QT_VERSION_STR="6.0.0",
        Qt=types.SimpleNamespace(
            AlignmentFlag=flag,
            Orientation=orientation,
            ScrollBarPolicy=scroll_policy,
            ItemFlag=flag,
            WindowType=flag,
        ),
        QTimer=types.SimpleNamespace(singleShot=lambda *_args, **_kwargs: None),
        QSettings=object,
    )
    qt_widgets = types.SimpleNamespace(
        QDialog=object,
        QFormLayout=types.SimpleNamespace(
            FieldGrowthPolicy=types.SimpleNamespace(AllNonFixedFieldsGrow=1)
        ),
        QMessageBox=types.SimpleNamespace(
            StandardButton=types.SimpleNamespace(Yes=1, Cancel=2)
        ),
    )
    qt = types.SimpleNamespace(
        QtCore=qt_core,
        QtGui=types.SimpleNamespace(
            QAction=object,
            QIcon=types.SimpleNamespace(fromTheme=lambda _name: None),
        ),
        QtWidgets=qt_widgets,
    )
    sys.modules["pymol"] = types.SimpleNamespace(Qt=qt, cmd=None)
    sys.modules["pymol.Qt"] = qt


_install_fake_pymol()

from FoldQC.plot_viewer import PlotDialog  # noqa: E402
from FoldQC.token_map import TokenInfo  # noqa: E402


class PlotViewerSelectionTests(unittest.TestCase):
    def _dialog_with_metadata(self, metadata):
        dialog = PlotDialog.__new__(PlotDialog)
        dialog._selection_metadata = metadata
        return dialog

    @staticmethod
    def _token(token_idx: int, obj_name: str, *, chain_id: str = "A") -> TokenInfo:
        return TokenInfo(
            token_idx=token_idx,
            chain_id=chain_id,
            res_num=token_idx + 1,
            res_name="ALA",
            is_hetatm=False,
            atom_name=None,
        )

    def test_grouped_bar_click_selects_all_tokens_for_bar(self) -> None:
        dialog = self._dialog_with_metadata(
            {
                "kind": "bars",
                "bar_token_indices": [[0, 2], [1], [3, 4]],
                "bar_x_positions": [0.0, 1.0, 2.0],
                "bar_widths": [0.8, 0.8, 0.8],
            }
        )

        tokens, rows, cols = dialog._tokens_for_click(0.2, 1.0)

        self.assertEqual(tokens, [0, 2])
        self.assertEqual(rows, [])
        self.assertEqual(cols, [])

    def test_grouped_bar_rectangle_unions_selected_bars(self) -> None:
        dialog = self._dialog_with_metadata(
            {
                "kind": "bars",
                "bar_token_indices": [[0, 2], [1], [3, 4]],
                "bar_x_positions": [0.0, 1.0, 2.0],
                "bar_widths": [0.8, 0.8, 0.8],
            }
        )

        tokens, _rows, _cols = dialog._tokens_for_rectangle(-0.5, None, 1.4, None)

        self.assertEqual(tokens, [0, 2, 1])

    def test_legacy_bar_selection_still_uses_nearest_token(self) -> None:
        dialog = self._dialog_with_metadata(
            {
                "kind": "bars",
                "token_indices": [10, 11],
                "x_positions": [0.0, 1.0],
            }
        )

        click_tokens, _rows, _cols = dialog._tokens_for_click(0.8, 1.0)
        rect_tokens, _rows, _cols = dialog._tokens_for_rectangle(0.5, None, 1.5, None)

        self.assertEqual(click_tokens, [11])
        self.assertEqual(rect_tokens, [11])

    def test_click_selection_accepts_any_stacked_bar_axis(self) -> None:
        top_ax = object()
        bottom_ax = object()
        dialog = self._dialog_with_metadata(
            {
                "kind": "bars",
                "bar_token_indices": [[0], [1]],
                "bar_x_positions": [0.0, 1.0],
                "bar_widths": [0.8, 0.8],
            }
        )
        dialog._selection_axes = [top_ax, bottom_ax]
        dialog._toolbar_is_active = lambda: False
        selected = []
        dialog._apply_viewer_selection = lambda tokens, rows, cols, additive=False: (
            selected.append((tokens, rows, cols, additive))
        )
        dialog._update_click_highlight = lambda *_args, **_kwargs: None
        event = types.SimpleNamespace(
            inaxes=bottom_ax,
            button=1,
            xdata=1.0,
            ydata=5.0,
            modifiers=frozenset(),
        )

        dialog._on_plot_click(event)

        self.assertEqual(selected, [([1], [], [], False)])

    def test_cmd_ctrl_and_alt_request_additive_selection(self) -> None:
        for modifier in ("cmd", "super", "ctrl", "control", "alt"):
            event = types.SimpleNamespace(modifiers=frozenset({modifier}), key=None)
            with self.subTest(modifier=modifier):
                self.assertTrue(PlotDialog._event_is_additive(event))

        legacy_event = types.SimpleNamespace(modifiers=None, key="cmd")
        self.assertTrue(PlotDialog._event_is_additive(legacy_event))

    def test_selection_controls_show_hint_and_add_disabled_icon_action(self) -> None:
        class FakeAction:
            def __init__(self, icon, text, parent):
                self.icon = icon
                self.text = text
                self.parent = parent
                self.tooltip = ""
                self.enabled = None
                self.triggered = types.SimpleNamespace(
                    connect=lambda slot: setattr(self, "slot", slot)
                )

            def setToolTip(self, text):
                self.tooltip = text

            def setEnabled(self, enabled):
                self.enabled = enabled

        hint = types.SimpleNamespace(visible=False)
        hint.setVisible = lambda visible: setattr(hint, "visible", visible)
        toolbar = types.SimpleNamespace(actions=[], separators=0)
        toolbar.addSeparator = lambda: setattr(
            toolbar, "separators", toolbar.separators + 1
        )
        toolbar.addAction = lambda action: toolbar.actions.append(action)
        toolbar.style = lambda: types.SimpleNamespace(
            standardIcon=lambda _icon: "fallback-icon"
        )
        dialog = self._dialog_with_metadata({"kind": "line"})
        dialog._selection_hint = hint
        dialog._toolbar = toolbar

        with (
            mock.patch("FoldQC.plot_viewer.QAction", FakeAction),
            mock.patch(
                "FoldQC.plot_viewer._make_clear_selection_icon",
                return_value="trash-icon",
            ),
        ):
            dialog._install_selection_controls()

        self.assertTrue(hint.visible)
        self.assertEqual(toolbar.separators, 1)
        self.assertEqual(len(toolbar.actions), 1)
        self.assertEqual(toolbar.actions[0].icon, "trash-icon")
        self.assertEqual(toolbar.actions[0].text, "")
        self.assertEqual(toolbar.actions[0].tooltip, "Clear plot selection")
        self.assertFalse(toolbar.actions[0].enabled)

    def test_toolbar_token_identity_formats_polymer_and_hetatm(self) -> None:
        polymer = TokenInfo(
            token_idx=44,
            chain_id="A",
            res_num=45,
            res_name="PHE",
            is_hetatm=False,
            atom_name=None,
        )
        ligand_atom = TokenInfo(
            token_idx=317,
            chain_id="L",
            res_num=501,
            res_name="lig",
            is_hetatm=True,
            atom_name="C1",
        )

        self.assertEqual(
            PlotDialog._format_token_identity(polymer),
            "token 44 · A:Phe45",
        )
        self.assertEqual(
            PlotDialog._format_token_identity(ligand_atom),
            "token 317 · L:LIG501/C1",
        )

    def test_line_toolbar_appends_token_only_near_plotted_position(self) -> None:
        figure = Figure()
        axis = figure.subplots()
        axis.format_coord = lambda x, y: f"native x={x:g}, y={y:g}"
        token_map = [
            TokenInfo(0, "A", 10, "GLY", False, None),
            TokenInfo(1, "B", 25, "ASP", False, None),
        ]
        dialog = self._dialog_with_metadata(
            {
                "kind": "line",
                "token_map": token_map,
                "token_indices": [0, 1],
                "x_positions": [0.0, 10.0],
            }
        )
        dialog._selection_axes = [axis]

        dialog._install_toolbar_formatters()

        self.assertEqual(
            axis.format_coord(10.2, 0.75),
            "native x=10.2, y=0.75 | token 1 · B:Asp25",
        )
        self.assertEqual(axis.format_coord(5.0, 0.75), "native x=5, y=0.75")

    def test_matrix_toolbar_maps_rows_and_columns_independently(self) -> None:
        figure = Figure()
        axis = figure.subplots()
        axis.format_coord = lambda x, y: f"native ({x:g}, {y:g})"
        token_map = [
            TokenInfo(0, "A", 45, "PHE", False, None),
            TokenInfo(1, "B", 120, "ASP", False, None),
        ]
        dialog = self._dialog_with_metadata(
            {
                "kind": "matrix",
                "token_map": token_map,
                "row_indices": [1],
                "col_indices": [0],
            }
        )
        dialog._selection_axes = [axis]

        dialog._install_toolbar_formatters()

        self.assertEqual(
            axis.format_coord(0.2, -0.1),
            "native (0.2, -0.1) | row: token 1 · B:Asp120 | col: token 0 · A:Phe45",
        )
        self.assertEqual(axis.format_coord(0.6, 0.0), "native (0.6, 0)")

    def test_matrix_toolbar_preserves_matplotlib_cell_value_once(self) -> None:
        from matplotlib.backend_bases import MouseEvent, NavigationToolbar2
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        figure = Figure()
        canvas = FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.imshow(np.array([[3.72]], dtype=np.float64))
        canvas.draw()
        pixel_x, pixel_y = axis.transData.transform((0.0, 0.0))
        event = MouseEvent("motion_notify_event", canvas, pixel_x, pixel_y)
        original_message = NavigationToolbar2._mouse_event_to_message(event)
        original_value_line = original_message.splitlines()[-1]
        dialog = self._dialog_with_metadata(
            {
                "kind": "matrix",
                "token_map": [TokenInfo(0, "A", 45, "PHE", False, None)],
                "row_indices": [0],
                "col_indices": [0],
            }
        )
        dialog._selection_axes = [axis]

        dialog._install_toolbar_formatters()
        enhanced_message = NavigationToolbar2._mouse_event_to_message(event)
        reflowed_message = PlotDialog._reflow_matrix_toolbar_message(enhanced_message)
        coordinate_line, detail_line = reflowed_message.splitlines()

        self.assertNotIn("row:", coordinate_line)
        self.assertIn("row: token 0 · A:Phe45", detail_line)
        self.assertIn(", col: token 0 · A:Phe45", detail_line)
        self.assertTrue(detail_line.endswith(original_value_line))
        self.assertEqual(reflowed_message.count(original_value_line), 1)

    def test_matrix_toolbar_set_message_applies_two_line_layout(self) -> None:
        messages = []
        toolbar = types.SimpleNamespace(set_message=messages.append)
        dialog = self._dialog_with_metadata({"kind": "matrix"})
        dialog._toolbar = toolbar

        dialog._install_matrix_toolbar_message_layout()
        toolbar.set_message(
            "(x, y) = (247.0, 173.0) | row: token 173 · A:Asp174 | "
            "col: token 247 · A:Leu248\n[1.06]"
        )

        self.assertEqual(
            messages,
            [
                "(x, y) = (247.0, 173.0)\n"
                "row: token 173 · A:Asp174, col: token 247 · A:Leu248 [1.06]"
            ],
        )

    def test_binding_fingerprint_toolbar_formats_both_axes(self) -> None:
        figure = Figure()
        axes = figure.subplots(2, 1)
        token_map = [TokenInfo(0, "A", 45, "PHE", False, None)]
        dialog = self._dialog_with_metadata(
            {
                "kind": "bars",
                "token_map": token_map,
                "token_indices": [0],
                "x_positions": [0.0],
            }
        )
        dialog._selection_axes = list(axes)
        original_outputs = [axis.format_coord(0.0, 1.0) for axis in axes]

        dialog._install_toolbar_formatters()

        for axis, original in zip(axes, original_outputs):
            self.assertEqual(
                axis.format_coord(0.0, 1.0),
                f"{original} | token 0 · A:Phe45",
            )

    def test_grouped_bar_and_invalid_metadata_keep_native_toolbar_output(self) -> None:
        figure = Figure()
        grouped_axis, invalid_axis = figure.subplots(2, 1)
        grouped_original = grouped_axis.format_coord(0.0, 1.0)
        grouped = self._dialog_with_metadata(
            {
                "kind": "bars",
                "token_map": [TokenInfo(0, "A", 1, "ALA", False, None)],
                "bar_token_indices": [[0]],
            }
        )
        grouped._selection_axes = [grouped_axis]
        grouped._install_toolbar_formatters()

        invalid_original = invalid_axis.format_coord(0.0, 1.0)
        invalid = self._dialog_with_metadata(
            {
                "kind": "line",
                "token_map": [TokenInfo(0, "A", 1, "ALA", False, None)],
                "token_indices": [0],
                "x_positions": [0.0, 1.0],
            }
        )
        invalid._selection_axes = [invalid_axis]
        invalid._install_toolbar_formatters()

        self.assertEqual(grouped_axis.format_coord(0.0, 1.0), grouped_original)
        self.assertEqual(invalid_axis.format_coord(0.0, 1.0), invalid_original)

    def test_merged_position_intervals_join_adjacent_tokens(self) -> None:
        self.assertEqual(
            PlotDialog._merged_position_intervals([3.0, 0.0, 1.0]),
            [(-0.5, 1.5), (2.5, 3.5)],
        )

    def test_bar_highlight_changes_edges_and_restores_original_style(self) -> None:
        figure = Figure()
        axis = figure.subplots()
        bars = axis.bar(
            [0.0, 1.0],
            [2.0, 3.0],
            width=0.8,
            edgecolor="white",
            linewidth=0.6,
        )
        dialog = self._dialog_with_metadata(
            {
                "kind": "bars",
                "bar_token_indices": [[0], [1]],
                "bar_x_positions": [0.0, 1.0],
                "bar_widths": [0.8, 0.8],
            }
        )
        dialog._selection_axes = [axis]
        dialog._highlight_artists = []
        dialog._matrix_boundary_artist = None
        dialog._bar_original_styles = []

        dialog._render_bar_highlights([1])

        self.assertAlmostEqual(bars[0].get_linewidth(), 0.6)
        self.assertAlmostEqual(bars[1].get_linewidth(), 2.0)
        np.testing.assert_allclose(bars[1].get_edgecolor()[:3], (0.4, 0.4, 0.4))

        dialog._restore_bar_styles()

        self.assertAlmostEqual(bars[1].get_linewidth(), 0.6)

    def test_bar_highlight_uses_middle_grey_for_plddt_class_bar(self) -> None:
        figure = Figure()
        axis = figure.subplots()
        bars = axis.bar(
            [0.0],
            [2.0],
            width=0.8,
            color=[(1.0, 0.494, 0.271)],
            edgecolor="none",
        )
        dialog = self._dialog_with_metadata(
            {
                "kind": "bars",
                "bar_token_indices": [[0]],
                "bar_x_positions": [0.0],
                "bar_widths": [0.8],
            }
        )
        dialog._selection_axes = [axis]
        dialog._highlight_artists = []
        dialog._matrix_boundary_artist = None
        dialog._bar_original_styles = []

        dialog._render_bar_highlights([0])

        np.testing.assert_allclose(bars[0].get_edgecolor()[:3], (0.4, 0.4, 0.4))

    def test_ensemble_summary_uses_spans_without_modifying_bar_edges(self) -> None:
        figure = Figure()
        axes = figure.subplots(2, 1)
        bars = [axis.bar([0.0, 1.0], [1.0, 2.0]) for axis in axes]
        original_edges = [
            [patch.get_edgecolor() for patch in container] for container in bars
        ]
        dialog = self._dialog_with_metadata(
            {
                "kind": "ensemble_site_summary",
                "member_obj_names": ["model_0", "model_1"],
                "member_x_positions": [0.0, 1.0],
                "member_widths": [0.9, 0.9],
            }
        )
        dialog._selection_axes = list(axes)
        dialog._highlight_artists = []
        dialog._matrix_boundary_artist = None

        dialog._render_group_span_highlights([1])

        self.assertEqual(len(dialog._highlight_artists), 2)
        for container, expected_edges in zip(bars, original_edges):
            self.assertEqual(
                [patch.get_edgecolor() for patch in container],
                expected_edges,
            )

    def test_line_highlight_merges_adjacent_selected_positions(self) -> None:
        figure = Figure()
        axis = figure.subplots()
        dialog = self._dialog_with_metadata(
            {
                "kind": "line",
                "token_indices": [0, 1, 2, 3],
                "x_positions": [0.0, 1.0, 2.0, 3.0],
            }
        )
        dialog._selection_axes = [axis]
        dialog._selected_token_indices = [0, 1, 3]
        dialog._highlight_artists = []
        dialog._matrix_boundary_artist = None

        dialog._render_span_highlights()

        self.assertEqual(len(dialog._highlight_artists), 2)

    def test_matrix_boundary_has_no_internal_edge_for_adjacent_cells(self) -> None:
        mask = np.array([[True, True], [False, False]], dtype=bool)

        segments = PlotDialog._matrix_boundary_segments(mask)

        normalized = {
            tuple(sorted((tuple(start), tuple(end)))) for start, end in segments
        }
        self.assertEqual(len(normalized), 6)
        self.assertNotIn(((0.5, -0.5), (0.5, 0.5)), normalized)

    def test_matrix_mask_replacement_and_addition_track_actual_cells(self) -> None:
        dialog = self._dialog_with_metadata(
            {
                "kind": "matrix",
                "row_indices": [0, 1, 2],
                "col_indices": [0, 1, 2],
            }
        )
        dialog._matrix_selection_mask = None

        dialog._update_matrix_mask({(0, 0), (0, 1)}, additive=False)
        dialog._update_matrix_mask({(2, 2)}, additive=True)

        np.testing.assert_array_equal(
            dialog._matrix_selection_mask,
            np.array(
                [
                    [True, True, False],
                    [False, False, False],
                    [False, False, True],
                ]
            ),
        )

        dialog._update_matrix_mask({(1, 1)}, additive=False)

        np.testing.assert_array_equal(
            dialog._matrix_selection_mask,
            np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, False],
                ]
            ),
        )

    def test_additive_matrix_selection_unions_tokens_rows_and_columns(self) -> None:
        cmd = types.SimpleNamespace(selections=[], enabled=[], refreshes=0)
        cmd.select = lambda name, expr: cmd.selections.append((name, expr))
        cmd.enable = lambda name: cmd.enabled.append(name)

        def refresh():
            cmd.refreshes += 1

        cmd.refresh = refresh
        sys.modules["pymol"].cmd = cmd

        token_map = [self._token(idx, "model") for idx in range(4)]
        dialog = self._dialog_with_metadata(
            {
                "kind": "matrix",
                "token_map": token_map,
                "obj_name": "model",
            }
        )
        dialog._selected_token_indices = []
        dialog._selected_row_indices = []
        dialog._selected_col_indices = []

        dialog._apply_viewer_selection([0, 1], [0], [1])
        dialog._apply_viewer_selection([2, 3], [2], [3], additive=True)

        self.assertEqual(dialog._selected_token_indices, [0, 1, 2, 3])
        self.assertEqual(dialog._selected_row_indices, [0, 2])
        self.assertEqual(dialog._selected_col_indices, [1, 3])
        self.assertEqual(
            cmd.selections[-3:],
            [
                (
                    "foldqc_plot_selection",
                    "(%model and polymer and chain A)",
                ),
                ("foldqc_plot_rows", "(%model and polymer and chain A and resi 1+3)"),
                ("foldqc_plot_cols", "(%model and polymer and chain A and resi 2+4)"),
            ],
        )

    def test_ensemble_token_selection_uses_object_metadata_for_compaction(self) -> None:
        token_maps = []
        for obj_name in ("model_0", "model_1"):
            token_maps.append(
                [
                    TokenInfo(
                        token_idx=idx,
                        chain_id="A",
                        res_num=idx + 1,
                        res_name="ALA",
                        is_hetatm=False,
                        atom_name=None,
                    )
                    for idx in range(3)
                ]
            )
        dialog = self._dialog_with_metadata(
            {
                "kind": "line",
                "token_maps": token_maps,
                "token_map_obj_names": ["model_0", "model_1"],
            }
        )

        self.assertEqual(
            dialog._selection_object_token_maps(),
            [("model_0", token_maps[0]), ("model_1", token_maps[1])],
        )

    def test_ensemble_token_selection_without_object_metadata_raises(self) -> None:
        token_map = [
            TokenInfo(
                token_idx=0,
                chain_id="A",
                res_num=1,
                res_name="ALA",
                is_hetatm=False,
                atom_name=None,
            )
        ]
        dialog = self._dialog_with_metadata(
            {
                "kind": "line",
                "token_maps": [token_map],
            }
        )

        with self.assertRaisesRegex(ValueError, "requires token_map_obj_names"):
            dialog._selection_object_token_maps()

    def test_toolbar_navigation_has_no_axis_limit_selection_callbacks(self) -> None:
        class FakeCallbacks:
            def __init__(self) -> None:
                self.connected = []

            def connect(self, signal, callback):
                self.connected.append((signal, callback))
                return len(self.connected)

        axis = types.SimpleNamespace(callbacks=FakeCallbacks())
        figure = types.SimpleNamespace(
            axes=[axis],
            _foldqc_viewer_selection={
                "kind": "line",
                "token_map": [object()],
                "obj_name": "model",
            },
        )
        dialog = PlotDialog.__new__(PlotDialog)
        dialog._figure = figure
        dialog._canvas = types.SimpleNamespace(mpl_connect=lambda *_args: 1)
        dialog._selection_metadata = None
        dialog._selection_ax = None
        dialog._selection_axes = []
        dialog._selector = None
        dialog._selectors = []
        dialog._mpl_cids = []
        dialog._install_selection_controls = lambda: None

        dialog._install_viewer_selection_bridge()

        self.assertEqual(axis.callbacks.connected, [])

    def test_ensemble_site_selection_enables_models_and_shows_site_sticks(self) -> None:
        cmd = types.SimpleNamespace(
            enabled=[],
            disabled=[],
            selections=[],
            shows=[],
            zooms=[],
            refreshes=0,
        )
        cmd.enable = lambda name: cmd.enabled.append(name)
        cmd.disable = lambda name: cmd.disabled.append(name)
        cmd.select = lambda name, expr: cmd.selections.append((name, expr))
        cmd.show = lambda representation, selection: cmd.shows.append(
            (representation, selection)
        )
        cmd.zoom = lambda selection: cmd.zooms.append(selection)

        def refresh():
            cmd.refreshes += 1

        cmd.refresh = refresh
        sys.modules["pymol"].cmd = cmd

        token_maps = [
            [self._token(idx, "model_0") for idx in range(2)],
            [self._token(idx, "model_1") for idx in range(3)],
        ]
        dialog = self._dialog_with_metadata(
            {
                "kind": "ensemble_site_summary",
                "member_obj_names": ["model_0", "model_1"],
                "member_token_maps": token_maps,
                "member_site_indices": [[0], [0, 2]],
                "member_x_positions": [0.0, 1.0],
                "member_widths": [0.9, 0.9],
                "selection_name": "foldqc_ensemble_site",
            }
        )

        self.assertEqual(dialog._members_in_interval(0.6, 1.4), [1])
        dialog._apply_ensemble_site_selection([1])

        self.assertEqual(cmd.enabled, ["model_1", "foldqc_ensemble_site"])
        self.assertEqual(cmd.disabled, ["model_0"])
        self.assertEqual(
            cmd.selections,
            [
                (
                    "foldqc_ensemble_site",
                    "(%model_1 and polymer and chain A and resi 1+3)",
                )
            ],
        )
        self.assertEqual(cmd.shows, [("sticks", "foldqc_ensemble_site")])
        self.assertEqual(cmd.zooms, ["foldqc_ensemble_site"])
        self.assertEqual(cmd.refreshes, 1)

    def test_clear_matrix_selection_resets_plot_and_pymol_state(self) -> None:
        cmd = types.SimpleNamespace(selections=[], refreshes=0)
        cmd.select = lambda name, expr: cmd.selections.append((name, expr))
        cmd.refresh = lambda: setattr(cmd, "refreshes", cmd.refreshes + 1)
        sys.modules["pymol"].cmd = cmd

        action = types.SimpleNamespace(enabled=True)
        action.setEnabled = lambda enabled: setattr(action, "enabled", enabled)
        canvas = types.SimpleNamespace(draws=0)
        canvas.draw_idle = lambda: setattr(canvas, "draws", canvas.draws + 1)
        dialog = self._dialog_with_metadata({"kind": "matrix"})
        dialog._selected_token_indices = [0]
        dialog._selected_row_indices = [0]
        dialog._selected_col_indices = [0]
        dialog._selected_ensemble_member_indices = []
        dialog._selected_bar_group_indices = []
        dialog._matrix_selection_mask = np.ones((1, 1), dtype=bool)
        dialog._highlight_artists = []
        dialog._matrix_boundary_artist = None
        dialog._bar_original_styles = []
        dialog._clear_selection_action = action
        dialog._canvas = canvas

        dialog._clear_plot_selection()

        self.assertEqual(
            cmd.selections,
            [
                ("foldqc_plot_selection", "none"),
                ("foldqc_plot_rows", "none"),
                ("foldqc_plot_cols", "none"),
            ],
        )
        self.assertFalse(np.any(dialog._matrix_selection_mask))
        self.assertFalse(action.enabled)
        self.assertEqual(canvas.draws, 1)

    def test_clear_ensemble_site_reenables_members_without_representation_commands(
        self,
    ) -> None:
        cmd = types.SimpleNamespace(
            selections=[], enabled=[], refreshes=0, shows=[], hides=[], zooms=[]
        )
        cmd.select = lambda name, expr: cmd.selections.append((name, expr))
        cmd.enable = lambda name: cmd.enabled.append(name)
        cmd.refresh = lambda: setattr(cmd, "refreshes", cmd.refreshes + 1)
        cmd.show = lambda *args: cmd.shows.append(args)
        cmd.hide = lambda *args: cmd.hides.append(args)
        cmd.zoom = lambda *args: cmd.zooms.append(args)
        sys.modules["pymol"].cmd = cmd

        action = types.SimpleNamespace(enabled=True)
        action.setEnabled = lambda enabled: setattr(action, "enabled", enabled)
        canvas = types.SimpleNamespace(draw_idle=lambda: None)
        dialog = self._dialog_with_metadata(
            {
                "kind": "ensemble_site_summary",
                "member_obj_names": ["model_0", "model_1"],
                "selection_name": "foldqc_ensemble_site",
            }
        )
        dialog._selected_token_indices = []
        dialog._selected_row_indices = []
        dialog._selected_col_indices = []
        dialog._selected_ensemble_member_indices = [1]
        dialog._selected_bar_group_indices = []
        dialog._matrix_selection_mask = None
        dialog._highlight_artists = []
        dialog._matrix_boundary_artist = None
        dialog._bar_original_styles = []
        dialog._clear_selection_action = action
        dialog._canvas = canvas

        dialog._clear_plot_selection()

        self.assertEqual(cmd.selections, [("foldqc_ensemble_site", "none")])
        self.assertEqual(cmd.enabled, ["model_0", "model_1"])
        self.assertEqual(cmd.shows, [])
        self.assertEqual(cmd.hides, [])
        self.assertEqual(cmd.zooms, [])


if __name__ == "__main__":
    unittest.main()
