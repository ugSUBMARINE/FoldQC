from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import painter  # noqa: E402
from FoldQC import palettes  # noqa: E402


def _token(idx: int):
    return types.SimpleNamespace(
        token_idx=idx,
        chain_id="A",
        res_num=idx + 1,
        is_hetatm=False,
        atom_name=None,
    )


class _Cmd:
    def __init__(self) -> None:
        self.set_color_calls: list[tuple[str, list[float]]] = []
        self.spectrum_calls: list[tuple] = []
        self.color_calls: list[tuple[str, str]] = []
        self.alter_calls: list[tuple[str, str]] = []
        self.delete_calls: list[str] = []
        self.ramp_new_calls: list[tuple[tuple, dict]] = []
        self.rebuild_calls = 0

    def set_color(self, name: str, rgb: list[float]) -> None:
        self.set_color_calls.append((name, rgb))

    def spectrum(self, *args, **kwargs) -> None:
        self.spectrum_calls.append((args, kwargs))

    def color(self, color_name: str, selection: str) -> None:
        self.color_calls.append((color_name, selection))

    def alter(self, obj_name: str, expr: str) -> None:
        self.alter_calls.append((obj_name, expr))

    def rebuild(self) -> None:
        self.rebuild_calls += 1

    def delete(self, name: str) -> None:
        self.delete_calls.append(name)

    def ramp_new(self, *args, **kwargs) -> None:
        self.ramp_new_calls.append((args, kwargs))


class PainterPaletteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_pymol = sys.modules.get("pymol")
        self.cmd = _Cmd()
        self.stored = types.SimpleNamespace()
        sys.modules["pymol"] = types.SimpleNamespace(cmd=self.cmd, stored=self.stored)

    def tearDown(self) -> None:
        if self.old_pymol is None:
            sys.modules.pop("pymol", None)
        else:
            sys.modules["pymol"] = self.old_pymol

    def test_native_reverse_palette_uses_spectrum_name_without_custom_colors(
        self,
    ) -> None:
        painter.paint_property_bulk(
            "obj",
            [_token(0), _token(1)],
            np.array([1.0, 2.0], dtype=np.float32),
            palette="blue_white_red",
            reverse_palette=True,
            rebuild=False,
        )

        self.assertEqual(self.cmd.set_color_calls, [])
        args, kwargs = self.cmd.spectrum_calls[0]
        self.assertEqual(args[:3], ("b", "red_white_blue", "obj"))
        self.assertEqual(kwargs["minimum"], 1.0)
        self.assertEqual(kwargs["maximum"], 2.0)

    def test_viridis_registers_custom_colors_before_spectrum(self) -> None:
        painter.paint_property_bulk(
            "obj",
            [_token(0), _token(1)],
            np.array([1.0, 2.0], dtype=np.float32),
            palette="viridis",
            rebuild=False,
        )

        self.assertGreater(len(self.cmd.set_color_calls), 0)
        args, _kwargs = self.cmd.spectrum_calls[0]
        palette = args[1]
        self.assertIn("foldqc_viridis_f_00", palette)
        self.assertEqual(len(palette.split()), len(self.cmd.set_color_calls))

    def test_categorical_labels_color_exact_integer_b_factors(self) -> None:
        painter.paint_categorical_labels_bulk(
            "obj",
            [_token(0), _token(1), _token(2)],
            np.array([0.0, 1.0, np.nan], dtype=np.float32),
            rebuild=False,
        )

        self.assertEqual(self.cmd.spectrum_calls, [])
        self.assertEqual(
            [name for name, _rgb in self.cmd.set_color_calls],
            ["foldqc_category_000", "foldqc_category_001"],
        )
        self.assertEqual(
            self.cmd.set_color_calls[0][1],
            list(palettes.categorical_color(0)),
        )
        self.assertIn(("foldqc_category_000", "obj and b = 0"), self.cmd.color_calls)
        self.assertIn(("foldqc_category_001", "obj and b = 1"), self.cmd.color_calls)
        self.assertIn(("grey70", "obj and b < 0"), self.cmd.color_calls)

    def test_plddt_class_coloring_uses_shared_palette_definitions(self) -> None:
        painter.paint_plddt_class_coloring("obj", rebuild=False)

        self.assertEqual(
            [name for name, _rgb in self.cmd.set_color_calls],
            [color.pymol_name for color in palettes.PLDDT_CLASS_COLORS],
        )
        self.assertEqual(
            self.cmd.set_color_calls[0][1],
            list(palettes.PLDDT_CLASS_COLORS[0].rgb),
        )
        self.assertIn(
            (
                palettes.PLDDT_CLASS_COLORS[0].pymol_name,
                f"obj and {palettes.PLDDT_CLASS_COLORS[0].bfactor_selection}",
            ),
            self.cmd.color_calls,
        )

    def test_show_colorbar_replaces_single_named_ramp_object(self) -> None:
        self.assertEqual(painter.COLORBAR_OBJECT_NAME, "foldqc_colorbar")

        painter.show_colorbar(
            "blue_white_red",
            False,
            0.0,
            1.0,
        )

        self.assertEqual(self.cmd.delete_calls, [painter.COLORBAR_OBJECT_NAME])
        self.assertEqual(len(self.cmd.ramp_new_calls), 1)
        args, kwargs = self.cmd.ramp_new_calls[0]
        self.assertEqual(
            args,
            (
                painter.COLORBAR_OBJECT_NAME,
                None,
                [0.0, 0.5, 1.0],
                ["blue", "white", "red"],
            ),
        )
        self.assertEqual(kwargs, {"quiet": 1})

    def test_show_colorbar_uses_reversed_palette_order(self) -> None:
        painter.show_colorbar("blue_white_red", True, 2.0, 8.0)

        args, kwargs = self.cmd.ramp_new_calls[0]
        self.assertEqual(
            args,
            (
                painter.COLORBAR_OBJECT_NAME,
                None,
                [2.0, 5.0, 8.0],
                ["red", "white", "blue"],
            ),
        )
        self.assertEqual(kwargs, {"quiet": 1})

    def test_show_colorbar_passes_custom_palette_rgb_stops(self) -> None:
        painter.show_colorbar("viridis", False, 0.0, 7.0)

        args, _kwargs = self.cmd.ramp_new_calls[0]
        self.assertEqual(args[0], painter.COLORBAR_OBJECT_NAME)
        self.assertIsNone(args[1])
        self.assertEqual(args[2], list(np.linspace(0.0, 7.0, 8)))
        self.assertEqual(len(args[3]), 8)
        self.assertIsInstance(args[3][0], list)
        self.assertAlmostEqual(args[3][0][0], 0.267004)


if __name__ == "__main__":
    unittest.main()
