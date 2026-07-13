from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import (
    mol_viewer,  # noqa: E402
    palettes,  # noqa: E402
)
from FoldQC.token_map import TokenInfo, TokenMap  # noqa: E402


def _token(idx: int) -> TokenInfo:
    return TokenInfo(idx, "A", idx + 1, "ALA", False, None)


def _token_map(*tokens: TokenInfo) -> TokenMap:
    return TokenMap(tokens)


class _Cmd:
    def __init__(self) -> None:
        self.set_color_calls: list[tuple[str, list[float]]] = []
        self.spectrum_calls: list[tuple] = []
        self.color_calls: list[tuple[str, str]] = []
        self.alter_calls: list[tuple[str, str]] = []
        self.alter_spaces: list[dict] = []
        self.delete_calls: list[str] = []
        self.ramp_new_calls: list[tuple[tuple, dict]] = []
        self.rebuild_calls = 0
        self.recolor_calls: list[str] = []
        self.color_indices = {"grey70": 1}

    def set_color(self, name: str, rgb: list[float]) -> None:
        self.set_color_calls.append((name, rgb))
        self.color_indices.setdefault(name, len(self.color_indices) + 1)

    def get_color_index(self, name: str) -> int:
        return self.color_indices.get(name, -1)

    def spectrum(self, *args, **kwargs) -> None:
        self.spectrum_calls.append((args, kwargs))

    def color(self, color_name: str, selection: str) -> None:
        self.color_calls.append((color_name, selection))

    def alter(self, obj_name: str, expr: str, *, space=None) -> None:
        self.alter_calls.append((obj_name, expr))
        self.alter_spaces.append(space or {})

    def get_model(self, _obj_name: str):
        return types.SimpleNamespace(
            atom=[
                types.SimpleNamespace(
                    index=index,
                    chain="A",
                    resi=str(index),
                    resn="ALA",
                    name="CA",
                    hetatm=False,
                )
                for index in range(1, 4)
            ]
        )

    def index(self, _obj_name: str):
        return [("obj", index) for index in range(1, 4)]

    def recolor(self, selection: str) -> None:
        self.recolor_calls.append(selection)

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
        mol_viewer.paint_property_bulk(
            "obj",
            _token_map(_token(0), _token(1)),
            np.array([1.0, 2.0], dtype=np.float32),
            palette="blue_white_red",
            reverse_palette=True,
            rebuild=False,
        )

        self.assertEqual(self.cmd.set_color_calls, [])
        args, kwargs = self.cmd.spectrum_calls[0]
        self.assertEqual(args[:3], ("b", "red_white_blue", "(obj)"))
        self.assertEqual(kwargs["minimum"], 1.0)
        self.assertEqual(kwargs["maximum"], 2.0)

    def test_viridis_quantizes_colors_without_spectrum(self) -> None:
        mol_viewer.paint_property_bulk(
            "obj",
            _token_map(_token(0), _token(1)),
            np.array([1.0, 2.0], dtype=np.float32),
            palette="viridis",
            rebuild=False,
        )

        self.assertEqual(len(self.cmd.set_color_calls), 256)
        self.assertEqual(self.cmd.spectrum_calls, [])
        self.assertEqual(self.cmd.recolor_calls, ["(obj)"])
        self.assertIn("color = foldqc_atom_colors[index]", self.cmd.alter_calls[0][1])
        self.assertEqual(
            self.cmd.alter_spaces[0]["foldqc_atom_values"][1:3], [1.0, 2.0]
        )

        self.cmd.set_color_calls.clear()
        mol_viewer.paint_property_bulk(
            "obj",
            _token_map(_token(0), _token(1)),
            np.array([1.0, 2.0], dtype=np.float32),
            palette="viridis",
            rebuild=False,
        )
        self.assertEqual(self.cmd.set_color_calls, [])

    def test_native_batch_uses_one_spectrum_for_all_objects(self) -> None:
        result = mol_viewer.paint_properties_bulk(
            [
                mol_viewer.PaintTarget(
                    "obj_0",
                    _token_map(_token(0), _token(1)),
                    np.array([0.0, 1.0], dtype=np.float32),
                ),
                mol_viewer.PaintTarget(
                    "obj_1",
                    _token_map(_token(0), _token(1)),
                    np.array([2.0, 3.0], dtype=np.float32),
                ),
            ],
            palette="blue_white_red",
            rebuild=False,
        )

        self.assertEqual((result.vmin, result.vmax), (0.0, 3.0))
        self.assertEqual(len(self.cmd.alter_calls), 2)
        self.assertEqual(len(self.cmd.spectrum_calls), 1)
        args, kwargs = self.cmd.spectrum_calls[0]
        self.assertEqual(args[:3], ("b", "blue_white_red", "(obj_0) or (obj_1)"))
        self.assertEqual(kwargs, {"minimum": 0.0, "maximum": 3.0})

    def test_custom_palette_clips_range_reverses_and_uses_nan_color(self) -> None:
        mol_viewer.paint_property_bulk(
            "obj",
            _token_map(_token(0), _token(1), _token(2)),
            np.array([-5.0, np.nan, 5.0], dtype=np.float32),
            palette="viridis",
            reverse_palette=True,
            vmin=0.0,
            vmax=1.0,
            rebuild=False,
        )

        first_name, first_rgb = self.cmd.set_color_calls[0]
        last_name, _last_rgb = self.cmd.set_color_calls[-1]
        self.assertIn("_r_q256_", first_name)
        np.testing.assert_allclose(first_rgb, palettes.VIRIDIS_STOPS[-1])
        atom_colors = self.cmd.alter_spaces[0]["foldqc_atom_colors"]
        self.assertEqual(atom_colors[1], self.cmd.color_indices[first_name])
        self.assertEqual(atom_colors[2], self.cmd.color_indices["grey70"])
        self.assertEqual(atom_colors[3], self.cmd.color_indices[last_name])

    def test_categorical_labels_color_exact_integer_b_factors(self) -> None:
        mol_viewer.paint_categorical_labels_bulk(
            "obj",
            _token_map(_token(0), _token(1), _token(2)),
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
        self.assertIn(
            ("foldqc_category_000", "((obj)) and b = 0"), self.cmd.color_calls
        )
        self.assertIn(
            ("foldqc_category_001", "((obj)) and b = 1"), self.cmd.color_calls
        )
        self.assertIn(("grey70", "((obj)) and b < 0"), self.cmd.color_calls)

    def test_categorical_batch_registers_shared_labels_once(self) -> None:
        mol_viewer.paint_categorical_labels_batch(
            [
                mol_viewer.PaintTarget(
                    "obj_0",
                    _token_map(_token(0), _token(1)),
                    np.array([0.0, 1.0], dtype=np.float32),
                ),
                mol_viewer.PaintTarget(
                    "obj_1",
                    _token_map(_token(0), _token(1)),
                    np.array([1.0, 2.0], dtype=np.float32),
                ),
            ],
            rebuild=False,
        )

        self.assertEqual(len(self.cmd.alter_calls), 2)
        self.assertEqual(
            [name for name, _rgb in self.cmd.set_color_calls],
            [
                "foldqc_category_000",
                "foldqc_category_001",
                "foldqc_category_002",
            ],
        )
        self.assertEqual(len(self.cmd.color_calls), 4)

    def test_plddt_class_coloring_uses_shared_palette_definitions(self) -> None:
        mol_viewer.paint_plddt_class_coloring("obj", rebuild=False)

        self.assertEqual(
            [name for name, _rgb in self.cmd.set_color_calls],
            [
                color.key if color.key == "plddt_nan" else f"plddt_{color.key}"
                for color in palettes.PLDDT_CLASS_COLORS
            ],
        )
        self.assertEqual(
            self.cmd.set_color_calls[0][1],
            list(palettes.PLDDT_CLASS_COLORS[0].rgb),
        )
        self.assertEqual(
            self.cmd.color_calls,
            [
                ("plddt_very_high", "obj and (b>90 or b=90)"),
                ("plddt_high", "obj and ((b<90 and b>70) or b=70)"),
                ("plddt_low", "obj and ((b<70 and b>50) or b=50)"),
                ("plddt_very_low", "obj and ((b<50 and b>0) or b=0)"),
                ("plddt_nan", "obj and (b<0)"),
            ],
        )

    def test_plddt_class_batch_registers_and_colors_classes_once(self) -> None:
        mol_viewer.paint_plddt_class_batch(
            [
                mol_viewer.PaintTarget(
                    "obj_0", _token_map(_token(0)), np.array([0.9], dtype=np.float32)
                ),
                mol_viewer.PaintTarget(
                    "obj_1", _token_map(_token(0)), np.array([0.6], dtype=np.float32)
                ),
            ],
            rebuild=False,
        )

        self.assertEqual(len(self.cmd.alter_calls), 2)
        self.assertEqual(
            len(self.cmd.set_color_calls), len(palettes.PLDDT_CLASS_COLORS)
        )
        self.assertEqual(len(self.cmd.color_calls), len(palettes.PLDDT_CLASS_COLORS))
        self.assertTrue(
            all(
                "(obj_0) or (obj_1)" in selection
                for _name, selection in self.cmd.color_calls
            )
        )
        self.assertAlmostEqual(
            self.cmd.alter_spaces[0]["foldqc_atom_values"][1], 90.0, places=5
        )
        self.assertAlmostEqual(
            self.cmd.alter_spaces[1]["foldqc_atom_values"][1], 60.0, places=5
        )

    def test_show_colorbar_replaces_single_named_ramp_object(self) -> None:
        self.assertEqual(mol_viewer.COLORBAR_OBJECT_NAME, "foldqc_colorbar")

        mol_viewer.show_colorbar(
            "blue_white_red",
            False,
            0.0,
            1.0,
        )

        self.assertEqual(self.cmd.delete_calls, [mol_viewer.COLORBAR_OBJECT_NAME])
        self.assertEqual(len(self.cmd.ramp_new_calls), 1)
        args, kwargs = self.cmd.ramp_new_calls[0]
        self.assertEqual(
            args,
            (
                mol_viewer.COLORBAR_OBJECT_NAME,
                None,
                [0.0, 0.5, 1.0],
                ["blue", "white", "red"],
            ),
        )
        self.assertEqual(kwargs, {"quiet": 1})

    def test_show_colorbar_uses_reversed_palette_order(self) -> None:
        mol_viewer.show_colorbar("blue_white_red", True, 2.0, 8.0)

        args, kwargs = self.cmd.ramp_new_calls[0]
        self.assertEqual(
            args,
            (
                mol_viewer.COLORBAR_OBJECT_NAME,
                None,
                [2.0, 5.0, 8.0],
                ["red", "white", "blue"],
            ),
        )
        self.assertEqual(kwargs, {"quiet": 1})

    def test_show_colorbar_passes_custom_palette_rgb_stops(self) -> None:
        mol_viewer.show_colorbar("viridis", False, 0.0, 7.0)

        args, _kwargs = self.cmd.ramp_new_calls[0]
        self.assertEqual(args[0], mol_viewer.COLORBAR_OBJECT_NAME)
        self.assertIsNone(args[1])
        self.assertEqual(args[2], list(np.linspace(0.0, 7.0, 8)))
        self.assertEqual(len(args[3]), 8)
        self.assertIsInstance(args[3][0], list)
        self.assertAlmostEqual(args[3][0][0], 0.267004)


if __name__ == "__main__":
    unittest.main()
