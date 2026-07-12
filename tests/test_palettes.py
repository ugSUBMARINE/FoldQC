from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import palettes  # noqa: E402


class PaletteTests(unittest.TestCase):
    def test_curated_gui_palettes_exclude_reversed_duplicates(self) -> None:
        keys = [spec.key for spec in palettes.iter_gui_palettes()]

        self.assertIn("blue_white_red", keys)
        self.assertIn("white_green", keys)
        self.assertIn("viridis", keys)
        self.assertNotIn("rainbow", keys)
        self.assertNotIn("rainbow2", keys)
        self.assertNotIn("red_white_blue", keys)
        self.assertNotIn("magenta_white_cyan", keys)

    def test_palette_specs_do_not_expose_viewer_encodings(self) -> None:
        spec = next(
            spec for spec in palettes.PALETTE_SPECS if spec.key == "blue_white_red"
        )

        self.assertFalse(hasattr(spec, "pymol"))
        self.assertEqual(spec.mpl, "coolwarm")

    def test_palette_keys_return_curated_base_keys_only(self) -> None:
        keys = palettes.palette_keys()

        self.assertIn("blue_white_red", keys)
        self.assertNotIn("red_white_blue", keys)

    def test_viridis_keeps_generic_rgb_stops(self) -> None:
        spec = next(spec for spec in palettes.PALETTE_SPECS if spec.key == "viridis")

        self.assertEqual(spec.rgb_stops, palettes.VIRIDIS_STOPS)

    def test_categorical_color_uses_shared_palette_and_deterministic_fallback(
        self,
    ) -> None:
        self.assertEqual(
            palettes.categorical_color(0),
            palettes.CATEGORICAL_STOPS[0],
        )
        self.assertEqual(
            palettes.categorical_color(19),
            palettes.CATEGORICAL_STOPS[19],
        )
        self.assertEqual(palettes.categorical_color(99), palettes.categorical_color(99))

    def test_plddt_class_colors_define_thresholds_and_bar_order(self) -> None:
        self.assertEqual(
            [color.minimum for color in palettes.PLDDT_CLASS_COLORS],
            [90.0, 70.0, 50.0, 0.0, None],
        )
        self.assertEqual(
            palettes.PLDDT_CLASS_BAR_COLORS,
            tuple(
                color.rgb
                for color in reversed(palettes.PLDDT_CLASS_COLORS)
                if color.key != "plddt_nan"
            ),
        )

    def test_matplotlib_reverse_uses_reverse_colormap_name(self) -> None:
        cmap, used_fallback = palettes.resolve_matplotlib_cmap("viridis", reverse=True)

        self.assertFalse(used_fallback)
        self.assertEqual(cmap.name, "viridis_r")


if __name__ == "__main__":
    unittest.main()
