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

    def test_native_reverse_palette_uses_pymol_name(self) -> None:
        resolved = palettes.resolve_pymol_palette("blue_white_red", reverse=True)

        self.assertEqual(resolved.palette, "red_white_blue")
        self.assertEqual(resolved.custom_colors, ())

    def test_palette_keys_return_curated_base_keys_only(self) -> None:
        keys = palettes.palette_keys()

        self.assertIn("blue_white_red", keys)
        self.assertNotIn("red_white_blue", keys)

    def test_native_green_palette_uses_pymol_reverse_name(self) -> None:
        forward = palettes.resolve_pymol_palette("white_green")
        reverse = palettes.resolve_pymol_palette("white_green", reverse=True)

        self.assertEqual(forward.palette, "white_green")
        self.assertEqual(reverse.palette, "green_white")

    def test_viridis_resolves_to_custom_color_list(self) -> None:
        resolved = palettes.resolve_pymol_palette("viridis", reverse=True)

        self.assertEqual(len(resolved.custom_colors), len(palettes.VIRIDIS_STOPS))
        self.assertEqual(
            resolved.palette,
            " ".join(color.name for color in resolved.custom_colors),
        )
        self.assertTrue(resolved.custom_colors[0].name.startswith("foldqc_viridis_r_"))

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

    def test_plddt_class_colors_define_pymol_and_bar_order(self) -> None:
        self.assertEqual(
            [color.pymol_name for color in palettes.PLDDT_CLASS_COLORS],
            [
                "plddt_very_high",
                "plddt_high",
                "plddt_low",
                "plddt_very_low",
                "plddt_nan",
            ],
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
