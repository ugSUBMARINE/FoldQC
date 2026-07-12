from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import (
    palettes,  # noqa: E402
    plots,  # noqa: E402
)


class PlotTests(unittest.TestCase):
    def test_selection_metadata_default_prefix_is_plugin_scoped(self) -> None:
        fig = plots.make_line_plot(
            np.array([0, 1], dtype=np.int32),
            [("values", np.array([0.5, 0.7], dtype=np.float32), None)],
        )
        try:
            token_map = [
                types.SimpleNamespace(chain_id="A", res_num=1, is_hetatm=False),
                types.SimpleNamespace(chain_id="A", res_num=2, is_hetatm=False),
            ]
            plots.attach_viewer_selection_metadata(
                fig,
                kind="line",
                token_map=token_map,
                obj_name="obj",
                token_maps=[token_map],
                token_map_obj_names=["obj"],
                token_indices=[0, 1],
            )
            self.assertEqual(
                fig._foldqc_viewer_selection["selection_prefix"],
                "foldqc_plot",
            )
            self.assertEqual(
                fig._foldqc_viewer_selection["token_map_obj_names"],
                ["obj"],
            )
        finally:
            plots.plt.close(fig)

    def test_selection_metadata_rejects_mismatched_object_names(self) -> None:
        fig = plots.make_line_plot(
            np.array([0], dtype=np.int32),
            [("values", np.array([0.5], dtype=np.float32), None)],
        )
        try:
            with self.assertRaisesRegex(ValueError, "one-to-one"):
                plots.attach_viewer_selection_metadata(
                    fig,
                    kind="line",
                    token_map=[object()],
                    obj_name="obj",
                    token_maps=[[object()]],
                    token_map_obj_names=["obj", "other"],
                    token_indices=[0],
                )
        finally:
            plots.plt.close(fig)

    def test_selection_metadata_requires_object_names_with_token_maps(self) -> None:
        fig = plots.make_line_plot(
            np.array([0], dtype=np.int32),
            [("values", np.array([0.5], dtype=np.float32), None)],
        )
        try:
            with self.assertRaisesRegex(ValueError, "provided together"):
                plots.attach_viewer_selection_metadata(
                    fig,
                    kind="line",
                    token_map=[object()],
                    obj_name="obj",
                    token_maps=[[object()]],
                    token_indices=[0],
                )
        finally:
            plots.plt.close(fig)

    def test_line_plot_draws_std_band_and_chain_boundaries(self) -> None:
        fig = plots.make_line_plot(
            np.array([0, 1, 5], dtype=np.int32),
            [
                (
                    "mean",
                    np.array([0.5, 0.7, 0.6], dtype=np.float32),
                    np.array([0.1, 0.2, 0.1], dtype=np.float32),
                )
            ],
            chain_boundaries=[3.0],
            chain_labels=[("A", 0.5), ("B", 5.0)],
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_xlabel(), "Token index")
            self.assertEqual(len(ax.collections), 1)
            self.assertTrue(any(line.get_xdata()[0] == 3.0 for line in ax.lines))
            self.assertEqual([text.get_text() for text in ax.texts], ["A", "B"])
        finally:
            plots.plt.close(fig)

    def test_line_plot_std_band_uses_line_color(self) -> None:
        fig = plots.make_line_plot(
            np.array([0, 1], dtype=np.int32),
            [
                (
                    "mean",
                    np.array([0.5, 0.7], dtype=np.float32),
                    np.array([0.1, 0.2], dtype=np.float32),
                    "#6baed6",
                )
            ],
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.lines[0].get_color(), "#6baed6")
            face = ax.collections[0].get_facecolor()[0]
            expected = plots.matplotlib.colors.to_rgba("#6baed6", alpha=0.18)
            np.testing.assert_allclose(face, expected)
        finally:
            plots.plt.close(fig)

    def test_line_plot_single_point_uses_marker(self) -> None:
        fig = plots.make_line_plot(
            np.array([5], dtype=np.int32),
            [("value", np.array([0.7], dtype=np.float32), None)],
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.lines[0].get_marker(), "o")
            self.assertEqual(ax.lines[0].get_linestyle(), "None")
            np.testing.assert_allclose(ax.lines[0].get_xdata(), np.array([5.0]))
            np.testing.assert_allclose(ax.lines[0].get_ydata(), np.array([0.7]))
        finally:
            plots.plt.close(fig)

    def test_line_plot_single_point_std_uses_errorbar_marker(self) -> None:
        fig = plots.make_line_plot(
            np.array([5], dtype=np.int32),
            [
                (
                    "mean",
                    np.array([0.7], dtype=np.float32),
                    np.array([0.2], dtype=np.float32),
                    "#1f77b4",
                )
            ],
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.lines[0].get_marker(), "o")
            self.assertEqual(ax.lines[0].get_color(), "#1f77b4")
            self.assertGreaterEqual(len(ax.collections), 1)
            self.assertIsNotNone(ax.get_legend())
        finally:
            plots.plt.close(fig)

    def test_line_plot_height_uses_golden_ratio(self) -> None:
        fig = plots.make_line_plot(
            np.arange(350, dtype=np.int32),
            [("values", np.linspace(0.0, 1.0, 350), None)],
        )
        try:
            width, height = fig.get_size_inches()
            self.assertAlmostEqual(width / height, plots.GOLDEN_RATIO)
        finally:
            plots.plt.close(fig)

    def test_line_plot_can_force_single_series_legend_and_color(self) -> None:
        fig = plots.make_line_plot(
            np.array([0, 1], dtype=np.int32),
            [("gap", np.array([0.5, 0.7], dtype=np.float32), None, "#1f77b4")],
            show_legend=True,
        )
        try:
            ax = fig.axes[0]
            self.assertIsNotNone(ax.get_legend())
            self.assertEqual(
                [text.get_text() for text in ax.get_legend().texts], ["gap"]
            )
            self.assertEqual(ax.lines[0].get_color(), "#1f77b4")
        finally:
            plots.plt.close(fig)

    def test_matrix_plot_sets_title_and_axis_labels(self) -> None:
        fig = plots.make_matrix_plot(
            np.arange(4, dtype=np.float32).reshape(2, 2),
            title="Test heatmap",
            xlabel="Columns",
            ylabel="Rows",
            colorbar_label="Error",
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_title(), "Test heatmap")
            self.assertEqual(ax.get_xlabel(), "Columns")
            self.assertEqual(ax.get_ylabel(), "Rows")
            self.assertEqual(fig.axes[1].get_ylabel(), "Error")
        finally:
            plots.plt.close(fig)

    def test_matrix_plot_uses_palette_mapping_and_chain_boundaries(self) -> None:
        token_map = [
            types.SimpleNamespace(chain_id="A", res_num=1, is_hetatm=False),
            types.SimpleNamespace(chain_id="B", res_num=2, is_hetatm=False),
        ]
        fig = plots.make_matrix_plot(
            np.arange(4, dtype=np.float32).reshape(2, 2),
            token_map=token_map,
            row_indices=[0, 1],
            col_indices=[1, 0],
            row_chain_boundaries=[0.5],
            col_chain_boundaries=[0.5],
            palette="blue_white_red",
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(
                [tick.get_text() for tick in ax.get_xticklabels()], ["B2", "A1"]
            )
            self.assertTrue(any(line.get_xdata()[0] == 0.5 for line in ax.lines))
            self.assertTrue(any(line.get_ydata()[0] == 0.5 for line in ax.lines))
        finally:
            plots.plt.close(fig)

    def test_square_matrix_plot_uses_square_axes_and_equal_aspect(self) -> None:
        fig = plots.make_matrix_plot(np.zeros((4, 4), dtype=np.float32))
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_box_aspect(), 1)
            self.assertEqual(ax.get_aspect(), 1.0)
        finally:
            plots.plt.close(fig)

    def test_rectangular_matrix_plot_uses_square_axes_and_auto_aspect(self) -> None:
        fig = plots.make_matrix_plot(np.zeros((4, 2), dtype=np.float32))
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_box_aspect(), 1)
            self.assertEqual(ax.get_aspect(), "auto")
        finally:
            plots.plt.close(fig)

    def test_matrix_plot_falls_back_for_unknown_palette(self) -> None:
        fig = plots.make_matrix_plot(
            np.zeros((2, 2), dtype=np.float32),
            title="Palette test",
            palette="not_a_matplotlib_palette",
        )
        try:
            self.assertIn("viridis", fig.axes[0].get_title())
        finally:
            plots.plt.close(fig)

    def test_matrix_plot_uses_greens_and_reversed_viridis(self) -> None:
        green_fig = plots.make_matrix_plot(
            np.zeros((2, 2), dtype=np.float32),
            title="Greens",
            palette="white_green",
        )
        viridis_fig = plots.make_matrix_plot(
            np.zeros((2, 2), dtype=np.float32),
            title="Viridis",
            palette="viridis",
            reverse_palette=True,
        )
        try:
            self.assertEqual(green_fig.axes[0].get_title(), "Greens")
            self.assertEqual(green_fig.axes[0].images[0].cmap.name, "Greens")
            self.assertEqual(
                viridis_fig.axes[0].images[0].cmap.name,
                "viridis_r",
            )
        finally:
            plots.plt.close(green_fig)
            plots.plt.close(viridis_fig)

    def test_matrix_plot_draws_cell_annotations_and_explicit_labels(self) -> None:
        fig = plots.make_matrix_plot(
            np.array([[0.1234, 0.9876]], dtype=np.float32),
            row_labels=["A"],
            col_labels=["A", "B"],
            cell_text=np.array([["0.123", "0.988"]], dtype=object),
            palette="white_green",
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.images[0].cmap.name, "Greens")
            self.assertEqual(
                [tick.get_text() for tick in ax.get_xticklabels()], ["A", "B"]
            )
            self.assertEqual([tick.get_text() for tick in ax.get_yticklabels()], ["A"])
            self.assertEqual([text.get_text() for text in ax.texts], ["0.123", "0.988"])
        finally:
            plots.plt.close(fig)

    def test_matrix_plot_draws_ensemble_cell_annotations(self) -> None:
        fig = plots.make_matrix_plot(
            np.array([[0.8]], dtype=np.float32),
            cell_text=np.array([["0.800 +/- 0.050"]], dtype=object),
        )
        try:
            self.assertEqual(fig.axes[0].texts[0].get_text(), "0.800 +/- 0.050")
        finally:
            plots.plt.close(fig)

    def test_matrix_plot_annotation_color_uses_rendered_cell_luminance(self) -> None:
        light_fig = plots.make_matrix_plot(
            np.array([[0.0]], dtype=np.float32),
            cell_text=np.array([["0.000"]], dtype=object),
            palette="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        dark_fig = plots.make_matrix_plot(
            np.array([[0.0]], dtype=np.float32),
            cell_text=np.array([["0.000"]], dtype=object),
            palette="Greys",
            reverse_palette=True,
            vmin=0.0,
            vmax=1.0,
        )
        try:
            self.assertEqual(light_fig.axes[0].texts[0].get_color(), "black")
            self.assertEqual(dark_fig.axes[0].texts[0].get_color(), "white")
        finally:
            plots.plt.close(light_fig)
            plots.plt.close(dark_fig)

    def test_plddt_class_bar_plot_counts_percentages_and_order(self) -> None:
        fig = plots.make_plddt_class_bar_plot(
            ["very low", "low", "high", "very high"],
            [2, 0, 1, 3],
            total=6,
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(
                [tick.get_text() for tick in ax.get_xticklabels()],
                ["very low", "low", "high", "very high"],
            )
            self.assertEqual(
                [int(patch.get_height()) for patch in ax.patches],
                [2, 0, 1, 3],
            )
            labels = [text.get_text() for text in ax.texts]
            self.assertIn("0\n0.0%", labels)
            self.assertIn("3\n50.0%", labels)
            np.testing.assert_allclose(
                ax.patches[0].get_facecolor()[0:3],
                palettes.PLDDT_CLASS_BAR_COLORS[0],
            )
        finally:
            plots.plt.close(fig)

    def test_categorical_bar_plot_uses_supplied_colors(self) -> None:
        fig = plots.make_categorical_bar_plot(
            ["0", "1"],
            [3, 2],
            colors=[(0.1, 0.2, 0.3), (0.8, 0.7, 0.6)],
        )
        try:
            ax = fig.axes[0]
            np.testing.assert_allclose(
                ax.patches[0].get_facecolor()[0:3],
                (0.1, 0.2, 0.3),
            )
            np.testing.assert_allclose(
                ax.patches[1].get_facecolor()[0:3],
                (0.8, 0.7, 0.6),
            )
        finally:
            plots.plt.close(fig)

    def test_histogram_plot_uses_sqrt_bin_count_capped_at_50(self) -> None:
        fig = plots.make_histogram_plot(np.arange(10000, dtype=np.float32))
        try:
            self.assertEqual(len(fig.axes[0].patches), plots.MAX_HISTOGRAM_BINS)
        finally:
            plots.plt.close(fig)

    def test_histogram_plot_ignores_nonfinite_values(self) -> None:
        fig = plots.make_histogram_plot(
            np.array([1.0, 2.0, np.nan, np.inf], dtype=np.float32)
        )
        try:
            total = sum(int(patch.get_height()) for patch in fig.axes[0].patches)
            self.assertEqual(total, 2)
        finally:
            plots.plt.close(fig)

    def test_ensemble_site_summary_plot_creates_grouped_metric_bars(self) -> None:
        fig = plots.make_ensemble_site_summary_plot(
            ["model_0", "model_1"],
            [
                ("mean pLDDT", np.array([0.8, 0.7], dtype=np.float32), "steelblue"),
                ("PAE mean", np.array([4.0, 5.0], dtype=np.float32), "tomato"),
                ("PDE mean", np.array([6.0, 7.0], dtype=np.float32), "goldenrod"),
            ],
        )
        try:
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(len(fig.axes[0].patches), 2)
            self.assertEqual(len(fig.axes[1].patches), 4)
            self.assertEqual(
                [tick.get_text() for tick in fig.axes[1].get_xticklabels()],
                ["model_0", "model_1"],
            )
            self.assertEqual(
                [text.get_text() for text in fig.axes[0].get_legend().texts],
                ["mean pLDDT"],
            )
            self.assertEqual(
                [text.get_text() for text in fig.axes[1].get_legend().texts],
                ["PAE mean", "PDE mean"],
            )
        finally:
            plots.plt.close(fig)

    def test_ensemble_site_summary_plot_uses_one_panel_for_plddt_only(self) -> None:
        fig = plots.make_ensemble_site_summary_plot(
            ["model_0", "model_1"],
            [("mean pLDDT", np.array([0.8, 0.7], dtype=np.float32), "steelblue")],
        )
        try:
            self.assertEqual(len(fig.axes), 1)
            self.assertEqual(fig.axes[0].get_ylabel(), "pLDDT")
        finally:
            plots.plt.close(fig)

    def test_binding_site_fingerprint_labels_include_residue_name(self) -> None:
        token_map = [
            types.SimpleNamespace(chain_id="A", res_num=42, res_name="ASP"),
            types.SimpleNamespace(chain_id="B", res_num=107, res_name="TYR"),
        ]
        fig = plots.make_binding_site_fingerprint(
            token_map,
            [0, 1],
            plddt=np.array([0.8, 0.9], dtype=np.float32),
        )
        try:
            labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
            self.assertEqual(labels, ["Asp-A42", "Tyr-B107"])
        finally:
            plots.plt.close(fig)

    def test_binding_site_fingerprint_height_uses_golden_ratio(self) -> None:
        token_map = [
            types.SimpleNamespace(chain_id="A", res_num=i, res_name="ASP")
            for i in range(10)
        ]
        fig = plots.make_binding_site_fingerprint(
            token_map,
            list(range(10)),
            plddt=np.linspace(0.5, 0.9, 10, dtype=np.float32),
        )
        try:
            width, height = fig.get_size_inches()
            self.assertAlmostEqual(width / height, plots.GOLDEN_RATIO)
        finally:
            plots.plt.close(fig)

    def test_binding_site_fingerprint_splits_scores_from_error_metrics(self) -> None:
        token_map = [
            types.SimpleNamespace(chain_id="A", res_num=42, res_name="ASP"),
            types.SimpleNamespace(chain_id="B", res_num=107, res_name="TYR"),
        ]
        fig = plots.make_binding_site_fingerprint(
            token_map,
            [0, 1],
            plddt=np.array([0.8, 0.9], dtype=np.float32),
            pae_to_ligand=np.array([4.0, 5.0], dtype=np.float32),
            pae_from_ligand=np.array([5.0, 6.0], dtype=np.float32),
            pde_to_ligand=np.array([6.0, 7.0], dtype=np.float32),
        )
        try:
            self.assertEqual(len(fig.axes), 2)
            width, height = fig.get_size_inches()
            self.assertAlmostEqual(
                height,
                (width / plots.GOLDEN_RATIO) * plots.STACKED_BAR_HEIGHT_MULTIPLIER,
            )
            self.assertEqual(
                [text.get_text() for text in fig.axes[0].get_legend().texts],
                ["pLDDT"],
            )
            self.assertEqual(
                [text.get_text() for text in fig.axes[1].get_legend().texts],
                ["PAE row mean (Å)", "PAE column mean (Å)", "PDE mean (Å)"],
            )
            self.assertEqual(len(fig.axes[0].patches), 2)
            self.assertEqual(len(fig.axes[1].patches), 6)
        finally:
            plots.plt.close(fig)

    def test_binding_site_fingerprint_accepts_error_bars(self) -> None:
        token_map = [
            types.SimpleNamespace(chain_id="A", res_num=42, res_name="ASP"),
            types.SimpleNamespace(chain_id="B", res_num=107, res_name="TYR"),
        ]
        fig = plots.make_binding_site_fingerprint(
            token_map,
            [0, 1],
            plddt=np.array([0.8, 0.9], dtype=np.float32),
            plddt_std=np.array([0.05, 0.04], dtype=np.float32),
        )
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_title(), "Binding-site confidence fingerprint")
            self.assertGreaterEqual(len(ax.containers), 1)
        finally:
            plots.plt.close(fig)

    def test_binding_site_fingerprint_accepts_interaction_probability(self) -> None:
        token_map = [
            types.SimpleNamespace(chain_id="A", res_num=42, res_name="ASP"),
            types.SimpleNamespace(chain_id="B", res_num=107, res_name="TYR"),
        ]
        fig = plots.make_binding_site_fingerprint(
            token_map,
            [0, 1],
            interaction_prob_to_ligand=np.array([0.2, 0.9], dtype=np.float32),
        )
        try:
            labels = [text.get_text() for text in fig.axes[0].get_legend().texts]
            self.assertIn("Interaction probability", labels)
        finally:
            plots.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
