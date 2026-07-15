from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import properties as P  # noqa: E402
from FoldQC.confidence import PredictionConfidence  # noqa: E402
from FoldQC.token_map import TokenInfo, TokenMap  # noqa: E402


def _token_map(*chain_ids: str) -> TokenMap:
    return TokenMap(
        tuple(
            TokenInfo(index, chain_id, index + 1, "ALA", False, None)
            for index, chain_id in enumerate(chain_ids)
        )
    )


class PropertiesTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("scipy") is not None,
        "scipy is required for complete-linkage PAE domain labels",
    )
    def test_pae_domain_labels_complete_linkage_avoids_diagonal_bridge(self) -> None:
        pae = np.array(
            [
                [0.0, 1.0, 4.0, 8.0],
                [1.0, 0.0, 1.0, 4.0],
                [4.0, 1.0, 0.0, 1.0],
                [8.0, 4.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        labels = P.pae_domain_labels(pae, threshold=2.0, method="complete_linkage")

        self.assertEqual(labels.dtype, np.float32)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_pae_continuous_affinity_preserves_distance_strength(self) -> None:
        sym_pae = np.array(
            [
                [0.0, 1.0, 4.0],
                [1.0, 0.0, 8.0],
                [4.0, 8.0, 0.0],
            ],
            dtype=np.float32,
        )

        affinity = P._pae_continuous_affinity(sym_pae, threshold=4.0)

        np.testing.assert_allclose(np.diag(affinity), np.ones(3))
        self.assertGreater(affinity[0, 1], affinity[0, 2])
        self.assertGreater(affinity[0, 2], affinity[1, 2])
        self.assertGreater(affinity[1, 2], 0.0)

    @unittest.skipUnless(
        importlib.util.find_spec("scipy") is not None,
        "scipy is required for spectral eigengap estimation",
    )
    def test_spectral_cluster_count_uses_eigengap_with_cap(self) -> None:
        affinity = np.eye(15, dtype=np.float64)

        count = P._spectral_cluster_count_from_eigengap(affinity, max_clusters=12)

        self.assertLessEqual(count, 12)
        self.assertGreaterEqual(count, 2)

    def test_pae_column_to_selection_uses_selected_rows(self) -> None:
        pae = np.array(
            [
                [0.0, 1.0, 8.0, 10.0],
                [2.0, 0.0, 6.0, 12.0],
                [4.0, 3.0, 0.0, 14.0],
                [5.0, 7.0, 9.0, 0.0],
            ],
            dtype=np.float32,
        )

        values = P.pae_column_to_selection(pae, [0, 2])

        np.testing.assert_allclose(
            values, np.array([2.0, 2.0, 4.0, 12.0], dtype=np.float32)
        )

    def test_pae_chain_summary_returns_row_and_column_within_other_means(
        self,
    ) -> None:
        pae = np.array(
            [
                [0.0, 2.0, 10.0],
                [4.0, 0.0, 12.0],
                [6.0, 8.0, 0.0],
            ],
            dtype=np.float32,
        )
        token_map = _token_map("A", "A", "B")

        row_within, row_other, col_within, col_other = P.pae_chain_summary(
            pae, token_map
        )

        np.testing.assert_allclose(
            row_within, np.array([1.0, 2.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            row_other, np.array([10.0, 12.0, 7.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            col_within, np.array([2.0, 1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            col_other, np.array([6.0, 8.0, 11.0], dtype=np.float32)
        )

    def test_pae_symmetric_mean_within_selection_only_sets_selected_tokens(
        self,
    ) -> None:
        pae = np.array(
            [
                [0.0, 1.0, 8.0, 10.0],
                [2.0, 0.0, 6.0, 12.0],
                [4.0, 3.0, 0.0, 14.0],
                [5.0, 7.0, 9.0, 0.0],
            ],
            dtype=np.float32,
        )

        values = P.pae_symmetric_mean_within_selection(pae, [1, 3])

        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isnan(values[2]))
        np.testing.assert_allclose(values[[1, 3]], np.array([4.75, 4.75]))

    def test_pae_symmetric_to_selection_for_contacts_only_sets_contacts(self) -> None:
        pae = np.array(
            [
                [0.0, 1.0, 8.0, 10.0],
                [2.0, 0.0, 6.0, 12.0],
                [4.0, 3.0, 0.0, 14.0],
                [5.0, 7.0, 9.0, 0.0],
            ],
            dtype=np.float32,
        )

        values = P.pae_symmetric_to_selection_for_contacts(
            pae,
            ref_indices=[0, 2],
            contact_indices=[1, 3],
        )

        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isnan(values[2]))
        np.testing.assert_allclose(values[[1, 3]], np.array([3.0, 9.5]))

    def test_pde_mean_within_selection_only_sets_selected_tokens(self) -> None:
        pde = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 4.0, 5.0],
                [2.0, 4.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ],
            dtype=np.float32,
        )

        values = P.pde_mean_within_selection(pde, [1, 3])

        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isnan(values[2]))
        np.testing.assert_allclose(values[[1, 3]], np.array([2.5, 2.5]))

    def test_pde_mean_within_selection_rejects_empty_selection(self) -> None:
        with self.assertRaises(ValueError):
            P.pde_mean_within_selection(np.zeros((2, 2), dtype=np.float32), [])

    def test_pde_chain_summary_returns_within_and_other_chain_means(self) -> None:
        pde = np.array(
            [
                [0.0, 2.0, 10.0, 12.0],
                [4.0, 0.0, 14.0, 16.0],
                [6.0, 8.0, 0.0, 18.0],
                [10.0, 12.0, 20.0, 0.0],
            ],
            dtype=np.float32,
        )
        token_map = _token_map("A", "A", "B", "B")

        within, other = P.pde_chain_summary(pde, token_map)

        np.testing.assert_allclose(
            within, np.array([1.0, 2.0, 9.0, 10.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            other, np.array([11.0, 15.0, 7.0, 11.0], dtype=np.float32)
        )

    def test_chain_summary_rejects_token_map_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "token_map length 1"):
            P.pae_chain_summary(
                np.zeros((2, 2), dtype=np.float32),
                [types.SimpleNamespace(token_idx=0, chain_id="A")],
            )

    def test_pde_contact_filtered_averages_only_reference_contacts(self) -> None:
        pde = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 4.0, 5.0],
                [2.0, 4.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ],
            dtype=np.float32,
        )
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        values = P.pde_contact_filtered(
            pde,
            coords,
            ref_indices=[1, 3],
            distance_cutoff=1.5,
        )

        np.testing.assert_allclose(values[[0, 1, 3]], np.array([1.0, 0.0, 0.0]))
        self.assertTrue(np.isnan(values[2]))

    def test_pde_contact_filtered_rejects_empty_selection(self) -> None:
        with self.assertRaises(ValueError):
            P.pde_contact_filtered(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
                [],
            )

    def test_pde_to_selection_for_contacts_only_sets_contact_tokens(self) -> None:
        pde = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 4.0, 5.0],
                [2.0, 4.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ],
            dtype=np.float32,
        )

        values = P.pde_to_selection_for_contacts(
            pde,
            ref_indices=[0, 2],
            contact_indices=[1, 3],
        )

        np.testing.assert_allclose(values[[1, 3]], np.array([2.5, 4.5]))
        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isnan(values[2]))

    def test_pde_to_selection_for_contacts_all_nan_when_no_contacts(self) -> None:
        values = P.pde_to_selection_for_contacts(
            np.zeros((2, 2), dtype=np.float32),
            ref_indices=[0],
            contact_indices=[],
        )

        self.assertTrue(np.isnan(values).all())

    def test_contact_probability_mean(self) -> None:
        contact_probs = np.array(
            [[1.0, 0.2, 0.4], [0.2, 1.0, 0.8]],
            dtype=np.float32,
        )

        values = P.contact_probability_mean(contact_probs)

        np.testing.assert_allclose(values, np.array([0.53333336, 0.6666667]))

    def test_contact_probability_to_selection(self) -> None:
        contact_probs = np.array(
            [[1.0, 0.2, 0.4], [0.2, 1.0, 0.8], [0.4, 0.8, 1.0]],
            dtype=np.float32,
        )

        values = P.contact_probability_to_selection(contact_probs, [0, 2])

        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isnan(values[2]))
        np.testing.assert_allclose(values[1], 0.5)

    def test_contact_probability_to_selection_rejects_empty_selection(self) -> None:
        with self.assertRaises(ValueError):
            P.contact_probability_to_selection(np.zeros((2, 2), dtype=np.float32), [])

    def test_pair_chain_iptm_matrix_uses_token_map_chain_order(self) -> None:
        token_map = _token_map("A", "A", "L")
        confidence = PredictionConfidence(
            pair_chain_iptm=np.array([[0.9, 0.7], [0.6, np.nan]])
        )

        matrix, labels = P.pair_chain_iptm_matrix(confidence, token_map)

        self.assertEqual(labels, ["A", "L"])
        np.testing.assert_allclose(
            matrix,
            np.array([[0.9, 0.7], [0.6, np.nan]], dtype=np.float32),
            equal_nan=True,
        )

    def test_pair_chain_iptm_matrix_fills_zero_diagonal_from_chain_ptm(self) -> None:
        token_map = _token_map("A", "B")
        confidence = PredictionConfidence(
            chain_ptm=np.array([0.91, 0.82]),
            pair_chain_iptm=np.array([[0.0, 0.7], [0.6, np.nan]]),
        )

        matrix, labels = P.pair_chain_iptm_matrix(confidence, token_map)

        self.assertEqual(labels, ["A", "B"])
        np.testing.assert_allclose(
            matrix,
            np.array([[0.91, 0.7], [0.6, 0.82]], dtype=np.float32),
        )

    def test_pair_chain_iptm_matrix_rejects_missing_typed_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "pairwise chain ipTM"):
            P.pair_chain_iptm_matrix(PredictionConfidence(), _token_map("A"))


if __name__ == "__main__":
    unittest.main()
