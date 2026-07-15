from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import palettes, plot_data
from FoldQC.confidence import PredictionConfidence
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap


def _token(idx: int, *, chain_id: str = "A", is_hetatm: bool = False) -> TokenInfo:
    return TokenInfo(
        token_idx=idx,
        chain_id=chain_id,
        residue_id=ResidueId(idx + 1),
        res_name="LIG" if is_hetatm else "ALA",
        is_hetatm=is_hetatm,
        atom_name=f"C{idx}" if is_hetatm else None,
    )


def test_chain_boundaries_preserve_selected_positions_and_blank_labels() -> None:
    token_map = TokenMap(
        (
            _token(0, chain_id="A"),
            _token(1, chain_id="A"),
            _token(2, chain_id=""),
            _token(3, chain_id="B"),
        )
    )

    boundaries, labels = plot_data.chain_boundaries(
        token_map, [0, 2, 3], original_x=True
    )

    assert boundaries == [1.0, 2.5]
    assert labels == [("A", 0.0), ("(blank)", 2.0), ("B", 3.0)]
    assert plot_data.chain_boundaries(token_map, []) == ([], [])


def test_has_multiple_token_chains_counts_hetatm_chains() -> None:
    assert plot_data.has_multiple_token_chains(
        TokenMap((_token(0, chain_id="A"), _token(1, chain_id="L", is_hetatm=True)))
    )
    assert not plot_data.has_multiple_token_chains(
        TokenMap((_token(0, chain_id="A"), _token(1, chain_id="A", is_hetatm=True)))
    )


def test_nan_mean_std_handles_vectors_matrices_and_missing_arrays() -> None:
    mean, std = plot_data.nan_mean_std(
        [
            np.array([1.0, np.nan, 3.0], dtype=np.float32),
            None,
            np.array([3.0, 5.0, np.nan], dtype=np.float32),
        ],
        3,
    )

    np.testing.assert_allclose(mean, np.array([2.0, 5.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(std, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    matrix_mean, matrix_std = plot_data.nan_mean_std(
        [
            np.array([[1.0, 3.0], [5.0, np.nan]], dtype=np.float32),
            np.array([[3.0, 7.0], [np.nan, 9.0]], dtype=np.float32),
        ],
        4,
    )
    np.testing.assert_allclose(
        matrix_mean, np.array([[2.0, 5.0], [5.0, 9.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        matrix_std, np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    )

    assert plot_data.nan_mean_std([None, None], 3) == (None, None)


def test_summary_series_for_single_model_returns_expected_labels_and_values() -> None:
    token_map = TokenMap(
        (
            _token(0, chain_id="A"),
            _token(1, chain_id="A"),
            _token(2, chain_id="B"),
        )
    )
    data = types.SimpleNamespace(
        pae=np.array(
            [
                [0.0, 2.0, 10.0],
                [4.0, 0.0, 12.0],
                [6.0, 8.0, 0.0],
            ],
            dtype=np.float32,
        ),
        pde=None,
    )

    series = plot_data.summary_series_for_data("pae", data, token_map)

    assert [item[0] for item in series] == [
        "row gap (other - within)",
        "column gap (other - within)",
    ]
    np.testing.assert_allclose(
        series[0][1], np.array([9.0, 10.0, 7.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        series[1][1], np.array([4.0, 7.0, 11.0], dtype=np.float32)
    )
    assert series[0][3] == "#1f77b4"
    assert series[1][3] == "#6baed6"


def test_summary_series_for_ensemble_aggregates_mean_and_std() -> None:
    token_map = TokenMap((_token(0, chain_id="A"), _token(1, chain_id="B")))
    data_items = [
        types.SimpleNamespace(pde=np.array([[0.0, 2.0], [4.0, 0.0]], dtype=np.float32)),
        types.SimpleNamespace(pde=np.array([[0.0, 4.0], [8.0, 0.0]], dtype=np.float32)),
    ]

    series = plot_data.summary_series_for_ensemble("pde", data_items, token_map)

    assert [item[0] for item in series] == [
        "gap (other - within) mean",
    ]
    np.testing.assert_allclose(series[0][1], np.array([3.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(series[0][2], np.array([1.0, 2.0], dtype=np.float32))


def test_plddt_class_distribution_groups_for_fraction_percent_and_nonfinite() -> None:
    labels, counts, groups, total = plot_data.plddt_class_distribution_groups(
        np.array([0.2, 0.55, 0.75, 0.95, np.nan], dtype=np.float32),
        [10, 11, 12, 13, 14],
    )

    assert labels == ["very low", "low", "high", "very high"]
    assert counts == [1, 1, 1, 1]
    assert groups == [[10], [11], [12], [13]]
    assert total == 4

    _labels, counts, groups, total = plot_data.plddt_class_distribution_groups(
        np.array([45.0, 65.0, 85.0, 95.0], dtype=np.float32),
        [0, 1, 2, 3],
    )
    assert counts == [1, 1, 1, 1]
    assert groups == [[0], [1], [2], [3]]
    assert total == 4

    labels, counts, groups, total = plot_data.plddt_class_distribution_groups(
        np.array([np.nan], dtype=np.float32), [0]
    )
    assert labels == ["very low", "low", "high", "very high"]
    assert counts == [0, 0, 0, 0]
    assert groups == [[], [], [], []]
    assert total == 0


def test_domain_label_distribution_groups_use_categorical_colors() -> None:
    labels, counts, groups, colors = plot_data.domain_label_distribution_groups(
        np.array([2.0, 1.0, 1.0, np.nan, 2.2], dtype=np.float32),
        [0, 1, 2, 3, 4],
    )

    assert labels == ["1", "2"]
    assert counts == [2, 2]
    assert groups == [[1, 2], [0, 4]]
    assert colors == [palettes.categorical_color(1), palettes.categorical_color(2)]


def test_histogram_distribution_groups_filter_nonfinite_and_include_last_edge() -> None:
    edges, groups, positions, widths = plot_data.histogram_distribution_groups(
        np.array([0.0, 1.0, 2.0, 3.0, np.nan], dtype=np.float32),
        [10, 11, 12, 13, 14],
        max_bins=2,
    )

    np.testing.assert_allclose(edges, np.array([0.0, 1.5, 3.0]))
    assert groups == [[10, 11], [12, 13]]
    assert positions == [0.75, 2.25]
    assert widths == [1.5, 1.5]

    counts, bins = plot_data.compute_histogram_bins(np.arange(10000), max_bins=50)
    assert len(counts) == 50
    assert len(bins) == 51
    with pytest.raises(ValueError, match="No finite values"):
        plot_data.compute_histogram_bins(np.array([np.nan]))


def test_format_matrix_cell_text_handles_std_and_nonfinite() -> None:
    matrix = np.array([[0.91234, np.nan], [0.7, 0.6]], dtype=np.float32)
    std = np.array([[0.1, 0.2], [np.nan, 0.05]], dtype=np.float32)

    assert plot_data.format_matrix_cell_text(matrix).tolist() == [
        ["0.912", None],
        ["0.700", "0.600"],
    ]
    assert plot_data.format_matrix_cell_text(matrix, std).tolist() == [
        ["0.912 +/- 0.100", None],
        ["0.700 +/- n/a", "0.600 +/- 0.050"],
    ]


def test_fingerprint_arrays_and_series_for_single_model() -> None:
    data = types.SimpleNamespace(
        token_plddt=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        token_plddt_source="structure_b_factor",
        pae=np.arange(9, dtype=np.float32).reshape(3, 3),
        pde=np.arange(9, dtype=np.float32).reshape(3, 3) + 100.0,
        contact_probs=np.array(
            [[1.0, 0.1, 0.3], [0.1, 1.0, 0.5], [0.3, 0.5, 1.0]],
            dtype=np.float32,
        ),
    )

    series = plot_data.fingerprint_series_for_single(data, [0, 2])

    np.testing.assert_array_equal(series["plddt"], data.token_plddt)
    np.testing.assert_allclose(
        series["pae_to_ligand"], np.array([1.0, 4.0, 7.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        series["pae_from_ligand"], np.array([3.0, 4.0, 5.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        series["pde_to_ligand"], np.array([101.0, 104.0, 107.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        series["interaction_prob_to_ligand"],
        np.array([np.nan, 0.3, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    assert series["plddt_std"] is None


def test_fingerprint_series_for_ensemble_aggregates_mean_and_std() -> None:
    data_items = [
        types.SimpleNamespace(
            token_plddt=np.array([0.8, 0.6], dtype=np.float32),
            token_plddt_source="provider_token",
            pae=None,
            pde=None,
            contact_probs=None,
        ),
        types.SimpleNamespace(
            token_plddt=np.array([1.0, 0.2], dtype=np.float32),
            token_plddt_source="provider_token",
            pae=None,
            pde=None,
            contact_probs=None,
        ),
    ]

    series = plot_data.fingerprint_series_for_ensemble(data_items, [0], size=2)

    np.testing.assert_allclose(series["plddt"], np.array([0.9, 0.4], dtype=np.float32))
    np.testing.assert_allclose(
        series["plddt_std"], np.array([0.1, 0.2], dtype=np.float32)
    )
    assert series["pae_to_ligand"] is None


def test_site_summary_values_use_finite_site_means() -> None:
    data = types.SimpleNamespace(
        token_plddt=np.array([0.8, np.nan, 0.6], dtype=np.float32),
        token_plddt_source="structure_b_factor",
        pae=np.array(
            [[0.0, 2.0, 4.0], [6.0, 0.0, 8.0], [10.0, 12.0, 0.0]],
            dtype=np.float32,
        ),
        pde=np.array(
            [[0.0, 1.0, np.nan], [3.0, 0.0, 5.0], [7.0, 9.0, 0.0]],
            dtype=np.float32,
        ),
    )

    assert plot_data.finite_mean(np.array([np.nan, 2.0, 4.0])) == 3.0
    assert np.isnan(plot_data.finite_mean(np.array([np.nan])))
    assert np.isnan(plot_data.within_site_matrix_mean(None, [0, 1]))
    assert np.isnan(plot_data.within_site_matrix_mean(data.pae, [0]))

    summary = plot_data.site_summary_values(data, [0, 1, 2])

    assert summary["plddt"] == pytest.approx(0.7)
    assert summary["pae"] == pytest.approx(7.0)
    assert summary["pde"] == pytest.approx(5.0)


def test_chain_iptm_matrix_plot_data_single_and_ensemble() -> None:
    token_map = TokenMap((_token(0, chain_id="A"), _token(1, chain_id="B")))
    data = types.SimpleNamespace(
        confidence=PredictionConfidence(
            pair_chain_iptm=np.array([[0.91234, 0.81234], [0.71234, 0.61234]])
        )
    )

    matrix, rows, cols, title, label, row_labels, col_labels, text = (
        plot_data.chain_iptm_matrix_plot_data(
            target_kind="single",
            data=data,
            token_map=token_map,
            title="Pairwise chain ipTM",
            label="ipTM",
        )
    )

    np.testing.assert_allclose(
        matrix, np.array([[0.91234, 0.81234], [0.71234, 0.61234]], dtype=np.float32)
    )
    assert rows == [0, 1]
    assert cols == [0, 1]
    assert title == "Pairwise chain ipTM"
    assert label == "ipTM"
    assert row_labels == ["A", "B"]
    assert col_labels == ["A", "B"]
    assert text.tolist() == [["0.912", "0.812"], ["0.712", "0.612"]]

    members = [
        types.SimpleNamespace(rank=0, data=data, token_map=token_map),
        types.SimpleNamespace(
            rank=1,
            token_map=token_map,
            data=types.SimpleNamespace(
                confidence=PredictionConfidence(
                    pair_chain_iptm=np.array([[1.0, 0.8], [0.6, 0.4]])
                )
            ),
        ),
    ]

    matrix, _rows, _cols, title, _label, row_labels, _col_labels, text = (
        plot_data.chain_iptm_matrix_plot_data(
            target_kind="ensemble_group",
            data=None,
            token_map=token_map,
            title="Pairwise chain ipTM",
            label="ipTM",
            members=members,
        )
    )

    np.testing.assert_allclose(
        matrix,
        np.array([[0.95617, 0.80617], [0.65617, 0.50617]], dtype=np.float32),
    )
    assert "ensemble mean" in title
    assert row_labels == ["A", "B"]
    assert text[0, 0].startswith("0.956 +/-")

    with pytest.raises(ValueError, match="Chain confidence data"):
        plot_data.chain_iptm_matrix_plot_data(
            target_kind="single",
            data=types.SimpleNamespace(confidence=None),
            token_map=token_map,
            title="Pairwise chain ipTM",
            label="ipTM",
        )


def test_chain_iptm_matrix_plot_data_uses_chain_ptm_diagonal_for_ensemble() -> None:
    token_map = TokenMap((_token(0, chain_id="A"), _token(1, chain_id="B")))
    members = [
        types.SimpleNamespace(
            rank=0,
            token_map=token_map,
            data=types.SimpleNamespace(
                confidence=PredictionConfidence(
                    chain_ptm=np.array([0.9, 0.8]),
                    pair_chain_iptm=np.array([[0.9, 0.7], [0.7, 0.8]]),
                )
            ),
        ),
        types.SimpleNamespace(
            rank=1,
            token_map=token_map,
            data=types.SimpleNamespace(
                confidence=PredictionConfidence(
                    chain_ptm=np.array([1.0, 0.6]),
                    pair_chain_iptm=np.array([[1.0, 0.5], [0.5, 0.6]]),
                )
            ),
        ),
    ]

    matrix, _rows, _cols, _title, _label, _row_labels, _col_labels, text = (
        plot_data.chain_iptm_matrix_plot_data(
            target_kind="ensemble_group",
            data=None,
            token_map=token_map,
            title="Pairwise chain ipTM",
            label="ipTM",
            members=members,
        )
    )

    np.testing.assert_allclose(
        matrix,
        np.array([[0.95, 0.6], [0.6, 0.7]], dtype=np.float32),
    )
    assert text[0, 0].startswith("0.950 +/-")
    assert text[1, 1].startswith("0.700 +/-")
