from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from FoldQC import compute, plots
from FoldQC.analysis import AnalysisRequest, AnalysisResolver, ComputedMetric
from FoldQC.confidence import PredictionConfidence
from FoldQC.context_service import ContextService
from FoldQC.ensemble import EnsembleMember, EnsembleState
from FoldQC.gui_services import ContextSelection
from FoldQC.gui_state import MetricContext, PluginState
from FoldQC.loader_models import (
    ModelFiles,
    PredictionData,
    PredictionFiles,
    ProviderInfo,
)
from FoldQC.model_state import ModelState
from FoldQC.plot_coordinator import PlotCoordinator
from FoldQC.plot_preparation import PlotPreparationService
from FoldQC.structure_index import StructureIndex
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap


@pytest.fixture
def plot_state(tmp_path):
    provider = ProviderInfo("test", "Test")
    token_map = TokenMap(
        tuple(
            TokenInfo(i, chain, ResidueId(i + 1), "ALA", False, None)
            for i, chain in enumerate(("A", "A", "B", "B"))
        )
    )
    files = PredictionFiles(
        "prediction",
        tmp_path,
        provider,
        models=[
            ModelFiles(
                rank,
                tmp_path / f"model_{rank}.cif",
                f"model {rank}",
                f"model_{rank}",
                capabilities=frozenset({"plddt", "pae", "pde", "contact_probs"}),
            )
            for rank in (0, 1)
        ],
    )
    states = {}
    for model in files.models:
        rank = model.rank
        plddt = np.array([0.3, 0.7, 0.6, 0.9], dtype=np.float32)
        plddt.setflags(write=False)
        matrix = (
            np.array(
                [[8, 6, 2, 3], [6, 8, 1, 2], [2, 1, 3, 4], [3, 2, 4, 3]],
                dtype=np.float32,
            )
            + rank
        )
        data = PredictionData(
            "prediction",
            rank,
            model.structure_path,
            provider,
            token_plddt=plddt,
            token_plddt_source="provider_token",
            pae=matrix,
            pde=matrix,
            contact_probs=matrix / 20,
            confidence=PredictionConfidence(
                pair_chain_iptm=np.array([[0.7, 0.3], [0.4, 0.8]]) + rank * 0.02,
            ),
        )
        index = StructureIndex(
            model.structure_path, "cif", token_map, 4, (0, 1, 2, 3), plddt
        )
        states[rank] = ModelState(rank, data, index)
    ensemble = EnsembleState(
        "prediction_ensemble",
        tuple(EnsembleMember(rank, f"model_{rank}") for rank in states),
        False,
        np.zeros(4),
        plddt,
        np.zeros(4),
    )
    return PluginState(files, states, 0, ensemble)


def resolve_plot(state, action, metric, ensemble, reference=()):
    target = "prediction_ensemble" if ensemble else "model_0"
    request = AnalysisRequest(
        action, target, metric, reference_selection="chain A" if reference else ""
    )
    resolved = AnalysisResolver().resolve(request, state)
    return resolved.with_member_contexts(
        tuple(
            MetricContext(reference_indices=reference) for _member in resolved.members
        )
    )


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize("reference", [(), (0, 1)])
@pytest.mark.parametrize(
    ("metric", "cmap"),
    [
        ("pae_row_mean", "viridis"),
        ("pde_mean", "viridis"),
        ("contact_prob_mean", "viridis"),
        ("chain_iptm", "viridis_r"),
        ("chain_ptm", "viridis_r"),
    ],
)
def test_matrix_appearance_defaults(plot_state, ensemble, reference, metric, cmap):
    resolved = resolve_plot(plot_state, "matrix", metric, ensemble, reference)
    prepared = PlotCoordinator(PlotPreparationService()).prepare(resolved, ())
    figure = prepared.figure
    try:
        ax = figure.axes[0]
        image = ax.images[0]
        assert image.cmap.name == cmap
        assert image.get_clim() == pytest.approx((0, np.max(image.get_array())))
        if metric in {"chain_iptm", "chain_ptm"}:
            assert image.get_array().shape == (2, 2)
            assert [tick.get_text() for tick in ax.get_xticklabels()] == ["A", "B"]
            assert len(ax.texts) == 4
            assert ("+/-" in ax.texts[0].get_text()) == ensemble
            expected = np.array([[0.7, 0.3], [0.4, 0.8]]) + (0.01 if ensemble else 0)
            np.testing.assert_allclose(image.get_array(), expected)
        else:
            indices = list(reference) if reference else list(range(4))
            metadata = figure._foldqc_viewer_selection
            assert metadata["token_map"] is resolved.target.token_map
            if metric == "pae_row_mean":
                assert metadata["row_indices"] == indices
                assert metadata["col_indices"] == list(range(4))
                assert image.get_array().shape == (len(indices), 4)
            else:
                assert metadata["row_indices"] == list(range(4))
                assert metadata["col_indices"] == indices
                assert image.get_array().shape == (4, len(indices))
            if ensemble:
                assert metadata["token_map_obj_names"] == ["model_0", "model_1"]
    finally:
        plots.plt.close(figure)


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize("action", ["line", "pae_summary", "pde_summary"])
def test_line_and_summary_plots_autoscale(plot_state, ensemble, action):
    resolved = resolve_plot(
        plot_state, action, "plddt" if action == "line" else None, ensemble
    )
    computed = tuple(
        ComputedMetric(
            member.rank,
            member.label,
            member.obj_name,
            member.model_state,
            member.metric_context,
            member.model_state.data.token_plddt,
        )
        for member in resolved.members
    )
    figure = (
        PlotCoordinator(PlotPreparationService()).prepare(resolved, computed).figure
    )
    try:
        ax = figure.axes[0]
        ymin, ymax = ax.get_ylim()
        values = np.concatenate(
            [
                line.get_ydata()
                for line in ax.lines
                if line.get_transform() == ax.transData
            ]
        )
        assert ymin < np.min(values)
        assert ymax > np.max(values)
        if action != "line":
            assert np.min(values) < 0
            assert ymin < 0
        metadata = figure._foldqc_viewer_selection
        assert metadata["token_indices"] == list(range(4))
        assert metadata["obj_name"] == resolved.target.obj_name
    finally:
        plots.plt.close(figure)


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize("reference", [(), (2, 3)])
def test_chain_ptm_line_values_and_reference_scope(plot_state, ensemble, reference):
    resolved = resolve_plot(plot_state, "line", "chain_ptm", ensemble, reference)
    computed = tuple(
        ComputedMetric(
            member.rank,
            member.label,
            member.obj_name,
            member.model_state,
            member.metric_context,
            compute.compute_metric(
                "chain_ptm", member.model_state.data, member.model_state.token_map
            ),
        )
        for member in resolved.members
    )
    preparation = PlotPreparationService()
    indices, series, ylabel = preparation.line_data(resolved, computed)
    expected = np.array([0.7, 0.7, 0.8, 0.8]) + (0.01 if ensemble else 0)
    selected = list(reference) if reference else list(range(4))
    np.testing.assert_array_equal(indices, selected)
    np.testing.assert_allclose(series[0][1], expected[selected])
    assert ylabel == "pTM"
    if ensemble:
        np.testing.assert_allclose(
            series[0][2], np.full(len(selected), 0.01), atol=1e-7
        )
    else:
        assert series[0][2] is None
    figure = PlotCoordinator(preparation).prepare(resolved, computed).figure
    try:
        assert figure.axes[0].get_ylabel() == "pTM"
        assert figure._foldqc_viewer_selection["token_indices"] == selected
    finally:
        plots.plt.close(figure)


def test_chain_ptm_vector_only_matrix_reports_missing_data(plot_state):
    plot_state.model_states[0].data.confidence = PredictionConfidence(
        chain_ptm=np.array([0.9, 0.8])
    )
    resolved = resolve_plot(plot_state, "matrix", "chain_ptm", False)
    with pytest.raises(ValueError, match="pairwise chain ipTM"):
        PlotCoordinator(PlotPreparationService()).prepare(resolved, ())


@pytest.mark.parametrize("ensemble", [False, True])
def test_chain_ptm_context_requires_scores_for_every_target_member(
    plot_state, ensemble
):
    context = ContextService(
        plot_state,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        {"chain_ptm": 0, "chain_iptm": 1},
    )
    context.set_selection(
        ContextSelection(
            target_name="prediction_ensemble" if ensemble else "model_0",
            metric_key="chain_ptm",
        )
    )
    assert dict(context.metric_availability())[0]
    plot_state.model_states[1].data.confidence = PredictionConfidence(
        chain_iptm=np.array([0.8, 0.7]),
        pair_chain_iptm=np.array([[np.nan, 0.8], [0.7, np.nan]]),
    )
    availability = dict(context.metric_availability())
    assert availability[0] is (not ensemble)
    assert availability[1]
    plot_state.model_states[0].data.confidence = PredictionConfidence(
        chain_ptm=np.array([0.9, np.nan])
    )
    assert dict(context.metric_availability())[0] is (not ensemble)
    plot_state.model_states[0].data.confidence = PredictionConfidence(ptm=0.9)
    assert not dict(context.metric_availability())[0]
