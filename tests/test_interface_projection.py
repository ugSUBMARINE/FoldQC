"""Residue projections preserve retained scores, directions and support."""

from dataclasses import replace

import numpy as np
import pytest
from FoldQC import metrics
from FoldQC.interface_analysis import (
    InterfaceAnalysisOptions,
    InterfaceProjectionSelection,
)
from FoldQC.interface_projection import project_interface_residues
from test_interface_analysis import compute, reference_index, score


def selection(model, key="ipsae", kind="ipsae_profile", side="both"):
    source = score(model, key)
    return InterfaceProjectionSelection(
        1, model.rank, source.metric_key, source.chain_a, source.partners, kind, side
    )


@pytest.mark.parametrize("cutoff", [10, 15])
@pytest.mark.parametrize(
    "side,indices", [("both", [0, 1, 2, 3]), ("a", [0, 1]), ("b", [2, 3])]
)
def test_profile_is_exact_retained_direction(cutoff, side, indices):
    model = compute(options=InterfaceAnalysisOptions(cutoff))
    request = selection(model, side=side)
    result = project_interface_residues(model, request, cutoff, lambda: None)
    expected = np.full(4, np.nan)
    retained = dict(score(model, "ipsae").residue_profile)
    for i in indices:
        expected[i] = retained[i]
    np.testing.assert_array_equal(result.values, expected)
    assert result.token_map is model.token_map
    assert result.values.flags.c_contiguous and not result.values.flags.writeable
    assert f"PAE <{cutoff} Å" in result.title


@pytest.mark.parametrize(
    "key", sorted(metrics.INTERFACE_PROJECTIONS["contributor_mask"].metric_keys)
)
@pytest.mark.parametrize("side", ["both", "a", "b"])
def test_masks_use_exact_support(key, side):
    model = compute()
    request = selection(model, key.removeprefix("interface_"), "contributor_mask", side)
    result = project_interface_residues(model, request, 10, lambda: None)
    source = score(model, key.removeprefix("interface_"))
    for i in range(4):
        in_scope = side == "both" or (side == "a" and i < 2) or (side == "b" and i >= 2)
        if in_scope:
            assert result.values[i] == int(i in source.support_a + source.support_b)
        else:
            assert np.isnan(result.values[i])
    if key == "interface_ilis":
        assert "contact-filtered support" in result.title
        assert "not every input" in result.description


def test_valid_empty_values_and_unavailable_are_distinct():
    model = compute(pae=np.full((4, 4), 30.0))
    for kind in ("ipsae_profile", "contributor_mask"):
        result = project_interface_residues(
            model, selection(model, kind=kind), 10, lambda: None
        )
        np.testing.assert_array_equal(result.values, 0)
    model = compute(pae=None)
    with pytest.raises(ValueError, match="PAE"):
        project_interface_residues(model, selection(model), 10, lambda: None)
    model = compute(reference_index(distant=True))
    result = project_interface_residues(
        model, selection(model, "pdockq", "contributor_mask"), 10, lambda: None
    )
    np.testing.assert_array_equal(result.values, 0)


def test_rejects_forged_profiles_support_and_unsupported_projection():
    model = compute()
    source = score(model, "ipsae")
    for profile in (
        source.residue_profile[:-1],
        source.residue_profile + source.residue_profile[:1],
        ((0, float("nan")),) + source.residue_profile[1:],
    ):
        broken = replace(model, scores=(replace(source, residue_profile=profile),))
        with pytest.raises(ValueError):
            project_interface_residues(broken, selection(model), 10, lambda: None)
    broken = replace(model, scores=(replace(source, support_a=(2,)),))
    with pytest.raises(ValueError, match="sides"):
        project_interface_residues(
            broken, selection(model, kind="contributor_mask"), 10, lambda: None
        )
    with pytest.raises(ValueError, match="no defined"):
        project_interface_residues(
            model, selection(model, "iptm", "contributor_mask"), 10, lambda: None
        )


def test_projection_cancellation():
    def cancelled():
        raise InterruptedError()

    model = compute()
    with pytest.raises(InterruptedError):
        project_interface_residues(model, selection(model), 10, cancelled)


def published_harness():
    from test_interface_analysis import harness

    parts = harness()
    *_, runner, operations, viewer, service, coordinator = parts
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions())
    runner.deliver(runner.task(lambda _: None))
    model = service.report.models[0]
    request = replace(selection(model), generation=service.report.generation)
    return parts, request


def test_captured_projection_routes_without_token_computation_or_acquisition():
    from FoldQC.analysis import ColorOptions

    (
        (state, files, presenter, runner, operations, viewer, service, coordinator),
        request,
    ) = published_harness()
    coordinator._metric_computation.reset_mock()
    coordinator.project_interface_residues(request, "interface_color")
    assert operations.is_busy
    assert coordinator._pending is None
    coordinator.invalidate_ui()  # unrelated appearance changes do not redirect the action
    runner.deliver(runner.task(lambda _: None))
    assert not operations.is_busy
    args = coordinator._coloring.execute_projection.call_args.args
    assert args[1].selection == request
    assert args[2] == ColorOptions("viridis", True, 0.0, 1.0)
    assert len(args[0].members) == 1
    coordinator._metric_computation.compute.assert_not_called()
    assert not presenter.notices


@pytest.mark.parametrize(
    "change", ["close", "cancel", "replace", "version", "index", "object", "target"]
)
def test_stale_or_cancelled_projection_cannot_paint(change):
    (
        (state, files, presenter, runner, operations, viewer, service, coordinator),
        request,
    ) = published_harness()
    coordinator.project_interface_residues(request, "interface_color")
    value = runner.task(lambda _: None)
    if change == "close":
        coordinator.interface_window_closed()
    elif change == "cancel":
        presenter.cancel_callbacks[-1]()
    elif change == "replace":
        state.pred_files = None
    elif change == "version":
        state.active_model_state._version += 1
    elif change == "index":
        state.active_model_state.structure_index = reference_index()
    elif change == "object":
        viewer.name_exists.return_value = False
    else:
        service.refresh_current("another_target")
    runner.deliver(value)
    assert not operations.is_busy
    coordinator._coloring.execute_projection.assert_not_called()
    assert not presenter.notices


def test_cancelled_projection_worker_and_late_error_are_silent():
    (*_, presenter, runner, operations, viewer, service, coordinator), request = (
        published_harness()
    )
    coordinator.project_interface_residues(request, "interface_color")
    task, request_id, on_error = runner.task, runner.request_id, runner.on_error
    coordinator.interface_window_closed()
    with pytest.raises(InterruptedError):
        task(lambda _: None)
    on_error(request_id, RuntimeError("late"))
    assert not presenter.notices


def test_plot_routes_through_guard_and_old_plot_cannot_select_or_clear():
    from FoldQC.plot_coordinator import PlotCoordinator
    from FoldQC.plot_preparation import PlotPreparationService
    from matplotlib import pyplot as plt

    (*_, presenter, runner, operations, viewer, service, coordinator), request = (
        published_harness()
    )
    coordinator._plots = PlotCoordinator(PlotPreparationService())
    prepared_plots = []
    presenter.show_plot = prepared_plots.append
    coordinator.project_interface_residues(request, "interface_profile")
    runner.deliver(runner.task(lambda _: None))
    prepared = prepared_plots[0]
    assert not operations.is_busy
    assert prepared.selection_port.select((0, 999))
    call = viewer.update_object_token_selection.call_args.args
    assert call[0] == "foldqc_interface" and call[1][0].token_indices == (0,)
    assert prepared.selection_port.clear()
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions(15))
    runner.deliver(runner.task(lambda _: None))
    viewer.reset_mock()
    assert not prepared.selection_port.select((0,))
    assert not prepared.selection_port.clear()
    viewer.update_object_token_selection.assert_not_called()
    viewer.delete_names.assert_not_called()
    plt.close(prepared.figure)


def test_failed_refresh_keeps_projection_generation_valid():
    (*_, presenter, runner, operations, viewer, service, coordinator), request = (
        published_harness()
    )
    before = service.report
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions(15))
    runner.on_error(runner.request_id, RuntimeError("failed refresh"))
    assert service.report is before
    assert service.projection_model(request) is before.models[0]


def test_projection_segments_respect_insertion_codes_and_gaps():
    from FoldQC.plot_data import interface_residue_segments
    from FoldQC.token_map import ResidueId, TokenInfo, TokenMap

    tm = TokenMap(
        tuple(
            TokenInfo(i, chain, ResidueId(num, code), "ALA", False, None)
            for i, (chain, num, code) in enumerate(
                [
                    ("A", 1, ""),
                    ("A", 1, "A"),
                    ("A", 1, "B"),
                    ("A", 2, ""),
                    ("A", 4, ""),
                    ("B", 5, ""),
                ]
            )
        )
    )
    assert interface_residue_segments(tm, np.zeros(6)) == (
        ("A", (0, 1, 2, 3)),
        ("A", (4,)),
        ("B", (5,)),
    )


def test_projection_paint_failure_rolls_back_and_does_not_publish_statistics():
    from unittest.mock import Mock

    from FoldQC.analysis import AnalysisRequest, AnalysisResolver, ColorOptions
    from FoldQC.gui_coloring import ColoringCoordinator
    from FoldQC.gui_services import ManagedColorbar, PaintBatchResult
    from FoldQC.viewer_transactions import ViewerTransactionError
    from test_viewer_transactions import FakeViewer

    (state, _, presenter, _, _, _, service, _), request = published_harness()
    projection = project_interface_residues(
        service.report.models[0], request, 10, lambda: None
    )
    viewer = FakeViewer()
    viewer.capture_paint_mappings = Mock(return_value="old mappings")
    viewer.restore_paint_mappings = Mock()
    viewer.managed_colorbar = ManagedColorbar("viridis", False, 0, 100, ("model_0",))
    previous_colorbar = viewer.managed_colorbar
    original = tuple(a.copy() for a in viewer.visuals["model_0"])

    def paint(*args, **kwargs):
        viewer.visuals["model_0"] = (np.array([1.0]), np.array([99]))
        return PaintBatchResult(0, 1, ())

    viewer.paint_continuous = paint
    viewer.fail_colorbar = True
    statistics = Mock()
    presenter.show_statistics = Mock()
    coloring = ColoringCoordinator(viewer, presenter, Mock(), Mock(), statistics)
    resolved = AnalysisResolver().resolve(
        AnalysisRequest("interface_color", "model_0", interface_projection=request),
        state,
    )
    with pytest.raises(ViewerTransactionError):
        coloring.execute_projection(
            resolved, projection, ColorOptions("viridis", True, 0, 1), lambda: True
        )
    for actual, expected in zip(viewer.visuals["model_0"], original):
        np.testing.assert_array_equal(actual, expected)
    assert viewer.managed_colorbar == previous_colorbar
    viewer.restore_paint_mappings.assert_called_once_with("old mappings")
    statistics.set_coloring_result.assert_not_called()
    presenter.show_statistics.assert_not_called()


def test_revalidate_after_mapping_confirmation_prevents_paint():
    from unittest.mock import Mock

    from FoldQC.analysis import AnalysisRequest, AnalysisResolver, ColorOptions
    from FoldQC.gui_coloring import ColoringCoordinator

    (state, _, presenter, _, _, viewer, service, _), request = published_harness()
    projection = project_interface_residues(
        service.report.models[0], request, 10, lambda: None
    )
    computation, statistics = Mock(), Mock()
    computation.confirm_token_overlap.side_effect = lambda *args: (
        service.window_closed() or True
    )
    coloring = ColoringCoordinator(viewer, presenter, Mock(), computation, statistics)
    resolved = AnalysisResolver().resolve(
        AnalysisRequest("interface_color", "model_0", interface_projection=request),
        state,
    )
    coloring.execute_projection(
        resolved,
        projection,
        ColorOptions("viridis", True, 0, 1),
        lambda: service.projection_model(request) is not None,
    )
    viewer.paint_continuous.assert_not_called()
    viewer.snapshot_atom_visuals.assert_not_called()
    viewer.restore_paint_mappings.assert_called_once()
    statistics.set_coloring_result.assert_not_called()


def test_reordered_tokens_extra_chain_and_ligand_keep_lossless_scope():
    from FoldQC.token_map import ResidueId, TokenInfo, TokenMap

    model = compute()
    old = score(model, "ipsae")
    order = {0: 3, 1: 0, 2: 4, 3: 2}
    tokens = [None] * 6
    for previous, current in order.items():
        tokens[current] = replace(model.token_map[previous], token_idx=current)
    tokens[1] = TokenInfo(1, "C", ResidueId(9, "B"), "ALA", False, None)
    tokens[5] = TokenInfo(5, "L", ResidueId(1), "HEM", True, "FE")
    remapped = replace(
        old,
        support_a=tuple(order[i] for i in old.support_a),
        support_b=tuple(order[i] for i in old.support_b),
        residue_profile=tuple((order[i], value) for i, value in old.residue_profile),
    )
    model = replace(
        model,
        token_map=TokenMap(tuple(tokens)),
        scores=(remapped,),
        protein_indices=(0, 1, 2, 3, 4),
    )
    result = project_interface_residues(model, selection(model), 10, lambda: None)
    for previous, current in order.items():
        assert result.values[current] == dict(old.residue_profile)[previous]
    assert np.isnan(result.values[[1, 5]]).all()
    assert model.token_map[0].resi == "2A"
    rest = replace(
        remapped,
        metric_key="interface_pdockq2_zhu",
        partners=("B", "C"),
        support_a=(3,),
        support_b=(1, 4),
    )
    model = replace(model, scores=(rest,))
    request = selection(model, "pdockq2_zhu", "contributor_mask", "b")
    result = project_interface_residues(model, request, 10, lambda: None)
    np.testing.assert_array_equal(result.values, [np.nan, 1, 0, np.nan, 1, np.nan])


def test_plot_port_does_not_keep_source_model_or_report_alive():
    import gc
    import weakref

    (
        (state, files, presenter, runner, operations, viewer, service, coordinator),
        request,
    ) = published_harness()
    port = service.projection_selection_port(request, (0, 1))
    model_ref = weakref.ref(state.active_model_state)
    report_ref = weakref.ref(service.report)
    coordinator.interface_window_closed()
    presenter.reports.clear()
    state.model_states.clear()
    state.pred_files = None
    runner.task = None
    gc.collect()
    assert model_ref() is None and report_ref() is None
    assert not port.select((0,)) and not port.clear()


def test_projection_plot_values_selection_metadata_and_mask_legend():
    from FoldQC.plot_coordinator import PlotCoordinator
    from FoldQC.plot_preparation import PlotPreparationService
    from matplotlib import pyplot as plt

    coordinator = PlotCoordinator(PlotPreparationService())
    model = compute()
    request = selection(model, kind="contributor_mask")
    projection = project_interface_residues(model, request, 10, lambda: None)
    prepared = coordinator.prepare_projection(projection)
    metadata = prepared.figure._foldqc_viewer_selection
    assert metadata["token_map"] is model.token_map
    assert metadata["token_indices"] == [0, 1, 2, 3]
    assert metadata["token_values"] == tuple(projection.values)
    ax = prepared.figure.axes[0]
    assert ax.get_ylim() == (0, 1)
    assert all(line.get_drawstyle() == "steps-mid" for line in ax.lines)
    assert {t.get_text() for t in ax.get_legend().get_texts()} >= {
        "Contributor",
        "Non-contributor",
    }
    number = prepared.figure.number
    prepared.close()
    assert not plt.fignum_exists(number)
