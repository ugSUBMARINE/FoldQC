"""Offline published-score, canonical geometry and report regression coverage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from FoldQC import metrics, properties
from FoldQC.analysis import (
    AnalysisPreflightError,
    AnalysisRequest,
    AnalysisResolver,
    build_data_load_plan,
)
from FoldQC.analysis_coordinator import AnalysisCoordinator
from FoldQC.confidence import PredictionConfidence
from FoldQC.data_acquisition import DataAcquisitionService
from FoldQC.export import interface_report_rows, write_interface_report
from FoldQC.gui_operations import GuiOperationCoordinator
from FoldQC.gui_services import ContextSelection
from FoldQC.interface_analysis import (
    InterfaceAnalysisOptions,
    InterfaceReport,
    InterfaceSelectionRequest,
)
from FoldQC.interface_compute import (
    aggregate_interface_models,
    compute_interface_model,
    validate_interface_dispatch,
)
from FoldQC.interface_report_data import heatmap_value, report_rows, score_details
from FoldQC.interface_service import INTERFACE_SELECTION, InterfaceReportService
from FoldQC.loader import load_prediction_data, scan_prediction_path
from FoldQC.loader_models import PredictionData
from FoldQC.model_state import ModelState
from FoldQC.structure_index import StructureGeometry, StructureIndex
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap
from test_gui_model_switching import (
    DeferredRunner,
    FakeDependencies,
    FakePresenter,
    FakeView,
    ImmediateScheduler,
    _state,
)

REFERENCE = json.loads(
    Path(__file__)
    .with_name("fixtures")
    .joinpath("interface_reference.json")
    .read_text()
)


def reference_index(*, distant=False, missing=False):
    tm = TokenMap(
        tuple(
            TokenInfo(
                i, chain, ResidueId(i + 1, "A" if i == 1 else ""), res, False, None
            )
            for i, (chain, res) in enumerate(
                zip(REFERENCE["chains"], REFERENCE["residues"], strict=True)
            )
        )
    )
    ca, cb = (
        np.array(REFERENCE["ca"], dtype=float),
        np.array(REFERENCE["cb"], dtype=float),
    )
    if distant:
        ca[2:] += 100
        cb[2:] += 100
    if missing:
        cb[0] = np.nan
    geometry = StructureGeometry(
        ca,
        cb,
        np.array(REFERENCE["ca_plddt"]),
        np.array(REFERENCE["cb_plddt"]),
        np.ones(4, dtype=bool),
    )
    return StructureIndex(
        Path("/tmp/model_0.cif"), "cif", tm, 4, (0, 1, 2, 3), np.ones(4), geometry
    )


def compute(index=None, pae="reference", **kwargs):
    if isinstance(pae, str):
        pae = np.array(REFERENCE["pae"], dtype=np.float64)
    return compute_interface_model(
        index or reference_index(),
        pae,
        kwargs.pop("confidence", None),
        kwargs.pop("options", InterfaceAnalysisOptions()),
        rank=0,
        label="Model 0",
        object_name="model_0",
        **kwargs,
    )


def score(model, key, chain="A"):
    return next(
        s
        for s in model.scores
        if s.metric_key == "interface_" + key and s.chain_a == chain
    )


def report(model=None):
    return InterfaceReport(
        "prediction",
        "test",
        "Test",
        "/tmp/input.zip",
        "model_0",
        InterfaceAnalysisOptions(),
        (model or compute(),),
    )


def test_published_golden_values_and_variants():
    model, expected = compute(), REFERENCE["expected"]
    assert score(model, "ipsae").forward == pytest.approx(expected["ipsae_10_ab"])
    assert score(model, "ipsae").reverse == pytest.approx(expected["ipsae_10_ba"])
    assert score(model, "lis_kim").value == pytest.approx(expected["lis_kim_mean"])
    assert score(model, "lis_kim").forward == pytest.approx(expected["lis_kim_ab"])
    assert score(model, "lis_kim").reverse == pytest.approx(expected["lis_kim_ba"])
    assert score(model, "lis_dunbrack").value == pytest.approx(
        expected["lis_dunbrack_max"]
    )
    assert score(model, "clis").value == pytest.approx(expected["clis_mean"])
    assert score(model, "ilis").value == pytest.approx(expected["ilis"])
    assert score(model, "pdockq").value == pytest.approx(expected["pdockq"])
    assert score(model, "pdockq2_dunbrack").forward == pytest.approx(
        expected["pdockq2_ab"]
    )
    assert score(model, "pdockq2_dunbrack").reverse == pytest.approx(
        expected["pdockq2_ba"]
    )
    assert score(model, "pdockq2_zhu").value == pytest.approx(expected["zhu_a"])
    assert score(model, "pdockq2_zhu", "B").value == pytest.approx(expected["zhu_b"])
    assert dict(score(model, "lis_kim").counts)["LIA_combined"] == 6
    assert dict(score(model, "clis").counts)["cLIA_combined"] == 4
    assert score(model, "pdockq2_dunbrack").interface_plddt == pytest.approx(190 / 3)
    assert score(model, "pdockq2_zhu").interface_plddt == pytest.approx(230 / 3)
    assert len(score(model, "ipsae").residue_profile) == 4
    assert score(model, "clis").support_a == (0, 1)
    assert score(model, "clis").support_b == (2,)


def test_empty_interfaces_preserve_each_reference_convention():
    model = compute(reference_index(distant=True), pae=np.full((4, 4), 30.0))
    for key in (
        "ipsae",
        "lis_kim",
        "lis_dunbrack",
        "clis",
        "ilis",
        "pdockq",
        "pdockq2_dunbrack",
    ):
        assert score(model, key).value == 0.0
    assert score(model, "pdockq2_zhu").value == pytest.approx(
        REFERENCE["expected"]["zhu_empty"]
    )
    assert "baseline" in score(model, "pdockq2_zhu").reason
    assert score(model, "pdockq2_zhu").status == "ok"


def test_missing_geometry_or_pae_is_not_zero_or_a_global_confidence_fallback():
    model = compute(reference_index(missing=True))
    assert score(model, "clis").value is None
    assert score(model, "ipsae").value is not None
    assert score(model, "pdockq2_zhu").value is not None
    model = compute(pae=None, confidence=PredictionConfidence(iptm=0.9, ptm=0.8))
    assert score(model, "pdockq").value is not None
    assert score(model, "ipsae").value is None
    assert score(model, "iptm").value is None
    missing = replace(reference_index(), geometry=None)
    assert score(compute(missing), "ipsae").value is not None
    assert score(compute(missing), "pdockq").value is None
    assert score(compute(pae=np.full((4, 4), -1.0)), "lis_kim").value is None
    with pytest.raises(ValueError, match="canonical token map"):
        compute(pae=np.ones((3, 3)))


def test_provider_directional_axes_and_preset_boundaries():
    model = compute(
        confidence=PredictionConfidence(
            pair_chain_iptm=np.array([[0.8, 0.2], [0.6, 0.9]])
        )
    )
    assert score(model, "iptm").forward == pytest.approx(0.2)
    assert score(model, "iptm").value == pytest.approx(0.6)
    partial = compute(
        confidence=PredictionConfidence(
            pair_chain_iptm=np.array([[0.8, 0.2], [np.nan, 0.9]])
        )
    )
    assert score(partial, "iptm").forward == pytest.approx(0.2)
    assert score(partial, "iptm").reverse is None
    assert score(partial, "iptm").value is None
    rows = [
        r
        for r in interface_report_rows(report(partial))
        if r["metric_key"] == "interface_iptm"
    ]
    assert rows[0]["status"] == "unavailable"
    assert rows[1]["direction"] == "a_to_b" and rows[1]["status"] == "ok"
    low = compute(pae=np.full((4, 4), 10.0))
    high = compute(pae=np.full((4, 4), 10.0), options=InterfaceAnalysisOptions(15))
    assert score(low, "ipsae").value == 0
    assert score(high, "ipsae").value > 0
    assert dict(score(high, "ipsae").parameters)["pae_cutoff_angstrom"] == "15"
    with pytest.raises(ValueError):
        InterfaceAnalysisOptions(12)
    assert properties.interface_contact_mask(
        np.array([[0.0, 0, 0]]), np.array([[8.0, 0, 0]]), lambda: None
    )[0, 0]


def test_tiling_and_cancellation():
    calls = []
    properties.interface_contact_mask(
        np.zeros((600, 3)), np.ones((3, 3)), lambda: calls.append(1)
    )
    assert len(calls) == 3

    def cancel():
        raise InterruptedError()

    with pytest.raises(InterruptedError):
        compute(check_cancel=cancel)


def test_disordered_tail_does_not_change_ipsae():
    block = np.array([[1.0, 2.0], [3.0, 4.0]])
    original, _ = properties.interface_ipsae(block, 10)
    padded = np.full((100, 200), 30.0)
    padded[:2, :2] = block
    assert properties.interface_ipsae(padded, 10)[0] == original


def test_chain_vs_rest_is_not_a_pairwise_mean():
    original = reference_index()
    tm = TokenMap(
        (
            *original.token_map.tokens,
            TokenInfo(4, "C", ResidueId(1), "ALA", False, None),
        )
    )
    geom = original.geometry
    index = StructureIndex(
        original.path,
        "cif",
        tm,
        5,
        tuple(range(5)),
        np.ones(5),
        StructureGeometry(
            np.vstack((geom.ca, [[0, 0, 3]])),
            np.vstack((geom.cb, [[0, 0, 4]])),
            np.append(geom.ca_plddt, 50),
            np.append(geom.cb_plddt, 40),
            np.ones(5, bool),
        ),
    )
    pae = np.full((5, 5), 1.0)
    pae[:4, :4] = REFERENCE["pae"]
    model = compute(index, pae=pae)
    assert score(model, "pdockq2_zhu").partners == ("B", "C")
    assert score(model, "pdockq2_zhu").value != pytest.approx(
        score(compute(), "pdockq2_zhu").value
    )


def test_ensemble_identity_groups_and_heatmap_direction():
    first = compute()
    second = replace(first, rank=1, label="Model 1")
    aggregates = aggregate_interface_models((first, second))
    assert all(a.n_models == 2 and a.std == 0 for a in aggregates)
    changed = replace(first.token_map[0], residue_id=ResidueId(99))
    third = replace(second, token_map=TokenMap((changed, *first.token_map.tokens[1:])))
    aggregates = aggregate_interface_models((first, third))
    assert all(a.n_models == 1 for a in aggregates)
    snapshot = replace(report(), models=(first, third), aggregates=aggregates)
    rows = report_rows(snapshot, "chain_pair")
    assert heatmap_value(rows, "A", "A", "interface_ipsae", 0, False) is None
    assert (
        heatmap_value(rows, "A", "B", "interface_ipsae", 0, True)
        == score(first, "ipsae").forward
    )
    assert (
        heatmap_value(rows, "B", "A", "interface_ipsae", 0, True)
        == score(first, "ipsae").reverse
    )
    assert heatmap_value(rows, "A", "B", "interface_ipsae", None, False) is None


def test_ensemble_matches_residue_sets_independently_of_token_and_chain_order():
    original = reference_index()
    permutation = [3, 2, 1, 0]
    geometry = original.geometry
    assert geometry is not None
    reordered = replace(
        original,
        token_map=TokenMap(
            tuple(
                replace(original.token_map[i], token_idx=j)
                for j, i in enumerate(permutation)
            )
        ),
        geometry=StructureGeometry(
            geometry.ca[permutation],
            geometry.cb[permutation],
            geometry.ca_plddt[permutation],
            geometry.cb_plddt[permutation],
            geometry.protein_mask[permutation],
        ),
    )
    pae = np.array(REFERENCE["pae"], dtype=float)
    first = compute(original)
    second = replace(
        compute(reordered, pae=pae[np.ix_(permutation, permutation)]), rank=1
    )
    aggregates = aggregate_interface_models((first, second))
    assert len(aggregates) == len([s for s in first.scores if s.value is not None])
    assert all(a.n_models == 2 and a.std == 0 for a in aggregates)
    ipsae = next(a for a in aggregates if a.metric_key == "interface_ipsae")
    assert ipsae.forward == score(first, "ipsae").forward
    assert ipsae.reverse == score(first, "ipsae").reverse
    rows = interface_report_rows(
        replace(report(), models=(first, second), aggregates=aggregates)
    )
    assert any(r.get("n_models") == 2 for r in rows)


def test_csv_complete_provenance_and_lossless_contributors(tmp_path):
    snapshot = report()
    rows = interface_report_rows(snapshot)
    assert {row["scope"] for row in rows} == {"chain_pair", "chain_rest"}
    directional = [r for r in rows if r["metric_key"] == "interface_ipsae"]
    assert {r["direction"] for r in directional} == {"summary", "a_to_b", "b_to_a"}
    assert json.loads(directional[0]["contributors_a"])[1]["residue_id"] == "2A"
    assert (
        directional[0]["implementation_revision"] == REFERENCE["references"]["Dunbrack"]
    )
    assert not any("palette" in key or "selection" in key for key in rows[0])
    destination = tmp_path / "interface.csv"
    write_interface_report(destination, snapshot)
    assert "interface_schema_version" in destination.read_text()
    row = report_rows(snapshot, "chain_pair")[0]
    assert "Original Zhu" in score_details(snapshot, row, "interface_ipsae")


def test_registry_has_separate_scopes_and_dispatch():
    validate_interface_dispatch()
    assert metrics.METRICS.require("interface_pdockq2_zhu").result_scope == "chain_rest"
    assert all(not s.plot_modes for s in metrics.METRICS if s.interface is not None)


def test_geometry_is_read_once_and_preserves_legacy_token_mapping(tmp_path):
    pdb = tmp_path / "model.pdb"

    def atom(
        serial, name, res, chain, number, x, y, z, b, record="ATOM", insertion=" "
    ):
        return f"{record:<6}{serial:5} {name:^4} {res:3} {chain}{number:4}{insertion}   {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b:6.2f}          {name[0]:>2}\n"

    pdb.write_text(
        atom(1, "CA", "GLY", "A", 1, 0, 0, 0, 80, insertion="A")
        + atom(2, "CA", "ALA", "B", 1, 0, 4, 0, 90)
        + atom(3, "CB", "ALA", "B", 1, 0, 5, 0, 70)
        + atom(4, "C1", "LIG", "L", 1, 0, 0, 0, 90, "HETATM")
    )
    with patch.object(Path, "open", wraps=pdb.open) as opened:
        index = StructureIndex.from_path(pdb)
    assert opened.call_count == 1
    assert index.atom_count == 4 and index.atom_to_token == (0, 1, 1, 2)
    assert index.token_map[0].resi == "1A"
    assert np.array_equal(index.geometry.cb[0], index.geometry.ca[0])
    assert index.geometry.cb_plddt[0] == 80
    assert not index.geometry.ca.flags.writeable
    assert not index.geometry.protein_mask[2]
    before = compute_interface_model(
        index,
        np.ones((3, 3)),
        None,
        InterfaceAnalysisOptions(),
        rank=0,
        label="x",
        object_name="x",
    )
    pdb.write_text("replaced file")
    after = compute_interface_model(
        index,
        np.ones((3, 3)),
        None,
        InterfaceAnalysisOptions(),
        rank=0,
        label="x",
        object_name="x",
    )
    assert before.scores == after.scores


class ReportPresenter(FakePresenter):
    def __init__(self):
        super().__init__()
        self.report_view = Mock()
        self.reports = []

    def show_interface_report(self, snapshot, observer):
        self.reports.append(snapshot)
        return self.report_view

    def raise_interface_report(self):
        return bool(self.reports)


def harness(with_pae=True):
    state, files = _state(with_pae=with_pae)
    index = reference_index()
    state.model_states[0] = ModelState(
        0,
        PredictionData(
            "prediction",
            0,
            index.path,
            files.provider,
            token_plddt=np.ones(4),
            token_plddt_source="provider_token",
            pae=np.asarray(REFERENCE["pae"], dtype=float) if with_pae else None,
        ),
        index,
    )
    presenter, runner = ReportPresenter(), DeferredRunner()
    operations = GuiOperationCoordinator(presenter, FakeView())
    viewer = Mock()
    service = InterfaceReportService(state, viewer, presenter, operations)
    operations.observe(service)
    context = Mock()
    context.selection = ContextSelection(target_name="model_0")
    data = DataAcquisitionService(
        state, presenter, ImmediateScheduler(), runner, operations
    )
    coordinator = AnalysisCoordinator(
        state,
        presenter,
        FakeDependencies(),
        data,
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        context,
        set(),
        runner,
        operations,
        service,
    )
    return state, files, presenter, runner, operations, viewer, service, coordinator


def test_background_report_does_not_call_token_computation():
    state, _, presenter, runner, operations, viewer, service, coordinator = harness()
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions(15))
    assert operations.is_busy
    runner.deliver(runner.task(lambda _: None))
    assert not operations.is_busy
    assert presenter.reports[0].options.ipsae_cutoff == 15
    coordinator._metric_computation.compute.assert_not_called()
    viewer.inspect_tokens.assert_not_called()
    coordinator.invalidate_ui()  # palette changes do not invalidate a completed report
    assert service.is_current()
    state.pred_files = None
    service.refresh_current()
    assert not service.is_current()
    presenter.report_view.set_current.assert_called_with(False)


def test_cancel_close_and_late_worker_results_preserve_report():
    *_, presenter, runner, operations, viewer, service, coordinator = harness()
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions())
    result = runner.task(lambda _: None)
    presenter.cancel_callbacks[-1]()
    runner.deliver(result)
    assert not presenter.reports and not operations.is_busy and not presenter.notices
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions())
    task = runner.task
    coordinator.interface_window_closed()
    with pytest.raises(InterruptedError):
        task(lambda _: None)


def test_lazy_request_captures_exact_preset_and_reuses_canonical_index():
    from FoldQC.lifecycle_support import DataLoadBatchResult

    state, files, presenter, runner, operations, viewer, service, coordinator = harness(
        False
    )
    canonical = state.active_model_state.structure_index
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions(15))
    requirement = coordinator._data._pending[2].requirements[0]
    incoming = PredictionData(
        "prediction", 0, canonical.path, files.provider, pae=np.ones((4, 4))
    )
    runner.deliver(DataLoadBatchResult(files, ((requirement, incoming),)))
    runner.deliver(runner.task(lambda _: None))
    assert presenter.reports[0].options.ipsae_cutoff == 15
    assert state.active_model_state.structure_index is canonical


def test_selection_exact_support_and_stale_rejection():
    state, _, presenter, _, operations, viewer, service, coordinator = harness()
    snapshot = report(compute(state.active_model_state.structure_index))
    resolved = AnalysisResolver().resolve(
        AnalysisRequest(
            "interfaces", "model_0", interface_options=InterfaceAnalysisOptions()
        ),
        state,
    )
    service.show(resolved, snapshot, coordinator)
    request = InterfaceSelectionRequest(0, score(snapshot.models[0], "clis"), "b")
    service.select(request)
    call = viewer.update_object_token_selection.call_args
    assert call.args[0] == INTERFACE_SELECTION
    assert call.args[1][0].token_indices == (2,)
    state.pred_files = None
    service.select(request)
    assert viewer.update_object_token_selection.call_count == 1
    service.clear_selection()
    viewer.delete_names.assert_called_once_with((INTERFACE_SELECTION,))


def test_partial_ensemble_plan_only_loads_advertised_pae(tmp_path):
    from test_analysis import _plugin_state

    state = _plugin_state(tmp_path)
    resolved = AnalysisResolver().resolve(
        AnalysisRequest(
            "interfaces",
            "prediction_ensemble",
            interface_options=InterfaceAnalysisOptions(),
        ),
        state,
    )
    assert resolved.coverage_policy == "available"
    plan = build_data_load_plan(resolved)
    assert [r.rank for r in plan.requirements] == [0]
    assert plan.requirements[0].capabilities == {"pae"}


def test_cif_original_geometry_fractional_confidence_and_exclusions(tmp_path):
    path = tmp_path / "prediction.cif"
    fields = "group_PDB label_atom_id auth_comp_id auth_seq_id pdbx_PDB_ins_code auth_asym_id Cartn_x Cartn_y Cartn_z B_iso_or_equiv type_symbol".split()
    path.write_text(
        "data_prediction\nloop_\n"
        + "\n".join("_atom_site." + f for f in fields)
        + "\n"
        + "ATOM CA ALA 1 . A 0 0 0 0.8 C\n"
        "ATOM CB ALA 1 . A 0 0 1 0.7 C\n"
        "ATOM CA GLY 2 A A 5 0 0 0.9 C\n"
        "ATOM CA ALA 1 . B 0 4 0 0.95 C\n"
        "ATOM CB ALA 1 . B 0 5 1 0.85 C\n"
        "ATOM P DA 1 . N 5 5 5 0.8 P\n"
        "HETATM CA MSE 1 . M 0 0 0 0.8 C\n"
        "HETATM C1 LIG 1 . L 0 0 0 0.8 C\n#\n"
    )
    index = StructureIndex.from_path(path)
    assert index.token_map[1].resi == "2A"
    assert index.atom_to_token == (0, 0, 1, 2, 2, 3, 4, 5)
    assert np.array_equal(
        index.geometry.protein_mask, [True, True, True, False, False, False]
    )
    assert index.geometry.ca_plddt[0] == 80 and index.geometry.cb_plddt[0] == 70
    result = compute_interface_model(
        index,
        np.ones((6, 6)),
        None,
        InterfaceAnalysisOptions(),
        rank=0,
        label="x",
        object_name="x",
    )
    assert result.chain_order == ("A", "B")
    assert "modified-residue" in result.diagnostics[0]
    assert all(s.chain_a not in {"N", "M", "L"} for s in result.scores)


def test_ambiguous_representative_disables_only_affected_metric(tmp_path):
    path = tmp_path / "altloc.cif"
    path.write_text(
        "data_x\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.auth_comp_id\n_atom_site.auth_seq_id\n_atom_site.auth_asym_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.B_iso_or_equiv\n"
        "ATOM CA ALA 1 A 0 0 0 80\nATOM CA ALA 1 A 0 1 0 80\nATOM CB ALA 1 A 0 0 1 80\n"
        "ATOM CA ALA 1 B 0 5 0 90\nATOM CB ALA 1 B 0 5 1 90\n#\n"
    )
    index = StructureIndex.from_path(path)
    result = compute_interface_model(
        index,
        np.ones((2, 2)),
        None,
        InterfaceAnalysisOptions(),
        rank=0,
        label="x",
        object_name="x",
    )
    assert score(result, "pdockq").value is not None
    assert score(result, "pdockq2_zhu").value is None
    assert score(result, "ipsae").value is not None
    path.write_text(path.read_text().replace("0 0 1 80", "0 0 1 ?"))
    unknown_confidence = compute(StructureIndex.from_path(path), pae=np.ones((2, 2)))
    assert score(unknown_confidence, "pdockq").value is None
    assert "pLDDT" in score(unknown_confidence, "pdockq").reason
    assert score(unknown_confidence, "ipsae").value is not None


@pytest.mark.parametrize("trailing_field", ["pdbx_PDB_model_num", "type_symbol"])
def test_cif_omitted_trailing_fields_preserve_atoms_and_interface_scores(
    tmp_path, trailing_field
):
    header = (
        "data_x\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n"
        "_atom_site.auth_comp_id\n_atom_site.auth_seq_id\n_atom_site.auth_asym_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        f"_atom_site.B_iso_or_equiv\n_atom_site.{trailing_field}\n"
    )
    rows = (
        "ATOM CA GLY 1 A 0 0 0 80",
        "ATOM CA ALA 1 B 0 4 0 90",
        "ATOM CB ALA 1 B 0 5 0 70",
    )
    trailing_value = "C" if trailing_field == "type_symbol" else "1"
    complete = tmp_path / "complete.cif"
    complete.write_text(header + "\n".join(f"{r} {trailing_value}" for r in rows))
    shortened = tmp_path / "shortened.cif"
    shortened.write_text(header + "\n".join(rows) + "\nATOM CA GLY\n")
    expected = StructureIndex.from_path(complete)
    with patch.object(Path, "open", wraps=shortened.open) as opened:
        actual = StructureIndex.from_path(shortened)
    assert opened.call_count == 1
    assert actual.atom_count == expected.atom_count == 3
    assert actual.token_map == expected.token_map
    assert actual.atom_to_token == expected.atom_to_token == (0, 1, 1)
    np.testing.assert_array_equal(actual.structure_plddt, expected.structure_plddt)
    for name in ("ca", "cb", "ca_plddt", "cb_plddt"):
        np.testing.assert_array_equal(
            getattr(actual.geometry, name), getattr(expected.geometry, name)
        )
    before = compute(expected, pae=np.ones((2, 2)))
    after = compute(actual, pae=np.ones((2, 2)))
    assert before.scores == after.scores
    assert dict(score(after, "pdockq").counts)["cb_contact_pairs"] == 1
    for key in ("clis", "ilis", "pdockq", "pdockq2_dunbrack", "pdockq2_zhu"):
        assert score(after, key).value is not None


@pytest.mark.parametrize("rank, contacts", tuple(enumerate((180, 188, 171, 176, 297))))
def test_boltz_api_example_retains_geometry_for_all_models(rank, contacts):
    # Large example predictions are optional local files, not test dependencies.
    root = Path(__file__).resolve().parents[1] / "examples/SbPZS_Boltz_Api"
    if not root.is_dir():
        pytest.skip("Local SbPZS Boltz API example is not installed.")
    files = scan_prediction_path(root)
    try:
        index = StructureIndex.from_path(files.structure_path(rank))
        data = load_prediction_data(files, rank, load_pae=True, structure_index=index)
        assert index.atom_count == 3498
        assert len(index.token_map) == 522
        result = compute_interface_model(
            index,
            data.pae,
            data.confidence,
            InterfaceAnalysisOptions(),
            rank=rank,
            label=files.model(rank).display_label,
            object_name=f"model_{rank}",
        )
        assert dict(score(result, "pdockq").counts)["cb_contact_pairs"] == contacts
        for key in ("clis", "ilis", "pdockq", "pdockq2_dunbrack", "pdockq2_zhu"):
            assert score(result, key).value is not None
        assert score(result, "pdockq2_zhu", "B").value is not None
    finally:
        files.close()


def test_stale_errors_are_silent_and_failed_refresh_preserves_snapshot():
    state, _, presenter, runner, operations, viewer, service, coordinator = harness()
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions())
    runner.deliver(runner.task(lambda _: None))
    snapshot = service.report
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions(15))
    runner.on_error(runner.request_id, "failure")
    assert service.report is snapshot
    assert presenter.notices[-1].code == "interface_computation_failed"
    presenter.notices.clear()
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions())
    state.pred_files = None
    runner.on_error(runner.request_id, "stale failure")
    assert service.report is snapshot and not presenter.notices


def test_close_cancels_exact_lazy_action_without_merge():
    from FoldQC.lifecycle_support import DataLoadBatchResult

    state, files, presenter, runner, operations, viewer, service, coordinator = harness(
        False
    )
    requirement = None
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions())
    requirement = coordinator._data._pending[2].requirements[0]
    coordinator.interface_window_closed()
    assert not operations.is_busy and coordinator._data._pending is None
    incoming = PredictionData(
        "prediction",
        0,
        state.active_model_state.structure_index.path,
        files.provider,
        pae=np.ones((4, 4)),
    )
    runner.deliver(DataLoadBatchResult(files, ((requirement, incoming),)))
    assert state.active_model_state.data.pae is None and not presenter.reports


def test_snapshot_weak_sources_do_not_retain_model_arrays():
    import gc
    from weakref import ref

    state, _, presenter, runner, operations, viewer, service, coordinator = harness()
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions())
    value = runner.task(lambda _: None)
    runner.deliver(value)
    del value
    runner.task = None
    weak = ref(state.active_model_state)
    state.model_states.clear()
    state.pred_files = None
    gc.collect()
    assert weak() is None
    assert service.report is not None  # compact report is still exportable
    assert not service.is_current()


def test_busy_close_defers_owned_selection_cleanup():
    state, _, presenter, runner, operations, viewer, service, coordinator = harness()
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions())
    runner.deliver(runner.task(lambda _: None))
    service.select(
        InterfaceSelectionRequest(0, score(service.report.models[0], "clis"))
    )
    lease = operations.begin("analysis", title="another action", label="busy")
    service.window_closed()
    viewer.delete_names.assert_not_called()
    operations.finish(lease)
    viewer.delete_names.assert_called_once_with((INTERFACE_SELECTION,))


def test_preset_is_captured_and_invalid_interface_metric_cannot_color():
    state, _, presenter, runner, operations, viewer, service, coordinator = harness()
    with pytest.raises(ValueError):
        AnalysisRequest("interfaces", "model_0")
    with pytest.raises(AnalysisPreflightError, match="Analyze interfaces"):
        AnalysisResolver().resolve(
            AnalysisRequest("color", "model_0", "interface_ipsae"), state
        )
    coordinator.analyze_current_interfaces(InterfaceAnalysisOptions(15))
    captured = runner.task(lambda _: None)
    coordinator.invalidate_ui()
    runner.deliver(captured)
    assert service.report is None and not presenter.reports


def test_aggregate_csv_statistics_have_correct_coverage():
    first = compute()
    second = replace(first, rank=1, label="Model 1")
    snapshot = replace(
        report(),
        models=(first, second),
        aggregates=aggregate_interface_models((first, second)),
    )
    rows = [
        r
        for r in interface_report_rows(snapshot)
        if r.get("aggregation") == "population_std"
    ]
    assert rows and all(r["value"] == 0 and r["n_models"] == 2 for r in rows)
    assert all(json.loads(r["member_ranks"]) == [0, 1] for r in rows)
