"""Geometry-only diagnostics and current-snapshot contact audit regressions."""

import csv
import json
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from FoldQC.export import (
    CONTACT_COLUMNS,
    interface_contact_rows,
    interface_report_rows,
    prepare_interface_contact_csv,
)
from FoldQC.interface_analysis import InterfaceContactExportRequest
from FoldQC.interface_compute import aggregate_interface_models, iter_interface_contacts
from FoldQC.interface_report_data import report_rows, score_details
from FoldQC.structure_index import StructureGeometry, StructureIndex
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap
from test_interface_analysis import compute, harness, reference_index, report, score


def diagnostic(model, scope="chain_pair", chain="A"):
    return next(d for d in model.contacts if d.scope == scope and d.chain_a == chain)


def pairs(index, d):
    return list(iter_interface_contacts(index, d, lambda: None))


def test_geometry_diagnostics_independent_of_scores():
    index = reference_index()
    model = compute(index, pae=None)
    g = index.geometry
    no_confidence = replace(
        index,
        geometry=replace(g, ca_plddt=np.full(4, np.nan), cb_plddt=np.full(4, np.nan)),
    )
    missing_confidence = compute(no_confidence, pae=None)
    assert model.contacts == missing_confidence.contacts == compute(index).contacts
    assert score(missing_confidence, "pdockq").value is None
    assert score(model, "pdockq2_zhu").value is None
    assert diagnostic(model, "chain_rest").pair_count > 0
    for d in model.contacts:
        audit = pairs(index, d)
        assert len(audit) == d.pair_count
        assert len({a for a, _, _ in audit}) == d.residues_a
        assert len({b for _, b, _ in audit}) == d.residues_b
    row = report_rows(report(model), "chain_pair")[0]
    for s in row.scores:
        text = score_details(report(model), row, s.metric_key)
        assert "Geometric contacts: 2 pairs" in text
        assert "Cβ, Cα for glycine; distance ≤8 Å" in text
        assert "Metric support counts" in text


def test_missing_geometry_never_exposes_partial_counts():
    index = reference_index(missing=True)
    model = compute(index)
    cb = diagnostic(model)
    assert (cb.pair_count, cb.residues_a, cb.residues_b) == (None, None, None)
    assert cb.reason and diagnostic(model, "chain_rest").pair_count is not None
    ca = index.geometry.ca.copy()
    ca[2] = np.nan
    model = compute(replace(index, geometry=replace(index.geometry, ca=ca)))
    assert all(d.pair_count is None for d in model.contacts)
    legacy = compute(replace(index, geometry=None))
    assert all(d.status == "unavailable" for d in legacy.contacts)
    with pytest.raises(ValueError, match="unavailable"):
        pairs(index, cb)


def synthetic_index():
    # Interleaved chains, insertion code, glycine and a third partner. Non-protein
    # and HETATM tokens overlap spatially but must never enter the contact audit.
    entries = (
        ("B", "ALA", False),
        ("A", "GLY", False),
        ("C", "ALA", False),
        ("A", "ALA", False),
        ("D", "DA", False),
        ("A", "MSE", True),
    )
    tm = TokenMap(
        tuple(
            TokenInfo(
                i,
                c,
                ResidueId(-2 if i == 1 else i + 1, "A" if i == 1 else ""),
                r,
                het,
                "CA" if het else None,
            )
            for i, (c, r, het) in enumerate(entries)
        )
    )
    coords = np.array(
        [[8, 0, 0], [0, 0, 0], [0, 8, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], float
    )
    g = StructureGeometry(
        coords,
        coords,
        np.full(6, 90),
        np.full(6, 90),
        np.array([True, True, True, True, False, True]),
    )
    return StructureIndex(
        Path("/tmp/synthetic.cif"), "cif", tm, 6, tuple(range(6)), np.full(6, 90), g
    )


def test_audit_boundaries_multiplicity_identity_and_rest_union(tmp_path):
    index = synthetic_index()
    model = compute(index, pae=None)
    d = diagnostic(model, "chain_rest")
    assert (d.pair_count, d.residues_a, d.residues_b) == (4, 2, 2)
    audit = pairs(index, d)
    assert audit == [(1, 0, 8.0), (1, 2, 8.0), (3, 0, 8.0), (3, 2, 8.0)]
    cb = next(
        d for d in model.contacts if d.scope == "chain_pair" and d.partners == ("A",)
    )
    rows = list(interface_contact_rows(report(model), model, cb, pairs(index, cb)))
    gly = next(r for r in rows if r["res_name_b"] == "GLY")
    assert (gly["atom_name_b"], gly["residue_number_b"], gly["insertion_code_b"]) == (
        "CA",
        -2,
        "A",
    )
    assert gly["atom_name_a"] == "CB"
    destination = tmp_path / "contacts.csv"
    prepared = prepare_interface_contact_csv(str(destination), rows, lambda: None)
    assert not destination.exists()
    prepared.commit(str(destination))
    prepared.close()
    with destination.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(CONTACT_COLUMNS)
        assert len(list(reader)) == 2
    beyond = index.geometry.ca.copy()
    beyond[0, 0] = np.nextafter(8.0, np.inf)
    changed = replace(index, geometry=replace(index.geometry, ca=beyond, cb=beyond))
    assert diagnostic(compute(changed, pae=None), "chain_rest").pair_count == 2


def test_report_csv_v2_and_empty_contact_csv(tmp_path):
    model = compute(reference_index(distant=True))
    assert all(
        (d.pair_count, d.residues_a, d.residues_b) == (0, 0, 0) for d in model.contacts
    )
    second = replace(model, rank=1)
    snapshot = replace(
        report(model),
        models=(model, second),
        aggregates=aggregate_interface_models((model, second)),
    )
    rows = interface_report_rows(snapshot)
    assert {r["interface_schema_version"] for r in rows} == {"2"}
    for row in rows:
        if "model_rank" in row:
            d = json.loads(row["contact_diagnostics"])
            assert d["status"] == "ok" and d["pair_count"] == 0
        else:
            assert row.get("contact_diagnostics", "") == ""
    aggregate = next(r for r in report_rows(snapshot, "chain_pair") if r.rank is None)
    assert "per model only" in score_details(snapshot, aggregate, "interface_pdockq")
    d = diagnostic(model)
    output = tmp_path / "empty.csv"
    prepared = prepare_interface_contact_csv(
        str(output),
        interface_contact_rows(
            snapshot, model, d, pairs(reference_index(distant=True), d)
        ),
        lambda: None,
    )
    prepared.commit(str(output))
    assert len(output.read_text().splitlines()) == 1


def ready_export(tmp_path):
    state, files, presenter, runner, operations, viewer, service, coordinator = (
        harness()
    )
    coordinator.open_interfaces()
    runner.deliver(runner.task(lambda _: None))
    output = tmp_path / "contacts.csv"
    output.write_text("previous output")
    request = InterfaceContactExportRequest(
        0, diagnostic(service.report.models[0]), str(output)
    )
    return (
        state,
        presenter,
        runner,
        operations,
        viewer,
        service,
        coordinator,
        output,
        request,
    )


def test_contact_service_commits_without_viewer_or_structure_reads(tmp_path):
    (
        state,
        presenter,
        runner,
        operations,
        viewer,
        service,
        coordinator,
        output,
        request,
    ) = ready_export(tmp_path)
    viewer.reset_mock()
    coordinator.export_interface_contacts(request)
    with patch.object(
        StructureIndex, "from_path", side_effect=AssertionError("reread")
    ):
        result = runner.task(lambda _: None)
    assert output.read_text() == "previous output"
    runner.deliver(result)
    assert "distance_angstrom" in output.read_text()
    assert not result.path.exists() and not operations.is_busy
    assert not viewer.update_object_token_selection.called
    assert set(call[0] for call in viewer.mock_calls) <= {"name_exists"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "invalidate",
    ["source", "version", "target", "object", "close", "cancel", "refresh"],
)
def test_contact_late_result_does_not_publish(tmp_path, invalidate):
    (
        state,
        presenter,
        runner,
        operations,
        viewer,
        service,
        coordinator,
        output,
        request,
    ) = ready_export(tmp_path)
    coordinator.export_interface_contacts(request)
    result = runner.task(lambda _: None)
    before_notices = len(presenter.notices)
    if invalidate == "source":
        state.pred_files = None
    elif invalidate == "version":
        state.active_model_state._version += 1
    elif invalidate == "target":
        service.refresh_current("another_target")
    elif invalidate == "object":
        viewer.name_exists.return_value = False
    elif invalidate == "close":
        coordinator.interface_window_closed()
    elif invalidate == "cancel":
        service._cancel_contact_export()
    else:
        service._report = replace(service.report)
    runner.deliver(result)
    assert output.read_text() == "previous output"
    assert not result.path.exists() and not operations.is_busy
    assert len(presenter.notices) == before_notices


def test_stale_or_forged_request_rejected_before_worker(tmp_path):
    (
        state,
        presenter,
        runner,
        operations,
        viewer,
        service,
        coordinator,
        output,
        request,
    ) = ready_export(tmp_path)
    old_task = runner.task
    coordinator.export_interface_contacts(
        replace(request, diagnostic=replace(request.diagnostic))
    )
    assert runner.task is old_task and not operations.is_busy
    service.refresh_current("another_target")
    coordinator.export_interface_contacts(request)
    assert runner.task is old_task and output.read_text() == "previous output"


def test_worker_cancel_and_write_failure_remove_partial_file(tmp_path):
    (
        state,
        presenter,
        runner,
        operations,
        viewer,
        service,
        coordinator,
        output,
        request,
    ) = ready_export(tmp_path)
    coordinator.export_interface_contacts(request)
    task = runner.task
    service._cancel_contact_export()
    with pytest.raises(InterruptedError):
        task(lambda _: None)

    def failed_rows():
        yield {"distance_angstrom": 1.5}
        raise OSError("write failed")

    with pytest.raises(OSError):
        prepare_interface_contact_csv(str(output), failed_rows(), lambda: None)
    assert not list(tmp_path.glob("*.tmp"))
    assert output.read_text() == "previous output"


def test_commit_failure_and_stale_error_cleanup(tmp_path):
    (
        state,
        presenter,
        runner,
        operations,
        viewer,
        service,
        coordinator,
        output,
        request,
    ) = ready_export(tmp_path)
    coordinator.export_interface_contacts(request)
    result = runner.task(lambda _: None)
    with patch.object(Path, "replace", side_effect=OSError("cannot replace")):
        runner.deliver(result)
    assert not result.path.exists() and not operations.is_busy
    assert output.read_text() == "previous output"
    coordinator.export_interface_contacts(request)
    state.pred_files = None
    before = len(presenter.notices)
    runner.on_error(runner.request_id, OSError("late failure"))
    assert len(presenter.notices) == before and not operations.is_busy


def test_contact_tiles_cancel_and_audit_mismatch(tmp_path):
    index = reference_index()
    d = diagnostic(compute(index))

    def cancel():
        raise InterruptedError("stop")

    with pytest.raises(InterruptedError):
        list(iter_interface_contacts(index, d, cancel))
    wrong = replace(d, pair_count=d.pair_count + 1)
    with pytest.raises(ValueError, match="does not match"):
        prepare_interface_contact_csv(
            str(tmp_path / "bad.csv"),
            interface_contact_rows(
                report(),
                compute(index),
                wrong,
                iter_interface_contacts(index, wrong, lambda: None),
            ),
            lambda: None,
        )
    assert list(tmp_path.iterdir()) == []


def test_close_failure_and_midstream_cancellation_cleanup(tmp_path):
    original = tempfile.NamedTemporaryFile

    @contextmanager
    def failing_close(**kwargs):
        with original(**kwargs) as stream:
            yield stream
        raise OSError("close failed")

    destination = str(tmp_path / "contacts.csv")
    with patch("FoldQC.export.tempfile.NamedTemporaryFile", failing_close):
        with pytest.raises(OSError, match="close failed"):
            prepare_interface_contact_csv(destination, [], lambda: None)
    assert list(tmp_path.iterdir()) == []
    checks = 0

    def cancel_after_one_row():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise InterruptedError("cancel midstream")

    with pytest.raises(InterruptedError):
        prepare_interface_contact_csv(
            destination, [{"distance_angstrom": 1.0}] * 2, cancel_after_one_row
        )
    assert list(tmp_path.iterdir()) == []


def test_completed_export_does_not_retain_canonical_geometry(tmp_path):
    import gc
    from weakref import ref

    (
        state,
        presenter,
        runner,
        operations,
        viewer,
        service,
        coordinator,
        output,
        request,
    ) = ready_export(tmp_path)
    geometry = ref(state.active_model_state.structure_index.geometry)
    model = ref(state.active_model_state)
    coordinator.export_interface_contacts(request)
    runner.deliver(runner.task(lambda _: None))
    runner.task = None  # A real runner releases its completed task closure.
    state.model_states.clear()
    state.pred_files = None
    gc.collect()
    assert model() is None and geometry() is None
    assert service.report is not None
    assert diagnostic(service.report.models[0]).pair_count == 2
