from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import FoldQC.ensemble as ensemble  # noqa: E402
from FoldQC.ensemble import (  # noqa: E402
    EnsembleMember,
    build_members,
    compute_metric_consensus,
    compute_per_token_rmsd,
    default_group_name,
    kabsch_transform,
    prepare_metrics,
    select_alignment_core,
    validate_members,
)
from FoldQC.mol_viewer import (  # noqa: E402
    load_models_as_objects,
    load_models_as_states,
)
from FoldQC.token_map import TokenInfo  # noqa: E402


def _token(idx: int, is_hetatm: bool = False) -> TokenInfo:
    atom_name = f"C{idx}" if is_hetatm else None
    return TokenInfo(
        token_idx=idx,
        chain_id="L" if is_hetatm else "A",
        res_num=idx + 1,
        res_name="LIG" if is_hetatm else "ALA",
        is_hetatm=is_hetatm,
        atom_name=atom_name,
    )


class EnsembleTests(unittest.TestCase):
    def test_default_pymol_names_are_plugin_scoped(self) -> None:
        class _Cmd:
            def __init__(self) -> None:
                self.loads: list[tuple] = []
                self.set_calls: list[tuple[str, str]] = []
                self.rebuild_calls = 0

            def get_names(self, _kind: str):
                return []

            def load(self, *args, **kwargs) -> None:
                self.loads.append((*args, kwargs))

            def set(self, name: str, value: str) -> None:
                self.set_calls.append((name, value))

            def rebuild(self) -> None:
                self.rebuild_calls += 1

        old_pymol = sys.modules.get("pymol")
        cmd = _Cmd()
        sys.modules["pymol"] = type("FakePymol", (), {"cmd": cmd})()
        try:
            state_obj = load_models_as_states([(0, Path("/tmp/a.cif"))])
            object_loads = load_models_as_objects([(0, Path("/tmp/a.cif"))])
        finally:
            if old_pymol is None:
                sys.modules.pop("pymol", None)
            else:
                sys.modules["pymol"] = old_pymol

        self.assertEqual(state_obj, "foldqc_ensemble")
        self.assertEqual(object_loads, [(0, "foldqc_model_0")])
        self.assertEqual(cmd.loads[0], ("/tmp/a.cif", "foldqc_ensemble", {"state": 1}))
        self.assertEqual(
            cmd.loads[1], ("/tmp/a.cif", "foldqc_model_0", {"quiet": 1, "zoom": 0})
        )

    def test_alignment_core_uses_only_high_confidence_polymer_tokens(self) -> None:
        token_map = [_token(0), _token(1), _token(2), _token(3, is_hetatm=True)]
        plddt = np.array([0.9, 0.81, 0.7, 0.99], dtype=np.float32)

        self.assertEqual(select_alignment_core(token_map, plddt, min_tokens=2), [0, 1])

    def test_alignment_core_falls_back_to_all_polymer_tokens(self) -> None:
        token_map = [_token(0), _token(1), _token(2), _token(3, is_hetatm=True)]
        plddt = np.array([0.9, 0.5, 0.4, 0.99], dtype=np.float32)

        self.assertEqual(select_alignment_core(token_map, plddt), [0, 1, 2])

    def test_per_token_rmsd_identical_coordinates_are_zero(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

        rmsd = compute_per_token_rmsd([coords, coords.copy()])

        np.testing.assert_allclose(rmsd, np.zeros(2, dtype=np.float32))

    def test_per_token_rmsd_detects_displaced_tokens(self) -> None:
        coords_a = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        coords_b = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, 1.0]], dtype=np.float32)

        rmsd = compute_per_token_rmsd([coords_a, coords_b])

        self.assertEqual(float(rmsd[0]), 0.0)
        self.assertGreater(float(rmsd[1]), 0.0)

    def test_metric_consensus_matches_numpy_mean_and_std(self) -> None:
        arrays = [
            np.array([0.8, 0.6], dtype=np.float32),
            np.array([1.0, 0.2], dtype=np.float32),
        ]

        mean, std = compute_metric_consensus(arrays)

        np.testing.assert_allclose(mean, np.array([0.9, 0.4], dtype=np.float32))
        np.testing.assert_allclose(std, np.array([0.1, 0.2], dtype=np.float32))

    def test_kabsch_transform_maps_mobile_to_target(self) -> None:
        mobile = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        rotation_true = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        translation_true = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        target = mobile @ rotation_true.T + translation_true

        rotation, translation = kabsch_transform(mobile, target)
        transformed = mobile @ rotation.T + translation

        np.testing.assert_allclose(transformed, target, atol=1e-6)


def _member(
    rank: int,
    *,
    obj_name: str | None = None,
    token_count: int = 3,
    plddt=None,
    structure_plddt=None,
) -> EnsembleMember:
    return EnsembleMember(
        rank=rank,
        obj_name=obj_name or f"target_model_{rank}",
        data=types.SimpleNamespace(
            rank=rank,
            structure_path=Path(f"/tmp/target_model_{rank}.cif"),
            plddt=plddt,
            structure_plddt=structure_plddt,
        ),
        token_map=[_token(i) for i in range(token_count)],
    )


def test_default_group_name_derives_current_gui_name() -> None:
    pred_files = types.SimpleNamespace(
        models=[types.SimpleNamespace(object_name="target_model_0")]
    )

    assert default_group_name(pred_files) == "target_model_ensemble"


def test_build_members_loads_data_and_reuses_reference_token_map(monkeypatch) -> None:
    pred_files = types.SimpleNamespace(
        models=[
            types.SimpleNamespace(
                object_name="target_model_0",
                rank=0,
                structure_path=Path("/tmp/target_model_0.cif"),
            ),
            types.SimpleNamespace(
                object_name="target_model_1",
                rank=1,
                structure_path=Path("/tmp/target_model_1.cif"),
            ),
        ],
    )
    loaded_calls = []
    load_data_calls = []

    def fake_load_models_as_objects(model_paths, *, obj_prefix, group_name):
        loaded_calls.append((model_paths, obj_prefix, group_name))
        return [(0, "target_model_0"), (1, "target_model_1")]

    def fake_load_prediction_data(pred, rank, **kwargs):
        load_data_calls.append((pred, rank, kwargs))
        return types.SimpleNamespace(
            rank=rank,
            structure_path=Path(f"/tmp/target_model_{rank}.cif"),
            plddt=None,
            structure_plddt=np.array([0.8, 0.9], dtype=np.float32),
        )

    def fake_build_token_map(structure_path):
        assert structure_path == Path("/tmp/target_model_0.cif")
        return [_token(0), _token(1, is_hetatm=True)]

    monkeypatch.setattr(ensemble, "load_models_as_objects", fake_load_models_as_objects)
    monkeypatch.setattr("FoldQC.loader.load_prediction_data", fake_load_prediction_data)
    monkeypatch.setattr("FoldQC.token_map.build_token_map", fake_build_token_map)

    group_name, members = build_members(pred_files)

    assert group_name == "target_model_ensemble"
    expected_model_paths = [(m.rank, m.structure_path) for m in pred_files.models]
    assert loaded_calls == [
        (
            expected_model_paths,
            "target_model",
            "target_model_ensemble",
        )
    ]
    assert [member.rank for member in members] == [0, 1]
    assert load_data_calls[0][2] == {
        "load_pae": False,
        "load_pde": False,
        "load_structure_plddt": True,
    }
    assert members[0].token_map is members[1].token_map


def test_validate_members_accepts_compatible_members() -> None:
    validate_members(
        [
            _member(0, plddt=np.ones(3), structure_plddt=np.ones(3)),
            _member(1, plddt=np.ones(3), structure_plddt=np.ones(3)),
        ]
    )


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ([], "No ensemble models were loaded."),
        (
            [_member(0, token_count=3), _member(1, token_count=2)],
            "Token count mismatch: target_model_1 maps to 2 tokens, "
            "but target_model_0 maps to 3 tokens.",
        ),
        (
            [_member(0, token_count=3), _member(1, token_count=3, plddt=np.ones(2))],
            "pLDDT length mismatch for model_1: 2 values for 3 tokens.",
        ),
        (
            [
                _member(0, token_count=3),
                _member(1, token_count=3, structure_plddt=np.ones(2)),
            ],
            "Structure pLDDT length mismatch for model_1: 2 values for 3 tokens.",
        ),
    ],
)
def test_validate_members_reports_existing_failure_messages(members, message) -> None:
    with pytest.raises(
        ValueError, match=message.replace("(", r"\(").replace(")", r"\)")
    ):
        validate_members(members)


def test_prepare_metrics_skip_alignment_uses_current_coordinates(monkeypatch) -> None:
    rmsd = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    monkeypatch.setattr(
        ensemble,
        "compute_aligned_per_token_rmsd",
        lambda members: rmsd,
    )
    members = [
        _member(0, plddt=None, structure_plddt=np.array([0.8, 0.9, 1.0])),
        _member(1, plddt=None, structure_plddt=np.array([0.6, 0.7, 0.8])),
    ]

    result = prepare_metrics(members, skip_alignment=True)

    assert result.aligned is False
    assert result.mode_label == "current coordinates"
    np.testing.assert_allclose(result.rmsd, rmsd)
    np.testing.assert_allclose(result.plddt_mean, np.array([0.7, 0.8, 0.9]))
    np.testing.assert_allclose(result.plddt_std, np.array([0.1, 0.1, 0.1]))
    assert members[0].data.plddt is None  # prepare_metrics must not mutate data


def test_current_coordinate_rmsd_reuses_each_object_inspection_for_painting(
    monkeypatch,
) -> None:
    members = [_member(0), _member(1)]
    calls = []

    def inspect(obj_name, _token_map):
        calls.append(obj_name)
        offset = 0.0 if obj_name.endswith("0") else 1.0
        return types.SimpleNamespace(
            paint_mapping=f"mapping:{obj_name}",
            representative_coords=np.full((3, 3), offset, dtype=np.float32),
        )

    monkeypatch.setattr(ensemble, "inspect_object_tokens", inspect)

    result = ensemble.compute_aligned_per_token_rmsd(members)

    assert calls == ["target_model_0", "target_model_1"]
    assert members[0].paint_mapping == "mapping:target_model_0"
    assert members[1].paint_mapping == "mapping:target_model_1"
    np.testing.assert_allclose(result, np.sqrt(3.0) / 2.0)


def test_prepare_metrics_aligns_to_rank_zero_or_first_member(monkeypatch) -> None:
    calls = []
    members = [
        _member(2, plddt=np.array([0.9, 0.9, 0.9], dtype=np.float32)),
        _member(3, plddt=np.array([0.7, 0.8, 0.9], dtype=np.float32)),
    ]

    def fake_select_alignment_core(token_map, plddt):
        calls.append(("core", token_map, plddt))
        return [0, 1, 2]

    def fake_align_objects_to_reference(input_members, core_indices, reference_rank):
        calls.append(("align", input_members, core_indices, reference_rank))
        return [
            np.zeros((3, 3), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
        ]

    monkeypatch.setattr(ensemble, "select_alignment_core", fake_select_alignment_core)
    monkeypatch.setattr(
        ensemble, "align_objects_to_reference", fake_align_objects_to_reference
    )
    monkeypatch.setattr(
        ensemble,
        "compute_aligned_per_token_rmsd",
        lambda _members: (_ for _ in ()).throw(AssertionError("should align first")),
    )

    result = prepare_metrics(members, skip_alignment=False)

    assert result.aligned is True
    assert result.mode_label == "automatic core alignment"
    assert calls[0] == ("core", members[0].token_map, members[0].data.plddt)
    assert calls[1] == ("align", members, [0, 1, 2], 2)
    np.testing.assert_allclose(
        result.rmsd,
        compute_per_token_rmsd(
            [
                np.zeros((3, 3), dtype=np.float32),
                np.ones((3, 3), dtype=np.float32),
            ]
        ),
    )


def test_prepare_metrics_prefers_rank_zero_alignment_reference(monkeypatch) -> None:
    calls = []
    members = [
        _member(2, plddt=np.array([0.7, 0.8, 0.9], dtype=np.float32)),
        _member(0, plddt=np.array([0.9, 0.9, 0.9], dtype=np.float32)),
    ]

    monkeypatch.setattr(
        ensemble,
        "select_alignment_core",
        lambda token_map, plddt: calls.append(("core", token_map, plddt)) or [0, 1, 2],
    )
    monkeypatch.setattr(
        ensemble,
        "align_objects_to_reference",
        lambda input_members, core_indices, reference_rank: (
            calls.append(("align", input_members, core_indices, reference_rank))
            or [
                np.zeros((3, 3), dtype=np.float32),
                np.ones((3, 3), dtype=np.float32),
            ]
        ),
    )

    prepare_metrics(members, skip_alignment=False)

    assert calls[0] == ("core", members[1].token_map, members[1].data.plddt)
    assert calls[1] == ("align", members, [0, 1, 2], 0)


def test_prepare_metrics_requires_plddt_for_alignment() -> None:
    members = [_member(0, plddt=None, structure_plddt=None)]

    with pytest.raises(
        ValueError, match="Automatic ensemble alignment requires pLDDT data."
    ):
        prepare_metrics(members, skip_alignment=False)


def test_prepare_metrics_requires_plddt_for_consensus(monkeypatch) -> None:
    monkeypatch.setattr(
        ensemble,
        "compute_aligned_per_token_rmsd",
        lambda _members: np.zeros(3, dtype=np.float32),
    )
    members = [
        _member(0, plddt=np.array([0.8, 0.9, 1.0])),
        _member(1, plddt=None, structure_plddt=None),
    ]

    with pytest.raises(ValueError, match="pLDDT data are not available for model_1."):
        prepare_metrics(members, skip_alignment=True)


if __name__ == "__main__":
    unittest.main()
