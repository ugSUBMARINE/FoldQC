from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.ensemble import (  # noqa: E402
    EnsembleMember,
    EnsembleState,
    calculate_alignment_plan,
    compute_metric_consensus,
    compute_per_token_rmsd,
    default_group_name,
    invert_rigid_transform,
    kabsch_transform,
    prepare_ensemble,
    select_alignment_core,
)
from FoldQC.model_state import ModelState  # noqa: E402
from FoldQC.structure_index import StructureIndex  # noqa: E402
from FoldQC.token_map import TokenInfo, TokenMap  # noqa: E402


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


def _index(path: Path, token_map: TokenMap) -> StructureIndex:
    plddt = np.zeros(len(token_map), dtype=np.float32)
    plddt.setflags(write=False)
    return StructureIndex(
        path=path,
        format="cif",
        token_map=token_map,
        atom_count=len(token_map),
        atom_to_token=tuple(range(len(token_map))),
        structure_plddt=plddt,
    )


class EnsembleTests(unittest.TestCase):
    def test_alignment_core_uses_only_high_confidence_polymer_tokens(self) -> None:
        token_map = TokenMap(
            (_token(0), _token(1), _token(2), _token(3, is_hetatm=True))
        )
        plddt = np.array([0.9, 0.81, 0.7, 0.99], dtype=np.float32)

        self.assertEqual(select_alignment_core(token_map, plddt, min_tokens=2), [0, 1])

    def test_alignment_core_falls_back_to_all_polymer_tokens(self) -> None:
        token_map = TokenMap(
            (_token(0), _token(1), _token(2), _token(3, is_hetatm=True))
        )
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

    def test_inverse_rigid_transform_restores_coordinates(self) -> None:
        coords = np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 1.0, 2.0]],
            dtype=np.float64,
        )
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        translation = np.array([2.0, 3.0, 4.0], dtype=np.float64)

        transformed = coords @ rotation.T + translation
        inverse_rotation, inverse_translation = invert_rigid_transform(
            rotation, translation
        )

        np.testing.assert_allclose(
            transformed @ inverse_rotation.T + inverse_translation,
            coords,
            atol=1e-6,
        )


def test_calculate_alignment_plan_uses_current_coordinate_arrays() -> None:
    token_map = TokenMap(tuple(_token(i) for i in range(3)))
    members = [
        types.SimpleNamespace(rank=0, token_map=token_map),
        types.SimpleNamespace(rank=1, token_map=token_map),
    ]
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    mobile = reference + np.array([4.0, -2.0, 1.0], dtype=np.float32)

    plan = calculate_alignment_plan(
        members,
        {0: reference, 1: mobile},
        reference_rank=0,
        core_indices=(0, 1, 2),
    )

    assert [transform.rank for transform in plan.transforms] == [1]
    np.testing.assert_allclose(plan.transformed_coords[0], reference, atol=1e-6)
    np.testing.assert_allclose(plan.transformed_coords[1], reference, atol=1e-6)
    np.testing.assert_allclose(plan.rmsd, np.zeros(3), atol=1e-6)


def test_default_group_name_derives_current_gui_name() -> None:
    pred_files = types.SimpleNamespace(
        models=[types.SimpleNamespace(object_name="target_model_0")]
    )

    assert default_group_name(pred_files) == "target_model_ensemble"


def test_ensemble_state_contains_only_rank_viewer_metadata_and_readonly_arrays() -> (
    None
):
    member = EnsembleMember(rank=2, obj_name="target_model_2")
    state = EnsembleState(
        group_name="target_ensemble",
        members=(member,),
        aligned=True,
        rmsd=np.array([0.1], dtype=np.float32),
        plddt_mean=np.array([0.8], dtype=np.float32),
        plddt_std=np.array([0.05], dtype=np.float32),
    )

    assert state.ranks == (2,)
    assert not hasattr(member, "data")
    assert not hasattr(member, "token_map")
    assert not hasattr(member, "model_state")
    assert state.rmsd.flags.writeable is False
    with pytest.raises(ValueError):
        state.rmsd[0] = 1.0


def test_prepare_ensemble_reuses_loaded_rank_and_prepares_missing_rank(
    monkeypatch,
) -> None:
    pred_files = types.SimpleNamespace(
        models=[
            types.SimpleNamespace(
                rank=0,
                object_name="target_model_0",
                structure_path=Path("/tmp/target_model_0.cif"),
                display_label="model_0",
            ),
            types.SimpleNamespace(
                rank=1,
                object_name="target_model_1",
                structure_path=Path("/tmp/target_model_1.cif"),
                display_label="model_1",
            ),
        ]
    )
    existing_data = types.SimpleNamespace(
        rank=0,
        structure_path=Path("/tmp/target_model_0.cif"),
        token_plddt=np.array([0.8, 0.9, 1.0], dtype=np.float32),
        token_plddt_source="structure_b_factor",
        pae=np.ones((3, 3), dtype=np.float32),
        pde=None,
        contact_probs=None,
    )
    existing_map = TokenMap((_token(0), _token(1), _token(2)))
    existing_index = _index(existing_data.structure_path, existing_map)
    existing = ModelState(0, existing_data, existing_index)
    load_calls = []
    map_paths = []
    phases = []

    def load_data(pred, rank, **kwargs):
        load_calls.append((pred, rank, kwargs))
        return types.SimpleNamespace(
            rank=rank,
            structure_path=Path(f"/tmp/target_model_{rank}.cif"),
            token_plddt=np.array([0.6, 0.7, 0.8], dtype=np.float32),
            token_plddt_source="provider_token",
        )

    def build_index(path):
        map_paths.append(path)
        token_map = TokenMap((_token(0), _token(1), _token(2)))
        return _index(path, token_map)

    monkeypatch.setattr("FoldQC.loader.load_prediction_data", load_data)
    monkeypatch.setattr("FoldQC.structure_index.StructureIndex.from_path", build_index)

    result = prepare_ensemble(
        pred_files,
        skip_alignment=False,
        existing_states_by_rank={0: existing},
        report_phase=phases.append,
    )

    assert [member.rank for member in result.members] == [0, 1]
    assert result.members[0].model_state is existing
    assert result.members[0].data is existing_data
    assert load_calls == [
        (
            pred_files,
            1,
            {
                "load_pae": False,
                "load_pde": False,
                "load_contact_probs": False,
                "load_token_plddt": True,
                "structure_index": result.members[1].model_state.structure_index,
            },
        )
    ]
    assert map_paths == [Path("/tmp/target_model_1.cif")]
    assert result.reference_rank == 0
    assert result.core_indices == (0, 1, 2)
    np.testing.assert_allclose(result.plddt_mean, np.array([0.7, 0.8, 0.9]))
    assert phases[-1] == "Validating ensemble token maps…"


def test_prepare_ensemble_stages_plddt_reload_without_mutating_existing_state(
    monkeypatch,
) -> None:
    model = types.SimpleNamespace(
        rank=0,
        object_name="target_model_0",
        structure_path=Path("/tmp/target_model_0.cif"),
        display_label="model_0",
    )
    pred_files = types.SimpleNamespace(models=[model])
    original_data = types.SimpleNamespace(
        rank=0,
        structure_path=model.structure_path,
        token_plddt=None,
        pae=np.ones((3, 3), dtype=np.float32),
        pde=None,
        contact_probs=None,
    )
    token_map = TokenMap((_token(0), _token(1), _token(2)))
    structure_index = _index(model.structure_path, token_map)
    existing = ModelState(0, original_data, structure_index)
    reloaded_data = types.SimpleNamespace(
        rank=0,
        structure_path=model.structure_path,
        token_plddt=np.array([0.7, 0.8, 0.9], dtype=np.float32),
        token_plddt_source="structure_b_factor",
        pae=np.ones((3, 3), dtype=np.float32),
        pde=None,
        contact_probs=None,
    )
    load_calls = []

    def load_data(prediction_files, rank, **kwargs):
        load_calls.append((prediction_files, rank, kwargs))
        return reloaded_data

    monkeypatch.setattr("FoldQC.loader.load_prediction_data", load_data)

    result = prepare_ensemble(
        pred_files,
        skip_alignment=True,
        existing_states_by_rank={0: existing},
    )

    prepared_state = result.members[0].model_state
    assert prepared_state is not existing
    assert prepared_state.data is reloaded_data
    assert prepared_state.token_map is token_map
    assert existing.data is original_data
    assert load_calls == [
        (
            pred_files,
            0,
            {
                "load_pae": False,
                "load_pde": False,
                "load_contact_probs": False,
                "load_token_plddt": True,
                "structure_index": structure_index,
            },
        )
    ]


def test_prepare_ensemble_rejects_different_ordered_tokens(monkeypatch) -> None:
    pred_files = types.SimpleNamespace(
        models=[
            types.SimpleNamespace(
                rank=rank,
                object_name=f"target_model_{rank}",
                display_label=f"model_{rank}",
                structure_path=Path(f"/tmp/target_model_{rank}.cif"),
            )
            for rank in (0, 1)
        ]
    )

    def load_data(_pred, rank, **_kwargs):
        return types.SimpleNamespace(
            rank=rank,
            structure_path=Path(f"/tmp/target_model_{rank}.cif"),
            token_plddt=np.ones(3, dtype=np.float32),
            token_plddt_source="provider_token",
        )

    indexes = iter(
        (
            _index(
                Path("/tmp/target_model_0.cif"),
                TokenMap((_token(0), _token(1), _token(2))),
            ),
            _index(
                Path("/tmp/target_model_1.cif"),
                TokenMap((_token(0), _token(1, is_hetatm=True), _token(2))),
            ),
        )
    )
    monkeypatch.setattr("FoldQC.loader.load_prediction_data", load_data)
    monkeypatch.setattr(
        "FoldQC.structure_index.StructureIndex.from_path", lambda _path: next(indexes)
    )

    with pytest.raises(ValueError, match="Token order mismatch for model_1"):
        prepare_ensemble(pred_files, skip_alignment=True)
