from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import export


def _token(
    idx: int,
    *,
    chain_id: str = "A",
    res_num: int | None = None,
    is_hetatm: bool = False,
):
    return types.SimpleNamespace(
        token_idx=idx,
        chain_id=chain_id,
        res_num=idx + 1 if res_num is None else res_num,
        res_name="LIG" if is_hetatm else "ALA",
        is_hetatm=is_hetatm,
        atom_name=f"C{idx}" if is_hetatm else None,
    )


class _PredictionFiles:
    def __init__(self, root: Path, ranks=(0, 1)) -> None:
        self.name = "target"
        self.provider = "boltz"
        self.input_path = root
        self.pred_dir = root
        self.models = [
            types.SimpleNamespace(rank=rank, display_label=f"rank {rank}")
            for rank in ranks
        ]

    def model(self, rank: int):
        return next(model for model in self.models if model.rank == rank)


def test_fieldnames_add_ensemble_columns_only_when_requested() -> None:
    base = export.fieldnames()
    with_ensemble = export.fieldnames(include_ensemble=True)

    assert base == export.BASE_COLUMNS
    assert with_ensemble == export.BASE_COLUMNS + export.ENSEMBLE_COLUMNS
    assert "ensemble_group" not in base
    assert "ensemble_group" in with_ensemble


def test_default_csv_export_path_uses_prediction_metadata(tmp_path: Path) -> None:
    pred_files = _PredictionFiles(tmp_path)
    pred_data = types.SimpleNamespace(rank=1)

    path = export.default_csv_export_path(pred_files, pred_data, "plddt")

    assert path == str(tmp_path / "target_rank1_plddt.csv")


def test_default_csv_export_path_falls_back_to_home(tmp_path: Path) -> None:
    path = export.default_csv_export_path(None, None, None, home=tmp_path)

    assert path == str(tmp_path / "foldqc_metric.csv")


def test_model_label_for_rank_returns_display_label_or_fallback(tmp_path: Path) -> None:
    pred_files = _PredictionFiles(tmp_path, ranks=(0,))

    assert export.model_label_for_rank(pred_files, 0) == "rank 0"
    assert export.model_label_for_rank(pred_files, 9, fallback="model_9") == "model_9"


def test_build_token_rows_formats_base_metadata_and_token_values(
    tmp_path: Path,
) -> None:
    pred_files = _PredictionFiles(tmp_path)
    data = types.SimpleNamespace(
        name="target",
        provider="boltz",
        rank=0,
        display_label="rank 0",
        structure_path=tmp_path / "target_model_0.cif",
    )
    token_map = [_token(0), _token(1, chain_id="L", is_hetatm=True)]

    rows = export.build_token_rows(
        pred_files=pred_files,
        data=data,
        token_map=token_map,
        values=np.array([1.25, np.nan], dtype=np.float32),
        metric_key="plddt",
        reference_selection="chain L",
        cutoff_angstrom=None,
        reference_indices=[1],
        contact_indices=[0],
    )

    assert len(rows) == 2
    first = rows[0]
    assert first["export_schema_version"] == "1"
    assert first["provider"] == "boltz"
    assert first["prediction_name"] == "target"
    assert first["input_path"] == str(tmp_path)
    assert first["structure_path"] == str(tmp_path / "target_model_0.cif")
    assert first["model_rank"] == 0
    assert first["model_label"] == "rank 0"
    assert first["metric_key"] == "plddt"
    assert first["metric_label"] == "pLDDT \u2014 continuous"
    assert first["value"] == 1.25
    assert first["value_units"] == "plddt"
    assert first["value_semantics"] == "higher_is_better"
    assert first["reference_selection"] == "chain L"
    assert first["cutoff_angstrom"] == ""
    assert first["token_index"] == 0
    assert first["token_type"] == "polymer_residue"
    assert first["chain_id"] == "A"
    assert first["res_num"] == 1
    assert first["res_name"] == "ALA"
    assert first["atom_name"] == ""
    assert first["is_hetatm"] == "false"
    assert first["is_reference_token"] == "false"
    assert first["is_contact_token"] == "true"

    second = rows[1]
    assert second["value"] == "nan"
    assert second["token_type"] == "ligand_atom"
    assert second["chain_id"] == "L"
    assert second["atom_name"] == "C1"
    assert second["is_hetatm"] == "true"
    assert second["is_reference_token"] == "true"
    assert second["is_contact_token"] == "false"


def test_build_token_rows_adds_ensemble_metadata(tmp_path: Path) -> None:
    pred_files = _PredictionFiles(tmp_path)
    data = types.SimpleNamespace(
        provider="boltz",
        name="target",
        rank=1,
        display_label="rank 1",
        structure_path=tmp_path / "target_model_1.cif",
    )

    rows = export.build_token_rows(
        pred_files=pred_files,
        data=data,
        token_map=[_token(0)],
        values=np.array([0.5], dtype=np.float32),
        metric_key="ensemble_plddt_std",
        metric_label="custom label",
        cutoff_angstrom=7.5,
        include_ensemble=True,
        ensemble_group="target_ensemble",
        ensemble_member_rank=1,
        ensemble_member_label="rank 1",
        ensemble_aligned=True,
        aggregate_kind="ensemble_std",
    )

    assert rows == [
        {
            "export_schema_version": "1",
            "provider": "boltz",
            "prediction_name": "target",
            "input_path": str(tmp_path),
            "structure_path": str(tmp_path / "target_model_1.cif"),
            "model_rank": 1,
            "model_label": "rank 1",
            "metric_key": "ensemble_plddt_std",
            "metric_label": "custom label",
            "value": 0.5,
            "value_units": "plddt",
            "value_semantics": "lower_is_better",
            "reference_selection": "",
            "cutoff_angstrom": 7.5,
            "token_index": 0,
            "token_type": "polymer_residue",
            "chain_id": "A",
            "res_num": 1,
            "res_name": "ALA",
            "atom_name": "",
            "is_hetatm": "false",
            "is_reference_token": "false",
            "is_contact_token": "false",
            "ensemble_group": "target_ensemble",
            "ensemble_member_rank": 1,
            "ensemble_member_label": "rank 1",
            "ensemble_aligned": "true",
            "aggregate_kind": "ensemble_std",
        }
    ]


def test_write_csv_uses_stable_header_and_row_values(tmp_path: Path) -> None:
    path = tmp_path / "tokens.csv"
    rows = [
        {
            "export_schema_version": "1",
            "provider": "boltz",
            "prediction_name": "target",
            "input_path": str(tmp_path),
            "structure_path": str(tmp_path / "target_model_0.cif"),
            "model_rank": 0,
            "model_label": "rank 0",
            "metric_key": "plddt",
            "metric_label": "pLDDT \u2014 continuous",
            "value": 1.25,
            "value_units": "plddt",
            "value_semantics": "higher_is_better",
            "reference_selection": "",
            "cutoff_angstrom": "",
            "token_index": 0,
            "token_type": "polymer_residue",
            "chain_id": "A",
            "res_num": 1,
            "res_name": "ALA",
            "atom_name": "",
            "is_hetatm": "false",
            "is_reference_token": "false",
            "is_contact_token": "false",
        }
    ]

    export.write_csv(path, rows)

    with path.open() as fh:
        reader = csv.DictReader(fh)
        written = list(reader)

    assert reader.fieldnames == export.BASE_COLUMNS
    assert written[0]["metric_key"] == "plddt"
    assert written[0]["value"] == "1.25"
