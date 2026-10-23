"""Server-format PDB exports from localfold use the existing AF3 loader."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from FoldQC.loader import (
    load_prediction_confidence_summaries,
    load_prediction_data,
    scan_prediction_path,
)
from FoldQC.structure_index import StructureIndex

PDB_TEXT = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00100.00           C
HETATM    3  C1  LIG L   1       2.000   0.000   0.000  1.00 40.00           C
END
"""


def _write_job(
    root: Path, prefix: str, rank: int = 0, *, truncated: bool = False
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    structure = root / f"{prefix}_model_{rank}.pdb"
    structure.write_text(PDB_TEXT)
    json_prefix = "" if truncated else f"{prefix}_"
    (root / f"{json_prefix}summary_confidences_{rank}.json").write_text(
        json.dumps(
            {
                "ranking_score": rank / 100,
                "ptm": 0.8,
                "iptm": 0.7,
                "chain_ptm": [0.8, 0.6],
                "chain_iptm": [0.7, 0.7],
                "chain_pair_iptm": [[0.8, 0.7], [0.7, 0.6]],
            }
        )
    )
    (root / f"{json_prefix}full_data_{rank}.json").write_text(
        json.dumps(
            {
                "atom_plddts": [80.0, 100.0, 40.0],
                "pae": [[0.0, 1.0], [2.0, 0.0]],
                "contact_probs": [[1.0, 0.25], [0.25, 1.0]],
            }
        )
    )
    return structure


@pytest.mark.parametrize("prefix", ["af3_1", "boltz2_1"])
def test_localfold_pdb_loads_through_server_provider(tmp_path: Path, prefix: str):
    structure = _write_job(tmp_path, prefix)
    files = scan_prediction_path(tmp_path)
    index = StructureIndex.from_path(structure)

    # Loading and comparison must reuse the index or compact summary, rather
    # than indexing the structure again.
    with mock.patch.object(StructureIndex, "from_path", side_effect=AssertionError):
        data = load_prediction_data(
            files, load_contact_probs=True, structure_index=index
        )
        summaries = load_prediction_confidence_summaries(files)

    assert files.provider.key == "af3_server"
    assert files.provider.label == "AlphaFold 3 Server"
    assert files.n_models == 1
    assert not files.supports_ensemble
    model = files.model(0)
    assert model.structure_path == structure
    assert data.structure_path == structure
    assert model.metadata["source_prefix"] == prefix
    assert model.summary_path == tmp_path / f"{prefix}_summary_confidences_0.json"
    assert model.confidence_path == tmp_path / f"{prefix}_full_data_0.json"
    assert model.capabilities == {"plddt", "pae", "contact_probs"}
    assert index.token_map.chain_order == ("A", "L")
    assert index.atom_to_token == (0, 0, 1)
    assert index.token_map[1].atom_name == "C1"
    assert data.token_plddt_source == "provider_atom_mean"
    np.testing.assert_allclose(data.token_plddt, [0.9, 0.4])
    np.testing.assert_array_equal(data.token_plddt, index.structure_plddt)
    np.testing.assert_allclose(data.pae, [[0.0, 1.0], [2.0, 0.0]])
    np.testing.assert_allclose(data.contact_probs, [[1.0, 0.25], [0.25, 1.0]])
    assert data.confidence.ptm == 0.8
    assert data.confidence.iptm == 0.7
    assert summaries[0].ptm == 0.8
    np.testing.assert_allclose(data.confidence.chain_ptm, [0.8, 0.6])
    np.testing.assert_allclose(data.confidence.chain_iptm, [0.7, 0.7])
    np.testing.assert_allclose(
        data.confidence.pair_chain_iptm, [[0.8, 0.7], [0.7, 0.6]]
    )
    for array in (
        data.token_plddt,
        data.pae,
        data.contact_probs,
        data.confidence.chain_ptm,
        data.confidence.chain_iptm,
        data.confidence.pair_chain_iptm,
    ):
        assert array.flags.c_contiguous
        assert not array.flags.writeable


def test_pdb_numeric_ranks_and_truncated_sidecars(tmp_path: Path):
    for rank in (10, 2, 0):
        _write_job(tmp_path, "custom_job", rank, truncated=True)
    files = scan_prediction_path(tmp_path)

    assert [model.rank for model in files.models] == [0, 2, 10]
    assert files.supports_ensemble
    for model in files.models:
        assert model.confidence_path.name == f"full_data_{model.rank}.json"
        assert model.summary_path.name == f"summary_confidences_{model.rank}.json"
        data = load_prediction_data(files, rank=model.rank)
        assert data.confidence.ranking_score == model.rank / 100


def test_cif_preferred_only_for_matching_model_basename(tmp_path: Path):
    pdb = _write_job(tmp_path, "custom_job")
    cif = pdb.with_suffix(".cif")
    # Discovery does not read the structure; its content is immaterial here.
    cif.write_text("data_test\n")
    other_pdb = _write_job(tmp_path, "custom_job", rank=1)
    files = scan_prediction_path(tmp_path)

    assert [model.structure_path for model in files.models] == [cif, other_pdb]
    assert files.n_models == 2


def test_pdb_matrix_flags_and_b_factor_fallback(tmp_path: Path):
    _write_job(tmp_path, "custom_job")
    files = scan_prediction_path(tmp_path)
    data = load_prediction_data(files, load_pae=False, load_contact_probs=False)
    assert data.pae is None
    assert data.contact_probs is None
    assert data.token_plddt_source == "provider_atom_mean"

    full_path = files.model(0).confidence_path
    full_path.write_text("{}")
    with mock.patch("FoldQC.providers.alphafold._load_json") as read_full:
        unloaded = load_prediction_data(
            files, load_pae=False, load_contact_probs=False, load_token_plddt=False
        )
    read_full.assert_not_called()
    assert unloaded.token_plddt is None
    fallback = load_prediction_data(files, load_pae=False)
    assert fallback.token_plddt_source == "structure_b_factor"
    np.testing.assert_allclose(fallback.token_plddt, [0.9, 0.4])


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("atom_plddts", [80.0], "does not match 3 atoms"),
        ("pae", [[0.0]], r"pae must have shape \(2, 2\)"),
        ("contact_probs", [[1.0]], r"contact_probs must have shape \(2, 2\)"),
    ],
)
def test_pdb_rejects_malformed_confidence_arrays(
    tmp_path: Path, field: str, value: list, error: str
):
    _write_job(tmp_path, "custom_job")
    files = scan_prediction_path(tmp_path)
    full_path = files.model(0).confidence_path
    payload = json.loads(full_path.read_text())
    payload[field] = value
    full_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=error):
        load_prediction_data(files, load_contact_probs=True)


@pytest.mark.parametrize("wrapper", ["", "download/prediction"])
def test_pdb_zip_preserves_input_path_and_owns_extraction(tmp_path: Path, wrapper: str):
    root = tmp_path / "source"
    _write_job(root, "custom_job")
    archive = tmp_path / "prediction.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in root.iterdir():
            zf.write(path, Path(wrapper) / path.name)

    files = scan_prediction_path(archive)
    extracted = files.pred_dir
    try:
        assert files.input_path == archive
        assert files.provider.key == "af3_server"
        assert files.structure_path(0).suffix == ".pdb"
        data = load_prediction_data(files, load_pae=False)
        assert data.pae is None
        assert extracted.exists()
        # Lazy reads still work after scanning and the initial load finish.
        data = load_prediction_data(files, load_contact_probs=True)
        assert data.pae.shape == (2, 2)
        assert data.contact_probs.shape == (2, 2)
    finally:
        files.close()
    assert not extracted.exists()
    files.close()


def test_server_missing_structure_error_mentions_both_formats(tmp_path: Path):
    (tmp_path / "custom_job_full_data_0.json").write_text("{}")
    (tmp_path / "custom_job_model_invalid.pdb").write_text(PDB_TEXT)
    with pytest.raises(ValueError, match="model CIF/PDB files"):
        scan_prediction_path(tmp_path)
