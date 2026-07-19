from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.loader import (  # noqa: E402
    load_prediction_confidence_summaries,
    load_prediction_data,
    scan_prediction_path,
)
from FoldQC.loader_discovery import (  # noqa: E402
    discover_prediction_candidates,
    scan_prediction_candidate,
)
from FoldQC.provider_errors import ProviderContractError  # noqa: E402

PDB_TEXT = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 80.00           C
HETATM    3  C1  LIG B   2       2.000   0.000   0.000  1.00 40.00           C
END
"""

CIF_TEXT = """data_test
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.auth_comp_id
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.B_iso_or_equiv
ATOM N ALA 1 A 90.0
ATOM CA ALA 1 A 80.0
HETATM C1 LIG 2 B 40.0
#
"""


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload))


def _write_sample(
    query_dir: Path,
    *,
    seed: int,
    sample: int,
    score: float,
    full: bool = True,
    clash: float = 0.0,
    chain_ptm: dict[str, float] | None = None,
) -> Path:
    seed_dir = query_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    base = f"{query_dir.name}_seed_{seed}_sample_{sample}"
    structure = seed_dir / f"{base}_model.pdb"
    structure.write_text(PDB_TEXT)
    _write_json(
        seed_dir / f"{base}_confidences_aggregated.json",
        {
            "sample_ranking_score": score,
            "avg_plddt": 88.5,
            "ptm": 0.8,
            "iptm": 0.7,
            "gpde": 0.4,
            "disorder": 0.1,
            "has_clash": clash,
            "chain_ptm": chain_ptm or {"B": 0.6, "A": 0.9},
            "chain_pair_iptm": {"(B, A)": 0.75},
            "bespoke_iptm": {"(A, B)": 0.65},
        },
    )
    if full:
        _write_json(
            seed_dir / f"{base}_confidences.json",
            {
                "plddt": [80.0, 100.0, 40.0],
                "pae": [[0.0, 1.0], [2.0, 0.0]],
                "pde": [[0.0, 0.5], [0.75, 0.0]],
            },
        )
    return structure


def test_openfold_query_scans_ranks_and_loads_lazy_data(tmp_path: Path) -> None:
    query = tmp_path / "query_a"
    _write_sample(query, seed=8, sample=2, score=0.7)
    _write_sample(query, seed=7, sample=1, score=0.9)

    files = scan_prediction_path(query)
    data = load_prediction_data(files, rank=0, load_pae=True, load_pde=True)

    assert files.provider.key == "openfold3"
    assert files.provider.label == "OpenFold3"
    assert files.name == "query_a"
    assert [model.rank for model in files.models] == [0, 1]
    assert [model.metadata["seed"] for model in files.models] == [7, 8]
    assert [model.metadata["sample"] for model in files.models] == [1, 2]
    assert files.models[0].display_label == "rank 0 - seed 7 sample 1"
    assert files.models[0].object_name == "query_a_model_0"
    assert files.models[0].capabilities == frozenset({"plddt", "pae", "pde"})
    np.testing.assert_allclose(data.token_plddt, [0.9, 0.4])
    assert data.token_plddt_source == "provider_atom_mean"
    np.testing.assert_allclose(data.pae, [[0.0, 1.0], [2.0, 0.0]])
    np.testing.assert_allclose(data.pde, [[0.0, 0.5], [0.75, 0.0]])
    assert data.confidence.ranking_score == 0.9
    assert data.confidence.complex_plddt == 88.5
    assert data.confidence.has_clash is False
    np.testing.assert_allclose(data.confidence.chain_ptm, [0.9, 0.6])
    np.testing.assert_allclose(
        data.confidence.pair_chain_iptm,
        [[0.9, 0.75], [0.75, 0.6]],
    )
    np.testing.assert_allclose(
        data.confidence.pair_bespoke_iptm,
        [[np.nan, 0.65], [0.65, np.nan]],
        equal_nan=True,
    )


def test_openfold_missing_full_confidence_uses_structure_plddt(
    tmp_path: Path,
) -> None:
    query = tmp_path / "query_a"
    _write_sample(query, seed=1, sample=1, score=0.5, full=False)

    files = scan_prediction_path(query)
    data = load_prediction_data(files, load_pae=True, load_pde=True)

    assert files.models[0].capabilities == frozenset({"plddt"})
    assert data.pae is None
    assert data.pde is None
    assert data.token_plddt_source == "structure_b_factor"
    np.testing.assert_allclose(data.token_plddt, [0.85, 0.4])


def test_openfold_full_confidence_file_is_lazy(tmp_path: Path) -> None:
    query = tmp_path / "query_a"
    _write_sample(query, seed=1, sample=1, score=0.5)
    next(query.glob("seed_*/*_confidences.json")).write_text("{not-json")

    files = scan_prediction_path(query)
    summaries = load_prediction_confidence_summaries(files)
    data = load_prediction_data(
        files,
        load_pae=False,
        load_pde=False,
        load_token_plddt=False,
    )

    assert summaries[0].ranking_score == 0.5
    assert data.confidence.ranking_score == 0.5
    assert data.token_plddt is None


def test_openfold_cif_structure_is_supported(tmp_path: Path) -> None:
    query = tmp_path / "query_a"
    structure = _write_sample(query, seed=1, sample=1, score=0.5, full=False)
    cif_path = structure.with_suffix(".cif")
    cif_path.write_text(CIF_TEXT)
    structure.unlink()

    files = scan_prediction_path(query)
    data = load_prediction_data(files)

    assert files.models[0].structure_path == cif_path
    np.testing.assert_allclose(data.token_plddt, [0.85, 0.4])


def test_openfold_direct_seed_selection_is_supported(tmp_path: Path) -> None:
    query = tmp_path / "query_a"
    _write_sample(query, seed=42, sample=1, score=0.5)

    discovery = discover_prediction_candidates(query / "seed_42")
    files = scan_prediction_candidate(discovery, discovery.candidates[0])

    assert [
        (item.provider.key, item.relative_path) for item in discovery.candidates
    ] == [("openfold3", ".")]
    assert files.name == "query_a"
    assert files.n_models == 1


def test_openfold_nested_multi_query_root_lists_queries_not_seeds(
    tmp_path: Path,
) -> None:
    wrapper = tmp_path / "wrapper"
    output = wrapper / "run" / "output"
    _write_sample(output / "query_b", seed=2, sample=1, score=0.6)
    _write_sample(output / "query_a", seed=1, sample=1, score=0.8)

    discovery = discover_prediction_candidates(wrapper)

    assert [item.relative_path for item in discovery.candidates] == [
        "run/output/query_a",
        "run/output/query_b",
    ]
    assert [item.provider.key for item in discovery.candidates] == [
        "openfold3",
        "openfold3",
    ]


def test_openfold_multi_query_archive_lists_queries(tmp_path: Path) -> None:
    output = tmp_path / "source" / "output"
    _write_sample(output / "query_b", seed=2, sample=1, score=0.6)
    _write_sample(output / "query_a", seed=1, sample=1, score=0.8)
    archive = tmp_path / "openfold.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(output.parent.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output.parent.parent))

    discovery = discover_prediction_candidates(archive)
    files = scan_prediction_candidate(discovery, discovery.candidates[0])

    assert [item.relative_path for item in discovery.candidates] == [
        "source/output/query_a",
        "source/output/query_b",
    ]
    assert files.input_path == archive
    assert files.name == "query_a"


def test_openfold_incomplete_output_is_not_detected(tmp_path: Path) -> None:
    query = tmp_path / "query_a"
    seed_dir = query / "seed_1"
    seed_dir.mkdir(parents=True)
    (seed_dir / "query_a_seed_1_sample_1_model.pdb").write_text(PDB_TEXT)

    with pytest.raises(ValueError, match="Could not recognize"):
        discover_prediction_candidates(query)


def test_openfold_rejects_ambiguous_clash_and_unknown_chains(
    tmp_path: Path,
) -> None:
    ambiguous = tmp_path / "ambiguous"
    _write_sample(ambiguous, seed=1, sample=1, score=0.5, clash=0.5)
    with pytest.raises(ProviderContractError, match="has_clash must be boolean"):
        load_prediction_data(scan_prediction_path(ambiguous))

    unknown = tmp_path / "unknown"
    _write_sample(
        unknown,
        seed=1,
        sample=1,
        score=0.5,
        chain_ptm={"A": 0.9, "X": 0.6},
    )
    with pytest.raises(ProviderContractError, match="unknown chain 'X'"):
        load_prediction_data(scan_prediction_path(unknown))


def test_openfold_rejects_malformed_atom_plddt(tmp_path: Path) -> None:
    query = tmp_path / "query_a"
    _write_sample(query, seed=1, sample=1, score=0.5)
    full_path = next(query.glob("seed_*/*_confidences.json"))
    payload = json.loads(full_path.read_text())
    payload["plddt"] = [80.0]
    _write_json(full_path, payload)

    with pytest.raises(ValueError, match="does not match 3 atoms"):
        load_prediction_data(scan_prediction_path(query))


def test_real_openfold_example_scans_and_loads() -> None:
    root = Path(__file__).resolve().parents[1] / "examples" / "openfold_results"
    discovery = discover_prediction_candidates(root)
    files = scan_prediction_candidate(discovery, discovery.candidates[0])
    data = load_prediction_data(files, rank=0, load_pae=True, load_pde=True)

    assert [
        (item.provider.key, item.relative_path) for item in discovery.candidates
    ] == [("openfold3", "protein-ligand-ion")]
    assert files.n_models == 5
    assert files.models[0].metadata["sample"] == 3
    assert data.token_plddt.shape == (190,)
    assert data.pae.shape == (190, 190)
    assert data.pde.shape == (190, 190)
    assert data.confidence.pair_bespoke_iptm.shape == (3, 3)
