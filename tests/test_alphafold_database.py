from __future__ import annotations

import gzip
import io
import json
from dataclasses import replace
from pathlib import Path
from unittest import mock
from urllib.error import URLError

import numpy as np
import pytest
from FoldQC import reports
from FoldQC.alphafold_database import (
    AlphaFoldDatabaseGateway,
    AlphaFoldDbEntry,
    normalize_uniprot_qualifier,
    parse_alphafold_db_entry,
)
from FoldQC.loader import load_prediction_data, scan_prediction_path
from FoldQC.providers.alphafold_database import (
    MARKER_NAME,
    MARKER_SCHEMA_VERSION,
    MODEL_NAME,
    PAE_NAME,
)
from FoldQC.structure_index import StructureIndex

CIF_TEXT = """data_test
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.auth_comp_id
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.B_iso_or_equiv
ATOM CA ALA 1 A 80.0
ATOM CA GLY 2 A 90.0
#
"""


def _payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "modelEntityId": "AF-Q5VSL9-F1",
        "uniprotAccession": "Q5VSL9",
        "uniprotDescription": "Test protein",
        "latestVersion": 6,
        "sequenceStart": 1,
        "sequenceEnd": 2,
        "isComplex": False,
        "globalMetricValue": 87.25,
        "cifUrl": "https://alphafold.ebi.ac.uk/files/model.cif",
        "paeDocUrl": "https://alphafold.ebi.ac.uk/files/pae.json",
    }
    payload.update(changes)
    return payload


def _entry(**changes: object) -> AlphaFoldDbEntry:
    entry = AlphaFoldDbEntry(
        model_id="AF-Q5VSL9-F1",
        accessions=("Q5VSL9",),
        composition=(),
        description="Test protein",
        version=6,
        sequence_start=1,
        sequence_end=2,
        is_complex=False,
        assembly_type=None,
        oligomeric_state=None,
        mean_plddt=87.25,
        iptm=None,
        ipsae=None,
        pdockq2=None,
        lis=None,
        cif_url="https://alphafold.ebi.ac.uk/files/model.cif",
        pae_url="https://alphafold.ebi.ac.uk/files/pae.json",
    )
    return replace(entry, **changes)


def _complex_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "modelEntityId": "AF-COMPLEX-1",
        "complexPredictionAccuracy_ipTM": 0.85,
        "complexPredictionAccuracy_ipSAE": 0.81,
        "complexPredictionAccuracy_pDockQ2": 0.40,
        "complexPredictionAccuracy_LIS": 0.41,
    }
    payload.update(changes)
    return payload


@pytest.mark.parametrize(
    ("value", "normalized"),
    (
        ("P12345", "P12345"),
        ("Q5VSL9", "Q5VSL9"),
        ("A0A024RBG1", "A0A024RBG1"),
        (" a0a024rbg1-12 ", "A0A024RBG1-12"),
    ),
)
def test_uniprot_qualifier_validation(value: str, normalized: str) -> None:
    assert normalize_uniprot_qualifier(value) == normalized


@pytest.mark.parametrize(
    "value",
    (
        "Q5VSL9garbage",
        "Q5VSL9-0",
        "AF-Q5VSL9-F1",
        "Q5VSL９",
        "P1234",
        "A0A024RBGI",
    ),
)
def test_uniprot_qualifier_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="valid UniProt"):
        normalize_uniprot_qualifier(value)


def test_api_entry_parser_normalizes_only_selection_fields() -> None:
    payload = _payload(
        uniprotAccession=["q5vsl9", "P12345"],
        uniprotDescription=["Alpha", "Beta"],
        isComplex=True,
        assemblyType="heteromer",
        oligomericState="A2B",
        complexComposition=[
            {
                "identifierType": "uniprotAccession",
                "identifier": "q5vsl9",
                "stoichiometry": 2,
            },
            {
                "identifierType": "uniprotAccession",
                "identifier": "P12345",
                "stoichiometry": 1,
            },
        ],
        sequenceStart=None,
        sequenceEnd=None,
        ipTM=0.91,
        ipSAE=0.88,
    )
    entry = parse_alphafold_db_entry(payload, index=0)

    assert entry.accessions == ("Q5VSL9", "P12345")
    assert entry.composition == (("Q5VSL9", 2), ("P12345", 1))
    assert "2×Q5VSL9 + P12345" in entry.display_label
    assert entry.description == "Alpha, Beta"
    assert entry.is_complex
    assert entry.oligomeric_state == "A2B"
    assert not hasattr(entry, "ipTM")
    payload["uniprotAccession"] = "CHANGED"
    assert entry.accessions == ("Q5VSL9", "P12345")


def test_api_entry_parser_supports_homomer_composition_and_optional_pae() -> None:
    entry = parse_alphafold_db_entry(
        _payload(
            isComplex=True,
            assemblyType="Homo",
            oligomericState="tetramer",
            complexComposition=[
                {
                    "identifierType": "uniprotAccession",
                    "identifier": "Q5VSL9",
                    "stoichiometry": 4,
                }
            ],
            paeDocUrl=None,
        ),
        index=0,
    )
    assert entry.composition_label == "4×Q5VSL9"
    assert entry.pae_url is None
    assert "Homo, tetramer" in entry.selection_description


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"modelEntityId": None}, "modelEntityId"),
        ({"cifUrl": None}, "cifUrl"),
        ({"latestVersion": "6"}, "latestVersion"),
        ({"globalMetricValue": "high"}, "globalMetricValue"),
        ({"globalMetricValue": 101.0}, "within"),
        ({"isComplex": "false"}, "isComplex"),
        ({"sequenceStart": 0}, "sequenceStart"),
        ({"sequenceStart": 10, "sequenceEnd": 2}, "precedes"),
        ({"complexComposition": {}}, "complexComposition"),
    ),
)
def test_api_entry_parser_rejects_malformed_required_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_alphafold_db_entry(_payload(**changes), index=1)


class _Response(io.BytesIO):
    def __init__(self, data: bytes, *, final_url: str | None = None) -> None:
        super().__init__(data)
        self._final_url = final_url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def geturl(self) -> str:
        return self._final_url or "https://alphafold.ebi.ac.uk/files/response"


def test_gateway_lookup_requests_complexes_and_preserves_order() -> None:
    requested: list[tuple[str, int]] = []
    payload = [_payload(), _payload(modelEntityId="AF-COMPLEX-1", isComplex=True)]

    def opener(request, *, timeout: int):
        requested.append((request.full_url, timeout))
        response = [_complex_payload()] if "/complex/" in request.full_url else payload
        return _Response(json.dumps(response).encode())

    entries = AlphaFoldDatabaseGateway(opener=opener).lookup("q5vsl9")

    assert [entry.model_id for entry in entries] == [
        "AF-Q5VSL9-F1",
        "AF-COMPLEX-1",
    ]
    assert requested[0][0].endswith("/prediction/Q5VSL9?include_complexes=true")
    assert requested[1][0].endswith("/complex/Q5VSL9")
    assert requested[0][1] > 0
    assert entries[1].iptm == 0.85
    assert entries[1].ipsae == 0.81
    assert entries[1].pdockq2 == 0.40
    assert entries[1].lis == 0.41


def test_gateway_skips_complex_metadata_when_complexes_are_excluded() -> None:
    requested: list[str] = []

    def opener(request, *, timeout: int):
        requested.append(request.full_url)
        return _Response(json.dumps([_payload()]).encode())

    entries = AlphaFoldDatabaseGateway(opener=opener).lookup(
        "Q5VSL9", include_complexes=False
    )

    assert len(entries) == 1
    assert requested[0].endswith("/prediction/Q5VSL9?include_complexes=false")
    assert len(requested) == 1


def test_gateway_rejects_malformed_complex_confidence() -> None:
    prediction = [_payload(modelEntityId="AF-COMPLEX-1", isComplex=True)]

    def opener(request, *, timeout: int):
        response = (
            [_complex_payload(complexPredictionAccuracy_ipSAE=1.2)]
            if "/complex/" in request.full_url
            else prediction
        )
        return _Response(json.dumps(response).encode())

    with pytest.raises(ValueError, match="ipSAE.*within 0–1"):
        AlphaFoldDatabaseGateway(opener=opener).lookup("Q5VSL9")


def test_gateway_reports_network_and_json_failures() -> None:
    def failed(_request, *, timeout: int):
        raise URLError("offline")

    with pytest.raises(ValueError, match="lookup.*offline"):
        AlphaFoldDatabaseGateway(opener=failed).lookup("Q5VSL9")

    def malformed(_request, *, timeout: int):
        return _Response(b"not json")

    with pytest.raises(ValueError, match="invalid JSON"):
        AlphaFoldDatabaseGateway(opener=malformed).lookup("Q5VSL9")


def test_gateway_materializes_owned_prediction_and_cleans_it_up() -> None:
    responses = {
        "https://alphafold.ebi.ac.uk/files/model.cif": CIF_TEXT.encode(),
        "https://alphafold.ebi.ac.uk/files/pae.json": json.dumps(
            [{"predicted_aligned_error": [[0.0, 1.0], [1.0, 0.0]]}]
        ).encode(),
    }

    def opener(request, *, timeout: int):
        return _Response(responses[request.full_url])

    files = AlphaFoldDatabaseGateway(opener=opener).materialize(
        _entry(iptm=0.85, ipsae=0.81, pdockq2=0.40, lis=0.41)
    )
    root = files.pred_dir
    assert files.provider.key == "alphafold_db"
    assert files.n_models == 1
    assert (root / MARKER_NAME).is_file()
    marker = json.loads((root / MARKER_NAME).read_text())
    assert marker["confidence"] == {
        "iptm": 0.85,
        "ipsae": 0.81,
        "pdockq2": 0.40,
        "lis": 0.41,
    }
    assert not any(root.glob("*.part"))

    files.close()
    files.close()
    assert not root.exists()


def test_gateway_decompresses_gzip_structure_and_pae_downloads() -> None:
    pae_text = json.dumps([{"predicted_aligned_error": [[0.0, 1.0], [1.0, 0.0]]}])
    responses = {
        "https://alphafold.ebi.ac.uk/files/model.cif": gzip.compress(CIF_TEXT.encode()),
        "https://alphafold.ebi.ac.uk/files/pae.json": gzip.compress(pae_text.encode()),
    }

    def opener(request, *, timeout: int):
        return _Response(responses[request.full_url])

    files = AlphaFoldDatabaseGateway(opener=opener).materialize(_entry())
    root = files.pred_dir
    try:
        assert (root / MODEL_NAME).read_text() == CIF_TEXT
        assert (root / PAE_NAME).read_text() == pae_text
        assert not any(root.glob("*.part"))
    finally:
        files.close()


def test_gateway_rejects_untrusted_download_and_cleans_partial_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "materialized"
    with mock.patch(
        "FoldQC.alphafold_database.tempfile.mkdtemp", return_value=str(root)
    ):
        with pytest.raises(ValueError, match="not trusted"):
            AlphaFoldDatabaseGateway().materialize(
                _entry(cif_url="https://example.org/model.cif", pae_url=None)
            )
    assert not root.exists()


def test_gateway_rejects_download_redirect_to_untrusted_host(tmp_path: Path) -> None:
    root = tmp_path / "redirected"
    root.mkdir()

    def opener(_request, *, timeout: int):
        return _Response(CIF_TEXT.encode(), final_url="https://example.org/model.cif")

    with mock.patch(
        "FoldQC.alphafold_database.tempfile.mkdtemp", return_value=str(root)
    ):
        with pytest.raises(ValueError, match="not trusted"):
            AlphaFoldDatabaseGateway(opener=opener).materialize(_entry(pae_url=None))
    assert not root.exists()


def test_gateway_cleans_successful_cif_when_advertised_pae_download_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "partial"
    root.mkdir()

    def opener(request, *, timeout: int):
        if request.full_url.endswith("model.cif"):
            return _Response(CIF_TEXT.encode())
        raise URLError("PAE unavailable")

    with mock.patch(
        "FoldQC.alphafold_database.tempfile.mkdtemp", return_value=str(root)
    ):
        with pytest.raises(ValueError, match="PAE.*unavailable"):
            AlphaFoldDatabaseGateway(opener=opener).materialize(_entry())
    assert not root.exists()


def _write_materialized(
    root: Path,
    *,
    pae: object | None,
    confidence: dict[str, object] | None = None,
) -> None:
    root.mkdir()
    (root / MODEL_NAME).write_text(CIF_TEXT)
    (root / MARKER_NAME).write_text(
        json.dumps(
            {
                "schema_version": MARKER_SCHEMA_VERSION,
                "model_id": "AF-Q5VSL9-F1",
                "display_label": "Monomer Q5VSL9 (AF-Q5VSL9-F1)",
                "confidence": confidence or {},
            }
        )
    )
    if pae is not None:
        (root / PAE_NAME).write_text(json.dumps(pae))


def test_provider_uses_b_factors_and_loads_downloaded_pae_lazily(
    tmp_path: Path,
) -> None:
    root = tmp_path / "afdb"
    _write_materialized(
        root,
        pae=[{"predicted_aligned_error": [[0.0, 2.0], [3.0, 0.0]]}],
    )
    files = scan_prediction_path(root)
    index = StructureIndex.from_path(files.structure_path(0))

    initial = load_prediction_data(
        files, rank=0, load_pae=False, load_pde=False, structure_index=index
    )
    assert files.provider.key == "alphafold_db"
    assert not files.supports_ensemble
    assert files.model(0).capabilities == frozenset({"plddt", "pae"})
    assert initial.pae is None
    assert initial.token_plddt_source == "structure_b_factor"
    np.testing.assert_allclose(initial.token_plddt, [0.8, 0.9])

    lazy = load_prediction_data(
        files, rank=0, load_pae=True, load_pde=False, structure_index=index
    )
    np.testing.assert_allclose(lazy.pae, [[0.0, 2.0], [3.0, 0.0]])
    assert lazy.pae is not None and not lazy.pae.flags.writeable


def test_provider_supports_plddt_only_when_pae_is_absent(tmp_path: Path) -> None:
    root = tmp_path / "afdb"
    _write_materialized(root, pae=None)
    files = scan_prediction_path(root)
    assert files.model(0).capabilities == frozenset({"plddt"})


def test_provider_loads_and_summarizes_complex_interface_confidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "afdb"
    _write_materialized(
        root,
        pae=None,
        confidence={"iptm": 0.85, "ipsae": 0.81, "pdockq2": 0.40, "lis": 0.41},
    )
    files = scan_prediction_path(root)
    index = StructureIndex.from_path(files.structure_path(0))
    data = load_prediction_data(
        files, rank=0, load_pae=False, load_pde=False, structure_index=index
    )

    assert data.confidence is not None
    assert data.confidence.iptm == 0.85
    assert data.confidence.ipsae == 0.81
    assert data.confidence.pdockq2 == 0.40
    assert data.confidence.lis == 0.41
    summary = reports.format_confidence_summary(data, index.token_map)
    assert "ipTM             : 0.85" in summary
    assert "ipSAE            : 0.81" in summary
    assert "pDockQ2          : 0.40" in summary
    assert "LIS              : 0.41" in summary
    assert "pLDDT read from structure B-factors" in summary


@pytest.mark.parametrize(
    "pae",
    (
        {"predicted_aligned_error": [[0.0]]},
        [],
        [{"other": [[0.0, 1.0], [1.0, 0.0]]}],
        [{"predicted_aligned_error": [[0.0]]}],
    ),
)
def test_provider_rejects_malformed_or_wrong_shape_pae(
    tmp_path: Path, pae: object
) -> None:
    root = tmp_path / "afdb"
    _write_materialized(root, pae=pae)
    files = scan_prediction_path(root)
    index = StructureIndex.from_path(files.structure_path(0))
    with pytest.raises(ValueError, match="PAE|pae"):
        load_prediction_data(
            files, rank=0, load_pae=True, load_pde=False, structure_index=index
        )
