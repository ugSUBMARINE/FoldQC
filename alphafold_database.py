"""Qt-free AlphaFold Database lookup and temporary materialization."""

from __future__ import annotations

import gzip
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from .loader_models import PredictionFiles
from .ownership import TemporaryDirectoryOwner
from .providers.alphafold_database import (
    MARKER_NAME,
    MARKER_SCHEMA_VERSION,
    MODEL_NAME,
    PAE_NAME,
)

API_BASE_URL = "https://alphafold.ebi.ac.uk/api"
TRUSTED_DOWNLOAD_HOST = "alphafold.ebi.ac.uk"
HTTP_TIMEOUT_SECONDS = 60
USER_AGENT = "FoldQC/0.2 AlphaFold-DB-client"

UNIPROT_QUALIFIER_PATTERN = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
    r"(?:-[1-9][0-9]*)?",
    flags=re.IGNORECASE | re.ASCII,
)


def normalize_uniprot_qualifier(value: str) -> str:
    qualifier = str(value).strip().upper()
    if not UNIPROT_QUALIFIER_PATTERN.fullmatch(qualifier):
        raise ValueError(
            "Enter a valid UniProt accession or isoform ID, for example "
            "Q5VSL9 or Q5VSL9-4."
        )
    return qualifier


@dataclass(frozen=True)
class AlphaFoldDbEntry:
    model_id: str
    accessions: tuple[str, ...]
    composition: tuple[tuple[str, int], ...]
    description: str
    version: int
    sequence_start: int | None
    sequence_end: int | None
    is_complex: bool
    assembly_type: str | None
    oligomeric_state: str | None
    mean_plddt: float
    iptm: float | None
    ipsae: float | None
    pdockq2: float | None
    lis: float | None
    cif_url: str
    pae_url: str | None

    @property
    def primary_accession(self) -> str:
        return self.accessions[0] if self.accessions else ""

    @property
    def display_label(self) -> str:
        kind = "Complex" if self.is_complex else "Monomer"
        composition = self.composition_label
        return f"{kind} {composition} ({self.model_id})"

    @property
    def composition_label(self) -> str:
        if self.composition:
            return " + ".join(
                f"{stoichiometry}×{accession}" if stoichiometry != 1 else accession
                for accession, stoichiometry in self.composition
            )
        return "/".join(self.accessions) or "unknown accession"

    @property
    def selection_description(self) -> str:
        details: list[str] = []
        if self.is_complex:
            states = tuple(
                value for value in (self.assembly_type, self.oligomeric_state) if value
            )
            if states:
                details.append(", ".join(dict.fromkeys(states)))
        elif self.sequence_start is not None and self.sequence_end is not None:
            details.append(f"residues {self.sequence_start}–{self.sequence_end}")
        details.append(f"v{self.version}")
        details.append(f"API average pLDDT {self.mean_plddt:.2f}")
        if self.description:
            details.append(self.description)
        return "; ".join(details)


class AlphaFoldDatabasePort(Protocol):
    def lookup(
        self, qualifier: str, *, include_complexes: bool = True
    ) -> tuple[AlphaFoldDbEntry, ...]: ...

    def materialize(self, entry: AlphaFoldDbEntry) -> PredictionFiles: ...


class UrlResponse(Protocol):
    def __enter__(self) -> UrlResponse: ...

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None: ...

    def read(self, amount: int = -1) -> bytes: ...

    def geturl(self) -> str: ...


class UrlOpen(Protocol):
    def __call__(self, request: Request, *, timeout: int) -> UrlResponse: ...


@dataclass(frozen=True)
class _ComplexConfidence:
    model_id: str
    iptm: float | None
    ipsae: float | None
    pdockq2: float | None
    lis: float | None


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _accessions(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list):
        values = tuple(item for item in value if isinstance(item, str))
    else:
        return ()
    return tuple(item.strip().upper() for item in values if item.strip())


def _complex_composition(value: object, *, context: str) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context}: complexComposition must be a list or null.")
    composition: list[tuple[str, int]] = []
    for item_index, item in enumerate(value):
        item_context = f"{context}: complexComposition item {item_index + 1}"
        if not isinstance(item, dict):
            raise ValueError(f"{item_context} must be an object.")
        identifier_type = item.get("identifierType")
        identifier = _optional_text(item.get("identifier"))
        stoichiometry = item.get("stoichiometry")
        if identifier_type != "uniprotAccession" or identifier is None:
            raise ValueError(f"{item_context} must identify a UniProt accession.")
        if (
            isinstance(stoichiometry, bool)
            or not isinstance(stoichiometry, int)
            or stoichiometry < 1
        ):
            raise ValueError(f"{item_context} has invalid stoichiometry.")
        composition.append((identifier.upper(), stoichiometry))
    return tuple(composition)


def _optional_int(value: object, field: str, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: {field} must be an integer or null.")
    return value


def _optional_score(payload: dict, field: str, context: str) -> float | None:
    if field not in payload or payload[field] is None:
        return None
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: {field} must be numeric or null.")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{context}: {field} must be within 0–1.")
    return score


def _parse_complex_confidence(payload: object, *, index: int) -> _ComplexConfidence:
    context = f"AlphaFold DB complex result {index + 1}"
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object.")
    model_id = _optional_text(payload.get("modelEntityId"))
    if model_id is None:
        raise ValueError(f"{context} has no modelEntityId.")
    return _ComplexConfidence(
        model_id=model_id,
        iptm=_optional_score(payload, "complexPredictionAccuracy_ipTM", context),
        ipsae=_optional_score(payload, "complexPredictionAccuracy_ipSAE", context),
        pdockq2=_optional_score(payload, "complexPredictionAccuracy_pDockQ2", context),
        lis=_optional_score(payload, "complexPredictionAccuracy_LIS", context),
    )


def parse_alphafold_db_entry(payload: object, *, index: int) -> AlphaFoldDbEntry:
    context = f"AlphaFold DB result {index + 1}"
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object.")
    model_id = _optional_text(payload.get("modelEntityId"))
    cif_url = _optional_text(payload.get("cifUrl"))
    if model_id is None:
        raise ValueError(f"{context} has no modelEntityId.")
    if cif_url is None:
        raise ValueError(f"{context} has no cifUrl.")
    version = _optional_int(payload.get("latestVersion"), "latestVersion", context)
    if version is None or version < 1:
        raise ValueError(f"{context}: latestVersion must be a positive integer.")
    raw_plddt = payload.get("globalMetricValue")
    if isinstance(raw_plddt, bool) or not isinstance(raw_plddt, (int, float)):
        raise ValueError(f"{context}: globalMetricValue must be numeric.")
    mean_plddt = float(raw_plddt)
    if not math.isfinite(mean_plddt) or not 0.0 <= mean_plddt <= 100.0:
        raise ValueError(f"{context}: globalMetricValue must be within 0–100.")
    is_complex = payload.get("isComplex", False)
    if not isinstance(is_complex, bool):
        raise ValueError(f"{context}: isComplex must be boolean.")
    description_value = payload.get("uniprotDescription")
    if isinstance(description_value, list):
        description = ", ".join(
            item.strip() for item in description_value if isinstance(item, str)
        )
    else:
        description = _optional_text(description_value) or ""
    composition = _complex_composition(
        payload.get("complexComposition"), context=context
    )
    accessions = tuple(accession for accession, _count in composition) or _accessions(
        payload.get("uniprotAccession")
    )
    sequence_start = _optional_int(
        payload.get("sequenceStart"), "sequenceStart", context
    )
    sequence_end = _optional_int(payload.get("sequenceEnd"), "sequenceEnd", context)
    if sequence_start is not None and sequence_start < 1:
        raise ValueError(f"{context}: sequenceStart must be positive.")
    if sequence_end is not None and sequence_end < 1:
        raise ValueError(f"{context}: sequenceEnd must be positive.")
    if (
        sequence_start is not None
        and sequence_end is not None
        and sequence_end < sequence_start
    ):
        raise ValueError(f"{context}: sequenceEnd precedes sequenceStart.")
    return AlphaFoldDbEntry(
        model_id=model_id,
        accessions=accessions,
        composition=composition,
        description=description,
        version=version,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        is_complex=is_complex,
        assembly_type=_optional_text(payload.get("assemblyType")),
        oligomeric_state=_optional_text(payload.get("oligomericState")),
        mean_plddt=mean_plddt,
        iptm=None,
        ipsae=None,
        pdockq2=None,
        lis=None,
        cif_url=cif_url,
        pae_url=_optional_text(payload.get("paeDocUrl")),
    )


class AlphaFoldDatabaseGateway:
    def __init__(self, *, opener: UrlOpen = urlopen) -> None:
        self._opener = opener

    def lookup(
        self, qualifier: str, *, include_complexes: bool = True
    ) -> tuple[AlphaFoldDbEntry, ...]:
        normalized = normalize_uniprot_qualifier(qualifier)
        query = urlencode({"include_complexes": str(include_complexes).lower()})
        url = f"{API_BASE_URL}/prediction/{normalized}?{query}"
        payload = self._read_json(url, label=f"AlphaFold DB lookup for {normalized}")
        if not isinstance(payload, list):
            raise ValueError("AlphaFold DB prediction response must be a JSON list.")
        entries = tuple(
            parse_alphafold_db_entry(item, index=index)
            for index, item in enumerate(payload)
        )
        if not include_complexes or not any(entry.is_complex for entry in entries):
            return entries
        complex_payload = self._read_json(
            f"{API_BASE_URL}/complex/{normalized}",
            label=f"AlphaFold DB complex lookup for {normalized}",
        )
        if not isinstance(complex_payload, list):
            raise ValueError("AlphaFold DB complex response must be a JSON list.")
        confidence_by_model = {
            confidence.model_id: confidence
            for index, item in enumerate(complex_payload)
            for confidence in (_parse_complex_confidence(item, index=index),)
        }
        return tuple(
            replace(
                entry,
                iptm=confidence.iptm,
                ipsae=confidence.ipsae,
                pdockq2=confidence.pdockq2,
                lis=confidence.lis,
            )
            if entry.is_complex
            and (confidence := confidence_by_model.get(entry.model_id)) is not None
            else entry
            for entry in entries
        )

    def materialize(self, entry: AlphaFoldDbEntry) -> PredictionFiles:
        root = Path(tempfile.mkdtemp(prefix="foldqc_afdb_"))
        owner = TemporaryDirectoryOwner(root)
        try:
            self._download(entry.cif_url, root / MODEL_NAME, label="structure")
            if entry.pae_url is not None:
                self._download(entry.pae_url, root / PAE_NAME, label="PAE")
            marker = {
                "schema_version": MARKER_SCHEMA_VERSION,
                "model_id": entry.model_id,
                "display_label": entry.display_label,
                "confidence": {
                    key: value
                    for key, value in (
                        ("iptm", entry.iptm),
                        ("ipsae", entry.ipsae),
                        ("pdockq2", entry.pdockq2),
                        ("lis", entry.lis),
                    )
                    if value is not None
                },
            }
            marker_part = root / f"{MARKER_NAME}.part"
            with marker_part.open("w", encoding="utf-8") as handle:
                json.dump(marker, handle, separators=(",", ":"))
            marker_part.replace(root / MARKER_NAME)
            from .providers.registry import BUILTIN_PROVIDERS

            files = BUILTIN_PROVIDERS.get("alphafold_db").scan(root)
            files.adopt_resource_owner(owner)
            return files
        except Exception:
            owner.close()
            raise

    def _read_json(self, url: str, *, label: str) -> object:
        request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )
        try:
            with self._opener(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                raw = response.read()
            return json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            raise ValueError(f"{label} failed with HTTP {exc.code}.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ValueError(f"{label} failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} returned invalid JSON.") from exc

    def _download(self, url: str, target: Path, *, label: str) -> None:
        self._validate_download_url(url, label=label)
        request = Request(url, headers={"User-Agent": USER_AGENT})
        part = target.with_name(f"{target.name}.part")
        try:
            with self._opener(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                self._validate_download_url(response.geturl(), label=label)
                with part.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            self._finalize_download(part, target)
        except HTTPError as exc:
            raise ValueError(
                f"AlphaFold DB {label} download failed with HTTP {exc.code}."
            ) from exc
        except (URLError, TimeoutError, OSError, EOFError) as exc:
            raise ValueError(f"AlphaFold DB {label} download failed: {exc}") from exc

    @staticmethod
    def _finalize_download(part: Path, target: Path) -> None:
        with part.open("rb") as handle:
            is_gzip = handle.read(2) == b"\x1f\x8b"
        if not is_gzip:
            part.replace(target)
            return
        expanded = target.with_name(f"{target.name}.expanded.part")
        with gzip.open(part, "rb") as source, expanded.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        expanded.replace(target)
        part.unlink()

    @staticmethod
    def _validate_download_url(url: object, *, label: str) -> None:
        if not isinstance(url, str):
            raise ValueError(f"AlphaFold DB {label} URL is not trusted: {url!r}")
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != TRUSTED_DOWNLOAD_HOST
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(f"AlphaFold DB {label} URL is not trusted: {url}")
