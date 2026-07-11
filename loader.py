"""
Loader
======
Discover and read prediction outputs from Boltz-2, Boltz Lab, Boltz API,
AlphaFold 3, AlphaFold Server, Chai-1 Discovery, Protenix, an archived
prediction output, or a single predicted structure file.

The public API remains intentionally small:

``scan_prediction_path(path)``
    Accepts an output directory, an archive containing an output directory, or
    a single CIF/PDB file and returns a provider-aware
    :class:`PredictionFiles`.

``scan_prediction_dir(path)``
    Compatibility wrapper for directory inputs.

``load_prediction_data(pred_files, rank, ...)``
    Loads one ranked model into the provider-neutral :class:`PredictionData`.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import stat
import tarfile
import tempfile
import weakref
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from .token_map import extract_structure_plddt, parse_structure_atoms

STRUCTURE_SUFFIXES = {".cif", ".pdb"}
ARCHIVE_EXTENSIONS = (".tar.gz", ".zip", ".tgz", ".tar")
PROVIDER_LABELS = {
    "boltz": "Boltz",
    "boltz_lab": "Boltz Lab",
    "boltz_api": "Boltz API",
    "alphafold3": "AlphaFold 3",
    "af3_server": "AlphaFold 3 Server",
    "chai1": "Chai-1 Discovery",
    "protenix": "Protenix",
    "structure_only": "Structure-only",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelFiles:
    """Paths and display metadata for one ranked model."""

    rank: int
    structure_path: Path
    display_label: str
    object_name: str
    confidence_path: Path | None = None
    summary_path: Path | None = None
    plddt_path: Path | None = None
    pae_path: Path | None = None
    pde_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionFiles:
    """Provider-aware paths found under one prediction source."""

    name: str
    pred_dir: Path
    provider: str = "boltz"
    input_path: Path | None = None
    models: list[ModelFiles] = field(default_factory=list)

    affinity_file: Path | None = None
    embeddings_file: Path | None = None
    capabilities: set[str] = field(default_factory=set)
    _temporary_directory: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def provider_label(self) -> str:
        return _provider_label(self.provider)

    @property
    def supports_ensemble(self) -> bool:
        return self.provider != "structure_only" and self.n_models >= 2

    @property
    def structure_files(self) -> list[tuple[int, Path]]:
        return [(model.rank, model.structure_path) for model in self.models]

    @structure_files.setter
    def structure_files(self, value: list[tuple[int, Path]]) -> None:
        self.models = [
            ModelFiles(
                rank=rank,
                structure_path=Path(path),
                display_label=f"model_{rank}",
                object_name=f"{self.name}_model_{rank}",
            )
            for rank, path in value
        ]

    @property
    def confidence_files(self) -> list[tuple[int, Path]]:
        return [
            (model.rank, model.confidence_path)
            for model in self.models
            if model.confidence_path is not None
        ]

    @confidence_files.setter
    def confidence_files(self, value: list[tuple[int, Path]]) -> None:
        by_rank = {rank: Path(path) for rank, path in value}
        for model in self.models:
            model.confidence_path = by_rank.get(model.rank)

    @property
    def plddt_files(self) -> list[tuple[int, Path]]:
        return [
            (model.rank, model.plddt_path)
            for model in self.models
            if model.plddt_path is not None
        ]

    @plddt_files.setter
    def plddt_files(self, value: list[tuple[int, Path]]) -> None:
        by_rank = {rank: Path(path) for rank, path in value}
        for model in self.models:
            model.plddt_path = by_rank.get(model.rank)

    @property
    def pae_files(self) -> list[tuple[int, Path]]:
        return [
            (model.rank, model.pae_path)
            for model in self.models
            if model.pae_path is not None
        ]

    @pae_files.setter
    def pae_files(self, value: list[tuple[int, Path]]) -> None:
        by_rank = {rank: Path(path) for rank, path in value}
        for model in self.models:
            model.pae_path = by_rank.get(model.rank)

    @property
    def pde_files(self) -> list[tuple[int, Path]]:
        return [
            (model.rank, model.pde_path)
            for model in self.models
            if model.pde_path is not None
        ]

    @pde_files.setter
    def pde_files(self, value: list[tuple[int, Path]]) -> None:
        by_rank = {rank: Path(path) for rank, path in value}
        for model in self.models:
            model.pde_path = by_rank.get(model.rank)

    @property
    def has_pae(self) -> bool:
        return "pae" in self.capabilities

    @property
    def has_pde(self) -> bool:
        return "pde" in self.capabilities

    @property
    def has_contact_probs(self) -> bool:
        return "contact_probs" in self.capabilities

    @property
    def has_plddt(self) -> bool:
        return "plddt" in self.capabilities

    @property
    def has_structure_plddt(self) -> bool:
        return "structure_plddt" in self.capabilities

    @property
    def has_affinity(self) -> bool:
        return self.affinity_file is not None

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings_file is not None

    def model(self, rank: int) -> ModelFiles:
        models = {model.rank: model for model in self.models}
        if rank not in models:
            raise KeyError(rank)
        return models[rank]

    def structure_path(self, rank: int) -> Path:
        return self.model(rank).structure_path

    def confidence_path(self, rank: int) -> Path | None:
        return self.model(rank).confidence_path

    def summary_path(self, rank: int) -> Path | None:
        return self.model(rank).summary_path

    def plddt_path(self, rank: int) -> Path | None:
        return self.model(rank).plddt_path

    def pae_path(self, rank: int) -> Path | None:
        return self.model(rank).pae_path

    def pde_path(self, rank: int) -> Path | None:
        return self.model(rank).pde_path


@dataclass
class PredictionData:
    """Arrays and metadata loaded for one ranked model."""

    name: str
    rank: int
    structure_path: Path
    provider: str = "boltz"
    display_label: str = ""

    structure_plddt: np.ndarray | None = None
    plddt: np.ndarray | None = None
    pae: np.ndarray | None = None
    pde: np.ndarray | None = None
    contact_probs: np.ndarray | None = None

    confidence: dict | None = None
    summary_confidence: dict | None = None
    affinity: dict | None = None

    embeddings_s: np.ndarray | None = None
    embeddings_z: np.ndarray | None = None


@dataclass(frozen=True)
class PredictionCandidate:
    """One loadable prediction source found below a selected path."""

    path: Path
    provider: str
    provider_label: str
    relative_path: str


@dataclass
class PredictionDiscovery:
    """Candidate list plus any temporary resources needed to load them."""

    input_path: Path
    candidates: tuple[PredictionCandidate, ...]
    _temporary_directory: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def scan(self, candidate: PredictionCandidate) -> PredictionFiles:
        """Load a selected candidate and transfer any extraction lifetime handle."""
        if candidate not in self.candidates:
            raise ValueError(f"Unknown prediction candidate: {candidate.path}")
        files = _scan_prediction_path_exact(candidate.path, input_path=self.input_path)
        if self._temporary_directory is not None:
            files._temporary_directory = self._temporary_directory
            self._temporary_directory = None
        return files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_prediction_path(path: str | Path) -> PredictionFiles:
    """Detect and scan a prediction output directory, archive, or structure."""
    discovery = discover_prediction_candidates(path)
    return discovery.scan(discovery.candidates[0])


def scan_prediction_dir(path: str | Path) -> PredictionFiles:
    """Discover prediction outputs in or below a selected directory."""
    selected_dir = Path(path).expanduser().resolve()
    if not selected_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {selected_dir}")
    discovery = discover_prediction_candidates(selected_dir)
    return discovery.scan(discovery.candidates[0])


def discover_prediction_candidates(path: str | Path) -> PredictionDiscovery:
    """Find loadable prediction candidates below a folder, archive, or structure."""
    source = Path(path).expanduser().resolve()
    if source.is_file():
        if _archive_kind(source) is not None:
            return _discover_archive_file(source)
        if source.suffix.lower() in STRUCTURE_SUFFIXES:
            return PredictionDiscovery(
                input_path=source,
                candidates=(
                    PredictionCandidate(
                        path=source,
                        provider="structure_only",
                        provider_label=_provider_label("structure_only"),
                        relative_path=source.name,
                    ),
                ),
            )
        raise ValueError(f"Unsupported file format: {source.name}")
    if source.is_dir():
        candidates = _prediction_candidates_in_tree(source)
        if not candidates:
            raise ValueError(
                f"Could not recognize prediction output format in {source}.\n"
                "Expected the folder to contain a Boltz prediction folder, "
                "a Boltz Lab output folder, a Boltz API output folder, "
                "an AlphaFold 3 output folder, an AlphaFold 3 Server output folder, "
                "a Chai-1 Discovery output folder, or a Protenix output folder; "
                "or select a single .cif/.pdb structure file."
            )
        return PredictionDiscovery(input_path=source, candidates=tuple(candidates))
    raise FileNotFoundError(f"Path does not exist: {source}")


def load_prediction_data(
    pred_files: PredictionFiles,
    rank: int = 0,
    load_pae: bool = True,
    load_pde: bool = True,
    load_embeddings: bool = False,
    load_structure_plddt: bool = True,
    load_contact_probs: bool = False,
    load_plddt: bool = True,
) -> PredictionData:
    """Load arrays and JSON metadata for one ranked model."""
    try:
        model = pred_files.model(rank)
    except KeyError as exc:
        raise ValueError(
            f"Rank {rank} not found. Available ranks: "
            + str(sorted(model.rank for model in pred_files.models))
        ) from exc

    data = PredictionData(
        name=pred_files.name,
        rank=rank,
        structure_path=model.structure_path,
        provider=pred_files.provider,
        display_label=model.display_label,
    )

    if load_structure_plddt:
        data.structure_plddt = extract_structure_plddt(model.structure_path)

    if model.summary_path is not None:
        data.summary_confidence = _load_json(model.summary_path)
        data.confidence = _normalise_confidence(data.summary_confidence)

    if pred_files.provider in {"boltz", "boltz_lab", "boltz_api"}:
        _load_boltz_model_data(
            pred_files,
            model,
            data,
            load_pae=load_pae,
            load_pde=load_pde,
            load_embeddings=load_embeddings,
            load_plddt=load_plddt,
        )
    elif pred_files.provider in {"alphafold3", "af3_server"}:
        _load_af3_model_data(
            model,
            data,
            load_pae=load_pae,
            load_contact_probs=load_contact_probs,
            load_plddt=load_plddt,
        )
    elif pred_files.provider == "chai1":
        _load_chai_model_data(
            model,
            data,
            load_pae=load_pae,
            load_pde=load_pde,
        )
    elif pred_files.provider == "protenix":
        _load_protenix_model_data(
            model,
            data,
            load_pae=load_pae,
            load_pde=load_pde,
            load_contact_probs=load_contact_probs,
            load_plddt=load_plddt,
        )
    elif pred_files.provider == "structure_only":
        # Nothing else to load.
        pass
    else:
        raise ValueError(f"Unsupported provider: {pred_files.provider}")

    if data.confidence is None and data.summary_confidence is not None:
        data.confidence = _normalise_confidence(data.summary_confidence)

    return data


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------


@dataclass
class _AF3Sample:
    """One AF3 seed/sample directory's discovered paths and metadata."""

    model_path: Path
    confidence_path: Path | None
    summary_path: Path | None
    summary: dict
    ranking_score: float | None
    seed: int | None
    sample: int | None


@dataclass
class _BoltzSampleCandidate:
    """One Boltz Lab / API prediction sample's discovered paths."""

    sample_index: int
    structure_path: Path
    pae_path: Path | None
    confidence: dict | None = None
    structure_confidence: float | None = None


@dataclass
class _ChaiCandidate:
    """One Chai-1 Discovery prediction candidate's discovered paths."""

    structure_path: Path
    model_idx: int | None
    explicit_rank: int | None
    confidence_path: Path | None
    pae_path: Path | None
    pde_path: Path | None
    aggregate_score: float | None


@dataclass
class _ProtenixCandidate:
    """One Protenix prediction sample's discovered paths."""

    structure_path: Path
    summary_path: Path
    full_data_path: Path | None
    source_prefix: str
    sample_rank: int
    seed: int | None
    ranking_score: float | None


def _prediction_dir_provider(pred_dir: Path) -> str | None:
    if _looks_like_af3_server(pred_dir):
        return "af3_server"
    if _looks_like_af3(pred_dir):
        return "alphafold3"
    if _looks_like_chai(pred_dir):
        return "chai1"
    if _looks_like_protenix(pred_dir):
        return "protenix"
    if _looks_like_boltz_api(pred_dir):
        return "boltz_api"
    if _looks_like_boltz_lab(pred_dir):
        return "boltz_lab"
    if _looks_like_boltz(pred_dir):
        return "boltz"
    return None


def _provider_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider)


class _TemporaryExtraction:
    """Keep extracted archive contents alive while PredictionFiles is referenced."""

    def __init__(self, root: Path) -> None:
        self.root = root
        weakref.finalize(self, shutil.rmtree, root, ignore_errors=True)


def _looks_like_boltz(pred_dir: Path) -> bool:
    name = pred_dir.name
    return bool(
        list(pred_dir.glob(f"{name}_model_*.cif"))
        or list(pred_dir.glob(f"{name}_model_*.pdb"))
    )


def _looks_like_boltz_lab(pred_dir: Path) -> bool:
    metrics_path = pred_dir / "metrics.json"
    return metrics_path.exists() and bool(_boltz_sample_structure_paths(pred_dir))


def _looks_like_boltz_api(pred_dir: Path) -> bool:
    prediction_dir = _boltz_api_prediction_dir(pred_dir)
    if prediction_dir is None:
        return False
    metrics = _load_optional_json(prediction_dir / "metrics.json")
    if _is_boltz_api_metrics(metrics):
        return True
    run = _load_optional_json(pred_dir / "run.json")
    return _is_boltz_api_output(run.get("output") if isinstance(run, dict) else None)


def _looks_like_af3(pred_dir: Path) -> bool:
    if _af3_model_path(pred_dir) is not None and (
        _has_af3_metadata(pred_dir) or _af3_ranking_scores_path(pred_dir) is not None
    ):
        return True

    has_root_ranking_scores = _af3_ranking_scores_path(pred_dir) is not None
    for sample_dir in pred_dir.glob("seed-*_sample-*"):
        if not sample_dir.is_dir():
            continue
        if _af3_model_path(sample_dir) is not None and (
            _has_af3_metadata(sample_dir) or has_root_ranking_scores
        ):
            return True
    return False


def _has_af3_metadata(pred_dir: Path) -> bool:
    return (
        _first_confidence_json(pred_dir) is not None
        or _af3_summary_path(pred_dir) is not None
    )


def _looks_like_af3_server(pred_dir: Path) -> bool:
    return bool(
        list(pred_dir.glob("*_full_data_*.json"))
        or list(pred_dir.glob("full_data_*.json"))
    )


def _looks_like_chai(pred_dir: Path) -> bool:
    structures = [
        path
        for suffix in STRUCTURE_SUFFIXES
        for path in pred_dir.glob(f"pred*{suffix}")
        if _parse_chai_stem(path.stem) is not None
    ]
    if not structures:
        return False
    return bool(
        list(pred_dir.glob("scores*.json"))
        or list(pred_dir.glob("scores*.npz"))
        or list(pred_dir.glob("pae*.npy"))
    )


def _looks_like_protenix(pred_dir: Path) -> bool:
    for predictions_dir in _protenix_prediction_dirs(pred_dir):
        for structure_path in _protenix_structure_paths(predictions_dir):
            parsed = _parse_protenix_stem(structure_path.stem)
            if parsed is None:
                continue
            prefix, sample_rank = parsed
            summary_path = (
                predictions_dir
                / f"{prefix}_summary_confidence_sample_{sample_rank}.json"
            )
            if summary_path.exists():
                return True
    return False


def _scan_zip_file(path: Path) -> PredictionFiles:
    discovery = _discover_archive_file(path)
    return discovery.scan(discovery.candidates[0])


def _archive_kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar", ".tar.gz", ".tgz")):
        return "tar"
    return None


def _archive_base_name(path: Path) -> str:
    name = path.name
    for suffix in ARCHIVE_EXTENSIONS:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _discover_archive_file(path: Path) -> PredictionDiscovery:
    kind = _archive_kind(path)
    if kind == "zip" and not zipfile.is_zipfile(path):
        raise ValueError(f"Invalid archive: {path}")
    if kind == "tar" and not tarfile.is_tarfile(path):
        raise ValueError(f"Invalid archive: {path}")
    if kind is None:
        raise ValueError(f"Unsupported archive format: {path}")

    archive_name = _safe_object_name(_archive_base_name(path))
    temp_root = Path(tempfile.mkdtemp(prefix=f"foldqc_{archive_name}_"))
    extract_root = temp_root / archive_name
    try:
        if kind == "zip":
            _extract_zip_safely(path, extract_root)
        else:
            _extract_tar_safely(path, extract_root)
        candidates = _prediction_candidates_in_tree(extract_root)
        if not candidates:
            raise ValueError(
                "Could not recognize prediction output format inside archive.\n"
                "Expected the archive to contain a Boltz, Boltz Lab, Boltz API, "
                "AlphaFold 3, AlphaFold 3 Server, Chai-1 Discovery, or Protenix "
                "output folder."
            )
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    return PredictionDiscovery(
        input_path=path,
        candidates=tuple(candidates),
        _temporary_directory=_TemporaryExtraction(temp_root),
    )


def _discover_zip_file(path: Path) -> PredictionDiscovery:
    return _discover_archive_file(path)


def _extract_zip_safely(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    destination_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = _safe_zip_member_path(member)
            if member_path is None:
                continue
            target = (destination / member_path).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(
                    f"Unsafe path in zip archive: {member.filename}"
                ) from exc

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(
                    f"Refusing to extract symlink from zip archive: {member.filename}"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _safe_zip_member_path(member: zipfile.ZipInfo) -> Path | None:
    raw_name = member.filename.replace("\\", "/")
    path = PurePosixPath(raw_name)
    windows_path = PureWindowsPath(raw_name)
    if not path.parts:
        return None
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part == ".." for part in path.parts)
    ):
        raise ValueError(f"Unsafe path in zip archive: {member.filename}")
    parts = [part for part in path.parts if part not in {"", "."}]
    if not parts:
        return None
    return Path(*parts)


def _extract_tar_safely(tar_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    destination_root = destination.resolve()
    with tarfile.open(tar_path, mode="r:*") as tf:
        for member in tf.getmembers():
            member_path = _safe_tar_member_path(member)
            if member_path is None:
                continue
            target = (destination / member_path).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe path in tar archive: {member.name}") from exc

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(
                    f"Unsupported member type in tar archive: {member.name}"
                )

            source = tf.extractfile(member)
            if source is None:
                raise ValueError(f"Could not extract tar member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as dst:
                shutil.copyfileobj(source, dst)


def _safe_tar_member_path(member: tarfile.TarInfo) -> Path | None:
    if member.issym() or member.islnk():
        raise ValueError(f"Refusing to extract link from tar archive: {member.name}")
    raw_name = member.name.replace("\\", "/")
    path = PurePosixPath(raw_name)
    windows_path = PureWindowsPath(raw_name)
    if not path.parts:
        return None
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part == ".." for part in path.parts)
    ):
        raise ValueError(f"Unsafe path in tar archive: {member.name}")
    parts = [part for part in path.parts if part not in {"", "."}]
    if not parts:
        return None
    return Path(*parts)


def _find_prediction_dir_in_extract(extract_root: Path) -> Path:
    return _find_prediction_dir_in_tree(
        extract_root,
        not_found_message=(
            "Could not recognize prediction output format inside archive.\n"
            "Expected the archive to contain a Boltz, Boltz Lab, Boltz API, "
            "AlphaFold 3, AlphaFold 3 Server, Chai-1 Discovery, or Protenix "
            "output folder."
        ),
    )


def _find_prediction_dir_in_tree(
    root: Path,
    *,
    not_found_message: str,
) -> Path:
    candidates = _prediction_candidates_in_tree(root)
    if not candidates:
        raise ValueError(not_found_message)
    return candidates[0].path


def _prediction_candidates_in_tree(root: Path) -> list[PredictionCandidate]:
    raw_candidates = []
    for path in _iter_candidate_prediction_dirs(root):
        provider = _prediction_dir_provider(path)
        if provider is None:
            continue
        raw_candidates.append(
            PredictionCandidate(
                path=path,
                provider=provider,
                provider_label=_provider_label(provider),
                relative_path=_candidate_relative_path(root, path),
            )
        )
    candidates = _filter_prediction_candidates(root, raw_candidates)
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.relative_path.casefold(),
            candidate.provider_label.casefold(),
        ),
    )


def _candidate_relative_path(root: Path, path: Path) -> str:
    if path == root:
        return "."
    return str(path.relative_to(root))


def _filter_prediction_candidates(
    root: Path,
    candidates: list[PredictionCandidate],
) -> list[PredictionCandidate]:
    candidates = [
        candidate
        for candidate in candidates
        if not _is_provider_internal_candidate(candidate, candidates)
    ]
    if any(candidate.path != root for candidate in candidates):
        candidates = [candidate for candidate in candidates if candidate.path != root]
    return candidates


def _is_provider_internal_candidate(
    candidate: PredictionCandidate,
    candidates: list[PredictionCandidate],
) -> bool:
    if candidate.provider == "alphafold3":
        seed, sample = _parse_af3_seed_sample(candidate.path.name)
        if seed is not None and sample is not None:
            return _has_ancestor_candidate(candidate, candidates, provider="alphafold3")
    if candidate.path.name == "predictions":
        return _has_ancestor_candidate(candidate, candidates, provider="protenix")
    if candidate.provider == "protenix" and candidate.path.name.startswith("seed_"):
        return _has_ancestor_candidate(candidate, candidates, provider="protenix")
    if candidate.provider == "boltz_api" and candidate.path.name == "prediction":
        return _has_ancestor_candidate(candidate, candidates, provider="boltz_api")
    return False


def _has_ancestor_candidate(
    candidate: PredictionCandidate,
    candidates: list[PredictionCandidate],
    *,
    provider: str | None = None,
) -> bool:
    for other in candidates:
        if other.path == candidate.path:
            continue
        if provider is not None and other.provider != provider:
            continue
        try:
            candidate.path.relative_to(other.path)
        except ValueError:
            continue
        return True
    return False


def _iter_candidate_prediction_dirs(extract_root: Path) -> list[Path]:
    candidates = [extract_root]
    candidates.extend(
        path
        for path in extract_root.rglob("*")
        if path.is_dir() and "__MACOSX" not in path.parts
    )
    return candidates


def _scan_prediction_path_exact(path: Path, *, input_path: Path) -> PredictionFiles:
    if path.is_file():
        files = _scan_structure_file(path)
        files.input_path = input_path
        return files
    if path.is_dir():
        return _scan_prediction_dir_exact(path, input_path=input_path)
    raise FileNotFoundError(f"Path does not exist: {path}")


def _scan_prediction_dir_exact(pred_dir: Path, *, input_path: Path) -> PredictionFiles:
    provider = _prediction_dir_provider(pred_dir)
    files: PredictionFiles
    if provider == "af3_server":
        files = _scan_af3_server_dir(pred_dir)
    elif provider == "alphafold3":
        files = _scan_af3_dir(pred_dir)
    elif provider == "chai1":
        files = _scan_chai_dir(pred_dir)
    elif provider == "protenix":
        files = _scan_protenix_dir(pred_dir)
    elif provider == "boltz_api":
        files = _scan_boltz_api_dir(pred_dir)
    elif provider == "boltz_lab":
        files = _scan_boltz_lab_dir(pred_dir)
    elif provider == "boltz":
        files = _scan_boltz_dir(pred_dir)
    else:
        raise ValueError(f"Unsupported provider in {pred_dir}: {provider}")
    files.input_path = input_path
    return files


def _scan_structure_file(path: Path) -> PredictionFiles:
    if path.suffix.lower() not in STRUCTURE_SUFFIXES:
        raise ValueError(f"Unsupported structure file format: {path.suffix}")
    name = path.stem
    return PredictionFiles(
        name=name,
        pred_dir=path.parent,
        provider="structure_only",
        input_path=path,
        models=[
            ModelFiles(
                rank=0,
                structure_path=path,
                display_label=path.name,
                object_name=_safe_object_name(name),
            )
        ],
        capabilities={"structure_plddt"},
    )


def _scan_boltz_dir(pred_dir: Path) -> PredictionFiles:
    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="boltz",
        input_path=pred_dir,
        capabilities={"structure_plddt"},
    )

    rank_re = re.compile(rf"{re.escape(name)}_model_(\d+)")

    def rank_of(p: Path) -> int:
        match = rank_re.search(p.stem)
        return int(match.group(1)) if match else -1

    structures_by_rank: dict[int, Path] = {}
    for path in sorted(pred_dir.glob(f"{name}_model_*.pdb"), key=rank_of):
        rank = rank_of(path)
        if rank >= 0:
            structures_by_rank[rank] = path
    for path in sorted(pred_dir.glob(f"{name}_model_*.cif"), key=rank_of):
        rank = rank_of(path)
        if rank >= 0:
            structures_by_rank[rank] = path

    if not structures_by_rank:
        raise ValueError(
            f"No structure files found in {pred_dir}.\n"
            "Expected files named like '<name>_model_<rank>.cif' or '.pdb'."
        )

    files.models = [
        ModelFiles(
            rank=rank,
            structure_path=path,
            display_label=f"model_{rank}",
            object_name=f"{_safe_object_name(name)}_model_{rank}",
        )
        for rank, path in sorted(structures_by_rank.items())
    ]

    def by_rank(pattern: str) -> dict[int, Path]:
        result = {}
        for path in sorted(pred_dir.glob(pattern), key=rank_of):
            rank = rank_of(path)
            if rank >= 0:
                result[rank] = path
        return result

    confidence = by_rank(f"confidence_{name}_model_*.json")
    plddt = by_rank(f"plddt_{name}_model_*.npz")
    pae = by_rank(f"pae_{name}_model_*.npz")
    pde = by_rank(f"pde_{name}_model_*.npz")
    for model in files.models:
        model.confidence_path = confidence.get(model.rank)
        model.plddt_path = plddt.get(model.rank)
        model.pae_path = pae.get(model.rank)
        model.pde_path = pde.get(model.rank)

    if plddt:
        files.capabilities.add("plddt")
    if pae:
        files.capabilities.add("pae")
    if pde:
        files.capabilities.add("pde")

    affinity_path = pred_dir / f"affinity_{name}.json"
    if affinity_path.exists():
        files.affinity_file = affinity_path

    embeddings_path = pred_dir / f"embeddings_{name}.npz"
    if embeddings_path.exists():
        files.embeddings_file = embeddings_path

    return files


def _scan_boltz_lab_dir(pred_dir: Path) -> PredictionFiles:
    candidates = _boltz_sample_candidates(pred_dir)
    if not candidates:
        raise ValueError(
            f"No Boltz Lab prediction files found in {pred_dir}.\n"
            "Expected files named like 'sample_<n>_predicted_structure.cif' "
            "or '.pdb' with a metrics.json file."
        )

    metrics = _load_optional_json(pred_dir / "metrics.json")
    sample_results = (
        metrics.get("sample_results", {}) if isinstance(metrics, dict) else {}
    )
    if not isinstance(sample_results, dict):
        sample_results = {}

    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="boltz_lab",
        input_path=pred_dir,
        capabilities={"structure_plddt"},
    )

    for rank, item in enumerate(sorted(candidates, key=lambda item: item.sample_index)):
        sample_index = item.sample_index
        confidence = sample_results.get(f"sample_{sample_index}")
        files.models.append(
            ModelFiles(
                rank=rank,
                structure_path=item.structure_path,
                display_label=f"sample {sample_index}",
                object_name=f"{_safe_object_name(name)}_model_{rank}",
                pae_path=item.pae_path,
                metadata={
                    "sample_index": sample_index,
                    "confidence": confidence if isinstance(confidence, dict) else None,
                },
            )
        )

    if any(model.pae_path is not None for model in files.models):
        files.capabilities.add("pae")
    return files


def _scan_boltz_api_dir(pred_dir: Path) -> PredictionFiles:
    prediction_dir = _boltz_api_prediction_dir(pred_dir)
    if prediction_dir is None:
        raise ValueError(
            f"No Boltz API prediction files found in {pred_dir}.\n"
            "Expected a run root with 'outputs/files/prediction' or an extracted "
            "'prediction' folder."
        )

    candidates = _boltz_sample_candidates(prediction_dir)
    if not candidates:
        raise ValueError(
            f"No Boltz API prediction files found in {prediction_dir}.\n"
            "Expected files named like 'sample_<n>_predicted_structure.cif' "
            "or '.pdb'."
        )

    metrics_by_sample = _boltz_api_metrics_by_sample(pred_dir, prediction_dir)
    for item in candidates:
        item.confidence = metrics_by_sample.get(item.sample_index)
        item.structure_confidence = _float_or_none(
            (item.confidence or {}).get("structure_confidence")
        )

    candidates.sort(
        key=lambda item: (
            item.structure_confidence is None,
            -float(item.structure_confidence or 0.0),
            item.sample_index,
        )
    )

    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="boltz_api",
        input_path=pred_dir,
        capabilities={"structure_plddt"},
    )

    for rank, item in enumerate(candidates):
        metadata: dict[str, Any] = {
            "sample_index": item.sample_index,
            "confidence": item.confidence,
        }
        if item.structure_confidence is not None:
            metadata["structure_confidence"] = item.structure_confidence
        files.models.append(
            ModelFiles(
                rank=rank,
                structure_path=item.structure_path,
                display_label=f"rank {rank} - sample {item.sample_index}",
                object_name=f"{_safe_object_name(name)}_model_{rank}",
                pae_path=item.pae_path,
                metadata=metadata,
            )
        )

    if any(model.pae_path is not None for model in files.models):
        files.capabilities.add("pae")
    return files


def _scan_af3_dir(pred_dir: Path) -> PredictionFiles:
    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="alphafold3",
        input_path=pred_dir,
        capabilities={"structure_plddt"},
    )

    ranking_scores = _load_af3_ranking_scores(pred_dir)
    samples: list[_AF3Sample] = []
    for sample_dir in sorted(pred_dir.glob("seed-*_sample-*")):
        if not sample_dir.is_dir():
            continue
        model_path = _af3_model_path(sample_dir)
        if model_path is None:
            continue
        confidence_path = _first_confidence_json(sample_dir)
        summary_path = _af3_summary_path(sample_dir)
        summary = _load_json(summary_path) if summary_path is not None else {}
        seed, sample = _parse_af3_seed_sample(sample_dir.name)
        samples.append(
            _AF3Sample(
                model_path=model_path,
                confidence_path=confidence_path,
                summary_path=summary_path,
                summary=summary,
                ranking_score=ranking_scores.get((seed, sample)),
                seed=seed,
                sample=sample,
            )
        )

    if samples:
        samples.sort(
            key=lambda item: (
                _ranking_sort_key(item.summary, item.ranking_score),
                item.seed if item.seed is not None else 0,
                item.sample if item.sample is not None else 0,
            )
        )
        for rank, item in enumerate(samples):
            label = "rank " + str(rank)
            if item.seed is not None and item.sample is not None:
                label += f" - seed {item.seed} sample {item.sample}"
            files.models.append(
                ModelFiles(
                    rank=rank,
                    structure_path=item.model_path,
                    display_label=label,
                    object_name=f"{_safe_object_name(name)}_model_{rank}",
                    confidence_path=item.confidence_path,
                    summary_path=item.summary_path,
                    metadata={"seed": item.seed, "sample": item.sample},
                )
            )
    else:
        model_path = _af3_model_path(pred_dir)
        if model_path is None:
            raise ValueError(f"No AlphaFold 3 model CIF found in {pred_dir}.")
        stem = model_path.stem.removesuffix("_model")
        files.models.append(
            ModelFiles(
                rank=0,
                structure_path=model_path,
                display_label="rank 0",
                object_name=f"{_safe_object_name(stem)}_model_0",
                confidence_path=_first_confidence_json(pred_dir),
                summary_path=_af3_summary_path(pred_dir),
            )
        )

    if any(model.confidence_path is not None for model in files.models):
        files.capabilities.update({"pae", "contact_probs", "plddt"})
    return files


def _scan_af3_server_dir(pred_dir: Path) -> PredictionFiles:
    """Discover ranked AlphaFold 3 Server output files."""
    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="af3_server",
        input_path=pred_dir,
        capabilities={"structure_plddt"},
    )

    model_re = re.compile(r"(.+)_model_(\d+)$")
    ranked_models: list[tuple[int, str, Path]] = []
    for path in sorted(pred_dir.glob("*_model_*.cif")):
        match = model_re.fullmatch(path.stem)
        if match is None:
            continue
        prefix, rank_text = match.groups()
        ranked_models.append((int(rank_text), prefix, path))

    if not ranked_models:
        raise ValueError(f"No AlphaFold 3 Server model CIFs found in {pred_dir}.")

    for rank, prefix, structure_path in sorted(ranked_models):
        full_data_path = _ranked_json_path(
            pred_dir,
            prefix,
            "full_data",
            rank,
        )
        summary_path = _ranked_json_path(
            pred_dir,
            prefix,
            "summary_confidences",
            rank,
        )
        files.models.append(
            ModelFiles(
                rank=rank,
                structure_path=structure_path,
                display_label=f"rank {rank}",
                object_name=f"{_safe_object_name(name)}_model_{rank}",
                confidence_path=full_data_path,
                summary_path=summary_path,
                metadata={"source_prefix": prefix},
            )
        )

    if any(model.confidence_path is not None for model in files.models):
        files.capabilities.update({"pae", "contact_probs", "plddt"})
    return files


def _scan_chai_dir(pred_dir: Path) -> PredictionFiles:
    """Discover Chai-1 Discovery ranked output files."""
    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="chai1",
        input_path=pred_dir,
        capabilities={"structure_plddt"},
    )

    candidates: list[_ChaiCandidate] = []
    for structure_path in _chai_structure_paths(pred_dir):
        parsed = _parse_chai_stem(structure_path.stem)
        if parsed is None:
            continue
        model_idx, explicit_rank = parsed
        suffix = structure_path.stem.removeprefix("pred")
        confidence_path = _chai_companion_path(pred_dir, "scores", suffix)
        pae_path = _chai_companion_path(pred_dir, "pae", suffix)
        pde_path = _chai_companion_path(pred_dir, "pde", suffix)
        candidates.append(
            _ChaiCandidate(
                structure_path=structure_path,
                model_idx=model_idx,
                explicit_rank=explicit_rank,
                confidence_path=confidence_path,
                pae_path=pae_path,
                pde_path=pde_path,
                aggregate_score=_chai_aggregate_score(confidence_path),
            )
        )

    if not candidates:
        raise ValueError(
            f"No Chai-1 Discovery structure files found in {pred_dir}.\n"
            "Expected files named like 'pred.rank_<rank>.cif', "
            "'pred.model_idx_<idx>.rank_<rank>.cif', or "
            "'pred.model_idx_<idx>.cif'."
        )

    explicit = [item for item in candidates if item.explicit_rank is not None]
    raw = [item for item in candidates if item.explicit_rank is None]
    explicit.sort(
        key=lambda item: (
            item.explicit_rank,
            item.model_idx if item.model_idx is not None else -1,
            item.structure_path.name,
        )
    )
    raw.sort(
        key=lambda item: (
            item.aggregate_score is None,
            -float(item.aggregate_score or 0.0),
            item.model_idx if item.model_idx is not None else 0,
            item.structure_path.name,
        )
    )

    used_ranks: set[int] = set()
    for item in explicit:
        rank = int(item.explicit_rank)  # type: ignore[arg-type]  # explicit list is filtered
        if rank in used_ranks:
            continue
        used_ranks.add(rank)
        _append_chai_model(files, item, rank)

    next_rank = 0
    for item in raw:
        while next_rank in used_ranks:
            next_rank += 1
        used_ranks.add(next_rank)
        _append_chai_model(files, item, next_rank)

    files.models.sort(key=lambda model: model.rank)
    if any(model.pae_path is not None for model in files.models):
        files.capabilities.add("pae")
    if any(model.pde_path is not None for model in files.models):
        files.capabilities.add("pde")
    return files


def _append_chai_model(
    files: PredictionFiles,
    item: _ChaiCandidate,
    rank: int,
) -> None:
    metadata: dict[str, Any] = {}
    if item.model_idx is not None:
        metadata["model_idx"] = item.model_idx
    if item.aggregate_score is not None:
        metadata["aggregate_score"] = item.aggregate_score
    label = f"rank {rank}"
    if item.model_idx is not None:
        label += f" - model {item.model_idx}"
    files.models.append(
        ModelFiles(
            rank=rank,
            structure_path=item.structure_path,
            display_label=label,
            object_name=f"{_safe_object_name(files.name)}_model_{rank}",
            confidence_path=item.confidence_path,
            pae_path=item.pae_path,
            pde_path=item.pde_path,
            metadata=metadata,
        )
    )


def _scan_protenix_dir(pred_dir: Path) -> PredictionFiles:
    """Discover Protenix seed/predictions output files."""
    candidates = _protenix_prediction_candidates(pred_dir)
    if not candidates:
        raise ValueError(
            f"No Protenix prediction files found in {pred_dir}.\n"
            "Expected files named like '<sample>_sample_<rank>.cif' with "
            "matching '<sample>_summary_confidence_sample_<rank>.json' files "
            "under a 'seed_<seed>/predictions' directory."
        )

    name = _protenix_prediction_name(candidates, pred_dir)
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="protenix",
        input_path=pred_dir,
        capabilities={"structure_plddt"},
    )

    candidates.sort(
        key=lambda item: (
            item.ranking_score is None,
            -float(item.ranking_score or 0.0),
            item.seed if item.seed is not None else 0,
            item.sample_rank,
            item.structure_path.name,
        )
    )

    for rank, item in enumerate(candidates):
        label = f"rank {rank}"
        if item.seed is not None:
            label += f" - seed {item.seed}"
        label += f" sample {item.sample_rank}"
        metadata: dict[str, Any] = {
            "sample_rank": item.sample_rank,
            "source_prefix": item.source_prefix,
        }
        if item.seed is not None:
            metadata["seed"] = item.seed
        if item.ranking_score is not None:
            metadata["ranking_score"] = item.ranking_score
        files.models.append(
            ModelFiles(
                rank=rank,
                structure_path=item.structure_path,
                display_label=label,
                object_name=f"{_safe_object_name(name)}_model_{rank}",
                confidence_path=item.full_data_path,
                summary_path=item.summary_path,
                metadata=metadata,
            )
        )

    if any(model.confidence_path is not None for model in files.models):
        files.capabilities.update({"pae", "pde", "contact_probs", "plddt"})
    return files


# ---------------------------------------------------------------------------
# Provider-specific loaders
# ---------------------------------------------------------------------------


def _load_boltz_model_data(
    pred_files: PredictionFiles,
    model: ModelFiles,
    data: PredictionData,
    *,
    load_pae: bool,
    load_pde: bool,
    load_embeddings: bool,
    load_plddt: bool,
) -> None:
    if load_plddt and model.plddt_path is not None:
        data.plddt = np.load(model.plddt_path)["plddt"]

    if load_pae and model.pae_path is not None:
        data.pae = np.load(model.pae_path)["pae"]

    if load_pde and model.pde_path is not None:
        data.pde = np.load(model.pde_path)["pde"]

    if model.confidence_path is not None:
        confidence = _load_json(model.confidence_path)
        data.confidence = _normalise_confidence(confidence)
    elif isinstance(model.metadata.get("confidence"), dict):
        data.confidence = _normalise_confidence(model.metadata["confidence"])

    if pred_files.affinity_file is not None:
        data.affinity = _load_json(pred_files.affinity_file)

    if load_embeddings and pred_files.embeddings_file is not None:
        emb = np.load(pred_files.embeddings_file)
        data.embeddings_s = emb["s"]
        data.embeddings_z = emb["z"]


def _load_af3_model_data(
    model: ModelFiles,
    data: PredictionData,
    *,
    load_pae: bool,
    load_contact_probs: bool,
    load_plddt: bool,
) -> None:
    needs_full = load_pae or load_contact_probs or load_plddt
    if not needs_full or model.confidence_path is None:
        return

    full = _load_json(model.confidence_path)
    full_summary = {
        key: value
        for key, value in full.items()
        if key not in {"pae", "contact_probs", "atom_plddts"}
    }
    data.confidence = _normalise_confidence(
        {**(data.summary_confidence or {}), **full_summary}
    )

    if load_pae and "pae" in full:
        data.pae = np.asarray(full["pae"], dtype=np.float32)
    if load_contact_probs and "contact_probs" in full:
        data.contact_probs = np.asarray(full["contact_probs"], dtype=np.float32)
    if load_plddt and "atom_plddts" in full:
        data.plddt = _collapse_atom_plddts_to_tokens(
            model.structure_path,
            np.asarray(full["atom_plddts"], dtype=np.float32),
        )


def _load_chai_model_data(
    model: ModelFiles,
    data: PredictionData,
    *,
    load_pae: bool,
    load_pde: bool,
) -> None:
    if model.confidence_path is not None:
        data.confidence = _normalise_confidence(
            _normalise_chai_confidence(_load_chai_scores(model.confidence_path))
        )

    if load_pae and model.pae_path is not None:
        data.pae = np.load(model.pae_path).astype(np.float32)

    if load_pde and model.pde_path is not None:
        data.pde = np.load(model.pde_path).astype(np.float32)


def _load_protenix_model_data(
    model: ModelFiles,
    data: PredictionData,
    *,
    load_pae: bool,
    load_pde: bool,
    load_contact_probs: bool,
    load_plddt: bool,
) -> None:
    needs_full = load_pae or load_pde or load_contact_probs or load_plddt
    if not needs_full or model.confidence_path is None:
        return

    full = _load_json(model.confidence_path)
    if load_pae:
        pae = _first_present(full, ("token_pair_pae", "pae"))
        if pae is not None:
            data.pae = _squeezed_float32_array(pae)
    if load_pde:
        pde = _first_present(full, ("token_pair_pde", "pde"))
        if pde is not None:
            data.pde = _squeezed_float32_array(pde)
    if load_contact_probs and "contact_probs" in full:
        data.contact_probs = _squeezed_float32_array(full["contact_probs"])
    if load_plddt:
        atom_plddt = _first_present(full, ("atom_plddt", "atom_plddts"))
        if atom_plddt is not None:
            data.plddt = _collapse_atom_plddts_to_tokens(
                model.structure_path,
                np.asarray(atom_plddt, dtype=np.float32),
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collapse_atom_plddts_to_tokens(
    structure_path: Path,
    atom_plddts: np.ndarray,
) -> np.ndarray:

    atoms = parse_structure_atoms(structure_path)
    if len(atoms) != len(atom_plddts):
        raise ValueError(
            f"AF3 atom_plddts length {len(atom_plddts)} does not match "
            f"{len(atoms)} atoms in {structure_path.name}."
        )

    values = np.asarray(atom_plddts, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmax(finite)) > 1.5:
        values = values / 100.0

    # Group values by residue, then average over residues. This is necessary because
    # AF3 atom_plddts are per-atom, but we want per-residue
    residue_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for atom, value in zip(atoms, values, strict=True):
        # skip heterocomponents since they always have one token per atom
        if not atom["hetatm"]:
            key = (atom["chain"], atom["resi"], atom["resn"])
            residue_values[key].append(float(value))

    # Rebuild in structure-token order rather than "all residues then ligands".
    out: list[float] = []
    emitted: set[tuple[str, int, str]] = set()
    for atom, value in zip(atoms, values, strict=True):
        if atom["hetatm"]:
            out.append(float(value))
        else:
            key = (atom["chain"], atom["resi"], atom["resn"])
            if key not in emitted:
                emitted.add(key)
                out.append(float(np.nanmean(residue_values[key])))

    return np.asarray(out, dtype=np.float32)


def _chai_structure_paths(pred_dir: Path) -> list[Path]:
    paths = [
        path
        for suffix in STRUCTURE_SUFFIXES
        for path in pred_dir.glob(f"pred*{suffix}")
    ]
    return sorted(paths, key=lambda path: path.name)


def _parse_chai_stem(stem: str) -> tuple[int | None, int | None] | None:
    match = re.fullmatch(r"pred(?:\.model_idx_(\d+))?(?:\.rank_(\d+))?", stem)
    if match is None:
        return None
    model_idx, rank = match.groups()
    if model_idx is None and rank is None:
        return None
    return (
        int(model_idx) if model_idx is not None else None,
        int(rank) if rank is not None else None,
    )


def _chai_companion_path(pred_dir: Path, stem: str, suffix: str) -> Path | None:
    if stem == "scores":
        json_path = pred_dir / f"{stem}{suffix}.json"
        if json_path.exists():
            return json_path
        npz_path = pred_dir / f"{stem}{suffix}.npz"
        if npz_path.exists():
            return npz_path
        return None
    path = pred_dir / f"{stem}{suffix}.npy"
    return path if path.exists() else None


def _chai_aggregate_score(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return _float_or_none(_load_chai_scores(path).get("aggregate_score"))
    except Exception:
        return None


def _load_chai_scores(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as scores:
            return {
                key: _numpy_score_to_python(key, scores[key]) for key in scores.files
            }
    raise ValueError(f"Unsupported Chai score file format: {path.suffix}")


def _numpy_score_to_python(key: str, value):
    array = np.asarray(value)
    scalar_keys = {
        "aggregate_score",
        "ptm",
        "iptm",
        "has_inter_chain_clashes",
    }
    if array.shape == ():
        return array.item()
    if key in scalar_keys and array.size == 1:
        return array.reshape(-1)[0].item()
    return array.tolist()


def _float_or_none(value) -> float | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _boltz_api_prediction_dir(pred_dir: Path) -> Path | None:
    if _is_boltz_api_prediction_leaf(pred_dir):
        return pred_dir
    nested = pred_dir / "outputs" / "files" / "prediction"
    if _is_boltz_api_prediction_leaf(nested):
        return nested
    return None


def _is_boltz_api_prediction_leaf(pred_dir: Path) -> bool:
    return (
        pred_dir.is_dir()
        and (pred_dir / "metrics.json").exists()
        and bool(_boltz_sample_structure_paths(pred_dir))
    )


def _boltz_sample_structure_paths(pred_dir: Path) -> dict[int, Path]:
    structures_by_sample: dict[int, Path] = {}
    for suffix in (".pdb", ".cif"):
        for path in pred_dir.glob(f"sample_*_predicted_structure{suffix}"):
            sample_index = _parse_boltz_sample_structure_stem(path.stem)
            if sample_index is not None:
                structures_by_sample[sample_index] = path
    return structures_by_sample


def _parse_boltz_sample_structure_stem(stem: str) -> int | None:
    match = re.fullmatch(r"sample_(\d+)_predicted_structure", stem)
    return int(match.group(1)) if match else None


def _boltz_sample_candidates(pred_dir: Path) -> list[_BoltzSampleCandidate]:
    candidates = []
    for sample_index, structure_path in _boltz_sample_structure_paths(pred_dir).items():
        pae_path = pred_dir / f"sample_{sample_index}_pae.npz"
        candidates.append(
            _BoltzSampleCandidate(
                sample_index=sample_index,
                structure_path=structure_path,
                pae_path=pae_path if pae_path.exists() else None,
            )
        )
    return sorted(candidates, key=lambda item: item.sample_index)


def _is_boltz_api_metrics(metrics: dict | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    return "all_sample_results" in metrics or "best_sample" in metrics


def _is_boltz_api_output(output: dict | None) -> bool:
    if not isinstance(output, dict):
        return False
    return "all_sample_results" in output or "best_sample" in output


def _boltz_api_metrics_by_sample(
    pred_dir: Path,
    prediction_dir: Path,
) -> dict[int, dict]:
    metrics = _load_optional_json(prediction_dir / "metrics.json")
    results = None
    if isinstance(metrics, dict):
        results = metrics.get("all_sample_results")

    if not isinstance(results, list) or not results:
        run = _load_optional_json(pred_dir / "run.json")
        output = run.get("output") if isinstance(run, dict) else None
        if isinstance(output, dict):
            results = output.get("all_sample_results")

    if not isinstance(results, list):
        return {}

    by_sample = {}
    for sample_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        metrics_for_sample = result.get("metrics", result)
        if isinstance(metrics_for_sample, dict):
            by_sample[sample_index] = metrics_for_sample
    return by_sample


def _protenix_prediction_candidates(pred_dir: Path) -> list[_ProtenixCandidate]:
    candidates: list[_ProtenixCandidate] = []
    for predictions_dir in _protenix_prediction_dirs(pred_dir):
        for structure_path in _protenix_structure_paths(predictions_dir):
            parsed = _parse_protenix_stem(structure_path.stem)
            if parsed is None:
                continue
            prefix, sample_rank = parsed
            summary_path = (
                predictions_dir
                / f"{prefix}_summary_confidence_sample_{sample_rank}.json"
            )
            if not summary_path.exists():
                continue
            full_data_path = (
                predictions_dir / f"{prefix}_full_data_sample_{sample_rank}.json"
            )
            summary = _load_json(summary_path)
            candidates.append(
                _ProtenixCandidate(
                    structure_path=structure_path,
                    summary_path=summary_path,
                    full_data_path=full_data_path if full_data_path.exists() else None,
                    source_prefix=prefix,
                    sample_rank=sample_rank,
                    seed=_parse_protenix_seed(predictions_dir),
                    ranking_score=_float_or_none(summary.get("ranking_score")),
                )
            )
    return candidates


def _protenix_prediction_dirs(pred_dir: Path) -> list[Path]:
    candidates = []
    if pred_dir.name == "predictions":
        candidates.append(pred_dir)
    direct = pred_dir / "predictions"
    if direct.is_dir():
        candidates.append(direct)
    candidates.extend(
        path for path in pred_dir.glob("seed_*/predictions") if path.is_dir()
    )
    return sorted(set(candidates))


def _protenix_structure_paths(predictions_dir: Path) -> list[Path]:
    paths = [
        path
        for suffix in STRUCTURE_SUFFIXES
        for path in predictions_dir.glob(f"*_sample_*{suffix}")
        if _parse_protenix_stem(path.stem) is not None
    ]
    return sorted(paths, key=lambda path: path.name)


def _parse_protenix_stem(stem: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"(.+)_sample_(\d+)", stem)
    if match is None:
        return None
    prefix, sample = match.groups()
    if prefix.endswith("_summary_confidence") or prefix.endswith("_full_data"):
        return None
    return prefix, int(sample)


def _parse_protenix_seed(predictions_dir: Path) -> int | None:
    for parent in [predictions_dir.parent, *predictions_dir.parents]:
        match = re.fullmatch(r"seed_(\d+)", parent.name)
        if match:
            return int(match.group(1))
    return None


def _protenix_prediction_name(
    candidates: list[_ProtenixCandidate], pred_dir: Path
) -> str:
    prefixes = {item.source_prefix for item in candidates}
    if len(prefixes) == 1:
        return next(iter(prefixes))
    return pred_dir.name


def _first_present(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _squeezed_float32_array(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    return array


def _normalise_confidence(confidence: dict | None) -> dict | None:
    if confidence is None:
        return None
    normalised = dict(confidence)

    if "structure_confidence" in normalised and "confidence_score" not in normalised:
        normalised["confidence_score"] = normalised["structure_confidence"]

    if "chain_ptm" in normalised and "chains_ptm" not in normalised:
        chain_ptm = normalised["chain_ptm"]
        if isinstance(chain_ptm, list):
            normalised["chains_ptm"] = {
                str(idx): value for idx, value in enumerate(chain_ptm)
            }

    if "chain_iptm" in normalised and "chains_iptm" not in normalised:
        chain_iptm = normalised["chain_iptm"]
        if isinstance(chain_iptm, list):
            normalised["chains_iptm"] = {
                str(idx): value for idx, value in enumerate(chain_iptm)
            }

    if "chain_pair_iptm" in normalised and "pair_chains_iptm" not in normalised:
        matrix = normalised["chain_pair_iptm"]
        if isinstance(matrix, list):
            normalised["pair_chains_iptm"] = _matrix_list_to_nested_dict(matrix)

    if "chain_pair_pae_min" in normalised and "pair_chains_pae_min" not in normalised:
        matrix = normalised["chain_pair_pae_min"]
        if isinstance(matrix, list):
            normalised["pair_chains_pae_min"] = _matrix_list_to_nested_dict(matrix)

    return normalised


def _normalise_chai_confidence(confidence: dict | None) -> dict | None:
    if confidence is None:
        return None
    normalised = dict(confidence)

    if "aggregate_score" in normalised and "ranking_score" not in normalised:
        normalised["ranking_score"] = normalised["aggregate_score"]

    if (
        "per_chain_ptm" in normalised
        and "chains_ptm" not in normalised
        and "chain_ptm" not in normalised
    ):
        chain_ptm = _chai_first_sample(normalised["per_chain_ptm"])
        if isinstance(chain_ptm, list):
            normalised["chains_ptm"] = {
                str(idx): value for idx, value in enumerate(chain_ptm)
            }

    if (
        "per_chain_pair_iptm" in normalised
        and "pair_chains_iptm" not in normalised
        and "chain_pair_iptm" not in normalised
    ):
        matrix = _chai_first_sample(normalised["per_chain_pair_iptm"])
        if isinstance(matrix, list):
            normalised["pair_chains_iptm"] = _matrix_list_to_nested_dict(matrix)

    if "has_inter_chain_clashes" in normalised and "has_clash" not in normalised:
        normalised["has_clash"] = normalised["has_inter_chain_clashes"]

    return normalised


def _chai_first_sample(value):
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def _matrix_list_to_nested_dict(matrix: list) -> dict[str, dict[str, float]]:
    return {
        str(i): {str(j): value for j, value in enumerate(row)}
        for i, row in enumerate(matrix)
        if isinstance(row, list)
    }


def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _first(paths) -> Path | None:
    return next(iter(sorted(paths)), None)


def _af3_model_path(path: Path) -> Path | None:
    """Return an AF3 model CIF using current or older truncated names."""
    return _first(path.glob("*_model.cif")) or (
        path / "model.cif" if (path / "model.cif").exists() else None
    )


def _af3_summary_path(path: Path) -> Path | None:
    """Return an AF3 summary JSON using current or older truncated names."""
    return _first(path.glob("*_summary_confidences.json")) or (
        path / "summary_confidences.json"
        if (path / "summary_confidences.json").exists()
        else None
    )


def _first_confidence_json(path: Path) -> Path | None:
    candidates = [
        candidate
        for candidate in path.glob("*_confidences.json")
        if candidate.name != "summary_confidences.json"
        and not candidate.name.endswith("_summary_confidences.json")
    ]
    return _first(candidates) or (
        path / "confidences.json" if (path / "confidences.json").exists() else None
    )


def _parse_af3_seed_sample(dirname: str) -> tuple[int | None, int | None]:
    match = re.fullmatch(r"seed-(\d+)_sample-(\d+)", dirname)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _load_af3_ranking_scores(
    pred_dir: Path,
) -> dict[tuple[int | None, int | None], float]:
    """Read AF3 ranking scores using (seed, sample) as the lookup key."""
    path = _af3_ranking_scores_path(pred_dir)
    if path is None:
        return {}

    scores: dict[tuple[int | None, int | None], float] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                seed = int(row["seed"])
                sample = int(row["sample"])
                score = float(row["ranking_score"])
            except (KeyError, TypeError, ValueError):
                continue
            scores[(seed, sample)] = score
    return scores


def _af3_ranking_scores_path(pred_dir: Path) -> Path | None:
    """Return current or older AF3 ranking-score CSV path."""
    prefixed = pred_dir / f"{pred_dir.name}_ranking_scores.csv"
    truncated = pred_dir / "ranking_scores.csv"
    if prefixed.exists():
        return prefixed
    if truncated.exists():
        return truncated
    return _first(pred_dir.glob("*_ranking_scores.csv"))


def _ranked_json_path(
    pred_dir: Path,
    prefix: str,
    stem: str,
    rank: int,
) -> Path | None:
    """Return prefixed or truncated ranked JSON path."""
    prefixed = pred_dir / f"{prefix}_{stem}_{rank}.json"
    truncated = pred_dir / f"{stem}_{rank}.json"
    if prefixed.exists():
        return prefixed
    if truncated.exists():
        return truncated
    return None


def _ranking_sort_key(summary: dict, ranking_score: float | None = None) -> float:
    value = ranking_score if ranking_score is not None else summary.get("ranking_score")
    try:
        return -float(value)  # type: ignore[arg-type]  # None/invalid caught below
    except (TypeError, ValueError):
        return float("inf")


def _safe_object_name(name: str) -> str:
    safe = re.sub(r"\W+", "_", name).strip("_")
    return safe or "prediction"
