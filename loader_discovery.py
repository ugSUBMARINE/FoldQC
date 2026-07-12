"""Provider-neutral path, tree, and archive discovery."""

from __future__ import annotations

import shutil
import stat
import tarfile
import tempfile
import weakref
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from .loader_models import PredictionCandidate, PredictionDiscovery, PredictionFiles
from .loader_utils import ARCHIVE_EXTENSIONS, _safe_object_name
from .providers.registry import BUILTIN_PROVIDERS


class _TemporaryExtraction:
    """Keep extracted archive contents alive while PredictionFiles is referenced."""

    def __init__(self, root: Path) -> None:
        self.root = root
        weakref.finalize(self, shutil.rmtree, root, ignore_errors=True)


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


def _candidate_relative_path(root: Path, path: Path) -> str:
    if path == root:
        return "."
    return str(path.relative_to(root))


def _iter_candidate_prediction_dirs(extract_root: Path) -> list[Path]:
    candidates = [extract_root]
    candidates.extend(
        path
        for path in extract_root.rglob("*")
        if path.is_dir() and "__MACOSX" not in path.parts
    )
    return candidates


def discover_prediction_candidates(path: str | Path) -> PredictionDiscovery:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        if _archive_kind(source) is not None:
            return _discover_archive_file(source)
        provider = BUILTIN_PROVIDERS.detect(source)
        if provider is not None:
            return PredictionDiscovery(
                input_path=source,
                candidates=(
                    PredictionCandidate(
                        path=source,
                        provider=provider.key,
                        provider_label=provider.label,
                        relative_path=source.name,
                    ),
                ),
            )
        raise ValueError(f"Unsupported file format: {source.name}")
    if source.is_dir():
        candidates = _prediction_candidates_in_tree(source)
        if not candidates:
            labels = ", ".join(BUILTIN_PROVIDERS.directory_labels)
            raise ValueError(
                f"Could not recognize prediction output format in {source}.\n"
                f"Expected a supported prediction folder ({labels}); or select "
                "a single .cif/.pdb structure file."
            )
        return PredictionDiscovery(input_path=source, candidates=tuple(candidates))
    raise FileNotFoundError(f"Path does not exist: {source}")


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
        (_extract_zip_safely if kind == "zip" else _extract_tar_safely)(
            path, extract_root
        )
        candidates = _prediction_candidates_in_tree(extract_root)
        if not candidates:
            labels = ", ".join(BUILTIN_PROVIDERS.directory_labels)
            raise ValueError(
                "Could not recognize prediction output format inside archive.\n"
                f"Expected the archive to contain a supported output folder ({labels})."
            )
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return PredictionDiscovery(
        input_path=path,
        candidates=tuple(candidates),
        _temporary_directory=_TemporaryExtraction(temp_root),
    )


def _prediction_candidates_in_tree(root: Path) -> list[PredictionCandidate]:
    raw = []
    for path in _iter_candidate_prediction_dirs(root):
        provider = BUILTIN_PROVIDERS.detect(path)
        if provider is None or path.is_file():
            continue
        raw.append(
            PredictionCandidate(
                path=path,
                provider=provider.key,
                provider_label=provider.label,
                relative_path=_candidate_relative_path(root, path),
            )
        )
    candidates = [
        candidate
        for candidate in raw
        if not any(
            provider.is_internal_candidate(candidate, raw)
            for provider in BUILTIN_PROVIDERS.providers
        )
    ]
    if any(candidate.path != root for candidate in candidates):
        candidates = [candidate for candidate in candidates if candidate.path != root]
    return sorted(
        candidates,
        key=lambda c: (c.relative_path.casefold(), c.provider_label.casefold()),
    )


def scan_prediction_path_exact(path: Path, *, input_path: Path) -> PredictionFiles:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    provider = BUILTIN_PROVIDERS.detect(path)
    if provider is None:
        raise ValueError(f"Unsupported provider in {path}: None")
    files = provider.scan(path)
    files.input_path = input_path
    return files
