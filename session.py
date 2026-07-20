"""Qt-independent persistence helpers for recent FoldQC predictions."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .alphafold_database import normalize_uniprot_qualifier

SETTINGS_ORGANIZATION = "FoldQC"
SETTINGS_APPLICATION = "FoldQC"
SETTINGS_KEY_RECENT_PREDICTIONS = "session/recent_predictions"
SETTINGS_KEY_RECENT_AFDB_ACCESSIONS = "session/recent_afdb_accessions"
SETTINGS_KEY_GEOMETRY = "session/geometry"
MAX_RECENT_PREDICTIONS = 10

# Keys written by the pre-history session format.  They are retained only for
# one-time migration and cleanup.
SETTINGS_KEY_PATH = "session/input_path"
SETTINGS_KEY_MODEL_RANK = "session/model_rank"
SETTINGS_KEY_METRIC = "session/metric_key"
SETTINGS_KEY_TARGET = "session/target_name"
SETTINGS_KEY_REFERENCE = "session/reference_text"
SETTINGS_KEY_CUTOFF = "session/cutoff_text"
SETTINGS_KEY_PALETTE = "session/palette_key"
SETTINGS_KEY_PALETTE_REVERSE = "session/palette_reverse"
SETTINGS_KEY_SCALE_MIN = "session/scale_min"
SETTINGS_KEY_SCALE_MAX = "session/scale_max"
LEGACY_SETTINGS_KEYS = (
    SETTINGS_KEY_PATH,
    SETTINGS_KEY_MODEL_RANK,
    SETTINGS_KEY_METRIC,
    SETTINGS_KEY_TARGET,
    SETTINGS_KEY_REFERENCE,
    SETTINGS_KEY_CUTOFF,
    SETTINGS_KEY_PALETTE,
    SETTINGS_KEY_PALETTE_REVERSE,
    SETTINGS_KEY_SCALE_MIN,
    SETTINGS_KEY_SCALE_MAX,
)


@dataclass(frozen=True)
class SessionState:
    """The only dialog state persisted across FoldQC sessions."""

    recent_predictions: tuple[str, ...] = ()
    recent_afdb_accessions: tuple[str, ...] = ()
    geometry: object | None = None


def normalize_prediction_path(path: str | Path) -> str:
    """Return a stable absolute history path without requiring it to exist."""
    text = str(path).strip()
    if not text:
        return ""
    return os.path.abspath(os.path.expanduser(text))


def _deduplication_key(path: str) -> str:
    return os.path.normcase(path)


def normalize_recent_predictions(paths: Iterable[object]) -> tuple[str, ...]:
    """Normalize an MRU sequence, preserving its first occurrence ordering."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in paths:
        if not isinstance(value, (str, Path)):
            continue
        path = normalize_prediction_path(value)
        key = _deduplication_key(path)
        if not path or key in seen:
            continue
        normalized.append(path)
        seen.add(key)
        if len(normalized) >= MAX_RECENT_PREDICTIONS:
            break
    return tuple(normalized)


def add_recent_prediction(
    recent_predictions: Iterable[object], path: str | Path
) -> tuple[str, ...]:
    """Move one successfully loaded prediction to the front of the MRU list."""
    return normalize_recent_predictions((path, *recent_predictions))


def remove_recent_prediction(
    recent_predictions: Iterable[object], path: str | Path
) -> tuple[str, ...]:
    """Remove one normalized path from an MRU list."""
    target = normalize_prediction_path(path)
    key = _deduplication_key(target)
    return normalize_recent_predictions(
        value
        for value in recent_predictions
        if _deduplication_key(normalize_prediction_path(str(value))) != key
    )


def normalize_recent_afdb_accessions(values: Iterable[object]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            accession = normalize_uniprot_qualifier(value)
        except ValueError:
            continue
        if accession in seen:
            continue
        normalized.append(accession)
        seen.add(accession)
        if len(normalized) >= MAX_RECENT_PREDICTIONS:
            break
    return tuple(normalized)


def add_recent_afdb_accession(
    recent_accessions: Iterable[object], accession: str
) -> tuple[str, ...]:
    return normalize_recent_afdb_accessions((accession, *recent_accessions))


def _read_recent_predictions(settings) -> tuple[str, ...]:
    raw = settings.value(SETTINGS_KEY_RECENT_PREDICTIONS, None)
    if raw is None:
        legacy_path = settings.value(SETTINGS_KEY_PATH, "")
        return normalize_recent_predictions((legacy_path,))
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return ()
    else:
        decoded = raw
    if not isinstance(decoded, (list, tuple)):
        return ()
    return normalize_recent_predictions(decoded)


def _read_recent_afdb_accessions(settings) -> tuple[str, ...]:
    raw = settings.value(SETTINGS_KEY_RECENT_AFDB_ACCESSIONS, None)
    if raw is None:
        return ()
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return ()
    else:
        decoded = raw
    if not isinstance(decoded, (list, tuple)):
        return ()
    return normalize_recent_afdb_accessions(decoded)


def read_session_state(settings) -> SessionState:
    """Read history and geometry, including the legacy last-path migration."""
    return SessionState(
        recent_predictions=_read_recent_predictions(settings),
        recent_afdb_accessions=_read_recent_afdb_accessions(settings),
        geometry=settings.value(SETTINGS_KEY_GEOMETRY, None),
    )


def write_session_state(settings, state: SessionState) -> None:
    """Persist only recent predictions and dialog geometry, then remove legacy keys."""
    recent = normalize_recent_predictions(state.recent_predictions)
    settings.setValue(
        SETTINGS_KEY_RECENT_PREDICTIONS,
        json.dumps(recent, separators=(",", ":")),
    )
    recent_afdb = normalize_recent_afdb_accessions(state.recent_afdb_accessions)
    settings.setValue(
        SETTINGS_KEY_RECENT_AFDB_ACCESSIONS,
        json.dumps(recent_afdb, separators=(",", ":")),
    )
    if state.geometry is not None:
        settings.setValue(SETTINGS_KEY_GEOMETRY, state.geometry)
    remove = getattr(settings, "remove", None)
    if callable(remove):
        for key in LEGACY_SETTINGS_KEYS:
            remove(key)
    sync = getattr(settings, "sync", None)
    if callable(sync):
        sync()
