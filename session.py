"""
Session-state helpers for FoldQC.

This module is intentionally Qt- and PyMOL-independent. Callers provide a
settings-like object with ``value`` and ``setValue`` methods, such as QSettings.
"""

from __future__ import annotations

from dataclasses import dataclass

SETTINGS_ORGANIZATION = "FoldQC"
SETTINGS_APPLICATION = "FoldQC"
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
SETTINGS_KEY_GEOMETRY = "session/geometry"


@dataclass(frozen=True)
class SessionState:
    """Serialized lightweight GUI session state."""

    path: str = ""
    model_rank: int | None = None
    metric_key: str = ""
    target_name: str = ""
    reference_text: str = ""
    cutoff_text: str = ""
    palette_key: str = ""
    palette_reversed: bool = False
    scale_min: str = ""
    scale_max: str = ""
    geometry: object | None = None


@dataclass
class PendingSessionRestore:
    """Saved UI state that may wait for path/model/object loading."""

    model_rank: int | None = None
    metric_key: str | None = None
    target_name: str | None = None


def settings_text(settings, key: str) -> str:
    """Read one settings value as text, returning an empty string if missing."""
    value = settings.value(key, "")
    if value is None:
        return ""
    return str(value)


def settings_int(settings, key: str) -> int | None:
    """Read one optional integer settings value."""
    value = settings.value(key, None)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def settings_bool(settings, key: str) -> bool:
    """Read one settings value as a bool across Qt/Python backends."""
    value = settings.value(key, False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def read_session_state(settings) -> SessionState:
    """Read the persisted FoldQC GUI session from a settings-like object."""
    return SessionState(
        path=settings_text(settings, SETTINGS_KEY_PATH),
        model_rank=settings_int(settings, SETTINGS_KEY_MODEL_RANK),
        metric_key=settings_text(settings, SETTINGS_KEY_METRIC),
        target_name=settings_text(settings, SETTINGS_KEY_TARGET),
        reference_text=settings_text(settings, SETTINGS_KEY_REFERENCE),
        cutoff_text=settings_text(settings, SETTINGS_KEY_CUTOFF),
        palette_key=settings_text(settings, SETTINGS_KEY_PALETTE),
        palette_reversed=settings_bool(settings, SETTINGS_KEY_PALETTE_REVERSE),
        scale_min=settings_text(settings, SETTINGS_KEY_SCALE_MIN),
        scale_max=settings_text(settings, SETTINGS_KEY_SCALE_MAX),
        geometry=settings.value(SETTINGS_KEY_GEOMETRY, None),
    )


def write_session_state(settings, state: SessionState) -> None:
    """Write a FoldQC GUI session to a settings-like object."""
    settings.setValue(SETTINGS_KEY_PATH, state.path)
    settings.setValue(
        SETTINGS_KEY_MODEL_RANK,
        "" if state.model_rank is None else state.model_rank,
    )
    settings.setValue(SETTINGS_KEY_METRIC, state.metric_key or "")
    settings.setValue(SETTINGS_KEY_TARGET, state.target_name)
    settings.setValue(SETTINGS_KEY_REFERENCE, state.reference_text)
    settings.setValue(SETTINGS_KEY_CUTOFF, state.cutoff_text)
    settings.setValue(SETTINGS_KEY_PALETTE, state.palette_key)
    settings.setValue(SETTINGS_KEY_PALETTE_REVERSE, state.palette_reversed)
    settings.setValue(SETTINGS_KEY_SCALE_MIN, state.scale_min)
    settings.setValue(SETTINGS_KEY_SCALE_MAX, state.scale_max)
    if state.geometry:
        settings.setValue(SETTINGS_KEY_GEOMETRY, state.geometry)
    sync = getattr(settings, "sync", None)
    if callable(sync):
        sync()
