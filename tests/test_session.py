from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import session


class FakeSettings:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = dict(values or {})
        self.synced = 0

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:
        self.values[key] = value

    def sync(self) -> None:
        self.synced += 1


class FakeSettingsWithoutSync:
    def __init__(self) -> None:
        self.values = {}

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:
        self.values[key] = value


def test_settings_text_handles_missing_none_and_values() -> None:
    settings = FakeSettings(
        {
            "none": None,
            "number": 7,
            "text": "value",
        }
    )

    assert session.settings_text(settings, "missing") == ""
    assert session.settings_text(settings, "none") == ""
    assert session.settings_text(settings, "number") == "7"
    assert session.settings_text(settings, "text") == "value"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("7", 7),
        (8, 8),
        ("bad", None),
        (object(), None),
    ],
)
def test_settings_int_coercion(raw, expected) -> None:
    assert session.settings_int(FakeSettings({"rank": raw}), "rank") == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("TRUE", True),
        (" yes ", True),
        ("on", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("0", False),
        ("", False),
    ],
)
def test_settings_bool_coercion(raw, expected: bool) -> None:
    assert session.settings_bool(FakeSettings({"flag": raw}), "flag") is expected


def test_read_session_state_defaults() -> None:
    state = session.read_session_state(FakeSettings())

    assert state == session.SessionState()


def test_read_session_state_populated_values() -> None:
    geometry = b"geometry"
    state = session.read_session_state(
        FakeSettings(
            {
                session.SETTINGS_KEY_PATH: "/tmp/prediction",
                session.SETTINGS_KEY_MODEL_RANK: "2",
                session.SETTINGS_KEY_METRIC: "pde_contact",
                session.SETTINGS_KEY_TARGET: "target_model_2",
                session.SETTINGS_KEY_REFERENCE: "chain A",
                session.SETTINGS_KEY_CUTOFF: "8.0",
                session.SETTINGS_KEY_PALETTE: "white_blue",
                session.SETTINGS_KEY_PALETTE_REVERSE: "true",
                session.SETTINGS_KEY_SCALE_MIN: "1",
                session.SETTINGS_KEY_SCALE_MAX: "10",
                session.SETTINGS_KEY_GEOMETRY: geometry,
            }
        )
    )

    assert state == session.SessionState(
        path="/tmp/prediction",
        model_rank=2,
        metric_key="pde_contact",
        target_name="target_model_2",
        reference_text="chain A",
        cutoff_text="8.0",
        palette_key="white_blue",
        palette_reversed=True,
        scale_min="1",
        scale_max="10",
        geometry=geometry,
    )


def test_write_session_state_serializes_values_and_syncs() -> None:
    geometry = b"geometry"
    settings = FakeSettings()

    session.write_session_state(
        settings,
        session.SessionState(
            path="/tmp/prediction",
            model_rank=2,
            metric_key="pde_contact",
            target_name="target_model_2",
            reference_text="chain A",
            cutoff_text="8.0",
            palette_key="white_blue",
            palette_reversed=True,
            scale_min="1",
            scale_max="10",
            geometry=geometry,
        ),
    )

    assert settings.values == {
        session.SETTINGS_KEY_PATH: "/tmp/prediction",
        session.SETTINGS_KEY_MODEL_RANK: 2,
        session.SETTINGS_KEY_METRIC: "pde_contact",
        session.SETTINGS_KEY_TARGET: "target_model_2",
        session.SETTINGS_KEY_REFERENCE: "chain A",
        session.SETTINGS_KEY_CUTOFF: "8.0",
        session.SETTINGS_KEY_PALETTE: "white_blue",
        session.SETTINGS_KEY_PALETTE_REVERSE: True,
        session.SETTINGS_KEY_SCALE_MIN: "1",
        session.SETTINGS_KEY_SCALE_MAX: "10",
        session.SETTINGS_KEY_GEOMETRY: geometry,
    }
    assert settings.synced == 1


def test_write_session_state_serializes_empty_optionals_and_skips_falsey_geometry() -> (
    None
):
    settings = FakeSettings()

    session.write_session_state(
        settings,
        session.SessionState(
            path="/tmp/prediction",
            model_rank=None,
            metric_key="",
            geometry=b"",
        ),
    )

    assert settings.values[session.SETTINGS_KEY_MODEL_RANK] == ""
    assert settings.values[session.SETTINGS_KEY_METRIC] == ""
    assert session.SETTINGS_KEY_GEOMETRY not in settings.values
    assert settings.synced == 1


def test_write_session_state_accepts_settings_without_sync() -> None:
    settings = FakeSettingsWithoutSync()

    session.write_session_state(settings, session.SessionState(path="/tmp/prediction"))

    assert settings.values[session.SETTINGS_KEY_PATH] == "/tmp/prediction"


def test_pending_session_restore_defaults() -> None:
    pending = session.PendingSessionRestore()

    assert pending.model_rank is None
    assert pending.metric_key is None
    assert pending.target_name is None
