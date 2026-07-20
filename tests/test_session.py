from __future__ import annotations

import json
from pathlib import Path

from FoldQC import session


class FakeSettings:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = dict(values or {})
        self.synced = 0
        self.removed: list[str] = []

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:
        self.values[key] = value

    def remove(self, key: str) -> None:
        self.removed.append(key)
        self.values.pop(key, None)

    def sync(self) -> None:
        self.synced += 1


class FakeSettingsWithoutOptionalMethods:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:
        self.values[key] = value


def test_prediction_paths_are_normalized_without_requiring_existence(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "missing" / "prediction"
    assert session.normalize_prediction_path(nested) == str(nested.absolute())
    assert session.normalize_prediction_path("  ") == ""


def test_recent_predictions_are_mru_deduplicated_and_limited(tmp_path: Path) -> None:
    paths = tuple(tmp_path / f"prediction_{index}" for index in range(12))
    recent = session.normalize_recent_predictions(paths)
    assert recent == tuple(str(path.absolute()) for path in paths[:10])

    moved = session.add_recent_prediction(recent, paths[4])
    assert moved[0] == str(paths[4].absolute())
    assert len(moved) == 10
    assert moved.count(str(paths[4].absolute())) == 1

    removed = session.remove_recent_prediction(moved, paths[4])
    assert str(paths[4].absolute()) not in removed


def test_recent_afdb_accessions_are_normalized_deduplicated_and_limited() -> None:
    accessions = tuple(f"P1234{index}" for index in range(10)) + (
        "Q5VSL9",
        "q5vsl9",
    )
    recent = session.normalize_recent_afdb_accessions(accessions)
    assert recent == accessions[:10]

    moved = session.add_recent_afdb_accession(recent, "p12344")
    assert moved[0] == "P12344"
    assert len(moved) == 10
    assert moved.count("P12344") == 1


def test_recent_afdb_accessions_discard_invalid_values() -> None:
    assert session.normalize_recent_afdb_accessions(
        (" q5vsl9-4 ", "AF-Q5VSL9-F1", "Q5VSL9-0", 7)
    ) == ("Q5VSL9-4",)


def test_read_session_state_defaults() -> None:
    assert session.read_session_state(FakeSettings()) == session.SessionState()


def test_read_session_state_decodes_and_normalizes_json(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    geometry = b"geometry"
    settings = FakeSettings(
        {
            session.SETTINGS_KEY_RECENT_PREDICTIONS: json.dumps(
                [str(first), str(second), str(first), 7]
            ),
            session.SETTINGS_KEY_RECENT_AFDB_ACCESSIONS: json.dumps(
                ["q5vsl9", "Q5VSL9", "P12345", "bad"]
            ),
            session.SETTINGS_KEY_GEOMETRY: geometry,
        }
    )

    assert session.read_session_state(settings) == session.SessionState(
        recent_predictions=(str(first.absolute()), str(second.absolute())),
        recent_afdb_accessions=("Q5VSL9", "P12345"),
        geometry=geometry,
    )


def test_read_session_state_rejects_malformed_history() -> None:
    for raw in ("not-json", json.dumps({"path": "/tmp/prediction"}), 7):
        state = session.read_session_state(
            FakeSettings({session.SETTINGS_KEY_RECENT_PREDICTIONS: raw})
        )
        assert state.recent_predictions == ()


def test_read_session_state_migrates_legacy_path_without_checking_it(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "no-longer-present"
    state = session.read_session_state(
        FakeSettings({session.SETTINGS_KEY_PATH: str(legacy)})
    )
    assert state.recent_predictions == (str(legacy.absolute()),)


def test_write_session_state_stores_only_history_and_geometry(tmp_path: Path) -> None:
    legacy_values = {key: "obsolete" for key in session.LEGACY_SETTINGS_KEYS}
    settings = FakeSettings(legacy_values)
    prediction = tmp_path / "prediction"

    session.write_session_state(
        settings,
        session.SessionState(
            recent_predictions=(str(prediction),),
            recent_afdb_accessions=("Q5VSL9",),
            geometry=b"geometry",
        ),
    )

    assert settings.values == {
        session.SETTINGS_KEY_RECENT_PREDICTIONS: json.dumps(
            [str(prediction.absolute())], separators=(",", ":")
        ),
        session.SETTINGS_KEY_RECENT_AFDB_ACCESSIONS: json.dumps(
            ["Q5VSL9"], separators=(",", ":")
        ),
        session.SETTINGS_KEY_GEOMETRY: b"geometry",
    }
    assert settings.removed == list(session.LEGACY_SETTINGS_KEYS)
    assert settings.synced == 1


def test_write_without_geometry_preserves_existing_geometry() -> None:
    settings = FakeSettings({session.SETTINGS_KEY_GEOMETRY: b"existing"})
    session.write_session_state(settings, session.SessionState())
    assert settings.values[session.SETTINGS_KEY_GEOMETRY] == b"existing"


def test_write_accepts_settings_without_remove_or_sync(tmp_path: Path) -> None:
    settings = FakeSettingsWithoutOptionalMethods()
    prediction = tmp_path / "prediction"
    session.write_session_state(
        settings,
        session.SessionState(recent_predictions=(str(prediction),)),
    )
    assert json.loads(
        str(settings.values[session.SETTINGS_KEY_RECENT_PREDICTIONS])
    ) == [str(prediction.absolute())]
