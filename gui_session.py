"""Qt adapter for FoldQC prediction-history persistence."""

from __future__ import annotations

from pathlib import Path

from . import session
from .compat import QSettings, QtWidgets


class QtSessionAdapter:
    def __init__(self, dialog: QtWidgets.QDialog) -> None:
        self._dialog = dialog
        self._recent_predictions: tuple[str, ...] = ()

    @staticmethod
    def _settings() -> QSettings:
        return QSettings(session.SETTINGS_ORGANIZATION, session.SETTINGS_APPLICATION)

    def restore(self) -> session.SessionState:
        settings = self._settings()
        state = session.read_session_state(settings)
        self._recent_predictions = state.recent_predictions
        # Persist the migrated representation immediately and remove obsolete
        # control-state keys without ever loading the legacy path.
        session.write_session_state(settings, state)
        return state

    def _save(self) -> None:
        geometry = (
            self._dialog.saveGeometry()
            if hasattr(self._dialog, "saveGeometry")
            else None
        )
        session.write_session_state(
            self._settings(),
            session.SessionState(self._recent_predictions, geometry),
        )

    def record_recent_prediction(self, path: str | Path) -> tuple[str, ...]:
        updated = session.add_recent_prediction(self._recent_predictions, path)
        previous = self._recent_predictions
        self._recent_predictions = updated
        try:
            self._save()
        except Exception:
            self._recent_predictions = previous
            raise
        return self._recent_predictions

    def remove_recent_prediction(self, path: str | Path) -> tuple[str, ...]:
        updated = session.remove_recent_prediction(self._recent_predictions, path)
        previous = self._recent_predictions
        self._recent_predictions = updated
        try:
            self._save()
        except Exception:
            self._recent_predictions = previous
            raise
        return self._recent_predictions

    def save_geometry(self) -> None:
        self._save()
