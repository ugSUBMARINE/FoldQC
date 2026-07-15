"""Qt adapter for persisted FoldQC dialog state."""

from __future__ import annotations

from . import session
from .compat import QSettings, QtWidgets
from .gui_layout import GuiWidgets


class QtSessionAdapter:
    def __init__(self, dialog: QtWidgets.QDialog, widgets: GuiWidgets) -> None:
        self._dialog = dialog
        self._widgets = widgets
        self._restoring: bool = False

    @property
    def restoring(self) -> bool:
        return self._restoring

    def set_restoring(self, restoring: bool) -> None:
        self._restoring = bool(restoring)

    @staticmethod
    def _settings() -> QSettings:
        return QSettings(session.SETTINGS_ORGANIZATION, session.SETTINGS_APPLICATION)

    def restore(self) -> session.SessionState:
        return session.read_session_state(self._settings())

    def save(self) -> None:
        if self._restoring:
            return
        widgets = self._widgets
        geometry = (
            self._dialog.saveGeometry()
            if hasattr(self._dialog, "saveGeometry")
            else None
        )
        state = session.SessionState(
            path=widgets._dir_edit.text(),
            model_rank=widgets._model_combo.currentData(),
            metric_key=widgets._prop_combo.currentData() or "",
            target_name=widgets._obj_combo.currentText(),
            reference_text=widgets._ref_edit.text(),
            cutoff_text=widgets._cutoff_edit.text(),
            palette_key=str(widgets._palette_combo.currentData()),
            palette_reversed=bool(widgets._palette_reverse_chk.isChecked()),
            scale_min=widgets._vmin_edit.text(),
            scale_max=widgets._vmax_edit.text(),
            geometry=geometry,
        )
        session.write_session_state(self._settings(), state)
