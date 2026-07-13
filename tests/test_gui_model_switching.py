from __future__ import annotations

import sys
import tempfile
import traceback
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _install_fake_pymol() -> types.SimpleNamespace:
    class _QtSignal:
        def __init__(self) -> None:
            self.callbacks = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

        def emit(self, *args) -> None:
            for callback in list(self.callbacks):
                callback(*args)

    class _SignalDescriptor:
        def __set_name__(self, _owner, name: str) -> None:
            self.name = name

        def __get__(self, instance, _owner):
            if instance is None:
                return self
            return instance.__dict__.setdefault(self.name, _QtSignal())

    class _QObject:
        def __init__(self, _parent=None) -> None:
            pass

    class _QRunnable:
        def __init__(self) -> None:
            pass

    class _QThreadPool:
        def __init__(self) -> None:
            self.max_threads = None

        def setMaxThreadCount(self, count: int) -> None:
            self.max_threads = count

        def start(self, runnable) -> None:
            runnable.run()

    class _QTimer:
        delays = []

        @classmethod
        def singleShot(cls, delay: int, callback) -> None:
            cls.delays.append(delay)
            callback()

    flag = types.SimpleNamespace(
        AlignLeft=1,
        AlignRight=2,
        AlignCenter=4,
        AlignTop=8,
        AlignBottom=16,
        AlignVCenter=32,
        ItemIsEnabled=64,
        WindowCloseButtonHint=128,
    )
    orientation = types.SimpleNamespace(Horizontal=1, Vertical=2)
    scroll_policy = types.SimpleNamespace(ScrollBarAlwaysOff=1, ScrollBarAsNeeded=2)
    qt_core = types.SimpleNamespace(
        QT_VERSION_STR="6.0.0",
        Qt=types.SimpleNamespace(
            AlignmentFlag=flag,
            Orientation=orientation,
            ScrollBarPolicy=scroll_policy,
            ItemFlag=flag,
            WindowType=flag,
        ),
        QObject=_QObject,
        QRunnable=_QRunnable,
        QThreadPool=_QThreadPool,
        QTimer=_QTimer,
        pyqtSignal=lambda *_args: _SignalDescriptor(),
    )

    class _QSettings:
        _store: dict[tuple[str, str], dict[str, object]] = {}

        def __init__(self, organization: str, application: str) -> None:
            self._key = (organization, application)
            self._values = self._store.setdefault(self._key, {})

        def value(self, key: str, default=None):
            return self._values.get(key, default)

        def setValue(self, key: str, value) -> None:
            self._values[key] = value

        def sync(self) -> None:
            pass

    qt_core.QSettings = _QSettings

    class _QDialog:
        def __init__(self, _parent=None) -> None:
            self.title = ""
            self.modal = True
            self.minimum_width = None
            self.visible = False
            self.raised = False
            self.activated = False
            self.closed = False

        def setWindowTitle(self, title: str) -> None:
            self.title = title

        def setModal(self, modal: bool) -> None:
            self.modal = bool(modal)

        def setMinimumWidth(self, width: int) -> None:
            self.minimum_width = width

        def show(self) -> None:
            self.visible = True

        def hide(self) -> None:
            self.visible = False

        def close(self) -> None:
            self.closed = True
            self.visible = False

        def raise_(self) -> None:
            self.raised = True

        def activateWindow(self) -> None:
            self.activated = True

    class _QProgressDialog(_QDialog):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.label_text = ""
            self.value_range = None
            self.cancel_button = "default"
            self.auto_close = True
            self.auto_reset = True
            self.minimum_duration = None
            self.window_flags = {}
            self.reset_count = 0

        def setLabelText(self, text: str) -> None:
            self.label_text = text

        def setRange(self, minimum: int, maximum: int) -> None:
            self.value_range = (minimum, maximum)

        def setCancelButton(self, button) -> None:
            self.cancel_button = button

        def setAutoClose(self, enabled: bool) -> None:
            self.auto_close = bool(enabled)

        def setAutoReset(self, enabled: bool) -> None:
            self.auto_reset = bool(enabled)

        def setMinimumDuration(self, duration: int) -> None:
            self.minimum_duration = duration

        def setWindowFlag(self, flag_value, enabled: bool) -> None:
            self.window_flags[flag_value] = bool(enabled)

        def reset(self) -> None:
            self.reset_count += 1

    class _QFormLayout:
        FieldGrowthPolicy = types.SimpleNamespace(AllNonFixedFieldsGrow=1)

    class _VBoxLayout:
        def __init__(self, _parent=None) -> None:
            self.widgets = []

        def addWidget(self, widget) -> None:
            self.widgets.append(widget)

    class _PushButton:
        def __init__(self, text: str = "") -> None:
            self.text = text
            self.tooltip = ""
            self.fixed_width = None
            self.enabled = True
            self.clicked = _Signal()

        def setAutoDefault(self, _enabled: bool) -> None:
            pass

        def setDefault(self, _enabled: bool) -> None:
            pass

        def setFixedWidth(self, width: int) -> None:
            self.fixed_width = width

        def setToolTip(self, text: str) -> None:
            self.tooltip = text

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = bool(enabled)

        def isEnabled(self) -> bool:
            return self.enabled

    class _TextBrowser:
        def __init__(self) -> None:
            self.text = ""
            self.read_only = False
            self.open_external_links = False
            self.minimum_height = None

        def setPlainText(self, text: str) -> None:
            self.text = text

        def setReadOnly(self, read_only: bool) -> None:
            self.read_only = bool(read_only)

        def setOpenExternalLinks(self, enabled: bool) -> None:
            self.open_external_links = bool(enabled)

        def setMinimumHeight(self, height: int) -> None:
            self.minimum_height = height

    class _MessageBox:
        Yes = 1
        Cancel = 2
        StandardButton = types.SimpleNamespace(Yes=Yes, Cancel=Cancel)
        warnings: list[tuple[str, str]] = []
        criticals: list[tuple[str, str]] = []
        infos: list[tuple[str, str]] = []
        questions: list[tuple[str, str]] = []
        question_response = Yes

        @classmethod
        def warning(cls, _parent, title: str, message: str) -> None:
            cls.warnings.append((title, message))

        @classmethod
        def critical(cls, _parent, title: str, message: str) -> None:
            cls.criticals.append((title, message))

        @classmethod
        def information(cls, _parent, title: str, message: str) -> None:
            cls.infos.append((title, message))

        @classmethod
        def question(
            cls,
            _parent,
            title: str,
            message: str,
            _buttons=None,
            _default_button=None,
        ) -> int:
            cls.questions.append((title, message))
            return cls.question_response

    class _Signal:
        def __init__(self) -> None:
            self.callbacks = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

    class _Action:
        def __init__(self, text: str, _parent=None) -> None:
            self._text = text
            self._enabled = True
            self.tooltip = ""
            self.status_tip = ""
            self.triggered = _Signal()

        def text(self) -> str:
            return self._text

        def setEnabled(self, enabled: bool) -> None:
            self._enabled = bool(enabled)

        def isEnabled(self) -> bool:
            return self._enabled

        def setToolTip(self, text: str) -> None:
            self.tooltip = text

        def setStatusTip(self, text: str) -> None:
            self.status_tip = text

    class _Menu:
        def __init__(self, _parent=None) -> None:
            self.actions = []

        def addAction(self, action) -> None:
            self.actions.append(action)

    qt_widgets = types.SimpleNamespace(
        QDialog=_QDialog,
        QProgressDialog=_QProgressDialog,
        QFormLayout=_QFormLayout,
        QMessageBox=_MessageBox,
        QMenu=_Menu,
        QPushButton=_PushButton,
        QTextBrowser=_TextBrowser,
        QVBoxLayout=_VBoxLayout,
    )
    qt = types.SimpleNamespace(
        QtCore=qt_core,
        QtGui=types.SimpleNamespace(
            QAction=_Action,
            QIcon=types.SimpleNamespace(fromTheme=lambda _name: None),
        ),
        QtWidgets=qt_widgets,
    )
    pymol = types.SimpleNamespace(Qt=qt, cmd=None)
    sys.modules["pymol"] = pymol
    sys.modules["pymol.Qt"] = qt
    return pymol


_PYMOL = _install_fake_pymol()

from FoldQC import gui_jobs, metrics, session  # noqa: E402
from FoldQC.gui import (  # noqa: E402
    APP_TITLE,
    PREDICTION_FILE_FILTER,
    FoldQCPluginDialog,
    _PlotTarget,
)
from FoldQC.gui_state import GuiState  # noqa: E402
from FoldQC.token_map import TokenInfo, TokenMap  # noqa: E402


def _new_dialog() -> FoldQCPluginDialog:
    dialog = FoldQCPluginDialog.__new__(FoldQCPluginDialog)
    dialog._state = GuiState()
    return dialog


SETTINGS_KEY_CUTOFF = session.SETTINGS_KEY_CUTOFF
SETTINGS_KEY_GEOMETRY = session.SETTINGS_KEY_GEOMETRY
SETTINGS_KEY_METRIC = session.SETTINGS_KEY_METRIC
SETTINGS_KEY_MODEL_RANK = session.SETTINGS_KEY_MODEL_RANK
SETTINGS_KEY_PALETTE = session.SETTINGS_KEY_PALETTE
SETTINGS_KEY_PALETTE_REVERSE = session.SETTINGS_KEY_PALETTE_REVERSE
SETTINGS_KEY_PATH = session.SETTINGS_KEY_PATH
SETTINGS_KEY_REFERENCE = session.SETTINGS_KEY_REFERENCE
SETTINGS_KEY_SCALE_MAX = session.SETTINGS_KEY_SCALE_MAX
SETTINGS_KEY_SCALE_MIN = session.SETTINGS_KEY_SCALE_MIN
SETTINGS_KEY_TARGET = session.SETTINGS_KEY_TARGET


class _PredictionFiles:
    name = "target"
    models = [
        types.SimpleNamespace(
            rank=0,
            object_name="target_model_0",
            structure_path=Path("/tmp/target_model_0.cif"),
            display_label="model_0",
        ),
        types.SimpleNamespace(
            rank=1,
            object_name="target_model_1",
            structure_path=Path("/tmp/target_model_1.cif"),
            display_label="model_1",
        ),
    ]

    def structure_path(self, rank: int) -> Path:
        return Path(f"/tmp/target_model_{rank}.cif")


class _SessionPredictionFiles:
    def __init__(self, root: Path, ranks=(0, 1)) -> None:
        self._root = root
        self.name = "target"
        self.models = [
            types.SimpleNamespace(
                rank=rank,
                display_label=f"model_{rank}",
                object_name=f"target_model_{rank}",
            )
            for rank in ranks
        ]
        self.has_pae = False
        self.has_pde = False
        self.has_contact_probs = False
        self.has_plddt = False
        self.has_structure_plddt = True
        self.supports_ensemble = len(ranks) > 1

    def structure_path(self, rank: int) -> Path:
        return self._root / f"target_model_{rank}.cif"

    def model(self, rank: int):
        return next(model for model in self.models if model.rank == rank)


class _Discovery:
    def __init__(self, candidates, files_by_candidate, input_path=None) -> None:
        self.candidates = tuple(candidates)
        self.files_by_candidate = list(files_by_candidate)
        self.input_path = input_path
        self.scanned = []

    def scan(self, candidate):
        self.scanned.append(candidate)
        for item, files in self.files_by_candidate:
            if item is candidate:
                return files
        raise KeyError(candidate)


class _ImmediateJobHandle:
    def __init__(self) -> None:
        self.is_abandoned = False

    def abandon(self) -> None:
        self.is_abandoned = True


class _ImmediateJobRunner:
    def __init__(self) -> None:
        self.disposed = []

    def submit(
        self,
        request_id,
        task,
        on_progress,
        on_result,
        on_error,
    ):
        handle = _ImmediateJobHandle()

        def report(label: str) -> None:
            if not handle.is_abandoned:
                on_progress(request_id, label)

        try:
            result = task(report)
        except Exception as exc:
            if not handle.is_abandoned:
                on_error(
                    request_id,
                    types.SimpleNamespace(
                        message=str(exc) or type(exc).__name__,
                        traceback_text=traceback.format_exc(),
                    ),
                )
        else:
            if not handle.is_abandoned:
                on_result(request_id, result)
        return handle

    def dispose(self, value) -> None:
        self.disposed.append(value)


class _ManualJobRunner:
    def __init__(self) -> None:
        self.jobs = []
        self.disposed = []

    def submit(
        self,
        request_id,
        task,
        on_progress,
        on_result,
        on_error,
    ):
        handle = _ImmediateJobHandle()
        self.jobs.append((request_id, task, on_progress, on_result, on_error, handle))
        return handle

    def run_next(self) -> None:
        request_id, task, on_progress, on_result, on_error, handle = self.jobs.pop(0)

        def report(label: str) -> None:
            if not handle.is_abandoned:
                on_progress(request_id, label)

        try:
            result = task(report)
        except Exception as exc:
            if not handle.is_abandoned:
                on_error(
                    request_id,
                    types.SimpleNamespace(
                        message=str(exc) or type(exc).__name__,
                        traceback_text=traceback.format_exc(),
                    ),
                )
        else:
            if handle.is_abandoned:
                self.dispose(result)
            else:
                on_result(request_id, result)

    def dispose(self, value) -> None:
        self.disposed.append(value)


def _candidate(name: str, provider: str = "af3_server", path: Path | None = None):
    return types.SimpleNamespace(
        path=Path(f"/tmp/{name}") if path is None else path,
        provider=provider,
        provider_label="AlphaFold 3 Server",
        relative_path=name,
    )


class _CsvPredictionFiles:
    def __init__(self, root: Path, ranks=(0, 1)) -> None:
        self.name = "target"
        self.provider = "boltz"
        self.input_path = root
        self.pred_dir = root
        self.models = [
            types.SimpleNamespace(rank=rank, display_label=f"rank {rank}")
            for rank in ranks
        ]

    def model(self, rank: int):
        return next(model for model in self.models if model.rank == rank)


class _Cmd:
    def __init__(self, objects=(), enabled=()) -> None:
        self.objects = set(objects)
        self.enabled = set(enabled)
        self.loads: list[tuple[str, str, int, int]] = []
        self.enabled_calls: list[str] = []
        self.disabled_calls: list[str] = []
        self.selections: list[tuple[str, str]] = []
        self.show_calls: list[tuple[str, str]] = []
        self.zoom_calls: list[str] = []
        self.refresh_calls = 0
        self.selection_models = {}
        self.get_model_calls = 0

    def get_names(self, _kind: str, *args, **kwargs):
        enabled_only = kwargs.get("enabled_only") or (args and args[0] == 1)
        return sorted(self.enabled if enabled_only else self.objects)

    def get_model(self, selection: str):
        self.get_model_calls += 1
        return self.selection_models.get(selection, types.SimpleNamespace(atom=[]))

    def index(self, selection: str):
        model = self.selection_models.get(selection, types.SimpleNamespace(atom=[]))
        return [
            (selection, int(getattr(atom, "index", fallback)))
            for fallback, atom in enumerate(model.atom, start=1)
        ]

    def load(self, path: str, obj_name: str, quiet: int = 1, zoom: int = 0) -> None:
        self.loads.append((path, obj_name, quiet, zoom))
        self.objects.add(obj_name)
        self.enabled.add(obj_name)

    def enable(self, obj_name: str) -> None:
        self.enabled_calls.append(obj_name)
        self.enabled.add(obj_name)

    def disable(self, obj_name: str) -> None:
        self.disabled_calls.append(obj_name)
        self.enabled.discard(obj_name)

    def select(self, name: str, expression: str) -> None:
        self.selections.append((name, expression))

    def show(self, representation: str, selection: str) -> None:
        self.show_calls.append((representation, selection))

    def zoom(self, selection: str) -> None:
        self.zoom_calls.append(selection)

    def refresh(self) -> None:
        self.refresh_calls += 1


class _TextBox:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def setPlainText(self, text: str) -> None:
        self.text = text


class _Label:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.tooltip = ""
        self.enabled = True
        self.word_wrap = False
        self.minimum_height = None
        self.alignment = None
        self.line_spacing = 14

    def setText(self, text: str) -> None:
        self.text = text

    def setToolTip(self, text: str) -> None:
        self.tooltip = text

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def setVisible(self, _visible: bool) -> None:
        pass

    def setWordWrap(self, enabled: bool) -> None:
        self.word_wrap = bool(enabled)

    def fontMetrics(self):
        return types.SimpleNamespace(lineSpacing=lambda: self.line_spacing)

    def setMinimumHeight(self, height: int) -> None:
        self.minimum_height = height

    def setAlignment(self, alignment) -> None:
        self.alignment = alignment


class _LineEdit:
    def __init__(self, text: str = "") -> None:
        self._text = text
        self.tooltip = ""
        self.enabled = True

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text

    def setToolTip(self, text: str) -> None:
        self.tooltip = text

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self.enabled


class _CheckBox:
    def __init__(self, checked: bool = False) -> None:
        self._checked = bool(checked)
        self.enabled = True

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        self._checked = bool(checked)

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


class _ComboItem:
    def __init__(self, flags: int) -> None:
        self._flags = flags

    def flags(self) -> int:
        return self._flags

    def setFlags(self, flags: int) -> None:
        self._flags = flags


class _ComboModel:
    def __init__(self, count: int, flags: int) -> None:
        self.flags = flags
        self.items = [_ComboItem(flags) for _ in range(count)]

    def item(self, row: int):
        return self.items[row]

    def append(self) -> None:
        self.items.append(_ComboItem(self.flags))


class _Combo:
    def __init__(self, count: int, flags: int) -> None:
        self._model = _ComboModel(count, flags)
        self._texts = ["" for _ in range(count)]
        self._data = [None for _ in range(count)]
        self._current = 0
        self.enabled = True

    def model(self):
        return self._model

    def count(self) -> int:
        return len(self._model.items)

    def addItem(self, text: str, data=None) -> None:
        self._model.append()
        self._texts.append(text)
        self._data.append(data)

    def clear(self) -> None:
        self._model.items.clear()
        self._texts.clear()
        self._data.clear()
        self._current = 0

    def blockSignals(self, _blocked: bool) -> None:
        pass

    def itemText(self, row: int) -> str:
        return self._texts[row]

    def itemData(self, row: int):
        return self._data[row]

    def currentData(self):
        return self._data[self._current]

    def currentText(self) -> str:
        return self._texts[self._current]

    def currentIndex(self) -> int:
        return self._current

    def setCurrentIndex(self, row: int) -> None:
        self._current = row

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self.enabled


def _set_metric_combo(dialog, flags: int) -> None:
    dialog._prop_combo = _Combo(len(metrics.PROPERTIES), flags)
    dialog._prop_combo_rows = {
        prop["key"]: row for row, prop in enumerate(metrics.PROPERTIES)
    }


def _token(
    idx: int,
    *,
    chain_id: str = "A",
    res_num: int | None = None,
    is_hetatm: bool = False,
) -> TokenInfo:
    return TokenInfo(
        token_idx=idx,
        chain_id=chain_id,
        res_num=idx + 1 if res_num is None else res_num,
        res_name="LIG" if is_hetatm else "ALA",
        is_hetatm=is_hetatm,
        atom_name=f"C{idx}" if is_hetatm else None,
    )


def _token_map(*tokens: TokenInfo) -> TokenMap:
    return TokenMap(tokens)


def _atom(
    chain: str,
    resi: int | str,
    resn: str = "ALA",
    *,
    hetatm: bool = False,
    name: str = "CA",
):
    return types.SimpleNamespace(
        chain=chain,
        resi=str(resi),
        resn=resn,
        hetatm=hetatm,
        name=name,
    )


def _dialog_with(cmd: _Cmd):
    _PYMOL.cmd = cmd
    dialog = _new_dialog()
    dialog._pred_files = _PredictionFiles()
    dialog.refreshed = 0
    dialog.selected: list[str] = []
    dialog.painted: list[tuple[str, str]] = []
    dialog._refresh_objects = lambda: setattr(dialog, "refreshed", dialog.refreshed + 1)
    dialog._select_object = lambda obj_name: dialog.selected.append(obj_name)
    dialog._apply_plddt_class_coloring = lambda key, obj_name: dialog.painted.append(
        (key, obj_name)
    )
    return dialog


class QtJobRunnerTests(unittest.TestCase):
    def test_runner_reports_progress_and_result_in_order(self) -> None:
        events = []
        runner = gui_jobs.QtJobRunner()

        runner.submit(
            7,
            lambda report: (report("phase one"), report("phase two"), "done")[-1],
            lambda request_id, label: events.append(("progress", request_id, label)),
            lambda request_id, result: events.append(("result", request_id, result)),
            lambda request_id, failure: events.append(
                ("error", request_id, failure.message)
            ),
        )

        self.assertEqual(
            events,
            [
                ("progress", 7, "phase one"),
                ("progress", 7, "phase two"),
                ("result", 7, "done"),
            ],
        )

    def test_runner_transports_exception_message_and_traceback(self) -> None:
        failures = []
        runner = gui_jobs.QtJobRunner()

        def fail(_report):
            raise ValueError("broken job")

        runner.submit(
            3,
            fail,
            lambda *_args: None,
            lambda *_args: self.fail("Failure must not emit a result"),
            lambda request_id, failure: failures.append((request_id, failure)),
        )

        self.assertEqual(failures[0][0], 3)
        self.assertEqual(failures[0][1].message, "broken job")
        self.assertIn("ValueError: broken job", failures[0][1].traceback_text)

    def test_abandoned_job_suppresses_progress_and_result(self) -> None:
        queued = []
        events = []
        runner = gui_jobs.QtJobRunner()

        with mock.patch.object(gui_jobs._POOL, "start", side_effect=queued.append):
            handle = runner.submit(
                11,
                lambda report: (report("late phase"), "late result")[-1],
                lambda *_args: events.append("progress"),
                lambda *_args: events.append("result"),
                lambda *_args: events.append("error"),
            )

        handle.abandon()
        queued[0].run()
        self.assertEqual(events, [])

    def test_dispose_releases_value_in_discard_runnable(self) -> None:
        queued = []
        cleanup_calls = []
        temporary = types.SimpleNamespace(
            cleanup=lambda: cleanup_calls.append("cleaned")
        )
        pred_files = types.SimpleNamespace(_temporary_directory=temporary)
        value = types.SimpleNamespace(pred_files=pred_files)
        runner = gui_jobs.QtJobRunner()

        with mock.patch.object(gui_jobs._POOL, "start", side_effect=queued.append):
            runner.dispose(value)

        self.assertIs(queued[0]._value, value)
        queued[0].run()
        self.assertEqual(cleanup_calls, ["cleaned"])
        self.assertIsNone(queued[0]._value)


class GuiModelSwitchingTests(unittest.TestCase):
    def setUp(self) -> None:
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.warnings.clear()
        msg.criticals.clear()
        msg.infos.clear()
        msg.questions.clear()
        msg.question_response = msg.Yes
        _PYMOL.Qt.QtCore.QSettings._store.clear()

    def _settings(self):
        dialog = _new_dialog()
        return dialog._settings()

    def _session_dialog(self):
        enabled = _PYMOL.Qt.QtCore.Qt.ItemFlag.ItemIsEnabled
        dialog = _new_dialog()
        dialog._pred_files = None
        dialog._pred_data = None
        dialog._token_map = None
        dialog._ensemble_members = None
        dialog._ensemble_group_name = None
        dialog._ensemble_aligned = False
        dialog._ensemble_rmsd = None
        dialog._ensemble_plddt_mean = None
        dialog._ensemble_plddt_std = None
        dialog._plot_windows = []
        dialog._loading_prediction = False
        dialog._prediction_load_request_id = 0
        dialog._restoring_settings = False
        dialog._pending_session_restore = types.SimpleNamespace(
            model_rank=None,
            metric_key=None,
            target_name=None,
        )
        dialog._dir_edit = _LineEdit()
        dialog._model_combo = _Combo(0, enabled)
        dialog._obj_combo = _Combo(0, enabled)
        dialog._prop_combo = _Combo(0, enabled)
        dialog._populate_property_combo()
        dialog._ref_label = _Label("Reference:")
        dialog._ref_edit = _LineEdit()
        dialog._cutoff_label = _Label("Cutoff (Å):")
        dialog._cutoff_edit = _LineEdit("5.0")
        dialog._preview_label = _Label("")
        dialog._plot_actions = {}
        dialog._conf_browser = _TextBox()
        dialog._stats_browser = _TextBox()
        dialog._palette_combo = _Combo(0, enabled)
        dialog._palette_combo.addItem("Viridis", "viridis")
        dialog._palette_combo.addItem("Blues", "white_blue")
        dialog._palette_reverse_chk = _CheckBox()
        dialog._vmin_edit = _LineEdit()
        dialog._vmax_edit = _LineEdit()
        dialog._dir_btn = _LineEdit()
        dialog._file_btn = _LineEdit()
        dialog._obj_refresh_btn = _LineEdit()
        dialog._apply_btn = _LineEdit()
        dialog._plot_btn = _LineEdit()
        dialog._export_csv_btn = _LineEdit()
        dialog._ensemble_btn = _LineEdit()
        dialog._guide_btn = _LineEdit()
        dialog._close_btn = _LineEdit()
        dialog._job_runner = _ImmediateJobRunner()
        dialog._active_load_handle = None
        dialog._load_progress_dialog = None
        dialog._progress_show_generation = 0
        dialog.setWindowTitle = lambda _title: None
        return dialog

    def _write_session_settings(self, **values) -> None:
        settings = self._settings()
        for key, value in values.items():
            settings.setValue(key, value)

    def _with_fake_plots(self, fake_plots, callback):
        import FoldQC

        old_plots = sys.modules.get("FoldQC.plots")
        old_plots_attr = getattr(FoldQC, "plots", None)
        sys.modules["FoldQC.plots"] = fake_plots
        FoldQC.plots = fake_plots
        try:
            return callback()
        finally:
            if old_plots_attr is None:
                try:
                    delattr(FoldQC, "plots")
                except AttributeError:
                    pass
            else:
                FoldQC.plots = old_plots_attr
            if old_plots is None:
                sys.modules.pop("FoldQC.plots", None)
            else:
                sys.modules["FoldQC.plots"] = old_plots

    def test_app_title_uses_foldqc_display_name(self) -> None:
        self.assertEqual(APP_TITLE, "FoldQC")

    def test_plot_type_choices_include_ensemble_site_summary(self) -> None:
        self.assertEqual(
            metrics.PLOT_TYPES,
            [
                ("Line", "line"),
                ("Distribution", "distribution"),
                ("Matrix", "matrix"),
                ("PAE summary", "pae_summary"),
                ("PDE summary", "pde_summary"),
                ("Binding-site fingerprint", "binding_site_fingerprint"),
                ("Ensemble site summary", "ensemble_site_summary"),
            ],
        )

    def test_restore_session_loads_existing_path_rank_with_lazy_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _SessionPredictionFiles(root, ranks=(0, 1))
            self._write_session_settings(
                **{
                    SETTINGS_KEY_PATH: str(root),
                    SETTINGS_KEY_MODEL_RANK: 1,
                    SETTINGS_KEY_METRIC: "plddt",
                    SETTINGS_KEY_TARGET: "target_model_1",
                }
            )
            dialog = self._session_dialog()
            _PYMOL.cmd = _Cmd(
                objects={"target_model_0", "target_model_1"},
                enabled={"target_model_0", "target_model_1"},
            )
            load_calls = []

            def fake_load_prediction_data(pred_files, rank, **kwargs):
                load_calls.append((pred_files, rank, kwargs))
                return types.SimpleNamespace(
                    provider="structure_only",
                    rank=rank,
                    structure_path=pred_files.structure_path(rank),
                    structure_plddt=np.array([0.8], dtype=np.float32),
                    plddt=None,
                    confidence=None,
                    summary_confidence=None,
                    pae=None,
                    pde=None,
                    contact_probs=None,
                )

            candidate = _candidate("target")
            discovery = _Discovery([candidate], [(candidate, files)])
            with (
                mock.patch(
                    "FoldQC.loader.discover_prediction_candidates",
                    return_value=discovery,
                ) as discover,
                mock.patch(
                    "FoldQC.loader.load_prediction_data",
                    side_effect=fake_load_prediction_data,
                ),
            ):
                dialog._restore_session_settings()

            discover.assert_called_once_with(str(root))
            self.assertEqual(dialog._model_combo.currentData(), 1)
            self.assertEqual(dialog._obj_combo.currentText(), "target_model_1")
            self.assertEqual(load_calls[0][1], 1)
            self.assertEqual(
                load_calls[0][2],
                {
                    "load_pae": False,
                    "load_pde": False,
                    "load_contact_probs": False,
                },
            )

    def test_restore_session_skips_missing_path_without_warning(self) -> None:
        missing = "/tmp/foldqc-session-memory-missing-path"
        self._write_session_settings(**{SETTINGS_KEY_PATH: missing})
        dialog = self._session_dialog()
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        with mock.patch("FoldQC.loader.discover_prediction_candidates") as discover:
            dialog._restore_session_settings()

        discover.assert_not_called()
        self.assertEqual(dialog._dir_edit.text(), missing)
        self.assertIsNone(dialog._pred_files)
        self.assertEqual(msg.warnings, [])

    def test_restore_session_restores_controls_without_valid_path(self) -> None:
        self._write_session_settings(
            **{
                SETTINGS_KEY_PATH: "/tmp/foldqc-session-memory-missing-path",
                SETTINGS_KEY_METRIC: "pae_to_sel",
                SETTINGS_KEY_REFERENCE: "resname LIG",
                SETTINGS_KEY_CUTOFF: "7.5",
                SETTINGS_KEY_PALETTE: "white_blue",
                SETTINGS_KEY_PALETTE_REVERSE: True,
                SETTINGS_KEY_SCALE_MIN: "0.1",
                SETTINGS_KEY_SCALE_MAX: "0.9",
            }
        )
        dialog = self._session_dialog()

        dialog._restore_session_settings()

        self.assertEqual(dialog._prop_combo.currentData(), "pae_to_sel")
        self.assertEqual(dialog._palette_combo.currentData(), "white_blue")
        self.assertTrue(dialog._palette_reverse_chk.isChecked())
        self.assertEqual(dialog._ref_edit.text(), "resname LIG")
        self.assertEqual(dialog._cutoff_edit.text(), "7.5")
        self.assertEqual(dialog._vmin_edit.text(), "0.1")
        self.assertEqual(dialog._vmax_edit.text(), "0.9")

    def test_browse_file_filter_includes_supported_archives(self) -> None:
        dialog = _new_dialog()
        dialog._dir_edit = _LineEdit("/tmp")
        dialog._load_prediction_dir = lambda: None
        dialog._raise_after_native_dialog = lambda: (_ for _ in ()).throw(
            AssertionError("Open dialog should return a path")
        )
        captured = {}

        def get_open_file_name(*args):
            captured["title"] = args[1]
            captured["filter"] = args[3]
            return ("/tmp/archive.tar.gz", "")

        old_dialog = getattr(_PYMOL.Qt.QtWidgets, "QFileDialog", None)
        _PYMOL.Qt.QtWidgets.QFileDialog = types.SimpleNamespace(
            getOpenFileName=get_open_file_name
        )
        try:
            dialog._browse_file()
        finally:
            if old_dialog is None:
                delattr(_PYMOL.Qt.QtWidgets, "QFileDialog")
            else:
                _PYMOL.Qt.QtWidgets.QFileDialog = old_dialog

        self.assertEqual(dialog._dir_edit.text(), "/tmp/archive.tar.gz")
        self.assertIn("archive", captured["title"])
        self.assertEqual(captured["filter"], PREDICTION_FILE_FILTER)
        self.assertIn("*.tar", captured["filter"])
        self.assertIn("*.tar.gz", captured["filter"])
        self.assertIn("*.tgz", captured["filter"])

    def test_restore_target_selects_only_present_pymol_object(self) -> None:
        dialog = self._session_dialog()
        dialog._pending_session_restore.target_name = "target_model_1"
        _PYMOL.cmd = _Cmd(
            objects={"target_model_0", "target_model_1"},
            enabled={"target_model_0", "target_model_1"},
        )

        dialog._refresh_objects()

        self.assertEqual(dialog._obj_combo.currentText(), "target_model_1")
        self.assertIsNone(dialog._pending_session_restore.target_name)

        dialog = self._session_dialog()
        dialog._pending_session_restore.target_name = "missing_target"
        _PYMOL.cmd = _Cmd(objects={"target_model_0"}, enabled={"target_model_0"})

        dialog._refresh_objects()

        self.assertEqual(dialog._obj_combo.currentText(), "target_model_0")
        self.assertEqual(dialog._pending_session_restore.target_name, "missing_target")

    def test_refresh_objects_excludes_foldqc_colorbar(self) -> None:
        dialog = self._session_dialog()
        _PYMOL.cmd = _Cmd(
            objects={"target_model_0", "foldqc_colorbar", "other_model"},
            enabled={"target_model_0", "foldqc_colorbar", "other_model"},
        )

        dialog._refresh_objects()

        names = [
            dialog._obj_combo.itemText(row) for row in range(dialog._obj_combo.count())
        ]
        self.assertEqual(names, ["other_model", "target_model_0"])
        self.assertNotIn("foldqc_colorbar", names)

    def test_restore_model_rank_ignores_missing_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _SessionPredictionFiles(root, ranks=(0,))
            self._write_session_settings(
                **{
                    SETTINGS_KEY_PATH: str(root),
                    SETTINGS_KEY_MODEL_RANK: 9,
                }
            )
            dialog = self._session_dialog()
            _PYMOL.cmd = _Cmd(objects={"target_model_0"}, enabled={"target_model_0"})
            loaded_ranks = []

            def fake_load_prediction_data(pred_files, rank, **_kwargs):
                loaded_ranks.append(rank)
                return types.SimpleNamespace(
                    provider="structure_only",
                    rank=rank,
                    structure_path=pred_files.structure_path(rank),
                    structure_plddt=np.array([0.8], dtype=np.float32),
                    plddt=None,
                    confidence=None,
                    summary_confidence=None,
                    pae=None,
                    pde=None,
                    contact_probs=None,
                )

            candidate = _candidate("target")
            discovery = _Discovery([candidate], [(candidate, files)])
            with (
                mock.patch(
                    "FoldQC.loader.discover_prediction_candidates",
                    return_value=discovery,
                ),
                mock.patch(
                    "FoldQC.loader.load_prediction_data",
                    side_effect=fake_load_prediction_data,
                ),
            ):
                dialog._restore_session_settings()

            self.assertEqual(dialog._model_combo.currentData(), 0)
            self.assertEqual(loaded_ranks, [0])

    def test_load_prediction_dir_single_candidate_loads_without_chooser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _SessionPredictionFiles(root, ranks=(0,))
            candidate = _candidate("target", path=root / "target")
            discovery = _Discovery([candidate], [(candidate, files)])
            dialog = self._session_dialog()
            dialog._dir_edit.setText(str(root))
            dialog._choose_prediction_candidate = lambda _candidates: (
                _ for _ in ()
            ).throw(AssertionError("Chooser should not be shown for one candidate"))
            _PYMOL.cmd = _Cmd()

            with (
                mock.patch(
                    "FoldQC.loader.discover_prediction_candidates",
                    return_value=discovery,
                ),
                mock.patch(
                    "FoldQC.loader.load_prediction_data",
                    return_value=types.SimpleNamespace(
                        provider="structure_only",
                        rank=0,
                        structure_path=files.structure_path(0),
                        structure_plddt=np.array([0.8], dtype=np.float32),
                        plddt=None,
                        confidence=None,
                        summary_confidence=None,
                        pae=None,
                        pde=None,
                        contact_probs=None,
                    ),
                ),
            ):
                dialog._load_prediction_dir()

        self.assertIs(dialog._pred_files, files)
        self.assertEqual(discovery.scanned, [candidate])
        self.assertEqual(dialog._model_combo.currentData(), 0)
        self.assertEqual(dialog._dir_edit.text(), str(root / "target"))
        self.assertEqual(
            _PYMOL.cmd.loads,
            [(str(files.structure_path(0)), "target_model_0", 1, 1)],
        )

    def test_load_prediction_dir_keeps_archive_input_path_in_text_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive.tar.gz"
            archive.write_text("archive")
            extracted = root / "foldqc_archive_x" / "archive" / "prediction"
            files = _SessionPredictionFiles(extracted, ranks=(0,))
            candidate = _candidate("prediction", provider="boltz_api", path=extracted)
            discovery = _Discovery(
                [candidate], [(candidate, files)], input_path=archive
            )
            dialog = self._session_dialog()
            dialog._dir_edit.setText(str(archive))
            dialog._choose_prediction_candidate = lambda _candidates: (
                _ for _ in ()
            ).throw(AssertionError("Chooser should not be shown for one candidate"))
            _PYMOL.cmd = _Cmd()

            with (
                mock.patch(
                    "FoldQC.loader.discover_prediction_candidates",
                    return_value=discovery,
                ),
                mock.patch(
                    "FoldQC.loader.load_prediction_data",
                    return_value=types.SimpleNamespace(
                        provider="boltz_api",
                        rank=0,
                        structure_path=files.structure_path(0),
                        structure_plddt=np.array([0.8], dtype=np.float32),
                        plddt=None,
                        confidence=None,
                        summary_confidence=None,
                        pae=None,
                        pde=None,
                        contact_probs=None,
                    ),
                ),
            ):
                dialog._load_prediction_dir()

        self.assertIs(dialog._pred_files, files)
        self.assertEqual(discovery.scanned, [candidate])
        self.assertEqual(dialog._dir_edit.text(), str(archive))

    def test_load_prediction_dir_keeps_single_structure_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = root / "model.cif"
            structure.write_text("data")
            files = _SessionPredictionFiles(root, ranks=(0,))
            candidate = _candidate(
                "model.cif", provider="structure_only", path=structure
            )
            discovery = _Discovery(
                [candidate],
                [(candidate, files)],
                input_path=structure,
            )
            dialog = self._session_dialog()
            dialog._dir_edit.setText(str(structure))
            _PYMOL.cmd = _Cmd()

            with (
                mock.patch(
                    "FoldQC.loader.discover_prediction_candidates",
                    return_value=discovery,
                ),
                mock.patch(
                    "FoldQC.loader.load_prediction_data",
                    return_value=types.SimpleNamespace(
                        provider="structure_only",
                        rank=0,
                        structure_path=files.structure_path(0),
                        structure_plddt=np.array([0.8], dtype=np.float32),
                        plddt=None,
                        confidence=None,
                        summary_confidence=None,
                        pae=None,
                        pde=None,
                        contact_probs=None,
                    ),
                ),
            ):
                dialog._load_prediction_dir()

        self.assertIs(dialog._pred_files, files)
        self.assertEqual(discovery.scanned, [candidate])
        self.assertEqual(dialog._dir_edit.text(), str(structure))

    def test_load_prediction_dir_multiple_candidates_loads_selected_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files_a = _SessionPredictionFiles(root / "a", ranks=(0,))
            files_b = _SessionPredictionFiles(root / "b", ranks=(0,))
            candidate_a = _candidate("a_job", path=root / "a_job")
            candidate_b = _candidate("b_job", path=root / "b_job")
            discovery = _Discovery(
                [candidate_a, candidate_b],
                [(candidate_a, files_a), (candidate_b, files_b)],
            )
            dialog = self._session_dialog()
            dialog._dir_edit.setText(str(root))
            chooser_calls = []
            dialog._choose_prediction_candidate = lambda candidates: (
                chooser_calls.append(tuple(candidates)) or candidates[1]
            )
            _PYMOL.cmd = _Cmd()

            with (
                mock.patch(
                    "FoldQC.loader.discover_prediction_candidates",
                    return_value=discovery,
                ),
                mock.patch(
                    "FoldQC.loader.load_prediction_data",
                    return_value=types.SimpleNamespace(
                        provider="structure_only",
                        rank=0,
                        structure_path=files_b.structure_path(0),
                        structure_plddt=np.array([0.8], dtype=np.float32),
                        plddt=None,
                        confidence=None,
                        summary_confidence=None,
                        pae=None,
                        pde=None,
                        contact_probs=None,
                    ),
                ),
            ):
                dialog._load_prediction_dir()

        self.assertEqual(chooser_calls, [(candidate_a, candidate_b)])
        self.assertEqual(discovery.scanned, [candidate_b])
        self.assertIs(dialog._pred_files, files_b)
        self.assertEqual(dialog._dir_edit.text(), str(root / "b_job"))

    def test_load_prediction_dir_cancel_keeps_previous_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = _SessionPredictionFiles(root / "previous", ranks=(0,))
            files_a = _SessionPredictionFiles(root / "a", ranks=(0,))
            candidate_a = _candidate("a_job", path=root / "a_job")
            candidate_b = _candidate("b_job", path=root / "b_job")
            discovery = _Discovery([candidate_a, candidate_b], [(candidate_a, files_a)])
            dialog = self._session_dialog()
            dialog._dir_edit.setText(str(root))
            dialog._pred_files = previous
            dialog._choose_prediction_candidate = lambda _candidates: None

            with mock.patch(
                "FoldQC.loader.discover_prediction_candidates",
                return_value=discovery,
            ):
                dialog._load_prediction_dir()

        self.assertIs(dialog._pred_files, previous)
        self.assertEqual(discovery.scanned, [])
        self.assertEqual(dialog._dir_edit.text(), str(root))
        self.assertEqual(dialog._job_runner.disposed, [discovery])

    def test_load_prediction_dir_discovery_error_shows_warning(self) -> None:
        dialog = self._session_dialog()
        dialog._dir_edit.setText("/tmp/foldqc-bad-prediction")
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        with mock.patch(
            "FoldQC.loader.discover_prediction_candidates",
            side_effect=ValueError("no prediction here"),
        ):
            dialog._load_prediction_dir()

        self.assertEqual(msg.warnings, [(APP_TITLE, "no prediction here")])
        self.assertFalse(dialog._loading_prediction)
        self.assertTrue(dialog._dir_edit.isEnabled())

    def test_prediction_loading_disables_conflicting_controls(self) -> None:
        dialog = self._session_dialog()
        dialog._dir_edit.setText("/tmp/prediction")
        dialog._job_runner = _ManualJobRunner()

        dialog._load_prediction_dir()

        self.assertTrue(dialog._loading_prediction)
        self.assertFalse(dialog._dir_edit.isEnabled())
        self.assertFalse(dialog._model_combo.isEnabled())
        self.assertFalse(dialog._apply_btn.isEnabled())
        self.assertTrue(dialog._close_btn.isEnabled())
        self.assertTrue(dialog._guide_btn.isEnabled())

    def test_progress_dialog_is_modeless_indeterminate_and_delayed(self) -> None:
        dialog = self._session_dialog()
        dialog._loading_prediction = True
        dialog._prediction_load_request_id = 4
        _PYMOL.Qt.QtCore.QTimer.delays.clear()

        dialog._schedule_load_progress(4, "Discovering prediction folders…")

        progress = dialog._load_progress_dialog
        self.assertFalse(progress.modal)
        self.assertEqual(progress.value_range, (0, 0))
        self.assertIsNone(progress.cancel_button)
        self.assertFalse(progress.auto_close)
        self.assertFalse(progress.auto_reset)
        self.assertEqual(progress.minimum_duration, 0)
        self.assertEqual(_PYMOL.Qt.QtCore.QTimer.delays, [300])
        self.assertFalse(
            progress.window_flags[_PYMOL.Qt.QtCore.Qt.WindowType.WindowCloseButtonHint]
        )

    def test_initial_provider_load_error_preserves_previous_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = _SessionPredictionFiles(root / "previous", ranks=(0,))
            files = _SessionPredictionFiles(root / "new", ranks=(0,))
            candidate = _candidate("new", path=root / "new")
            discovery = _Discovery([candidate], [(candidate, files)])
            dialog = self._session_dialog()
            dialog._pred_files = previous
            dialog._dir_edit.setText(str(root))
            msg = _PYMOL.Qt.QtWidgets.QMessageBox

            with (
                mock.patch(
                    "FoldQC.loader.discover_prediction_candidates",
                    return_value=discovery,
                ),
                mock.patch(
                    "FoldQC.loader.load_prediction_data",
                    side_effect=ValueError("could not read confidence"),
                ),
            ):
                dialog._load_prediction_dir()

        self.assertIs(dialog._pred_files, previous)
        self.assertFalse(dialog._loading_prediction)
        self.assertEqual(
            msg.warnings,
            [(APP_TITLE, "could not read confidence")],
        )

    def test_initial_pymol_load_error_preserves_previous_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = _SessionPredictionFiles(root / "previous", ranks=(0,))
            previous_data = object()
            previous_members = [object()]
            files = _SessionPredictionFiles(root / "new", ranks=(0,))
            candidate = _candidate("new", path=root / "new")
            discovery = _Discovery([candidate], [(candidate, files)])
            loaded_data = types.SimpleNamespace(
                provider="structure_only",
                rank=0,
                structure_path=files.structure_path(0),
                structure_plddt=np.array([0.8], dtype=np.float32),
                plddt=None,
                confidence=None,
                summary_confidence=None,
                pae=None,
                pde=None,
                contact_probs=None,
            )
            dialog = self._session_dialog()
            dialog._pred_files = previous
            dialog._pred_data = previous_data
            dialog._ensemble_members = previous_members
            dialog._ensemble_group_name = "previous_ensemble"
            dialog._model_combo.addItem("previous model", 7)
            dialog._model_combo.setCurrentIndex(0)
            dialog._dir_edit.setText(str(root))
            msg = _PYMOL.Qt.QtWidgets.QMessageBox

            with (
                mock.patch(
                    "FoldQC.loader.discover_prediction_candidates",
                    return_value=discovery,
                ),
                mock.patch(
                    "FoldQC.loader.load_prediction_data",
                    return_value=loaded_data,
                ),
                mock.patch(
                    "FoldQC.gui_loading.ensure_structure_object",
                    side_effect=RuntimeError("viewer unavailable"),
                ),
            ):
                dialog._load_prediction_dir()

        self.assertIs(dialog._pred_files, previous)
        self.assertIs(dialog._pred_data, previous_data)
        self.assertIs(dialog._ensemble_members, previous_members)
        self.assertEqual(dialog._ensemble_group_name, "previous_ensemble")
        self.assertEqual(dialog._model_combo.currentData(), 7)
        self.assertEqual(dialog._dir_edit.text(), str(root))
        self.assertFalse(dialog._loading_prediction)
        self.assertEqual(len(dialog._job_runner.disposed), 1)
        self.assertIs(dialog._job_runner.disposed[0].pred_files, files)
        self.assertEqual(
            msg.warnings,
            [
                (
                    APP_TITLE,
                    "Could not load or show target_model_0.cif:\nviewer unavailable",
                )
            ],
        )

    def test_close_abandons_late_discovery_without_pymol_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _SessionPredictionFiles(root / "new", ranks=(0,))
            candidate = _candidate("new", path=root / "new")
            discovery = _Discovery([candidate], [(candidate, files)])
            dialog = self._session_dialog()
            dialog._dir_edit.setText(str(root))
            dialog._job_runner = _ManualJobRunner()
            _PYMOL.cmd = _Cmd()

            with mock.patch(
                "FoldQC.loader.discover_prediction_candidates",
                return_value=discovery,
            ):
                dialog._load_prediction_dir()
                request_id = dialog._prediction_load_request_id
                dialog.closeEvent(object())
                dialog._job_runner.run_next()

        self.assertGreater(dialog._prediction_load_request_id, request_id)
        self.assertIsNone(dialog._pred_files)
        self.assertEqual(_PYMOL.cmd.loads, [])
        self.assertEqual(dialog._job_runner.disposed, [discovery])

    def test_close_event_saves_session_settings(self) -> None:
        dialog = self._session_dialog()
        dialog._dir_edit.setText("/tmp/prediction")
        dialog._model_combo.addItem("model_2", 2)
        dialog._model_combo.setCurrentIndex(0)
        dialog._obj_combo.addItem("target_model_2")
        dialog._obj_combo.setCurrentIndex(0)
        dialog._select_property("pde_contact")
        dialog._ref_edit.setText("chain A")
        dialog._cutoff_edit.setText("8.0")
        dialog._palette_combo.setCurrentIndex(1)
        dialog._palette_reverse_chk.setChecked(True)
        dialog._vmin_edit.setText("1")
        dialog._vmax_edit.setText("10")
        dialog.saveGeometry = lambda: b"geometry-bytes"

        dialog.closeEvent(object())

        values = self._settings()._values
        self.assertEqual(values[SETTINGS_KEY_PATH], "/tmp/prediction")
        self.assertEqual(values[SETTINGS_KEY_MODEL_RANK], 2)
        self.assertEqual(values[SETTINGS_KEY_METRIC], "pde_contact")
        self.assertEqual(values[SETTINGS_KEY_TARGET], "target_model_2")
        self.assertEqual(values[SETTINGS_KEY_REFERENCE], "chain A")
        self.assertEqual(values[SETTINGS_KEY_CUTOFF], "8.0")
        self.assertEqual(values[SETTINGS_KEY_PALETTE], "white_blue")
        self.assertEqual(values[SETTINGS_KEY_PALETTE_REVERSE], True)
        self.assertEqual(values[SETTINGS_KEY_SCALE_MIN], "1")
        self.assertEqual(values[SETTINGS_KEY_SCALE_MAX], "10")
        self.assertEqual(values[SETTINGS_KEY_GEOMETRY], b"geometry-bytes")

    def test_property_metadata_exposes_groups_and_tiers(self) -> None:
        by_key = {prop["key"]: prop for prop in metrics.PROPERTIES}

        self.assertEqual(by_key["plddt_class"]["group"], "pLDDT")
        self.assertEqual(by_key["plddt"]["group"], "pLDDT")
        self.assertTrue(by_key["plddt_class"]["needs_any_plddt"])
        self.assertTrue(by_key["plddt"]["needs_any_plddt"])
        self.assertEqual(by_key["pae_row_mean"]["group"], "PAE")
        self.assertEqual(by_key["pae_row_mean"]["tier"], "normal")
        self.assertEqual(by_key["pae_col_mean"]["tier"], "normal")
        self.assertEqual(by_key["pae_contact"]["tier"], "advanced")
        self.assertEqual(by_key["pae_domain_complete"]["tier"], "experimental")
        self.assertEqual(by_key["pde_within_sel"]["tier"], "advanced")
        self.assertEqual(by_key["pde_contact"]["tier"], "advanced")
        self.assertEqual(by_key["pae_domain_spectral"]["tier"], "experimental")
        self.assertEqual(by_key["chain_iptm"]["group"], "Chain/interface")
        self.assertEqual(by_key["chain_iptm"]["tier"], "normal")

    def test_property_combo_uses_group_headers_and_tier_labels(self) -> None:
        enabled = _PYMOL.Qt.QtCore.Qt.ItemFlag.ItemIsEnabled
        dialog = _new_dialog()
        dialog._prop_combo = _Combo(0, enabled)

        dialog._populate_property_combo()

        texts = [
            dialog._prop_combo.itemText(row)
            for row in range(dialog._prop_combo.count())
        ]
        self.assertIn("pLDDT", texts)
        self.assertIn("PAE", texts)
        self.assertIn("PDE", texts)
        self.assertIn("Chain/interface", texts)
        plddt_rows = [
            text
            for text in texts
            if text.startswith("  pLDDT") and "Ensemble" not in text
        ]
        self.assertEqual(
            plddt_rows,
            ["  pLDDT — quality classes", "  pLDDT — continuous"],
        )
        self.assertIn("  PAE — row mean", texts)
        self.assertNotIn("  PAE — row mean [Advanced]", texts)
        self.assertIn("  PAE — column mean to selection [Advanced]", texts)
        self.assertIn("  PAE — contact-filtered to selection [Advanced]", texts)
        self.assertIn("  PAE — domain labels (complete linkage) [Experimental]", texts)
        self.assertIn(
            "  PAE — domain labels (spectral clustering) [Experimental]", texts
        )
        self.assertIn("  PDE — contact-filtered to selection [Advanced]", texts)
        self.assertIn("  Chain ipTM", texts)
        self.assertIn("  PAE — row mean to selection [Advanced]", texts)
        self.assertIn("  Interaction probability — mean to selection [Advanced]", texts)
        self.assertLess(
            texts.index("  PDE — mean"),
            texts.index("  PDE — within-chain mean"),
        )
        self.assertLess(
            texts.index("  PDE — within-chain mean"),
            texts.index("  PDE — mean to selection [Advanced]"),
        )
        pae_header_row = texts.index("PAE")
        self.assertFalse(
            dialog._prop_combo.model().item(pae_header_row).flags() & enabled
        )
        self.assertEqual(
            dialog._prop_combo_rows["chain_iptm"],
            texts.index("  Chain ipTM"),
        )

    def test_show_selected_plot_dispatches_selected_handler(self) -> None:
        dialog = _new_dialog()
        called = []
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "plddt")
        dialog._ref_edit = _LineEdit("resname LIG")
        dialog._current_target_kind = lambda: "ensemble_group"
        dialog._ensemble_members = [object()]
        dialog._has_fingerprint_data = lambda: True
        dialog._show_line_plot = lambda: called.append("line")
        dialog._show_distribution_plot = lambda: called.append("distribution")
        dialog._show_matrix_plot = lambda: called.append("matrix")
        dialog._show_binding_site_fingerprint = lambda: called.append("fingerprint")
        dialog._show_ensemble_site_summary = lambda: called.append("site")

        dialog._show_selected_plot("ensemble_site_summary")

        self.assertEqual(called, ["site"])

    def test_show_selected_plot_rejects_domain_label_line_and_ensemble_plots(
        self,
    ) -> None:
        dialog = _new_dialog()
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.infos.clear()
        dialog._prop_combo = types.SimpleNamespace(
            currentData=lambda: "pae_domain_complete"
        )
        dialog._ref_edit = _LineEdit("resname LIG")
        dialog._current_target_kind = lambda: "ensemble_group"
        dialog._ensemble_members = [object()]
        dialog._has_fingerprint_data = lambda: True
        dialog._show_line_plot = lambda: (_ for _ in ()).throw(
            AssertionError("Line plot should be blocked")
        )
        dialog._show_ensemble_site_summary = lambda: (_ for _ in ()).throw(
            AssertionError("Ensemble summary should be blocked")
        )

        for plot_type in ("line", "ensemble_site_summary"):
            dialog._show_selected_plot(plot_type)

        self.assertEqual(len(msg.infos), 2)
        self.assertIn("PAE domain labels", msg.infos[0][1])

    def test_plot_menu_actions_update_from_reference_and_metric(self) -> None:
        action_cls = _PYMOL.Qt.QtGui.QAction
        dialog = _new_dialog()
        dialog._plot_actions = {
            key: action_cls(label) for label, key in metrics.PLOT_TYPES
        }
        dialog._current_target_kind = lambda: "single"
        dialog._ensemble_members = None
        dialog._has_fingerprint_data = lambda: True
        dialog._ref_edit = _LineEdit("")
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "pde_contact")

        dialog._update_plot_actions()

        self.assertFalse(dialog._plot_actions["line"].isEnabled())
        self.assertFalse(dialog._plot_actions["binding_site_fingerprint"].isEnabled())
        self.assertTrue(dialog._plot_actions["matrix"].isEnabled())

        dialog._ref_edit.setText("resname LIG")
        dialog._update_plot_actions()

        self.assertFalse(dialog._plot_actions["line"].isEnabled())
        self.assertTrue(dialog._plot_actions["binding_site_fingerprint"].isEnabled())

        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "pae_contact")
        dialog._update_plot_actions()

        self.assertFalse(dialog._plot_actions["line"].isEnabled())
        self.assertTrue(dialog._plot_actions["distribution"].isEnabled())
        self.assertTrue(dialog._plot_actions["matrix"].isEnabled())

        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "chain_iptm")
        dialog._update_plot_actions()

        self.assertTrue(dialog._plot_actions["matrix"].isEnabled())
        self.assertFalse(dialog._plot_actions["distribution"].isEnabled())
        self.assertIn("chain ipTM", dialog._plot_actions["distribution"].tooltip)

    def test_plot_menu_contains_all_current_plot_actions(self) -> None:
        action_cls = _PYMOL.Qt.QtGui.QAction
        dialog = _new_dialog()
        dialog._plot_actions = {
            key: action_cls(label) for label, key in metrics.PLOT_TYPES
        }

        self.assertEqual(
            set(dialog._plot_actions), {key for _label, key in metrics.PLOT_TYPES}
        )

    def _context_dialog(self, metric: str, *, ref: str = "", cutoff: str = "5.0"):
        dialog = _new_dialog()
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: metric)
        dialog._obj_combo = types.SimpleNamespace(currentText=lambda: "target_model_0")
        dialog._ref_label = _Label("Reference:")
        dialog._ref_edit = _LineEdit(ref)
        dialog._cutoff_label = _Label("Cutoff (Å):")
        dialog._cutoff_edit = _LineEdit(cutoff)
        dialog._preview_label = _Label("")
        dialog._ensemble_members = None
        dialog._ensemble_group_name = None
        dialog._selected_ensemble_member = lambda _obj_name: None
        dialog._has_fingerprint_data = lambda: False
        return dialog

    def test_help_guide_dialog_contains_common_recipes(self) -> None:
        dialog = _new_dialog()
        dialog._guide_dialog = None

        dialog._show_guide()

        guide = dialog._guide_dialog
        self.assertEqual(guide.title, "FoldQC Quick Guide")
        self.assertFalse(guide.modal)
        self.assertTrue(guide.visible)
        text = guide._foldqc_guide_text
        self.assertIn("pLDDT - quality classes", text)
        self.assertIn('"resname LIG" or "organic"', text)
        self.assertIn("PDE - contact-filtered to selection", text)
        self.assertIn('"chain B"', text)
        self.assertIn("Chain ipTM with Plot > Matrix", text)
        self.assertIn("domain labels are categorical/experimental", text)
        self.assertIn("Load Ensemble", text)

        dialog._show_guide()

        self.assertIs(dialog._guide_dialog, guide)
        self.assertTrue(guide.raised)
        self.assertTrue(guide.activated)

    def test_preview_widgets_reserve_five_vertically_centered_lines(self) -> None:
        dialog = _new_dialog()
        caption = _Label("Preview:")
        preview = _Label("")
        flags = _PYMOL.Qt.QtCore.Qt.AlignmentFlag
        left_vcenter = flags.AlignLeft | flags.AlignVCenter

        dialog._configure_preview_widgets(caption, preview)

        self.assertTrue(preview.word_wrap)
        self.assertEqual(preview.minimum_height, preview.line_spacing * 5 + 8)
        self.assertEqual(caption.minimum_height, preview.minimum_height)
        self.assertEqual(preview.alignment, left_vcenter)
        self.assertEqual(caption.alignment, left_vcenter)

    def test_context_controls_disable_unused_reference_but_keep_cutoff_enabled(
        self,
    ) -> None:
        dialog = self._context_dialog("chain_iptm")

        dialog._update_context_controls()

        self.assertEqual(dialog._ref_label.text, "Reference selection:")
        self.assertFalse(dialog._ref_edit.isEnabled())
        self.assertEqual(dialog._cutoff_label.text, "Cutoff (Å):")
        self.assertTrue(dialog._cutoff_edit.isEnabled())

    def test_context_controls_enable_reference_for_selection_metric(self) -> None:
        dialog = self._context_dialog("pae_to_sel")

        dialog._update_context_controls()

        self.assertEqual(dialog._ref_label.text, "Reference selection:")
        self.assertTrue(dialog._ref_edit.isEnabled())
        self.assertEqual(dialog._cutoff_label.text, "Cutoff (Å):")
        self.assertTrue(dialog._cutoff_edit.isEnabled())

    def test_context_controls_enable_site_fields_for_plot_workflows(self) -> None:
        dialog = self._context_dialog("plddt")
        dialog._has_fingerprint_data = lambda: True

        dialog._update_context_controls()

        self.assertEqual(dialog._ref_label.text, "Reference selection:")
        self.assertTrue(dialog._ref_edit.isEnabled())
        self.assertEqual(dialog._cutoff_label.text, "Cutoff (Å):")
        self.assertTrue(dialog._cutoff_edit.isEnabled())

    def test_context_controls_label_cutoff_for_contact_and_domain_metrics(self) -> None:
        contact = self._context_dialog("pde_contact")
        pae_contact = self._context_dialog("pae_contact")
        domain = self._context_dialog("pae_domain_complete")

        contact._update_context_controls()
        pae_contact._update_context_controls()
        domain._update_context_controls()

        self.assertEqual(contact._ref_label.text, "Reference selection:")
        self.assertTrue(contact._ref_edit.isEnabled())
        self.assertEqual(contact._cutoff_label.text, "Cutoff (Å):")
        self.assertTrue(contact._cutoff_edit.isEnabled())
        self.assertEqual(pae_contact._ref_label.text, "Reference selection:")
        self.assertTrue(pae_contact._ref_edit.isEnabled())
        self.assertEqual(pae_contact._cutoff_label.text, "Cutoff (Å):")
        self.assertTrue(pae_contact._cutoff_edit.isEnabled())
        self.assertEqual(domain._cutoff_label.text, "PAE threshold (Å):")
        self.assertTrue(domain._cutoff_edit.isEnabled())

    def test_metric_preview_updates_label_from_rules(self) -> None:
        dialog = self._context_dialog("pde_contact", ref="resname LIG")

        dialog._update_metric_preview()

        self.assertIn(
            'Colors polymer binding-site residues in the target within 5 Å of "resname LIG"',
            dialog._preview_label.text,
        )
        self.assertIn("mean PDE to the reference", dialog._preview_label.text)
        self.assertIn(
            'restricted to tokens selected by "resname LIG"',
            dialog._preview_label.text,
        )

    def test_metric_preview_describes_pae_contact(self) -> None:
        dialog = self._context_dialog("pae_contact", ref="resname LIG")

        dialog._update_metric_preview()

        self.assertIn("mean PAE to the reference", dialog._preview_label.text)
        self.assertIn(
            'restricted to tokens selected by "resname LIG"',
            dialog._preview_label.text,
        )

    def test_metric_preview_gives_actionable_missing_reference_hint(self) -> None:
        dialog = self._context_dialog("pae_to_sel")

        dialog._update_metric_preview()

        self.assertEqual(
            dialog._preview_label.text,
            "Requires a reference selection, such as a chain, ligand, or residue set.",
        )

    def test_metric_preview_gives_contact_hint_when_reference_missing(self) -> None:
        dialog = self._context_dialog("pde_contact", cutoff="6.0")

        dialog._update_metric_preview()

        self.assertIn(
            "reference selection and contact cutoff", dialog._preview_label.text
        )
        self.assertIn("chain, ligand, or residue set", dialog._preview_label.text)

    def test_metric_preview_gives_actionable_missing_ensemble_hint(self) -> None:
        dialog = self._context_dialog("ensemble_rmsd")

        dialog._update_metric_preview()

        self.assertIn("Load Ensemble", dialog._preview_label.text)
        self.assertIn("to use this metric", dialog._preview_label.text)

    def test_metric_preview_describes_selected_ensemble_group(self) -> None:
        dialog = self._context_dialog("plddt")
        dialog._ensemble_members = [object(), object()]
        dialog._ensemble_group_name = "target_ensemble"
        dialog._obj_combo = types.SimpleNamespace(currentText=lambda: "target_ensemble")

        dialog._update_metric_preview()

        self.assertIn("all members of the ensemble", dialog._preview_label.text)
        self.assertIn(
            "line plots show the member mean and standard deviation",
            dialog._preview_label.text,
        )
        self.assertNotIn("target_ensemble", dialog._preview_label.text)

    def test_chain_iptm_preview_mentions_matrix(self) -> None:
        dialog = self._context_dialog("chain_iptm")

        dialog._update_metric_preview()

        self.assertIn("Matrix", dialog._preview_label.text)
        self.assertIn("pairwise chain ipTM", dialog._preview_label.text)

    def test_domain_label_preview_mentions_categorical_threshold_behavior(self) -> None:
        dialog = self._context_dialog("pae_domain_complete", cutoff="7.0")

        dialog._update_metric_preview()

        self.assertIn("categorical", dialog._preview_label.text)
        self.assertIn("threshold", dialog._preview_label.text)

    def test_missing_model_object_is_loaded_selected_and_painted(self) -> None:
        cmd = _Cmd()
        dialog = _dialog_with(cmd)

        obj_name = dialog._ensure_model_object(1)

        self.assertEqual(obj_name, "target_model_1")
        self.assertEqual(
            cmd.loads, [("/tmp/target_model_1.cif", "target_model_1", 1, 1)]
        )
        self.assertEqual(cmd.enabled_calls, [])
        self.assertEqual(dialog.selected, ["target_model_1"])
        self.assertEqual(dialog.painted, [("plddt_class", "target_model_1")])

    def test_disabled_model_object_is_enabled_without_reload_or_repaint(self) -> None:
        cmd = _Cmd(objects={"target_model_1"}, enabled=set())
        dialog = _dialog_with(cmd)

        obj_name = dialog._ensure_model_object(1)

        self.assertEqual(obj_name, "target_model_1")
        self.assertEqual(cmd.loads, [])
        self.assertEqual(cmd.enabled_calls, ["target_model_1"])
        self.assertEqual(dialog.selected, ["target_model_1"])
        self.assertEqual(dialog.painted, [])

    def test_visible_model_object_is_not_reloaded_enabled_or_repainted(self) -> None:
        cmd = _Cmd(objects={"target_model_1"}, enabled={"target_model_1"})
        dialog = _dialog_with(cmd)

        obj_name = dialog._ensure_model_object(1)

        self.assertEqual(obj_name, "target_model_1")
        self.assertEqual(cmd.loads, [])
        self.assertEqual(cmd.enabled_calls, [])
        self.assertEqual(dialog.selected, ["target_model_1"])
        self.assertEqual(dialog.painted, [])

    def test_ensemble_group_is_ordered_before_model_targets(self) -> None:
        dialog = _new_dialog()
        dialog._ensemble_group_name = "target_ensemble"
        dialog._ensemble_members = [
            types.SimpleNamespace(rank=0, obj_name="target_model_0"),
            types.SimpleNamespace(rank=1, obj_name="target_model_1"),
            types.SimpleNamespace(rank=2, obj_name="target_model_2"),
        ]

        ordered = dialog._ordered_target_names(
            [
                "other_object",
                "target_model_2",
                "target_ensemble",
                "target_model_1",
                "target_model_0",
            ]
        )

        self.assertEqual(
            ordered,
            [
                "target_ensemble",
                "target_model_0",
                "target_model_1",
                "target_model_2",
                "other_object",
            ],
        )

    def test_target_names_are_sorted_without_active_ensemble(self) -> None:
        dialog = _new_dialog()
        dialog._ensemble_group_name = None
        dialog._ensemble_members = None

        ordered = dialog._ordered_target_names(["z_obj", "a_obj", "B_obj"])

        self.assertEqual(ordered, ["a_obj", "B_obj", "z_obj"])

    def test_confidence_summary_update_writes_text_box(self) -> None:
        dialog = _new_dialog()
        dialog._conf_browser = _TextBox()
        dialog._pred_data = types.SimpleNamespace(
            provider="boltz",
            confidence={
                "confidence_score": 0.67,
                "ptm": 0.96,
                "iptm": 0.95,
                "ligand_iptm": 0.94,
                "protein_iptm": 0.93,
                "complex_plddt": 0.61,
                "complex_iplddt": 0.62,
                "complex_pde": 0.43,
                "complex_ipde": 0.44,
                "chains_ptm": {"1": 0.8, "0": 0.9},
            },
            affinity={
                "affinity_pred_value": -1.234,
                "affinity_probability_binary": 0.8765,
            },
        )

        dialog._update_confidence_summary()

        text = dialog._conf_browser.text
        self.assertIn("provider         : Boltz-2", text)
        self.assertIn("protein_iptm", text)
        self.assertIn("complex_iplddt", text)
        self.assertIn("complex_ipde", text)
        self.assertIn("chains_ptm:", text)
        self.assertLess(text.index("chain 0"), text.index("chain 1"))
        self.assertIn("affinity_pred_value", text)
        self.assertIn("affinity_probability", text)

    def test_continuous_plddt_warns_when_missing(self) -> None:
        dialog = _new_dialog()
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.warnings.clear()
        self.assertIsNone(
            dialog._compute_property_for(
                "plddt",
                None,
                types.SimpleNamespace(structure_plddt=None, plddt=None),
                [],
                "target_model_0",
            )
        )
        self.assertEqual(len(msg.warnings), 1)
        self.assertIn("pLDDT data are not available", msg.warnings[0][1])

    def test_plddt_class_coloring_uses_selected_fallback_source(self) -> None:
        dialog = _new_dialog()
        structure_values = np.array([0.9, 0.8], dtype=np.float32)
        provider_values = np.array([0.1, 0.2], dtype=np.float32)
        token_map = _token_map(_token(0), _token(1))
        dialog._pred_data = types.SimpleNamespace(
            structure_plddt=structure_values,
            plddt=provider_values,
        )
        dialog._token_map = token_map
        dialog._build_token_map_if_needed = lambda _obj_name: None
        dialog._validate_token_count = lambda values, tm, _obj: self.assertEqual(
            len(values), len(tm)
        )
        dialog._prepare_paint_mapping = lambda *_args: object()
        dialog._confirm_token_overlap_for_coloring = lambda *_args, **_kwargs: True
        dialog.setWindowTitle = lambda _title: None
        dialog._update_statistics_for_single = lambda *_args, **_kwargs: None

        with (
            mock.patch("FoldQC.gui_coloring.paint_plddt_class_coloring") as paint,
            mock.patch("FoldQC.gui_coloring.delete_colorbar"),
        ):
            dialog._apply_plddt_class_coloring("plddt_class", "target_model_0")

        np.testing.assert_array_equal(
            paint.call_args.kwargs["values"], structure_values
        )

        dialog._pred_data = types.SimpleNamespace(
            structure_plddt=None,
            plddt=provider_values,
        )
        with (
            mock.patch("FoldQC.gui_coloring.paint_plddt_class_coloring") as paint,
            mock.patch("FoldQC.gui_coloring.delete_colorbar"),
        ):
            dialog._apply_plddt_class_coloring("plddt_class", "target_model_0")

        np.testing.assert_array_equal(paint.call_args.kwargs["values"], provider_values)

    def test_single_statistics_update_writes_text_box(self) -> None:
        dialog = _new_dialog()
        dialog._stats_browser = _TextBox()

        dialog._update_statistics_for_single(
            "plddt",
            "target_model_0",
            np.array([0.8, 0.9], dtype=np.float32),
        )

        self.assertIn("pLDDT — continuous", dialog._stats_browser.text)
        self.assertIn("Target: target_model_0", dialog._stats_browser.text)
        self.assertIn("mean", dialog._stats_browser.text)

    def test_failed_property_compute_preserves_previous_statistics(self) -> None:
        dialog = _new_dialog()
        dialog._stats_browser = _TextBox("Previous statistics")
        dialog._ensemble_members = None
        dialog._pred_data = object()
        dialog._token_map = object()
        dialog._get_obj_name = lambda: "target_model_0"
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "plddt")
        dialog._palette_combo = types.SimpleNamespace(
            currentData=lambda: "blue_white_red"
        )
        dialog._palette_reverse_chk = types.SimpleNamespace(isChecked=lambda: False)
        dialog._get_vmin_vmax = lambda: (None, None)
        dialog._ref_edit = types.SimpleNamespace(text=lambda: "")
        dialog._build_token_map_if_needed = lambda _obj_name: None
        dialog._compute_property_for = lambda *_args: None

        dialog._apply_coloring()

        self.assertEqual(dialog._stats_browser.text, "Previous statistics")

    def _coloring_dialog_for_overlap(self, obj_name: str = "other"):
        dialog = _new_dialog()
        dialog._ensemble_members = None
        dialog._ensemble_group_name = None
        dialog._pred_data = types.SimpleNamespace(
            structure_path=Path("/tmp/target_model_0.cif"),
            structure_plddt=np.array([0.8, 0.9], dtype=np.float32),
            plddt=None,
        )
        dialog._pred_files = object()
        dialog._token_map = _token_map(_token(0, res_num=1), _token(1, res_num=2))
        dialog._get_obj_name = lambda: obj_name
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "plddt")
        dialog._selected_palette = lambda: ("blue_white_red", False)
        dialog._get_vmin_vmax = lambda: (None, None)
        dialog._ref_edit = types.SimpleNamespace(text=lambda: "")
        dialog._ensure_current_data_for_property = lambda _prop: None
        dialog._build_token_map_if_needed = lambda _obj_name: None
        dialog.setWindowTitle = lambda _title: None
        dialog._update_statistics_for_single = lambda *_args, **_kwargs: None
        return dialog

    def test_coloring_mismatch_warning_cancel_stops_before_paint(self) -> None:
        cmd = _Cmd(objects=("other",), enabled=("other",))
        cmd.selection_models["other"] = types.SimpleNamespace(
            atom=[_atom("B", 10, "GLY"), _atom("B", 11, "SER")]
        )
        _PYMOL.cmd = cmd
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.question_response = msg.Cancel
        dialog = self._coloring_dialog_for_overlap()

        with (
            mock.patch("FoldQC.gui_coloring.paint_property") as paint,
            mock.patch("FoldQC.gui_coloring.show_colorbar"),
        ):
            dialog._apply_coloring()

        self.assertEqual(len(msg.questions), 1)
        self.assertIn("low overlap", msg.questions[0][1])
        paint.assert_not_called()

    def test_coloring_mismatch_continue_is_cached_for_same_target(self) -> None:
        cmd = _Cmd(objects=("other",), enabled=("other",))
        cmd.selection_models["other"] = types.SimpleNamespace(
            atom=[_atom("B", 10, "GLY"), _atom("B", 11, "SER")]
        )
        _PYMOL.cmd = cmd
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.question_response = msg.Yes
        dialog = self._coloring_dialog_for_overlap()

        with (
            mock.patch("FoldQC.gui_coloring.paint_property") as paint,
            mock.patch("FoldQC.gui_coloring.show_colorbar"),
        ):
            paint.return_value = (0.8, 0.9)
            dialog._apply_coloring()
            dialog._apply_coloring()

        self.assertEqual(len(msg.questions), 1)
        self.assertEqual(paint.call_count, 2)
        self.assertEqual(cmd.get_model_calls, 1)

    def test_coloring_overlap_is_rechecked_after_atom_indices_change(self) -> None:
        cmd = _Cmd(objects=("other",), enabled=("other",))
        cmd.selection_models["other"] = types.SimpleNamespace(
            atom=[_atom("B", 10, "GLY"), _atom("B", 11, "SER")]
        )
        _PYMOL.cmd = cmd
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.question_response = msg.Yes
        dialog = self._coloring_dialog_for_overlap()

        with (
            mock.patch("FoldQC.gui_coloring.paint_property") as paint,
            mock.patch("FoldQC.gui_coloring.show_colorbar"),
        ):
            paint.return_value = (0.8, 0.9)
            dialog._apply_coloring()
            changed_atoms = [_atom("B", 10, "GLY"), _atom("B", 11, "SER")]
            changed_atoms[0].index = 3
            changed_atoms[1].index = 4
            cmd.selection_models["other"] = types.SimpleNamespace(atom=changed_atoms)
            dialog._apply_coloring()

        self.assertEqual(len(msg.questions), 2)
        self.assertEqual(cmd.get_model_calls, 2)
        self.assertEqual(paint.call_count, 2)

    def test_coloring_subset_target_does_not_warn(self) -> None:
        cmd = _Cmd(objects=("partial",), enabled=("partial",))
        cmd.selection_models["partial"] = types.SimpleNamespace(
            atom=[_atom("A", 1, "ALA"), _atom("A", 1, "ALA", name="N")]
        )
        _PYMOL.cmd = cmd
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        dialog = self._coloring_dialog_for_overlap("partial")

        with (
            mock.patch("FoldQC.gui_coloring.paint_property") as paint,
            mock.patch("FoldQC.gui_coloring.show_colorbar"),
        ):
            paint.return_value = (0.8, 0.9)
            dialog._apply_coloring()

        self.assertEqual(msg.questions, [])
        paint.assert_called_once()

    def test_successful_overlap_mapping_is_reused_without_get_model(self) -> None:
        cmd = _Cmd(objects=("partial",), enabled=("partial",))
        cmd.selection_models["partial"] = types.SimpleNamespace(
            atom=[_atom("A", 1, "ALA"), _atom("A", 2, "ALA")]
        )
        _PYMOL.cmd = cmd
        dialog = self._coloring_dialog_for_overlap("partial")

        with (
            mock.patch("FoldQC.gui_coloring.paint_property") as paint,
            mock.patch("FoldQC.gui_coloring.show_colorbar"),
        ):
            paint.return_value = (0.8, 0.9)
            dialog._apply_coloring()
            dialog._apply_coloring()

        self.assertEqual(cmd.get_model_calls, 1)
        self.assertEqual(paint.call_count, 2)

    def test_clearing_token_context_also_clears_paint_mappings(self) -> None:
        dialog = _new_dialog()
        dialog._token_map = _token_map(_token(0))
        dialog._token_map_obj = "target_model_0"
        dialog._token_map_structure_path = Path("/tmp/target_model_0.cif")
        dialog._paint_mappings = {("path", "target_model_0"): object()}
        dialog._accepted_token_overlap_warnings = {("path", "target_model_0")}

        dialog._clear_token_map_cache()

        self.assertIsNone(dialog._token_map)
        self.assertEqual(dialog._paint_mappings, {})
        self.assertEqual(dialog._accepted_token_overlap_warnings, set())

    def test_plddt_class_coloring_uses_token_overlap_warning(self) -> None:
        cmd = _Cmd(objects=("other",), enabled=("other",))
        cmd.selection_models["other"] = types.SimpleNamespace(
            atom=[_atom("B", 10, "GLY"), _atom("B", 11, "SER")]
        )
        _PYMOL.cmd = cmd
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.question_response = msg.Cancel
        dialog = _new_dialog()
        dialog._pred_data = types.SimpleNamespace(
            structure_path=Path("/tmp/target_model_0.cif"),
            structure_plddt=np.array([0.8, 0.9], dtype=np.float32),
            plddt=None,
        )
        dialog._token_map = _token_map(_token(0, res_num=1), _token(1, res_num=2))
        dialog._build_token_map_if_needed = lambda _obj_name: None
        dialog.setWindowTitle = lambda _title: None
        dialog._update_statistics_for_single = lambda *_args, **_kwargs: None

        with (
            mock.patch("FoldQC.gui_coloring.paint_plddt_class_coloring") as paint,
            mock.patch("FoldQC.gui_coloring.delete_colorbar"),
        ):
            dialog._apply_plddt_class_coloring("plddt_class", "other")

        self.assertEqual(len(msg.questions), 1)
        paint.assert_not_called()

    def test_selected_palette_uses_combo_data_and_reverse_checkbox(self) -> None:
        dialog = _new_dialog()
        dialog._palette_combo = types.SimpleNamespace(
            currentData=lambda: "green_white",
            currentText=lambda: "Green-white",
        )
        dialog._palette_reverse_chk = types.SimpleNamespace(isChecked=lambda: True)

        self.assertEqual(dialog._selected_palette(), ("green_white", True))

    def test_chain_iptm_is_disabled_without_confidence_data(self) -> None:
        enabled = _PYMOL.Qt.QtCore.Qt.ItemFlag.ItemIsEnabled
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(
            has_pae=False,
            has_pde=False,
            has_contact_probs=False,
            has_plddt=False,
            has_structure_plddt=True,
            supports_ensemble=False,
        )
        dialog._pred_data = types.SimpleNamespace(
            plddt=None,
            structure_plddt=np.array([0.9], dtype=np.float32),
            confidence=None,
            summary_confidence=None,
        )
        _set_metric_combo(dialog, enabled)

        dialog._update_property_availability()

        chain_row = next(
            row
            for row, prop in enumerate(metrics.PROPERTIES)
            if prop["key"] == "chain_iptm"
        )
        plddt_row = next(
            row for row, prop in enumerate(metrics.PROPERTIES) if prop["key"] == "plddt"
        )
        self.assertFalse(dialog._prop_combo.model().item(chain_row).flags() & enabled)
        self.assertTrue(dialog._prop_combo.model().item(plddt_row).flags() & enabled)

    def test_chain_iptm_is_enabled_with_confidence_data(self) -> None:
        enabled = _PYMOL.Qt.QtCore.Qt.ItemFlag.ItemIsEnabled
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(
            has_pae=False,
            has_pde=False,
            has_contact_probs=False,
            has_plddt=False,
            has_structure_plddt=False,
            supports_ensemble=False,
        )
        dialog._pred_data = types.SimpleNamespace(
            plddt=None,
            structure_plddt=None,
            confidence={"chains_iptm": {"0": 0.8}},
            summary_confidence=None,
        )
        _set_metric_combo(dialog, enabled)

        dialog._update_property_availability()

        chain_row = next(
            row
            for row, prop in enumerate(metrics.PROPERTIES)
            if prop["key"] == "chain_iptm"
        )
        self.assertTrue(dialog._prop_combo.model().item(chain_row).flags() & enabled)

    def test_chain_iptm_is_disabled_when_confidence_lacks_chain_scores(self) -> None:
        enabled = _PYMOL.Qt.QtCore.Qt.ItemFlag.ItemIsEnabled
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(
            has_pae=False,
            has_pde=False,
            has_contact_probs=False,
            has_plddt=False,
            has_structure_plddt=True,
            supports_ensemble=False,
        )
        dialog._pred_data = types.SimpleNamespace(
            plddt=None,
            structure_plddt=np.array([0.9], dtype=np.float32),
            confidence={
                "confidence_score": 0.91,
                "chains_ptm": {},
                "pair_chains_iptm": {},
            },
            summary_confidence=None,
        )
        _set_metric_combo(dialog, enabled)

        dialog._update_property_availability()

        chain_row = next(
            row
            for row, prop in enumerate(metrics.PROPERTIES)
            if prop["key"] == "chain_iptm"
        )
        plddt_row = next(
            row for row, prop in enumerate(metrics.PROPERTIES) if prop["key"] == "plddt"
        )
        self.assertFalse(dialog._prop_combo.model().item(chain_row).flags() & enabled)
        self.assertTrue(dialog._prop_combo.model().item(plddt_row).flags() & enabled)

    def test_ensemble_properties_require_loaded_ensemble(self) -> None:
        enabled = _PYMOL.Qt.QtCore.Qt.ItemFlag.ItemIsEnabled
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(
            has_pae=False,
            has_pde=False,
            has_contact_probs=False,
            has_plddt=True,
            has_structure_plddt=True,
            supports_ensemble=True,
        )
        dialog._pred_data = types.SimpleNamespace(
            plddt=np.array([0.8], dtype=np.float32),
            structure_plddt=np.array([0.8], dtype=np.float32),
            confidence=None,
            summary_confidence=None,
        )
        dialog._ensemble_members = None
        _set_metric_combo(dialog, enabled)

        dialog._update_property_availability()

        ensemble_rows = [
            row
            for row, prop in enumerate(metrics.PROPERTIES)
            if prop.get("ensemble_level", False)
        ]
        for row in ensemble_rows:
            self.assertFalse(dialog._prop_combo.model().item(row).flags() & enabled)

        dialog._ensemble_members = [types.SimpleNamespace(rank=0)]
        _set_metric_combo(dialog, enabled)
        dialog._update_property_availability()

        for row in ensemble_rows:
            self.assertTrue(dialog._prop_combo.model().item(row).flags() & enabled)

    def test_ensemble_plddt_properties_accept_structure_plddt_only(self) -> None:
        enabled = _PYMOL.Qt.QtCore.Qt.ItemFlag.ItemIsEnabled
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(
            has_pae=False,
            has_pde=False,
            has_contact_probs=False,
            has_plddt=False,
            has_structure_plddt=True,
            supports_ensemble=True,
        )
        dialog._pred_data = types.SimpleNamespace(
            plddt=None,
            structure_plddt=np.array([0.8], dtype=np.float32),
            confidence=None,
            summary_confidence=None,
        )
        dialog._ensemble_members = [types.SimpleNamespace(rank=0)]
        _set_metric_combo(dialog, enabled)

        dialog._update_property_availability()

        for key in ("ensemble_plddt_mean", "ensemble_plddt_std"):
            row = next(
                row for row, prop in enumerate(metrics.PROPERTIES) if prop["key"] == key
            )
            self.assertTrue(dialog._prop_combo.model().item(row).flags() & enabled)

    def test_pae_domain_label_properties_are_pae_gated(self) -> None:
        keys = {prop["key"] for prop in metrics.PROPERTIES}
        self.assertIn("pae_domain_complete", keys)
        self.assertIn("pae_domain_spectral", keys)

        enabled = _PYMOL.Qt.QtCore.Qt.ItemFlag.ItemIsEnabled
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(
            has_pae=False,
            has_pde=False,
            has_contact_probs=False,
            has_plddt=False,
            has_structure_plddt=False,
            supports_ensemble=False,
        )
        dialog._pred_data = types.SimpleNamespace(
            plddt=None,
            structure_plddt=None,
            confidence=None,
            summary_confidence=None,
        )
        _set_metric_combo(dialog, enabled)

        dialog._update_property_availability()

        complete_row = next(
            row
            for row, prop in enumerate(metrics.PROPERTIES)
            if prop["key"] == "pae_domain_complete"
        )
        spectral_row = next(
            row
            for row, prop in enumerate(metrics.PROPERTIES)
            if prop["key"] == "pae_domain_spectral"
        )
        self.assertFalse(
            dialog._prop_combo.model().item(complete_row).flags() & enabled
        )
        self.assertFalse(
            dialog._prop_combo.model().item(spectral_row).flags() & enabled
        )

        dialog._pred_files.has_pae = True
        _set_metric_combo(dialog, enabled)
        dialog._update_property_availability()

        self.assertTrue(dialog._prop_combo.model().item(complete_row).flags() & enabled)
        self.assertTrue(dialog._prop_combo.model().item(spectral_row).flags() & enabled)

    def test_ensemble_group_target_routes_apply_to_all_members(self) -> None:
        dialog = _new_dialog()
        members = [
            types.SimpleNamespace(obj_name="target_model_0"),
            types.SimpleNamespace(obj_name="target_model_1"),
        ]
        captured = []
        dialog._ensemble_group_name = "target_ensemble"
        dialog._ensemble_members = members
        dialog._get_obj_name = lambda: "target_ensemble"
        dialog._apply_ensemble_coloring = lambda target_members: captured.append(
            target_members
        )

        dialog._apply_coloring()

        self.assertEqual(captured, [members])

    def test_ensemble_member_target_routes_apply_to_that_member_only(self) -> None:
        dialog = _new_dialog()
        members = [
            types.SimpleNamespace(obj_name="target_model_0"),
            types.SimpleNamespace(obj_name="target_model_1"),
        ]
        captured = []
        dialog._ensemble_group_name = "target_ensemble"
        dialog._ensemble_members = members
        dialog._get_obj_name = lambda: "target_model_1"
        dialog._apply_ensemble_coloring = lambda target_members: captured.append(
            target_members
        )

        dialog._apply_coloring()

        self.assertEqual(captured, [[members[1]]])

    def test_contact_cutoff_rejects_invalid_values(self) -> None:
        dialog = _new_dialog()
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        for text in ("abc", "0", "-1", "nan", "inf"):
            msg.warnings.clear()
            dialog._cutoff_edit = _LineEdit(text)

            self.assertIsNone(dialog._get_cutoff_threshold())
            self.assertEqual(len(msg.warnings), 1)
            self.assertIn("Cutoff / threshold", msg.warnings[0][1])

    def test_pae_domain_complete_uses_cutoff_and_method(self) -> None:
        dialog = _new_dialog()
        dialog._cutoff_edit = _LineEdit("6.25")
        # Lambda returns False for "spectral", so any wrong method produces None.
        dialog._pae_domain_dependency_available = lambda method: (
            method == "complete_linkage"
        )
        data = types.SimpleNamespace(
            pae=np.zeros((3, 3), dtype=np.float32),
            structure_plddt=None,
            plddt=None,
        )

        # All-zero PAE: threshold=6.25 puts all 3 tokens in one cluster.
        values = dialog._compute_property_for(
            "pae_domain_complete", None, data, [], "target_model_0"
        )

        self.assertIsNotNone(values)
        self.assertEqual(values.shape, (3,))

    def test_pae_domain_spectral_uses_cutoff_and_method(self) -> None:
        dialog = _new_dialog()
        dialog._cutoff_edit = _LineEdit("8.5")
        # Lambda returns False for "complete_linkage", so wrong method gives None.
        dialog._pae_domain_dependency_available = lambda method: method == "spectral"
        data = types.SimpleNamespace(
            pae=np.zeros((3, 3), dtype=np.float32),
            structure_plddt=None,
            plddt=None,
        )

        # All-zero PAE: threshold=8.5 puts all 3 tokens in one cluster.
        values = dialog._compute_property_for(
            "pae_domain_spectral", None, data, [], "target_model_0"
        )

        self.assertIsNotNone(values)
        self.assertEqual(values.shape, (3,))

    def test_pae_domain_dependency_warning_for_missing_scipy(self) -> None:
        dialog = _new_dialog()
        dialog._cutoff_edit = _LineEdit("5.0")
        data = types.SimpleNamespace(
            pae=np.zeros((2, 2), dtype=np.float32),
            structure_plddt=None,
            plddt=None,
        )
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.warnings.clear()

        with mock.patch(
            "FoldQC.gui_metrics.importlib.util.find_spec", return_value=None
        ):
            values = dialog._compute_property_for(
                "pae_domain_complete", None, data, [], "target_model_0"
            )

        self.assertIsNone(values)
        self.assertEqual(len(msg.warnings), 1)
        self.assertIn("SciPy", msg.warnings[0][1])

    def test_pae_domain_dependency_warning_for_missing_sklearn(self) -> None:
        dialog = _new_dialog()
        dialog._cutoff_edit = _LineEdit("5.0")
        data = types.SimpleNamespace(
            pae=np.zeros((2, 2), dtype=np.float32),
            structure_plddt=None,
            plddt=None,
        )
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.warnings.clear()

        def _find_spec(name: str):
            if name == "scipy":
                return object()
            return None

        with mock.patch(
            "FoldQC.gui_metrics.importlib.util.find_spec", side_effect=_find_spec
        ):
            values = dialog._compute_property_for(
                "pae_domain_spectral", None, data, [], "target_model_0"
            )

        self.assertIsNone(values)
        self.assertEqual(len(msg.warnings), 1)
        self.assertIn("scikit-learn", msg.warnings[0][1])

    def test_pae_domain_line_ylabel_is_domain_label(self) -> None:
        self.assertEqual(metrics.line_ylabel("pae_domain_complete"), "Domain label")
        self.assertEqual(metrics.line_ylabel("pae_domain_spectral"), "Domain label")

    def test_ensemble_domain_labels_are_painted_per_member_categorically(self) -> None:
        dialog = _new_dialog()
        token_map = [_token(0), _token(1), _token(2)]
        members = [
            types.SimpleNamespace(
                rank=0,
                obj_name="target_model_0",
                data=types.SimpleNamespace(),
                token_map=token_map,
            ),
            types.SimpleNamespace(
                rank=1,
                obj_name="target_model_1",
                data=types.SimpleNamespace(),
                token_map=token_map,
            ),
        ]
        member_arrays = {
            "target_model_0": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "target_model_1": np.array([0.0, 1.0, 1.0], dtype=np.float32),
        }
        dialog._ensemble_group_name = "target_ensemble"
        dialog._selected_palette = lambda: ("blue_white_red", False)
        dialog._get_vmin_vmax = lambda: (None, None)
        dialog._ref_edit = _LineEdit("")
        dialog._ensure_member_data_for_property = lambda *_args: None
        dialog._compute_property_for = lambda _key, _ref, _data, _tm, obj_name: (
            member_arrays[obj_name]
        )
        dialog._validate_token_count = lambda values, tm, _obj: self.assertEqual(
            len(values), len(tm)
        )
        dialog._prepare_paint_mapping = lambda *_args: object()
        dialog._confirm_token_overlap_for_coloring = lambda *_args, **_kwargs: True
        stats_calls = []
        dialog._update_statistics_for_members = lambda *args, **kwargs: (
            stats_calls.append((args, kwargs))
        )
        dialog.setWindowTitle = lambda _title: None

        with (
            mock.patch(
                "FoldQC.gui_coloring.paint_categorical_labels_batch",
                return_value=types.SimpleNamespace(vmin=0.0, vmax=1.0),
            ) as categorical,
            mock.patch("FoldQC.gui_coloring.paint_properties_bulk") as continuous,
            mock.patch("FoldQC.gui_coloring.show_colorbar") as show_colorbar,
            mock.patch("FoldQC.gui_coloring.delete_colorbar") as delete_colorbar,
        ):
            dialog._apply_individual_property_to_ensemble(
                "pae_domain_complete",
                {"needs_pae": True},
                members,
            )

        categorical.assert_called_once()
        targets = categorical.call_args.args[0]
        self.assertEqual(
            [target.obj_name for target in targets],
            ["target_model_0", "target_model_1"],
        )
        np.testing.assert_array_equal(
            targets[0].values,
            member_arrays["target_model_0"],
        )
        np.testing.assert_array_equal(
            targets[1].values,
            member_arrays["target_model_1"],
        )
        continuous.assert_not_called()
        show_colorbar.assert_not_called()
        delete_colorbar.assert_called_once()
        self.assertTrue(stats_calls[0][1]["include_domain_labels"])

    def test_line_plot_data_uses_new_continuous_plddt_key(self) -> None:
        dialog = _new_dialog()
        token_map = [_token(0), _token(1)]
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(
                structure_plddt=np.array([0.9, 0.8], dtype=np.float32),
                plddt=np.array([0.1, 0.2], dtype=np.float32),
            ),
            token_map=token_map,
        )

        x_values, series, ylabel = dialog._compute_line_plot_data("plddt", target, [])

        np.testing.assert_array_equal(x_values, np.array([0, 1], dtype=np.int32))
        self.assertEqual(series[0][0], "pLDDT — continuous")
        np.testing.assert_array_equal(
            series[0][1], np.array([0.9, 0.8], dtype=np.float32)
        )
        self.assertEqual(ylabel, "pLDDT")

    def test_to_selection_line_plot_data_uses_all_tokens_when_reference_exists(
        self,
    ) -> None:
        dialog = _new_dialog()
        dialog._ref_edit = _LineEdit("chain L")
        token_map = [_token(i) for i in range(4)]
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(),
            token_map=token_map,
        )
        values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        dialog._compute_property_for = lambda *_args: values

        for key in (
            "pae_to_sel",
            "pae_col_to_sel",
            "pae_sym_sel",
            "pae_contact",
            "pde_to_sel",
            "pde_contact",
            "contact_prob_to_sel",
        ):
            for plot_type in ("line", "distribution"):
                x_values, series, _ylabel = dialog._compute_line_plot_data(
                    key, target, [1, 3], plot_type=plot_type
                )
                np.testing.assert_array_equal(
                    x_values, np.array([0, 1, 2, 3], dtype=np.int32)
                )
                np.testing.assert_array_equal(series[0][1], values)

    def test_reference_scoped_line_plot_data_still_restricts_other_metrics(
        self,
    ) -> None:
        dialog = _new_dialog()
        dialog._ref_edit = _LineEdit("chain L")
        token_map = [_token(i) for i in range(4)]
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(
                pde=np.arange(16, dtype=np.float32).reshape(4, 4)
            ),
            token_map=token_map,
        )
        dialog._compute_property_for = lambda *_args: np.array(
            [np.nan, 1.0, np.nan, 3.0], dtype=np.float32
        )

        x_values, series, _ylabel = dialog._compute_line_plot_data(
            "pde_within_sel", target, [1, 3], plot_type="distribution"
        )

        np.testing.assert_array_equal(x_values, np.array([1, 3], dtype=np.int32))
        np.testing.assert_array_equal(
            series[0][1], np.array([1.0, 3.0], dtype=np.float32)
        )

    def test_contact_probability_to_selection_line_keeps_context_tokens(self) -> None:
        dialog = _new_dialog()
        dialog._ref_edit = _LineEdit("chain L")
        contact_probs = np.array(
            [
                [1.0, 0.1, 0.4],
                [0.1, 1.0, 0.7],
                [0.4, 0.7, 1.0],
            ],
            dtype=np.float32,
        )
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(contact_probs=contact_probs),
            token_map=[_token(0), _token(1), _token(2)],
        )

        import FoldQC.gui_metrics as gui_module

        old_selection = gui_module.selection_to_token_indices
        try:
            gui_module.selection_to_token_indices = lambda _tm, _sel, obj_name="all": [
                0,
                2,
            ]
            x_values, series, _ylabel = dialog._compute_line_plot_data(
                "contact_prob_to_sel", target, [0, 2]
            )
        finally:
            gui_module.selection_to_token_indices = old_selection

        np.testing.assert_array_equal(x_values, np.array([0, 1, 2], dtype=np.int32))
        np.testing.assert_allclose(
            series[0][1],
            np.array([np.nan, 0.4, np.nan], dtype=np.float32),
            equal_nan=True,
        )

    def test_csv_export_single_model_rows_follow_current_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dialog = _new_dialog()
            dialog._pred_files = _CsvPredictionFiles(root, ranks=(0,))
            data = types.SimpleNamespace(
                name="target",
                provider="boltz",
                rank=0,
                display_label="rank 0",
                structure_path=root / "target_model_0.cif",
                structure_plddt=np.array([0.91, 0.42], dtype=np.float32),
                plddt=None,
            )
            token_map = [_token(0), _token(1, chain_id="L", is_hetatm=True)]
            target = _PlotTarget(
                kind="single",
                label="target_model_0",
                obj_name="target_model_0",
                data=data,
                token_map=token_map,
            )
            dialog._pred_data = data
            dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "plddt")
            dialog._ref_edit = _LineEdit("chain L")
            dialog._resolve_plot_target = lambda: target
            dialog._ensure_current_data_for_property = lambda _prop: None

            rows = dialog._build_csv_export_rows()

        self.assertEqual(len(rows), 2)
        first = rows[0]
        self.assertEqual(first["metric_key"], "plddt")
        self.assertEqual(first["model_label"], "rank 0")
        self.assertEqual(first["value"], float(np.float32(0.91)))
        self.assertEqual(first["token_type"], "polymer_residue")
        self.assertEqual(rows[1]["token_type"], "ligand_atom")
        for excluded in (
            "palette",
            "palette_reversed",
            "scale_min",
            "scale_max",
            "object_name",
            "pymol_selection",
        ):
            self.assertNotIn(excluded, first)

    def test_csv_export_reference_and_contact_flags_follow_current_inputs(
        self,
    ) -> None:
        dialog = _new_dialog()
        root = Path("/tmp/foldqc-export")
        dialog._pred_files = _CsvPredictionFiles(root, ranks=(0,))
        pde = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
        data = types.SimpleNamespace(
            name="target",
            provider="boltz",
            rank=0,
            display_label="rank 0",
            structure_path=root / "target_model_0.cif",
            pde=pde,
            structure_plddt=None,
            plddt=None,
            confidence=None,
        )
        token_map = [
            _token(0, chain_id="L", is_hetatm=True),
            _token(1, chain_id="A"),
            _token(2, chain_id="A"),
        ]
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=data,
            token_map=token_map,
        )
        dialog._pred_data = data
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "pde_contact")
        dialog._ref_edit = _LineEdit("resname LIG")
        dialog._cutoff_edit = _LineEdit("7.5")
        dialog._resolve_plot_target = lambda: target
        dialog._ensure_current_data_for_property = lambda _prop: None
        dialog._binding_site_token_indices = lambda *_args: [1, 2]

        import FoldQC.gui_plots as gui_module

        old_selection = gui_module.selection_to_token_indices

        def fake_selection_to_token_indices(_tm, selection, obj_name="all"):
            if selection == "resname LIG":
                return [0]
            return [0, 1, 2]

        try:
            gui_module.selection_to_token_indices = fake_selection_to_token_indices
            rows = dialog._build_csv_export_rows()
        finally:
            gui_module.selection_to_token_indices = old_selection

        self.assertEqual(rows[0]["reference_selection"], "resname LIG")
        self.assertEqual(rows[0]["cutoff_angstrom"], 7.5)
        self.assertEqual(rows[0]["is_reference_token"], "true")
        self.assertEqual(rows[0]["is_contact_token"], "false")
        self.assertEqual(rows[1]["is_reference_token"], "false")
        self.assertEqual(rows[1]["is_contact_token"], "true")
        self.assertEqual(rows[2]["is_contact_token"], "true")
        self.assertEqual(rows[0]["value"], "nan")
        self.assertEqual(rows[1]["value"], 1.0)
        self.assertEqual(rows[2]["value"], 2.0)

    def test_csv_export_pae_contact_uses_contact_flags_and_symmetric_values(
        self,
    ) -> None:
        dialog = _new_dialog()
        root = Path("/tmp/foldqc-export")
        dialog._pred_files = _CsvPredictionFiles(root, ranks=(0,))
        pae = np.array(
            [
                [0.0, 1.0, 8.0],
                [2.0, 0.0, 6.0],
                [4.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
        data = types.SimpleNamespace(
            name="target",
            provider="alphafold3",
            rank=0,
            display_label="rank 0",
            structure_path=root / "target_model_0.cif",
            pae=pae,
            structure_plddt=None,
            plddt=None,
            confidence=None,
        )
        token_map = [
            _token(0, chain_id="L", is_hetatm=True),
            _token(1, chain_id="A"),
            _token(2, chain_id="A"),
        ]
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=data,
            token_map=token_map,
        )
        dialog._pred_data = data
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "pae_contact")
        dialog._ref_edit = _LineEdit("resname LIG")
        dialog._cutoff_edit = _LineEdit("7.5")
        dialog._resolve_plot_target = lambda: target
        dialog._ensure_current_data_for_property = lambda _prop: None
        dialog._binding_site_token_indices = lambda *_args: [1, 2]

        import FoldQC.gui_plots as gui_module

        old_selection = gui_module.selection_to_token_indices

        def fake_selection_to_token_indices(_tm, selection, obj_name="all"):
            if selection == "resname LIG":
                return [0]
            return [0, 1, 2]

        try:
            gui_module.selection_to_token_indices = fake_selection_to_token_indices
            rows = dialog._build_csv_export_rows()
        finally:
            gui_module.selection_to_token_indices = old_selection

        self.assertEqual(rows[0]["cutoff_angstrom"], 7.5)
        self.assertEqual(rows[0]["is_reference_token"], "true")
        self.assertEqual(rows[0]["is_contact_token"], "false")
        self.assertEqual(rows[1]["is_contact_token"], "true")
        self.assertEqual(rows[2]["is_contact_token"], "true")
        self.assertEqual(rows[1]["value"], 1.5)
        self.assertEqual(rows[2]["value"], 6.0)

    def test_csv_export_ensemble_group_per_model_metric_adds_member_columns(
        self,
    ) -> None:
        dialog = _new_dialog()
        root = Path("/tmp/foldqc-export")
        dialog._pred_files = _CsvPredictionFiles(root, ranks=(0, 1))
        dialog._ensemble_group_name = "target_ensemble"
        dialog._ensemble_aligned = True
        token_map = [_token(0), _token(1)]
        members = [
            types.SimpleNamespace(
                rank=0,
                obj_name="target_model_0",
                data=types.SimpleNamespace(
                    rank=0,
                    display_label="rank 0",
                    provider="boltz",
                    name="target",
                    structure_path=root / "target_model_0.cif",
                    structure_plddt=np.array([0.8, 0.9], dtype=np.float32),
                    plddt=None,
                ),
                token_map=token_map,
            ),
            types.SimpleNamespace(
                rank=1,
                obj_name="target_model_1",
                data=types.SimpleNamespace(
                    rank=1,
                    display_label="rank 1",
                    provider="boltz",
                    name="target",
                    structure_path=root / "target_model_1.cif",
                    structure_plddt=np.array([0.7, 0.6], dtype=np.float32),
                    plddt=None,
                ),
                token_map=token_map,
            ),
        ]
        target = _PlotTarget(
            kind="ensemble_group",
            label="target_ensemble",
            obj_name="target_model_0",
            data=None,
            token_map=token_map,
            members=members,
        )
        dialog._ensemble_members = members
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "plddt_class")
        dialog._ref_edit = _LineEdit("")
        dialog._resolve_plot_target = lambda: target
        dialog._ensure_member_data_for_property = lambda *_args: None

        rows = dialog._build_csv_export_rows()

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["metric_key"], "plddt_class")
        self.assertEqual(rows[0]["value_units"], "plddt")
        self.assertEqual(rows[0]["ensemble_group"], "target_ensemble")
        self.assertEqual(rows[0]["ensemble_member_rank"], 0)
        self.assertEqual(rows[0]["ensemble_member_label"], "rank 0")
        self.assertEqual(rows[0]["ensemble_aligned"], "true")
        self.assertEqual(rows[0]["aggregate_kind"], "ensemble_member")
        self.assertEqual(rows[2]["ensemble_member_rank"], 1)
        self.assertEqual(rows[2]["ensemble_member_label"], "rank 1")

    def test_csv_export_ensemble_level_metric_adds_aggregate_columns(self) -> None:
        dialog = _new_dialog()
        root = Path("/tmp/foldqc-export")
        dialog._pred_files = _CsvPredictionFiles(root, ranks=(0, 1))
        dialog._ensemble_group_name = "target_ensemble"
        dialog._ensemble_aligned = False
        dialog._ensemble_members = [types.SimpleNamespace(rank=0)]
        dialog._ensemble_plddt_mean = np.array([0.75, 0.80], dtype=np.float32)
        dialog._ensemble_plddt_std = np.array([0.05, 0.10], dtype=np.float32)
        token_map = [_token(0), _token(1)]
        member = types.SimpleNamespace(
            rank=0,
            obj_name="target_model_0",
            data=types.SimpleNamespace(
                rank=0,
                display_label="rank 0",
                provider="boltz",
                name="target",
                structure_path=root / "target_model_0.cif",
            ),
            token_map=token_map,
        )
        target = _PlotTarget(
            kind="ensemble_group",
            label="target_ensemble",
            obj_name="target_model_0",
            data=None,
            token_map=token_map,
            members=[member],
        )
        dialog._prop_combo = types.SimpleNamespace(
            currentData=lambda: "ensemble_plddt_mean"
        )
        dialog._ref_edit = _LineEdit("")
        dialog._resolve_plot_target = lambda: target

        rows = dialog._build_csv_export_rows()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["metric_key"], "ensemble_plddt_mean")
        self.assertEqual(rows[0]["value_units"], "plddt")
        self.assertEqual(rows[0]["ensemble_group"], "target_ensemble")
        self.assertEqual(rows[0]["ensemble_member_rank"], "")
        self.assertEqual(rows[0]["ensemble_member_label"], "")
        self.assertEqual(rows[0]["ensemble_aligned"], "false")
        self.assertEqual(rows[0]["aggregate_kind"], "ensemble_mean")
        self.assertEqual(rows[1]["value"], float(np.float32(0.80)))

    def test_export_csv_uses_save_file_dialog_and_adds_csv_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dialog = _new_dialog()
            dialog._pred_files = _CsvPredictionFiles(root, ranks=(0,))
            dialog._pred_data = types.SimpleNamespace(rank=0)
            dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "plddt")
            captured = []
            dialog._export_csv_to_path = lambda path: captured.append(Path(path))
            dialog._raise_after_native_dialog = lambda: (_ for _ in ()).throw(
                AssertionError("Save dialog should return a path")
            )
            old_dialog = getattr(_PYMOL.Qt.QtWidgets, "QFileDialog", None)
            _PYMOL.Qt.QtWidgets.QFileDialog = types.SimpleNamespace(
                getSaveFileName=lambda *_args: (str(root / "tokens"), "")
            )
            try:
                dialog._export_csv()
            finally:
                if old_dialog is None:
                    delattr(_PYMOL.Qt.QtWidgets, "QFileDialog")
                else:
                    _PYMOL.Qt.QtWidgets.QFileDialog = old_dialog

        self.assertEqual(captured, [root / "tokens.csv"])

    def test_ensemble_line_plot_data_uses_cached_plddt_mean_and_std(self) -> None:
        dialog = _new_dialog()
        dialog._ensemble_members = [types.SimpleNamespace(rank=0)]
        dialog._ensemble_rmsd = None
        dialog._ensemble_plddt_mean = np.array([0.8, 0.9], dtype=np.float32)
        dialog._ensemble_plddt_std = np.array([0.1, 0.2], dtype=np.float32)
        token_map = [_token(0), _token(1)]
        target = _PlotTarget(
            kind="ensemble_group",
            label="target_ensemble",
            obj_name="target_model_0",
            data=None,
            token_map=token_map,
            members=[],
        )

        x_values, series, ylabel = dialog._compute_line_plot_data(
            "ensemble_plddt_mean", target, [1]
        )

        np.testing.assert_array_equal(x_values, np.array([1], dtype=np.int32))
        self.assertEqual(ylabel, "pLDDT")
        self.assertEqual(series[0][0], "Ensemble pLDDT mean")
        np.testing.assert_allclose(series[0][1], np.array([0.9], dtype=np.float32))
        np.testing.assert_allclose(series[0][2], np.array([0.2], dtype=np.float32))

    def test_distribution_plot_dispatches_plddt_classes_to_bar_plot(self) -> None:
        dialog = _new_dialog()
        token_map = [_token(i) for i in range(5)]
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(),
            token_map=token_map,
        )
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "plddt_class")
        dialog._resolve_reference_indices = lambda *args, **kwargs: []
        dialog._ensure_current_data_for_property = lambda _prop: None
        dialog._pred_data = target.data
        dialog._compute_line_plot_data = lambda *_args, **_kwargs: (
            np.arange(5, dtype=np.int32),
            [
                (
                    "pLDDT",
                    np.array([0.2, 0.55, 0.75, 0.95, np.nan], dtype=np.float32),
                    None,
                )
            ],
            "pLDDT",
        )
        shown = []
        dialog._show_plot_figure = lambda fig, title: shown.append((fig, title))
        bar_calls = []
        metadata_calls = []
        fake_plots = types.SimpleNamespace(
            make_plddt_class_bar_plot=lambda *args, **kwargs: (
                bar_calls.append((args, kwargs)) or "class-figure"
            ),
            attach_viewer_selection_metadata=lambda *args, **kwargs: (
                metadata_calls.append((args, kwargs)) or args[0]
            ),
        )

        self._with_fake_plots(fake_plots, dialog._show_distribution_plot)

        self.assertEqual(len(bar_calls), 1)
        self.assertEqual(bar_calls[0][0][0], ["very low", "low", "high", "very high"])
        self.assertEqual(bar_calls[0][0][1], [1, 1, 1, 1])
        self.assertEqual(bar_calls[0][1]["total"], 4)
        self.assertIn("\n(target_model_0)", bar_calls[0][1]["title"])
        self.assertEqual(
            metadata_calls[0][1]["bar_token_indices"], [[0], [1], [2], [3]]
        )
        self.assertEqual(shown[0][0], "class-figure")

    def test_distribution_plot_dispatches_continuous_values_to_histogram(self) -> None:
        dialog = _new_dialog()
        token_map = [_token(i) for i in range(4)]
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(),
            token_map=token_map,
        )
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "pae_row_mean")
        dialog._resolve_reference_indices = lambda *args, **kwargs: []
        dialog._ensure_current_data_for_property = lambda _prop: None
        dialog._pred_data = target.data
        dialog._compute_line_plot_data = lambda *_args, **_kwargs: (
            np.arange(4, dtype=np.int32),
            [
                (
                    "pLDDT",
                    np.array([0.2, 1.0, 2.0, np.nan], dtype=np.float32),
                    None,
                )
            ],
            "pLDDT",
        )
        shown = []
        dialog._show_plot_figure = lambda fig, title: shown.append((fig, title))
        hist_calls = []
        metadata_calls = []
        fake_plots = types.SimpleNamespace(
            MAX_HISTOGRAM_BINS=50,
            compute_histogram_bins=lambda *_args, **_kwargs: (
                np.array([2, 1], dtype=np.int64),
                np.array([0.0, 1.5, 3.0], dtype=np.float64),
            ),
            make_histogram_plot=lambda *args, **kwargs: (
                hist_calls.append((args, kwargs)) or "hist-figure"
            ),
            attach_viewer_selection_metadata=lambda *args, **kwargs: (
                metadata_calls.append((args, kwargs)) or args[0]
            ),
        )

        self._with_fake_plots(fake_plots, dialog._show_distribution_plot)

        self.assertEqual(len(hist_calls), 1)
        np.testing.assert_allclose(
            hist_calls[0][1]["bin_edges"], np.array([0.2, 1.1, 2.0])
        )
        self.assertEqual(metadata_calls[0][1]["bar_token_indices"], [[0, 1], [2]])
        np.testing.assert_allclose(
            metadata_calls[0][1]["bar_x_positions"], [0.65, 1.55]
        )
        np.testing.assert_allclose(metadata_calls[0][1]["bar_widths"], [0.9, 0.9])
        self.assertEqual(shown[0][0], "hist-figure")

    def test_distribution_plot_dispatches_domain_labels_to_categorical_bars(
        self,
    ) -> None:
        dialog = _new_dialog()
        token_map = [_token(i) for i in range(4)]
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(),
            token_map=token_map,
        )
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(
            currentData=lambda: "pae_domain_complete"
        )
        dialog._resolve_reference_indices = lambda *args, **kwargs: []
        dialog._ensure_current_data_for_property = lambda _prop: None
        dialog._pred_data = target.data
        dialog._compute_line_plot_data = lambda *_args, **_kwargs: (
            np.arange(4, dtype=np.int32),
            [
                (
                    "PAE domain labels",
                    np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float32),
                    None,
                )
            ],
            "Domain label",
        )
        shown = []
        dialog._show_plot_figure = lambda fig, title: shown.append((fig, title))
        cat_calls = []
        metadata_calls = []
        fake_plots = types.SimpleNamespace(
            make_categorical_bar_plot=lambda *args, **kwargs: (
                cat_calls.append((args, kwargs)) or "category-figure"
            ),
            attach_viewer_selection_metadata=lambda *args, **kwargs: (
                metadata_calls.append((args, kwargs)) or args[0]
            ),
        )

        self._with_fake_plots(fake_plots, dialog._show_distribution_plot)

        self.assertEqual(len(cat_calls), 1)
        self.assertEqual(cat_calls[0][0][0], ["0", "1", "2"])
        self.assertEqual(cat_calls[0][0][1], [1, 2, 1])
        self.assertEqual(
            cat_calls[0][1]["colors"][0],
            (0.1216, 0.4667, 0.7059),
        )
        self.assertEqual(metadata_calls[0][1]["bar_token_indices"], [[0], [1, 2], [3]])
        self.assertEqual(metadata_calls[0][1]["bar_x_positions"], [0, 1, 2])
        self.assertEqual(metadata_calls[0][1]["bar_widths"], [0.8, 0.8, 0.8])
        self.assertEqual(shown[0][0], "category-figure")

    def test_distribution_plot_rejects_chain_iptm(self) -> None:
        dialog = _new_dialog()
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(),
            token_map=[_token(0)],
        )
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "chain_iptm")
        dialog._resolve_reference_indices = lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(AssertionError("chain ipTM should not resolve Reference"))
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        dialog._show_distribution_plot()

        self.assertEqual(len(msg.infos), 1)
        self.assertIn("not available for chain ipTM", msg.infos[0][1])

    def test_distribution_plot_rejects_domain_labels_for_ensemble_group(self) -> None:
        dialog = _new_dialog()
        target = _PlotTarget(
            kind="ensemble_group",
            label="target_ensemble",
            obj_name="target_model_0",
            data=None,
            token_map=[_token(0)],
            members=[],
        )
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(
            currentData=lambda: "pae_domain_complete"
        )
        dialog._resolve_reference_indices = lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(AssertionError("Domain ensemble distribution should stop early"))
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.infos.clear()

        dialog._show_distribution_plot()

        self.assertEqual(len(msg.infos), 1)
        self.assertIn("not pooled across an ensemble", msg.infos[0][1])

    def test_distribution_plot_honors_reference_required_validation(self) -> None:
        dialog = _new_dialog()
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(),
            token_map=[_token(0)],
        )
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "pae_to_sel")
        dialog._resolve_reference_indices = lambda *args, **kwargs: None
        dialog._compute_line_plot_data = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Reference validation should stop distribution plotting")
        )

        dialog._show_distribution_plot()

    def test_line_plot_warns_when_restricted_series_has_no_finite_values(self) -> None:
        dialog = _new_dialog()
        token_map = [_token(0), _token(1)]
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(),
            token_map=token_map,
        )
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(
            currentData=lambda: "contact_prob_to_sel"
        )
        dialog._resolve_reference_indices = lambda *args, **kwargs: [0, 1]
        dialog._ensure_current_data_for_property = lambda _prop: None
        dialog._pred_data = target.data
        dialog._compute_line_plot_data = lambda *_args, **_kwargs: (
            np.array([0, 1], dtype=np.int32),
            [
                (
                    "Interaction probability",
                    np.array([np.nan, np.nan], dtype=np.float32),
                    None,
                )
            ],
            "Interaction probability",
        )
        shown = []
        dialog._show_plot_figure = lambda fig, title: shown.append((fig, title))
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        dialog._show_line_plot()

        self.assertEqual(shown, [])
        self.assertEqual(len(msg.warnings), 1)
        self.assertIn("No finite values", msg.warnings[0][1])

    def test_pde_contact_line_plot_is_blocked_by_direct_handler(self) -> None:
        dialog = _new_dialog()
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(),
            token_map=[_token(0)],
        )
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "pde_contact")
        dialog._resolve_reference_indices = lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(
            AssertionError("PDE contact line plot should stop before Reference lookup")
        )
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        dialog._show_line_plot()

        self.assertEqual(len(msg.infos), 1)
        self.assertIn("PDE contact-filtered", msg.infos[0][1])

    def test_pae_contact_line_plot_is_blocked_by_direct_handler(self) -> None:
        dialog = _new_dialog()
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(),
            token_map=[_token(0)],
        )
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "pae_contact")
        dialog._resolve_reference_indices = lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(
            AssertionError("PAE contact line plot should stop before Reference lookup")
        )
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        dialog._show_line_plot()

        self.assertEqual(len(msg.infos), 1)
        self.assertIn("PAE contact-filtered", msg.infos[0][1])

    def test_pae_summary_plot_data_lazy_loads_single_model_and_scopes_x_only(
        self,
    ) -> None:
        dialog = _new_dialog()
        token_map = _token_map(
            _token(0, chain_id="A"),
            _token(1, chain_id="A"),
            _token(2, chain_id="B"),
        )
        dialog._pred_files = types.SimpleNamespace(has_pae=True, has_pde=False)
        dialog._pred_data = types.SimpleNamespace(
            rank=0,
            pae=None,
            pde=None,
            contact_probs=None,
            structure_plddt=None,
            plddt=None,
        )
        load_calls = []

        def reload_data(rank, **flags):
            load_calls.append((rank, flags))
            return types.SimpleNamespace(
                rank=rank,
                pae=np.array(
                    [
                        [0.0, 2.0, 10.0],
                        [4.0, 0.0, 12.0],
                        [6.0, 8.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                pde=None,
                contact_probs=None,
                structure_plddt=None,
                plddt=None,
            )

        dialog._reload_prediction_data = reload_data
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=dialog._pred_data,
            token_map=token_map,
        )

        x_values, series, ylabel = dialog._compute_summary_plot_data(
            "pae", target, [2, 0]
        )

        np.testing.assert_array_equal(x_values, np.array([2, 0], dtype=np.int32))
        self.assertEqual(ylabel, "PAE gap (Å)")
        self.assertEqual(load_calls[0][1]["load_pae"], True)
        self.assertEqual(series[0][0], "row gap (other - within)")
        np.testing.assert_allclose(series[0][1], np.array([7.0, 9.0], dtype=np.float32))
        np.testing.assert_allclose(
            series[1][1], np.array([11.0, 4.0], dtype=np.float32)
        )
        self.assertEqual(series[0][3], "#1f77b4")
        self.assertEqual(series[1][3], "#6baed6")

    def test_pde_summary_plot_data_lazy_loads_ensemble_member(self) -> None:
        dialog = _new_dialog()
        token_map = _token_map(_token(0, chain_id="A"), _token(1, chain_id="B"))
        member = types.SimpleNamespace(
            rank=1,
            obj_name="target_model_1",
            data=types.SimpleNamespace(pde=None),
            token_map=token_map,
        )
        calls = []

        def ensure(member_arg, **kwargs):
            calls.append((member_arg, kwargs))
            member_arg.data = types.SimpleNamespace(
                pde=np.array([[0.0, 2.0], [4.0, 0.0]], dtype=np.float32)
            )

        dialog._ensure_member_data_for_plot = ensure
        target = _PlotTarget(
            kind="ensemble_member",
            label="target_model_1",
            obj_name="target_model_1",
            data=member.data,
            token_map=token_map,
            members=[member],
        )

        _x_values, series, ylabel = dialog._compute_summary_plot_data("pde", target, [])

        self.assertEqual(ylabel, "PDE gap (Å)")
        self.assertEqual(calls[0][1], {"load_pae": False, "load_pde": True})
        self.assertEqual(series[0][0], "gap (other - within)")
        np.testing.assert_allclose(series[0][1], np.array([2.0, 4.0], dtype=np.float32))

    def test_summary_plot_handler_uses_line_metadata_and_reference_scope(self) -> None:
        dialog = _new_dialog()
        token_map = _token_map(_token(0, chain_id="A"), _token(1, chain_id="B"))
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(
                pde=np.array([[0.0, 2.0], [4.0, 0.0]], dtype=np.float32)
            ),
            token_map=token_map,
        )
        dialog._resolve_plot_target = lambda: target
        dialog._pred_files = types.SimpleNamespace(has_pae=False, has_pde=True)
        dialog._pred_data = None
        dialog._ref_edit = _LineEdit("chain B")
        dialog._resolve_reference_indices = lambda *args, **kwargs: [1]
        dialog._get_vmin_vmax = lambda: (None, None)
        shown = []
        dialog._show_plot_figure = lambda fig, title: shown.append((fig, title))

        plot_calls = []
        metadata_calls = []
        fake_plots = types.SimpleNamespace(
            make_line_plot=lambda *args, **kwargs: (
                plot_calls.append((args, kwargs)) or "summary-figure"
            ),
            attach_viewer_selection_metadata=lambda *args, **kwargs: (
                metadata_calls.append((args, kwargs)) or args[0]
            ),
        )

        self._with_fake_plots(fake_plots, lambda: dialog._show_summary_plot("pde"))

        self.assertEqual(shown, [("summary-figure", "PDE summary (target_model_0)")])
        np.testing.assert_array_equal(
            plot_calls[0][0][0], np.array([1], dtype=np.int32)
        )
        self.assertEqual(plot_calls[0][1]["ylabel"], "PDE gap (Å)")
        self.assertEqual(plot_calls[0][1]["show_legend"], True)
        self.assertEqual(plot_calls[0][0][1][0][0], "gap (other - within)")
        np.testing.assert_allclose(plot_calls[0][0][1][0][1], np.array([4.0]))
        self.assertEqual(metadata_calls[0][1]["kind"], "line")
        self.assertEqual(metadata_calls[0][1]["token_indices"], [1])
        self.assertEqual(metadata_calls[0][1]["x_positions"], [1])

    def test_ensemble_summary_plot_uses_member_maps_for_selection_metadata(
        self,
    ) -> None:
        dialog = _new_dialog()
        token_map = _token_map(_token(0, chain_id="A"), _token(1, chain_id="B"))
        members = [
            types.SimpleNamespace(
                rank=0,
                obj_name="target_model_0",
                data=types.SimpleNamespace(
                    pde=np.array([[0.0, 2.0], [4.0, 0.0]], dtype=np.float32)
                ),
                token_map=token_map,
            ),
            types.SimpleNamespace(
                rank=1,
                obj_name="target_model_1",
                data=types.SimpleNamespace(
                    pde=np.array([[0.0, 4.0], [8.0, 0.0]], dtype=np.float32)
                ),
                token_map=token_map,
            ),
        ]
        target = _PlotTarget(
            kind="ensemble_group",
            label="target_ensemble",
            obj_name="target_model_0",
            data=None,
            token_map=token_map,
            members=members,
        )
        dialog._resolve_plot_target = lambda: target
        dialog._pred_files = types.SimpleNamespace(has_pae=False, has_pde=True)
        dialog._pred_data = None
        dialog._ref_edit = _LineEdit("")
        dialog._resolve_reference_indices = lambda *args, **kwargs: []
        dialog._get_vmin_vmax = lambda: (None, None)
        dialog._ensure_member_data_for_plot = lambda *args, **kwargs: None
        shown = []
        dialog._show_plot_figure = lambda fig, title: shown.append((fig, title))

        metadata_calls = []
        fake_plots = types.SimpleNamespace(
            make_line_plot=lambda *args, **kwargs: "ensemble-summary-figure",
            attach_viewer_selection_metadata=lambda *args, **kwargs: (
                metadata_calls.append((args, kwargs)) or args[0]
            ),
        )

        self._with_fake_plots(fake_plots, lambda: dialog._show_summary_plot("pde"))

        self.assertEqual(
            shown, [("ensemble-summary-figure", "PDE summary (target_ensemble)")]
        )
        self.assertEqual(metadata_calls[0][1]["token_maps"], [token_map, token_map])
        self.assertEqual(
            metadata_calls[0][1]["token_map_obj_names"],
            ["target_model_0", "target_model_1"],
        )

    def test_ensemble_distribution_metadata_targets_all_member_token_maps(self) -> None:
        dialog = _new_dialog()
        token_map = [_token(0), _token(1)]
        members = [
            types.SimpleNamespace(
                rank=0, obj_name="target_model_0", token_map=token_map
            ),
            types.SimpleNamespace(
                rank=1, obj_name="target_model_1", token_map=token_map
            ),
        ]
        target = _PlotTarget(
            kind="ensemble_group",
            label="target_ensemble",
            obj_name="target_model_0",
            data=None,
            token_map=token_map,
            members=members,
        )
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "ensemble_rmsd")
        dialog._resolve_reference_indices = lambda *args, **kwargs: []
        dialog._compute_line_plot_data = lambda *_args, **_kwargs: (
            np.arange(2, dtype=np.int32),
            [("RMSD", np.array([0.2, 1.0], dtype=np.float32), None)],
            "Distance / error (Å)",
        )
        dialog._show_plot_figure = lambda *_args: None
        metadata_calls = []
        fake_plots = types.SimpleNamespace(
            MAX_HISTOGRAM_BINS=50,
            compute_histogram_bins=lambda *_args, **_kwargs: (
                np.array([1, 1], dtype=np.int64),
                np.array([0.0, 0.5, 1.5], dtype=np.float64),
            ),
            make_histogram_plot=lambda *args, **kwargs: "hist-figure",
            attach_viewer_selection_metadata=lambda *args, **kwargs: (
                metadata_calls.append((args, kwargs)) or args[0]
            ),
        )

        self._with_fake_plots(fake_plots, dialog._show_distribution_plot)

        self.assertEqual(metadata_calls[0][1]["token_maps"], [token_map, token_map])
        self.assertEqual(
            metadata_calls[0][1]["token_map_obj_names"],
            ["target_model_0", "target_model_1"],
        )
        self.assertEqual(metadata_calls[0][1]["bar_token_indices"], [[0], [1]])

    def test_ensemble_site_summary_uses_reference_contact_site_and_within_site_matrices(
        self,
    ) -> None:
        cmd = _Cmd()
        _PYMOL.cmd = cmd
        token_map = [
            _token(0, chain_id="L", res_num=1, is_hetatm=True),
            _token(1, chain_id="A", res_num=10),
            _token(2, chain_id="A", res_num=11),
        ]
        data = types.SimpleNamespace(
            structure_plddt=np.array([0.8, np.nan, 0.6], dtype=np.float32),
            plddt=None,
            pae=np.array(
                [[0.0, 2.0, 4.0], [6.0, 0.0, 8.0], [10.0, 12.0, 0.0]],
                dtype=np.float32,
            ),
            pde=np.array(
                [[0.0, 1.0, np.nan], [3.0, 0.0, 5.0], [7.0, 9.0, 0.0]],
                dtype=np.float32,
            ),
        )
        member = types.SimpleNamespace(
            rank=0,
            obj_name="target_model_0",
            data=data,
            token_map=token_map,
        )
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(has_pae=True, has_pde=True)
        dialog._ensemble_members = [member]
        dialog._ensure_member_data_for_plot = lambda *args, **kwargs: None
        dialog._binding_site_token_indices = lambda *args, **kwargs: [1, 2]
        cmd.selection_models["(lig) and target_model_0"] = types.SimpleNamespace(
            atom=[
                types.SimpleNamespace(
                    chain="L",
                    resi="1",
                    hetatm=True,
                    name="C0",
                )
            ]
        )

        members, labels, series, site_indices = (
            dialog._compute_ensemble_site_summary_data("lig", 5.0)
        )

        self.assertEqual(members, [member])
        self.assertEqual(labels, ["model_0"])
        self.assertEqual(site_indices, [[0, 1, 2]])
        self.assertEqual(
            [row[0] for row in series],
            ["mean pLDDT", "PAE mean", "PDE mean"],
        )
        np.testing.assert_allclose(series[0][1], np.array([0.7], dtype=np.float32))
        np.testing.assert_allclose(series[1][1], np.array([7.0], dtype=np.float32))
        np.testing.assert_allclose(series[2][1], np.array([5.0], dtype=np.float32))

    def test_show_ensemble_site_summary_requires_active_ensemble(self) -> None:
        dialog = _new_dialog()
        dialog._ref_edit = _LineEdit("lig")
        dialog._cutoff_edit = _LineEdit("5.0")
        dialog._ensemble_members = None
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        dialog._show_ensemble_site_summary()

        self.assertEqual(len(msg.infos), 1)
        self.assertIn("Load Ensemble", msg.infos[0][1])

    def test_show_ensemble_site_summary_warns_when_no_metrics_are_available(
        self,
    ) -> None:
        dialog = _new_dialog()
        dialog._ref_edit = _LineEdit("lig")
        dialog._cutoff_edit = _LineEdit("5.0")
        dialog._ensemble_members = [types.SimpleNamespace(rank=0)]
        dialog._compute_ensemble_site_summary_data = lambda *_args: (
            [types.SimpleNamespace(rank=0)],
            ["model_0"],
            [],
            [[0]],
        )
        dialog._show_plot_figure = lambda *_args: (_ for _ in ()).throw(
            AssertionError("Empty ensemble site summary should not be shown")
        )
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        dialog._show_ensemble_site_summary()

        self.assertEqual(len(msg.warnings), 1)
        self.assertIn("No pLDDT, PAE, or PDE", msg.warnings[0][1])

    def test_ensemble_matrix_plot_data_averages_models_and_preserves_ref_order(
        self,
    ) -> None:
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(has_pae=False, has_pde=True)
        members = [
            types.SimpleNamespace(
                rank=0,
                obj_name="target_model_0",
                data=types.SimpleNamespace(pae=None, pde=None),
            ),
            types.SimpleNamespace(
                rank=1,
                obj_name="target_model_1",
                data=types.SimpleNamespace(pae=None, pde=None),
            ),
        ]
        matrices = [
            np.arange(9, dtype=np.float32).reshape(3, 3),
            np.arange(9, dtype=np.float32).reshape(3, 3) + 10.0,
        ]

        def ensure(
            member, *, load_pae=False, load_pde=False, load_structure_plddt=False
        ):
            self.assertFalse(load_pae)
            self.assertTrue(load_pde)
            member.data.pde = matrices[member.rank]

        dialog._ensure_member_data_for_plot = ensure
        token_map = [_token(0), _token(1), _token(2)]
        target = _PlotTarget(
            kind="ensemble_group",
            label="target_ensemble",
            obj_name="target_model_0",
            data=None,
            token_map=token_map,
            members=members,
        )

        (
            matrix,
            row_indices,
            col_indices,
            title,
            label,
            row_labels,
            col_labels,
            cell_text,
        ) = dialog._compute_matrix_plot_data("pde_mean", target, [2, 0])

        expected = (matrices[0] + matrices[1]) / 2.0
        np.testing.assert_allclose(matrix, expected[:, [2, 0]])
        self.assertEqual(row_indices, [0, 1, 2])
        self.assertEqual(col_indices, [2, 0])
        self.assertIn("ensemble mean", title)
        self.assertEqual(label, "PDE (Å)")
        self.assertIsNone(row_labels)
        self.assertIsNone(col_labels)
        self.assertIsNone(cell_text)

    def test_pae_row_mean_matrix_uses_reference_rows(self) -> None:
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(has_pae=True, has_pde=False)
        matrix_data = np.arange(16, dtype=np.float32).reshape(4, 4)
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(pae=matrix_data),
            token_map=[_token(i) for i in range(4)],
        )

        (
            matrix,
            row_indices,
            col_indices,
            _title,
            label,
            _row_labels,
            _col_labels,
            _cell_text,
        ) = dialog._compute_matrix_plot_data("pae_row_mean", target, [2, 0])

        np.testing.assert_allclose(matrix, matrix_data[[2, 0], :])
        self.assertEqual(row_indices, [2, 0])
        self.assertEqual(col_indices, [0, 1, 2, 3])
        self.assertEqual(label, "PAE (Å)")

    def test_pae_column_to_selection_matrix_uses_reference_rows(self) -> None:
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(has_pae=True, has_pde=False)
        matrix_data = np.arange(16, dtype=np.float32).reshape(4, 4)
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(pae=matrix_data),
            token_map=[_token(i) for i in range(4)],
        )

        (
            matrix,
            row_indices,
            col_indices,
            _title,
            label,
            _row_labels,
            _col_labels,
            _cell_text,
        ) = dialog._compute_matrix_plot_data("pae_col_to_sel", target, [2, 0])

        np.testing.assert_allclose(matrix, matrix_data[[2, 0], :])
        self.assertEqual(row_indices, [2, 0])
        self.assertEqual(col_indices, [0, 1, 2, 3])
        self.assertEqual(label, "PAE (Å)")

    def test_pae_symmetric_within_selection_matrix_uses_reference_square(
        self,
    ) -> None:
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(has_pae=True, has_pde=False)
        matrix_data = np.arange(16, dtype=np.float32).reshape(4, 4)
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(pae=matrix_data),
            token_map=[_token(i) for i in range(4)],
        )

        (
            matrix,
            row_indices,
            col_indices,
            _title,
            _label,
            _row_labels,
            _col_labels,
            _cell_text,
        ) = dialog._compute_matrix_plot_data("pae_sym_within_sel", target, [2, 0])

        np.testing.assert_allclose(matrix, matrix_data[np.ix_([2, 0], [2, 0])])
        self.assertEqual(row_indices, [2, 0])
        self.assertEqual(col_indices, [2, 0])

    def test_chain_iptm_matrix_plot_data_uses_confidence_json(self) -> None:
        dialog = _new_dialog()
        token_map = _token_map(
            _token(0, chain_id="A"),
            _token(1, chain_id="A"),
            _token(2, chain_id="L", is_hetatm=True),
        )
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(
                confidence={
                    "pair_chains_iptm": {
                        "0": {"0": 0.91234, "1": 0.81234},
                        "1": {"0": 0.71234, "1": 0.61234},
                    }
                }
            ),
            token_map=token_map,
        )

        (
            matrix,
            row_indices,
            col_indices,
            title,
            label,
            row_labels,
            col_labels,
            cell_text,
        ) = dialog._compute_matrix_plot_data("chain_iptm", target, [1])

        np.testing.assert_allclose(
            matrix,
            np.array([[0.91234, 0.81234], [0.71234, 0.61234]], dtype=np.float32),
        )
        self.assertEqual(row_indices, [0, 1])
        self.assertEqual(col_indices, [0, 1])
        self.assertEqual(title, "Pairwise chain ipTM")
        self.assertEqual(label, "ipTM")
        self.assertEqual(row_labels, ["A", "L"])
        self.assertEqual(col_labels, ["A", "L"])
        self.assertEqual(cell_text.tolist(), [["0.912", "0.812"], ["0.712", "0.612"]])

    def test_ensemble_chain_iptm_matrix_plot_data_uses_mean_and_std(self) -> None:
        dialog = _new_dialog()
        token_map = _token_map(_token(0, chain_id="A"), _token(1, chain_id="B"))
        members = [
            types.SimpleNamespace(
                rank=0,
                token_map=token_map,
                data=types.SimpleNamespace(
                    confidence={
                        "pair_chains_iptm": {
                            "0": {"0": 0.8, "1": 0.6},
                            "1": {"0": 0.4, "1": 0.2},
                        }
                    }
                ),
            ),
            types.SimpleNamespace(
                rank=1,
                token_map=token_map,
                data=types.SimpleNamespace(
                    confidence={
                        "pair_chains_iptm": {
                            "0": {"0": 1.0, "1": 0.8},
                            "1": {"0": 0.6, "1": 0.4},
                        }
                    }
                ),
            ),
        ]
        target = _PlotTarget(
            kind="ensemble_group",
            label="target_ensemble",
            obj_name="target_model_0",
            data=None,
            token_map=token_map,
            members=members,
        )

        (
            matrix,
            _row_indices,
            _col_indices,
            title,
            _label,
            row_labels,
            col_labels,
            cell_text,
        ) = dialog._compute_matrix_plot_data("chain_iptm", target, [])

        np.testing.assert_allclose(
            matrix,
            np.array([[0.9, 0.7], [0.5, 0.3]], dtype=np.float32),
        )
        self.assertIn("ensemble mean", title)
        self.assertEqual(row_labels, ["A", "B"])
        self.assertEqual(col_labels, ["A", "B"])
        self.assertEqual(
            cell_text.tolist(),
            [
                ["0.900 +/- 0.100", "0.700 +/- 0.100"],
                ["0.500 +/- 0.100", "0.300 +/- 0.100"],
            ],
        )

    def test_chain_iptm_matrix_plot_data_rejects_missing_pair_matrix(self) -> None:
        dialog = _new_dialog()
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(confidence={}),
            token_map=[_token(0, chain_id="A")],
        )

        with self.assertRaisesRegex(ValueError, "pair_chains_iptm"):
            dialog._compute_matrix_plot_data("chain_iptm", target, [])

    def test_show_matrix_plot_accepts_chain_iptm_without_reference_resolution(
        self,
    ) -> None:
        dialog = _new_dialog()
        token_map = _token_map(_token(0, chain_id="A"), _token(1, chain_id="B"))
        target = _PlotTarget(
            kind="single",
            label="target_model_0",
            obj_name="target_model_0",
            data=types.SimpleNamespace(
                confidence={
                    "pair_chains_iptm": {
                        "0": {"0": 0.9, "1": 0.8},
                        "1": {"0": 0.7, "1": 0.6},
                    }
                }
            ),
            token_map=token_map,
        )
        shown = []
        dialog._resolve_plot_target = lambda: target
        dialog._prop_combo = types.SimpleNamespace(currentData=lambda: "chain_iptm")
        dialog._resolve_reference_indices = lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(AssertionError("chain ipTM should not resolve Reference"))
        dialog._get_vmin_vmax = lambda: (None, None)
        dialog._selected_palette = lambda: ("blue_white_red", False)
        dialog._show_plot_figure = lambda fig, title: shown.append((fig, title))
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        dialog._show_matrix_plot()

        self.assertEqual(msg.infos, [])
        self.assertEqual(len(shown), 1)
        self.assertIn("Pairwise chain ipTM", shown[0][1])

    def test_ensemble_fingerprint_data_uses_mean_and_std(self) -> None:
        dialog = _new_dialog()
        dialog._pred_files = types.SimpleNamespace(has_pae=False, has_pde=False)
        token_map = [_token(0), _token(1)]
        members = [
            types.SimpleNamespace(
                rank=0,
                obj_name="target_model_0",
                data=types.SimpleNamespace(
                    structure_plddt=None,
                    plddt=np.array([0.8, 0.6], dtype=np.float32),
                    pae=None,
                    pde=None,
                ),
            ),
            types.SimpleNamespace(
                rank=1,
                obj_name="target_model_1",
                data=types.SimpleNamespace(
                    structure_plddt=None,
                    plddt=np.array([1.0, 0.2], dtype=np.float32),
                    pae=None,
                    pde=None,
                ),
            ),
        ]
        dialog._ensure_member_data_for_plot = lambda *args, **kwargs: None
        target = _PlotTarget(
            kind="ensemble_group",
            label="target_ensemble",
            obj_name="target_model_0",
            data=None,
            token_map=token_map,
            members=members,
        )

        series = dialog._compute_fingerprint_data(target, [0])

        np.testing.assert_allclose(
            series["plddt"], np.array([0.9, 0.4], dtype=np.float32)
        )
        np.testing.assert_allclose(
            series["plddt_std"], np.array([0.1, 0.2], dtype=np.float32)
        )
        self.assertIsNone(series["pae_to_ligand"])
        self.assertIsNone(series["pae_from_ligand"])
        self.assertIsNone(series["pde_to_ligand"])

    def test_fingerprint_warns_when_reference_selection_is_empty(self) -> None:
        dialog = _new_dialog()
        dialog._pred_data = object()
        dialog._get_obj_name = lambda: "target_model_0"
        dialog._ref_edit = _LineEdit("")
        msg = _PYMOL.Qt.QtWidgets.QMessageBox

        dialog._show_binding_site_fingerprint()

        self.assertEqual(len(msg.warnings), 1)
        self.assertIn("reference selection", msg.warnings[0][1])

    def test_fingerprint_uses_cutoff_object_and_filters_to_polymer_tokens(self) -> None:
        dialog = _new_dialog()
        token_map = [
            _token(0, chain_id="L", is_hetatm=True),
            _token(1, chain_id="A"),
            _token(2, chain_id="L", is_hetatm=True),
            _token(3, chain_id="A"),
        ]
        plddt = np.array([0.2, 0.8, 0.3, 0.9], dtype=np.float32)
        pae = np.arange(16, dtype=np.float32).reshape(4, 4)
        pde = np.arange(16, dtype=np.float32).reshape(4, 4) + 100.0
        dialog._pred_data = types.SimpleNamespace(
            structure_plddt=plddt,
            plddt=None,
            pae=pae,
            pde=pde,
        )
        dialog._token_map = token_map
        dialog._get_obj_name = lambda: "target_model_0"
        dialog._ref_edit = _LineEdit("resname LIG")
        dialog._cutoff_edit = _LineEdit("7.5")
        dialog._build_token_map_if_needed = lambda _obj_name: None

        calls = []

        def fake_selection_to_token_indices(tm, selection, obj_name="all"):
            calls.append((tm, selection, obj_name))
            if selection == "resname LIG":
                return [0, 2]
            return [0, 1, 2, 3]

        plot_calls = []
        metadata_calls = []
        fake_plots = types.SimpleNamespace(
            MAX_BINDING_SITE_RESIDUES=40,
            make_binding_site_fingerprint=lambda *args, **kwargs: (
                plot_calls.append((args, kwargs)) or "fingerprint-figure"
            ),
            attach_viewer_selection_metadata=lambda *args, **kwargs: (
                metadata_calls.append((args, kwargs)) or args[0]
            ),
            save_and_show=lambda _fig: None,
        )
        viewer_calls = []
        fake_plot_viewer = types.SimpleNamespace(
            show_figure=lambda *args, **kwargs: (
                viewer_calls.append((args, kwargs)) or "plot-dialog"
            )
        )

        import FoldQC
        import FoldQC.gui_metrics as metrics_gui_module
        import FoldQC.gui_plots as plots_gui_module

        old_selection = plots_gui_module.selection_to_token_indices
        old_nearby = metrics_gui_module.tokens_within_distance
        nearby_calls = []
        old_plots = sys.modules.get("FoldQC.plots")
        old_plots_attr = getattr(FoldQC, "plots", None)
        old_plot_viewer = sys.modules.get("FoldQC.plot_viewer")
        old_plot_viewer_attr = getattr(FoldQC, "plot_viewer", None)
        sys.modules["FoldQC.plots"] = fake_plots
        FoldQC.plots = fake_plots
        sys.modules["FoldQC.plot_viewer"] = fake_plot_viewer
        FoldQC.plot_viewer = fake_plot_viewer
        try:
            plots_gui_module.selection_to_token_indices = (
                fake_selection_to_token_indices
            )
            metrics_gui_module.tokens_within_distance = lambda *args: (
                nearby_calls.append(args) or [0, 1, 2, 3]
            )

            dialog._show_binding_site_fingerprint()
        finally:
            plots_gui_module.selection_to_token_indices = old_selection
            metrics_gui_module.tokens_within_distance = old_nearby
            if old_plots_attr is None:
                try:
                    delattr(FoldQC, "plots")
                except AttributeError:
                    pass
            else:
                FoldQC.plots = old_plots_attr
            if old_plots is None:
                sys.modules.pop("FoldQC.plots", None)
            else:
                sys.modules["FoldQC.plots"] = old_plots
            if old_plot_viewer_attr is None:
                try:
                    delattr(FoldQC, "plot_viewer")
                except AttributeError:
                    pass
            else:
                FoldQC.plot_viewer = old_plot_viewer_attr
            if old_plot_viewer is None:
                sys.modules.pop("FoldQC.plot_viewer", None)
            else:
                sys.modules["FoldQC.plot_viewer"] = old_plot_viewer

        self.assertEqual(calls[0], (token_map, "resname LIG", "target_model_0"))
        self.assertEqual(
            nearby_calls[0], (token_map, "target_model_0", "resname LIG", 7.5)
        )
        self.assertEqual(len(plot_calls), 1)
        args, kwargs = plot_calls[0]
        self.assertIs(args[0], token_map)
        self.assertEqual(args[1], [1, 3])
        np.testing.assert_array_equal(kwargs["plddt"], plddt)
        np.testing.assert_allclose(
            kwargs["pae_to_ligand"],
            np.array([1.0, 5.0, 9.0, 13.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            kwargs["pae_from_ligand"],
            np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            kwargs["pde_to_ligand"],
            np.array([101.0, 105.0, 109.0, 113.0], dtype=np.float32),
        )
        self.assertEqual(len(viewer_calls), 1)
        self.assertEqual(viewer_calls[0][0][0], "fingerprint-figure")
        self.assertEqual(dialog._plot_windows, ["plot-dialog"])
        self.assertEqual(len(metadata_calls), 1)
        self.assertEqual(metadata_calls[0][1]["token_indices"], [1, 3])

    def test_fingerprint_warns_when_binding_site_residue_limit_is_exceeded(
        self,
    ) -> None:
        dialog = _new_dialog()
        token_map = [_token(i) for i in range(5)]
        plddt = np.linspace(0.5, 0.9, len(token_map), dtype=np.float32)
        dialog._pred_data = types.SimpleNamespace(
            rank=0,
            structure_plddt=plddt,
            plddt=None,
            pae=None,
            pde=None,
            contact_probs=None,
        )
        dialog._pred_files = types.SimpleNamespace(
            has_pae=False,
            has_pde=False,
            has_contact_probs=False,
        )
        dialog._token_map = token_map
        dialog._get_obj_name = lambda: "target_model_0"
        dialog._ref_edit = _LineEdit("chain B")
        dialog._cutoff_edit = _LineEdit("5.0")
        dialog._build_token_map_if_needed = lambda _obj_name: None
        dialog._reload_prediction_data = lambda *args, **kwargs: dialog._pred_data
        dialog._plot_windows = []

        def fake_selection_to_token_indices(tm, selection, obj_name="all"):
            if selection == "chain B":
                return [0]
            return [1, 2, 3, 4]

        plot_calls = []
        metadata_calls = []
        fake_plots = types.SimpleNamespace(
            MAX_BINDING_SITE_RESIDUES=2,
            make_binding_site_fingerprint=lambda *args, **kwargs: (
                plot_calls.append((args, kwargs)) or "fingerprint-figure"
            ),
            attach_viewer_selection_metadata=lambda *args, **kwargs: (
                metadata_calls.append((args, kwargs)) or args[0]
            ),
            save_and_show=lambda _fig: None,
        )
        viewer_calls = []
        fake_plot_viewer = types.SimpleNamespace(
            show_figure=lambda *args, **kwargs: (
                viewer_calls.append((args, kwargs)) or "plot-dialog"
            )
        )

        import FoldQC
        import FoldQC.gui_metrics as metrics_gui_module
        import FoldQC.gui_plots as plots_gui_module

        old_selection = plots_gui_module.selection_to_token_indices
        old_nearby = metrics_gui_module.tokens_within_distance
        old_plots = sys.modules.get("FoldQC.plots")
        old_plots_attr = getattr(FoldQC, "plots", None)
        old_plot_viewer = sys.modules.get("FoldQC.plot_viewer")
        old_plot_viewer_attr = getattr(FoldQC, "plot_viewer", None)
        sys.modules["FoldQC.plots"] = fake_plots
        FoldQC.plots = fake_plots
        sys.modules["FoldQC.plot_viewer"] = fake_plot_viewer
        FoldQC.plot_viewer = fake_plot_viewer
        try:
            plots_gui_module.selection_to_token_indices = (
                fake_selection_to_token_indices
            )
            metrics_gui_module.tokens_within_distance = lambda *_args: [1, 2, 3, 4]

            dialog._show_binding_site_fingerprint()
        finally:
            plots_gui_module.selection_to_token_indices = old_selection
            metrics_gui_module.tokens_within_distance = old_nearby
            if old_plots_attr is None:
                try:
                    delattr(FoldQC, "plots")
                except AttributeError:
                    pass
            else:
                FoldQC.plots = old_plots_attr
            if old_plots is None:
                sys.modules.pop("FoldQC.plots", None)
            else:
                sys.modules["FoldQC.plots"] = old_plots
            if old_plot_viewer_attr is None:
                try:
                    delattr(FoldQC, "plot_viewer")
                except AttributeError:
                    pass
            else:
                FoldQC.plot_viewer = old_plot_viewer_attr
            if old_plot_viewer is None:
                sys.modules.pop("FoldQC.plot_viewer", None)
            else:
                sys.modules["FoldQC.plot_viewer"] = old_plot_viewer

        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        self.assertEqual(len(msg.warnings), 1)
        self.assertIn("found 4 polymer residues", msg.warnings[0][1])
        self.assertIn("Only the first 2 residues", msg.warnings[0][1])
        self.assertEqual(len(plot_calls), 1)
        self.assertEqual(plot_calls[0][0][1], [1, 2, 3, 4])
        self.assertEqual(metadata_calls[0][1]["token_indices"], [1, 2])
        self.assertEqual(len(viewer_calls), 1)

    def test_show_plot_figure_uses_qt_viewer_and_keeps_reference(self) -> None:
        dialog = _new_dialog()
        dialog._plot_windows = []
        viewer_calls = []
        fake_plot_viewer = types.SimpleNamespace(
            show_figure=lambda *args, **kwargs: (
                viewer_calls.append((args, kwargs)) or "plot-dialog"
            )
        )

        import FoldQC

        old_plot_viewer = sys.modules.get("FoldQC.plot_viewer")
        old_plot_viewer_attr = getattr(FoldQC, "plot_viewer", None)
        sys.modules["FoldQC.plot_viewer"] = fake_plot_viewer
        FoldQC.plot_viewer = fake_plot_viewer
        try:
            dialog._show_plot_figure("figure", "Plot title")
        finally:
            if old_plot_viewer_attr is None:
                try:
                    delattr(FoldQC, "plot_viewer")
                except AttributeError:
                    pass
            else:
                FoldQC.plot_viewer = old_plot_viewer_attr
            if old_plot_viewer is None:
                sys.modules.pop("FoldQC.plot_viewer", None)
            else:
                sys.modules["FoldQC.plot_viewer"] = old_plot_viewer

        self.assertEqual(len(viewer_calls), 1)
        args, kwargs = viewer_calls[0]
        self.assertEqual(args, ("figure",))
        self.assertEqual(kwargs["title"], "Plot title")
        self.assertIs(kwargs["parent"], dialog)
        self.assertEqual(dialog._plot_windows, ["plot-dialog"])

        kwargs["on_close"]("plot-dialog")
        self.assertEqual(dialog._plot_windows, [])

    def test_show_plot_figure_falls_back_to_external_viewer(self) -> None:
        dialog = _new_dialog()
        dialog._plot_windows = []
        saved = []
        fake_plot_viewer = types.SimpleNamespace(
            show_figure=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("no qtagg")
            )
        )
        fake_plots = types.SimpleNamespace(
            save_and_show=lambda fig: saved.append(fig) or "/tmp/plot.png"
        )
        msg = _PYMOL.Qt.QtWidgets.QMessageBox
        msg.criticals.clear()

        import FoldQC

        old_plots = sys.modules.get("FoldQC.plots")
        old_plots_attr = getattr(FoldQC, "plots", None)
        old_plot_viewer = sys.modules.get("FoldQC.plot_viewer")
        old_plot_viewer_attr = getattr(FoldQC, "plot_viewer", None)
        sys.modules["FoldQC.plots"] = fake_plots
        FoldQC.plots = fake_plots
        sys.modules["FoldQC.plot_viewer"] = fake_plot_viewer
        FoldQC.plot_viewer = fake_plot_viewer
        try:
            dialog._show_plot_figure("figure", "Plot title")
        finally:
            if old_plots_attr is None:
                try:
                    delattr(FoldQC, "plots")
                except AttributeError:
                    pass
            else:
                FoldQC.plots = old_plots_attr
            if old_plots is None:
                sys.modules.pop("FoldQC.plots", None)
            else:
                sys.modules["FoldQC.plots"] = old_plots
            if old_plot_viewer_attr is None:
                try:
                    delattr(FoldQC, "plot_viewer")
                except AttributeError:
                    pass
            else:
                FoldQC.plot_viewer = old_plot_viewer_attr
            if old_plot_viewer is None:
                sys.modules.pop("FoldQC.plot_viewer", None)
            else:
                sys.modules["FoldQC.plot_viewer"] = old_plot_viewer

        self.assertEqual(saved, ["figure"])
        self.assertEqual(dialog._plot_windows, [])
        self.assertEqual(msg.criticals, [])

    def test_pde_contact_uses_all_atom_contact_selection_and_excludes_reference(
        self,
    ) -> None:
        dialog = _new_dialog()
        dialog._cutoff_edit = _LineEdit("8.25")
        pde = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 4.0, 5.0],
                [2.0, 4.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ],
            dtype=np.float32,
        )
        data = types.SimpleNamespace(
            pde=pde,
            confidence=None,
            plddt=None,
            structure_plddt=None,
        )
        token_map = [
            _token(0, chain_id="L", is_hetatm=True),
            _token(1, chain_id="A"),
            _token(2, chain_id="L", is_hetatm=True),
            _token(3, chain_id="A"),
        ]

        import FoldQC.gui_metrics as gui_module

        old_selection = gui_module.selection_to_token_indices
        old_nearby = gui_module.tokens_within_distance
        calls = []
        nearby_calls = []

        def fake_selection_to_token_indices(tm, selection, obj_name="all"):
            calls.append((tm, selection, obj_name))
            if selection == "resname LIG":
                return [0, 2]
            return [0, 1, 2, 3]

        try:
            gui_module.selection_to_token_indices = fake_selection_to_token_indices
            gui_module.tokens_within_distance = lambda *args: (
                nearby_calls.append(args) or [0, 1, 2, 3]
            )

            values = dialog._compute_property_for(
                "pde_contact", "resname LIG", data, token_map, "target_model_0"
            )
        finally:
            gui_module.selection_to_token_indices = old_selection
            gui_module.tokens_within_distance = old_nearby

        self.assertEqual(calls[0], (token_map, "resname LIG", "target_model_0"))
        self.assertEqual(
            nearby_calls[0], (token_map, "target_model_0", "resname LIG", 8.25)
        )
        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isnan(values[2]))
        np.testing.assert_allclose(values[[1, 3]], np.array([2.5, 4.5]))

    def test_pae_contact_uses_all_atom_contact_selection_and_symmetric_mean(
        self,
    ) -> None:
        dialog = _new_dialog()
        dialog._cutoff_edit = _LineEdit("8.25")
        pae = np.array(
            [
                [0.0, 1.0, 8.0, 10.0],
                [2.0, 0.0, 6.0, 12.0],
                [4.0, 3.0, 0.0, 14.0],
                [5.0, 7.0, 9.0, 0.0],
            ],
            dtype=np.float32,
        )
        data = types.SimpleNamespace(
            pae=pae,
            confidence=None,
            plddt=None,
            structure_plddt=None,
        )
        token_map = [
            _token(0, chain_id="L", is_hetatm=True),
            _token(1, chain_id="A"),
            _token(2, chain_id="L", is_hetatm=True),
            _token(3, chain_id="A"),
        ]

        import FoldQC.gui_metrics as gui_module

        old_selection = gui_module.selection_to_token_indices
        old_nearby = gui_module.tokens_within_distance
        calls = []
        nearby_calls = []

        def fake_selection_to_token_indices(tm, selection, obj_name="all"):
            calls.append((tm, selection, obj_name))
            if selection == "resname LIG":
                return [0, 2]
            return [0, 1, 2, 3]

        try:
            gui_module.selection_to_token_indices = fake_selection_to_token_indices
            gui_module.tokens_within_distance = lambda *args: (
                nearby_calls.append(args) or [0, 1, 2, 3]
            )

            values = dialog._compute_property_for(
                "pae_contact", "resname LIG", data, token_map, "target_model_0"
            )
        finally:
            gui_module.selection_to_token_indices = old_selection
            gui_module.tokens_within_distance = old_nearby

        self.assertEqual(calls[0], (token_map, "resname LIG", "target_model_0"))
        self.assertEqual(
            nearby_calls[0], (token_map, "target_model_0", "resname LIG", 8.25)
        )
        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isnan(values[2]))
        np.testing.assert_allclose(values[[1, 3]], np.array([3.0, 9.5]))

    def test_interaction_probability_to_selection_dispatches_from_gui(self) -> None:
        dialog = _new_dialog()
        contact_probs = np.array(
            [
                [1.0, 0.1, 0.4],
                [0.1, 1.0, 0.7],
                [0.4, 0.7, 1.0],
            ],
            dtype=np.float32,
        )
        data = types.SimpleNamespace(contact_probs=contact_probs)
        token_map = [_token(0), _token(1), _token(2)]

        import FoldQC.gui_metrics as gui_module

        old_selection = gui_module.selection_to_token_indices
        try:
            gui_module.selection_to_token_indices = lambda _tm, _sel, obj_name="all": [
                0,
                2,
            ]
            values = dialog._compute_property_for(
                "contact_prob_to_sel", "chain L", data, token_map, "target_model_0"
            )
        finally:
            gui_module.selection_to_token_indices = old_selection

        np.testing.assert_allclose(
            values,
            np.array([np.nan, 0.4, np.nan]),
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
