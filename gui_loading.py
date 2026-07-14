"""Prediction/model lifecycle and contextual GUI coordination."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import ensemble, gui_rules, metrics, plot_data, reports
from .compat import ItemIsEnabled, QtCore, QtWidgets, WindowCloseButtonHint
from .mol_viewer import (
    add_objects_to_group,
    delete_viewer_names,
    ensure_structure_object,
    get_group_members,
    get_object_list,
    get_viewer_name,
    inspect_object_tokens,
    load_structure_object_if_missing,
    rebuild,
    remove_objects_from_group,
    run_with_updates_suspended,
    transform_object,
    viewer_name_exists,
)

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .token_map import TokenMap

_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


@dataclass(frozen=True)
class InitialLoadResult:
    """Provider files and initial lazy data prepared by a background job."""

    pred_files: object
    pred_data: object
    token_map: TokenMap
    rank: int
    display_path: Path


@dataclass(frozen=True)
class ModelSwitchResult:
    """One ranked model prepared without touching Qt widgets or PyMOL."""

    pred_files: object
    rank: int
    data: object
    token_map: TokenMap
    _owns_prediction_files: bool = False


@dataclass(frozen=True)
class DataLoadItem:
    """One current-model or ensemble-member lazy reload request."""

    slot: str
    rank: int
    model_label: str
    flags: tuple[tuple[str, bool], ...]
    original_data: object
    member: object | None = None
    phase_arrays: tuple[str, ...] = ()

    def load_kwargs(self) -> dict[str, bool]:
        return dict(self.flags)


@dataclass(frozen=True)
class DataLoadBatchResult:
    """Atomically committable lazy data returned by one worker task."""

    pred_files: object
    loaded: tuple[tuple[DataLoadItem, object], ...]
    _owns_prediction_files: bool = False


@dataclass
class EnsembleViewerTransaction:
    """Main-thread state for an incrementally committed ensemble."""

    request_id: int
    prepared: ensemble.PreparedEnsemble
    previous_target: str = ""
    created_objects: list[str] = field(default_factory=list)
    inspections: dict[int, object] = field(default_factory=dict)
    applied_transforms: list[ensemble.AlignmentTransform] = field(default_factory=list)
    group_existed: bool = False
    previous_group_members: tuple[str, ...] = ()
    group_additions: tuple[str, ...] = ()
    previous_members: list | None = None
    previous_group_name: str | None = None
    previous_aligned: bool = False
    previous_rmsd: np.ndarray | None = None
    previous_plddt_mean: np.ndarray | None = None
    previous_plddt_std: np.ndarray | None = None


def _session_path_for_candidate(discovery, candidate) -> Path:
    input_path = getattr(discovery, "input_path", None)
    if input_path is not None:
        input_path = Path(input_path)
        if input_path.is_file():
            return input_path
    return Path(candidate.path)


def _discovery_phase(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(_ARCHIVE_SUFFIXES):
        return "Extracting archive and discovering predictions…"
    if lowered.endswith((".cif", ".pdb")):
        return "Inspecting structure file…"
    return "Discovering prediction folders…"


def _discover_prediction(path: str, report_phase):
    from .loader import discover_prediction_candidates

    report_phase(_discovery_phase(path))
    return discover_prediction_candidates(path)


def _scan_and_load_initial_prediction(
    discovery,
    candidate,
    preferred_rank: int | None,
    report_phase,
) -> InitialLoadResult:
    from .loader import load_prediction_data
    from .token_map import build_token_map

    display_path = _session_path_for_candidate(discovery, candidate)
    report_phase(f"Scanning {candidate.provider_label} output…")
    pred_files = discovery.scan(candidate)
    if not pred_files.models:
        raise ValueError("No ranked model files were found.")

    available_ranks = {model.rank for model in pred_files.models}
    rank = (
        preferred_rank
        if preferred_rank is not None and preferred_rank in available_ranks
        else pred_files.models[0].rank
    )
    model = pred_files.model(rank)
    report_phase(f"Loading {model.display_label} data…")
    pred_data = load_prediction_data(
        pred_files,
        rank,
        load_pae=False,
        load_pde=False,
        load_contact_probs=False,
    )
    report_phase(f"Preparing {model.display_label} token map…")
    token_map = build_token_map(pred_data.structure_path)
    return InitialLoadResult(pred_files, pred_data, token_map, rank, display_path)


def _load_rank_data(pred_files, rank: int, report_phase) -> ModelSwitchResult:
    from .loader import load_prediction_data
    from .token_map import build_token_map

    model = pred_files.model(rank)
    report_phase(f"Loading {model.display_label} data…")
    data = load_prediction_data(
        pred_files,
        rank,
        load_pae=False,
        load_pde=False,
        load_contact_probs=False,
    )
    report_phase(f"Preparing {model.display_label} token map…")
    token_map = build_token_map(data.structure_path)
    return ModelSwitchResult(pred_files, rank, data, token_map)


def _load_data_batch(pred_files, items: tuple[DataLoadItem, ...], report_phase):
    from .loader import load_prediction_data

    loaded = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        arrays = " and ".join(item.phase_arrays) or "metric data"
        suffix = f" ({index}/{total})" if total > 1 else ""
        report_phase(f"Loading {arrays} for {item.model_label}{suffix}…")
        data = load_prediction_data(
            pred_files,
            item.rank,
            **item.load_kwargs(),
        )
        loaded.append((item, data))
    return DataLoadBatchResult(pred_files, tuple(loaded))


def _prepare_ensemble_job(
    pred_files,
    skip_alignment: bool,
    existing_data_by_rank: dict[int, object],
    report_phase,
):
    return ensemble.prepare_ensemble(
        pred_files,
        skip_alignment=skip_alignment,
        existing_data_by_rank=existing_data_by_rank,
        report_phase=report_phase,
    )


def _score_table_has_values(value) -> bool:
    return isinstance(value, (dict, list)) and bool(value)


def _pair_score_table_has_values(value) -> bool:
    if isinstance(value, dict):
        return any(_score_table_has_values(row) for row in value.values())
    if isinstance(value, list):
        return any(_score_table_has_values(row) for row in value)
    return False


def _confidence_has_chain_iptm_metric_data(confidence) -> bool:
    if not isinstance(confidence, dict):
        return False
    return any(
        _score_table_has_values(confidence.get(key))
        for key in ("chains_iptm", "chain_iptm", "chains_ptm")
    ) or any(
        _pair_score_table_has_values(confidence.get(key))
        for key in ("pair_chains_iptm", "chain_pair_iptm")
    )


class GuiLoadingController:
    def _gui_job_is_busy(self) -> bool:
        return bool(
            getattr(self, "_loading_prediction", False)
            or getattr(self, "_loading_data", False)
        )

    def _next_gui_job_request_id(self) -> int:
        self._gui_job_request_id += 1
        return self._gui_job_request_id

    def _load_prediction_dir(self) -> None:
        """Start background discovery for the selected prediction path."""
        path = self._dir_edit.text().strip()
        if not path:
            return
        if self._gui_job_is_busy():
            return

        self._loading_prediction = True
        request_id = self._next_gui_job_request_id()
        self._prediction_load_request_id = request_id
        self._set_prediction_load_controls_enabled(False)
        self._schedule_load_progress(request_id, _discovery_phase(path))

        handle = self._job_runner.submit(
            request_id,
            lambda report: _discover_prediction(path, report),
            self._on_prediction_load_progress,
            self._on_prediction_discovery_ready,
            self._on_prediction_load_error,
        )
        if self._prediction_load_is_active(request_id):
            self._active_load_handle = handle

    def _session_path_for_loaded_candidate(self, discovery, candidate) -> Path:
        """Return the path to show/save after loading one discovery candidate."""
        return _session_path_for_candidate(discovery, candidate)

    def _prediction_load_is_active(self, request_id: int) -> bool:
        return bool(
            self._loading_prediction and request_id == self._prediction_load_request_id
        )

    def _data_load_is_active(self, request_id: int) -> bool:
        return bool(self._loading_data and request_id == self._data_load_request_id)

    def _load_progress_is_active(self, request_id: int) -> bool:
        return self._prediction_load_is_active(request_id) or self._data_load_is_active(
            request_id
        )

    def _on_prediction_load_progress(self, request_id: int, label: str) -> None:
        if not self._prediction_load_is_active(request_id):
            return
        dialog = self._ensure_load_progress_dialog()
        dialog.setLabelText(label)

    def _on_prediction_discovery_ready(self, request_id: int, discovery) -> None:
        if not self._prediction_load_is_active(request_id):
            self._job_runner.dispose(discovery)
            return
        self._active_load_handle = None

        if len(discovery.candidates) == 1:
            candidate = discovery.candidates[0]
        else:
            self._pause_load_progress()
            candidate = self._choose_prediction_candidate(discovery.candidates)
        if candidate is None:
            self._job_runner.dispose(discovery)
            self._finish_prediction_load(request_id)
            return

        if len(discovery.candidates) != 1:
            self._schedule_load_progress(
                request_id,
                f"Scanning {candidate.provider_label} output…",
            )
        else:
            self._on_prediction_load_progress(
                request_id,
                f"Scanning {candidate.provider_label} output…",
            )

        preferred_rank = getattr(self._pending_session_restore, "model_rank", None)
        handle = self._job_runner.submit(
            request_id,
            lambda report: _scan_and_load_initial_prediction(
                discovery,
                candidate,
                preferred_rank,
                report,
            ),
            self._on_prediction_load_progress,
            self._on_initial_prediction_ready,
            self._on_prediction_load_error,
        )
        if self._prediction_load_is_active(request_id):
            self._active_load_handle = handle

    def _on_initial_prediction_ready(
        self, request_id: int, result: InitialLoadResult
    ) -> None:
        if not self._prediction_load_is_active(request_id):
            self._job_runner.dispose(result)
            return
        self._active_load_handle = None
        structure_name = Path(result.pred_data.structure_path).name
        self._on_prediction_load_progress(
            request_id,
            f"Loading {structure_name} into PyMOL…",
        )
        QtCore.QTimer.singleShot(
            0,
            lambda: self._commit_initial_prediction(request_id, result),
        )

    def _commit_initial_prediction(
        self, request_id: int, result: InitialLoadResult
    ) -> None:
        if not self._prediction_load_is_active(request_id):
            self._job_runner.dispose(result)
            return

        model = result.pred_files.model(result.rank)
        obj_name = model.object_name
        structure_path = result.pred_files.structure_path(result.rank)
        try:
            did_load = ensure_structure_object(
                structure_path,
                obj_name,
                zoom=True,
            )
        except Exception as exc:
            logger.exception("Could not load the initial prediction model into PyMOL")
            self._job_runner.dispose(result)
            self._finish_prediction_load(request_id)
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                f"Could not load or show {structure_path.name}:\n{exc}",
            )
            return

        try:
            self._pred_files = result.pred_files
            self._ensemble_members = None
            self._ensemble_group_name = None
            self._ensemble_aligned = False
            self._ensemble_rmsd = None
            self._ensemble_plddt_mean = None
            self._ensemble_plddt_std = None
            self._dir_edit.setText(str(result.display_path))

            self._model_combo.blockSignals(True)
            try:
                self._model_combo.clear()
                for model in self._pred_files.models:
                    self._model_combo.addItem(model.display_label, model.rank)
                self._select_model_rank(result.rank)
            finally:
                self._model_combo.blockSignals(False)

            self._pending_session_restore.model_rank = None
            self._activate_model_data(
                result.rank,
                result.pred_data,
                prepared_object=(obj_name, did_load),
                prepared_token_map=result.token_map,
            )
        except Exception as exc:
            logger.exception("Could not activate the initial prediction model")
            self._finish_prediction_load(request_id)
            QtWidgets.QMessageBox.warning(self, APP_TITLE, str(exc))
            return

        self._finish_prediction_load(request_id, save_session=True)

    def _on_prediction_load_error(self, request_id: int, failure) -> None:
        if not self._prediction_load_is_active(request_id):
            return
        logger.error("Background prediction load failed:\n%s", failure.traceback_text)
        self._finish_prediction_load(request_id)
        QtWidgets.QMessageBox.warning(self, APP_TITLE, failure.message)

    def _finish_prediction_load(
        self,
        request_id: int,
        *,
        save_session: bool = False,
    ) -> None:
        if request_id != self._prediction_load_request_id:
            return
        self._active_load_handle = None
        self._loading_prediction = False
        self._hide_load_progress()
        self._set_prediction_load_controls_enabled(True)
        self._refresh_contextual_ui()
        if save_session:
            self._save_session_settings()

    def _abandon_prediction_load(self) -> None:
        """Detach the dialog from a running job without blocking for completion."""
        self._abandon_active_gui_job()

    def _abandon_active_gui_job(self) -> None:
        """Invalidate any active GUI job without waiting for its worker."""
        handle = getattr(self, "_active_load_handle", None)
        if handle is not None:
            handle.abandon()
        if getattr(self, "_active_ensemble_viewer_transaction", None) is not None:
            self._rollback_ensemble_viewer_transaction(refresh_gui=False)
        request_id = self._next_gui_job_request_id()
        self._prediction_load_request_id = request_id
        self._data_load_request_id = request_id
        self._active_load_handle = None
        self._active_data_continuation = None
        self._model_switch_previous_data = None
        self._model_switch_previous_token_context = None
        self._loading_prediction = False
        self._loading_data = False
        self._hide_load_progress()
        self._set_prediction_load_controls_enabled(True)

    def _set_prediction_load_controls_enabled(self, enabled: bool) -> None:
        names = (
            "_dir_edit",
            "_dir_btn",
            "_file_btn",
            "_model_combo",
            "_obj_combo",
            "_obj_refresh_btn",
            "_prop_combo",
            "_ref_edit",
            "_cutoff_edit",
            "_palette_combo",
            "_palette_reverse_chk",
            "_vmin_edit",
            "_vmax_edit",
            "_apply_btn",
            "_plot_btn",
            "_export_csv_btn",
            "_ensemble_btn",
        )
        for name in names:
            widget = getattr(self, name, None)
            if widget is not None and hasattr(widget, "setEnabled"):
                widget.setEnabled(enabled)
        if not enabled:
            for action in getattr(self, "_plot_actions", {}).values():
                action.setEnabled(False)

    def _ensure_load_progress_dialog(self):
        dialog = getattr(self, "_load_progress_dialog", None)
        if dialog is not None:
            return dialog
        dialog = QtWidgets.QProgressDialog(self)
        dialog.setWindowTitle(f"{APP_TITLE} – Loading")
        dialog.setModal(False)
        dialog.setRange(0, 0)
        dialog.setCancelButton(None)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        if hasattr(dialog, "setWindowFlag"):
            dialog.setWindowFlag(WindowCloseButtonHint, False)
        self._load_progress_dialog = dialog
        return dialog

    def _schedule_load_progress(self, request_id: int, label: str) -> None:
        dialog = self._ensure_load_progress_dialog()
        dialog.setLabelText(label)
        self._progress_show_generation = (
            getattr(self, "_progress_show_generation", 0) + 1
        )
        generation = self._progress_show_generation
        QtCore.QTimer.singleShot(
            300,
            lambda: self._show_load_progress(request_id, generation),
        )

    def _show_load_progress(self, request_id: int, generation: int) -> None:
        if not self._load_progress_is_active(request_id):
            return
        if generation != self._progress_show_generation:
            return
        dialog = self._ensure_load_progress_dialog()
        dialog.show()
        if hasattr(dialog, "raise_"):
            dialog.raise_()

    def _pause_load_progress(self) -> None:
        self._progress_show_generation = (
            getattr(self, "_progress_show_generation", 0) + 1
        )
        dialog = getattr(self, "_load_progress_dialog", None)
        if dialog is not None:
            dialog.hide()

    def _hide_load_progress(self) -> None:
        self._pause_load_progress()
        dialog = getattr(self, "_load_progress_dialog", None)
        if dialog is not None:
            dialog.reset()
            dialog.hide()

    def _choose_prediction_candidate(self, candidates):
        """Let the user pick one prediction directory from multiple candidates."""
        if not candidates:
            return None
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select prediction")
        if hasattr(dialog, "setModal"):
            dialog.setModal(True)
        if hasattr(dialog, "setMinimumWidth"):
            dialog.setMinimumWidth(520)

        layout = QtWidgets.QVBoxLayout(dialog)
        table = QtWidgets.QTableWidget(len(candidates), 2, dialog)
        table.setHorizontalHeaderLabels(["Directory", "Provider"])
        for row, candidate in enumerate(candidates):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(candidate.relative_path))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(candidate.provider_label))
        if hasattr(table, "setCurrentCell"):
            table.setCurrentCell(0, 0)
        if hasattr(table, "resizeColumnsToContents"):
            table.resizeColumnsToContents()
        header = (
            table.horizontalHeader() if hasattr(table, "horizontalHeader") else None
        )
        if header is not None and hasattr(header, "setStretchLastSection"):
            header.setStretchLastSection(True)
        layout.addWidget(table)

        button_box_cls = QtWidgets.QDialogButtonBox
        standard_button = getattr(button_box_cls, "StandardButton", button_box_cls)
        button_box = button_box_cls(standard_button.Ok | standard_button.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        exec_result = dialog.exec()
        dialog_code = getattr(QtWidgets.QDialog, "DialogCode", QtWidgets.QDialog)
        accepted = getattr(dialog_code, "Accepted", 1)
        if exec_result != accepted:
            return None
        row = table.currentRow() if hasattr(table, "currentRow") else 0
        if row < 0:
            row = 0
        return candidates[row]

    def _expected_object_name(self, rank: int) -> str:
        """Return the canonical viewer object name for one model rank."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        try:
            return self._pred_files.model(rank).object_name
        except Exception:
            return f"{self._pred_files.name}_model_{rank}"

    def _ensure_model_object(self, rank: int, *, paint: bool = True) -> str | None:
        """Load or enable the viewer object for *rank*, then select it."""
        if self._pred_files is None or not self._pred_files.models:
            return None
        obj_name = self._expected_object_name(rank)
        path = self._pred_files.structure_path(rank)
        try:
            did_load = ensure_structure_object(path, obj_name, zoom=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, f"Could not load or show {path.name}:\n{exc}"
            )
            return None

        self._activate_prepared_model_object(obj_name, did_load, paint=paint)
        return obj_name

    def _activate_prepared_model_object(
        self,
        obj_name: str,
        did_load: bool,
        *,
        paint: bool = True,
    ) -> None:
        """Select and optionally paint an object already ensured in PyMOL."""
        self._refresh_objects()
        self._select_object(obj_name)
        if paint and did_load:
            try:
                self._apply_plddt_class_coloring("plddt_class", obj_name)
            except Exception:
                pass  # coloring failure must not abort model selection

    def _on_model_changed(self) -> None:
        """Start a transactional background load for the selected rank."""
        if self._pred_files is None:
            return
        rank = self._model_combo.currentData()
        if rank is None:
            return
        committed_rank = getattr(self._pred_data, "rank", None)
        if rank == committed_rank or self._gui_job_is_busy():
            return

        pred_files = self._pred_files
        self._model_switch_previous_data = self._pred_data
        self._model_switch_previous_token_context = self._capture_token_context()
        self._loading_data = True
        request_id = self._next_gui_job_request_id()
        self._data_load_request_id = request_id
        self._set_prediction_load_controls_enabled(False)
        model = pred_files.model(rank)
        self._schedule_load_progress(
            request_id,
            f"Loading {model.display_label} data…",
        )
        handle = self._job_runner.submit(
            request_id,
            lambda report: _load_rank_data(pred_files, rank, report),
            self._on_data_load_progress,
            self._on_model_switch_ready,
            self._on_model_switch_error,
        )
        if self._data_load_is_active(request_id):
            self._active_load_handle = handle

    def _on_data_load_progress(self, request_id: int, label: str) -> None:
        if not self._data_load_is_active(request_id):
            return
        self._ensure_load_progress_dialog().setLabelText(label)

    def _on_model_switch_ready(
        self,
        request_id: int,
        result: ModelSwitchResult,
    ) -> None:
        if not self._data_load_is_active(request_id):
            self._job_runner.dispose(result)
            return
        if result.pred_files is not self._pred_files:
            self._job_runner.dispose(result)
            self._finish_data_load(request_id)
            return
        self._active_load_handle = None
        model = result.pred_files.model(result.rank)
        self._on_data_load_progress(
            request_id,
            f"Loading {model.display_label} into PyMOL…",
        )
        QtCore.QTimer.singleShot(
            0,
            lambda: self._commit_model_switch(request_id, result),
        )

    def _commit_model_switch(
        self,
        request_id: int,
        result: ModelSwitchResult,
    ) -> None:
        if not self._data_load_is_active(request_id):
            self._job_runner.dispose(result)
            return
        if result.pred_files is not self._pred_files:
            self._job_runner.dispose(result)
            self._finish_data_load(request_id)
            return

        model = result.pred_files.model(result.rank)
        structure_path = result.pred_files.structure_path(result.rank)
        try:
            did_load = ensure_structure_object(
                structure_path,
                model.object_name,
                zoom=True,
            )
        except Exception as exc:
            logger.exception("Could not load the selected prediction model into PyMOL")
            self._rollback_model_switch()
            self._finish_data_load(request_id, save_session=True)
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                f"Could not load or show {structure_path.name}:\n{exc}",
            )
            return

        try:
            self._activate_model_data(
                result.rank,
                result.data,
                prepared_object=(model.object_name, did_load),
                prepared_token_map=result.token_map,
            )
        except Exception as exc:
            logger.exception("Could not activate the selected prediction model")
            self._rollback_model_switch()
            self._finish_data_load(request_id, save_session=True)
            QtWidgets.QMessageBox.warning(self, APP_TITLE, str(exc))
            return

        self._finish_data_load(request_id, save_session=True)

    def _on_model_switch_error(self, request_id: int, failure) -> None:
        if not self._data_load_is_active(request_id):
            return
        logger.error("Background model switch failed:\n%s", failure.traceback_text)
        self._rollback_model_switch()
        self._finish_data_load(request_id, save_session=True)
        QtWidgets.QMessageBox.warning(self, APP_TITLE, failure.message)

    def _restore_committed_model_rank(self) -> None:
        rank = getattr(self._pred_data, "rank", None)
        if rank is None:
            return
        self._model_combo.blockSignals(True)
        try:
            self._select_model_rank(rank)
        finally:
            self._model_combo.blockSignals(False)

    def _rollback_model_switch(self) -> None:
        previous = getattr(self, "_model_switch_previous_data", None)
        restored_data = previous is not None and self._pred_data is not previous
        if restored_data:
            self._pred_data = previous
        token_context = getattr(self, "_model_switch_previous_token_context", None)
        if token_context is not None:
            self._restore_token_context(token_context)
        if restored_data:
            self._update_confidence_summary()
            self._update_property_availability()
        self._restore_committed_model_rank()

    def _finish_data_load(
        self,
        request_id: int,
        *,
        save_session: bool = False,
    ) -> None:
        if request_id != self._data_load_request_id:
            return
        self._active_load_handle = None
        self._active_data_continuation = None
        self._model_switch_previous_data = None
        self._model_switch_previous_token_context = None
        self._loading_data = False
        self._hide_load_progress()
        self._set_prediction_load_controls_enabled(True)
        self._refresh_contextual_ui()
        if save_session:
            self._save_session_settings()

    def _defer_action_for_data(
        self,
        target,
        requested_flags: dict[str, bool],
        continuation,
        *,
        error_title: str,
    ) -> bool:
        """Submit missing lazy arrays and resume *continuation* after commit."""
        if self._pred_files is None:
            return False
        items = self._data_load_items_for_target(target, requested_flags)
        if not items:
            return False
        if self._gui_job_is_busy():
            return True

        pred_files = self._pred_files
        self._loading_data = True
        request_id = self._next_gui_job_request_id()
        self._data_load_request_id = request_id
        self._active_data_continuation = continuation
        self._active_data_error_title = error_title
        self._set_prediction_load_controls_enabled(False)
        first = items[0]
        arrays = " and ".join(first.phase_arrays) or "metric data"
        self._schedule_load_progress(
            request_id,
            f"Loading {arrays} for {first.model_label}…",
        )
        batch = tuple(items)
        handle = self._job_runner.submit(
            request_id,
            lambda report: _load_data_batch(pred_files, batch, report),
            self._on_data_load_progress,
            self._on_lazy_data_ready,
            self._on_lazy_data_error,
        )
        if self._data_load_is_active(request_id):
            self._active_load_handle = handle
        return True

    def _data_load_items_for_target(
        self,
        target,
        requested_flags: dict[str, bool],
    ) -> list[DataLoadItem]:
        if target is None:
            return []
        slots = []
        if target.kind == "single":
            if target.data is self._pred_data and self._pred_data is not None:
                slots.append(("current", self._pred_data, None))
        elif target.kind == "ensemble_member":
            for member in target.members or []:
                slots.append(("member", member.data, member))
        elif target.kind == "ensemble_group":
            for member in sorted(target.members or [], key=lambda item: item.rank):
                slots.append(("member", member.data, member))

        items = []
        seen = set()
        for slot, data, member in slots:
            key = (slot, id(member) if member is not None else 0)
            if key in seen:
                continue
            seen.add(key)
            item = self._data_load_item(
                slot,
                data,
                member,
                requested_flags,
            )
            if item is not None:
                items.append(item)
        return items

    def _data_load_item(
        self,
        slot: str,
        data,
        member,
        requested_flags: dict[str, bool],
    ) -> DataLoadItem | None:
        flag_attrs = {
            "load_pae": ("pae", "PAE"),
            "load_pde": ("pde", "PDE"),
            "load_contact_probs": ("contact_probs", "interaction probabilities"),
            "load_token_plddt": ("token_plddt", "pLDDT"),
        }
        requested = {
            name: bool(requested_flags.get(name, False)) for name in flag_attrs
        }
        missing = [
            name
            for name, (attr, _label) in flag_attrs.items()
            if requested[name] and getattr(data, attr, None) is None
        ]
        if not missing:
            return None

        flags = {
            name: requested[name] or getattr(data, attr, None) is not None
            for name, (attr, _label) in flag_attrs.items()
        }
        phase_arrays = tuple(dict.fromkeys(flag_attrs[name][1] for name in missing))
        rank = int(member.rank if member is not None else data.rank)
        model = self._pred_files.model(rank)
        return DataLoadItem(
            slot=slot,
            rank=rank,
            model_label=model.display_label,
            flags=tuple(flags.items()),
            original_data=data,
            member=member,
            phase_arrays=phase_arrays,
        )

    def _on_lazy_data_ready(
        self,
        request_id: int,
        result: DataLoadBatchResult,
    ) -> None:
        if not self._data_load_is_active(request_id):
            self._job_runner.dispose(result)
            return
        if (
            result.pred_files is not self._pred_files
            or not self._lazy_result_is_current(result)
        ):
            self._job_runner.dispose(result)
            self._finish_data_load(request_id)
            return

        try:
            for item, data in result.loaded:
                self._validate_lazy_loaded_item(item, data)
        except Exception as exc:
            title = getattr(
                self,
                "_active_data_error_title",
                f"{APP_TITLE} - error",
            )
            self._finish_data_load(request_id)
            QtWidgets.QMessageBox.critical(self, title, str(exc))
            return

        self._active_load_handle = None
        for item, data in result.loaded:
            if item.slot == "current":
                self._pred_data = data
            else:
                item.member.data = data

        continuation = self._active_data_continuation
        self._on_data_load_progress(request_id, "Preparing requested action…")
        QtCore.QTimer.singleShot(
            0,
            lambda: self._resume_lazy_action(request_id, continuation),
        )

    def _lazy_result_is_current(self, result: DataLoadBatchResult) -> bool:
        for item, _data in result.loaded:
            if item.slot == "current":
                if self._pred_data is not item.original_data:
                    return False
            elif item.member is None or item.member.data is not item.original_data:
                return False
            elif item.member not in (self._ensemble_members or []):
                return False
        return True

    def _validate_lazy_loaded_item(self, item: DataLoadItem, data) -> None:
        flags = item.load_kwargs()
        fields = (
            ("load_pae", "pae", "PAE"),
            ("load_pde", "pde", "PDE"),
            ("load_contact_probs", "contact_probs", "interaction probabilities"),
            ("load_token_plddt", "token_plddt", "pLDDT"),
        )
        missing = [
            label
            for flag, attr, label in fields
            if flags.get(flag, False) and getattr(data, attr, None) is None
        ]
        if missing:
            raise ValueError(
                f"{item.model_label} did not provide required prediction data: "
                + ", ".join(missing)
            )

    def _resume_lazy_action(self, request_id: int, continuation) -> None:
        if not self._data_load_is_active(request_id):
            return
        try:
            if continuation is not None:
                continuation()
        except Exception as exc:
            logger.exception("Could not resume the requested action")
            title = getattr(
                self,
                "_active_data_error_title",
                f"{APP_TITLE} - error",
            )
            QtWidgets.QMessageBox.critical(self, title, str(exc))
        finally:
            self._finish_data_load(request_id)

    def _on_lazy_data_error(self, request_id: int, failure) -> None:
        if not self._data_load_is_active(request_id):
            return
        logger.error("Background lazy-data load failed:\n%s", failure.traceback_text)
        title = getattr(
            self,
            "_active_data_error_title",
            f"{APP_TITLE} - error",
        )
        self._finish_data_load(request_id)
        QtWidgets.QMessageBox.critical(self, title, failure.message)

    def _activate_model_data(
        self,
        rank: int,
        data,
        *,
        prepared_object: tuple[str, bool] | None = None,
        prepared_token_map: TokenMap | None = None,
    ) -> None:
        """Commit loaded model data and perform main-thread viewer/UI updates."""
        obj_name = (
            prepared_object[0]
            if prepared_object is not None
            else self._expected_object_name(rank)
        )
        self._pred_data = data
        self._clear_token_map_cache()
        if prepared_token_map is not None:
            self._token_map = prepared_token_map
            self._token_map_obj = obj_name
            self._token_map_structure_path = data.structure_path
        self._update_confidence_summary()
        self._update_property_availability()
        pending_metric = getattr(self._pending_session_restore, "metric_key", None)
        if pending_metric:
            if not self._select_property_if_available(pending_metric):
                self._select_first_available_property()
            self._pending_session_restore.metric_key = None
        else:
            self._select_first_available_property()
        if prepared_object is None:
            self._ensure_model_object(rank, paint=True)
        else:
            prepared_obj_name, did_load = prepared_object
            self._activate_prepared_model_object(
                prepared_obj_name, did_load, paint=True
            )
        pending_target = getattr(self._pending_session_restore, "target_name", None)
        if pending_target and self._combo_contains_text(
            self._obj_combo, pending_target
        ):
            self._select_object(pending_target)
            self._pending_session_restore.target_name = None
        self._refresh_contextual_ui()

    def _refresh_objects(self) -> None:
        """Re-populate the molecular-viewer target dropdown."""
        try:
            additional = (
                [self._ensemble_group_name] if self._ensemble_group_name else []
            )
            names = get_object_list(additional_names=additional)
            names = self._ordered_target_names(names)
        except Exception:
            names = []

        self._obj_combo.blockSignals(True)
        self._obj_combo.clear()
        for n in names:
            self._obj_combo.addItem(n)
            self._style_target_combo_item(self._obj_combo.count() - 1, n)
        pending_target = getattr(self._pending_session_restore, "target_name", None)
        if pending_target and self._combo_contains_text(
            self._obj_combo, pending_target
        ):
            self._select_object(pending_target)
            if not getattr(self, "_loading_prediction", False):
                self._pending_session_restore.target_name = None
        self._obj_combo.blockSignals(False)
        self._refresh_contextual_ui()

    def _ordered_target_names(self, names: list[str]) -> list[str]:
        """Return target names in stable display order."""
        group_name = self._ensemble_group_name
        members = sorted(self._ensemble_members or [], key=lambda member: member.rank)
        member_names = [member.obj_name for member in members]

        name_set = set(names)
        ordered = []
        if group_name in name_set:
            ordered.append(group_name)
        ordered.extend(name for name in member_names if name in name_set)

        handled = set(ordered)
        ordered.extend(
            sorted((name for name in names if name not in handled), key=str.casefold)
        )
        return ordered

    def _style_target_combo_item(self, row: int, name: str) -> None:
        """Visually distinguish the ensemble group in the target dropdown."""
        if name != self._ensemble_group_name:
            return
        item = self._obj_combo.model().item(row)
        if item is None:
            return
        font = item.font()
        font.setBold(True)
        font.setItalic(True)
        item.setFont(font)

    def _on_property_changed(self) -> None:
        """Refresh controls whose meaning depends on the selected property."""
        self._ref_label.setVisible(True)
        self._ref_edit.setVisible(True)
        self._refresh_contextual_ui()

    def _update_confidence_summary(self) -> None:
        """Fill the confidence text browser from loaded data."""
        self._conf_browser.setPlainText(
            reports.format_confidence_summary(self._pred_data)
        )

    def _update_property_availability(self) -> None:
        """Grey out combo items whose required data is not available."""
        if self._pred_data is None or self._pred_files is None:
            return
        has_pae = getattr(self._pred_files, "has_pae", False)
        has_pde = getattr(self._pred_files, "has_pde", False)
        has_contact_probs = getattr(self._pred_files, "has_contact_probs", False)
        has_plddt = (
            getattr(self._pred_files, "has_plddt", False)
            or getattr(self._pred_data, "token_plddt", None) is not None
        )
        has_confidence = (
            getattr(self._pred_data, "confidence", None) is not None
            or getattr(self._pred_data, "summary_confidence", None) is not None
        )
        has_chain_iptm = self._has_chain_iptm_metric_data()
        has_ensemble = bool(getattr(self, "_ensemble_members", None))

        model = self._prop_combo.model()
        for row, prop in enumerate(metrics.PROPERTIES):
            combo_row = self._property_combo_row(prop["key"])
            if combo_row is None:
                continue
            available = True
            if prop["needs_pae"] and not has_pae:
                available = False
            if prop["needs_pde"] and not has_pde:
                available = False
            if prop.get("needs_plddt", False) and not has_plddt:
                available = False
            if prop.get("needs_contact_probs", False) and not has_contact_probs:
                available = False
            if prop.get("needs_confidence", False) and not has_confidence:
                available = False
            if prop["key"] == "chain_iptm" and not has_chain_iptm:
                available = False
            if prop.get("ensemble_level", False) and not has_ensemble:
                available = False
            item = model.item(combo_row)
            if item is not None:
                flags = item.flags()
                if available:
                    item.setFlags(flags | ItemIsEnabled)
                else:
                    item.setFlags(flags & ~ItemIsEnabled)

    def _has_chain_iptm_metric_data(self) -> bool:
        """Return whether loaded confidence has data for the Chain ipTM metric."""
        if self._pred_data is None:
            return False
        for attr in ("confidence", "summary_confidence"):
            confidence = getattr(self._pred_data, attr, None)
            if _confidence_has_chain_iptm_metric_data(confidence):
                return True
        return False

    def _select_first_available_property(self) -> None:
        """Move the property combo away from a disabled item after loading."""
        model = self._prop_combo.model()
        current = self._prop_combo.currentIndex()
        if current >= 0:
            item = model.item(current)
            if item is not None and item.flags() & ItemIsEnabled:
                return
        for prop in metrics.PROPERTIES:
            row = self._property_combo_row(prop["key"])
            if row is None:
                continue
            item = model.item(row)
            if item is not None and item.flags() & ItemIsEnabled:
                self._prop_combo.setCurrentIndex(row)
                return

    def _clear_token_map_cache(self) -> None:
        """Drop token and viewer mapping state after changing prediction context."""
        self._token_map = None
        self._token_map_obj = None  # type: ignore[attr-defined]
        self._token_map_structure_path = None  # type: ignore[attr-defined]
        self._paint_mappings = {}
        self._accepted_token_overlap_warnings = set()

    def _capture_token_context(self) -> tuple:
        """Snapshot token and viewer mappings for transactional model switching."""
        return (
            self._token_map,
            self._token_map_obj,
            self._token_map_structure_path,
            dict(self._paint_mappings),
            set(self._accepted_token_overlap_warnings),
        )

    def _restore_token_context(self, context: tuple) -> None:
        """Restore a token-context snapshot after a failed model switch."""
        (
            self._token_map,
            self._token_map_obj,
            self._token_map_structure_path,
            paint_mappings,
            accepted_warnings,
        ) = context
        self._paint_mappings = paint_mappings
        self._accepted_token_overlap_warnings = accepted_warnings

    def _property_combo_row(self, key: str) -> int | None:
        """Return the combo row registered for a metric key."""
        return self._prop_combo_rows.get(key)

    def _current_target_kind(self) -> str:
        """Return a lightweight target kind without resolving token maps or loading data."""
        try:
            obj_name = self._get_obj_name()
        except Exception:
            obj_name = None
        if not obj_name:
            return "none"
        if obj_name == getattr(self, "_ensemble_group_name", None):
            return "ensemble_group"
        if self._selected_ensemble_member(obj_name) is not None:
            return "ensemble_member"
        return "single"

    def _has_fingerprint_data(self) -> bool:
        """Return whether fingerprint plotting has any source family available."""
        pred_files = getattr(self, "_pred_files", None)
        pred_data = getattr(self, "_pred_data", None)
        if pred_files is not None:
            if (
                getattr(pred_files, "has_pae", False)
                or getattr(pred_files, "has_pde", False)
                or getattr(pred_files, "has_contact_probs", False)
                or getattr(pred_files, "has_plddt", False)
            ):
                return True
        if pred_data is not None:
            if (
                getattr(pred_data, "pae", None) is not None
                or getattr(pred_data, "pde", None) is not None
                or getattr(pred_data, "contact_probs", None) is not None
                or getattr(pred_data, "token_plddt", None) is not None
            ):
                return True
        for member in getattr(self, "_ensemble_members", None) or []:
            data = getattr(member, "data", None)
            if data is None:
                continue
            if (
                getattr(data, "pae", None) is not None
                or getattr(data, "pde", None) is not None
                or getattr(data, "contact_probs", None) is not None
                or getattr(data, "token_plddt", None) is not None
            ):
                return True
        return False

    def _has_matrix_data_family(self, family: str) -> bool:
        """Return whether a matrix family is available from files or loaded data."""
        pred_files = getattr(self, "_pred_files", None)
        pred_data = getattr(self, "_pred_data", None)
        if family == "pae":
            return bool(
                getattr(pred_files, "has_pae", False)
                or getattr(pred_data, "pae", None) is not None
            )
        if family == "pde":
            return bool(
                getattr(pred_files, "has_pde", False)
                or getattr(pred_data, "pde", None) is not None
            )
        return False

    def _current_target_has_multiple_chains(self) -> bool:
        """Return whether the current target token map has multiple chains."""
        try:
            obj_name = self._get_obj_name()
        except Exception:
            obj_name = None
        if not obj_name:
            return False

        if obj_name == getattr(self, "_ensemble_group_name", None):
            members = getattr(self, "_ensemble_members", None) or []
            if not members:
                return False
            return plot_data.has_multiple_token_chains(members[0].token_map)

        member = self._selected_ensemble_member(obj_name)
        if member is not None:
            return plot_data.has_multiple_token_chains(member.token_map)

        if getattr(self, "_pred_data", None) is None:
            return False
        try:
            self._build_token_map_if_needed(obj_name)
        except Exception:
            return False
        return plot_data.has_multiple_token_chains(self._token_map)

    def _update_plot_actions(self) -> None:
        """Refresh plot menu action availability from current GUI state."""
        actions = getattr(self, "_plot_actions", None)
        if not actions:
            return
        metric_key = self._prop_combo.currentData()
        target_kind = self._current_target_kind()
        has_reference = bool(self._ref_edit.text().strip())
        has_ensemble = bool(getattr(self, "_ensemble_members", None))
        has_fingerprint_data = self._has_fingerprint_data()
        has_pae_data = self._has_matrix_data_family("pae")
        has_pde_data = self._has_matrix_data_family("pde")
        has_multiple_chains = self._current_target_has_multiple_chains()
        for label, plot_type in metrics.PLOT_TYPES:
            action = actions.get(plot_type)
            if action is None:
                continue
            state = gui_rules.plot_action_state(
                plot_type,
                metric_key,
                target_kind,
                has_reference,
                has_ensemble,
                has_fingerprint_data=has_fingerprint_data,
                has_pae_data=has_pae_data,
                has_pde_data=has_pde_data,
                has_multiple_chains=has_multiple_chains,
            )
            action.setEnabled(state.enabled)
            tip = state.reason or f"Show {label.lower()}."
            if hasattr(action, "setToolTip"):
                action.setToolTip(tip)
            if hasattr(action, "setStatusTip"):
                action.setStatusTip(tip)

    def _refresh_contextual_ui(self) -> None:
        """Refresh plot actions, contextual fields, and preview text together."""
        self._update_plot_actions()
        self._update_context_controls()
        self._update_metric_preview()

    def _update_context_controls(self) -> None:
        """Apply contextual Reference and cutoff control states."""
        key = self._prop_combo.currentData()
        context = gui_rules.field_context(
            key,
            self._current_target_kind(),
            bool(getattr(self, "_ensemble_members", None)),
            self._has_fingerprint_data(),
        )
        self._ref_label.setText(context.ref_label)
        self._ref_label.setToolTip(context.ref_tooltip)
        self._ref_edit.setEnabled(context.ref_enabled)
        self._ref_edit.setToolTip(context.ref_tooltip)
        self._cutoff_label.setText(context.cutoff_label)
        self._cutoff_label.setToolTip(context.cutoff_tooltip)
        self._cutoff_edit.setEnabled(context.cutoff_enabled)
        self._cutoff_edit.setToolTip(context.cutoff_tooltip)

    def _update_metric_preview(self) -> None:
        """Show compact practical text for the selected metric and inputs."""
        preview = getattr(self, "_preview_label", None)
        if preview is None:
            return
        key = self._prop_combo.currentData()
        ref_sel = self._ref_edit.text().strip()
        target_kind = self._current_target_kind()
        cutoff_edit = getattr(self, "_cutoff_edit", None)
        cutoff_text = cutoff_edit.text() if cutoff_edit is not None else ""
        preview.setText(
            gui_rules.metric_preview_text(
                key,
                target_kind,
                ref_sel,
                cutoff_text,
                bool(getattr(self, "_ensemble_members", None)),
            )
        )

    def _set_statistics_text(self, text: str) -> None:
        """Update the statistics panel when it exists."""
        browser = getattr(self, "_stats_browser", None)
        if browser is not None:
            browser.setPlainText(text)

    def _update_statistics_for_single(
        self,
        key: str,
        target_name: str,
        values: np.ndarray,
        *,
        include_plddt_classes: bool = False,
        include_chain_stats: bool = False,
        include_domain_labels: bool = False,
        token_map=None,
    ) -> None:
        """Show statistics for one successfully painted target."""
        self._set_statistics_text(
            reports.format_statistics_report(
                key,
                target_name,
                [(target_name, values, token_map)],
                include_plddt_classes=include_plddt_classes,
                include_chain_stats=include_chain_stats,
                include_domain_labels=include_domain_labels,
            )
        )

    def _update_statistics_for_members(
        self,
        key: str,
        target_label: str,
        member_values: list[tuple[object, np.ndarray]],
        *,
        include_plddt_classes: bool = False,
        include_chain_stats: bool = False,
        include_domain_labels: bool = False,
    ) -> None:
        """Show statistics for successfully painted ensemble targets."""
        entries = [
            (member.obj_name, values, getattr(member, "token_map", None))
            for member, values in member_values
        ]
        self._set_statistics_text(
            reports.format_statistics_report(
                key,
                target_label,
                entries,
                include_plddt_classes=include_plddt_classes,
                include_chain_stats=include_chain_stats,
                include_domain_labels=include_domain_labels,
            )
        )

    def _show_ensemble(self) -> None:
        """Prepare an ensemble in the worker, then commit it through PyMOL."""
        if self._gui_job_is_busy():
            return
        if self._pred_files is None:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, "No prediction output loaded."
            )
            return
        if not self._pred_files.supports_ensemble:
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Ensemble mode requires at least two model files.",
            )
            return

        skip_alignment = self._ask_skip_ensemble_alignment()
        if skip_alignment is None:
            return

        pred_files = self._pred_files
        existing_data = self._existing_ensemble_data_by_rank()
        self._loading_data = True
        request_id = self._next_gui_job_request_id()
        self._data_load_request_id = request_id
        self._set_prediction_load_controls_enabled(False)
        first_model = pred_files.models[0]
        self._schedule_load_progress(
            request_id,
            f"Preparing {first_model.display_label} ensemble data…",
        )
        handle = self._job_runner.submit(
            request_id,
            lambda report: _prepare_ensemble_job(
                pred_files,
                skip_alignment,
                existing_data,
                report,
            ),
            self._on_data_load_progress,
            self._on_ensemble_prepared,
            self._on_ensemble_preparation_error,
        )
        if self._data_load_is_active(request_id):
            self._active_load_handle = handle

    def _existing_ensemble_data_by_rank(self) -> dict[int, object]:
        """Return current rank data suitable for reuse by ensemble preparation."""
        existing: dict[int, object] = {}
        current = self._pred_data
        if current is not None and getattr(current, "rank", None) is not None:
            existing[int(current.rank)] = current
        for member in self._ensemble_members or []:
            rank = int(member.rank)
            previous = existing.get(rank)
            if previous is None or self._prediction_data_array_count(
                member.data
            ) > self._prediction_data_array_count(previous):
                existing[rank] = member.data
        return existing

    @staticmethod
    def _prediction_data_array_count(data: object) -> int:
        fields = ("token_plddt", "pae", "pde", "contact_probs")
        return sum(getattr(data, field, None) is not None for field in fields)

    def _on_ensemble_prepared(
        self,
        request_id: int,
        prepared: ensemble.PreparedEnsemble,
    ) -> None:
        if not self._data_load_is_active(request_id):
            self._job_runner.dispose(prepared)
            return
        if prepared.pred_files is not self._pred_files:
            self._job_runner.dispose(prepared)
            self._finish_data_load(request_id)
            return

        self._active_load_handle = None
        previous_target = ""
        obj_combo = getattr(self, "_obj_combo", None)
        if obj_combo is not None and hasattr(obj_combo, "currentText"):
            previous_target = obj_combo.currentText()
        try:
            group_existed = viewer_name_exists(prepared.group_name)
            previous_group_members = get_group_members(prepared.group_name)
        except Exception as exc:
            logger.exception("Could not inspect the target PyMOL ensemble group")
            self._finish_data_load(request_id)
            QtWidgets.QMessageBox.critical(
                self,
                f"{APP_TITLE} - error",
                str(exc),
            )
            return
        transaction = EnsembleViewerTransaction(
            request_id=request_id,
            prepared=prepared,
            previous_target=previous_target,
            group_existed=group_existed,
            previous_group_members=previous_group_members,
            previous_members=self._ensemble_members,
            previous_group_name=self._ensemble_group_name,
            previous_aligned=self._ensemble_aligned,
            previous_rmsd=self._ensemble_rmsd,
            previous_plddt_mean=self._ensemble_plddt_mean,
            previous_plddt_std=self._ensemble_plddt_std,
        )
        self._active_ensemble_viewer_transaction = transaction
        QtCore.QTimer.singleShot(
            0,
            lambda: self._load_next_ensemble_object(transaction, 0),
        )

    def _ensemble_transaction_is_active(
        self,
        transaction: EnsembleViewerTransaction,
    ) -> bool:
        return bool(
            self._data_load_is_active(transaction.request_id)
            and self._active_ensemble_viewer_transaction is transaction
            and transaction.prepared.pred_files is self._pred_files
        )

    def _load_next_ensemble_object(
        self,
        transaction: EnsembleViewerTransaction,
        index: int,
    ) -> None:
        if not self._ensemble_transaction_is_active(transaction):
            return
        members = transaction.prepared.members
        if index >= len(members):
            QtCore.QTimer.singleShot(
                0,
                lambda: self._inspect_next_ensemble_object(transaction, 0),
            )
            return

        member = members[index]
        self._on_data_load_progress(
            transaction.request_id,
            f"Loading {member.model_label} into PyMOL… ({index + 1}/{len(members)})",
        )
        try:
            did_load = load_structure_object_if_missing(
                member.structure_path,
                member.obj_name,
            )
            if did_load:
                transaction.created_objects.append(member.obj_name)
        except Exception as exc:
            self._fail_ensemble_viewer_transaction(transaction, exc)
            return
        QtCore.QTimer.singleShot(
            0,
            lambda: self._load_next_ensemble_object(transaction, index + 1),
        )

    def _inspect_next_ensemble_object(
        self,
        transaction: EnsembleViewerTransaction,
        index: int,
    ) -> None:
        if not self._ensemble_transaction_is_active(transaction):
            return
        members = transaction.prepared.members
        if index >= len(members):
            QtCore.QTimer.singleShot(
                0,
                lambda: self._align_and_group_ensemble(transaction),
            )
            return

        member = members[index]
        self._on_data_load_progress(
            transaction.request_id,
            f"Inspecting {member.model_label} coordinates… "
            f"({index + 1}/{len(members)})",
        )
        try:
            if not viewer_name_exists(member.obj_name):
                raise ValueError(f"PyMOL object '{member.obj_name}' no longer exists.")
            transaction.inspections[member.rank] = inspect_object_tokens(
                member.obj_name,
                member.token_map,
            )
        except Exception as exc:
            self._fail_ensemble_viewer_transaction(transaction, exc)
            return
        QtCore.QTimer.singleShot(
            0,
            lambda: self._inspect_next_ensemble_object(transaction, index + 1),
        )

    def _align_and_group_ensemble(
        self,
        transaction: EnsembleViewerTransaction,
    ) -> None:
        if not self._ensemble_transaction_is_active(transaction):
            return
        prepared = transaction.prepared
        coords = {
            rank: inspection.representative_coords
            for rank, inspection in transaction.inspections.items()
        }
        try:
            if prepared.skip_alignment:
                rmsd = ensemble.compute_per_token_rmsd(
                    [coords[member.rank] for member in prepared.members]
                )
                transforms: tuple[ensemble.AlignmentTransform, ...] = ()
            else:
                reference = next(
                    member
                    for member in prepared.members
                    if member.rank == prepared.reference_rank
                )
                self._on_data_load_progress(
                    transaction.request_id,
                    f"Aligning ensemble to {reference.model_label}…",
                )
                plan = ensemble.calculate_alignment_plan(
                    prepared.members,
                    coords,
                    reference_rank=prepared.reference_rank,
                    core_indices=prepared.core_indices,
                )
                transforms = plan.transforms
                rmsd = plan.rmsd

                def apply_transforms() -> None:
                    for transform in transforms:
                        member = next(
                            item
                            for item in prepared.members
                            if item.rank == transform.rank
                        )
                        transform_object(
                            member.obj_name,
                            transform.rotation,
                            transform.translation,
                        )
                        transaction.applied_transforms.append(transform)

                run_with_updates_suspended(apply_transforms)

            self._on_data_load_progress(
                transaction.request_id,
                "Grouping ensemble objects…",
            )
            object_names = tuple(member.obj_name for member in prepared.members)
            previous_members = set(transaction.previous_group_members)
            transaction.group_additions = tuple(
                name for name in object_names if name not in previous_members
            )
            run_with_updates_suspended(
                lambda: add_objects_to_group(prepared.group_name, object_names)
            )
        except Exception as exc:
            self._fail_ensemble_viewer_transaction(transaction, exc)
            return

        self._commit_ensemble_transaction(transaction, rmsd)

    def _commit_ensemble_transaction(
        self,
        transaction: EnsembleViewerTransaction,
        rmsd: np.ndarray,
    ) -> None:
        if not self._ensemble_transaction_is_active(transaction):
            return
        prepared = transaction.prepared
        members = [
            ensemble.EnsembleMember(
                rank=member.rank,
                obj_name=member.obj_name,
                data=member.data,
                token_map=member.token_map,
                paint_mapping=transaction.inspections[member.rank].paint_mapping,
            )
            for member in prepared.members
        ]
        self._ensemble_members = members
        self._ensemble_group_name = prepared.group_name
        self._ensemble_aligned = not prepared.skip_alignment
        self._ensemble_rmsd = rmsd
        self._ensemble_plddt_mean = prepared.plddt_mean
        self._ensemble_plddt_std = prepared.plddt_std
        try:
            self._refresh_objects()
            self._select_object(prepared.group_name)
            self._update_property_availability()
            self._select_property("ensemble_rmsd")
        except Exception as exc:
            self._restore_previous_ensemble_state(transaction)
            self._fail_ensemble_viewer_transaction(transaction, exc)
            return
        self._active_ensemble_viewer_transaction = None

        mode_label = (
            "current coordinates"
            if prepared.skip_alignment
            else "automatic core alignment"
        )
        self._finish_data_load(transaction.request_id, save_session=True)
        QtWidgets.QMessageBox.information(
            self,
            APP_TITLE,
            f"Loaded {len(members)} ensemble models into group "
            f"'{prepared.group_name}'.\n"
            f"RMSD was computed using {mode_label}.\n\n"
            "Use Apply Coloring to color the selected target.",
        )

    def _restore_previous_ensemble_state(
        self,
        transaction: EnsembleViewerTransaction,
    ) -> None:
        self._ensemble_members = transaction.previous_members
        self._ensemble_group_name = transaction.previous_group_name
        self._ensemble_aligned = transaction.previous_aligned
        self._ensemble_rmsd = transaction.previous_rmsd
        self._ensemble_plddt_mean = transaction.previous_plddt_mean
        self._ensemble_plddt_std = transaction.previous_plddt_std

    def _on_ensemble_preparation_error(self, request_id: int, failure) -> None:
        if not self._data_load_is_active(request_id):
            return
        logger.error(
            "Background ensemble preparation failed:\n%s", failure.traceback_text
        )
        self._finish_data_load(request_id)
        QtWidgets.QMessageBox.critical(
            self,
            f"{APP_TITLE} - error",
            failure.message,
        )

    def _fail_ensemble_viewer_transaction(
        self,
        transaction: EnsembleViewerTransaction,
        exc: Exception,
    ) -> None:
        if not self._ensemble_transaction_is_active(transaction):
            return
        logger.exception("Could not load or align the ensemble in PyMOL")
        self._rollback_ensemble_viewer_transaction(refresh_gui=True)
        self._finish_data_load(transaction.request_id, save_session=True)
        QtWidgets.QMessageBox.critical(
            self,
            f"{APP_TITLE} - error",
            str(exc),
        )

    def _rollback_ensemble_viewer_transaction(
        self,
        *,
        refresh_gui: bool,
    ) -> None:
        transaction = getattr(self, "_active_ensemble_viewer_transaction", None)
        if transaction is None:
            return
        prepared = transaction.prepared

        try:
            if transaction.group_existed:
                remove_objects_from_group(
                    prepared.group_name,
                    transaction.group_additions,
                )
            else:
                delete_viewer_names((prepared.group_name,))
        except Exception:
            logger.exception("Could not restore the previous ensemble group")

        try:
            members_by_rank = {member.rank: member for member in prepared.members}

            def revert_transforms() -> None:
                for transform in reversed(transaction.applied_transforms):
                    member = members_by_rank[transform.rank]
                    if not viewer_name_exists(member.obj_name):
                        continue
                    rotation, translation = ensemble.invert_rigid_transform(
                        transform.rotation,
                        transform.translation,
                    )
                    transform_object(member.obj_name, rotation, translation)

            if transaction.applied_transforms:
                run_with_updates_suspended(revert_transforms)
        except Exception:
            logger.exception("Could not restore transformed ensemble objects")

        try:
            delete_viewer_names(reversed(transaction.created_objects))
            if transaction.created_objects:
                rebuild()
        except Exception:
            logger.exception("Could not remove partially loaded ensemble objects")
        finally:
            self._active_ensemble_viewer_transaction = None

        if refresh_gui:
            try:
                self._refresh_objects()
                if transaction.previous_target:
                    self._select_object(transaction.previous_target)
            except Exception:
                logger.exception("Could not refresh the viewer after ensemble rollback")

    def _ask_skip_ensemble_alignment(self) -> bool | None:
        """Return True for expert-mode no-align, False for auto-align, None on cancel."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"{APP_TITLE} Ensemble")
        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel(
            "Load all models as separate objects and group them.\n"
            "By default, models are aligned to model_0 using a high-confidence "
            "protein core."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        checkbox = QtWidgets.QCheckBox(
            f"Use current {VIEWER_NAME} coordinates; do not align"
        )
        checkbox.setToolTip(
            "Skip automatic core alignment and compute ensemble RMSD from the current "
            f"{VIEWER_NAME} object coordinates."
        )
        layout.addWidget(checkbox)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch()
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        ok_btn.setToolTip("Load the ensemble with the selected alignment option.")
        cancel_btn.setToolTip(
            "Close this dialog without loading or updating the ensemble."
        )
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        button_row.addWidget(ok_btn)
        button_row.addWidget(cancel_btn)
        layout.addLayout(button_row)

        return checkbox.isChecked() if dialog.exec() == 1 else None
