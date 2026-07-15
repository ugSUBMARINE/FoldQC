"""Prediction/model lifecycle and contextual GUI coordination."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ensemble, gui_rules, metrics, plot_data, reports
from .compat import ItemIsEnabled, QtCore, QtWidgets, WindowCloseButtonHint
from .loader_models import DataCapability, PredictionData, PredictionFiles
from .model_state import ModelState, ModelStateSnapshot
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

_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


@dataclass
class InitialLoadResult:
    """Provider files and initial lazy data prepared by a background job."""

    _pred_files: PredictionFiles | None
    model_state: ModelState
    display_path: Path

    @property
    def rank(self) -> int:
        return self.model_state.rank

    @property
    def pred_files(self) -> PredictionFiles:
        if self._pred_files is None:
            raise RuntimeError("InitialLoadResult ownership was already transferred.")
        return self._pred_files

    def take_prediction_files(self) -> PredictionFiles:
        files = self.pred_files
        self._pred_files = None
        return files

    def close(self) -> None:
        if self._pred_files is not None:
            self._pred_files.close()
            self._pred_files = None


@dataclass(frozen=True)
class ModelSwitchResult:
    """One ranked model prepared without touching Qt widgets or PyMOL."""

    pred_files: PredictionFiles
    model_state: ModelState

    @property
    def rank(self) -> int:
        return self.model_state.rank


@dataclass(frozen=True)
class DataLoadItem:
    """One current-model or ensemble-member lazy reload request."""

    rank: int
    model_label: str
    capabilities: frozenset[DataCapability]
    model_state: ModelState
    expected_version: int
    expected_ensemble: ensemble.EnsembleState | None = None
    phase_arrays: tuple[str, ...] = ()

    def load_kwargs(self) -> dict[str, bool]:
        return {
            "load_pae": "pae" in self.capabilities,
            "load_pde": "pde" in self.capabilities,
            "load_contact_probs": "contact_probs" in self.capabilities,
            "load_token_plddt": "plddt" in self.capabilities,
        }


@dataclass(frozen=True)
class DataLoadBatchResult:
    """Atomically committable lazy data returned by one worker task."""

    pred_files: PredictionFiles
    loaded: tuple[tuple[DataLoadItem, PredictionData], ...]


@dataclass(frozen=True)
class ModelStoreSnapshot:
    """Restorable model-store membership, contents, and active rank."""

    active_rank: int | None
    entries: tuple[tuple[int, ModelState, ModelStateSnapshot], ...]


@dataclass(frozen=True)
class InitialPredictionSnapshot:
    """GUI state restored if initial prediction activation fails."""

    pred_files: PredictionFiles | None
    model_store: ModelStoreSnapshot
    ensemble_state: ensemble.EnsembleState | None
    display_path: str
    model_items: tuple[tuple[str, object], ...]
    selected_model_rank: object | None
    viewer_context: tuple


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
    previous_ensemble: ensemble.EnsembleState | None = None
    previous_model_store: ModelStoreSnapshot | None = None
    previous_viewer_context: tuple | None = None


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
    from .loader import load_prediction_data, scan_prediction_candidate
    from .structure_index import StructureIndex

    display_path = _session_path_for_candidate(discovery, candidate)
    report_phase(f"Scanning {candidate.provider_label} output…")
    pred_files = scan_prediction_candidate(discovery, candidate)
    try:
        if not pred_files.models:
            raise ValueError("No ranked model files were found.")

        available_ranks = {model.rank for model in pred_files.models}
        rank = (
            preferred_rank
            if preferred_rank is not None and preferred_rank in available_ranks
            else pred_files.models[0].rank
        )
        model = pred_files.model(rank)
        report_phase(f"Indexing {model.display_label} structure…")
        structure_index = StructureIndex.from_path(
            pred_files.structure_path(model.rank)
        )
        report_phase(f"Loading {model.display_label} data…")
        pred_data = load_prediction_data(
            pred_files,
            rank,
            load_pae=False,
            load_pde=False,
            load_contact_probs=False,
            structure_index=structure_index,
        )
        return InitialLoadResult(
            pred_files,
            ModelState(
                rank=rank,
                data=pred_data,
                structure_index=structure_index,
            ),
            display_path,
        )
    except Exception:
        pred_files.close()
        raise


def _load_rank_data(pred_files, rank: int, report_phase) -> ModelSwitchResult:
    from .loader import load_prediction_data
    from .structure_index import StructureIndex

    model = pred_files.model(rank)
    report_phase(f"Indexing {model.display_label} structure…")
    structure_index = StructureIndex.from_path(pred_files.structure_path(model.rank))
    report_phase(f"Loading {model.display_label} data…")
    data = load_prediction_data(
        pred_files,
        rank,
        load_pae=False,
        load_pde=False,
        load_contact_probs=False,
        structure_index=structure_index,
    )
    return ModelSwitchResult(
        pred_files,
        ModelState(rank=rank, data=data, structure_index=structure_index),
    )


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
            structure_index=item.model_state.structure_index,
            **item.load_kwargs(),
        )
        loaded.append((item, data))
    return DataLoadBatchResult(pred_files, tuple(loaded))


def _prepare_ensemble_job(
    pred_files,
    skip_alignment: bool,
    existing_states_by_rank: dict[int, ModelState],
    report_phase,
):
    return ensemble.prepare_ensemble(
        pred_files,
        skip_alignment=skip_alignment,
        existing_states_by_rank=existing_states_by_rank,
        report_phase=report_phase,
    )


class GuiLoadingController:
    def _capture_model_store(self) -> ModelStoreSnapshot:
        return ModelStoreSnapshot(
            active_rank=self._active_model_rank,
            entries=tuple(
                (rank, state, state.snapshot())
                for rank, state in self._model_states.items()
            ),
        )

    def _restore_model_store(self, snapshot: ModelStoreSnapshot) -> None:
        restored = {}
        for rank, state, state_snapshot in snapshot.entries:
            state.restore(state_snapshot)
            restored[rank] = state
        self._model_states = restored
        self._active_model_rank = snapshot.active_rank

    def _commit_model_state(
        self,
        incoming: ModelState,
        *,
        reset_store: bool = False,
        activate: bool = True,
    ) -> ModelState:
        if reset_store:
            canonical = incoming
            self._model_states = {incoming.rank: canonical}
        else:
            canonical = self._model_states.get(incoming.rank)
            if canonical is None:
                canonical = incoming
                states = dict(self._model_states)
                states[incoming.rank] = canonical
                self._model_states = states
            elif canonical is not incoming:
                canonical.validate_structure_index(incoming.structure_index)
                canonical.merge_data(incoming.data)
        if activate:
            self._active_model_rank = incoming.rank
        return canonical

    def _capture_initial_prediction_context(self) -> InitialPredictionSnapshot:
        model_items = tuple(
            (self._model_combo.itemText(index), self._model_combo.itemData(index))
            for index in range(self._model_combo.count())
        )
        return InitialPredictionSnapshot(
            pred_files=self._pred_files,
            model_store=self._capture_model_store(),
            ensemble_state=self._ensemble,
            display_path=self._dir_edit.text(),
            model_items=model_items,
            selected_model_rank=(
                self._model_combo.currentData() if self._model_combo.count() else None
            ),
            viewer_context=self._capture_viewer_mapping_context(),
        )

    def _restore_initial_prediction_context(
        self, snapshot: InitialPredictionSnapshot
    ) -> None:
        self._pred_files = snapshot.pred_files
        self._restore_model_store(snapshot.model_store)
        self._ensemble = snapshot.ensemble_state
        self._restore_viewer_mapping_context(snapshot.viewer_context)
        self._dir_edit.setText(snapshot.display_path)
        self._model_combo.blockSignals(True)
        try:
            self._model_combo.clear()
            for label, rank in snapshot.model_items:
                self._model_combo.addItem(label, rank)
            if snapshot.selected_model_rank is not None:
                self._select_model_rank(snapshot.selected_model_rank)
        finally:
            self._model_combo.blockSignals(False)
        self._update_confidence_summary()
        self._update_property_availability()
        self._refresh_objects()

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
        structure_name = Path(result.model_state.data.structure_path).name
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
            snapshot = self._capture_initial_prediction_context()
            new_pred_files = result.take_prediction_files()
            self._pred_files = new_pred_files
            self._ensemble = None
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
            self._activate_model_state(
                result.model_state,
                reset_store=True,
                prepared_object=(obj_name, did_load),
            )
        except Exception as exc:
            logger.exception("Could not activate the initial prediction model")
            if "snapshot" in locals():
                try:
                    self._restore_initial_prediction_context(snapshot)
                except Exception:
                    logger.exception(
                        "Could not fully restore the previous prediction state"
                    )
            if "new_pred_files" in locals():
                self._job_runner.dispose(new_pred_files)
            if did_load:
                try:
                    delete_viewer_names([obj_name])
                except Exception:
                    logger.exception(
                        "Could not remove the newly loaded prediction object"
                    )
            self._finish_prediction_load(request_id)
            QtWidgets.QMessageBox.warning(self, APP_TITLE, str(exc))
            return

        if (
            snapshot.pred_files is not None
            and snapshot.pred_files is not self._pred_files
        ):
            self._job_runner.dispose(snapshot.pred_files)
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
        self._model_switch_previous_store = None
        self._model_switch_previous_viewer_context = None
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
        else:
            self._update_ensemble_button_state()

    def _ensemble_load_is_available(self) -> bool:
        pred_files = self._pred_files
        return bool(
            not self._gui_job_is_busy()
            and pred_files is not None
            and getattr(pred_files, "supports_ensemble", False)
            and self._ensemble is None
        )

    def _update_ensemble_button_state(self) -> None:
        """Enable ensemble creation only when it is currently meaningful."""
        button = getattr(self, "_ensemble_btn", None)
        if button is None or not hasattr(button, "setEnabled"):
            return

        pred_files = self._pred_files
        busy = self._gui_job_is_busy()
        supports_ensemble = bool(
            pred_files is not None and getattr(pred_files, "supports_ensemble", False)
        )
        ensemble_loaded = self._ensemble is not None
        button.setEnabled(self._ensemble_load_is_available())

        if not hasattr(button, "setToolTip"):
            return
        if busy:
            tooltip = "Ensemble loading is unavailable while another task is running."
        elif pred_files is None:
            tooltip = "Load a prediction with at least two models first."
        elif not supports_ensemble:
            tooltip = "Ensemble mode requires at least two model files."
        elif ensemble_loaded:
            tooltip = "The ensemble for this prediction is already loaded."
        else:
            tooltip = (
                "Load all ranked models as an ensemble and compute "
                "ensemble-level metrics."
            )
        button.setToolTip(tooltip)

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
        committed_rank = self._active_model_rank
        if rank == committed_rank or self._gui_job_is_busy():
            return

        pred_files = self._pred_files
        self._model_switch_previous_store = self._capture_model_store()
        self._model_switch_previous_viewer_context = (
            self._capture_viewer_mapping_context()
        )
        self._loading_data = True
        request_id = self._next_gui_job_request_id()
        self._data_load_request_id = request_id
        self._set_prediction_load_controls_enabled(False)
        model = pred_files.model(rank)
        cached_state = self._model_states.get(rank)
        if cached_state is not None:
            self._schedule_load_progress(
                request_id,
                f"Showing {model.display_label}…",
            )
            QtCore.QTimer.singleShot(
                0,
                lambda: self._commit_model_switch(
                    request_id,
                    ModelSwitchResult(pred_files, cached_state),
                ),
            )
            return
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
            self._activate_model_state(
                result.model_state,
                prepared_object=(model.object_name, did_load),
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
        rank = self._active_model_rank
        if rank is None:
            return
        self._model_combo.blockSignals(True)
        try:
            self._select_model_rank(rank)
        finally:
            self._model_combo.blockSignals(False)

    def _rollback_model_switch(self) -> None:
        snapshot = getattr(self, "_model_switch_previous_store", None)
        restored = snapshot is not None
        if snapshot is not None:
            self._restore_model_store(snapshot)
        viewer_context = getattr(self, "_model_switch_previous_viewer_context", None)
        if viewer_context is not None:
            self._restore_viewer_mapping_context(viewer_context)
        if restored:
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
        self._model_switch_previous_store = None
        self._model_switch_previous_viewer_context = None
        self._loading_data = False
        self._hide_load_progress()
        self._set_prediction_load_controls_enabled(True)
        self._refresh_contextual_ui()
        if save_session:
            self._save_session_settings()

    def _defer_action_for_data(
        self,
        target,
        requested_capabilities: frozenset[DataCapability],
        continuation,
        *,
        error_title: str,
        allow_partial: bool = False,
    ) -> bool:
        """Submit missing lazy arrays and resume *continuation* after commit."""
        if self._pred_files is None:
            return False
        try:
            items = self._data_load_items_for_target(
                target, requested_capabilities, allow_partial=allow_partial
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, error_title, str(exc))
            return True
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
        requested_capabilities: frozenset[DataCapability],
        *,
        allow_partial: bool = False,
    ) -> list[DataLoadItem]:
        if target is None:
            return []
        expected_ensemble = (
            self._ensemble if target.kind.startswith("ensemble") else None
        )
        if target.kind.startswith("ensemble") and expected_ensemble is None:
            raise ValueError("The ensemble target is no longer active.")
        slots = []
        for state in target.model_states:
            if self._model_states.get(state.rank) is not state:
                raise ValueError(
                    f"Model {state.rank} is no longer present in the canonical "
                    "model store."
                )
            slots.append(state)

        capability_attrs = {
            "pae": "pae",
            "pde": "pde",
            "contact_probs": "contact_probs",
            "plddt": "token_plddt",
        }
        unavailable: list[str] = []
        model_getter = getattr(self._pred_files, "model", None)
        for model_state in slots:
            requested_missing = [
                capability
                for capability, attr in capability_attrs.items()
                if capability in requested_capabilities
                and getattr(model_state.data, attr, None) is None
            ]
            if not requested_missing:
                continue
            model = model_getter(model_state.rank) if callable(model_getter) else None
            missing = [
                capability
                for capability in requested_missing
                if model is None or not model.supports(capability)
            ]
            if missing and not allow_partial:
                label = (
                    f"model_{model_state.rank}"
                    if model is None
                    else model.display_label
                )
                unavailable.append(f"{label} ({', '.join(missing)})")
        if unavailable:
            raise ValueError(
                "The requested data are unavailable for: " + "; ".join(unavailable)
            )

        items = []
        seen = set()
        for model_state in slots:
            key = id(model_state)
            if key in seen:
                continue
            seen.add(key)
            item = self._data_load_item(
                model_state,
                requested_capabilities,
                expected_ensemble=expected_ensemble,
                allow_partial=allow_partial,
            )
            if item is not None:
                items.append(item)
        return items

    def _data_load_item(
        self,
        model_state: ModelState,
        requested_capabilities: frozenset[DataCapability],
        *,
        expected_ensemble: ensemble.EnsembleState | None,
        allow_partial: bool = False,
    ) -> DataLoadItem | None:
        data = model_state.data
        capability_attrs = {
            "pae": ("pae", "PAE"),
            "pde": ("pde", "PDE"),
            "contact_probs": ("contact_probs", "interaction probabilities"),
            "plddt": ("token_plddt", "pLDDT"),
        }
        requested = set(requested_capabilities)
        if allow_partial:
            model_getter = getattr(self._pred_files, "model", None)
            model = model_getter(model_state.rank) if callable(model_getter) else None
            for capability, (attr, _label) in capability_attrs.items():
                if getattr(data, attr, None) is None and (
                    model is None or not model.supports(capability)
                ):
                    requested.discard(capability)
        missing = [
            capability
            for capability, (attr, _label) in capability_attrs.items()
            if capability in requested and getattr(data, attr, None) is None
        ]
        if not missing:
            return None

        phase_arrays = tuple(
            dict.fromkeys(capability_attrs[name][1] for name in missing)
        )
        rank = model_state.rank
        model = self._pred_files.model(rank)
        return DataLoadItem(
            rank=rank,
            model_label=model.display_label,
            capabilities=frozenset(missing),
            model_state=model_state,
            expected_version=model_state.version,
            expected_ensemble=expected_ensemble,
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

        snapshots = []
        try:
            for item, data in result.loaded:
                self._validate_lazy_loaded_item(item, data)
                item.model_state.validate_merge(data)
            snapshots = [
                (item.model_state, item.model_state.snapshot())
                for item, _data in result.loaded
            ]
            for item, data in result.loaded:
                item.model_state.merge_data(data)
        except Exception as exc:
            for state, snapshot in reversed(snapshots):
                state.restore(snapshot)
            title = getattr(
                self,
                "_active_data_error_title",
                f"{APP_TITLE} - error",
            )
            self._finish_data_load(request_id)
            QtWidgets.QMessageBox.critical(self, title, str(exc))
            return

        self._active_load_handle = None
        continuation = self._active_data_continuation
        self._on_data_load_progress(request_id, "Preparing requested action…")
        QtCore.QTimer.singleShot(
            0,
            lambda: self._resume_lazy_action(request_id, continuation),
        )

    def _lazy_result_is_current(self, result: DataLoadBatchResult) -> bool:
        for item, _data in result.loaded:
            if self._model_states.get(item.rank) is not item.model_state:
                return False
            if item.model_state.version != item.expected_version:
                return False
            if item.expected_ensemble is not None:
                if self._ensemble is not item.expected_ensemble:
                    return False
                if item.rank not in item.expected_ensemble.ranks:
                    return False
        return True

    def _validate_lazy_loaded_item(self, item: DataLoadItem, data) -> None:
        fields = (
            ("pae", "pae", "PAE"),
            ("pde", "pde", "PDE"),
            ("contact_probs", "contact_probs", "interaction probabilities"),
            ("plddt", "token_plddt", "pLDDT"),
        )
        missing = [
            label
            for capability, attr, label in fields
            if capability in item.capabilities and getattr(data, attr, None) is None
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

    def _activate_model_state(
        self,
        model_state: ModelState,
        *,
        reset_store: bool = False,
        prepared_object: tuple[str, bool] | None = None,
    ) -> None:
        """Commit loaded model data and perform main-thread viewer/UI updates."""
        rank = model_state.rank
        self._commit_model_state(model_state, reset_store=reset_store)
        self._clear_viewer_mapping_cache()
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
            ensemble_state = self._ensemble
            additional = [] if ensemble_state is None else [ensemble_state.group_name]
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
        ensemble_state = self._ensemble
        group_name = None if ensemble_state is None else ensemble_state.group_name
        members = sorted(
            () if ensemble_state is None else ensemble_state.members,
            key=lambda member: member.rank,
        )
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
        ensemble_state = self._ensemble
        if ensemble_state is None or name != ensemble_state.group_name:
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
        state = self._active_model_state
        self._conf_browser.setPlainText(
            reports.format_confidence_summary(None if state is None else state.data)
        )

    def _update_property_availability(self) -> None:
        """Grey out combo items whose required data is not available."""
        state = self._active_model_state
        if state is None or self._pred_files is None:
            return
        has_pae = self._target_all_supports_family("pae")
        has_pde = self._target_all_supports_family("pde")
        has_contact_probs = self._target_all_supports_family("contact_probs")
        has_plddt = self._target_all_supports_family("plddt")
        target_states = self._current_target_model_states()
        has_confidence = bool(target_states) and all(
            target_state.data.confidence is not None for target_state in target_states
        )
        has_chain_iptm = self._has_chain_iptm_metric_data()
        has_ensemble = self._ensemble is not None

        model = self._prop_combo.model()
        family_available = {
            "pae": has_pae,
            "pde": has_pde,
            "plddt": has_plddt,
            "contact_probs": has_contact_probs,
            "confidence": has_confidence,
        }
        for spec in metrics.METRICS:
            combo_row = self._property_combo_row(spec.key)
            if combo_row is None:
                continue
            available = True
            if any(not family_available[item] for item in spec.requirements):
                available = False
            if spec.key == "chain_iptm" and not has_chain_iptm:
                available = False
            if spec.ensemble_level and not has_ensemble:
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
        states = self._current_target_model_states()
        return bool(states) and all(
            state.data.confidence is not None and state.data.confidence.has_chain_iptm
            for state in states
        )

    def _select_first_available_property(self) -> None:
        """Move the property combo away from a disabled item after loading."""
        model = self._prop_combo.model()
        current = self._prop_combo.currentIndex()
        if current >= 0:
            item = model.item(current)
            if item is not None and item.flags() & ItemIsEnabled:
                return
        for spec in metrics.METRICS:
            row = self._property_combo_row(spec.key)
            if row is None:
                continue
            item = model.item(row)
            if item is not None and item.flags() & ItemIsEnabled:
                self._prop_combo.setCurrentIndex(row)
                return

    def _clear_viewer_mapping_cache(self) -> None:
        """Drop object-specific mapping state after changing prediction context."""
        self._paint_mappings = {}
        self._accepted_token_overlap_warnings = set()

    def _capture_viewer_mapping_context(self) -> tuple:
        """Snapshot object mappings for transactional model switching."""
        return (
            dict(self._paint_mappings),
            set(self._accepted_token_overlap_warnings),
        )

    def _restore_viewer_mapping_context(self, context: tuple) -> None:
        """Restore object mappings after a failed model switch."""
        (
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
        ensemble_state = getattr(self, "_ensemble", None)
        if ensemble_state is not None and obj_name == ensemble_state.group_name:
            return "ensemble_group"
        if self._selected_ensemble_member(obj_name) is not None:
            return "ensemble_member"
        return "single"

    def _current_target_model_states(self) -> tuple[ModelState, ...]:
        """Return canonical model states addressed by the viewer target."""
        ensemble_state = getattr(self, "_ensemble", None)
        kind = self._current_target_kind()
        if kind == "ensemble_group" and ensemble_state is not None:
            return tuple(
                state
                for member in sorted(ensemble_state.members, key=lambda item: item.rank)
                if (state := self._model_states.get(member.rank)) is not None
            )
        if kind == "ensemble_member":
            member = self._selected_ensemble_member(self._get_obj_name())
            state = None if member is None else self._model_states.get(member.rank)
            return () if state is None else (state,)
        state = self._active_model_state
        return () if state is None else (state,)

    def _state_supports_family(self, state: ModelState, family: str) -> bool:
        data_attr = "token_plddt" if family == "plddt" else family
        if getattr(state.data, data_attr, None) is not None:
            return True
        if self._pred_files is None:
            return False
        model_getter = getattr(self._pred_files, "model", None)
        if not callable(model_getter):
            return False
        return model_getter(state.rank).supports(family)

    def _target_all_supports_family(self, family: str) -> bool:
        states = self._current_target_model_states()
        return bool(states) and all(
            self._state_supports_family(state, family) for state in states
        )

    def _target_any_supports_family(self, family: str) -> bool:
        return any(
            self._state_supports_family(state, family)
            for state in self._current_target_model_states()
        )

    def _has_fingerprint_data(self) -> bool:
        """Return whether fingerprint plotting has any source family available."""
        return any(
            self._target_any_supports_family(family)
            for family in ("pae", "pde", "contact_probs", "plddt")
        )

    def _has_matrix_data_family(self, family: str) -> bool:
        """Return whether a matrix family is available from files or loaded data."""
        if family == "pae":
            return self._target_all_supports_family("pae")
        if family == "pde":
            return self._target_all_supports_family("pde")
        return False

    def _current_target_has_multiple_chains(self) -> bool:
        """Return whether the current target token map has multiple chains."""
        try:
            obj_name = self._get_obj_name()
        except Exception:
            obj_name = None
        if not obj_name:
            return False

        ensemble_state = getattr(self, "_ensemble", None)
        if ensemble_state is not None and obj_name == ensemble_state.group_name:
            members = ensemble_state.members
            if not members:
                return False
            state = self._model_states.get(members[0].rank)
            return bool(
                state is not None
                and plot_data.has_multiple_token_chains(state.token_map)
            )

        member = self._selected_ensemble_member(obj_name)
        if member is not None:
            state = self._model_states.get(member.rank)
            return bool(
                state is not None
                and plot_data.has_multiple_token_chains(state.token_map)
            )

        try:
            state = self._require_active_model_state()
        except Exception:
            return False
        return plot_data.has_multiple_token_chains(state.token_map)

    def _update_plot_actions(self) -> None:
        """Refresh plot menu action availability from current GUI state."""
        actions = getattr(self, "_plot_actions", None)
        if not actions:
            return
        metric_key = self._prop_combo.currentData()
        target_kind = self._current_target_kind()
        has_reference = bool(self._ref_edit.text().strip())
        has_ensemble = self._ensemble is not None
        has_fingerprint_data = self._has_fingerprint_data()
        has_pae_data = self._has_matrix_data_family("pae")
        has_pde_data = self._has_matrix_data_family("pde")
        has_multiple_chains = self._current_target_has_multiple_chains()
        for spec in metrics.PLOTS:
            action = actions.get(spec.key)
            if action is None:
                continue
            state = gui_rules.plot_action_state(
                spec.key,
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
            tip = state.reason or f"Show {spec.label.lower()}."
            if hasattr(action, "setToolTip"):
                action.setToolTip(tip)
            if hasattr(action, "setStatusTip"):
                action.setStatusTip(tip)

    def _refresh_contextual_ui(self) -> None:
        """Refresh plot actions, contextual fields, and preview text together."""
        self._update_ensemble_button_state()
        self._update_property_availability()
        self._update_plot_actions()
        self._update_context_controls()
        self._update_metric_preview()

    def _update_context_controls(self) -> None:
        """Apply contextual Reference and cutoff control states."""
        key = self._prop_combo.currentData()
        context = gui_rules.field_context(
            key,
            self._current_target_kind(),
            self._ensemble is not None,
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
                self._ensemble is not None,
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
        if not self._ensemble_load_is_available():
            return

        skip_alignment = self._ask_skip_ensemble_alignment()
        if skip_alignment is None:
            return

        pred_files = self._pred_files
        existing_states = dict(self._model_states)
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
                existing_states,
                report,
            ),
            self._on_data_load_progress,
            self._on_ensemble_prepared,
            self._on_ensemble_preparation_error,
        )
        if self._data_load_is_active(request_id):
            self._active_load_handle = handle

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
            previous_ensemble=self._ensemble,
            previous_model_store=self._capture_model_store(),
            previous_viewer_context=self._capture_viewer_mapping_context(),
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
        try:
            canonical_states = {
                member.rank: self._commit_model_state(
                    member.model_state,
                    activate=False,
                )
                for member in prepared.members
            }
            members = tuple(
                ensemble.EnsembleMember(
                    rank=member.rank,
                    obj_name=member.obj_name,
                )
                for member in prepared.members
            )
            for member in members:
                state = canonical_states[member.rank]
                key = self._paint_mapping_cache_key(state.data, member.obj_name)
                self._paint_mappings[key] = transaction.inspections[
                    member.rank
                ].paint_mapping
            self._ensemble = ensemble.EnsembleState(
                group_name=prepared.group_name,
                members=members,
                aligned=not prepared.skip_alignment,
                rmsd=rmsd,
                plddt_mean=prepared.plddt_mean,
                plddt_std=prepared.plddt_std,
            )
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
        if transaction.previous_model_store is not None:
            self._restore_model_store(transaction.previous_model_store)
        self._ensemble = transaction.previous_ensemble
        if transaction.previous_viewer_context is not None:
            self._restore_viewer_mapping_context(transaction.previous_viewer_context)

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
