"""PredictionLifecycle application workflow."""

from __future__ import annotations

from .gui_services import LifecycleUiUpdate
from .lifecycle_support import (
    APP_TITLE,
    InitialLoadResult,
    InitialPredictionSnapshot,
    ModelState,
    ModelStoreSnapshot,
    ModelSwitchResult,
    Notice,
    Path,
    ProgressRequest,
    SelectionItem,
    SelectionRequest,
    _discover_prediction,
    _discovery_phase,
    _load_rank_data,
    _scan_and_load_initial_prediction,
    _session_path_for_candidate,
    delete_viewer_names,
    ensure_structure_object,
    logger,
)


class PredictionLifecycleWorkflow:
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
        self._presenter.update_progress(
            self._load_progress_operation(request_id), label
        )

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
        self.application.scheduler.call_soon(
            lambda: self._commit_initial_prediction(request_id, result)
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
            did_load = self._viewer_operation(
                "ensure_structure_object",
                ensure_structure_object,
                structure_path,
                obj_name,
                zoom=True,
            )
        except Exception as exc:
            logger.exception("Could not load the initial prediction model into PyMOL")
            self._job_runner.dispose(result)
            self._finish_prediction_load(request_id)
            self._presenter.present_notice(
                Notice(
                    "initial_viewer_load_failed",
                    f"Could not load or show {structure_path.name}:\n{exc}",
                    title=APP_TITLE,
                )
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
                    self._viewer_operation(
                        "delete_names", delete_viewer_names, [obj_name]
                    )
                except Exception:
                    logger.exception(
                        "Could not remove the newly loaded prediction object"
                    )
            self._finish_prediction_load(request_id)
            self._presenter.present_notice(
                Notice("initial_activation_failed", str(exc), title=APP_TITLE)
            )
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
        self._presenter.present_notice(
            Notice("prediction_load_failed", failure.message, title=APP_TITLE)
        )

    def _finish_prediction_load(
        self,
        request_id: int,
        *,
        save_session: bool = False,
    ) -> LifecycleUiUpdate | None:
        if request_id != self._prediction_load_request_id:
            return None
        self._active_load_handle = None
        self._loading_prediction = False
        self._hide_load_progress()
        self._set_prediction_load_controls_enabled(True)
        self._refresh_contextual_ui()
        if save_session:
            self._save_session_settings()
        return LifecycleUiUpdate(
            selected_rank=self._active_model_rank,
            refresh_context=True,
            save_session=save_session,
        )

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
        self._active_deferred_analysis = None
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

    @staticmethod
    def _load_progress_operation(request_id: int) -> str:
        return f"foldqc-load-{request_id}"

    def _schedule_load_progress(self, request_id: int, label: str) -> None:
        self._presenter.start_progress(
            ProgressRequest(
                operation_id=self._load_progress_operation(request_id),
                title=f"{APP_TITLE} – Loading",
                label=label,
                delay_ms=300,
            )
        )

    def _pause_load_progress(self) -> None:
        request_id = max(
            getattr(self, "_prediction_load_request_id", 0),
            getattr(self, "_data_load_request_id", 0),
        )
        self._presenter.finish_progress(self._load_progress_operation(request_id))

    def _hide_load_progress(self) -> None:
        self._pause_load_progress()

    def _choose_prediction_candidate(self, candidates):
        """Let the user pick one prediction directory from multiple candidates."""
        if not candidates:
            return None
        selected = self._presenter.select_item(
            SelectionRequest(
                code="prediction_candidate",
                title="Select prediction",
                message="Prediction directory:",
                items=tuple(
                    SelectionItem(
                        key=str(index),
                        label=candidate.relative_path,
                        description=candidate.provider_label,
                    )
                    for index, candidate in enumerate(candidates)
                ),
                default_key="0",
            )
        )
        return None if selected is None else candidates[int(selected)]

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
            did_load = self._viewer_operation(
                "ensure_structure_object",
                ensure_structure_object,
                path,
                obj_name,
                zoom=True,
            )
        except Exception as exc:
            self._presenter.present_notice(
                Notice(
                    "model_viewer_load_failed",
                    f"Could not load or show {path.name}:\n{exc}",
                    title=APP_TITLE,
                )
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
            self.application.scheduler.call_soon(
                lambda: self._commit_model_switch(
                    request_id,
                    ModelSwitchResult(pred_files, cached_state),
                )
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
        self._presenter.update_progress(
            self._load_progress_operation(request_id), label
        )

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
        self.application.scheduler.call_soon(
            lambda: self._commit_model_switch(request_id, result)
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
            did_load = self._viewer_operation(
                "ensure_structure_object",
                ensure_structure_object,
                structure_path,
                model.object_name,
                zoom=True,
            )
        except Exception as exc:
            logger.exception("Could not load the selected prediction model into PyMOL")
            self._rollback_model_switch()
            self._finish_data_load(request_id, save_session=True)
            self._presenter.present_notice(
                Notice(
                    "model_switch_viewer_failed",
                    f"Could not load or show {structure_path.name}:\n{exc}",
                    title=APP_TITLE,
                )
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
            self._presenter.present_notice(
                Notice("model_switch_failed", str(exc), title=APP_TITLE)
            )
            return

        self._finish_data_load(request_id, save_session=True)

    def _on_model_switch_error(self, request_id: int, failure) -> None:
        if not self._data_load_is_active(request_id):
            return
        logger.error("Background model switch failed:\n%s", failure.traceback_text)
        self._rollback_model_switch()
        self._finish_data_load(request_id, save_session=True)
        self._presenter.present_notice(
            Notice("model_switch_failed", failure.message, title=APP_TITLE)
        )

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
