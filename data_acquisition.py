"""DataAcquisition application workflow."""

from __future__ import annotations

from .gui_services import LifecycleUiUpdate
from .lifecycle_support import (
    APP_TITLE,
    DataCapability,
    DataLoadBatchResult,
    DataLoadRequirement,
    DeferredAnalysisAction,
    ModelState,
    Notice,
    PredictionFiles,
    _load_data_batch,
    ensemble,
    logger,
)


class DataAcquisitionWorkflow:
    def _finish_data_load(
        self,
        request_id: int,
        *,
        save_session: bool = False,
    ) -> LifecycleUiUpdate | None:
        if request_id != self._data_load_request_id:
            return None
        self._active_load_handle = None
        self._active_deferred_analysis = None
        self._model_switch_previous_store = None
        self._model_switch_previous_viewer_context = None
        self._loading_data = False
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

    def _defer_action_for_data(
        self,
        target,
        requested_capabilities: frozenset[DataCapability],
        *,
        error_title: str,
        allow_partial: bool = False,
        deferred_action: DeferredAnalysisAction,
    ) -> bool:
        """Submit missing lazy arrays and resume *continuation* after commit."""
        if self._pred_files is None:
            return False
        try:
            items = None
            if not allow_partial and isinstance(self._pred_files, PredictionFiles):
                resolved, plan = self.services.analysis.resolve_and_plan(
                    deferred_action.request
                )
                if resolved.required_capabilities == requested_capabilities:
                    items = list(plan.requirements)
            if items is None:
                items = self._data_load_items_for_target(
                    target, requested_capabilities, allow_partial=allow_partial
                )
        except ValueError as exc:
            self._presenter.present_notice(
                Notice("data_preflight_failed", str(exc), title=error_title)
            )
            return True
        if not items:
            return False
        if self._gui_job_is_busy():
            return True

        pred_files = self._pred_files
        self._loading_data = True
        request_id = self._next_gui_job_request_id()
        self._data_load_request_id = request_id
        self._active_deferred_analysis = deferred_action
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
    ) -> list[DataLoadRequirement]:
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
    ) -> DataLoadRequirement | None:
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
        return DataLoadRequirement(
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
            self._presenter.present_notice(
                Notice(
                    "lazy_merge_failed",
                    str(exc),
                    severity="error",
                    title=title,
                )
            )
            return

        self._active_load_handle = None
        continuation = self._active_deferred_analysis
        self._on_data_load_progress(request_id, "Preparing requested action…")
        self.application.scheduler.call_soon(
            lambda: self._resume_lazy_action(request_id, continuation)
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

    def _validate_lazy_loaded_item(self, item: DataLoadRequirement, data) -> None:
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
                self.services.analysis.resume(continuation)
        except Exception as exc:
            logger.exception("Could not resume the requested action")
            title = getattr(
                self,
                "_active_data_error_title",
                f"{APP_TITLE} - error",
            )
            self._presenter.present_notice(
                Notice(
                    "analysis_resume_failed",
                    str(exc),
                    severity="error",
                    title=title,
                )
            )
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
        self._presenter.present_notice(
            Notice(
                "lazy_load_failed",
                failure.message,
                severity="error",
                title=title,
            )
        )
