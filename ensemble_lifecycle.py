"""Transactional, scheduler-driven ensemble activation."""

from __future__ import annotations

import logging

import numpy as np

from . import ensemble
from .context_service import ContextService
from .gui_services import (
    ContextSelection,
    GuiScheduler,
    JobRunner,
    OperationCoordinatorPort,
    OperationLease,
    ViewerPort,
)
from .gui_state import PluginState
from .lifecycle_support import (
    APP_TITLE,
    VIEWER_NAME,
    EnsembleActivationTransaction,
    _prepare_ensemble_job,
)
from .prediction_lifecycle import PredictionLifecycleService
from .presentation import ChoiceOption, ChoiceRequest, Notice, PresentationPort

logger = logging.getLogger(__name__)

ALIGNMENT_CORE_SELECTION_NAME = "foldqc_alignment_core"


class EnsembleLifecycleService:
    def __init__(
        self,
        state: PluginState,
        viewer: ViewerPort,
        presenter: PresentationPort,
        scheduler: GuiScheduler,
        job_runner: JobRunner,
        operations: OperationCoordinatorPort,
        prediction: PredictionLifecycleService,
        context: ContextService,
    ) -> None:
        self._state = state
        self._viewer = viewer
        self._presenter = presenter
        self._scheduler = scheduler
        self._job_runner = job_runner
        self._operations = operations
        self._prediction = prediction
        self._context = context
        self._transaction: EnsembleActivationTransaction | None = None

    def activate(self, previous_target: str = "") -> None:
        files = self._state.pred_files
        if files is None or len(files.models) < 2:
            return
        skip_alignment = self._ask_skip_alignment()
        if skip_alignment is None:
            return
        first = files.models[0]
        lease = self._operations.begin(
            "ensemble",
            title=f"{APP_TITLE} – Loading",
            label=f"Preparing {first.display_label} ensemble data…",
            cancellable=True,
            on_cancel=lambda: self.rollback(refresh_context=True),
        )
        if lease is None:
            return
        existing = dict(self._state.model_states)
        handle = self._job_runner.submit(
            lease.request_id,
            lambda report: _prepare_ensemble_job(
                files, skip_alignment, existing, report
            ),
            self._on_progress,
            lambda request_id, value: self._on_prepared(
                request_id, value, previous_target
            ),
            self._on_error,
        )
        self._operations.attach(lease, handle)

    def _on_progress(self, request_id: int, label: str) -> None:
        lease = self._operations.active
        if lease is not None and lease.request_id == request_id:
            self._operations.update(lease, label)

    def _on_prepared(
        self, request_id: int, value: object, previous_target: str
    ) -> None:
        lease = self._operations.active
        if lease is None or lease.request_id != request_id:
            self._job_runner.dispose(value)
            return
        if not isinstance(value, ensemble.PreparedEnsemble):
            self._fail(lease, "The ensemble worker returned an unexpected result.")
            return
        if value.pred_files is not self._state.pred_files:
            self._job_runner.dispose(value)
            self._operations.finish(lease)
            return
        try:
            transaction = EnsembleActivationTransaction(
                request_id,
                value,
                previous_target=previous_target,
                group_existed=self._viewer.name_exists(value.group_name),
                previous_group_members=self._viewer.group_members(value.group_name),
                previous_ensemble=self._state.ensemble,
                previous_model_store=self._prediction.capture_model_store(),
                previous_viewer_context=self._viewer.capture_paint_mappings(),
            )
        except Exception as exc:
            self._fail(lease, str(exc))
            return
        self._transaction = transaction
        self._scheduler.call_soon(lambda: self._load_member(transaction, 0))

    def _is_current(self, transaction: EnsembleActivationTransaction) -> bool:
        lease = self._operations.active
        return bool(
            lease is not None
            and lease.request_id == transaction.request_id
            and self._transaction is transaction
            and transaction.prepared.pred_files is self._state.pred_files
        )

    def _load_member(
        self, transaction: EnsembleActivationTransaction, index: int
    ) -> None:
        if not self._is_current(transaction):
            return
        members = transaction.prepared.members
        if index >= len(members):
            self._scheduler.call_soon(lambda: self._inspect_member(transaction, 0))
            return
        member = members[index]
        self._on_progress(
            transaction.request_id,
            f"Loading {member.model_label}… ({index + 1}/{len(members)})",
        )
        try:
            if self._viewer.load_structure_object_if_missing(
                member.structure_path, member.obj_name
            ):
                transaction.created_objects.append(member.obj_name)
        except Exception as exc:
            self._fail_transaction(transaction, exc)
            return
        self._scheduler.call_soon(lambda: self._load_member(transaction, index + 1))

    def _inspect_member(
        self, transaction: EnsembleActivationTransaction, index: int
    ) -> None:
        if not self._is_current(transaction):
            return
        members = transaction.prepared.members
        if index >= len(members):
            self._scheduler.call_soon(lambda: self._align_and_group(transaction))
            return
        member = members[index]
        self._on_progress(
            transaction.request_id,
            f"Inspecting {member.model_label}… ({index + 1}/{len(members)})",
        )
        try:
            if not self._viewer.name_exists(member.obj_name):
                raise ValueError(f"Viewer object '{member.obj_name}' no longer exists.")
            transaction.inspections[member.rank] = self._viewer.inspect_tokens(
                member.obj_name, member.token_map
            )
        except Exception as exc:
            self._fail_transaction(transaction, exc)
            return
        self._scheduler.call_soon(lambda: self._inspect_member(transaction, index + 1))

    def _align_and_group(self, transaction: EnsembleActivationTransaction) -> None:
        if not self._is_current(transaction):
            return
        prepared = transaction.prepared
        coordinates = {
            rank: inspection.representative_coords
            for rank, inspection in transaction.inspections.items()
        }
        try:
            if prepared.skip_alignment:
                rmsd = ensemble.compute_per_token_rmsd(
                    [coordinates[member.rank] for member in prepared.members]
                )
            else:
                plan = ensemble.calculate_alignment_plan(
                    prepared.members,
                    coordinates,
                    reference_rank=prepared.reference_rank,
                    core_indices=prepared.core_indices,
                )

                def apply_transforms() -> None:
                    members = {member.rank: member for member in prepared.members}
                    for transform in plan.transforms:
                        member = members[transform.rank]
                        self._viewer.transform(
                            member.obj_name,
                            transform.rotation,
                            transform.translation,
                        )
                        transaction.applied_transforms.append(transform)

                self._viewer.run_suspended(apply_transforms)
                rmsd = plan.rmsd
            object_names = tuple(member.obj_name for member in prepared.members)
            previous = set(transaction.previous_group_members)
            transaction.group_additions = tuple(
                name for name in object_names if name not in previous
            )
            self._viewer.run_suspended(
                lambda: self._viewer.add_to_group(prepared.group_name, object_names)
            )
        except Exception as exc:
            self._fail_transaction(transaction, exc)
            return
        self._commit(transaction, rmsd)

    def _commit(
        self, transaction: EnsembleActivationTransaction, rmsd: np.ndarray
    ) -> None:
        prepared = transaction.prepared
        try:
            canonical = {
                member.rank: self._prediction.commit_model_state(
                    member.model_state, activate=False
                )
                for member in prepared.members
            }
            mappings = self._viewer.capture_paint_mappings()
            members = tuple(
                ensemble.EnsembleMember(member.rank, member.obj_name)
                for member in prepared.members
            )
            for member in prepared.members:
                model_state = canonical[member.rank]
                key = (
                    str(model_state.data.structure_path),
                    member.obj_name,
                )
                mappings[key] = transaction.inspections[member.rank].paint_mapping
            self._viewer.restore_paint_mappings(mappings)
            self._state.ensemble = ensemble.EnsembleState(
                prepared.group_name,
                members,
                not prepared.skip_alignment,
                rmsd,
                prepared.plddt_mean,
                prepared.plddt_std,
            )
            self._context.set_selection(
                ContextSelection(
                    prepared.group_name,
                    "ensemble_rmsd",
                    self._context.selection.reference_selection,
                    self._context.selection.cutoff_text,
                )
            )
            self._context.refresh_objects(prepared.group_name)
            reference = next(
                member
                for member in prepared.members
                if member.rank == prepared.reference_rank
            )
            self._viewer.update_token_selection(
                ALIGNMENT_CORE_SELECTION_NAME,
                () if prepared.skip_alignment else prepared.core_indices,
                ((reference.obj_name, reference.token_map),),
            )
        except Exception as exc:
            self._fail_transaction(transaction, exc)
            return
        self._transaction = None
        lease = self._operations.active
        if lease is not None:
            self._operations.finish(lease)
        mode = (
            "current coordinates"
            if prepared.skip_alignment
            else "automatic core alignment"
        )
        selection_note = (
            ""
            if prepared.skip_alignment
            else f"\nAlignment core: '{ALIGNMENT_CORE_SELECTION_NAME}'."
        )
        self._presenter.present_notice(
            Notice(
                "ensemble_loaded",
                f"Loaded {len(members)} ensemble models into group "
                f"'{prepared.group_name}'.\nRMSD was computed using {mode}."
                f"{selection_note}",
                severity="information",
                title=APP_TITLE,
            )
        )

    def _restore_previous(self, transaction: EnsembleActivationTransaction) -> None:
        if transaction.previous_model_store is not None:
            self._prediction.restore_model_store(transaction.previous_model_store)
        self._state.ensemble = transaction.previous_ensemble
        if transaction.previous_viewer_context is not None:
            self._viewer.restore_paint_mappings(transaction.previous_viewer_context)

    def _fail_transaction(
        self, transaction: EnsembleActivationTransaction, exc: Exception
    ) -> None:
        self.rollback(refresh_context=True)
        lease = self._operations.active
        if lease is not None:
            self._fail(lease, str(exc))

    def rollback(self, *, refresh_context: bool) -> None:
        transaction = self._transaction
        if transaction is None:
            return
        prepared = transaction.prepared
        try:
            if transaction.group_existed:
                self._viewer.remove_from_group(
                    prepared.group_name, transaction.group_additions
                )
            else:
                self._viewer.delete_names((prepared.group_name,))
        except Exception:
            logger.exception("Could not restore the previous ensemble group")
        try:
            members = {member.rank: member for member in prepared.members}

            def revert() -> None:
                for transform in reversed(transaction.applied_transforms):
                    member = members[transform.rank]
                    if self._viewer.name_exists(member.obj_name):
                        rotation, translation = ensemble.invert_rigid_transform(
                            transform.rotation, transform.translation
                        )
                        self._viewer.transform(member.obj_name, rotation, translation)

            if transaction.applied_transforms:
                self._viewer.run_suspended(revert)
        except Exception:
            logger.exception("Could not restore ensemble transforms")
        try:
            self._viewer.delete_names(tuple(reversed(transaction.created_objects)))
            if transaction.created_objects:
                self._viewer.rebuild()
        except Exception:
            logger.exception("Could not remove partial ensemble objects")
        self._restore_previous(transaction)
        self._transaction = None
        if refresh_context:
            self._context.refresh_objects(transaction.previous_target)

    def _on_error(self, request_id: int, failure: object) -> None:
        lease = self._operations.active
        if lease is None or lease.request_id != request_id:
            return
        traceback_text = getattr(failure, "traceback_text", "")
        if traceback_text:
            logger.error("Background ensemble preparation failed:\n%s", traceback_text)
        self._fail(lease, str(getattr(failure, "message", failure)))

    def _fail(self, lease: OperationLease, message: str) -> None:
        self._operations.finish(lease)
        self._presenter.present_notice(
            Notice(
                "ensemble_activation_failed",
                message,
                severity="error",
                title=f"{APP_TITLE} - error",
            )
        )

    def _ask_skip_alignment(self) -> bool | None:
        selected = self._presenter.choose(
            ChoiceRequest(
                "ensemble_alignment",
                f"{APP_TITLE} Ensemble",
                "Choose automatic high-confidence core alignment or keep the "
                f"current {VIEWER_NAME} coordinates.",
                (
                    ChoiceOption("align", "Align automatically"),
                    ChoiceOption("current", "Use current coordinates"),
                    ChoiceOption("cancel", "Cancel", role="reject"),
                ),
                default_key="align",
            )
        )
        if selected in {None, "cancel"}:
            return None
        return selected == "current"

    def close(self) -> None:
        if self._transaction is not None:
            self.rollback(refresh_context=False)
