"""EnsembleLifecycle application workflow."""

from __future__ import annotations

from .lifecycle_support import (
    APP_TITLE,
    VIEWER_NAME,
    ChoiceOption,
    ChoiceRequest,
    EnsembleActivationTransaction,
    Notice,
    _prepare_ensemble_job,
    add_objects_to_group,
    delete_viewer_names,
    ensemble,
    get_group_members,
    inspect_object_tokens,
    load_structure_object_if_missing,
    logger,
    np,
    rebuild,
    remove_objects_from_group,
    run_with_updates_suspended,
    transform_object,
    viewer_name_exists,
)


class EnsembleLifecycleWorkflow:
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
            group_existed = self._viewer_operation(
                "name_exists", viewer_name_exists, prepared.group_name
            )
            previous_group_members = self._viewer_operation(
                "group_members", get_group_members, prepared.group_name
            )
        except Exception as exc:
            logger.exception("Could not inspect the target PyMOL ensemble group")
            self._finish_data_load(request_id)
            self._presenter.present_notice(
                Notice(
                    "ensemble_inspection_failed",
                    str(exc),
                    severity="error",
                    title=f"{APP_TITLE} - error",
                )
            )
            return
        transaction = EnsembleActivationTransaction(
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
        self.application.scheduler.call_soon(
            lambda: self._load_next_ensemble_object(transaction, 0)
        )

    def _ensemble_transaction_is_active(
        self,
        transaction: EnsembleActivationTransaction,
    ) -> bool:
        return bool(
            self._data_load_is_active(transaction.request_id)
            and self._active_ensemble_viewer_transaction is transaction
            and transaction.prepared.pred_files is self._pred_files
        )

    def _load_next_ensemble_object(
        self,
        transaction: EnsembleActivationTransaction,
        index: int,
    ) -> None:
        if not self._ensemble_transaction_is_active(transaction):
            return
        members = transaction.prepared.members
        if index >= len(members):
            self.application.scheduler.call_soon(
                lambda: self._inspect_next_ensemble_object(transaction, 0)
            )
            return

        member = members[index]
        self._on_data_load_progress(
            transaction.request_id,
            f"Loading {member.model_label} into PyMOL… ({index + 1}/{len(members)})",
        )
        try:
            did_load = self._viewer_operation(
                "load_structure_object_if_missing",
                load_structure_object_if_missing,
                member.structure_path,
                member.obj_name,
            )
            if did_load:
                transaction.created_objects.append(member.obj_name)
        except Exception as exc:
            self._fail_ensemble_viewer_transaction(transaction, exc)
            return
        self.application.scheduler.call_soon(
            lambda: self._load_next_ensemble_object(transaction, index + 1)
        )

    def _inspect_next_ensemble_object(
        self,
        transaction: EnsembleActivationTransaction,
        index: int,
    ) -> None:
        if not self._ensemble_transaction_is_active(transaction):
            return
        members = transaction.prepared.members
        if index >= len(members):
            self.application.scheduler.call_soon(
                lambda: self._align_and_group_ensemble(transaction)
            )
            return

        member = members[index]
        self._on_data_load_progress(
            transaction.request_id,
            f"Inspecting {member.model_label} coordinates… "
            f"({index + 1}/{len(members)})",
        )
        try:
            if not self._viewer_operation(
                "name_exists", viewer_name_exists, member.obj_name
            ):
                raise ValueError(f"PyMOL object '{member.obj_name}' no longer exists.")
            transaction.inspections[member.rank] = self._viewer_operation(
                "inspect_tokens",
                inspect_object_tokens,
                member.obj_name,
                member.token_map,
            )
        except Exception as exc:
            self._fail_ensemble_viewer_transaction(transaction, exc)
            return
        self.application.scheduler.call_soon(
            lambda: self._inspect_next_ensemble_object(transaction, index + 1)
        )

    def _align_and_group_ensemble(
        self,
        transaction: EnsembleActivationTransaction,
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
                        self._viewer_operation(
                            "transform",
                            transform_object,
                            member.obj_name,
                            transform.rotation,
                            transform.translation,
                        )
                        transaction.applied_transforms.append(transform)

                self._viewer_operation(
                    "run_suspended", run_with_updates_suspended, apply_transforms
                )

            self._on_data_load_progress(
                transaction.request_id,
                "Grouping ensemble objects…",
            )
            object_names = tuple(member.obj_name for member in prepared.members)
            previous_members = set(transaction.previous_group_members)
            transaction.group_additions = tuple(
                name for name in object_names if name not in previous_members
            )
            self._viewer_operation(
                "run_suspended",
                run_with_updates_suspended,
                lambda: self._viewer_operation(
                    "add_to_group",
                    add_objects_to_group,
                    prepared.group_name,
                    object_names,
                ),
            )
        except Exception as exc:
            self._fail_ensemble_viewer_transaction(transaction, exc)
            return

        self._commit_ensemble_transaction(transaction, rmsd)

    def _commit_ensemble_transaction(
        self,
        transaction: EnsembleActivationTransaction,
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
        self._presenter.present_notice(
            Notice(
                "ensemble_loaded",
                f"Loaded {len(members)} ensemble models into group "
                f"'{prepared.group_name}'.\n"
                f"RMSD was computed using {mode_label}.\n\n"
                "Use Apply Coloring to color the selected target.",
                severity="information",
                title=APP_TITLE,
            )
        )

    def _restore_previous_ensemble_state(
        self,
        transaction: EnsembleActivationTransaction,
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
        self._presenter.present_notice(
            Notice(
                "ensemble_preparation_failed",
                failure.message,
                severity="error",
                title=f"{APP_TITLE} - error",
            )
        )

    def _fail_ensemble_viewer_transaction(
        self,
        transaction: EnsembleActivationTransaction,
        exc: Exception,
    ) -> None:
        if not self._ensemble_transaction_is_active(transaction):
            return
        logger.exception("Could not load or align the ensemble in PyMOL")
        self._rollback_ensemble_viewer_transaction(refresh_gui=True)
        self._finish_data_load(transaction.request_id, save_session=True)
        self._presenter.present_notice(
            Notice(
                "ensemble_activation_failed",
                str(exc),
                severity="error",
                title=f"{APP_TITLE} - error",
            )
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
                self._viewer_operation(
                    "remove_from_group",
                    remove_objects_from_group,
                    prepared.group_name,
                    transaction.group_additions,
                )
            else:
                self._viewer_operation(
                    "delete_names",
                    delete_viewer_names,
                    (prepared.group_name,),
                )
        except Exception:
            logger.exception("Could not restore the previous ensemble group")

        try:
            members_by_rank = {member.rank: member for member in prepared.members}

            def revert_transforms() -> None:
                for transform in reversed(transaction.applied_transforms):
                    member = members_by_rank[transform.rank]
                    if not self._viewer_operation(
                        "name_exists", viewer_name_exists, member.obj_name
                    ):
                        continue
                    rotation, translation = ensemble.invert_rigid_transform(
                        transform.rotation,
                        transform.translation,
                    )
                    self._viewer_operation(
                        "transform",
                        transform_object,
                        member.obj_name,
                        rotation,
                        translation,
                    )

            if transaction.applied_transforms:
                self._viewer_operation(
                    "run_suspended", run_with_updates_suspended, revert_transforms
                )
        except Exception:
            logger.exception("Could not restore transformed ensemble objects")

        try:
            self._viewer_operation(
                "delete_names",
                delete_viewer_names,
                tuple(reversed(transaction.created_objects)),
            )
            if transaction.created_objects:
                self._viewer_operation("rebuild", rebuild)
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
        selected = self._presenter.choose(
            ChoiceRequest(
                code="ensemble_alignment",
                title=f"{APP_TITLE} Ensemble",
                message=(
                    "Load all models as separate objects and group them. "
                    "Choose automatic high-confidence core alignment or keep "
                    f"the current {VIEWER_NAME} coordinates."
                ),
                options=(
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
