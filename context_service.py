"""Typed target, metric, plot, and preview context derivation."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from . import gui_rules, metrics, plot_data, reports
from .ensemble import EnsembleMember
from .gui_services import (
    ContextSelection,
    ContextViewState,
    DialogViewPort,
    ModelChoice,
    TargetChoice,
    ViewerPort,
)
from .gui_state import PluginState
from .loader_models import DataCapability
from .model_state import ModelState
from .presentation import PresentationPort
from .token_map import TokenMap

TargetKind = Literal["none", "single", "ensemble_member", "ensemble_group"]


class ContextService:
    """Derive deterministic view state without reading or mutating widgets."""

    def __init__(
        self,
        state: PluginState,
        viewer: ViewerPort,
        presenter: PresentationPort,
        view: DialogViewPort,
        metric_rows: dict[str, int],
    ) -> None:
        self._state = state
        self._viewer = viewer
        self._presenter = presenter
        self._view = view
        self._metric_rows = dict(metric_rows)
        self._selection = ContextSelection()

    @property
    def selection(self) -> ContextSelection:
        return self._selection

    def set_selection(self, selection: ContextSelection) -> None:
        self._selection = selection

    def refresh(self, selection: ContextSelection | None = None) -> ContextViewState:
        if selection is not None:
            self._selection = selection
        state = self.derive_view_state()
        self._view.apply_context(state)
        return state

    def refresh_objects(self, preferred_target: str | None = None) -> ContextViewState:
        ensemble_state = self._state.ensemble
        additional = () if ensemble_state is None else (ensemble_state.group_name,)
        try:
            names = self._ordered_target_names(
                self._viewer.object_names(additional_names=additional)
            )
        except Exception:
            names = []
        selected = preferred_target or self._selection.target_name
        if selected not in names:
            selected = names[0] if names else ""
        self._selection = replace(self._selection, target_name=selected)
        state = self.derive_view_state(target_names=names)
        self._view.apply_context(state)
        return state

    def _ordered_target_names(self, names: list[str]) -> list[str]:
        ensemble_state = self._state.ensemble
        group_name = None if ensemble_state is None else ensemble_state.group_name
        members = sorted(
            () if ensemble_state is None else ensemble_state.members,
            key=lambda member: member.rank,
        )
        member_names = [member.obj_name for member in members]
        available = set(names)
        ordered: list[str] = []
        if group_name in available:
            ordered.append(group_name)
        ordered.extend(name for name in member_names if name in available)
        handled = set(ordered)
        ordered.extend(
            sorted((name for name in names if name not in handled), key=str.casefold)
        )
        return ordered

    def target_kind(self) -> TargetKind:
        target_name = self._selection.target_name
        if not target_name:
            return "none"
        ensemble_state = self._state.ensemble
        if ensemble_state is not None and target_name == ensemble_state.group_name:
            return "ensemble_group"
        if self.selected_ensemble_member(target_name) is not None:
            return "ensemble_member"
        return "single"

    def selected_ensemble_member(self, target_name: str) -> EnsembleMember | None:
        ensemble_state = self._state.ensemble
        if ensemble_state is None:
            return None
        return next(
            (
                member
                for member in ensemble_state.members
                if member.obj_name == target_name
            ),
            None,
        )

    def target_model_states(self) -> tuple[ModelState, ...]:
        ensemble_state = self._state.ensemble
        kind = self.target_kind()
        if kind == "ensemble_group" and ensemble_state is not None:
            return tuple(
                model_state
                for member in sorted(ensemble_state.members, key=lambda item: item.rank)
                if (model_state := self._state.model_states.get(member.rank))
                is not None
            )
        if kind == "ensemble_member":
            member = self.selected_ensemble_member(self._selection.target_name)
            model_state = (
                None if member is None else self._state.model_states.get(member.rank)
            )
            return () if model_state is None else (model_state,)
        active = self._state.active_model_state
        return () if active is None else (active,)

    def state_supports_family(self, state: ModelState, family: DataCapability) -> bool:
        data_attr = "token_plddt" if family == "plddt" else family
        if getattr(state.data, data_attr, None) is not None:
            return True
        files = self._state.pred_files
        return bool(files is not None and files.model(state.rank).supports(family))

    def target_all_supports_family(self, family: DataCapability) -> bool:
        states = self.target_model_states()
        return bool(states) and all(
            self.state_supports_family(state, family) for state in states
        )

    def target_any_supports_family(self, family: DataCapability) -> bool:
        return any(
            self.state_supports_family(state, family)
            for state in self.target_model_states()
        )

    def has_fingerprint_data(self) -> bool:
        return any(
            self.target_any_supports_family(family)
            for family in ("pae", "pde", "contact_probs", "plddt")
        )

    def has_matrix_data_family(self, family: Literal["pae", "pde"]) -> bool:
        return family in {"pae", "pde"} and self.target_all_supports_family(family)

    def current_target_has_multiple_chains(self) -> bool:
        states = self.target_model_states()
        return bool(states and plot_data.has_multiple_token_chains(states[0].token_map))

    def ensemble_action_state(self) -> tuple[bool, str]:
        files = self._state.pred_files
        if files is None:
            return False, "Load a prediction with at least two models first."
        if not files.supports_ensemble:
            return False, "Ensemble mode requires at least two model files."
        if self._state.ensemble is not None:
            return False, "The ensemble for this prediction is already loaded."
        return (
            True,
            "Load all ranked models as an ensemble and compute ensemble-level metrics.",
        )

    def metric_availability(self) -> tuple[tuple[int, bool], ...]:
        if self._state.active_model_state is None or self._state.pred_files is None:
            return ()
        target_states = self.target_model_states()
        family_available = {
            family: self.target_all_supports_family(family)
            for family in ("pae", "pde", "plddt", "contact_probs")
        }
        family_available["confidence"] = bool(target_states) and all(
            state.data.confidence is not None for state in target_states
        )
        has_chain_iptm = bool(target_states) and all(
            state.data.confidence is not None and state.data.confidence.has_chain_iptm
            for state in target_states
        )
        availability: list[tuple[int, bool]] = []
        for spec in metrics.METRICS:
            row = self._metric_rows.get(spec.key)
            if row is None:
                continue
            available = all(family_available[item] for item in spec.requirements)
            if spec.key == "chain_iptm":
                available = available and has_chain_iptm
            if spec.ensemble_level:
                available = available and self._state.ensemble is not None
            availability.append((row, available))
        return tuple(availability)

    def first_available_metric(self) -> str | None:
        available_rows = {
            row for row, available in self.metric_availability() if available
        }
        return next(
            (
                spec.key
                for spec in metrics.METRICS
                if self._metric_rows.get(spec.key) in available_rows
            ),
            None,
        )

    def plot_availability(self) -> tuple[tuple[str, bool, str], ...]:
        selection = self._selection
        availability: list[tuple[str, bool, str]] = []
        for spec in metrics.PLOTS:
            state = gui_rules.plot_action_state(
                spec.key,
                selection.metric_key,
                self.target_kind(),
                bool(selection.reference_selection),
                self._state.ensemble is not None,
                has_fingerprint_data=self.has_fingerprint_data(),
                has_pae_data=self.has_matrix_data_family("pae"),
                has_pde_data=self.has_matrix_data_family("pde"),
                has_multiple_chains=self.current_target_has_multiple_chains(),
            )
            availability.append(
                (spec.key, state.enabled, state.reason or f"Show {spec.label.lower()}.")
            )
        return tuple(availability)

    def derive_view_state(
        self, *, target_names: list[str] | None = None
    ) -> ContextViewState:
        selection = self._selection
        target_kind = self.target_kind()
        field = gui_rules.field_context(
            selection.metric_key,
            target_kind,
            self._state.ensemble is not None,
            self.has_fingerprint_data(),
        )
        if target_names is None:
            ensemble_state = self._state.ensemble
            additional = () if ensemble_state is None else (ensemble_state.group_name,)
            try:
                target_names = self._ordered_target_names(
                    self._viewer.object_names(additional_names=additional)
                )
            except Exception:
                target_names = []
        ensemble_state = self._state.ensemble
        group_name = None if ensemble_state is None else ensemble_state.group_name
        member_names = {
            member.obj_name
            for member in (() if ensemble_state is None else ensemble_state.members)
        }
        targets = tuple(
            TargetChoice(
                name,
                (
                    "ensemble_group"
                    if name == group_name
                    else "ensemble_member"
                    if name in member_names
                    else "single"
                ),
            )
            for name in target_names
        )
        files = self._state.pred_files
        ensemble_enabled, ensemble_tooltip = self.ensemble_action_state()
        model_choices = (
            ()
            if files is None
            else tuple(
                ModelChoice(model.rank, model.display_label) for model in files.models
            )
        )
        active = self._state.active_model_state
        return ContextViewState(
            metric_availability=self.metric_availability(),
            plot_availability=self.plot_availability(),
            reference_label=field.ref_label,
            reference_tooltip=field.ref_tooltip,
            reference_enabled=field.ref_enabled,
            cutoff_label=field.cutoff_label,
            cutoff_tooltip=field.cutoff_tooltip,
            cutoff_enabled=field.cutoff_enabled,
            confidence_text=reports.format_confidence_summary(
                None if active is None else active.data
            ),
            preview_text=gui_rules.metric_preview_text(
                selection.metric_key,
                target_kind,
                selection.reference_selection,
                selection.cutoff_text,
                self._state.ensemble is not None,
            ),
            ensemble_enabled=ensemble_enabled,
            ensemble_tooltip=ensemble_tooltip,
            model_choices=model_choices,
            target_choices=targets,
            selected_rank=self._state.active_model_rank,
            selected_target=selection.target_name or None,
        )

    def show_statistics_for_single(
        self,
        key: str,
        target_name: str,
        values: np.ndarray,
        *,
        include_plddt_classes: bool = False,
        include_chain_stats: bool = False,
        include_domain_labels: bool = False,
        token_map: TokenMap | None = None,
    ) -> None:
        self._presenter.show_statistics(
            reports.format_statistics_report(
                key,
                target_name,
                [(target_name, values, token_map)],
                include_plddt_classes=include_plddt_classes,
                include_chain_stats=include_chain_stats,
                include_domain_labels=include_domain_labels,
            )
        )

    def show_statistics_for_members(
        self,
        key: str,
        target_label: str,
        member_values: list[tuple[EnsembleMember, np.ndarray]],
        *,
        include_plddt_classes: bool = False,
        include_chain_stats: bool = False,
        include_domain_labels: bool = False,
    ) -> None:
        entries = [
            (
                member.obj_name,
                values,
                (
                    None
                    if (state := self._state.model_states.get(member.rank)) is None
                    else state.token_map
                ),
            )
            for member, values in member_values
        ]
        self._presenter.show_statistics(
            reports.format_statistics_report(
                key,
                target_label,
                entries,
                include_plddt_classes=include_plddt_classes,
                include_chain_stats=include_chain_stats,
                include_domain_labels=include_domain_labels,
            )
        )
