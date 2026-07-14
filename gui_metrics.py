"""Lazy data loading and metric-computation GUI coordination."""

from __future__ import annotations

import numpy as np

from . import compute, metrics
from .compat import MessageBoxStandardButton, QtWidgets
from .gui_state import MetricContext
from .mol_viewer import (
    ObjectPaintMapping,
    ensure_object_paint_mapping,
    get_viewer_name,
    selection_to_token_indices,
    tokens_within_distance,
)
from .token_map import TokenMap

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()


class MetricController:
    def _require_active_model_state(self):
        """Return the active canonical model state or raise a user-facing error."""
        state = self._active_model_state
        if state is None:
            raise ValueError("No prediction data loaded; cannot resolve token map.")
        return state

    def _compute_property_for(
        self,
        key: str,
        ref_sel: str | None,
        data,
        tm: TokenMap,
        obj_name: str,
    ):
        """Resolve GUI/viewer context and dispatch one per-model metric."""

        def _need_ref():
            if not ref_sel:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "This property requires a reference selection.\n"
                    "Enter a viewer selection in the Reference field.",
                )
                return None
            indices = selection_to_token_indices(tm, ref_sel, obj_name=obj_name)
            if not indices:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    f"Reference selection '{ref_sel}' matched no tokens in {obj_name}.",
                )
                return None
            return indices

        ref_indices = None
        contact_indices = None
        cutoff = None

        if key == "ensemble_rmsd":
            QtWidgets.QMessageBox.information(
                self,
                APP_TITLE,
                "Ensemble RMSD requires all models to be loaded.\n"
                "Use the Ensemble… button.",
            )
            return None

        if (
            key == "contact_prob_to_sel"
            and getattr(data, "contact_probs", None) is None
        ):
            return self._compute_metric_with_messages(key, data, tm)

        if key in {
            "pae_to_sel",
            "pae_col_to_sel",
            "pae_sym_sel",
            "pae_sym_within_sel",
            "pae_contact",
            "pde_to_sel",
            "pde_within_sel",
            "pde_contact",
            "contact_prob_to_sel",
        }:
            ref_indices = _need_ref()
            if ref_indices is None:
                return None

        if key in ("pae_domain_complete", "pae_domain_spectral"):
            cutoff = self._get_cutoff_threshold()
            if cutoff is None:
                return None
            method = compute.pae_domain_method(key)
            if not self._pae_domain_dependency_available(method):
                return None

        if key in metrics.CONTACT_FILTERED_METRICS:
            cutoff = self._get_cutoff_threshold()
            if cutoff is None:
                return None
            contact_indices = self._binding_site_token_indices(
                tm, obj_name, ref_sel, ref_indices, cutoff
            )
            if not contact_indices:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "No polymer binding-site residues were found within "
                    f"{cutoff:g} Å of the reference selection.",
                )
                return None

        return self._compute_metric_with_messages(
            key,
            data,
            tm,
            ref_indices=ref_indices,
            contact_indices=contact_indices,
            cutoff=cutoff,
        )

    def _compute_property_from_context(
        self,
        key: str,
        data,
        tm: TokenMap,
        context: MetricContext,
    ):
        """Dispatch a metric using already-resolved export context."""
        cutoff = context.cutoff_angstrom
        if key in ("pae_domain_complete", "pae_domain_spectral"):
            method = compute.pae_domain_method(key)
            if not self._pae_domain_dependency_available(method):
                return None
        return self._compute_metric_with_messages(
            key,
            data,
            tm,
            ref_indices=list(context.reference_indices),
            contact_indices=list(context.contact_indices),
            cutoff=cutoff,
        )

    def _compute_metric_with_messages(
        self,
        key: str,
        data,
        tm: TokenMap,
        *,
        ref_indices: list[int] | None = None,
        contact_indices: list[int] | None = None,
        cutoff: float | None = None,
    ):
        """Call pure compute dispatch and translate expected errors to GUI text."""
        try:
            return compute.compute_metric(
                key,
                data,
                tm,
                ref_indices=ref_indices,
                contact_indices=contact_indices,
                cutoff=cutoff,
            )
        except compute.MissingMetricDataError:
            if key == "plddt":
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, "pLDDT data are not available for this model."
                )
            elif key.startswith("contact_prob"):
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, "Interaction probability data are not available."
                )
            elif key == "chain_iptm":
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, "Confidence JSON not available."
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, "Required metric data are not available."
                )
            return None
        except compute.MissingReferenceError:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "This property requires a reference selection.\n"
                f"Enter a {VIEWER_NAME} selection in the Reference field.",
            )
            return None
        except compute.MissingContactError:
            QtWidgets.QMessageBox.warning(
                self,
                APP_TITLE,
                "No polymer binding-site residues were found within "
                "the cutoff of the reference selection.",
            )
            return None
        except compute.UnsupportedMetricError:
            if key == "ensemble_rmsd":
                QtWidgets.QMessageBox.information(
                    self,
                    APP_TITLE,
                    "Ensemble RMSD requires all models to be loaded.\n"
                    "Use the Ensemble… button.",
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self, APP_TITLE, f"Unknown property key: {key}"
                )
            return None
        except compute.MetricComputationError as exc:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, str(exc))
            return None

    def _pae_domain_dependency_available(self, method: str) -> bool:
        """Offer to install a missing PAE domain-label dependency."""
        feature = {
            "complete_linkage": "pae_domain_complete",
            "spectral": "pae_domain_spectral",
        }.get(method)
        if feature is None:
            return True
        label = (
            "complete-linkage PAE domain labels"
            if method == "complete_linkage"
            else "spectral PAE domain labels"
        )
        return self._ensure_feature_dependencies((feature,), feature_label=label)

    def _compute_ensemble_property(self, key: str) -> np.ndarray:
        """Return an ensemble-level per-token array."""
        ensemble_state = getattr(self, "_ensemble", None)
        if ensemble_state is None:
            raise ValueError("No active ensemble.")

        if key == "ensemble_rmsd":
            return ensemble_state.rmsd

        if key in ("ensemble_plddt_mean", "ensemble_plddt_std"):
            return (
                ensemble_state.plddt_mean
                if key == "ensemble_plddt_mean"
                else ensemble_state.plddt_std
            )

        raise ValueError(f"Unknown ensemble property: {key}")

    def _validate_token_count(self, values, token_map: TokenMap, obj_name: str) -> None:
        """Raise a helpful error if a property array does not match a token map."""
        if values is None or token_map is None:
            raise ValueError("No values or token map available for coloring.")
        if len(values) != len(token_map):
            raise ValueError(
                f"Token count mismatch for {obj_name}: property has {len(values)} "
                f"values, but the loaded structure maps to {len(token_map)} tokens. "
                f"Check that the {VIEWER_NAME} object belongs to the selected prediction model."
            )

    def _confirm_token_overlap_for_coloring(
        self,
        token_map: TokenMap,
        obj_name: str,
        data=None,
        *,
        threshold: float = 0.50,
        mapping: ObjectPaintMapping | None = None,
    ) -> bool:
        """Warn when the selected viewer object barely overlaps the token map."""
        if token_map is None:
            return True

        cache_key = self._paint_mapping_cache_key(data, obj_name)
        try:
            if mapping is None:
                mapping = self._prepare_paint_mapping(token_map, obj_name, data)
            overlap = mapping.overlap
        except Exception:
            return True
        accepted = getattr(self, "_accepted_token_overlap_warnings", set())
        if cache_key in accepted:
            return True

        if overlap.target_tokens <= 0 or overlap.target_coverage >= threshold:
            return True

        pct = overlap.target_coverage * 100.0
        pred_pct = overlap.prediction_coverage * 100.0
        message = (
            f"The selected viewer target '{obj_name}' has low overlap with the "
            "loaded prediction token map.\n\n"
            f"Matched target tokens: {overlap.matched_target_tokens} / "
            f"{overlap.target_tokens} ({pct:.1f}%).\n"
            f"Matched prediction tokens: {overlap.matched_prediction_tokens} / "
            f"{overlap.prediction_tokens} ({pred_pct:.1f}%).\n\n"
            "Coloring this target may be meaningless if it is unrelated to the "
            "prediction, but it can be useful for deliberate copies or partial "
            "models. Apply the coloring anyway?"
        )
        buttons = MessageBoxStandardButton.Yes | MessageBoxStandardButton.Cancel
        result = QtWidgets.QMessageBox.question(
            self,
            APP_TITLE,
            message,
            buttons,
            MessageBoxStandardButton.Cancel,
        )
        if result != MessageBoxStandardButton.Yes:
            return False

        accepted.add(cache_key)
        self._accepted_token_overlap_warnings = accepted
        return True

    def _paint_mapping_cache_key(self, data, obj_name: str) -> tuple[str, str]:
        structure_path = getattr(data, "structure_path", None)
        if structure_path is None:
            state = getattr(self, "_active_model_state", None)
            structure_path = getattr(
                None if state is None else state.data, "structure_path", ""
            )
        return str(structure_path), str(obj_name)

    def _prepare_paint_mapping(
        self,
        token_map: TokenMap,
        obj_name: str,
        data=None,
    ) -> ObjectPaintMapping:
        """Return a valid cached atom-index mapping for one viewer object."""
        cache_key = self._paint_mapping_cache_key(data, obj_name)
        existing = getattr(self, "_paint_mappings", {}).get(cache_key)
        mapping, rebuilt = ensure_object_paint_mapping(obj_name, token_map, existing)
        mappings = dict(getattr(self, "_paint_mappings", {}))
        mappings[cache_key] = mapping
        self._paint_mappings = mappings
        if rebuilt:
            accepted = set(getattr(self, "_accepted_token_overlap_warnings", set()))
            accepted.discard(cache_key)
            self._accepted_token_overlap_warnings = accepted
        return mapping

    def _binding_site_token_indices(
        self,
        token_map: TokenMap,
        obj_name: str,
        ref_sel: str,
        ref_indices: list[int],
        cutoff: float,
    ) -> list[int]:
        """Return polymer tokens with any atom within *cutoff* Å of reference."""
        raw_binding_indices = tokens_within_distance(
            token_map, obj_name, ref_sel, cutoff
        )
        ref_set = set(ref_indices)
        return [
            idx
            for idx in raw_binding_indices
            if idx not in ref_set and not token_map[idx].is_hetatm
        ]

    def _ensure_current_data_for_property(self, prop: dict) -> None:
        """Assert that asynchronous preflight supplied the property's arrays."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        state = self._require_active_model_state()
        flags = metrics.metric_load_flags(prop)
        self._require_loaded_data(state.data, flags)

    def _ensure_member_data_for_property(self, member, prop: dict) -> None:
        """Assert that asynchronous preflight supplied a member's arrays."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        flags = metrics.metric_load_flags(prop)
        state = self._canonical_state_for_ensemble_member(member)
        self._require_loaded_data(state.data, flags)

    def _ensure_member_data_for_plot(
        self,
        member,
        *,
        load_pae: bool = False,
        load_pde: bool = False,
        load_contact_probs: bool = False,
        load_token_plddt: bool = False,
    ) -> None:
        """Assert that asynchronous preflight supplied plot arrays."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        state = self._canonical_state_for_ensemble_member(member)
        self._require_loaded_data(
            state.data,
            {
                "load_pae": load_pae,
                "load_pde": load_pde,
                "load_contact_probs": load_contact_probs,
                "load_token_plddt": load_token_plddt,
            },
        )

    def _require_loaded_data(
        self,
        data,
        flags: dict[str, bool],
    ) -> None:
        """Raise if a ready-data continuation is missing required arrays."""
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
                "Required prediction data are not available: " + ", ".join(missing)
            )
