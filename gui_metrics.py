"""Lazy data loading and metric-computation GUI coordination."""

from __future__ import annotations

import importlib.util

import numpy as np

from . import compute, metrics
from .compat import MessageBoxStandardButton, QtWidgets
from .gui_state import MetricContext
from .mol_viewer import (
    compare_token_map_to_object,
    get_viewer_name,
    selection_to_token_indices,
    tokens_within_distance,
)

APP_TITLE = "FoldQC"
VIEWER_NAME = get_viewer_name()


class MetricController:
    def _build_token_map_if_needed(self, obj_name: str) -> None:
        """(Re-)build the token map if the object changed.

        The structure path from the loaded prediction data is passed to
        ``build_token_map`` so that HETATM atom order is read from the file
        rather than from a viewer's potentially reordered internal model.
        """
        current_obj = getattr(self, "_token_map_obj", None)
        current_path = getattr(self, "_token_map_structure_path", None)
        structure_path = (
            None if self._pred_data is None else self._pred_data.structure_path
        )
        if (
            self._token_map is None
            or current_obj != obj_name
            or current_path != structure_path
        ):
            if self._pred_data is None:
                raise ValueError("No prediction data loaded; cannot build token map.")
            from .token_map import build_token_map

            self._token_map = build_token_map(self._pred_data.structure_path)
            self._token_map_obj = obj_name  # type: ignore[attr-defined]
            self._token_map_structure_path = self._pred_data.structure_path

    def _compute_property_for(
        self,
        key: str,
        ref_sel: str | None,
        data,
        tm,
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
        tm,
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
        tm,
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
        """Warn and return False when a PAE domain-label dependency is missing."""
        if method == "complete_linkage":
            if importlib.util.find_spec("scipy") is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "PAE domain labels (complete linkage) require SciPy. "
                    f"Install scipy in the Python environment used by {VIEWER_NAME}.",
                )
                return False
            return True
        if method == "spectral":
            missing = []
            if importlib.util.find_spec("scipy") is None:
                missing.append("SciPy")
            if importlib.util.find_spec("sklearn") is None:
                missing.append("scikit-learn")
            if missing:
                QtWidgets.QMessageBox.warning(
                    self,
                    APP_TITLE,
                    "PAE domain labels (spectral clustering) require "
                    f"{' and '.join(missing)}. Install the missing package(s) "
                    f"in the Python environment used by {VIEWER_NAME}.",
                )
                return False
            return True
        return True

    def _compute_ensemble_property(self, key: str) -> np.ndarray:
        """Return an ensemble-level per-token array."""
        members = self._ensemble_members or []
        if not members:
            raise ValueError("No active ensemble.")

        if key == "ensemble_rmsd":
            if self._ensemble_rmsd is None:
                raise ValueError(
                    "Ensemble RMSD has not been computed. Use the Ensemble… button again."
                )
            return self._ensemble_rmsd

        if key in ("ensemble_plddt_mean", "ensemble_plddt_std"):
            if self._ensemble_plddt_mean is None or self._ensemble_plddt_std is None:
                raise ValueError(
                    "Ensemble pLDDT consensus has not been computed. "
                    "Use the Ensemble… button again."
                )
            return (
                self._ensemble_plddt_mean
                if key == "ensemble_plddt_mean"
                else self._ensemble_plddt_std
            )

        raise ValueError(f"Unknown ensemble property: {key}")

    def _validate_token_count(self, values, token_map, obj_name: str) -> None:
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
        token_map,
        obj_name: str,
        data=None,
        *,
        threshold: float = 0.50,
    ) -> bool:
        """Warn when the selected viewer object barely overlaps the token map."""
        if token_map is None:
            return True

        structure_path = getattr(data, "structure_path", None)
        if structure_path is None:
            structure_path = getattr(
                getattr(self, "_pred_data", None), "structure_path", ""
            )
        cache_key = (str(structure_path), str(obj_name))
        accepted = getattr(self, "_accepted_token_overlap_warnings", set())
        if cache_key in accepted:
            return True

        try:
            overlap = compare_token_map_to_object(token_map, obj_name)
        except Exception:
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

    def _binding_site_token_indices(
        self,
        token_map,
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
        """Reload current single-model data if a lazy property needs more arrays."""
        if self._pred_files is None or self._pred_data is None:
            raise ValueError("No prediction output loaded.")
        flags = metrics.metric_load_flags(prop)
        data = self._pred_data
        if prop.get("needs_any_plddt", False) and (
            getattr(data, "structure_plddt", None) is not None
            or getattr(data, "plddt", None) is not None
        ):
            flags["load_structure_plddt"] = False
            flags["load_plddt"] = False
        needs_reload = (
            (flags["load_pae"] and getattr(data, "pae", None) is None)
            or (flags["load_pde"] and getattr(data, "pde", None) is None)
            or (
                flags["load_contact_probs"]
                and getattr(data, "contact_probs", None) is None
            )
            or (
                flags["load_structure_plddt"]
                and getattr(data, "structure_plddt", None) is None
            )
            or (flags["load_plddt"] and getattr(data, "plddt", None) is None)
        )
        if needs_reload:
            self._pred_data = self._reload_prediction_data(
                data.rank,
                load_pae=flags["load_pae"] or getattr(data, "pae", None) is not None,
                load_pde=flags["load_pde"] or getattr(data, "pde", None) is not None,
                load_contact_probs=flags["load_contact_probs"]
                or getattr(data, "contact_probs", None) is not None,
                load_structure_plddt=flags["load_structure_plddt"]
                or getattr(data, "structure_plddt", None) is not None,
                load_plddt=flags["load_plddt"]
                or getattr(data, "plddt", None) is not None,
            )

    def _reload_prediction_data(self, rank: int, **flags):
        """Load one model while preserving the provider-aware loader defaults."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        from .loader import load_prediction_data

        return load_prediction_data(self._pred_files, rank, **flags)

    def _ensure_member_data_for_property(self, member, prop: dict) -> None:
        """Reload member data with large matrices only when the property needs them."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        flags = metrics.metric_load_flags(prop)
        needs_pae = flags["load_pae"]
        needs_pde = flags["load_pde"]
        needs_contact_probs = flags["load_contact_probs"]
        needs_structure_plddt = flags["load_structure_plddt"]
        needs_plddt = flags["load_plddt"]
        if prop.get("needs_any_plddt", False) and (
            getattr(member.data, "structure_plddt", None) is not None
            or getattr(member.data, "plddt", None) is not None
        ):
            needs_structure_plddt = False
            needs_plddt = False
        if (
            (needs_pae and getattr(member.data, "pae", None) is None)
            or (needs_pde and getattr(member.data, "pde", None) is None)
            or (
                needs_contact_probs
                and getattr(member.data, "contact_probs", None) is None
            )
            or (
                needs_structure_plddt
                and getattr(member.data, "structure_plddt", None) is None
            )
            or (needs_plddt and getattr(member.data, "plddt", None) is None)
        ):
            member.data = self._reload_prediction_data(
                member.rank,
                load_pae=needs_pae or getattr(member.data, "pae", None) is not None,
                load_pde=needs_pde or getattr(member.data, "pde", None) is not None,
                load_contact_probs=needs_contact_probs
                or getattr(member.data, "contact_probs", None) is not None,
                load_structure_plddt=needs_structure_plddt
                or getattr(member.data, "structure_plddt", None) is not None,
                load_plddt=needs_plddt
                or getattr(member.data, "plddt", None) is not None,
            )

    def _ensure_member_data_for_plot(
        self,
        member,
        *,
        load_pae: bool = False,
        load_pde: bool = False,
        load_contact_probs: bool = False,
        load_structure_plddt: bool = False,
        load_plddt: bool = False,
    ) -> None:
        """Reload an ensemble member while preserving already-loaded plot arrays."""
        if self._pred_files is None:
            raise ValueError("No prediction output loaded.")
        data = member.data
        if (
            load_structure_plddt
            and load_plddt
            and (
                getattr(data, "structure_plddt", None) is not None
                or getattr(data, "plddt", None) is not None
            )
        ):
            load_structure_plddt = False
            load_plddt = False
        needs_reload = (
            (load_pae and getattr(data, "pae", None) is None)
            or (load_pde and getattr(data, "pde", None) is None)
            or (load_contact_probs and getattr(data, "contact_probs", None) is None)
            or (load_structure_plddt and getattr(data, "structure_plddt", None) is None)
            or (load_plddt and getattr(data, "plddt", None) is None)
        )
        if not needs_reload:
            return

        member.data = self._reload_prediction_data(
            member.rank,
            load_pae=load_pae or getattr(data, "pae", None) is not None,
            load_pde=load_pde or getattr(data, "pde", None) is not None,
            load_contact_probs=load_contact_probs
            or getattr(data, "contact_probs", None) is not None,
            load_structure_plddt=load_structure_plddt
            or getattr(data, "structure_plddt", None) is not None,
            load_plddt=load_plddt or getattr(data, "plddt", None) is not None,
        )
