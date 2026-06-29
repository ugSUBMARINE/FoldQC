"""
Metric computation dispatch for FoldQC.

This module is intentionally Qt- and PyMOL-independent. Callers resolve PyMOL
selections, contact shells, cutoffs, and lazy-loaded inputs before dispatching.
"""

from __future__ import annotations

import numpy as np

from . import properties as P


class MetricComputationError(Exception):
    """Base class for metric computation errors."""


class MissingMetricDataError(MetricComputationError):
    """Required prediction data are unavailable."""


class MissingReferenceError(MetricComputationError):
    """A metric needs reference token indices but none were supplied."""


class MissingCutoffError(MetricComputationError):
    """A metric needs a cutoff value but none was supplied."""


class MissingContactError(MetricComputationError):
    """A contact-filtered metric needs explicit contact token indices."""


class UnsupportedMetricError(MetricComputationError):
    """The requested metric key is not supported by per-model dispatch."""


def plddt_values_for(data) -> tuple[np.ndarray | None, str]:
    """Return preferred pLDDT values and a short source label."""
    if data is None:
        return None, ""
    structure_values = getattr(data, "structure_plddt", None)
    if structure_values is not None:
        return structure_values, "structure B-factors"
    provider_values = getattr(data, "plddt", None)
    if provider_values is not None:
        return provider_values, "provider pLDDT"
    return None, ""


def pae_domain_method(key: str) -> str:
    """Return the properties.py domain-label method for a metric key."""
    if key == "pae_domain_complete":
        return "complete_linkage"
    if key == "pae_domain_spectral":
        return "spectral"
    raise UnsupportedMetricError(f"Unknown PAE domain-label metric: {key}")


def compute_metric(
    key: str,
    data,
    token_map,
    *,
    ref_indices: list[int] | None = None,
    contact_indices: list[int] | None = None,
    cutoff: float | None = None,
) -> np.ndarray:
    """Compute a per-token metric array for one model."""
    if key == "plddt":
        values, _source_label = plddt_values_for(data)
        if values is None:
            raise MissingMetricDataError("pLDDT data are not available.")
        return P.plddt_values(values)

    if key == "pae_row_mean":
        return P.pae_row_mean(_required_attr(data, "pae", "PAE data"))

    if key == "pae_col_mean":
        return P.pae_col_mean(_required_attr(data, "pae", "PAE data"))

    if key == "pae_to_sel":
        return P.pae_to_selection(
            _required_attr(data, "pae", "PAE data"), _required_ref(ref_indices)
        )

    if key == "pae_col_to_sel":
        return P.pae_column_to_selection(
            _required_attr(data, "pae", "PAE data"), _required_ref(ref_indices)
        )

    if key == "pae_sym_sel":
        return P.pae_symmetric_to_selection(
            _required_attr(data, "pae", "PAE data"), _required_ref(ref_indices)
        )

    if key == "pae_sym_within_sel":
        return P.pae_symmetric_mean_within_selection(
            _required_attr(data, "pae", "PAE data"), _required_ref(ref_indices)
        )

    if key == "pae_contact":
        ref = _required_ref(ref_indices)
        contact = _required_contact(contact_indices)
        return P.pae_symmetric_to_selection_for_contacts(
            _required_attr(data, "pae", "PAE data"),
            ref,
            contact,
        )

    if key in ("pae_domain_complete", "pae_domain_spectral"):
        if cutoff is None:
            raise MissingCutoffError("PAE domain labels require a cutoff.")
        return P.pae_domain_labels(
            _required_attr(data, "pae", "PAE data"),
            threshold=float(cutoff),
            method=pae_domain_method(key),
        )

    if key == "pde_mean":
        return P.pde_mean(_required_attr(data, "pde", "PDE data"))

    if key == "pde_chain_mean":
        return P.pde_mean_within_chain(
            _required_attr(data, "pde", "PDE data"), token_map
        )

    if key == "pde_to_sel":
        return P.pde_to_selection(
            _required_attr(data, "pde", "PDE data"), _required_ref(ref_indices)
        )

    if key == "pde_within_sel":
        return P.pde_mean_within_selection(
            _required_attr(data, "pde", "PDE data"), _required_ref(ref_indices)
        )

    if key == "pde_contact":
        ref = _required_ref(ref_indices)
        contact = _required_contact(contact_indices)
        return P.pde_to_selection_for_contacts(
            _required_attr(data, "pde", "PDE data"),
            ref,
            contact,
        )

    if key == "contact_prob_mean":
        return P.contact_probability_mean(
            _required_attr(data, "contact_probs", "Interaction probability data")
        )

    if key == "contact_prob_to_sel":
        return P.contact_probability_to_selection(
            _required_attr(data, "contact_probs", "Interaction probability data"),
            _required_ref(ref_indices),
        )

    if key == "chain_iptm":
        return P.chain_iptm_values(
            _required_attr(data, "confidence", "Confidence JSON"), token_map
        )

    raise UnsupportedMetricError(f"Unknown property key: {key}")


def _required_attr(data, attr: str, label: str):
    value = getattr(data, attr, None)
    if value is None:
        raise MissingMetricDataError(f"{label} are not available.")
    return value


def _required_ref(ref_indices: list[int] | None) -> list[int]:
    if not ref_indices:
        raise MissingReferenceError("Reference token indices are required.")
    return list(ref_indices)


def _required_contact(contact_indices: list[int] | None) -> list[int]:
    if not contact_indices:
        raise MissingContactError("Contact token indices are required.")
    return list(contact_indices)
