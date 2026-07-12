"""
Metric computation dispatch for FoldQC.

This module is intentionally Qt- and PyMOL-independent. Callers resolve PyMOL
selections, contact shells, cutoffs, and lazy-loaded inputs before dispatching.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import properties as P

if TYPE_CHECKING:
    from .loader import PredictionData
    from .token_map import TokenInfo


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


@dataclass(frozen=True)
class _Dispatch:
    """Descriptor for one dispatchable metric in _DISPATCH."""

    data_attr: str  # PredictionData attribute to extract
    func: Callable  # properties.py function to call
    data_label: str  # human label for MissingMetricDataError
    needs_ref: bool = False  # append _required_ref(ref_indices) to args
    needs_contact: bool = False  # append _required_contact(contact_indices)
    needs_token_map: bool = False  # append token_map to args
    cutoff_kwarg: str | None = None  # pass cutoff as this keyword arg
    extra_kwargs: dict | None = None  # additional static keyword args


_DISPATCH: dict[str, _Dispatch] = {
    # PAE — simple
    "pae_row_mean": _Dispatch("pae", P.pae_row_mean, "PAE data"),
    "pae_col_mean": _Dispatch("pae", P.pae_col_mean, "PAE data"),
    # PAE — needs reference
    "pae_to_sel": _Dispatch("pae", P.pae_to_selection, "PAE data", needs_ref=True),
    "pae_col_to_sel": _Dispatch(
        "pae", P.pae_column_to_selection, "PAE data", needs_ref=True
    ),
    "pae_sym_sel": _Dispatch(
        "pae", P.pae_symmetric_to_selection, "PAE data", needs_ref=True
    ),
    "pae_sym_within_sel": _Dispatch(
        "pae", P.pae_symmetric_mean_within_selection, "PAE data", needs_ref=True
    ),
    # PAE — needs reference + contact indices
    "pae_contact": _Dispatch(
        "pae",
        P.pae_symmetric_to_selection_for_contacts,
        "PAE data",
        needs_ref=True,
        needs_contact=True,
    ),
    # PAE — domain labels (cutoff + method)
    "pae_domain_complete": _Dispatch(
        "pae",
        P.pae_domain_labels,
        "PAE data",
        cutoff_kwarg="threshold",
        extra_kwargs={"method": "complete_linkage"},
    ),
    "pae_domain_spectral": _Dispatch(
        "pae",
        P.pae_domain_labels,
        "PAE data",
        cutoff_kwarg="threshold",
        extra_kwargs={"method": "spectral"},
    ),
    # PDE — simple
    "pde_mean": _Dispatch("pde", P.pde_mean, "PDE data"),
    # PDE — needs token map
    "pde_chain_mean": _Dispatch(
        "pde", P.pde_mean_within_chain, "PDE data", needs_token_map=True
    ),
    # PDE — needs reference
    "pde_to_sel": _Dispatch("pde", P.pde_to_selection, "PDE data", needs_ref=True),
    "pde_within_sel": _Dispatch(
        "pde", P.pde_mean_within_selection, "PDE data", needs_ref=True
    ),
    # PDE — needs reference + contact indices
    "pde_contact": _Dispatch(
        "pde",
        P.pde_to_selection_for_contacts,
        "PDE data",
        needs_ref=True,
        needs_contact=True,
    ),
    # Interaction probability — simple
    "contact_prob_mean": _Dispatch(
        "contact_probs", P.contact_probability_mean, "Interaction probability data"
    ),
    # Interaction probability — needs reference
    "contact_prob_to_sel": _Dispatch(
        "contact_probs",
        P.contact_probability_to_selection,
        "Interaction probability data",
        needs_ref=True,
    ),
    # Chain-level confidence — needs token map
    "chain_iptm": _Dispatch(
        "confidence", P.chain_iptm_values, "Confidence JSON", needs_token_map=True
    ),
}


def plddt_values_for(data: PredictionData | None) -> tuple[np.ndarray | None, str]:
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
    data: PredictionData,
    token_map: list[TokenInfo],
    *,
    ref_indices: list[int] | None = None,
    contact_indices: list[int] | None = None,
    cutoff: float | None = None,
) -> np.ndarray:
    """Compute a per-token metric array for one model."""
    # plddt_class uses the same continuous array as plddt; the GUI chooses
    # the categorical painter based on the key, not on compute_metric's output.
    if key in ("plddt", "plddt_class"):
        values, _source_label = plddt_values_for(data)
        if values is None:
            raise MissingMetricDataError("pLDDT data are not available.")
        return P.plddt_values(values)

    entry = _DISPATCH.get(key)
    if entry is None:
        raise UnsupportedMetricError(f"Unknown property key: {key}")

    data_value = _required_attr(data, entry.data_attr, entry.data_label)
    args: list = [data_value]
    kwargs: dict = dict(entry.extra_kwargs or {})

    if entry.needs_ref:
        args.append(_required_ref(ref_indices))
    if entry.needs_contact:
        args.append(_required_contact(contact_indices))
    if entry.needs_token_map:
        args.append(token_map)
    if entry.cutoff_kwarg:
        if cutoff is None:
            raise MissingCutoffError("PAE domain labels require a cutoff.")
        kwargs[entry.cutoff_kwarg] = float(cutoff)

    return entry.func(*args, **kwargs)


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
