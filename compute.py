"""
Metric computation dispatch for FoldQC.

This module is intentionally Qt- and viewer-independent. Callers resolve viewer
selections, contact shells, cutoffs, and lazy-loaded inputs before dispatching.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import metrics
from . import properties as P

if TYPE_CHECKING:
    from .loader import PredictionData
    from .loader_models import PlddtSource
    from .token_map import TokenMap


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
    requirement: metrics.DataRequirement
    needs_ref: bool = False  # append _required_ref(ref_indices) to args
    needs_contact: bool = False  # append _required_contact(contact_indices)
    needs_cutoff: bool = False  # cutoff is required by context resolution
    needs_token_map: bool = False  # append token_map to args
    cutoff_kwarg: str | None = None  # pass cutoff as this keyword arg
    extra_kwargs: dict | None = None  # additional static keyword args


_DISPATCH: dict[str, _Dispatch] = {
    "plddt": _Dispatch("token_plddt", P.plddt_values, "pLDDT data", "plddt"),
    "plddt_class": _Dispatch("token_plddt", P.plddt_values, "pLDDT data", "plddt"),
    # PAE — simple
    "pae_row_mean": _Dispatch("pae", P.pae_row_mean, "PAE data", "pae"),
    "pae_col_mean": _Dispatch("pae", P.pae_col_mean, "PAE data", "pae"),
    # PAE — needs reference
    "pae_to_sel": _Dispatch(
        "pae", P.pae_to_selection, "PAE data", "pae", needs_ref=True
    ),
    "pae_col_to_sel": _Dispatch(
        "pae", P.pae_column_to_selection, "PAE data", "pae", needs_ref=True
    ),
    "pae_sym_sel": _Dispatch(
        "pae", P.pae_symmetric_to_selection, "PAE data", "pae", needs_ref=True
    ),
    "pae_sym_within_sel": _Dispatch(
        "pae",
        P.pae_symmetric_mean_within_selection,
        "PAE data",
        "pae",
        needs_ref=True,
    ),
    # PAE — needs reference + contact indices
    "pae_contact": _Dispatch(
        "pae",
        P.pae_symmetric_to_selection_for_contacts,
        "PAE data",
        "pae",
        needs_ref=True,
        needs_contact=True,
        needs_cutoff=True,
    ),
    # PAE — domain labels (cutoff + method)
    "pae_domain_complete": _Dispatch(
        "pae",
        P.pae_domain_labels,
        "PAE data",
        "pae",
        cutoff_kwarg="threshold",
        extra_kwargs={"method": "complete_linkage"},
    ),
    "pae_domain_spectral": _Dispatch(
        "pae",
        P.pae_domain_labels,
        "PAE data",
        "pae",
        cutoff_kwarg="threshold",
        extra_kwargs={"method": "spectral"},
    ),
    # PDE — simple
    "pde_mean": _Dispatch("pde", P.pde_mean, "PDE data", "pde"),
    # PDE — needs token map
    "pde_chain_mean": _Dispatch(
        "pde", P.pde_mean_within_chain, "PDE data", "pde", needs_token_map=True
    ),
    # PDE — needs reference
    "pde_to_sel": _Dispatch(
        "pde", P.pde_to_selection, "PDE data", "pde", needs_ref=True
    ),
    "pde_within_sel": _Dispatch(
        "pde", P.pde_mean_within_selection, "PDE data", "pde", needs_ref=True
    ),
    # PDE — needs reference + contact indices
    "pde_contact": _Dispatch(
        "pde",
        P.pde_to_selection_for_contacts,
        "PDE data",
        "pde",
        needs_ref=True,
        needs_contact=True,
        needs_cutoff=True,
    ),
    # Interaction probability — simple
    "contact_prob_mean": _Dispatch(
        "contact_probs",
        P.contact_probability_mean,
        "Interaction probability data",
        "contact_probs",
    ),
    # Interaction probability — needs reference
    "contact_prob_to_sel": _Dispatch(
        "contact_probs",
        P.contact_probability_to_selection,
        "Interaction probability data",
        "contact_probs",
        needs_ref=True,
    ),
    # Chain-level confidence — needs token map
    "chain_iptm": _Dispatch(
        "confidence",
        P.chain_iptm_values,
        "Confidence data",
        "confidence",
        needs_token_map=True,
    ),
}


def validate_dispatch_registry() -> None:
    """Fail fast when metric metadata and executable dispatch drift apart."""
    expected = {spec.key for spec in metrics.METRICS if not spec.ensemble_level}
    actual = set(_DISPATCH)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"Metric dispatch mismatch; missing={missing!r}, extra={extra!r}."
        )
    for key, entry in _DISPATCH.items():
        spec = metrics.METRICS.require(key)
        errors = []
        if spec.requirements != {entry.requirement}:
            errors.append(
                f"data requirements {sorted(spec.requirements)!r} != "
                f"{[entry.requirement]!r}"
            )
        if entry.needs_ref != spec.needs_reference:
            errors.append("reference requirement")
        if entry.needs_contact != spec.needs_contact_shell:
            errors.append("contact-shell requirement")
        if (entry.needs_cutoff or bool(entry.cutoff_kwarg)) != spec.needs_cutoff:
            errors.append("cutoff requirement")
        if errors:
            raise ValueError(
                f"Metric {key!r} dispatch disagrees on {', '.join(errors)}."
            )


validate_dispatch_registry()


def plddt_values_for(
    data: PredictionData | None,
) -> tuple[np.ndarray | None, PlddtSource | None]:
    """Return provider-selected token pLDDT values and their provenance."""
    if data is None:
        return None, None
    values = getattr(data, "token_plddt", None)
    if values is None:
        return None, None
    return values, getattr(data, "token_plddt_source", None)


def compute_metric(
    key: str,
    data: PredictionData,
    token_map: TokenMap,
    *,
    ref_indices: list[int] | None = None,
    contact_indices: list[int] | None = None,
    cutoff: float | None = None,
) -> np.ndarray:
    """Compute a per-token metric array for one model."""
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
