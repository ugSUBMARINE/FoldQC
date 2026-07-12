"""
Metric metadata for FoldQC.

This module is intentionally independent of Qt and molecular viewers. It owns metric
labels, grouping metadata, data requirements, plot metadata, and export
units/semantics; numeric transformations remain in :mod:`properties`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MetricSpec:
    """Structured descriptor for one user-visible FoldQC metric."""

    key: str
    label: str
    group: str
    tier: str = "normal"
    needs_ref: bool = False
    needs_pae: bool = False
    needs_pde: bool = False
    needs_plddt: bool = False
    needs_structure_plddt: bool = False
    needs_any_plddt: bool = False
    needs_contact_probs: bool = False
    needs_confidence: bool = False
    ensemble_level: bool = False
    # Format string with optional named placeholders {target_text}, {ref_sel}, {cutoff}.
    # Rendered by gui_rules.metric_preview_text into the combo-box preview label.
    preview_template: str = ""

    def as_property_dict(self) -> dict[str, object]:
        """Return the legacy dict descriptor shape used by GUI code."""
        return asdict(self)


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="plddt_class",
        label="pLDDT — quality classes",
        group="pLDDT",
        needs_any_plddt=True,
        preview_template="Applies AlphaFold pLDDT confidence classes to {target_text}.",
    ),
    MetricSpec(
        key="plddt",
        label="pLDDT — continuous",
        group="pLDDT",
        needs_any_plddt=True,
        preview_template="Colors {target_text} by continuous local confidence (pLDDT).",
    ),
    MetricSpec(
        key="pae_row_mean",
        label="PAE — row mean",
        group="PAE",
        needs_pae=True,
        preview_template=(
            "Colors each token in {target_text} by how well the rest of the model"
            " is positioned when aligned on that token."
        ),
    ),
    MetricSpec(
        key="pae_col_mean",
        label="PAE — column mean",
        group="PAE",
        needs_pae=True,
        preview_template=(
            "Colors each token in {target_text} by its average positional uncertainty"
            " across all alignment frames."
        ),
    ),
    MetricSpec(
        key="pae_to_sel",
        label="PAE — row mean to selection",
        group="PAE",
        tier="advanced",
        needs_ref=True,
        needs_pae=True,
        preview_template=(
            'Colors each token in {target_text} by directional PAE from that token to "{ref_sel}".'
        ),
    ),
    MetricSpec(
        key="pae_col_to_sel",
        label="PAE — column mean to selection",
        group="PAE",
        tier="advanced",
        needs_ref=True,
        needs_pae=True,
        preview_template=(
            'Colors each token in {target_text} by directional PAE from "{ref_sel}" to that token.'
        ),
    ),
    MetricSpec(
        key="pae_sym_sel",
        label="PAE — symmetric mean to selection",
        group="PAE",
        tier="advanced",
        needs_ref=True,
        needs_pae=True,
        preview_template=(
            "Colors {target_text} by bidirectional mean PAE between each token and"
            ' "{ref_sel}".'
        ),
    ),
    MetricSpec(
        key="pae_sym_within_sel",
        label="PAE — symmetric mean within selection",
        group="PAE",
        tier="advanced",
        needs_ref=True,
        needs_pae=True,
        preview_template=(
            'Colors only tokens in "{ref_sel}" within {target_text} by their'
            " internal symmetric PAE."
        ),
    ),
    MetricSpec(
        key="pae_contact",
        label="PAE — contact-filtered to selection",
        group="PAE",
        tier="advanced",
        needs_ref=True,
        needs_pae=True,
        preview_template=(
            "Colors polymer binding-site residues in {target_text} within {cutoff} of"
            ' "{ref_sel}" by mean PAE to the reference.'
        ),
    ),
    MetricSpec(
        key="pae_domain_complete",
        label="PAE — domain labels (complete linkage)",
        group="PAE",
        tier="experimental",
        needs_pae=True,
        preview_template=(
            "Colors {target_text} with categorical rigid-domain labels by grouping"
            " tokens whose pairwise symmetric PAE stays within the {cutoff} threshold."
        ),
    ),
    MetricSpec(
        key="pae_domain_spectral",
        label="PAE — domain labels (spectral clustering)",
        group="PAE",
        tier="experimental",
        needs_pae=True,
        preview_template=(
            "Colors {target_text} with categorical heuristic PAE domain labels by"
            " spectral clustering of a symmetric PAE affinity graph using the"
            " {cutoff} threshold as a scale."
        ),
    ),
    MetricSpec(
        key="pde_mean",
        label="PDE — mean",
        group="PDE",
        needs_pde=True,
        preview_template=(
            "Colors each token in {target_text} by its average predicted distance"
            " error to all other tokens."
        ),
    ),
    MetricSpec(
        key="pde_chain_mean",
        label="PDE — within-chain mean",
        group="PDE",
        needs_pde=True,
        preview_template=(
            "Colors each token in {target_text} by predicted distance error within"
            " its own chain."
        ),
    ),
    MetricSpec(
        key="pde_to_sel",
        label="PDE — mean to selection",
        group="PDE",
        tier="advanced",
        needs_ref=True,
        needs_pde=True,
        preview_template=(
            "Colors each token in {target_text} by predicted distance error"
            ' to "{ref_sel}".'
        ),
    ),
    MetricSpec(
        key="pde_within_sel",
        label="PDE — within-selection mean",
        group="PDE",
        tier="advanced",
        needs_ref=True,
        needs_pde=True,
        preview_template=(
            'Colors only tokens in "{ref_sel}" within {target_text} by their'
            " internal predicted distance error."
        ),
    ),
    MetricSpec(
        key="pde_contact",
        label="PDE — contact-filtered to selection",
        group="PDE",
        tier="advanced",
        needs_ref=True,
        needs_pde=True,
        preview_template=(
            "Colors polymer binding-site residues in {target_text} within {cutoff} of"
            ' "{ref_sel}" by mean PDE to the reference.'
        ),
    ),
    MetricSpec(
        key="contact_prob_mean",
        label="Interaction probability — mean",
        group="Interaction probability",
        needs_contact_probs=True,
        preview_template=(
            "Colors each token in {target_text} by its average predicted interaction"
            " probability across the model."
        ),
    ),
    MetricSpec(
        key="contact_prob_to_sel",
        label="Interaction probability — mean to selection",
        group="Interaction probability",
        tier="advanced",
        needs_ref=True,
        needs_contact_probs=True,
        preview_template=(
            "Colors each token in {target_text} by predicted interaction probability"
            ' with "{ref_sel}".'
        ),
    ),
    MetricSpec(
        key="ensemble_rmsd",
        label="Ensemble RMSD, aligned",
        group="Ensemble",
        ensemble_level=True,
        preview_template=(
            "Colors {target_text} by per-token coordinate variation in the loaded"
            " ensemble after alignment."
        ),
    ),
    MetricSpec(
        key="ensemble_plddt_mean",
        label="Ensemble pLDDT mean",
        group="Ensemble",
        needs_any_plddt=True,
        ensemble_level=True,
        preview_template=(
            "Colors {target_text} by the mean pLDDT at each token across models in"
            " the loaded ensemble."
        ),
    ),
    MetricSpec(
        key="ensemble_plddt_std",
        label="Ensemble pLDDT std",
        group="Ensemble",
        needs_any_plddt=True,
        ensemble_level=True,
        preview_template=(
            "Colors {target_text} by how much pLDDT varies at each token across"
            " models in the loaded ensemble."
        ),
    ),
    MetricSpec(
        key="chain_iptm",
        label="Chain ipTM",
        group="Chain/interface",
        needs_confidence=True,
        preview_template=(
            "Colors chains in {target_text} by chain-level ipTM; use Plot >"
            " Matrix for pairwise chain ipTM."
        ),
    ),
)

METRIC_BY_KEY: dict[str, MetricSpec] = {spec.key: spec for spec in METRICS}

# Dict-shaped descriptors are kept in this module for the current GUI migration
# step.  gui.py intentionally does not re-export them.
PROPERTIES: list[dict[str, object]] = [spec.as_property_dict() for spec in METRICS]
PROPERTY_BY_KEY: dict[str, dict[str, object]] = {
    str(prop["key"]): prop for prop in PROPERTIES
}

PLOT_TYPES: list[tuple[str, str]] = [
    ("Line", "line"),
    ("Distribution", "distribution"),
    ("Matrix", "matrix"),
    ("PAE summary", "pae_summary"),
    ("PDE summary", "pde_summary"),
    ("Binding-site fingerprint", "binding_site_fingerprint"),
    ("Ensemble site summary", "ensemble_site_summary"),
]

PLDDT_CLASS_STATS: list[tuple[str, float | None, float | None]] = [
    ("Very high (>=90)", 90.0, None),
    ("High (70-90)", 70.0, 90.0),
    ("Low (50-70)", 50.0, 70.0),
    ("Very low (<50)", None, 50.0),
]
PLDDT_CLASS_PLOT_LABELS: dict[str, str] = {
    "Very high (>=90)": "very high",
    "High (70-90)": "high",
    "Low (50-70)": "low",
    "Very low (<50)": "very low",
}

METRIC_UNITS_AND_SEMANTICS: dict[str, tuple[str, str]] = {
    "plddt_class": ("plddt", "higher_is_better"),
    "plddt": ("plddt", "higher_is_better"),
    "pae_row_mean": ("angstrom", "lower_is_better"),
    "pae_col_mean": ("angstrom", "lower_is_better"),
    "pae_to_sel": ("angstrom", "lower_is_better"),
    "pae_col_to_sel": ("angstrom", "lower_is_better"),
    "pae_sym_sel": ("angstrom", "lower_is_better"),
    "pae_sym_within_sel": ("angstrom", "lower_is_better"),
    "pae_contact": ("angstrom", "lower_is_better"),
    "pae_domain_complete": ("label", "categorical_label"),
    "pae_domain_spectral": ("label", "categorical_label"),
    "pde_mean": ("angstrom", "lower_is_better"),
    "pde_chain_mean": ("angstrom", "lower_is_better"),
    "pde_to_sel": ("angstrom", "lower_is_better"),
    "pde_within_sel": ("angstrom", "lower_is_better"),
    "pde_contact": ("angstrom", "lower_is_better"),
    "contact_prob_mean": ("probability", "higher_is_better"),
    "contact_prob_to_sel": ("probability", "higher_is_better"),
    "chain_iptm": ("iptm", "higher_is_better"),
    "ensemble_rmsd": ("angstrom", "lower_is_better"),
    "ensemble_plddt_mean": ("plddt", "higher_is_better"),
    "ensemble_plddt_std": ("plddt", "lower_is_better"),
}


def _get_field(spec_or_dict: MetricSpec | Mapping[str, object], key: str, default=None):
    """Return one metric descriptor field from either supported shape."""
    if isinstance(spec_or_dict, MetricSpec):
        return getattr(spec_or_dict, key, default)
    return spec_or_dict.get(key, default)


def is_domain_label_metric(key: str | None) -> bool:
    """Return True for categorical PAE domain-label metrics."""
    return bool(key) and str(key).startswith("pae_domain")


def metric_label(key: str) -> str:
    """Return the user-facing label for a metric key."""
    spec = METRIC_BY_KEY.get(key)
    return spec.label if spec is not None else key


def property_combo_label(spec_or_dict: MetricSpec | Mapping[str, object]) -> str:
    """Return the grouped combo display label for a metric descriptor."""
    label = str(_get_field(spec_or_dict, "label", _get_field(spec_or_dict, "key", "")))
    tier = _get_field(spec_or_dict, "tier", "normal")
    if tier == "advanced":
        label = f"{label} [Advanced]"
    elif tier == "experimental":
        label = f"{label} [Experimental]"
    return f"  {label}"


def metric_load_flags(
    spec_or_dict: MetricSpec | Mapping[str, object],
) -> dict[str, bool]:
    """Return loader flags needed to compute one metric."""
    needs_any_plddt = bool(_get_field(spec_or_dict, "needs_any_plddt", False))
    return {
        "load_pae": bool(_get_field(spec_or_dict, "needs_pae", False)),
        "load_pde": bool(_get_field(spec_or_dict, "needs_pde", False)),
        "load_contact_probs": bool(
            _get_field(spec_or_dict, "needs_contact_probs", False)
        ),
        "load_structure_plddt": bool(
            _get_field(spec_or_dict, "needs_structure_plddt", False) or needs_any_plddt
        ),
        "load_plddt": bool(
            _get_field(spec_or_dict, "needs_plddt", False) or needs_any_plddt
        ),
    }


def line_compute_key(key: str) -> str:
    """Map paint-only pLDDT class entries to scalar pLDDT data."""
    return "plddt" if key == "plddt_class" else key


_CONTEXT_TO_SELECTION_METRICS = {
    "pae_to_sel",
    "pae_col_to_sel",
    "pae_sym_sel",
    "pae_contact",
    "pde_to_sel",
    "pde_contact",
    "contact_prob_to_sel",
}

CONTACT_FILTERED_METRICS = {"pae_contact", "pde_contact"}


def plot_uses_reference_scope(metric_key: str, plot_type: str) -> bool:
    """Return whether a plot should restrict displayed tokens to Reference."""
    if plot_type == "distribution" and metric_key == "chain_iptm":
        return False
    if plot_type in {"line", "distribution"}:
        return metric_key not in _CONTEXT_TO_SELECTION_METRICS
    if plot_type == "matrix":
        return metric_key != "chain_iptm"
    return False


def line_ylabel(key: str) -> str:
    """Return a compact y-axis label for one metric key."""
    if key.startswith("pae_domain"):
        return "Domain label"
    if key.startswith("pae"):
        return "PAE (Å)"
    if key.startswith("pde") or key == "ensemble_rmsd":
        return "Distance / error (Å)"
    if key.startswith("contact_prob"):
        return "Interaction probability"
    if key.startswith("plddt") or key.startswith("ensemble_plddt"):
        return "pLDDT"
    if key == "chain_iptm":
        return "ipTM"
    return "Value"


def matrix_source_for_metric(key: str) -> tuple[str, str, str] | None:
    """Return matrix attribute, title, and colorbar label for a metric key."""
    if key.startswith("pae"):
        return "pae", "Predicted Aligned Error (Å)", "PAE (Å)"
    if key.startswith("pde"):
        return "pde", "Predicted Distance Error (Å)", "PDE (Å)"
    if key.startswith("contact_prob"):
        return "contact_probs", "Interaction probability", "Probability"
    if key == "chain_iptm":
        return "chain_iptm", "Pairwise chain ipTM", "ipTM"
    return None


def ensemble_aggregate_kind(key: str) -> str:
    """Return export aggregate kind for ensemble-level metrics."""
    return {
        "ensemble_rmsd": "ensemble_rmsd",
        "ensemble_plddt_mean": "ensemble_mean",
        "ensemble_plddt_std": "ensemble_std",
    }.get(key, "ensemble_mean")


def metric_units_and_semantics(key: str) -> tuple[str, str]:
    """Return machine-readable value units and interpretation semantics."""
    return METRIC_UNITS_AND_SEMANTICS.get(key, ("", ""))
