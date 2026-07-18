"""
Pure GUI decision rules for FoldQC.

This module contains UI state decisions that do not need Qt or a molecular viewer.
The dialog gathers current widget/session state, calls these helpers, and then
applies the returned text/enabled state to widgets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import metrics


@dataclass(frozen=True)
class PlotActionState:
    """Enabled state and user-facing reason for one plot action."""

    enabled: bool
    reason: str = ""


@dataclass(frozen=True)
class FieldContext:
    """User-facing state for contextual Reference and cutoff controls."""

    ref_label: str
    ref_enabled: bool
    ref_tooltip: str
    cutoff_label: str
    cutoff_enabled: bool
    cutoff_tooltip: str


def plot_action_state(
    plot_type: str,
    metric_key: str | None,
    target_kind: str,
    has_reference: bool,
    has_ensemble: bool,
    *,
    has_fingerprint_data: bool = True,
    has_pae_data: bool = False,
    has_pde_data: bool = False,
    has_multiple_chains: bool = False,
) -> PlotActionState:
    """Return central availability state for one plot menu action."""
    plot_spec = metrics.PLOTS.find(plot_type)
    if plot_spec is None:
        return PlotActionState(False, f"Unknown plot type: {plot_type}")
    if not metric_key and plot_spec.requires_metric:
        return PlotActionState(False, "Select a Color by metric before plotting.")
    spec = metrics.METRICS.find(metric_key)
    has_target = target_kind != "none"
    if plot_type != "ensemble_site_summary" and not has_target:
        return PlotActionState(False, "Select a viewer target before plotting.")

    if plot_type == "line":
        if spec is not None and spec.is_domain_label:
            return PlotActionState(
                False, "PAE domain labels are categorical; use Distribution instead."
            )
        if spec is not None and spec.needs_contact_shell:
            metric_label = "PAE/PDE" if metric_key == "pae_contact" else "PDE"
            return PlotActionState(
                False,
                f"{metric_label} contact-filtered values are sparse; use "
                "Distribution or Matrix.",
            )
        if spec is not None and spec.needs_reference and not has_reference:
            return PlotActionState(
                False, "This line plot requires a reference selection."
            )
        return PlotActionState(True)

    if plot_type == "distribution":
        if metric_key == "chain_iptm":
            return PlotActionState(
                False, "Distribution plots are not available for chain ipTM."
            )
        if (
            spec is not None
            and spec.is_domain_label
            and target_kind == "ensemble_group"
        ):
            return PlotActionState(
                False,
                "PAE domain labels are member-local; choose a single model or member.",
            )
        if spec is not None and spec.needs_reference and not has_reference:
            return PlotActionState(
                False, "This distribution requires a reference selection."
            )
        return PlotActionState(True)

    if plot_type == "matrix":
        if spec is None or spec.matrix is None:
            return PlotActionState(
                False,
                "Matrix plots are only available for PAE, PDE, interaction "
                "probability, and chain ipTM.",
            )
        return PlotActionState(True)

    if plot_type == "pae_summary":
        if not has_pae_data:
            return PlotActionState(False, "PAE summary requires PAE data.")
        if not has_multiple_chains:
            return PlotActionState(
                False, "PAE summary requires a target with more than one chain."
            )
        return PlotActionState(True)

    if plot_type == "pde_summary":
        if not has_pde_data:
            return PlotActionState(False, "PDE summary requires PDE data.")
        if not has_multiple_chains:
            return PlotActionState(
                False, "PDE summary requires a target with more than one chain."
            )
        return PlotActionState(True)

    if plot_type == "binding_site_fingerprint":
        if not has_reference:
            return PlotActionState(
                False, "Binding-site fingerprint requires a reference selection."
            )
        if not has_fingerprint_data:
            return PlotActionState(
                False,
                "Binding-site fingerprint requires pLDDT, PAE, PDE, or "
                "interaction probability data.",
            )
        return PlotActionState(True)

    if plot_type == "ensemble_site_summary":
        if spec is not None and spec.is_domain_label:
            return PlotActionState(
                False, "Ensemble site summary is not available for PAE domain labels."
            )
        if not has_reference:
            return PlotActionState(
                False, "Ensemble site summary requires a reference selection."
            )
        if not has_ensemble:
            return PlotActionState(
                False, "Load an ensemble before showing the ensemble site summary."
            )
        return PlotActionState(True)

    return PlotActionState(False, f"Unknown plot type: {plot_type}")


def field_context(
    metric_key: str | None,
    target_kind: str,
    has_ensemble: bool,
    has_fingerprint_data: bool,
) -> FieldContext:
    """Return contextual labels and enabled states for Reference/cutoff."""
    spec = metrics.METRICS.find(metric_key)
    has_target = target_kind != "none"
    supports_site_plot = has_target and has_fingerprint_data
    supports_ensemble_site = (
        has_ensemble and has_target and not (spec is not None and spec.is_domain_label)
    )

    needs_metric_ref = bool(spec is not None and spec.needs_reference)
    if needs_metric_ref:
        ref_tooltip = (
            "Viewer selection used by this to-selection metric, mapped back "
            "to FoldQC tokens."
        )
        ref_enabled = True
    elif supports_site_plot or supports_ensemble_site:
        ref_tooltip = (
            "Optional viewer selection used by binding-site fingerprint and "
            "ensemble site summary plots."
        )
        ref_enabled = True
    else:
        ref_tooltip = (
            "Reference is not used by the selected metric or currently "
            "available plot actions."
        )
        ref_enabled = False
    ref_label = "Reference selection:"

    if spec is not None and spec.is_domain_label:
        cutoff_label = "PAE threshold (Å):"
        cutoff_tooltip = "PAE threshold used to assign categorical domain labels."
        cutoff_enabled = True
    elif spec is not None and spec.needs_contact_shell:
        cutoff_label = "Cutoff (Å):"
        cutoff_tooltip = (
            "Distance cutoff for contact-filtered values against the reference "
            "selection."
        )
        cutoff_enabled = True
    else:
        cutoff_label = "Cutoff (Å):"
        cutoff_tooltip = (
            "Positive distance cutoff or PAE threshold used by metrics and "
            "site-focused plots when applicable."
        )
        cutoff_enabled = True

    return FieldContext(
        ref_label=ref_label,
        ref_enabled=ref_enabled,
        ref_tooltip=ref_tooltip,
        cutoff_label=cutoff_label,
        cutoff_enabled=cutoff_enabled,
        cutoff_tooltip=cutoff_tooltip,
    )


def preview_cutoff_text(cutoff_text: str | None) -> str:
    """Return a compact cutoff value for preview text."""
    text = "5.0" if cutoff_text is None else cutoff_text.strip()
    if not text:
        text = "5.0"
    try:
        value = float(text)
    except ValueError:
        return "the cutoff"
    if not math.isfinite(value) or value <= 0.0:
        return "the cutoff"
    return f"{value:g} Å"


def _metric_preview_base(
    metric_key: str | None,
    target_kind: str,
    reference_selection: str,
    cutoff_text: str | None,
    has_ensemble: bool,
    *,
    generalize_reference: bool,
) -> str:
    if not metric_key:
        return "Select a Color by metric."

    spec = metrics.METRICS.find(metric_key)
    ref_sel = reference_selection.strip()
    target_text = (
        "all members of the ensemble"
        if target_kind == "ensemble_group"
        else "the target"
    )

    if spec is not None and spec.ensemble_level and not has_ensemble:
        return 'Load an ensemble with "Load Ensemble" to use this metric.'

    if spec is not None and spec.needs_reference and not ref_sel:
        if spec.needs_contact_shell:
            return (
                "Requires a reference selection and contact cutoff, such as a "
                "chain, ligand, or residue set."
            )
        return (
            "Requires a reference selection, such as a chain, ligand, or residue set."
        )

    template = spec.preview_template if spec is not None else ""
    if template:
        preview = template.format(
            target_text=target_text,
            ref_sel=(
                "the reference selection"
                if generalize_reference and ref_sel
                else ref_sel
            ),
            cutoff=preview_cutoff_text(cutoff_text),
        )
    else:
        preview = f"Colors {target_text} by {metrics.metric_label(metric_key)}."

    return preview


def metric_preview_summary(
    metric_key: str | None,
    target_kind: str,
    reference_selection: str,
    cutoff_text: str | None,
    has_ensemble: bool,
) -> str:
    """Return a short meaning-focused explanation for the stable preview row."""
    return _metric_preview_base(
        metric_key,
        target_kind,
        reference_selection,
        cutoff_text,
        has_ensemble,
        generalize_reference=True,
    )


def metric_preview_text(
    metric_key: str | None,
    target_kind: str,
    reference_selection: str,
    cutoff_text: str | None,
    has_ensemble: bool,
) -> str:
    """Return the complete practical explanation for preview details."""
    preview = _metric_preview_base(
        metric_key,
        target_kind,
        reference_selection,
        cutoff_text,
        has_ensemble,
        generalize_reference=False,
    )
    if not metric_key:
        return preview
    preview = _append_ensemble_plot_guidance(preview, metric_key, target_kind)
    ref_sel = reference_selection.strip()
    return _append_reference_plot_guidance(preview, metric_key, ref_sel)


def _append_ensemble_plot_guidance(
    preview: str,
    metric_key: str,
    target_kind: str,
) -> str:
    """Describe actual ensemble aggregation for the selected metric's plots."""
    if target_kind != "ensemble_group":
        return preview

    if metric_key == "ensemble_rmsd":
        return f"{preview} Plots show the shared ensemble RMSD values."
    if metric_key == "ensemble_plddt_mean":
        return (
            f"{preview} Line plots show ensemble mean pLDDT with standard "
            "deviation; distribution plots use the mean values."
        )
    if metric_key == "ensemble_plddt_std":
        return f"{preview} Plots show the ensemble pLDDT standard-deviation values."

    plot_details = []
    spec = metrics.METRICS.find(metric_key)
    if spec is not None and not spec.is_domain_label:
        plot_details.append("line plots show the member mean and standard deviation")
    if metric_key != "chain_iptm" and spec is not None and not spec.is_domain_label:
        plot_details.append("distribution plots use the member mean")
    if spec is not None and spec.matrix is not None:
        if metric_key == "chain_iptm":
            plot_details.append(
                "matrix plots show the member mean with standard-deviation annotations"
            )
        else:
            plot_details.append("matrix plots show the member mean")

    if not plot_details:
        return preview
    return f"{preview} For ensemble plots, {'; '.join(plot_details)}."


def _append_reference_plot_guidance(
    preview: str,
    metric_key: str,
    reference_selection: str,
) -> str:
    """Append plot-scope guidance when Reference affects token-indexed plots."""
    if not reference_selection:
        return preview

    restricted_plots = []
    spec = metrics.METRICS.find(metric_key)
    if (
        spec is not None
        and not spec.is_domain_label
        and "line" in spec.reference_scoped_plots
    ):
        restricted_plots.append("line plot x-ranges")
    if (
        spec is not None
        and spec.matrix is not None
        and "matrix" in spec.reference_scoped_plots
    ):
        if metric_key in {"pae_row_mean", "pae_col_to_sel"}:
            restricted_plots.append("matrix plot rows")
        elif metric_key == "pae_sym_within_sel":
            restricted_plots.append("matrix plot rows and columns")
        else:
            restricted_plots.append("matrix plot columns")
    if spec is not None and "distribution" in spec.reference_scoped_plots:
        restricted_plots.append("distribution plots")
    if not restricted_plots:
        return preview

    if len(restricted_plots) == 1:
        plot_text = restricted_plots[0]
    elif len(restricted_plots) == 2:
        plot_text = " and ".join(restricted_plots)
    else:
        plot_text = ", ".join(restricted_plots[:-1])
        plot_text += f", and {restricted_plots[-1]}"
    return (
        f"{preview} {plot_text.capitalize()} are restricted to tokens selected by "
        f'"{reference_selection}".'
    )
