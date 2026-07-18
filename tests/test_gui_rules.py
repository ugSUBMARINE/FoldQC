from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import gui_rules  # noqa: E402


def _enabled(
    plot_type: str,
    metric_key: str | None,
    *,
    target_kind: str = "single",
    has_reference: bool = False,
    has_ensemble: bool = False,
    has_fingerprint_data: bool = True,
    has_pae_data: bool = False,
    has_pde_data: bool = False,
    has_multiple_chains: bool = False,
) -> bool:
    return gui_rules.plot_action_state(
        plot_type,
        metric_key,
        target_kind,
        has_reference,
        has_ensemble,
        has_fingerprint_data=has_fingerprint_data,
        has_pae_data=has_pae_data,
        has_pde_data=has_pde_data,
        has_multiple_chains=has_multiple_chains,
    ).enabled


def test_plot_action_state_for_representative_metrics() -> None:
    assert _enabled("line", "plddt")
    assert _enabled("distribution", "plddt")
    assert not _enabled("matrix", "plddt")

    assert _enabled("matrix", "pae_row_mean")
    assert not _enabled("line", "pae_domain_spectral")
    assert _enabled("distribution", "pae_domain_spectral")
    assert not _enabled(
        "distribution", "pae_domain_spectral", target_kind="ensemble_group"
    )

    assert not _enabled("line", "pde_contact")
    assert not _enabled("line", "pde_contact", has_reference=True)
    assert _enabled("distribution", "pde_contact", has_reference=True)
    assert _enabled("matrix", "pde_contact")
    assert not _enabled("line", "pae_contact", has_reference=True)
    assert _enabled("distribution", "pae_contact", has_reference=True)
    assert _enabled("matrix", "pae_contact")

    assert not _enabled("line", "contact_prob_to_sel")
    assert _enabled("line", "contact_prob_to_sel", has_reference=True)
    assert _enabled("matrix", "contact_prob_to_sel")

    assert _enabled("matrix", "chain_iptm")
    assert not _enabled("distribution", "chain_iptm")
    assert _enabled(
        "line",
        "ensemble_rmsd",
        target_kind="ensemble_group",
        has_ensemble=True,
    )
    assert not _enabled(
        "matrix",
        "ensemble_rmsd",
        target_kind="ensemble_group",
        has_ensemble=True,
    )
    assert _enabled(
        "pae_summary",
        None,
        has_pae_data=True,
        has_multiple_chains=True,
    )
    assert _enabled(
        "pde_summary",
        "plddt",
        has_pde_data=True,
        has_multiple_chains=True,
    )


@pytest.mark.parametrize(
    ("plot_type", "metric_key", "kwargs", "reason"),
    [
        (
            "line",
            None,
            {},
            "Select a Color by metric before plotting.",
        ),
        (
            "line",
            "plddt",
            {"target_kind": "none"},
            "Select a viewer target before plotting.",
        ),
        (
            "line",
            "pae_domain_complete",
            {},
            "PAE domain labels are categorical; use Distribution instead.",
        ),
        (
            "binding_site_fingerprint",
            "plddt",
            {"has_reference": True, "has_fingerprint_data": False},
            "Binding-site fingerprint requires pLDDT, PAE, PDE, or "
            "interaction probability data.",
        ),
        (
            "ensemble_site_summary",
            "plddt",
            {"has_reference": True},
            "Load an ensemble before showing the ensemble site summary.",
        ),
        (
            "pae_summary",
            None,
            {"has_multiple_chains": True},
            "PAE summary requires PAE data.",
        ),
        (
            "pde_summary",
            None,
            {"has_pde_data": True},
            "PDE summary requires a target with more than one chain.",
        ),
        (
            "not_a_plot",
            "plddt",
            {},
            "Unknown plot type: not_a_plot",
        ),
        (
            "line",
            "pae_contact",
            {"has_reference": True},
            "PAE/PDE contact-filtered values are sparse; use Distribution or Matrix.",
        ),
        (
            "line",
            "pde_contact",
            {"has_reference": True},
            "PDE contact-filtered values are sparse; use Distribution or Matrix.",
        ),
    ],
)
def test_plot_action_state_reasons(
    plot_type: str,
    metric_key: str | None,
    kwargs: dict[str, object],
    reason: str,
) -> None:
    state_kwargs = {
        "target_kind": "single",
        "has_reference": False,
        "has_ensemble": False,
        "has_fingerprint_data": True,
        "has_pae_data": False,
        "has_pde_data": False,
        "has_multiple_chains": False,
    }
    state_kwargs.update(kwargs)
    state = gui_rules.plot_action_state(
        plot_type,
        metric_key,
        **state_kwargs,
    )

    assert state.enabled is False
    assert state.reason == reason


@pytest.mark.parametrize(
    (
        "metric_key",
        "target_kind",
        "has_ensemble",
        "has_fingerprint_data",
        "ref_label",
        "ref_enabled",
        "cutoff_label",
        "cutoff_enabled",
    ),
    [
        (
            "chain_iptm",
            "single",
            False,
            False,
            "Reference selection:",
            False,
            "Cutoff (Å):",
            True,
        ),
        (
            "pae_to_sel",
            "single",
            False,
            False,
            "Reference selection:",
            True,
            "Cutoff (Å):",
            True,
        ),
        (
            "plddt",
            "single",
            False,
            True,
            "Reference selection:",
            True,
            "Cutoff (Å):",
            True,
        ),
        (
            "pae_contact",
            "single",
            False,
            False,
            "Reference selection:",
            True,
            "Cutoff (Å):",
            True,
        ),
        (
            "pde_contact",
            "single",
            False,
            False,
            "Reference selection:",
            True,
            "Cutoff (Å):",
            True,
        ),
        (
            "pae_domain_complete",
            "single",
            False,
            False,
            "Reference selection:",
            False,
            "PAE threshold (Å):",
            True,
        ),
        (
            "plddt",
            "ensemble_group",
            True,
            False,
            "Reference selection:",
            True,
            "Cutoff (Å):",
            True,
        ),
    ],
)
def test_field_context(
    metric_key: str,
    target_kind: str,
    has_ensemble: bool,
    has_fingerprint_data: bool,
    ref_label: str,
    ref_enabled: bool,
    cutoff_label: str,
    cutoff_enabled: bool,
) -> None:
    context = gui_rules.field_context(
        metric_key,
        target_kind,
        has_ensemble,
        has_fingerprint_data,
    )

    assert context.ref_label == ref_label
    assert context.ref_enabled is ref_enabled
    assert context.cutoff_label == cutoff_label
    assert context.cutoff_enabled is cutoff_enabled


@pytest.mark.parametrize(
    ("cutoff_text", "expected"),
    [
        ("", "5 Å"),
        ("5.0", "5 Å"),
        ("7.5", "7.5 Å"),
        ("abc", "the cutoff"),
        ("0", "the cutoff"),
        ("-1", "the cutoff"),
        ("nan", "the cutoff"),
    ],
)
def test_preview_cutoff_text(cutoff_text: str, expected: str) -> None:
    assert gui_rules.preview_cutoff_text(cutoff_text) == expected


@pytest.mark.parametrize(
    ("metric_key", "ref_sel", "cutoff", "has_ensemble", "expected"),
    [
        (
            "pae_row_mean",
            "",
            "5.0",
            False,
            "how well the rest of the model",
        ),
        ("pae_to_sel", "", "5.0", False, "such as a chain"),
        (
            "pae_col_to_sel",
            "chain B",
            "5.0",
            False,
            'directional PAE from "chain B" to that token',
        ),
        (
            "pae_sym_within_sel",
            "chain B",
            "5.0",
            False,
            'only tokens in "chain B"',
        ),
        (
            "pae_contact",
            "resname LIG",
            "5.0",
            False,
            'within 5 Å of "resname LIG"',
        ),
        (
            "pde_contact",
            "resname LIG",
            "5.0",
            False,
            'within 5 Å of "resname LIG"',
        ),
        (
            "pae_domain_complete",
            "",
            "6.0",
            False,
            "categorical rigid-domain labels",
        ),
        (
            "pae_domain_spectral",
            "",
            "7.5",
            False,
            "spectral clustering",
        ),
        ("ensemble_rmsd", "", "5.0", False, "Load Ensemble"),
        (
            "pae_to_sel",
            "chain B",
            "5.0",
            False,
            'directional PAE from that token to "chain B"',
        ),
        ("chain_iptm", "", "5.0", False, "chain-level ipTM"),
        ("plddt", "", "5.0", False, "continuous local confidence"),
        ("ensemble_plddt_std", "", "5.0", True, "pLDDT varies"),
    ],
)
def test_metric_preview_text(
    metric_key: str,
    ref_sel: str,
    cutoff: str,
    has_ensemble: bool,
    expected: str,
) -> None:
    text = gui_rules.metric_preview_text(
        metric_key,
        "target_model_0",
        ref_sel,
        cutoff,
        has_ensemble,
    )

    assert expected in text


def test_metric_preview_text_handles_no_metric() -> None:
    assert (
        gui_rules.metric_preview_text(None, "target_model_0", "", "5.0", False)
        == "Select a Color by metric."
    )


def test_metric_preview_summary_preserves_meaning_without_variable_details() -> None:
    reference = "chain B and resi 10-250"
    summary = gui_rules.metric_preview_summary(
        "pae_to_sel",
        "target_model_0",
        reference,
        "5.0",
        False,
    )
    details = gui_rules.metric_preview_text(
        "pae_to_sel",
        "target_model_0",
        reference,
        "5.0",
        False,
    )

    assert "directional PAE" in summary
    assert '"the reference selection"' in summary
    assert reference not in summary
    assert reference in details
    assert "Matrix plot columns" in details
    assert "Matrix plot columns" not in summary


def test_metric_preview_text_appends_reference_plot_restriction() -> None:
    text = gui_rules.metric_preview_text(
        "plddt",
        "target_model_0",
        "chain B",
        "5.0",
        False,
    )

    assert "continuous local confidence" in text
    assert (
        "Line plot x-ranges and distribution plots are restricted "
        'to tokens selected by "chain B".'
    ) in text
    assert "matrix plot x-ranges" not in text
    assert "\n\nAttention: Line plot x-ranges" in text


def test_metric_details_extend_the_compact_preview() -> None:
    summary = gui_rules.metric_preview_summary(
        "pae_row_mean", "target_model_0", "", "5.0", False
    )
    details = gui_rules.metric_preview_text(
        "pae_row_mean", "target_model_0", "", "5.0", False
    )

    assert details.startswith(summary)
    assert "mean of PAE[i, :]" in details
    assert "Lower values identify better-anchored tokens" in details


def test_domain_preview_restriction_excludes_unavailable_line_plot() -> None:
    text = gui_rules.metric_preview_text(
        "pae_domain_complete",
        "target_model_0",
        "chain B",
        "5.0",
        False,
    )

    assert (
        "Matrix plot columns and distribution plots are restricted "
        'to tokens selected by "chain B".'
    ) in text
    assert "line plot x-ranges" not in text


def test_pae_row_mean_preview_mentions_matrix_rows() -> None:
    text = gui_rules.metric_preview_text(
        "pae_row_mean",
        "target_model_0",
        "chain B",
        "5.0",
        False,
    )

    assert (
        "Line plot x-ranges, matrix plot rows, and distribution plots are "
        'restricted to tokens selected by "chain B".'
    ) in text
    assert "matrix plot x-ranges" not in text


@pytest.mark.parametrize(
    "metric_key",
    [
        "pae_to_sel",
        "pae_sym_sel",
        "pae_contact",
        "pde_to_sel",
        "pde_contact",
        "contact_prob_to_sel",
    ],
)
def test_to_selection_preview_restriction_mentions_only_matrix_scope(
    metric_key: str,
) -> None:
    text = gui_rules.metric_preview_text(
        metric_key,
        "target_model_0",
        "chain B",
        "5.0",
        False,
    )

    assert 'Matrix plot columns are restricted to tokens selected by "chain B".' in text
    assert "Line plot x-ranges" not in text
    assert "distribution plots are restricted" not in text


def test_pae_column_to_selection_preview_mentions_matrix_rows() -> None:
    text = gui_rules.metric_preview_text(
        "pae_col_to_sel",
        "target_model_0",
        "chain B",
        "5.0",
        False,
    )

    assert 'Matrix plot rows are restricted to tokens selected by "chain B".' in text
    assert "matrix plot x-ranges" not in text


def test_pae_symmetric_within_selection_preview_mentions_matrix_rows_and_columns() -> (
    None
):
    text = gui_rules.metric_preview_text(
        "pae_sym_within_sel",
        "target_model_0",
        "chain B",
        "5.0",
        False,
    )

    assert "Line plot x-ranges" in text
    assert "matrix plot rows and columns" in text
    assert "distribution plots" in text


def test_metric_preview_text_omits_plot_restriction_without_reference() -> None:
    text = gui_rules.metric_preview_text(
        "plddt",
        "target_model_0",
        "",
        "5.0",
        False,
    )

    assert "restricted to tokens selected by" not in text


def test_chain_iptm_preview_restriction_only_mentions_line_plot() -> None:
    text = gui_rules.metric_preview_text(
        "chain_iptm",
        "target_model_0",
        "chain B",
        "5.0",
        False,
    )

    assert "Plot > Matrix" in text
    assert 'Line plot x-ranges are restricted to tokens selected by "chain B".' in text
    assert "matrix plot x-ranges" not in text
    assert "distribution plots are restricted" not in text


def test_single_target_preview_uses_generic_target_name() -> None:
    text = gui_rules.metric_preview_text(
        "plddt",
        "single",
        "",
        "5.0",
        False,
    )

    assert "the target" in text
    assert "target_model_0" not in text


def test_ensemble_preview_describes_member_coloring_and_plot_aggregation() -> None:
    text = gui_rules.metric_preview_text(
        "pae_row_mean",
        "ensemble_group",
        "",
        "5.0",
        True,
    )

    assert "all members of the ensemble" in text
    assert "line plots show the member mean and standard deviation" in text
    assert "distribution plots use the member mean" in text
    assert "matrix plots show the member mean" in text


def test_ensemble_domain_preview_only_describes_available_matrix_aggregation() -> None:
    text = gui_rules.metric_preview_text(
        "pae_domain_complete",
        "ensemble_group",
        "",
        "5.0",
        True,
    )

    assert "all members of the ensemble" in text
    assert "matrix plots show the member mean" in text
    assert "line plots show" not in text
    assert "distribution plots use" not in text


def test_ensemble_chain_iptm_preview_describes_line_and_matrix_std() -> None:
    text = gui_rules.metric_preview_text(
        "chain_iptm",
        "ensemble_group",
        "",
        "5.0",
        True,
    )

    assert "all members of the ensemble" in text
    assert "line plots show the member mean and standard deviation" in text
    assert (
        "matrix plots show the member mean with standard-deviation annotations" in text
    )
    assert "distribution plots use" not in text


@pytest.mark.parametrize(
    ("metric_key", "expected", "unexpected"),
    [
        (
            "ensemble_rmsd",
            "Plots show the shared ensemble RMSD values.",
            "member mean and standard deviation",
        ),
        (
            "ensemble_plddt_mean",
            "Line plots show ensemble mean pLDDT with standard deviation",
            "matrix plots show",
        ),
        (
            "ensemble_plddt_std",
            "Plots show the ensemble pLDDT standard-deviation values.",
            "member mean and standard deviation",
        ),
    ],
)
def test_ensemble_level_preview_describes_actual_aggregate(
    metric_key: str,
    expected: str,
    unexpected: str,
) -> None:
    text = gui_rules.metric_preview_text(
        metric_key,
        "ensemble_group",
        "",
        "5.0",
        True,
    )

    assert "all members of the ensemble" in text
    assert expected in text
    assert unexpected not in text
