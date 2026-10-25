"""Pure residue projections of retained interface results; no data acquisition."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import metrics
from .interface_analysis import (
    InterfaceModelResult,
    InterfaceProjectionResult,
    InterfaceProjectionSelection,
    InterfaceScoreResult,
)


def projection_unavailable_reason(score: InterfaceScoreResult, kind: str) -> str:
    spec = metrics.INTERFACE_PROJECTIONS.get(kind)
    if spec is None or score.metric_key not in spec.metric_keys:
        return "This metric has no defined residue projection of this kind."
    if score.value is None:
        return score.reason or "This metric is unavailable."
    if kind == "ipsae_profile" and not score.residue_profile:
        return "The snapshot has no retained ipSAE residue profile."
    return ""


def projection_score(
    model: InterfaceModelResult, selection: InterfaceProjectionSelection
) -> InterfaceScoreResult:
    if model.rank != selection.rank:
        raise ValueError("Projection model does not match the captured member.")
    matches = tuple(
        score
        for score in model.scores
        if (score.metric_key, score.chain_a, score.partners)
        == (selection.metric_key, selection.chain_a, selection.partners)
    )
    if len(matches) != 1:
        raise ValueError("The selected interface result is not in this snapshot.")
    return matches[0]


def project_interface_residues(
    model: InterfaceModelResult,
    selection: InterfaceProjectionSelection,
    ipsae_cutoff: int,
    check_cancel: Callable[[], None],
) -> InterfaceProjectionResult:
    check_cancel()
    score = projection_score(model, selection)
    reason = projection_unavailable_reason(score, selection.kind)
    if reason:
        raise ValueError(reason)
    tm = model.token_map
    eligible = set(model.protein_indices)
    if any(i < 0 or i >= len(tm) or tm[i].is_hetatm for i in eligible):
        raise ValueError("Invalid protein indices in interface snapshot.")
    a = {i for i in eligible if tm[i].chain_id == score.chain_a}
    b = {i for i in eligible if tm[i].chain_id in score.partners}
    if not a or not b or a & b:
        raise ValueError("Interface projection requires distinct protein partners.")
    scope = a | b if selection.side == "both" else a if selection.side == "a" else b
    values = np.full(len(tm), np.nan, dtype=np.float64)
    partner_label = ", ".join(score.partners)
    metric = metrics.METRICS.require(score.metric_key)
    assert metric.interface is not None
    if selection.kind == "ipsae_profile":
        profile = dict(score.residue_profile)
        if len(profile) != len(score.residue_profile) or set(profile) != a | b:
            raise ValueError(
                "ipSAE profile must cover the exact interface protein tokens once."
            )
        for offset, (i, value) in enumerate(profile.items()):
            if offset % 1024 == 0:
                check_cancel()
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    "ipSAE profile values must be finite scores in [0, 1]."
                )
            if i in scope:
                values[i] = value
        direction = (
            f"{score.chain_a} → {partner_label} and {partner_label} → {score.chain_a}"
            if selection.side == "both"
            else f"{score.chain_a} → {partner_label}"
            if selection.side == "a"
            else f"{partner_label} → {score.chain_a}"
        )
        label = f"ipSAE profile · {direction} · PAE <{ipsae_cutoff} Å"
        description = "Alignment-residue ipSAE; each chain retains its own direction. Zero is a valid score."
    else:
        if not set(score.support_a) <= a or not set(score.support_b) <= b:
            raise ValueError("Contributor indices do not match the interface sides.")
        values[list(scope)] = 0
        support = (set(score.support_a) | set(score.support_b)) & scope
        values[list(support)] = 1
        side = (
            f"{score.chain_a} + {partner_label}"
            if selection.side == "both"
            else score.chain_a
            if selection.side == "a"
            else partner_label
        )
        label = f"{metric.interface.report_label} contributor mask · {score.chain_a} versus {partner_label} · {side}"
        if score.metric_key == "interface_ipsae":
            label += f" · PAE <{ipsae_cutoff} Å"
        description = "Orange: contributor (1); light grey: eligible non-contributor (0); grey70: outside scope. Support combines directions where applicable."
        if score.metric_key == "interface_ilis":
            label = label.replace("contributor mask", "contact-filtered support")
            description += (
                " iLIS shows its cLIS component's support, not every input to iLIS."
            )
    check_cancel()
    return InterfaceProjectionResult(
        selection,
        tm,
        model.object_name,
        values,
        f"{model.label} · {label}",
        description,
    )
