"""Pure table, heatmap and details projections for an interface snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from . import metrics
from .interface_analysis import (
    InterfaceContactDiagnostics,
    InterfaceReport,
    InterfaceScoreResult,
)
from .interface_compute import interface_metric_specs
from .interface_projection import projection_unavailable_reason


def residue_action_reason(
    row: InterfaceTableRow | None, key: str, kind: str, *, current: bool, busy: bool
) -> str:
    if busy:
        return "Wait for the active operation to finish."
    if row is None:
        return "Select an interface row first."
    if row.rank is None:
        return "Choose an individual model; aggregate rows have no residue projection."
    if not current:
        return "This snapshot is stale. Analyze the current target again."
    score = row.score(key)
    if score is None:
        return "The selected interface metric is unavailable."
    return projection_unavailable_reason(score, kind)


@dataclass(frozen=True)
class InterfaceTableRow:
    label: str
    rank: int | None
    chain_a: str
    partners: tuple[str, ...]
    scores: tuple[InterfaceScoreResult, ...]
    contacts: InterfaceContactDiagnostics | None = None

    def score(self, key: str) -> InterfaceScoreResult | None:
        return next((s for s in self.scores if s.metric_key == key), None)


def report_rows(report: InterfaceReport, scope: str) -> tuple[InterfaceTableRow, ...]:
    visible = {spec.key for spec in report_metric_specs(scope)}
    rows: list[InterfaceTableRow] = []
    for model in report.models:
        pairs: dict[tuple[str, tuple[str, ...]], list[InterfaceScoreResult]] = {}
        for score in model.scores:
            if score.metric_key in visible:
                pairs.setdefault((score.chain_a, score.partners), []).append(score)
        rows.extend(
            InterfaceTableRow(
                model.label,
                model.rank,
                a,
                b,
                tuple(scores),
                next(
                    (
                        d
                        for d in model.contacts
                        if d.scope == scope and d.chain_a == a and d.partners == b
                    ),
                    None,
                ),
            )
            for (a, b), scores in pairs.items()
        )
    aggregates: dict[
        tuple[str, tuple[str, ...], tuple[tuple[str, str, str], ...]],
        list[InterfaceScoreResult],
    ] = {}
    ranks: dict[
        tuple[str, tuple[str, ...], tuple[tuple[str, str, str], ...]], set[int]
    ] = {}
    for agg in report.aggregates:
        if agg.metric_key not in visible:
            continue
        key = agg.chain_a, agg.partners, agg.identities
        aggregates.setdefault(key, []).append(
            InterfaceScoreResult(
                agg.metric_key,
                agg.chain_a,
                agg.partners,
                agg.mean,
                agg.forward,
                agg.reverse,
                reason=f"Ensemble: n={agg.n_models}; population SD={agg.std:.6g}; min={agg.minimum:.6g}; max={agg.maximum:.6g}. Ranks: {agg.ranks}",
                counts=(("n_models", agg.n_models),),
            )
        )
        ranks.setdefault(key, set()).update(agg.ranks)
    rows.extend(
        InterfaceTableRow(
            f"Ensemble mean (ranks {', '.join(map(str, sorted(ranks[key])))})",
            None,
            key[0],
            key[1],
            tuple(scores),
        )
        for key, scores in aggregates.items()
    )
    return tuple(rows)


def report_metric_specs(scope: str) -> tuple[metrics.MetricSpec, ...]:
    return tuple(
        spec
        for spec in interface_metric_specs()
        if spec.result_scope == scope
        and spec.interface is not None
        and spec.interface.report_label is not None
    )


def pair_metric_specs() -> tuple[metrics.MetricSpec, ...]:
    return report_metric_specs("chain_pair")


def heatmap_value(
    rows: tuple[InterfaceTableRow, ...],
    chain_a: str,
    chain_b: str,
    key: str,
    rank: int | None,
    directional: bool,
) -> float | None:
    if chain_a == chain_b:
        return None
    matches = [
        row
        for row in rows
        if row.rank == rank and {row.chain_a, *row.partners} == {chain_a, chain_b}
    ]
    # Incompatible residue identities must never be combined into one heatmap cell.
    if len(matches) != 1:
        return None
    row = matches[0]
    score = row.score(key)
    if score is None:
        return None
    if directional:
        return score.forward if row.chain_a == chain_a else score.reverse
    return score.value


def score_details(report: InterfaceReport, row: InterfaceTableRow, key: str) -> str:
    score = row.score(key)
    spec = metrics.METRICS.require(key)
    definition = spec.interface
    assert definition is not None
    lines = [
        f"{row.label}: {row.chain_a} → {', '.join(row.partners)}",
        contact_details(row.contacts),
        "",
        spec.label,
        spec.details_template,
        "",
        f"Scope: {'Chain pair' if spec.result_scope == 'chain_pair' else 'Chain versus rest'}; summary: {definition.aggregation}",
        f"Implementation revision: {definition.revision}",
    ]
    if score is None:
        return "\n".join(lines + ["N/A: score unavailable for this row."])
    lines.extend(
        [
            f"Value: {'N/A' if score.value is None else f'{score.value:.8g}'}",
            f"A→B: {score.forward}; B→A: {score.reverse}",
            f"Parameters: {dict(score.parameters)}",
            f"Confidence source: {score.confidence_source}",
            f"Metric support counts: {dict(score.counts)}",
            "LIA/cLIA count directional qualifying cells, not areas or unique residues.",
            f"Interface pLDDT: {score.interface_plddt}",
            score.reason,
        ]
    )
    model = next((m for m in report.models if m.rank == row.rank), None)
    if model is not None:
        for name, indices in (("A", score.support_a), ("B/rest", score.support_b)):
            lines.append(
                f"Contributors {name} ({len(indices)}): "
                + ", ".join(
                    f"{model.token_map[i].chain_id}/{model.token_map[i].resi} {model.token_map[i].res_name}"
                    for i in indices
                )
            )
        lines.extend(model.diagnostics)
    return "\n".join(lines)


def contact_details(diagnostic: InterfaceContactDiagnostics | None) -> str:
    """Compact geometry summary shared by details and contact tooltips."""
    if diagnostic is None:
        return "Geometric contacts: N/A. Diagnostics are available per model only."
    atoms = "Cβ, Cα for glycine" if diagnostic.convention == "cb_ca_gly" else "Cα"
    definition = f"{atoms}; distance ≤{diagnostic.cutoff_angstrom:g} Å."
    if diagnostic.pair_count is None:
        return f"Geometric contacts: N/A. {definition} {diagnostic.reason}"
    partner = "B" if diagnostic.scope == "chain_pair" else "rest"
    return (
        f"Geometric contacts: {diagnostic.pair_count} pairs; "
        f"unique residues A: {diagnostic.residues_a}, {partner}: {diagnostic.residues_b}. "
        f"{definition}"
    )
