"""
Report text builders for FoldQC.

This module is intentionally independent of Qt and molecular viewers. GUI code passes in
loaded data and computed arrays, then writes the returned text to widgets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import metrics
from .confidence import (
    AffinityConfidence,
    ConfidenceFieldSpec,
    PredictionConfidence,
)
from .loader_models import PredictionFiles, ProviderInfo
from .presentation import (
    ModelComparisonColumn,
    ModelComparisonRequest,
    ModelComparisonRow,
)

if TYPE_CHECKING:
    from .token_map import TokenMap


def format_optional_float(value, *, precision: int = 4) -> str:
    """Format a possibly missing numeric value."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def format_confidence_summary(pred_data, token_map: TokenMap | None = None) -> str:
    """Render one provider's typed confidence data through its schema."""
    if pred_data is None:
        return "No confidence data loaded."
    provider = getattr(pred_data, "provider", None)
    if not isinstance(provider, ProviderInfo):
        raise TypeError("PredictionData.provider must be ProviderInfo.")
    spec = provider.confidence_summary
    if spec.informational_text is not None:
        return spec.informational_text

    confidence = getattr(pred_data, "confidence", None)
    provider_line = f"provider         : {provider.label}"
    if confidence is None:
        lines = [provider_line, "No confidence data loaded."]
        if spec.note_text is not None:
            lines += ["", spec.note_text]
        return "\n".join(lines)
    if not isinstance(confidence, PredictionConfidence):
        raise TypeError("PredictionData.confidence must be PredictionConfidence.")

    lines = [provider_line]
    affinity: AffinityConfidence | None = confidence.affinity
    for field in spec.fields:
        source = affinity if field.source == "affinity" else confidence
        value = None if source is None else getattr(source, field.attribute)
        if value is None and field.omit_when_missing:
            continue
        if isinstance(value, bool):
            formatted = str(value)
        else:
            formatted = format_optional_float(value, precision=field.precision)
        lines.append(f"{field.label:<17}: {formatted}{field.suffix}")

    for section in spec.sections:
        values = getattr(confidence, section.attribute)
        if values is None or not np.isfinite(values).any():
            continue
        lines += ["", f"{section.label}:"]
        for index, value in enumerate(values):
            lines.append(f"  chain {index}: {format_optional_float(value)}")

    for section in spec.matrix_sections:
        values = getattr(confidence, section.attribute)
        if values is None or not np.isfinite(values).any():
            continue
        labels = (
            tuple(token_map.chain_order)
            if token_map is not None and len(token_map.chain_order) == values.shape[0]
            else tuple(str(index) for index in range(values.shape[0]))
        )
        entries = [
            (labels[row], labels[column], values[row, column])
            for row in range(values.shape[0])
            for column in range(row + 1, values.shape[1])
            if np.isfinite(values[row, column])
        ]
        if not entries:
            continue
        lines += ["", f"{section.label}:"]
        for row_label, column_label, value in entries:
            lines.append(
                f"  chains {row_label} / {column_label}: {format_optional_float(value)}"
            )
    if spec.note_text is not None:
        lines += ["", spec.note_text]
    return "\n".join(lines)


def build_model_comparison(
    pred_files: PredictionFiles,
    confidences: tuple[PredictionConfidence | None, ...],
    *,
    selected_rank: int | None,
) -> ModelComparisonRequest:
    """Build a compact provider-schema table for all discovered model ranks."""
    if len(confidences) != len(pred_files.models):
        raise ValueError("Confidence summaries must correspond to discovered models.")
    spec = pred_files.provider.confidence_summary
    fields = tuple(
        field
        for field in spec.fields
        if field.include_in_model_comparison
        and (
            not field.omit_when_missing
            or any(
                _comparison_value(confidence, field) is not None
                for confidence in confidences
            )
        )
    )
    columns = tuple(ModelComparisonColumn(field.label) for field in fields)
    rows = tuple(
        ModelComparisonRow(
            model.rank,
            model.display_label,
            tuple(
                _format_comparison_value(_comparison_value(confidence, field), field)
                for field in fields
            ),
        )
        for model, confidence in zip(pred_files.models, confidences)
    )
    return ModelComparisonRequest(
        title="Compare ranked models",
        provider_label=pred_files.provider.label,
        columns=columns,
        rows=rows,
        selected_rank=selected_rank,
    )


def _comparison_value(
    confidence: PredictionConfidence | None,
    field: ConfidenceFieldSpec,
) -> object | None:
    if confidence is None:
        return None
    source = confidence.affinity if field.source == "affinity" else confidence
    return None if source is None else getattr(source, field.attribute)


def _format_comparison_value(
    value: object | None,
    field: ConfidenceFieldSpec,
) -> str:
    if isinstance(value, (bool, np.bool_)):
        formatted = str(bool(value))
    else:
        formatted = format_optional_float(value, precision=field.precision)
    return f"{formatted}{field.suffix}" if value is not None else formatted


def format_value(value: float) -> str:
    """Format one numeric statistic compactly and consistently."""
    return f"{value:.4g}"


def format_numeric_statistics(values: np.ndarray) -> list[str]:
    """Return finite-value summary lines for one per-token array."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    ignored = int(arr.size - finite.size)
    if ignored:
        lines = [
            f"tokens        : {arr.size} (finite: {finite.size}, ignored: {ignored})",
        ]
    else:
        lines = [f"tokens        : {arr.size}"]
    if finite.size == 0:
        lines.append("No finite values.")
        return lines

    q1, median, q3 = np.percentile(finite, [25.0, 50.0, 75.0])
    lines += [
        f"mean          : {format_value(float(np.mean(finite)))}",
        f"std           : {format_value(float(np.std(finite)))}",
        f"min           : {format_value(float(np.min(finite)))}",
        f"Q1            : {format_value(float(q1))}",
        f"median        : {format_value(float(median))}",
        f"Q3            : {format_value(float(q3))}",
        f"max           : {format_value(float(np.max(finite)))}",
    ]
    return lines


def format_plddt_class_statistics(values: np.ndarray) -> list[str]:
    """Return pLDDT quality-class count and percentage lines."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    lines = ["pLDDT classes:"]
    if finite.size == 0:
        lines.append("  No finite pLDDT values.")
        return lines

    plddt_pct = finite * 100.0 if float(np.max(finite)) <= 1.5 else finite
    for label, lower, upper in metrics.PLDDT_CLASS_STATS:
        mask = np.ones(plddt_pct.shape, dtype=bool)
        if lower is not None:
            mask &= plddt_pct >= lower
        if upper is not None:
            mask &= plddt_pct < upper
        count = int(np.count_nonzero(mask))
        percent = 100.0 * count / float(plddt_pct.size)
        lines.append(f"  {label:<17}: {count} ({percent:.1f}%)")
    return lines


def format_chain_statistics(
    values: np.ndarray, token_map: TokenMap | None
) -> list[str]:
    """Return per-chain numeric summaries for a token-indexed array."""
    if token_map is None:
        return []
    arr = np.asarray(values, dtype=np.float64).ravel()
    chains: list[str] = []
    indices_by_chain: dict[str, list[int]] = {}
    for idx, tok in enumerate(token_map):
        if idx >= arr.size:
            break
        chain_id = str(getattr(tok, "chain_id", "") or "(blank)")
        if chain_id not in indices_by_chain:
            chains.append(chain_id)
            indices_by_chain[chain_id] = []
        indices_by_chain[chain_id].append(idx)

    if not chains:
        return []

    lines = ["", "By chain:"]
    for chain_id in chains:
        lines.append(f"Chain {chain_id}")
        chain_values = arr[indices_by_chain[chain_id]]
        lines.extend(format_numeric_statistics(chain_values))
    return lines


def format_domain_label_statistics(values: np.ndarray) -> list[str]:
    """Return cluster-label counts for categorical domain labels."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    lines = ["domain labels:"]
    if finite.size == 0:
        lines.append("  No finite domain labels.")
        return lines
    labels, counts = np.unique(finite.astype(int), return_counts=True)
    lines.append(f"  clusters      : {len(labels)}")
    for label, count in zip(labels, counts):
        lines.append(f"  label {int(label):<7}: {int(count)} tokens")
    return lines


def format_statistics_report(
    metric_key: str,
    target_label: str,
    entries: list[tuple],
    *,
    include_plddt_classes: bool = False,
    include_chain_stats: bool = False,
    include_domain_labels: bool = False,
) -> str:
    """Build the full statistics panel text for one applied coloring."""
    normalised_entries = []
    for entry in entries:
        if len(entry) == 2:
            name, values = entry
            token_map = None
        else:
            name, values, token_map = entry
        normalised_entries.append((name, values, token_map))

    lines = [
        metrics.metric_label(metric_key),
        f"Target: {target_label}",
        "",
    ]
    if not normalised_entries:
        lines.append("No values were painted.")
        return "\n".join(lines)

    if len(normalised_entries) > 1 and not include_domain_labels:
        pooled = np.concatenate(
            [np.asarray(values).ravel() for _, values, _ in normalised_entries]
        )
        lines.append("Overall (pooled)")
        lines.extend(format_numeric_statistics(pooled))
        if include_plddt_classes:
            lines.extend(format_plddt_class_statistics(pooled))
        lines.append("")
    elif len(normalised_entries) > 1 and include_domain_labels:
        lines.append("Overall: not pooled; domain labels are member-local.")
        lines.append("")

    for idx, (name, values, token_map) in enumerate(normalised_entries):
        lines.append(name)
        if include_domain_labels:
            lines.extend(format_domain_label_statistics(values))
        else:
            lines.extend(format_numeric_statistics(values))
        if include_plddt_classes:
            lines.extend(format_plddt_class_statistics(values))
        if include_chain_stats:
            lines.extend(format_chain_statistics(values, token_map))
        if idx != len(normalised_entries) - 1:
            lines.append("")
    return "\n".join(lines)
