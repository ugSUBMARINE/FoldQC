"""
Report text builders for FoldQC.

This module is intentionally independent of Qt and PyMOL.  GUI code passes in
loaded data and computed arrays, then writes the returned text to widgets.
"""

from __future__ import annotations

import numpy as np

from . import metrics


def provider_display_label(provider: str) -> str:
    """Return the user-facing provider label used in summaries."""
    return {
        "boltz": "Boltz-2",
        "alphafold3": "AlphaFold 3",
        "af3_server": "AlphaFold 3 Server",
        "chai1": "Chai-1 Discovery",
        "protenix": "Protenix",
        "structure_only": "Structure-only",
    }.get(provider, provider)


def sorted_chain_items(values: dict):
    """Sort chain-indexed JSON dictionaries numerically when possible."""

    def key(item):
        chain_key, _ = item
        try:
            return (0, int(chain_key))
        except (TypeError, ValueError):
            return (1, str(chain_key))

    return sorted(values.items(), key=key)


def iter_chain_values(values):
    """Yield chain-indexed values from dicts or lists in stable order."""
    if isinstance(values, list):
        return [(str(idx), value) for idx, value in enumerate(values)]
    if isinstance(values, dict):
        return sorted_chain_items(values)
    return []


def format_optional_float(value, *, precision: int = 4) -> str:
    """Format a possibly missing numeric value."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_confidence_value(values: dict, key: str) -> str:
    """Format one confidence dictionary value."""
    return format_optional_float(values.get(key))


def format_confidence_summary(pred_data) -> str:
    """Build confidence summary text from loaded prediction data."""
    if pred_data is None:
        return "No confidence data loaded."

    provider = getattr(pred_data, "provider", "unknown")
    provider_line = f"provider         : {provider_display_label(provider)}"

    if provider == "structure_only":
        return "Structure-only input: pLDDT read from B-factors."

    if provider in {"alphafold3", "af3_server", "chai1", "protenix"}:
        conf = getattr(pred_data, "confidence", None) or getattr(
            pred_data, "summary_confidence", None
        )
        if not conf:
            return f"{provider_line}\nNo confidence data loaded."
        disorder = conf.get("fraction_disordered", conf.get("disorder"))
        lines = [
            provider_line,
            f"ranking_score    : {_format_confidence_value(conf, 'ranking_score')}",
            f"ptm              : {_format_confidence_value(conf, 'ptm')}",
            f"iptm             : {_format_confidence_value(conf, 'iptm')}",
            f"fraction_disord. : {format_optional_float(disorder)}",
            f"has_clash        : {conf.get('has_clash', 'n/a')}",
        ]
        if "gpde" in conf:
            lines.append(f"gpde             : {_format_confidence_value(conf, 'gpde')}")
        chain_ptm = conf.get("chains_ptm") or conf.get("chain_ptm")
        if chain_ptm:
            lines += ["", "chain_ptm:"]
            for chain_key, value in iter_chain_values(chain_ptm):
                lines.append(f"  chain {chain_key}: {format_optional_float(value)}")
        chain_iptm = conf.get("chains_iptm") or conf.get("chain_iptm")
        if chain_iptm:
            lines += ["", "chain_iptm:"]
            for chain_key, value in iter_chain_values(chain_iptm):
                lines.append(f"  chain {chain_key}: {format_optional_float(value)}")
        return "\n".join(lines)

    conf = getattr(pred_data, "confidence", None)
    if conf is None:
        return f"{provider_line}\nNo confidence data loaded."

    lines = [
        provider_line,
        f"confidence_score : {_format_confidence_value(conf, 'confidence_score')}",
        f"ptm              : {_format_confidence_value(conf, 'ptm')}",
        f"iptm             : {_format_confidence_value(conf, 'iptm')}",
        f"ligand_iptm      : {_format_confidence_value(conf, 'ligand_iptm')}",
        f"protein_iptm     : {_format_confidence_value(conf, 'protein_iptm')}",
        f"complex_plddt    : {_format_confidence_value(conf, 'complex_plddt')}",
        f"complex_iplddt   : {_format_confidence_value(conf, 'complex_iplddt')}",
        f"complex_pde      : {_format_confidence_value(conf, 'complex_pde')} Å",
        f"complex_ipde     : {_format_confidence_value(conf, 'complex_ipde')} Å",
    ]
    chains_ptm = conf.get("chains_ptm", {})
    if chains_ptm:
        lines += ["", "chains_ptm:"]
        for chain_key, value in sorted_chain_items(chains_ptm):
            lines.append(f"  chain {chain_key}: {format_optional_float(value)}")
    affinity = getattr(pred_data, "affinity", None)
    if affinity:
        affinity_pred_value = format_optional_float(
            affinity.get("affinity_pred_value"), precision=3
        )
        lines += [
            "",
            f"affinity_pred_value       : {affinity_pred_value}  (log₁₀[IC₅₀/μM])",
            f"affinity_probability      : "
            f"{format_optional_float(affinity.get('affinity_probability_binary'))}",
        ]
    return "\n".join(lines)


def format_value(value: float) -> str:
    """Format one numeric statistic compactly and consistently."""
    return f"{value:.4g}"


def format_numeric_statistics(values: np.ndarray) -> list[str]:
    """Return finite-value summary lines for one per-token array."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    ignored = int(arr.size - finite.size)
    lines = [
        f"tokens        : {arr.size} (finite: {finite.size})",
    ]
    if ignored:
        lines.append(f"ignored       : {ignored}")
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


def format_chain_statistics(values: np.ndarray, token_map) -> list[str]:
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
