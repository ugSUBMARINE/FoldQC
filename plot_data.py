"""
Pure plot-data preparation helpers for FoldQC.

This module intentionally has no Qt, PyMOL, or Matplotlib imports. GUI code
handles lazy loading, PyMOL selections, and figure construction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from . import metrics, properties
from .palettes import categorical_color

MAX_HISTOGRAM_BINS = 50


def chain_boundaries(
    token_map,
    indices: list[int] | None = None,
    *,
    original_x: bool = False,
) -> tuple[list[float], list[tuple[str, float]]]:
    """Return chain transition positions and labels for displayed tokens."""
    seq = list(range(len(token_map))) if indices is None else list(indices)
    if not seq:
        return [], []

    boundaries: list[float] = []
    labels: list[tuple[str, float]] = []
    run_start = 0
    current_chain = str(getattr(token_map[seq[0]], "chain_id", "") or "(blank)")

    def _pos(display_idx: int) -> float:
        return float(seq[display_idx] if original_x else display_idx)

    def _finish_run(end_display_idx: int, chain_id: str) -> None:
        labels.append((chain_id, (_pos(run_start) + _pos(end_display_idx)) / 2.0))

    for display_idx in range(1, len(seq)):
        chain_id = str(
            getattr(token_map[seq[display_idx]], "chain_id", "") or "(blank)"
        )
        if chain_id == current_chain:
            continue
        previous_pos = _pos(display_idx - 1)
        current_pos = _pos(display_idx)
        boundaries.append((previous_pos + current_pos) / 2.0)
        _finish_run(display_idx - 1, current_chain)
        run_start = display_idx
        current_chain = chain_id

    _finish_run(len(seq) - 1, current_chain)
    return boundaries, labels


def has_multiple_token_chains(token_map) -> bool:
    """Return True when a token map contains more than one chain ID."""
    chains = {str(getattr(tok, "chain_id", "") or "(blank)") for tok in token_map}
    return len(chains) > 1


def line_member_load_flags(key: str) -> tuple[bool, bool, bool, bool, bool]:
    """Return PAE/PDE/contact/pLDDT load flags for one line property."""
    compute_key = metrics.line_compute_key(key)
    return (
        compute_key.startswith("pae"),
        compute_key.startswith("pde"),
        compute_key.startswith("contact_prob"),
        compute_key == "plddt",
        compute_key == "plddt",
    )


def nan_mean_std(
    arrays: list[np.ndarray | None],
    size: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute mean/std while treating missing arrays as all-NaN."""
    if not arrays or all(arr is None for arr in arrays):
        return None, None
    sample_shape = next(
        np.asarray(arr, dtype=np.float32).shape for arr in arrays if arr is not None
    )
    filled = [
        np.full(sample_shape or (size,), np.nan, dtype=np.float32)
        if arr is None
        else np.asarray(arr, dtype=np.float32)
        for arr in arrays
    ]
    stack = np.stack(filled, axis=0)
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    sums = np.where(finite, stack, 0.0).sum(axis=0)
    mean = np.full(sample_shape or (size,), np.nan, dtype=np.float32)
    np.divide(sums, counts, out=mean, where=counts > 0)
    centered = np.where(finite, stack - mean, 0.0)
    variance = np.full(sample_shape or (size,), np.nan, dtype=np.float32)
    np.divide(
        (centered * centered).sum(axis=0),
        counts,
        out=variance,
        where=counts > 0,
    )
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def summary_series_for_data(
    kind: str,
    data,
    token_map,
) -> list[
    tuple[str, np.ndarray, np.ndarray | None]
    | tuple[str, np.ndarray, np.ndarray | None, str]
]:
    """Return PAE/PDE summary line series for one already-loaded model."""
    if kind == "pae":
        pae = getattr(data, "pae", None)
        if pae is None:
            raise ValueError("PAE matrix is not available for this model.")
        row_within, row_other, col_within, col_other = properties.pae_chain_summary(
            pae, token_map
        )
        return [
            ("row gap (other - within)", row_other - row_within, None, "#1f77b4"),
            ("column gap (other - within)", col_other - col_within, None, "#6baed6"),
        ]
    if kind == "pde":
        pde = getattr(data, "pde", None)
        if pde is None:
            raise ValueError("PDE matrix is not available for this model.")
        within, other = properties.pde_chain_summary(pde, token_map)
        return [
            ("gap (other - within)", other - within, None),
        ]
    raise ValueError(f"Unknown summary kind: {kind}")


def summary_series_for_ensemble(
    kind: str,
    data_items: list,
    token_map,
    token_maps: list | None = None,
) -> list[
    tuple[str, np.ndarray, np.ndarray | None]
    | tuple[str, np.ndarray, np.ndarray | None, str]
]:
    """Return PAE/PDE summary line series aggregated across ensemble models."""
    if not data_items:
        raise ValueError("No ensemble models are available for this summary plot.")
    if token_maps is None:
        token_maps = [token_map for _data in data_items]
    if len(token_maps) != len(data_items):
        raise ValueError("token_maps must match data_items.")

    by_label: dict[str, list[np.ndarray | None]] = {}
    labels: list[str] = []
    colors: dict[str, str | None] = {}
    for data, member_token_map in zip(data_items, token_maps, strict=True):
        series = summary_series_for_data(kind, data, member_token_map)
        if not labels:
            labels = [item[0] for item in series]
            by_label = {label: [] for label in labels}
            colors = {item[0]: item[3] if len(item) == 4 else None for item in series}
        elif [item[0] for item in series] != labels:
            raise ValueError("Ensemble summary series are incompatible.")
        for item in series:
            label, values = item[0], item[1]
            by_label[label].append(np.asarray(values, dtype=np.float32))

    aggregated: list[
        tuple[str, np.ndarray, np.ndarray | None]
        | tuple[str, np.ndarray, np.ndarray | None, str]
    ] = []
    for label in labels:
        mean, std = nan_mean_std(by_label[label], len(token_map))
        if mean is None:
            raise ValueError("No ensemble values are available for this summary plot.")
        color = colors.get(label)
        item_label = f"{label} mean"
        if color is None:
            aggregated.append((item_label, mean, std))
        else:
            aggregated.append((item_label, mean, std, color))
    return aggregated


def plddt_class_distribution_groups(
    values: np.ndarray,
    token_indices: list[int],
) -> tuple[list[str], list[int], list[list[int]], int]:
    """Return class labels, counts, token groups, and finite total for pLDDT."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        labels = [
            metrics.PLDDT_CLASS_PLOT_LABELS[label]
            for label, _lower, _upper in reversed(metrics.PLDDT_CLASS_STATS)
        ]
        return labels, [0 for _ in labels], [[] for _ in labels], 0

    scale = 100.0 if float(np.max(finite)) <= 1.5 else 1.0
    pct = arr * scale
    labels: list[str] = []
    counts: list[int] = []
    groups: list[list[int]] = []
    for label, lower, upper in reversed(metrics.PLDDT_CLASS_STATS):
        mask = np.isfinite(pct)
        if lower is not None:
            mask &= pct >= lower
        if upper is not None:
            mask &= pct < upper
        group = [
            int(token_idx)
            for token_idx, selected in zip(token_indices, mask)
            if bool(selected)
        ]
        labels.append(metrics.PLDDT_CLASS_PLOT_LABELS[label])
        counts.append(len(group))
        groups.append(group)
    return labels, counts, groups, int(finite.size)


def domain_label_distribution_groups(
    values: np.ndarray,
    token_indices: list[int],
) -> tuple[list[str], list[int], list[list[int]], list[tuple[float, float, float]]]:
    """Return domain-label category labels, counts, token groups, and colors."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(arr)
    labels: list[str] = []
    counts: list[int] = []
    groups: list[list[int]] = []
    colors: list[tuple[float, float, float]] = []
    rounded = np.full(arr.shape, -1, dtype=np.int64)
    rounded[finite] = np.rint(arr[finite]).astype(np.int64)
    for label in sorted({int(value) for value in rounded[finite]}):
        mask = finite & (rounded == label)
        group = [
            int(token_idx)
            for token_idx, selected in zip(token_indices, mask)
            if bool(selected)
        ]
        labels.append(str(label))
        counts.append(len(group))
        groups.append(group)
        colors.append(categorical_color(label))
    return labels, counts, groups, colors


def compute_histogram_bins(
    values: Sequence[float] | np.ndarray,
    *,
    max_bins: int = MAX_HISTOGRAM_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-value histogram counts and bin edges."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("No finite values are available for the histogram.")

    max_bins = max(1, int(max_bins))
    n_bins = min(max_bins, max(1, int(math.ceil(math.sqrt(float(finite.size))))))
    counts, edges = np.histogram(finite, bins=n_bins)
    return counts.astype(np.int64), edges.astype(np.float64)


def histogram_distribution_groups(
    values: np.ndarray,
    token_indices: list[int],
    *,
    max_bins: int = MAX_HISTOGRAM_BINS,
) -> tuple[np.ndarray, list[list[int]], list[float], list[float]]:
    """Return histogram edges, token groups, bar positions, and bar widths."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    _counts, edges = compute_histogram_bins(arr, max_bins=max_bins)
    finite = np.isfinite(arr)
    groups: list[list[int]] = []
    for bin_idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_idx == len(edges) - 2:
            mask = finite & (arr >= left) & (arr <= right)
        else:
            mask = finite & (arr >= left) & (arr < right)
        groups.append(
            [
                int(token_idx)
                for token_idx, selected in zip(token_indices, mask)
                if bool(selected)
            ]
        )
    widths = np.diff(edges).astype(np.float64)
    positions = (edges[:-1] + widths / 2.0).astype(np.float64)
    return edges, groups, positions.tolist(), widths.tolist()


def format_matrix_cell_text(
    matrix: np.ndarray,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Return cell annotation strings for finite matrix values."""
    values = np.asarray(matrix, dtype=np.float64)
    errors = None if std is None else np.asarray(std, dtype=np.float64)
    text = np.full(values.shape, None, dtype=object)
    for idx in np.ndindex(values.shape):
        value = values[idx]
        if not np.isfinite(value):
            continue
        if errors is None:
            text[idx] = f"{value:.3f}"
            continue
        err = errors[idx]
        if np.isfinite(err):
            text[idx] = f"{value:.3f} +/- {err:.3f}"
        else:
            text[idx] = f"{value:.3f} +/- n/a"
    return text


def fingerprint_arrays_for_data(
    data,
    ref_indices: list[int],
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Return pLDDT, PAE/PDE, and interaction-to-ref arrays for one model."""
    plddt = getattr(data, "structure_plddt", None)
    if plddt is None:
        plddt = getattr(data, "plddt", None)
    pae_to_ref = None
    pae_from_ref = None
    pae = getattr(data, "pae", None)
    if pae is not None:
        pae_to_ref = properties.pae_to_selection(pae, ref_indices)
        ref = np.asarray(ref_indices, dtype=int)
        pae_from_ref = pae[ref, :].mean(axis=0).astype(np.float32)
    pde = getattr(data, "pde", None)
    pde_to_ref = None if pde is None else properties.pde_to_selection(pde, ref_indices)
    contact_probs = getattr(data, "contact_probs", None)
    contact_to_ref = (
        None
        if contact_probs is None
        else properties.contact_probability_to_selection(contact_probs, ref_indices)
    )
    return plddt, pae_to_ref, pae_from_ref, pde_to_ref, contact_to_ref


def finite_mean(values: np.ndarray) -> float:
    """Return finite-value mean or NaN when no finite values exist."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def within_site_matrix_mean(
    matrix: np.ndarray | None, site_indices: list[int]
) -> float:
    """Return finite off-diagonal mean for matrix entries within a site."""
    if matrix is None or len(site_indices) < 2:
        return float("nan")
    arr = np.asarray(matrix, dtype=np.float64)
    idx = np.asarray(site_indices, dtype=int)
    sub = arr[np.ix_(idx, idx)]
    mask = np.ones(sub.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    finite = sub[mask]
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def site_summary_values(data, site_indices: list[int]) -> dict[str, float]:
    """Return mean pLDDT, PAE, and PDE values for a resolved site."""
    plddt = getattr(data, "structure_plddt", None)
    if plddt is None:
        plddt = getattr(data, "plddt", None)
    plddt_mean = (
        float("nan")
        if plddt is None
        else finite_mean(np.asarray(plddt, dtype=np.float64)[site_indices])
    )
    return {
        "plddt": plddt_mean,
        "pae": within_site_matrix_mean(getattr(data, "pae", None), site_indices),
        "pde": within_site_matrix_mean(getattr(data, "pde", None), site_indices),
    }


def fingerprint_series_for_single(
    data,
    ref_indices: list[int],
) -> dict[str, np.ndarray | None]:
    """Return fingerprint series for one already-loaded model."""
    plddt, pae_to_ref, pae_from_ref, pde_to_ref, contact_to_ref = (
        fingerprint_arrays_for_data(data, ref_indices)
    )
    return {
        "plddt": plddt,
        "plddt_std": None,
        "pae_to_ligand": pae_to_ref,
        "pae_to_ligand_std": None,
        "pae_from_ligand": pae_from_ref,
        "pae_from_ligand_std": None,
        "pde_to_ligand": pde_to_ref,
        "pde_to_ligand_std": None,
        "interaction_prob_to_ligand": contact_to_ref,
        "interaction_prob_to_ligand_std": None,
    }


def fingerprint_series_for_ensemble(
    data_items: list,
    ref_indices: list[int],
    *,
    size: int,
) -> dict[str, np.ndarray | None]:
    """Return mean/std fingerprint series for already-loaded ensemble data."""
    plddt_arrays = []
    pae_arrays = []
    pae_from_arrays = []
    pde_arrays = []
    contact_arrays = []
    for data in data_items:
        plddt, pae_to_ref, pae_from_ref, pde_to_ref, contact_to_ref = (
            fingerprint_arrays_for_data(data, ref_indices)
        )
        plddt_arrays.append(plddt)
        pae_arrays.append(pae_to_ref)
        pae_from_arrays.append(pae_from_ref)
        pde_arrays.append(pde_to_ref)
        contact_arrays.append(contact_to_ref)

    plddt_mean, plddt_std = nan_mean_std(plddt_arrays, size)
    pae_mean, pae_std = nan_mean_std(pae_arrays, size)
    pae_from_mean, pae_from_std = nan_mean_std(pae_from_arrays, size)
    pde_mean, pde_std = nan_mean_std(pde_arrays, size)
    contact_mean, contact_std = nan_mean_std(contact_arrays, size)
    return {
        "plddt": plddt_mean,
        "plddt_std": plddt_std,
        "pae_to_ligand": pae_mean,
        "pae_to_ligand_std": pae_std,
        "pae_from_ligand": pae_from_mean,
        "pae_from_ligand_std": pae_from_std,
        "pde_to_ligand": pde_mean,
        "pde_to_ligand_std": pde_std,
        "interaction_prob_to_ligand": contact_mean,
        "interaction_prob_to_ligand_std": contact_std,
    }


def chain_iptm_matrix_plot_data(
    *,
    target_kind: str,
    data,
    token_map,
    title: str,
    label: str,
    members: list | None = None,
) -> tuple[
    np.ndarray,
    list[int],
    list[int],
    str,
    str,
    list[str],
    list[str],
    np.ndarray,
]:
    """Return chain-level ipTM matrix data for single models or ensembles."""
    if target_kind == "ensemble_group":
        matrices = []
        chain_labels: list[str] | None = None
        for member in members or []:
            if member.data.confidence is None:
                raise ValueError(
                    f"Confidence JSON is not available for model_{member.rank}."
                )
            matrix, labels = properties.pair_chains_iptm_matrix(
                member.data.confidence, member.token_map
            )
            if chain_labels is None:
                chain_labels = labels
            elif labels != chain_labels:
                raise ValueError(
                    "Ensemble models have incompatible chain order for chain ipTM."
                )
            matrices.append(matrix)
        if not matrices or chain_labels is None:
            raise ValueError("No ensemble chain ipTM values are available.")
        mean, std = nan_mean_std(matrices, matrices[0].size)
        if mean is None or std is None:
            raise ValueError("No ensemble chain ipTM values are available.")
        matrix = mean.reshape(matrices[0].shape)
        std_matrix = std.reshape(matrices[0].shape)
        return (
            matrix,
            list(range(matrix.shape[0])),
            list(range(matrix.shape[1])),
            f"{title} — ensemble mean",
            label,
            chain_labels,
            chain_labels,
            format_matrix_cell_text(matrix, std_matrix),
        )

    if data.confidence is None:
        raise ValueError("Confidence JSON is not available for this model.")
    matrix, chain_labels = properties.pair_chains_iptm_matrix(
        data.confidence, token_map
    )
    return (
        matrix,
        list(range(matrix.shape[0])),
        list(range(matrix.shape[1])),
        title,
        label,
        chain_labels,
        chain_labels,
        format_matrix_cell_text(matrix),
    )
