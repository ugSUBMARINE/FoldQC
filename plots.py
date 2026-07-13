"""
Plots
=====
Matplotlib-based figures for token-indexed line plots, PAE/PDE-style matrices,
and binding-site fingerprints.

Figure-building functions in this module are viewer- and Qt-independent. The
GUI embeds their figures in a host Qt dialog when Matplotlib's Qt canvas is
available.  If Qt embedding fails, the GUI can pass a figure to
``save_and_show()`` to write a temporary PNG and open it with the system
viewer.

The ``"Agg"`` backend is forced at module import time so matplotlib never
tries to create its own window from this module.
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import tempfile
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

# Force non-interactive Agg backend before any pyplot import.
# Must happen before 'import matplotlib.pyplot'.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from .palettes import PLDDT_CLASS_BAR_COLORS, resolve_matplotlib_cmap
from .plot_data import MAX_HISTOGRAM_BINS, compute_histogram_bins

if TYPE_CHECKING:
    from .token_map import TokenMap

MAX_TICKS = 60
GOLDEN_RATIO = (1.0 + 5.0**0.5) / 2.0
MAX_BINDING_SITE_RESIDUES = 40
STACKED_BAR_HEIGHT_MULTIPLIER = 1.2
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_file(path: str) -> None:
    """Open *path* with the system's default viewer."""
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", path])
    elif system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", path])


def save_and_show(fig: plt.Figure) -> str:
    """Save *fig* to a temp PNG, close it, open it, and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _open_file(tmp.name)
    return tmp.name


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    """Return finite min/max for *values*, falling back to 0..1."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _token_label(tok) -> str:
    """Return a compact residue/atom label for one token."""
    prefix = f"{tok.chain_id}{tok.res_num}"
    if getattr(tok, "is_hetatm", False) and getattr(tok, "atom_name", None):
        return f"{prefix}:{tok.atom_name}"
    return prefix


def attach_viewer_selection_metadata(
    fig: plt.Figure,
    *,
    kind: str,
    token_map: TokenMap,
    obj_name: str,
    token_maps: Sequence[TokenMap] | None = None,
    token_map_obj_names: Sequence[str] | None = None,
    token_indices: Sequence[int] | None = None,
    x_positions: Sequence[float] | None = None,
    row_indices: Sequence[int] | None = None,
    col_indices: Sequence[int] | None = None,
    bar_token_indices: Sequence[Sequence[int]] | None = None,
    bar_x_positions: Sequence[float] | None = None,
    bar_widths: Sequence[float] | None = None,
    selection_prefix: str = "foldqc_plot",
) -> plt.Figure:
    """Attach token-selection metadata for :mod:`plot_viewer`.

    The metadata is plain Python data; the embedded Qt dialog delegates viewer
    selection behavior through :mod:`mol_viewer`.
    """
    if kind not in {"line", "bars", "matrix"}:
        raise ValueError(f"Unsupported viewer plot-selection kind: {kind!r}")

    metadata: dict[str, Any] = {
        "kind": kind,
        "token_map": token_map,
        "obj_name": obj_name,
        "selection_prefix": selection_prefix,
    }
    if token_maps is not None or token_map_obj_names is not None:
        if token_maps is None or token_map_obj_names is None:
            raise ValueError(
                "token_maps and token_map_obj_names must be provided together."
            )
        if len(token_map_obj_names) != len(token_maps):
            raise ValueError(
                "token_map_obj_names must correspond one-to-one with token_maps."
            )
        metadata["token_maps"] = list(token_maps)
        metadata["token_map_obj_names"] = [str(name) for name in token_map_obj_names]
    if token_indices is not None:
        metadata["token_indices"] = [int(i) for i in token_indices]
    if x_positions is not None:
        metadata["x_positions"] = [float(x) for x in x_positions]
    if row_indices is not None:
        metadata["row_indices"] = [int(i) for i in row_indices]
    if col_indices is not None:
        metadata["col_indices"] = [int(i) for i in col_indices]
    if bar_token_indices is not None:
        metadata["bar_token_indices"] = [
            [int(i) for i in group] for group in bar_token_indices
        ]
    if bar_x_positions is not None:
        metadata["bar_x_positions"] = [float(x) for x in bar_x_positions]
    if bar_widths is not None:
        metadata["bar_widths"] = [float(w) for w in bar_widths]

    setattr(fig, "_foldqc_viewer_selection", metadata)
    return fig


def attach_ensemble_site_summary_metadata(
    fig: plt.Figure,
    *,
    members: Sequence[Any],
    site_indices: Sequence[Sequence[int]],
    selection_name: str = "foldqc_ensemble_site",
) -> plt.Figure:
    """Attach member-activation metadata for ensemble site summary plots."""
    member_list = list(members)
    if len(member_list) != len(site_indices):
        raise ValueError("members and site_indices must have the same length.")
    metadata: dict[str, Any] = {
        "kind": "ensemble_site_summary",
        "member_obj_names": [str(member.obj_name) for member in member_list],
        "member_token_maps": [member.token_map for member in member_list],
        "member_site_indices": [[int(i) for i in indices] for indices in site_indices],
        "member_x_positions": [float(i) for i in range(len(member_list))],
        "member_widths": [0.9 for _member in member_list],
        "selection_name": selection_name,
    }
    setattr(fig, "_foldqc_viewer_selection", metadata)
    return fig


# ---------------------------------------------------------------------------
# Line plot
# ---------------------------------------------------------------------------


def make_line_plot(
    x_values: Sequence[int] | np.ndarray,
    series: Sequence[
        tuple[str, np.ndarray, np.ndarray | None]
        | tuple[str, np.ndarray, np.ndarray | None, str]
    ],
    title: str = "Line plot",
    ylabel: str = "Value",
    ymin: float | None = None,
    ymax: float | None = None,
    chain_boundaries: Sequence[float] | None = None,
    chain_labels: Sequence[tuple[str, float]] | None = None,
    show_legend: bool | None = None,
) -> plt.Figure:
    """Create a token-indexed line plot.

    ``series`` contains ``(label, values, std_or_none)`` tuples, optionally
    with a fourth color field.  When a standard-deviation array is provided it
    is shown as a shaded band.
    """
    x = np.asarray(x_values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x_values must be one-dimensional.")
    if not series:
        raise ValueError("At least one line series is required.")

    fig_w = max(7, min(16, max(1, x.size) / 35 + 5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w / GOLDEN_RATIO))

    all_values = []
    for item in series:
        if len(item) == 3:
            label, values, std = item
            color = None
        elif len(item) == 4:
            label, values, std, color = item
        else:
            raise ValueError("Line series must contain 3 or 4 fields.")
        y = np.asarray(values, dtype=np.float64)
        if y.shape != x.shape:
            raise ValueError(
                f"Line series '{label}' has shape {y.shape}, expected {x.shape}."
            )
        line_kwargs = {"linewidth": 1.6, "label": label}
        if color is not None:
            line_kwargs["color"] = color
        all_values.append(y)
        if std is not None:
            err = np.asarray(std, dtype=np.float64)
            if err.shape != x.shape:
                raise ValueError(
                    f"Line series '{label}' std has shape {err.shape}, expected {x.shape}."
                )
            lower = y - err
            upper = y + err
            all_values.extend([lower, upper])
            if x.size == 1:
                err_kwargs = dict(line_kwargs)
                err_kwargs.update(
                    {
                        "fmt": "o",
                        "markersize": 4.5,
                        "elinewidth": 1.2,
                        "capsize": 3.0,
                    }
                )
                ax.errorbar(x, y, yerr=err, **err_kwargs)
            else:
                (line,) = ax.plot(x, y, **line_kwargs)
                ax.fill_between(
                    x,
                    lower,
                    upper,
                    alpha=0.18,
                    linewidth=0.0,
                    color=line.get_color(),
                )
        elif x.size == 1:
            point_kwargs = dict(line_kwargs)
            point_kwargs.update({"marker": "o", "markersize": 4.5, "linestyle": "none"})
            ax.plot(x, y, **point_kwargs)
        else:
            ax.plot(x, y, **line_kwargs)

    for pos in chain_boundaries or []:
        ax.axvline(float(pos), color="0.25", linewidth=0.8, alpha=0.45)

    if chain_labels:
        y0, y1 = ax.get_ylim()
        label_y = y1 - (y1 - y0) * 0.04
        for label, pos in chain_labels:
            ax.text(
                float(pos),
                label_y,
                str(label),
                ha="center",
                va="top",
                fontsize=8,
                color="0.25",
            )

    if ymin is not None or ymax is not None:
        current_min, current_max = _finite_range(np.concatenate(all_values))
        ax.set_ylim(
            current_min if ymin is None else float(ymin),
            current_max if ymax is None else float(ymax),
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Token index", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if show_legend is None:
        show_legend = len(series) > 1 or series[0][2] is not None
    if show_legend:
        ax.legend(fontsize=8)
    ax.grid(True, axis="y", color="0.88", linewidth=0.7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------


def make_plddt_class_bar_plot(
    labels: Sequence[str],
    counts: Sequence[int],
    *,
    total: int | None = None,
    title: str = "pLDDT quality classes",
    colors: Sequence[str] | None = None,
) -> plt.Figure:
    """Create a pLDDT quality-class count bar plot."""
    labels = [str(label) for label in labels]
    counts_arr = np.asarray(counts, dtype=np.int64)
    if counts_arr.ndim != 1:
        raise ValueError("counts must be one-dimensional.")
    if len(labels) != counts_arr.size:
        raise ValueError("labels and counts must have the same length.")

    if total is None:
        total = int(np.sum(counts_arr))
    total = max(0, int(total))
    bar_colors = list(colors or PLDDT_CLASS_BAR_COLORS)
    if len(bar_colors) < counts_arr.size:
        repeats = int(math.ceil(counts_arr.size / max(1, len(bar_colors))))
        bar_colors = (bar_colors * repeats)[: counts_arr.size]

    fig, ax = plt.subplots(figsize=(6.0, 6.0 / GOLDEN_RATIO))
    x = np.arange(counts_arr.size, dtype=np.float64)
    bars = ax.bar(x, counts_arr, width=0.8, color=bar_colors[: counts_arr.size])

    ymax = max(1.0, float(np.max(counts_arr)) if counts_arr.size else 1.0)
    offset = ymax * 0.04
    for bar, count in zip(bars, counts_arr):
        percent = 0.0 if total == 0 else 100.0 * int(count) / float(total)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(count) + offset,
            f"{int(count)}\n{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Tokens", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, ymax + offset * 4.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_categorical_bar_plot(
    labels: Sequence[str],
    counts: Sequence[int],
    *,
    title: str = "Category distribution",
    colors: Sequence[Any] | None = None,
) -> plt.Figure:
    """Create a categorical count bar plot."""
    labels = [str(label) for label in labels]
    counts_arr = np.asarray(counts, dtype=np.int64)
    if counts_arr.ndim != 1:
        raise ValueError("counts must be one-dimensional.")
    if len(labels) != counts_arr.size:
        raise ValueError("labels and counts must have the same length.")

    bar_colors = list(colors or ["steelblue"])
    if len(bar_colors) < counts_arr.size:
        repeats = int(math.ceil(counts_arr.size / max(1, len(bar_colors))))
        bar_colors = (bar_colors * repeats)[: counts_arr.size]

    fig_w = max(5.0, min(14.0, counts_arr.size * 0.35 + 4.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w / GOLDEN_RATIO))
    x = np.arange(counts_arr.size, dtype=np.float64)
    bars = ax.bar(
        x,
        counts_arr,
        width=0.8,
        color=bar_colors[: counts_arr.size],
        edgecolor="white",
        linewidth=0.6,
    )

    ymax = max(1.0, float(np.max(counts_arr)) if counts_arr.size else 1.0)
    offset = ymax * 0.04
    for bar, count in zip(bars, counts_arr):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(count) + offset,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Tokens", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_ylim(0.0, ymax + offset * 3.0)
    ax.grid(True, axis="y", color="0.88", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_histogram_plot(
    values: Sequence[float] | np.ndarray,
    *,
    title: str = "Distribution",
    xlabel: str = "Value",
    max_bins: int = MAX_HISTOGRAM_BINS,
    bin_edges: Sequence[float] | np.ndarray | None = None,
) -> plt.Figure:
    """Create a histogram for finite continuous per-token values."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("No finite values are available for the histogram.")

    if bin_edges is None:
        counts, edges = compute_histogram_bins(finite, max_bins=max_bins)
    else:
        edges = np.asarray(bin_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_edges must contain at least two values.")
        counts, edges = np.histogram(finite, bins=edges)
        counts = counts.astype(np.int64)
        edges = edges.astype(np.float64)

    widths = np.diff(edges)
    fig_w = max(6.4, min(12.0, counts.size * 0.18 + 5.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w / GOLDEN_RATIO))
    ax.bar(
        edges[:-1],
        counts,
        width=widths,
        align="edge",
        color="steelblue",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Tokens", fontsize=9)
    ax.grid(True, axis="y", color="0.88", linewidth=0.7)
    fig.tight_layout()
    return fig


def _draw_grouped_bar_series(
    ax,
    x: np.ndarray,
    series: Sequence[tuple[str, np.ndarray, np.ndarray | None, str]],
    *,
    alpha: float = 0.85,
) -> None:
    """Draw one grouped-bar panel on *ax*."""
    k = len(series)
    width = 0.8 / k
    for i, (label, values, yerr, color) in enumerate(series):
        offset = (i - (k - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            yerr=yerr,
            width=width,
            label=label,
            color=color,
            alpha=alpha,
            capsize=2 if yerr is not None else 0,
            error_kw={"linewidth": 0.8, "alpha": alpha},
        )


# ---------------------------------------------------------------------------
# Ensemble site summary
# ---------------------------------------------------------------------------


def make_ensemble_site_summary_plot(
    member_labels: Sequence[str],
    series: Sequence[tuple[str, np.ndarray, str]],
    *,
    title: str = "Ensemble site summary",
) -> plt.Figure:
    """Create grouped bars summarizing ligand-site quality for ensemble members."""
    labels = [str(label) for label in member_labels]
    if not labels:
        raise ValueError("At least one ensemble member is required.")
    if not series:
        raise ValueError("At least one metric series is required.")

    n = len(labels)
    x = np.arange(n, dtype=np.float64)
    fig_w = max(7.0, min(16.0, n * 0.45 + 5.5))
    plddt_series = []
    error_series = []
    for metric_label, values, color in series:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (n,):
            raise ValueError(
                f"Series '{metric_label}' has shape {arr.shape}, expected {(n,)}."
            )
        row = (metric_label, arr, None, color)
        if "plddt" in metric_label.lower():
            plddt_series.append(row)
        else:
            error_series.append(row)

    split_axes = bool(plddt_series and error_series)
    fig_h = (
        fig_w / GOLDEN_RATIO * (STACKED_BAR_HEIGHT_MULTIPLIER if split_axes else 1.0)
    )
    if split_axes:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(fig_w, fig_h),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.15]},
        )
        ax_plddt, ax_error = axes
        ax_plddt.set_title(title, fontsize=11)
        _draw_grouped_bar_series(ax_plddt, x, plddt_series, alpha=0.88)
        ax_plddt.set_ylabel("pLDDT", fontsize=9)
        ax_plddt.set_ylim(0.0, 1.05)
        ax_plddt.legend(fontsize=8)
        ax_plddt.tick_params(axis="x", labelbottom=False)

        _draw_grouped_bar_series(ax_error, x, error_series, alpha=0.88)
        ax_error.set_ylabel("Error (Å)", fontsize=9)
        ax_error.legend(fontsize=8)
        ax = ax_error
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_title(title, fontsize=11)
        _draw_grouped_bar_series(ax, x, plddt_series or error_series, alpha=0.88)
        ax.set_ylabel(
            "pLDDT" if plddt_series and not error_series else "Mean error (Å)",
            fontsize=9,
        )
        if plddt_series and not error_series:
            ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)

    finite_concat = (
        np.concatenate(
            [
                row[1][np.isfinite(row[1])]
                for row in error_series
                if np.any(np.isfinite(row[1]))
            ]
        )
        if error_series and any(np.any(np.isfinite(row[1])) for row in error_series)
        else np.array([], dtype=np.float64)
    )
    if finite_concat.size:
        y_min = min(0.0, float(np.min(finite_concat)))
        y_max = float(np.max(finite_concat))
        if y_min == y_max:
            y_max = y_min + 1.0
        padding = (y_max - y_min) * 0.08
        ax.set_ylim(y_min, y_max + padding)

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45 if n > 8 else 0,
        ha="right" if n > 8 else "center",
        fontsize=8,
    )
    # remove grid for now since it can be too visually noisy
    # ax.grid(True, axis="y", color="0.88", linewidth=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Matrix plot (PAE / PDE)
# ---------------------------------------------------------------------------


def make_matrix_plot(
    matrix: np.ndarray,
    title: str = "Matrix",
    token_map: TokenMap | None = None,
    row_indices: Sequence[int] | None = None,
    col_indices: Sequence[int] | None = None,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    cell_text: Sequence[Sequence[str | None]] | None = None,
    row_chain_boundaries: Sequence[float] | None = None,
    col_chain_boundaries: Sequence[float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    palette: str = "blue_white_red",
    reverse_palette: bool = False,
    xlabel: str = "Token j",
    ylabel: str = "Token i",
    colorbar_label: str = "Value (Å)",
) -> plt.Figure:
    """Create a matrix plot with optional token labels and chain borders."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    n_rows, n_cols = matrix.shape
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        _, vmax = _finite_range(matrix)

    cmap, used_fallback = resolve_matplotlib_cmap(palette, reverse=reverse_palette)
    display_title = title
    if used_fallback:
        display_title = f"{title} (palette '{palette}' -> viridis)"

    fig_size = max(5, min(12, max(n_rows, n_cols) / 10 + 3))
    image_aspect = "equal" if n_rows == n_cols else "auto"

    fig, ax = plt.subplots(figsize=(fig_size + 1.0, fig_size))
    ax.set_box_aspect(1)
    im = ax.imshow(
        matrix,
        aspect=image_aspect,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=9)

    ax.set_title(display_title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    if col_labels is not None and len(col_labels) <= MAX_TICKS:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    if row_labels is not None and len(row_labels) <= MAX_TICKS:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)

    if token_map is not None and row_labels is None and col_labels is None:
        rows = list(range(n_rows)) if row_indices is None else list(row_indices)
        cols = list(range(n_cols)) if col_indices is None else list(col_indices)
        if len(cols) <= MAX_TICKS:
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(
                [_token_label(token_map[i]) for i in cols],
                rotation=90,
                fontsize=6,
            )
        if len(rows) <= MAX_TICKS:
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([_token_label(token_map[i]) for i in rows], fontsize=6)

    if cell_text is not None:
        texts = np.asarray(cell_text, dtype=object)
        if texts.shape != matrix.shape:
            raise ValueError("cell_text must have the same shape as matrix.")
        for row in range(n_rows):
            for col in range(n_cols):
                text = texts[row, col]
                if text is None or text == "":
                    continue
                value = matrix[row, col]
                luminance = 1.0
                if np.isfinite(value):
                    r, g, b, _ = im.cmap(im.norm(value))
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                color = "black" if luminance > 0.55 else "white"
                ax.text(
                    col,
                    row,
                    str(text),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    for pos in row_chain_boundaries or []:
        ax.axhline(float(pos), color="0.15", linewidth=0.8, alpha=0.55)
    for pos in col_chain_boundaries or []:
        ax.axvline(float(pos), color="0.15", linewidth=0.8, alpha=0.55)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Binding-site fingerprint bar chart
# ---------------------------------------------------------------------------


def make_binding_site_fingerprint(
    token_map: TokenMap,
    binding_site_indices: list[int],
    plddt: np.ndarray | None = None,
    plddt_std: np.ndarray | None = None,
    pae_to_ligand: np.ndarray | None = None,
    pae_to_ligand_std: np.ndarray | None = None,
    pae_from_ligand: np.ndarray | None = None,
    pae_from_ligand_std: np.ndarray | None = None,
    pde_to_ligand: np.ndarray | None = None,
    pde_to_ligand_std: np.ndarray | None = None,
    interaction_prob_to_ligand: np.ndarray | None = None,
    interaction_prob_to_ligand_std: np.ndarray | None = None,
    title: str = "Binding-site confidence fingerprint",
    max_residues: int = MAX_BINDING_SITE_RESIDUES,
) -> plt.Figure:
    """Create a grouped bar chart for binding-site confidence values.

    Parameters
    ----------
    binding_site_indices:
        Token indices of binding-site residues (e.g. within 5 Å of ligand).
    max_residues:
        Truncate to the first *max_residues* residues for readability.
    """
    indices = binding_site_indices[:max_residues]
    labels = [
        f"{token_map[i].res_name.title()}-{token_map[i].chain_id}{token_map[i].res_num}"
        for i in indices
    ]

    series: list[tuple[str, np.ndarray, np.ndarray | None, str]] = []
    if plddt is not None:
        yerr = None if plddt_std is None else plddt_std[indices]
        series.append(("pLDDT", plddt[indices], yerr, "steelblue"))
    if pae_to_ligand is not None:
        yerr = None if pae_to_ligand_std is None else pae_to_ligand_std[indices]
        series.append(("PAE row mean (Å)", pae_to_ligand[indices], yerr, "tomato"))
    if pae_from_ligand is not None:
        yerr = None if pae_from_ligand_std is None else pae_from_ligand_std[indices]
        series.append(("PAE column mean (Å)", pae_from_ligand[indices], yerr, "orchid"))
    if pde_to_ligand is not None:
        yerr = None if pde_to_ligand_std is None else pde_to_ligand_std[indices]
        series.append(("PDE mean (Å)", pde_to_ligand[indices], yerr, "goldenrod"))
    if interaction_prob_to_ligand is not None:
        yerr = (
            None
            if interaction_prob_to_ligand_std is None
            else interaction_prob_to_ligand_std[indices]
        )
        series.append(
            (
                "Interaction probability",
                interaction_prob_to_ligand[indices],
                yerr,
                "seagreen",
            )
        )

    if not series:
        raise ValueError("At least one fingerprint series must be provided.")

    n = len(indices)
    x = np.arange(n)
    score_series = [
        row
        for row in series
        if row[0] == "pLDDT" or row[0] == "Interaction probability"
    ]
    error_series = [
        row for row in series if row[0] not in {"pLDDT", "Interaction probability"}
    ]
    split_axes = bool(score_series and error_series)
    fig_w = max(8, n * 0.35)
    fig_h = (
        fig_w / GOLDEN_RATIO * (STACKED_BAR_HEIGHT_MULTIPLIER if split_axes else 1.0)
    )
    if split_axes:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(fig_w, fig_h),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.15]},
        )
        ax_score, ax_error = axes
        ax_score.set_title(title, fontsize=11)
        _draw_grouped_bar_series(ax_score, x, score_series)
        ax_score.set_ylabel("Score", fontsize=9)
        ax_score.set_ylim(0.0, 1.05)
        ax_score.legend(fontsize=8)
        ax_score.tick_params(axis="x", labelbottom=False)

        _draw_grouped_bar_series(ax_error, x, error_series)
        ax_error.set_ylabel("Error (Å)", fontsize=9)
        ax_error.legend(fontsize=8)
        ax = ax_error
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_title(title, fontsize=11)
        _draw_grouped_bar_series(ax, x, series)
        ax.set_ylabel(
            "Score"
            if score_series and not error_series
            else "Error (Å)"
            if error_series and not score_series
            else "Score / Error (Å)"
        )
        if score_series and not error_series:
            ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    fig.tight_layout()
    return fig
