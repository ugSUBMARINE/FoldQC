"""
CSV export helpers for FoldQC.

This module is intentionally viewer-independent. GUI code resolves selections,
computes metric arrays, and passes token maps plus provenance here for CSV row
assembly and writing.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import metrics

if TYPE_CHECKING:
    from .loader import PredictionData
    from .token_map import TokenInfo

SCHEMA_VERSION = "1"

BASE_COLUMNS = [
    "export_schema_version",
    "provider",
    "prediction_name",
    "input_path",
    "structure_path",
    "model_rank",
    "model_label",
    "metric_key",
    "metric_label",
    "value",
    "value_units",
    "value_semantics",
    "reference_selection",
    "cutoff_angstrom",
    "token_index",
    "token_type",
    "chain_id",
    "res_num",
    "res_name",
    "atom_name",
    "is_hetatm",
    "is_reference_token",
    "is_contact_token",
]

ENSEMBLE_COLUMNS = [
    "ensemble_group",
    "ensemble_member_rank",
    "ensemble_member_label",
    "ensemble_aligned",
    "aggregate_kind",
]


def fieldnames(*, include_ensemble: bool = False) -> list[str]:
    """Return CSV fieldnames for token-level exports."""
    names = list(BASE_COLUMNS)
    if include_ensemble:
        names.extend(ENSEMBLE_COLUMNS)
    return names


def default_csv_export_path(
    pred_files, pred_data, metric_key: str | None, *, home: str | Path | None = None
) -> str:
    """Return a practical default CSV destination path."""
    metric = metric_key or "metric"
    home_path = Path.home() if home is None else Path(home)
    if pred_files is None:
        return str(home_path / f"foldqc_{metric}.csv")
    rank = 0 if pred_data is None else getattr(pred_data, "rank", 0)
    name = getattr(pred_files, "name", "prediction")
    pred_dir = getattr(pred_files, "pred_dir", home_path) or home_path
    return str(Path(pred_dir) / f"{name}_rank{rank}_{metric}.csv")


def model_label_for_rank(pred_files, rank: int, *, fallback: str = "") -> str:
    """Return the display label for one prediction rank."""
    try:
        return str(pred_files.model(rank).display_label)
    except Exception:
        return fallback


def build_token_rows(
    *,
    pred_files,
    data: PredictionData,
    token_map: list[TokenInfo],
    values,
    metric_key: str,
    metric_label: str | None = None,
    reference_selection: str = "",
    cutoff_angstrom: float | None = None,
    reference_indices: Iterable[int] | None = None,
    contact_indices: Iterable[int] | None = None,
    include_ensemble: bool = False,
    ensemble_group: str = "",
    ensemble_member_rank: int | None = None,
    ensemble_member_label: str = "",
    ensemble_aligned: bool | None = None,
    aggregate_kind: str = "single_model",
) -> list[dict[str, object]]:
    """Build token-level CSV rows from a computed metric array."""
    unit, semantics = metrics.metric_units_and_semantics(metric_key)
    label = (
        metric_label if metric_label is not None else metrics.metric_label(metric_key)
    )
    arr = np.asarray(values)
    ref_set = set(reference_indices or [])
    contact_set = set(contact_indices or [])
    model_rank = int(getattr(data, "rank", 0))
    model_label = getattr(data, "display_label", "") or model_label_for_rank(
        pred_files, model_rank
    )
    structure_path = str(getattr(data, "structure_path", "") or "")

    common = {
        "export_schema_version": SCHEMA_VERSION,
        "provider": getattr(pred_files, "provider", getattr(data, "provider", "")),
        "prediction_name": getattr(pred_files, "name", getattr(data, "name", "")),
        "input_path": str(getattr(pred_files, "input_path", "") or ""),
        "structure_path": structure_path,
        "model_rank": model_rank,
        "model_label": model_label,
        "metric_key": metric_key,
        "metric_label": label,
        "value_units": unit,
        "value_semantics": semantics,
        "reference_selection": reference_selection,
        "cutoff_angstrom": "" if cutoff_angstrom is None else float(cutoff_angstrom),
    }

    rows: list[dict[str, object]] = []
    for tok in token_map:
        token_idx = int(getattr(tok, "token_idx", len(rows)))
        value = arr[token_idx]
        row = {
            **common,
            "value": _csv_value(value),
            "token_index": token_idx,
            "token_type": (
                "ligand_atom" if getattr(tok, "is_hetatm", False) else "polymer_residue"
            ),
            "chain_id": getattr(tok, "chain_id", ""),
            "res_num": getattr(tok, "res_num", ""),
            "res_name": getattr(tok, "res_name", ""),
            "atom_name": getattr(tok, "atom_name", "") or "",
            "is_hetatm": _bool_text(getattr(tok, "is_hetatm", False)),
            "is_reference_token": _bool_text(token_idx in ref_set),
            "is_contact_token": _bool_text(token_idx in contact_set),
        }
        if include_ensemble:
            row.update(
                {
                    "ensemble_group": ensemble_group,
                    "ensemble_member_rank": (
                        ""
                        if ensemble_member_rank is None
                        else int(ensemble_member_rank)
                    ),
                    "ensemble_member_label": ensemble_member_label,
                    "ensemble_aligned": (
                        "" if ensemble_aligned is None else _bool_text(ensemble_aligned)
                    ),
                    "aggregate_kind": aggregate_kind,
                }
            )
        rows.append(row)
    return rows


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write token rows with a stable CSV header."""
    include_ensemble = any(any(key in row for key in ENSEMBLE_COLUMNS) for row in rows)
    with Path(path).open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames(include_ensemble=include_ensemble)
        )
        writer.writeheader()
        writer.writerows(rows)


def _csv_value(value) -> float | str:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(scalar):
        return "nan"
    return scalar


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"
