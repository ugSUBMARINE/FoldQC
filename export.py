"""
CSV export helpers for FoldQC.

This module is intentionally viewer-independent. GUI code resolves selections,
computes metric arrays, and passes token maps plus provenance here for CSV row
assembly and writing.
"""

from __future__ import annotations

import csv
import json
import tempfile
import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import metrics
from .interface_analysis import (
    InterfaceContactDiagnostics,
    InterfaceModelResult,
    InterfaceReport,
)

if TYPE_CHECKING:
    from .loader_models import PredictionData, PredictionFiles
    from .token_map import TokenMap

SCHEMA_VERSION = "2"

INTERFACE_SCHEMA_VERSION = "2"
INTERFACE_COLUMNS = (
    "interface_schema_version",
    "provider",
    "prediction_name",
    "input_path",
    "target_name",
    "model_rank",
    "model_label",
    "structure_path",
    "scope",
    "chain_a",
    "partner_chains",
    "metric_key",
    "metric_label",
    "value_units",
    "value_semantics",
    "implementation",
    "implementation_revision",
    "citation",
    "parameters",
    "direction",
    "aggregation",
    "confidence_source",
    "value",
    "status",
    "reason",
    "counts",
    "contact_diagnostics",
    "interface_plddt",
    "contributors_a",
    "contributors_b",
    "n_models",
    "member_ranks",
    "residue_identities",
)


def interface_report_rows(report: InterfaceReport) -> list[dict[str, object]]:
    """Complete snapshot export, independent of UI filtering, sorting or tab."""
    rows: list[dict[str, object]] = []
    base: dict[str, object] = {
        "interface_schema_version": INTERFACE_SCHEMA_VERSION,
        "provider": report.provider,
        "prediction_name": report.prediction_name,
        "input_path": report.input_path,
        "target_name": report.target_name,
    }

    def metadata(key: str) -> dict[str, object]:
        spec = metrics.METRICS.require(key)
        definition = spec.interface
        if definition is None:
            raise ValueError("Interface export requires an interface metric.")
        return {
            "metric_key": key,
            "metric_label": spec.label,
            "scope": spec.result_scope,
            "value_units": spec.value_unit,
            "value_semantics": spec.value_semantics,
            "implementation": definition.implementation,
            "implementation_revision": definition.revision,
            "citation": definition.citation,
        }

    for model in report.models:

        def identities(indices: tuple[int, ...]) -> str:
            return json.dumps(
                [
                    {
                        "chain_id": model.token_map[i].chain_id,
                        "residue_id": model.token_map[i].resi,
                        "res_name": model.token_map[i].res_name,
                        "token_index": i,
                    }
                    for i in indices
                ],
                ensure_ascii=False,
            )

        for score in model.scores:
            spec = metrics.METRICS.require(score.metric_key)
            assert spec.interface is not None
            diagnostic = next(
                (
                    d
                    for d in model.contacts
                    if d.scope == spec.result_scope
                    and d.chain_a == score.chain_a
                    and d.partners == score.partners
                ),
                None,
            )
            common = (
                base
                | metadata(score.metric_key)
                | {
                    "model_rank": model.rank,
                    "model_label": model.label,
                    "structure_path": model.structure_path,
                    "chain_a": score.chain_a,
                    "partner_chains": json.dumps(score.partners),
                    "parameters": json.dumps(dict(score.parameters)),
                    "confidence_source": score.confidence_source,
                    "status": score.status,
                    "reason": score.reason,
                    "counts": json.dumps(dict(score.counts)),
                    "contact_diagnostics": ""
                    if diagnostic is None
                    else json.dumps(
                        asdict(diagnostic) | {"status": diagnostic.status},
                        ensure_ascii=False,
                    ),
                    "interface_plddt": ""
                    if score.interface_plddt is None
                    else score.interface_plddt,
                    "contributors_a": identities(score.support_a),
                    "contributors_b": identities(score.support_b),
                    "n_models": int(score.value is not None),
                    "member_ranks": json.dumps([model.rank]),
                }
            )
            rows.append(
                common
                | {
                    "direction": "summary",
                    "aggregation": spec.interface.aggregation,
                    "value": "" if score.value is None else score.value,
                }
            )
            for direction, value in (
                ("a_to_b", score.forward),
                ("b_to_a", score.reverse),
            ):
                if value is not None:
                    rows.append(
                        common
                        | {
                            "direction": direction,
                            "aggregation": "directional",
                            "value": value,
                            "status": "ok",
                            "reason": "" if score.value is None else score.reason,
                            "n_models": 1,
                        }
                    )
    for aggregate in report.aggregates:
        # Parameters and input provenance are identical within this report/key.
        exemplar = next(
            score
            for model in report.models
            if model.rank in aggregate.ranks
            for score in model.scores
            if score.metric_key == aggregate.metric_key
            and (
                {score.chain_a, *score.partners}
                == {aggregate.chain_a, *aggregate.partners}
                if metrics.METRICS.require(score.metric_key).result_scope
                == "chain_pair"
                else score.chain_a == aggregate.chain_a
                and set(score.partners) == set(aggregate.partners)
            )
        )
        common = (
            base
            | metadata(aggregate.metric_key)
            | {
                "chain_a": aggregate.chain_a,
                "partner_chains": json.dumps(aggregate.partners),
                "parameters": json.dumps(dict(exemplar.parameters)),
                "confidence_source": exemplar.confidence_source,
                "status": "ok",
                "n_models": aggregate.n_models,
                "member_ranks": json.dumps(aggregate.ranks),
                "residue_identities": json.dumps(aggregate.identities),
                "direction": "summary",
            }
        )
        for name, value in (
            ("mean", aggregate.mean),
            ("population_std", aggregate.std),
            ("minimum", aggregate.minimum),
            ("maximum", aggregate.maximum),
        ):
            rows.append(common | {"aggregation": name, "value": value})
        for direction, value in (
            ("a_to_b", aggregate.forward),
            ("b_to_a", aggregate.reverse),
        ):
            if value is not None:
                rows.append(
                    common
                    | {"direction": direction, "aggregation": "mean", "value": value}
                )
    return rows


def write_interface_report(path: str | Path, report: InterfaceReport) -> None:
    """Write the interface schema without changing the token export schema."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=INTERFACE_COLUMNS)
        writer.writeheader()
        writer.writerows(interface_report_rows(report))


CONTACT_SCHEMA_VERSION = "1"
CONTACT_COLUMNS = (
    "contact_schema_version",
    "provider",
    "prediction_name",
    "input_path",
    "target_name",
    "model_rank",
    "model_label",
    "structure_path",
    "scope",
    "chain_a",
    "partner_chains",
    "convention",
    "cutoff_angstrom",
    "comparison",
    "token_index_a",
    "chain_id_a",
    "residue_number_a",
    "insertion_code_a",
    "res_name_a",
    "atom_name_a",
    "token_index_b",
    "chain_id_b",
    "residue_number_b",
    "insertion_code_b",
    "res_name_b",
    "atom_name_b",
    "distance_angstrom",
)


def interface_contact_rows(
    report: InterfaceReport,
    model: InterfaceModelResult,
    diagnostic: InterfaceContactDiagnostics,
    pairs: Iterable[tuple[int, int, float]],
) -> Iterator[dict[str, object]]:
    """One row per geometric contact, with lossless canonical endpoints."""
    base: dict[str, object] = {
        "contact_schema_version": CONTACT_SCHEMA_VERSION,
        "provider": report.provider,
        "prediction_name": report.prediction_name,
        "input_path": report.input_path,
        "target_name": report.target_name,
        "model_rank": model.rank,
        "model_label": model.label,
        "structure_path": model.structure_path,
        "scope": diagnostic.scope,
        "chain_a": diagnostic.chain_a,
        "partner_chains": json.dumps(diagnostic.partners),
        "convention": diagnostic.convention,
        "cutoff_angstrom": diagnostic.cutoff_angstrom,
        "comparison": diagnostic.comparison,
    }
    for a, b, distance in pairs:
        row = base | {"distance_angstrom": distance}
        for side, i in (("a", a), ("b", b)):
            token = model.token_map[i]
            row.update(
                {
                    f"token_index_{side}": i,
                    f"chain_id_{side}": token.chain_id,
                    f"residue_number_{side}": token.residue_id.number,
                    f"insertion_code_{side}": token.residue_id.insertion_code,
                    f"res_name_{side}": token.res_name,
                    f"atom_name_{side}": "CA"
                    if diagnostic.convention == "ca" or token.res_name.upper() == "GLY"
                    else "CB",
                }
            )
        yield row


class PreparedContactCsv:
    """Own an adjacent temporary output until validated commit or disposal."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cleanup = weakref.finalize(self, path.unlink, missing_ok=True)

    def close(self) -> None:
        self._cleanup()

    def commit(self, destination: str) -> None:
        self.path.replace(destination)
        self._cleanup.detach()


def prepare_interface_contact_csv(
    destination: str,
    rows: Iterable[dict[str, object]],
    check_cancel: Callable[[], None],
) -> PreparedContactCsv:
    """Stream an audit to an owned sibling file; never publish from a worker."""
    output = Path(destination)
    check_cancel()
    prepared: PreparedContactCsv | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            prepared = PreparedContactCsv(Path(stream.name))
            writer = csv.DictWriter(stream, fieldnames=CONTACT_COLUMNS)
            writer.writeheader()
            for row in rows:
                check_cancel()
                writer.writerow(row)
            check_cancel()
            stream.flush()
    except BaseException:
        if prepared is not None:
            prepared.close()
        raise
    return prepared


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
    "residue_id",
    "insertion_code",
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
    pred_files,
    pred_data,
    metric_key: str | None,
    *,
    ensemble: bool = False,
    home: str | Path | None = None,
) -> str:
    """Return a practical default CSV destination path."""
    metric = metric_key or "metric"
    home_path = Path.home() if home is None else Path(home)
    if pred_files is None:
        return str(home_path / f"foldqc_{metric}.csv")
    name = getattr(pred_files, "name", "prediction")
    pred_dir = getattr(pred_files, "pred_dir", home_path) or home_path
    if ensemble:
        filename_metric = (
            metric if metric.startswith("ensemble_") else f"ensemble_{metric}"
        )
        return str(Path(pred_dir) / f"{name}_{filename_metric}.csv")
    rank = 0 if pred_data is None else getattr(pred_data, "rank", 0)
    return str(Path(pred_dir) / f"{name}_rank{rank}_{metric}.csv")


def model_label_for_rank(pred_files, rank: int, *, fallback: str = "") -> str:
    """Return the display label for one prediction rank."""
    try:
        return str(pred_files.model(rank).display_label)
    except Exception:
        return fallback


def build_token_rows(
    *,
    pred_files: PredictionFiles,
    data: PredictionData,
    token_map: TokenMap,
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
    metric_spec = metrics.METRICS.require(metric_key)
    unit, semantics = metric_spec.value_unit, metric_spec.value_semantics
    label = metric_label if metric_label is not None else metric_spec.label
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
        "provider": pred_files.provider.key,
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
            "residue_id": getattr(tok, "resi", ""),
            "insertion_code": getattr(tok, "insertion_code", ""),
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
