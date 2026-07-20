"""Materialized AlphaFold Protein Structure Database provider."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..confidence import ConfidenceSummarySpec
from ..loader_models import ModelFiles, PredictionData, PredictionFiles
from ..loader_utils import _safe_object_name
from .base import BaseProvider, LoadOptions

MARKER_NAME = "foldqc_afdb.json"
MODEL_NAME = "model.cif"
PAE_NAME = "pae.json"
MARKER_SCHEMA_VERSION = 1

AFDB_CONFIDENCE_SUMMARY = ConfidenceSummarySpec(
    informational_text=(
        "AlphaFold DB input: pLDDT read from structure B-factors; "
        "PAE loaded on demand when available."
    )
)


def _load_marker(path: Path) -> tuple[str, str]:
    try:
        with (path / MARKER_NAME).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read AlphaFold DB marker in {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"AlphaFold DB marker in {path} must be a JSON object.")
    if payload.get("schema_version") != MARKER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported AlphaFold DB marker version in {path}.")
    model_id = payload.get("model_id")
    display_label = payload.get("display_label")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"AlphaFold DB marker in {path} has no model ID.")
    if not isinstance(display_label, str) or not display_label.strip():
        raise ValueError(f"AlphaFold DB marker in {path} has no display label.")
    return model_id.strip(), display_label.strip()


def _load_pae(path: Path) -> np.ndarray:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read AlphaFold DB PAE file {path}: {exc}") from exc
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(
            f"AlphaFold DB PAE file {path} must contain exactly one record."
        )
    record = payload[0]
    if not isinstance(record, dict) or "predicted_aligned_error" not in record:
        raise ValueError(
            f"AlphaFold DB PAE file {path} has no predicted_aligned_error matrix."
        )
    try:
        return np.asarray(record["predicted_aligned_error"], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"AlphaFold DB PAE file {path} contains a malformed matrix."
        ) from exc


class AlphaFoldDatabaseProvider(BaseProvider):
    key, label = "alphafold_db", "AlphaFold DB (EBI)"
    supports_ensemble = False
    confidence_summary = AFDB_CONFIDENCE_SUMMARY

    def detect(self, path: Path) -> bool:
        return (
            path.is_dir()
            and (path / MARKER_NAME).is_file()
            and (path / MODEL_NAME).is_file()
        )

    def scan(self, path: Path) -> PredictionFiles:
        if not self.detect(path):
            raise ValueError(
                f"No materialized AlphaFold DB prediction found in {path}."
            )
        model_id, display_label = _load_marker(path)
        pae_path = path / PAE_NAME
        files = self.prediction_files(name=model_id, pred_dir=path)
        files.models = [
            ModelFiles(
                rank=0,
                structure_path=path / MODEL_NAME,
                display_label=display_label,
                object_name=_safe_object_name(model_id),
                pae_path=pae_path if pae_path.is_file() else None,
                capabilities=frozenset(
                    {"plddt", "pae"} if pae_path.is_file() else {"plddt"}
                ),
            )
        ]
        return files

    def load_model_data(
        self,
        pred_files: PredictionFiles,
        model: ModelFiles,
        data: PredictionData,
        options: LoadOptions,
        *,
        structure_index,
    ) -> None:
        if options.load_pae and model.pae_path is not None:
            data.pae = _load_pae(model.pae_path)
