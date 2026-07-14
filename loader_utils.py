"""Shared loader helpers independent of concrete providers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

STRUCTURE_SUFFIXES = {".cif", ".pdb"}
ARCHIVE_EXTENSIONS = (".tar.gz", ".zip", ".tgz", ".tar")


def _float_or_none(value) -> float | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _squeezed_float32_array(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    return array


def _normalise_confidence(confidence: dict | None) -> dict | None:
    if confidence is None:
        return None
    normalised = dict(confidence)

    if "structure_confidence" in normalised and "confidence_score" not in normalised:
        normalised["confidence_score"] = normalised["structure_confidence"]

    if "chain_ptm" in normalised and "chains_ptm" not in normalised:
        chain_ptm = normalised["chain_ptm"]
        if isinstance(chain_ptm, list):
            normalised["chains_ptm"] = {
                str(idx): value for idx, value in enumerate(chain_ptm)
            }

    if "chain_iptm" in normalised and "chains_iptm" not in normalised:
        chain_iptm = normalised["chain_iptm"]
        if isinstance(chain_iptm, list):
            normalised["chains_iptm"] = {
                str(idx): value for idx, value in enumerate(chain_iptm)
            }

    if "chain_pair_iptm" in normalised and "pair_chains_iptm" not in normalised:
        matrix = normalised["chain_pair_iptm"]
        if isinstance(matrix, list):
            normalised["pair_chains_iptm"] = _matrix_list_to_nested_dict(matrix)

    if "chain_pair_pae_min" in normalised and "pair_chains_pae_min" not in normalised:
        matrix = normalised["chain_pair_pae_min"]
        if isinstance(matrix, list):
            normalised["pair_chains_pae_min"] = _matrix_list_to_nested_dict(matrix)

    return normalised


def _matrix_list_to_nested_dict(matrix: list) -> dict[str, dict[str, float]]:
    return {
        str(i): {str(j): value for j, value in enumerate(row)}
        for i, row in enumerate(matrix)
        if isinstance(row, list)
    }


def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _first(paths) -> Path | None:
    return next(iter(sorted(paths)), None)


def _safe_object_name(name: str) -> str:
    safe = re.sub(r"\W+", "_", name).strip("_")
    return safe or "prediction"
