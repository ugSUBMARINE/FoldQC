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
