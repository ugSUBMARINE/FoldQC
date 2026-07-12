"""Shared loader helpers independent of concrete providers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .token_map import parse_structure_atoms

STRUCTURE_SUFFIXES = {".cif", ".pdb"}
ARCHIVE_EXTENSIONS = (".tar.gz", ".zip", ".tgz", ".tar")


def _collapse_atom_plddts_to_tokens(
    structure_path: Path,
    atom_plddts: np.ndarray,
) -> np.ndarray:

    atoms = parse_structure_atoms(structure_path)
    if len(atoms) != len(atom_plddts):
        raise ValueError(
            f"AF3 atom_plddts length {len(atom_plddts)} does not match "
            f"{len(atoms)} atoms in {structure_path.name}."
        )

    values = np.asarray(atom_plddts, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmax(finite)) > 1.5:
        values = values / 100.0

    # Group values by residue, then average over residues. This is necessary because
    # AF3 atom_plddts are per-atom, but we want per-residue
    residue_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for atom, value in zip(atoms, values, strict=True):
        # skip heterocomponents since they always have one token per atom
        if not atom["hetatm"]:
            key = (atom["chain"], atom["resi"], atom["resn"])
            residue_values[key].append(float(value))

    # Rebuild in structure-token order rather than "all residues then ligands".
    out: list[float] = []
    emitted: set[tuple[str, int, str]] = set()
    for atom, value in zip(atoms, values, strict=True):
        if atom["hetatm"]:
            out.append(float(value))
        else:
            key = (atom["chain"], atom["resi"], atom["resn"])
            if key not in emitted:
                emitted.add(key)
                out.append(float(np.nanmean(residue_values[key])))

    return np.asarray(out, dtype=np.float32)


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
