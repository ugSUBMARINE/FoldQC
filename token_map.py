"""
Token map
=========
Build a bidirectional mapping between prediction token indices and PyMOL atom
selections so that per-token arrays (pLDDT, PAE rows, PDE rows, …) can be
painted onto a loaded structure.

Token definition
----------------
- Polymer chains (protein, DNA, RNA): **one token per residue**.
  All atoms belonging to a residue share that token index; the token’s
  representative coordinate is the Cα (protein) or C1′ (nucleotide).
- Ligand chains (HETATM): **one token per heavy atom**.

The token order follows the chain order in the CIF/PDB file, which in turn
reflects the provider's original output order.

Critical implementation note
-----------------------------
PyMOL **re-sorts HETATM atoms** after loading a structure file. This order
can differ from the prediction output's ligand atom order. Token indices are
therefore derived by reading the original CIF/PDB file directly; PyMOL is
only used to build selection strings — which match by atom *name* and are
therefore unaffected by the internal ordering.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SAFE_OBJECT_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_SAFE_SELECTOR_NAME = re.compile(r"^[A-Za-z0-9_']+$")


@dataclass
class TokenInfo:
    """Mapping from one prediction token to a PyMOL selection and metadata."""

    token_idx: int  # 0-based prediction token index
    chain_id: str  # PyMOL chain ID, e.g. "A"
    res_num: int  # residue sequence number from the structure file
    res_name: str  # residue or ligand name, e.g. "ALA", "SAH", "LIG"
    is_hetatm: bool  # True for ligand atoms, False for polymer residues

    # For HETATM tokens only: individual atom name (e.g. "C1", "N3").
    # None for polymer tokens (all atoms of the residue share the token).
    atom_name: str | None

    # PyMOL selection string that selects exactly the atom(s) for this token.
    # Polymer:  "chain A and resi 42"
    # Ligand:   "chain C and resi 1 and name C1"
    pymol_selection: str


@dataclass(frozen=True)
class TokenOverlapSummary:
    """Identity overlap between a prediction token map and a PyMOL object."""

    prediction_tokens: int
    target_tokens: int
    matched_prediction_tokens: int
    matched_target_tokens: int
    target_coverage: float
    prediction_coverage: float


def _compact_integer_ranges(values: Iterable[int]) -> str:
    """Return sorted integers as PyMOL ``resi`` ranges joined by ``+``."""
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return ""

    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return "+".join(ranges)


def _chain_selector(chain_id: str) -> str | None:
    """Return a compact, safe PyMOL chain selector value."""
    if not chain_id:
        return '""'
    if _SAFE_SELECTOR_NAME.fullmatch(chain_id):
        return chain_id
    return None


def compact_pymol_selection_expression(
    token_indices: Iterable[int],
    object_token_maps: Sequence[tuple[str, Sequence[Any]]],
) -> str:
    """Build a compact PyMOL expression for token indices across objects.

    Polymer residues are grouped into ``resi`` ranges per object and chain.
    HETATM tokens are grouped per object, chain, residue number, and residue
    name, with atom names combined into one comma-separated ``name`` selector.
    Tokens whose identifiers cannot be represented safely fall back to their
    existing exact ``pymol_selection`` strings.
    """
    indices = list(dict.fromkeys(int(index) for index in token_indices))
    clauses: list[str] = []
    seen_clauses: set[str] = set()

    def add_clause(clause: str) -> None:
        wrapped = f"({clause})"
        if clause and wrapped not in seen_clauses:
            seen_clauses.add(wrapped)
            clauses.append(wrapped)

    for object_name, token_map in object_token_maps:
        if object_name is None or not str(object_name):
            raise ValueError("Every token map must have a non-empty object name.")
        selected_indices = {
            token_idx for token_idx in indices if 0 <= token_idx < len(token_map)
        }
        all_polymer_indices: dict[str, set[int]] = {}
        all_ligand_indices: dict[tuple[str, int, str], set[int]] = {}
        polymer_residues: dict[str, set[int]] = {}
        ligand_atoms: dict[tuple[str, int, str], list[str]] = {}
        fallback_selections: list[str] = []
        compact_object = (
            str(object_name) if _SAFE_OBJECT_NAME.fullmatch(str(object_name)) else None
        )

        for token_idx, token in enumerate(token_map):
            if not hasattr(token, "chain_id") or not hasattr(token, "is_hetatm"):
                continue
            chain_id = str(token.chain_id)
            if not bool(token.is_hetatm):
                all_polymer_indices.setdefault(chain_id, set()).add(token_idx)
                continue
            if not hasattr(token, "res_num") or not hasattr(token, "res_name"):
                continue
            try:
                residue_number = int(token.res_num)
            except (TypeError, ValueError):
                continue
            key = (chain_id, residue_number, str(token.res_name or ""))
            all_ligand_indices.setdefault(key, set()).add(token_idx)

        for token_idx in indices:
            if token_idx not in selected_indices:
                continue
            token = token_map[token_idx]
            required_fields = (
                "token_idx",
                "chain_id",
                "res_num",
                "res_name",
                "is_hetatm",
                "atom_name",
                "pymol_selection",
            )
            missing_fields = [
                field for field in required_fields if not hasattr(token, field)
            ]
            if missing_fields:
                raise ValueError(
                    f"Token {token_idx} for object {object_name!r} is missing "
                    f"required TokenInfo fields: {', '.join(missing_fields)}."
                )

            fallback = getattr(token, "pymol_selection")
            if not isinstance(fallback, str) or not fallback:
                raise ValueError(
                    f"Token {token_idx} for object {object_name!r} has no valid "
                    "pymol_selection fallback."
                )
            chain_id = str(token.chain_id)
            chain_selector = _chain_selector(chain_id)

            try:
                residue_number = int(token.res_num)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Token {token_idx} for object {object_name!r} has an invalid "
                    f"res_num: {token.res_num!r}."
                ) from None

            if compact_object is None or chain_selector is None:
                fallback_selections.append(fallback)
                continue

            if not bool(token.is_hetatm):
                polymer_residues.setdefault(chain_id, set()).add(residue_number)
                continue

            residue_name = str(token.res_name or "")
            atom_name = str(token.atom_name or "")
            if not (
                _SAFE_SELECTOR_NAME.fullmatch(residue_name)
                and _SAFE_SELECTOR_NAME.fullmatch(atom_name)
            ):
                fallback_selections.append(fallback)
                continue
            ligand_atoms.setdefault(
                (chain_id, residue_number, residue_name), []
            ).append(atom_name)

        if compact_object is not None:
            for chain_id, residue_numbers in polymer_residues.items():
                all_chain_indices = all_polymer_indices.get(chain_id, set())
                if all_chain_indices and all_chain_indices <= selected_indices:
                    add_clause(
                        f"%{compact_object} and polymer and chain "
                        f"{_chain_selector(chain_id)}"
                    )
                    continue
                residue_selector = _compact_integer_ranges(residue_numbers)
                add_clause(
                    f"%{compact_object} and polymer and chain "
                    f"{_chain_selector(chain_id)} and resi {residue_selector}"
                )

            for (
                chain_id,
                residue_number,
                residue_name,
            ), atom_names in ligand_atoms.items():
                ligand_key = (chain_id, residue_number, residue_name)
                all_residue_indices = all_ligand_indices.get(ligand_key, set())
                if all_residue_indices and all_residue_indices <= selected_indices:
                    add_clause(
                        f"%{compact_object} and hetatm and chain "
                        f"{_chain_selector(chain_id)} "
                        f"and resi {residue_number} and resn {residue_name}"
                    )
                    continue
                unique_atom_names = list(dict.fromkeys(atom_names))
                add_clause(
                    f"%{compact_object} and hetatm and chain "
                    f"{_chain_selector(chain_id)} and resi {residue_number} "
                    f"and resn {residue_name} and name {','.join(unique_atom_names)}"
                )

        for selection in fallback_selections:
            add_clause(selection)

    return " or ".join(clauses)


def _parse_cif_atoms(cif_path: str | Path) -> list[dict]:
    """Read every atom record from an mmCIF file in file order.

    Returns a list of dicts with keys ``hetatm``, ``name``, ``resn``,
    ``resi`` (int), ``chain``, and ``b`` in the exact order they appear in
    the ``_atom_site`` loop.  Column positions are discovered from the loop
    headers so the function is robust to future column reordering.
    """
    col: dict[str, int] = {}  # header tag → zero-based column index
    atoms: list[dict] = []
    in_atom_site = False
    header_count = 0

    with Path(cif_path).open() as fh:
        for raw in fh:
            stripped = raw.strip()

            # A bare loop_ token resets the current loop
            if stripped == "loop_":
                col.clear()
                header_count = 0
                in_atom_site = False
                continue

            # Collect _atom_site column headers
            if stripped.startswith("_atom_site."):
                tag = stripped.split(".", 1)[1].split()[0]
                col[tag] = header_count
                header_count += 1
                in_atom_site = True
                continue

            if not in_atom_site or not col:
                continue

            # End of the _atom_site block
            if (
                stripped.startswith("_")
                or stripped == "loop_"
                or stripped.startswith("#")
            ):
                in_atom_site = False
                continue

            if not stripped:
                continue

            # Only ATOM / HETATM data lines
            if not (stripped.startswith("ATOM") or stripped.startswith("HETATM")):
                continue

            try:
                parts = shlex.split(stripped, comments=False, posix=True)
            except ValueError:
                continue
            if len(parts) <= max(col.values()):
                continue  # malformed line, skip

            resn_col = col.get("auth_comp_id", col.get("label_comp_id"))
            if resn_col is None:
                continue
            b_col = col.get("B_iso_or_equiv")
            b_factor = float(parts[b_col]) if b_col is not None else np.nan

            atoms.append(
                {
                    "hetatm": parts[col["group_PDB"]] == "HETATM",
                    "name": parts[col["label_atom_id"]],
                    "resn": parts[resn_col],
                    "resi": int(parts[col["auth_seq_id"]]),
                    "chain": parts[col["auth_asym_id"]],
                    "b": b_factor,
                }
            )

    return atoms


def _parse_pdb_atoms(pdb_path: str | Path) -> list[dict]:
    """Read every ATOM/HETATM record from a PDB file in file order."""
    atoms: list[dict] = []
    with Path(pdb_path).open() as fh:
        for raw in fh:
            record = raw[0:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue

            altloc = raw[16:17].strip()
            if altloc not in {"", "A"}:
                continue

            try:
                resi = int(raw[22:26].strip())
            except ValueError:
                continue

            try:
                b_factor = float(raw[60:66].strip())
            except ValueError:
                b_factor = np.nan

            atoms.append(
                {
                    "hetatm": record == "HETATM",
                    "name": raw[12:16].strip(),
                    "resn": raw[17:20].strip(),
                    "resi": resi,
                    "chain": raw[21:22].strip(),
                    "b": b_factor,
                }
            )

    return atoms


def parse_structure_atoms(structure_path: str | Path) -> list[dict]:
    """Read ATOM/HETATM records from a CIF or PDB file in file order."""
    path = Path(structure_path)
    suffix = path.suffix.lower()
    if suffix == ".cif":
        return _parse_cif_atoms(path)
    if suffix == ".pdb":
        return _parse_pdb_atoms(path)
    raise ValueError(f"Unsupported structure format: {path.suffix}")


def extract_structure_plddt(structure_path: str | Path) -> np.ndarray:
    """Extract per-token pLDDT from structure B-factors on a 0-1 scale.

    Prediction structures are expected to write pLDDT × 100 into the B-factor
    column.  This function parses the original CIF/PDB file and collapses atom
    records to the same token order used by :func:`build_token_map`: one token
    per polymer residue and one token per HETATM atom.
    """
    atoms = parse_structure_atoms(structure_path)
    if not atoms:
        raise ValueError(f"No atom records found in {structure_path}")

    values: list[float] = []
    seen_residues: set[tuple[str, int, str]] = set()
    for atom in atoms:
        if atom["hetatm"]:
            values.append(float(atom["b"]) / 100.0)
            continue

        key = (atom["chain"], atom["resi"], atom["resn"])
        if key not in seen_residues:
            seen_residues.add(key)
            values.append(float(atom["b"]) / 100.0)

    return np.array(values, dtype=np.float32)


def build_token_map(obj_name: str, structure_path: str | Path) -> list[TokenInfo]:
    """Build the ordered token map for a loaded PyMOL object.

    Token indices are assigned by reading the CIF/PDB file in order, **not** by
    iterating ``cmd.get_model()``.  PyMOL alphabetises HETATM atoms internally,
    so using the structure file is essential for correct token-atom
    correspondence in ligands.

    PyMOL is not queried while building the map; *obj_name* is only interpolated
    into selection strings. Those selections are built from structure-file atom
    names and therefore match the correct atoms regardless of PyMOL's internal
    ordering.

    Parameters
    ----------
    obj_name:
        PyMOL object name — used solely to build selection strings.
    structure_path:
        Path to the original CIF/PDB file that was loaded into *obj_name*.
    """
    structure_atoms = parse_structure_atoms(structure_path)
    if not structure_atoms:
        raise ValueError(f"No atom records found in {structure_path}")

    token_map: list[TokenInfo] = []
    token_idx = 0
    seen_residues: dict[
        tuple[str, int, str], int
    ] = {}  # (chain, resi, resn) → token_idx

    for atom in structure_atoms:
        chain = atom["chain"]
        resi = atom["resi"]
        resn = atom["resn"]
        name = atom["name"]

        if atom["hetatm"]:
            # Ligand heavy atom: one token per atom in original file order
            sel = f"/{obj_name}//{chain}/{resi}/{name}"
            token_map.append(
                TokenInfo(
                    token_idx=token_idx,
                    chain_id=chain,
                    res_num=resi,
                    res_name=resn,
                    is_hetatm=True,
                    atom_name=name,
                    pymol_selection=sel,
                )
            )
            token_idx += 1
        else:
            # Polymer residue: one token per unique (chain, resi, resn)
            key = (chain, resi, resn)
            if key not in seen_residues:
                seen_residues[key] = token_idx
                sel = f"/{obj_name}//{chain}/{resi}/"
                token_map.append(
                    TokenInfo(
                        token_idx=token_idx,
                        chain_id=chain,
                        res_num=resi,
                        res_name=resn,
                        is_hetatm=False,
                        atom_name=None,
                        pymol_selection=sel,
                    )
                )
                token_idx += 1

    return token_map


def _token_identity(token: TokenInfo) -> tuple[str, int, str, str | None]:
    """Return a residue/atom identity for comparing token maps to PyMOL models."""
    if bool(token.is_hetatm):
        return (
            str(token.chain_id),
            int(token.res_num),
            str(token.res_name),
            str(token.atom_name or ""),
        )
    return (str(token.chain_id), int(token.res_num), str(token.res_name), None)


def _model_token_identities(model) -> set[tuple[str, int, str, str | None]]:
    """Return token identities represented by a PyMOL model object."""
    identities: set[tuple[str, int, str, str | None]] = set()
    for atom in getattr(model, "atom", []) or []:
        try:
            resi = int(atom.resi)
        except (TypeError, ValueError):
            continue
        chain = str(getattr(atom, "chain", ""))
        resn = str(getattr(atom, "resn", ""))
        if bool(getattr(atom, "hetatm", False)):
            identities.add((chain, resi, resn, str(getattr(atom, "name", ""))))
        else:
            identities.add((chain, resi, resn, None))
    return identities


def compare_token_map_to_pymol_object(
    token_map: list[TokenInfo],
    obj_name: str,
) -> TokenOverlapSummary:
    """Compare prediction token identities with identities present in *obj_name*.

    Polymer tokens are compared by ``chain + resi + resn``. HETATM tokens also
    include atom name. PyMOL is imported lazily so this helper remains safe to
    import outside PyMOL.
    """
    from pymol import cmd  # lazy import

    prediction_identities = {_token_identity(token) for token in token_map}
    model = cmd.get_model(obj_name)
    target_identities = _model_token_identities(model)
    matched = prediction_identities & target_identities
    prediction_total = len(prediction_identities)
    target_total = len(target_identities)
    matched_total = len(matched)
    return TokenOverlapSummary(
        prediction_tokens=prediction_total,
        target_tokens=target_total,
        matched_prediction_tokens=matched_total,
        matched_target_tokens=matched_total,
        target_coverage=(matched_total / target_total if target_total else 1.0),
        prediction_coverage=(
            matched_total / prediction_total if prediction_total else 1.0
        ),
    )


def selection_to_token_indices(
    token_map: list[TokenInfo],
    pymol_selection: str,
    obj_name: str = "all",
) -> list[int]:
    """Return the token indices covered by a PyMOL selection string.

    Parameters
    ----------
    token_map:
        Built by :func:`build_token_map`.
    pymol_selection:
        Any valid PyMOL selection expression, e.g. ``"chain C"`` or
        ``"byres (chain A within 5 of chain C)"``.
    obj_name:
        Object to restrict the selection to (default ``"all"``).

    Returns
    -------
    list[int]
        Sorted list of matching token indices.
    """
    from pymol import cmd  # lazy import

    # Resolve the selection to a set of (chain, resi, name) tuples
    model = cmd.get_model(f"({pymol_selection}) and {obj_name}")
    if model is None:
        return []

    # Build lookup sets for fast intersection
    polymer_residues: set[tuple[str, int]] = set()
    hetatm_atoms: set[tuple[str, int, str]] = set()

    for atom in model.atom:
        chain = atom.chain
        resi = int(atom.resi)
        if atom.hetatm:
            hetatm_atoms.add((chain, resi, atom.name))
        else:
            polymer_residues.add((chain, resi))

    result: list[int] = []
    for tok in token_map:
        if tok.is_hetatm:
            if (tok.chain_id, tok.res_num, tok.atom_name) in hetatm_atoms:
                result.append(tok.token_idx)
        else:
            if (tok.chain_id, tok.res_num) in polymer_residues:
                result.append(tok.token_idx)

    return sorted(result)
