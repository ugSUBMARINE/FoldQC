"""
Token map
=========
Build viewer-independent prediction-token metadata from original structure
files so per-token arrays can be mapped onto a loaded structure by a viewer
adapter.

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
Molecular viewers may reorder HETATM atoms after loading a structure file.
Token indices are therefore derived exclusively from the original CIF/PDB file
and ligand tokens retain atom names for stable viewer-side lookup.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import overload

import numpy as np


@dataclass(frozen=True, slots=True)
class TokenInfo:
    """Viewer-independent identity and metadata for one prediction token."""

    token_idx: int  # 0-based prediction token index
    chain_id: str  # structure-file chain ID, e.g. "A"
    res_num: int  # residue sequence number from the structure file
    res_name: str  # residue or ligand name, e.g. "ALA", "SAH", "LIG"
    is_hetatm: bool  # True for ligand atoms, False for polymer residues

    # For HETATM tokens only: individual atom name (e.g. "C1", "N3").
    # None for polymer tokens (all atoms of the residue share the token).
    atom_name: str | None


TokenIdentity = tuple[str, int, str, str | None]
PolymerTokenKey = tuple[str, int]
HetatmTokenKey = tuple[str, int, str]


@dataclass(frozen=True, slots=True)
class TokenMap(Sequence[TokenInfo]):
    """Immutable prediction-token sequence with reusable derived metadata.

    Equality and hashing depend only on :attr:`tokens`. Hashing therefore
    scales with the number of tokens and is intended only for small, bounded
    caches rather than process-wide memoization.
    """

    tokens: tuple[TokenInfo, ...]
    chain_order: tuple[str, ...] = field(
        init=False, repr=False, compare=False, hash=False
    )
    chain_to_indices: Mapping[str, tuple[int, ...]] = field(
        init=False, repr=False, compare=False, hash=False
    )
    chain_id_to_index: Mapping[str, int] = field(
        init=False, repr=False, compare=False, hash=False
    )
    polymer_indices: tuple[int, ...] = field(
        init=False, repr=False, compare=False, hash=False
    )
    polymer_token_by_residue: Mapping[PolymerTokenKey, int] = field(
        init=False, repr=False, compare=False, hash=False
    )
    hetatm_token_by_atom: Mapping[HetatmTokenKey, int] = field(
        init=False, repr=False, compare=False, hash=False
    )
    token_identities: frozenset[TokenIdentity] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        tokens = tuple(self.tokens)
        object.__setattr__(self, "tokens", tokens)

        chain_order: list[str] = []
        chain_to_indices: dict[str, list[int]] = {}
        polymer_indices: list[int] = []
        polymer_tokens: dict[PolymerTokenKey, int] = {}
        hetatm_tokens: dict[HetatmTokenKey, int] = {}
        identities: set[TokenIdentity] = set()
        last_chain: str | None = None

        for position, token in enumerate(tokens):
            if token.token_idx != position:
                raise ValueError(
                    "TokenMap requires dense token indices matching tuple positions; "
                    f"position {position} contains token_idx {token.token_idx}."
                )

            chain_id = str(token.chain_id)
            residue_number = token.res_num
            residue_name = str(token.res_name)
            token_idx = int(token.token_idx)

            if chain_id != last_chain:
                chain_order.append(chain_id)
                last_chain = chain_id
            chain_to_indices.setdefault(chain_id, []).append(token_idx)

            if token.is_hetatm:
                atom_name = str(token.atom_name or "")
                hetatm_tokens[(chain_id, residue_number, atom_name)] = token_idx
                identity_atom_name: str | None = atom_name
            else:
                polymer_indices.append(token_idx)
                polymer_tokens[(chain_id, residue_number)] = token_idx
                identity_atom_name = None
            identities.add((chain_id, residue_number, residue_name, identity_atom_name))

        chain_indices = {
            chain_id: tuple(indices) for chain_id, indices in chain_to_indices.items()
        }
        chain_id_to_index = {
            chain_id: chain_idx for chain_idx, chain_id in enumerate(chain_order)
        }
        object.__setattr__(self, "chain_order", tuple(chain_order))
        object.__setattr__(self, "chain_to_indices", MappingProxyType(chain_indices))
        object.__setattr__(
            self, "chain_id_to_index", MappingProxyType(chain_id_to_index)
        )
        object.__setattr__(self, "polymer_indices", tuple(polymer_indices))
        object.__setattr__(
            self, "polymer_token_by_residue", MappingProxyType(polymer_tokens)
        )
        object.__setattr__(
            self, "hetatm_token_by_atom", MappingProxyType(hetatm_tokens)
        )
        object.__setattr__(self, "token_identities", frozenset(identities))

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[TokenInfo]:
        return iter(self.tokens)

    @overload
    def __getitem__(self, index: int) -> TokenInfo: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TokenInfo, ...]: ...

    def __getitem__(self, index: int | slice) -> TokenInfo | tuple[TokenInfo, ...]:
        return self.tokens[index]


@dataclass(frozen=True)
class TokenOverlapSummary:
    """Identity overlap between a prediction token map and a viewer object."""

    prediction_tokens: int
    target_tokens: int
    matched_prediction_tokens: int
    matched_target_tokens: int
    target_coverage: float
    prediction_coverage: float


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


def build_token_map(structure_path: str | Path) -> TokenMap:
    """Build an ordered, viewer-independent token map.

    Token indices are assigned by reading the CIF/PDB file in order, **not** by
    querying a molecular viewer. Viewers may reorder HETATM atoms internally,
    so using the original structure file is essential for correct ligand
    token-atom correspondence.

    Parameters
    ----------
    structure_path:
        Path to the original CIF/PDB prediction file.
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
            token_map.append(
                TokenInfo(
                    token_idx=token_idx,
                    chain_id=chain,
                    res_num=resi,
                    res_name=resn,
                    is_hetatm=True,
                    atom_name=name,
                )
            )
            token_idx += 1
        else:
            # Polymer residue: one token per unique (chain, resi, resn)
            key = (chain, resi, resn)
            if key not in seen_residues:
                seen_residues[key] = token_idx
                token_map.append(
                    TokenInfo(
                        token_idx=token_idx,
                        chain_id=chain,
                        res_num=resi,
                        res_name=resn,
                        is_hetatm=False,
                        atom_name=None,
                    )
                )
                token_idx += 1

    return TokenMap(tuple(token_map))
