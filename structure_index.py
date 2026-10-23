"""One-pass, viewer-independent indexing of prediction structures.

Frozen records omit automatic slots for Python 3.10+ compatibility; see the
dataclass compatibility note in :mod:`token_map`.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import numpy as np

from .token_map import ResidueId, TokenInfo, TokenMap, is_hydrogen

StructureFormat = Literal["cif", "pdb"]


@dataclass(frozen=True)
class _AtomRecord:
    hetatm: bool
    name: str
    res_name: str
    residue_id: ResidueId
    chain_id: str
    b_factor: float
    element: str
    xyz: tuple[float, float, float] = (np.nan, np.nan, np.nan)


PROTEIN_RESIDUES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
    "ASX GLX SEC PYL UNK MSE".split()
)


@dataclass(frozen=True, eq=False)
class StructureGeometry:
    """Original representative atoms in canonical token order (never viewer data)."""

    ca: np.ndarray
    cb: np.ndarray
    ca_plddt: np.ndarray
    cb_plddt: np.ndarray
    protein_mask: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.protein_mask)
        for name in ("ca", "cb", "ca_plddt", "cb_plddt", "protein_mask"):
            value = np.array(
                getattr(self, name),
                dtype=bool if name == "protein_mask" else np.float64,
                order="C",
                copy=True,
            )
            shape = (n, 3) if name in {"ca", "cb"} else (n,)
            if value.shape != shape or (
                name != "protein_mask" and np.isinf(value).any()
            ):
                raise ValueError(f"Invalid structure geometry {name}.")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def protein_token_indices(
    token_map: TokenMap, geometry: StructureGeometry | None = None
) -> tuple[int, ...]:
    """Protein polymer tokens only; HETATM atom tokens are never retokenized."""
    return tuple(
        token.token_idx
        for token in token_map
        if not token.is_hetatm
        and (
            token.res_name.upper() in PROTEIN_RESIDUES
            or (geometry is not None and geometry.protein_mask[token.token_idx])
        )
    )


def _coordinate(
    parts: list[str], columns: tuple[int | None, ...]
) -> tuple[float, float, float]:
    try:
        values = tuple(float(parts[c]) if c is not None else np.nan for c in columns)
        return (values[0], values[1], values[2])
    except (ValueError, IndexError):
        return (np.nan, np.nan, np.nan)


def _geometry(
    atoms: list[_AtomRecord], token_map: TokenMap, mapping: tuple[int | None, ...]
) -> StructureGeometry:
    n = len(token_map)
    ca = np.full((n, 3), np.nan)
    cb = np.full((n, 3), np.nan)
    ca_conf = np.full(n, np.nan)
    cb_conf = np.full(n, np.nan)
    names: dict[int, set[str]] = {}
    representatives: dict[tuple[int, str], list[_AtomRecord]] = {}
    for atom, index in zip(atoms, mapping, strict=True):
        if index is None or token_map[index].is_hetatm:
            continue
        names.setdefault(index, set()).add(atom.name)
        if atom.name in {"CA", "CB"}:
            representatives.setdefault((index, atom.name), []).append(atom)
    mask = np.array(
        [
            not token.is_hetatm
            and (
                token.res_name.upper() in PROTEIN_RESIDUES
                or {"N", "CA", "C"} <= names.get(token.token_idx, set())
            )
            for token in token_map
        ]
    )
    for (index, name), matches in representatives.items():
        # Ambiguous representatives remain missing, rather than choosing an altloc.
        if len(matches) != 1:
            continue
        atom = matches[0]
        coords, confidence = (ca, ca_conf) if name == "CA" else (cb, cb_conf)
        coords[index] = atom.xyz
        confidence[index] = atom.b_factor
    # FoldQC accepts percentage or fractional structure confidence. Normalize once,
    # consistently with structure_plddt, then expose the published 0–100 scale.
    for values in (ca_conf, cb_conf):
        fractional = np.isfinite(values) & (values <= 1.5)
        values[fractional] *= 100.0
        values[(values < 0) | (values > 100)] = np.nan
    for token in token_map:
        if token.res_name.upper() == "GLY":
            cb[token.token_idx] = ca[token.token_idx]
            cb_conf[token.token_idx] = ca_conf[token.token_idx]
    return StructureGeometry(ca, cb, ca_conf, cb_conf, mask)


@dataclass(frozen=True)
class StructureIndex:
    """Reusable token and atom metadata parsed from one structure-file read."""

    path: Path
    format: StructureFormat
    token_map: TokenMap
    atom_count: int
    atom_to_token: tuple[int | None, ...]
    structure_plddt: np.ndarray
    geometry: StructureGeometry | None = None

    def __post_init__(self) -> None:
        path = Path(self.path)
        atom_to_token = tuple(
            None if index is None else int(index) for index in self.atom_to_token
        )
        values = np.array(self.structure_plddt, dtype=np.float32, copy=True)
        if self.format not in {"cif", "pdb"}:
            raise ValueError(f"Unsupported structure format: {self.format}")
        if self.atom_count < 0 or len(atom_to_token) != self.atom_count:
            raise ValueError(
                "StructureIndex atom-to-token mapping must match its atom count."
            )
        if values.ndim != 1 or len(values) != len(self.token_map):
            raise ValueError(
                "StructureIndex structure pLDDT must contain one value per token."
            )
        if self.geometry is not None and len(self.geometry.protein_mask) != len(
            self.token_map
        ):
            raise ValueError("Structure geometry must match the canonical token map.")
        if np.isinf(values).any():
            raise ValueError(
                "StructureIndex structure pLDDT must not contain infinity."
            )
        if any(
            index is not None and (index < 0 or index >= len(self.token_map))
            for index in atom_to_token
        ):
            raise ValueError("StructureIndex atom-to-token mapping is out of range.")
        values.setflags(write=False)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "atom_to_token", atom_to_token)
        object.__setattr__(self, "structure_plddt", values)

    @classmethod
    def from_path(cls, structure_path: str | Path) -> StructureIndex:
        """Read and index a CIF or PDB structure exactly once."""
        path = Path(structure_path)
        suffix = path.suffix.lower()
        if suffix == ".cif":
            structure_format: StructureFormat = "cif"
            parser = _parse_cif_atoms
        elif suffix == ".pdb":
            structure_format = "pdb"
            parser = _parse_pdb_atoms
        else:
            raise ValueError(f"Unsupported structure format: {path.suffix}")

        try:
            with path.open(encoding="utf-8") as handle:
                atoms = parser(handle)
        except ValueError as exc:
            raise ValueError(f"Failed to parse {path}: {exc}") from exc
        if not atoms:
            raise ValueError(f"No atom records found in {path}")

        token_map, atom_to_token, structure_plddt = _index_atoms(atoms)
        if not token_map:
            raise ValueError(
                f"Structure {path} contains atom records but no supported tokens."
            )
        return cls(
            path=path,
            format=structure_format,
            token_map=token_map,
            atom_count=len(atoms),
            atom_to_token=atom_to_token,
            structure_plddt=structure_plddt,
            geometry=_geometry(atoms, token_map, atom_to_token),
        )

    def collapse_atom_plddt(self, atom_plddt: np.ndarray) -> np.ndarray:
        """Average atom confidence into this index's prediction-token order."""
        values = np.asarray(atom_plddt, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(
                f"Atom pLDDT array for {self.path.name} must be one-dimensional; "
                f"got shape {values.shape}."
            )
        if len(values) != self.atom_count:
            raise ValueError(
                f"Atom pLDDT length {len(values)} does not match "
                f"{self.atom_count} atoms in {self.path.name}."
            )
        if np.isinf(values).any():
            raise ValueError(
                f"Atom pLDDT array for {self.path.name} must not contain infinity."
            )
        return _collapse_atom_values(values, self.atom_to_token, len(self.token_map))

    def matches_path(self, structure_path: str | Path) -> bool:
        """Return whether *structure_path* identifies this indexed file."""
        return self.path.resolve() == Path(structure_path).resolve()


def _parse_cif_atoms(handle: TextIO) -> list[_AtomRecord]:
    col: dict[str, int] = {}
    atoms: list[_AtomRecord] = []
    in_atom_site = False
    header_count = 0

    for raw in handle:
        stripped = raw.strip()
        if stripped == "loop_":
            col.clear()
            header_count = 0
            in_atom_site = False
            continue
        if stripped.startswith("_atom_site."):
            tag = stripped.split(".", 1)[1].split()[0]
            col[tag] = header_count
            header_count += 1
            in_atom_site = True
            continue
        if not in_atom_site or not col:
            continue
        if stripped.startswith("_") or stripped == "loop_" or stripped.startswith("#"):
            in_atom_site = False
            continue
        if not stripped or not stripped.startswith(("ATOM", "HETATM")):
            continue

        try:
            parts = shlex.split(stripped, comments=False, posix=True)
        except ValueError:
            continue
        res_name_col = col.get("auth_comp_id", col.get("label_comp_id"))
        if res_name_col is None:
            continue
        # Some Boltz exports omit trailing, unused fields such as the model
        # number. Keep these atom rows when their identity fields are present;
        # missing optional coordinates/confidence remain explicitly undefined.
        required_columns = (
            col["group_PDB"],
            col["label_atom_id"],
            col["auth_seq_id"],
            col["auth_asym_id"],
            res_name_col,
        )
        if len(parts) <= max(required_columns):
            continue
        b_col = col.get("B_iso_or_equiv")
        try:
            b_factor = (
                float(parts[b_col])
                if b_col is not None and b_col < len(parts)
                else np.nan
            )
        except ValueError:
            b_factor = np.nan
        insertion_col = col.get("pdbx_PDB_ins_code")
        insertion_code = (
            parts[insertion_col]
            if insertion_col is not None and insertion_col < len(parts)
            else ""
        )
        element_col = col.get("type_symbol")
        atoms.append(
            _AtomRecord(
                hetatm=parts[col["group_PDB"]] == "HETATM",
                name=parts[col["label_atom_id"]],
                res_name=parts[res_name_col],
                residue_id=ResidueId(int(parts[col["auth_seq_id"]]), insertion_code),
                chain_id=parts[col["auth_asym_id"]],
                b_factor=b_factor,
                element=parts[element_col]
                if element_col is not None and element_col < len(parts)
                else "",
                xyz=_coordinate(
                    parts, (col.get("Cartn_x"), col.get("Cartn_y"), col.get("Cartn_z"))
                ),
            )
        )
    return atoms


def _parse_pdb_atoms(handle: TextIO) -> list[_AtomRecord]:
    atoms: list[_AtomRecord] = []
    for raw in handle:
        record = raw[0:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        if raw[16:17].strip() not in {"", "A"}:
            continue
        try:
            res_num = int(raw[22:26].strip())
        except ValueError:
            continue
        try:
            b_factor = float(raw[60:66].strip())
        except ValueError:
            b_factor = np.nan
        atoms.append(
            _AtomRecord(
                hetatm=record == "HETATM",
                name=raw[12:16].strip(),
                res_name=raw[17:20].strip(),
                residue_id=ResidueId(res_num, raw[26:27]),
                chain_id=raw[21:22].strip(),
                b_factor=b_factor,
                element=raw[76:78].strip(),
                xyz=_coordinate([raw[30:38], raw[38:46], raw[46:54]], (0, 1, 2)),
            )
        )
    return atoms


def _index_atoms(
    atoms: list[_AtomRecord],
) -> tuple[TokenMap, tuple[int | None, ...], np.ndarray]:
    tokens: list[TokenInfo] = []
    atom_to_token: list[int | None] = []
    polymer_tokens: dict[tuple[str, ResidueId], tuple[int, str]] = {}
    hetatm_tokens: set[tuple[str, ResidueId, str, str]] = set()
    residue_names: dict[tuple[str, ResidueId], str] = {}

    for atom in atoms:
        residue_key = (atom.chain_id, atom.residue_id)
        existing_name = residue_names.get(residue_key)
        if existing_name is not None and existing_name != atom.res_name:
            raise ValueError(
                "Conflicting residue names for "
                f"{atom.chain_id}/{atom.residue_id}: "
                f"{existing_name!r} and {atom.res_name!r}."
            )
        residue_names[residue_key] = atom.res_name
        if atom.hetatm:
            if is_hydrogen(atom.element, atom.name):
                atom_to_token.append(None)
                continue
            identity = (
                atom.chain_id,
                atom.residue_id,
                atom.res_name,
                atom.name,
            )
            if identity in hetatm_tokens:
                raise ValueError(f"Duplicate HETATM token identity: {identity!r}")
            hetatm_tokens.add(identity)
            token_idx = len(tokens)
            tokens.append(
                TokenInfo(
                    token_idx=token_idx,
                    chain_id=atom.chain_id,
                    residue_id=atom.residue_id,
                    res_name=atom.res_name,
                    is_hetatm=True,
                    atom_name=atom.name,
                )
            )
        else:
            key = residue_key
            existing = polymer_tokens.get(key)
            if existing is None:
                token_idx = len(tokens)
                polymer_tokens[key] = (token_idx, atom.res_name)
                tokens.append(
                    TokenInfo(
                        token_idx=token_idx,
                        chain_id=atom.chain_id,
                        residue_id=atom.residue_id,
                        res_name=atom.res_name,
                        is_hetatm=False,
                        atom_name=None,
                    )
                )
            else:
                token_idx = existing[0]
        atom_to_token.append(token_idx)

    b_factors = np.asarray([atom.b_factor for atom in atoms], dtype=np.float32)
    if np.isinf(b_factors).any():
        raise ValueError("Structure B-factors must not contain infinity.")
    structure_plddt = _collapse_atom_values(
        b_factors, tuple(atom_to_token), len(tokens)
    )
    return (
        TokenMap(tuple(tokens)),
        tuple(atom_to_token),
        structure_plddt,
    )


def _collapse_atom_values(
    atom_values: np.ndarray,
    atom_to_token: tuple[int | None, ...],
    token_count: int,
) -> np.ndarray:
    """Normalize and average finite atom values into canonical token order."""
    values = np.asarray(atom_values, dtype=np.float32)
    percentage = np.isfinite(values) & (values > 1.5)
    if percentage.any():
        values = values.copy()
        values[percentage] /= 100.0

    grouped: list[list[float]] = [[] for _ in range(token_count)]
    for token_idx, value in zip(atom_to_token, values, strict=True):
        if token_idx is not None and np.isfinite(value):
            grouped[token_idx].append(float(value))
    return np.asarray(
        [float(np.mean(group)) if group else float("nan") for group in grouped],
        dtype=np.float32,
    )
