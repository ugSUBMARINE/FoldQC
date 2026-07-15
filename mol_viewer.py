"""PyMOL implementation of FoldQC's molecular-viewer boundary.

All viewer commands and PyMOL selection syntax live in this module. Other
modules exchange viewer-independent token metadata and call the focused
functions below. PyMOL imports stay lazy so the package remains importable in
plain Python and in tests with a fake viewer.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from .palettes import (
    BUILTIN_PALETTE_KEYS,
    PALETTE_SPECS,
    PLDDT_CLASS_COLORS,
    categorical_color,
)
from .token_map import ResidueId, TokenMap, TokenOverlapSummary, is_hydrogen

T = TypeVar("T")

BUILTIN_PALETTES: list[str] = list(BUILTIN_PALETTE_KEYS)
NAN_COLOR_DEFAULT = "grey70"
COLORBAR_OBJECT_NAME = "foldqc_colorbar"

_SAFE_OBJECT_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_SAFE_SELECTOR_NAME = re.compile(r"^[A-Za-z0-9_']+$")

_PYMOL_PALETTES: dict[str, tuple[str, str | None]] = {
    "white_blue": ("white_blue", "blue_white"),
    "white_red": ("white_red", "red_white"),
    "white_green": ("white_green", "green_white"),
    "blue_white_red": ("blue_white_red", "red_white_blue"),
    "green_white_red": ("green_white_red", "red_white_green"),
    "cyan_white_magenta": ("cyan_white_magenta", "magenta_white_cyan"),
    "yellow_white_magenta": ("yellow_white_magenta", "magenta_white_yellow"),
    "rainbow": ("rainbow", "rainbow_rev"),
    "rainbow2": ("rainbow2", "rainbow2_rev"),
}
_PALETTE_SPECS_BY_KEY = {spec.key: spec for spec in PALETTE_SPECS}


@dataclass(frozen=True)
class _ColorDef:
    name: str
    rgb: tuple[float, float, float]


@dataclass(frozen=True)
class _PaletteResolution:
    palette: str
    custom_colors: tuple[_ColorDef, ...] = ()


@dataclass(frozen=True)
class ObjectPaintMapping:
    """Stable mapping from one PyMOL object's atom indices to prediction tokens."""

    obj_name: str
    atom_index_fingerprint: tuple[int, ...]
    atom_token_indices: np.ndarray
    atom_count: int
    max_atom_index: int
    overlap: TokenOverlapSummary


@dataclass(frozen=True)
class ObjectTokenInspection:
    """One object snapshot reused for painting metadata and representative coordinates."""

    paint_mapping: ObjectPaintMapping
    representative_coords: np.ndarray


@dataclass(frozen=True)
class PaintTarget:
    """One object and per-token array participating in a paint operation."""

    obj_name: str
    token_map: TokenMap
    values: np.ndarray
    mapping: ObjectPaintMapping | None = None


@dataclass(frozen=True)
class PaintBatchResult:
    """Resolved range and mappings from a batch paint operation."""

    vmin: float
    vmax: float
    mappings: tuple[ObjectPaintMapping, ...]


def get_viewer_name() -> str:
    """Return the user-facing name of the active molecular viewer."""
    return "PyMOL"


def get_selection_examples() -> dict[str, str]:
    """Return backend-specific selection examples for viewer-facing help."""
    return {
        "general": '"chain C" or "resname LIG"',
        "ligand": '"resname LIG" or "organic"',
        "chain": '"chain B"',
    }


def get_object_list(
    *,
    additional_names: Iterable[str] = (),
    excluded_names: Iterable[str] = (COLORBAR_OBJECT_NAME,),
) -> list[str]:
    """Return structure object names, optionally including known group names."""
    from pymol import cmd

    excluded = set(excluded_names)
    names = [
        str(name)
        for name in (cmd.get_names("objects") or [])
        if str(name) not in excluded
    ]
    try:
        all_names = set(cmd.get_names("all") or [])
    except Exception:
        all_names = set()
    for name in additional_names:
        name = str(name)
        if name and name not in excluded and name not in names:
            if not all_names or name in all_names:
                names.insert(0, name)
    return names


def is_object_enabled(obj_name: str) -> bool:
    """Return whether an object is enabled in the viewer."""
    from pymol import cmd

    try:
        enabled = set(cmd.get_names("objects", enabled_only=1) or [])
    except TypeError:
        enabled = set(cmd.get_names("objects", 1) or [])
    except Exception:
        return True
    return obj_name in enabled


def enable_object(obj_name: str) -> None:
    from pymol import cmd

    cmd.enable(obj_name)


def disable_object(obj_name: str) -> None:
    from pymol import cmd

    cmd.disable(obj_name)


def ensure_structure_object(
    path: str | Path,
    obj_name: str,
    *,
    zoom: bool = True,
) -> bool:
    """Load or enable one structure object; return true when it was loaded."""
    from pymol import cmd

    current_objects = set(cmd.get_names("objects") or [])
    if obj_name in current_objects:
        if not is_object_enabled(obj_name):
            cmd.enable(obj_name)
        return False
    cmd.load(str(path), obj_name, quiet=1, zoom=int(bool(zoom)))
    return True


def load_structure_object_if_missing(path: str | Path, obj_name: str) -> bool:
    """Load one object without changing the state of an existing object."""
    from pymol import cmd

    current_objects = set(cmd.get_names("objects") or [])
    if obj_name in current_objects:
        return False
    cmd.load(str(path), obj_name, quiet=1, zoom=0)
    return True


def viewer_name_exists(name: str) -> bool:
    """Return whether *name* currently exists as an object, group, or selection."""
    from pymol import cmd

    try:
        return name in set(cmd.get_names("all") or [])
    except Exception:
        return name in set(cmd.get_names("objects") or [])


def get_group_members(group_name: str) -> tuple[str, ...]:
    """Return object members of a viewer group, or an empty tuple."""
    from pymol import cmd

    if not viewer_name_exists(group_name):
        return ()
    return tuple(str(name) for name in (cmd.get_object_list(f"({group_name})") or []))


def add_objects_to_group(group_name: str, object_names: Iterable[str]) -> None:
    """Add objects to one viewer group."""
    from pymol import cmd

    for obj_name in object_names:
        cmd.group(group_name, str(obj_name), "add")


def remove_objects_from_group(group_name: str, object_names: Iterable[str]) -> None:
    """Remove objects from one viewer group when it still exists."""
    from pymol import cmd

    if not viewer_name_exists(group_name):
        return
    for obj_name in object_names:
        cmd.group(group_name, str(obj_name), "remove")


def delete_viewer_names(names: Iterable[str]) -> None:
    """Delete existing viewer names."""
    from pymol import cmd

    for name in names:
        if viewer_name_exists(str(name)):
            cmd.delete(str(name))


def run_with_updates_suspended(func: Callable[[], T]) -> T:
    """Run a callback with viewport updates suspended and rebuild afterward."""
    from pymol import cmd

    try:
        cmd.set("suspend_updates", "on")
    except Exception:
        pass
    try:
        return func()
    finally:
        try:
            cmd.set("suspend_updates", "off")
            cmd.rebuild()
        except Exception:
            pass


def snapshot_atom_visuals(obj_name: str):
    """Capture the atom-index fingerprint, B-factors, and color indices."""
    from pymol import cmd

    from .gui_services import AtomVisualSnapshot

    model = cmd.get_model(obj_name)
    atoms = tuple(getattr(model, "atom", ()) or ())
    return AtomVisualSnapshot(
        obj_name=str(obj_name),
        atom_indices=tuple(int(atom.index) for atom in atoms),
        b_factors=np.asarray(
            [float(getattr(atom, "b", 0.0)) for atom in atoms], dtype=np.float32
        ),
        color_indices=np.asarray(
            [int(getattr(atom, "color", 0)) for atom in atoms], dtype=np.int32
        ),
    )


def restore_atom_visuals(snapshot) -> None:
    """Restore a snapshot, rejecting objects whose atom identity changed."""
    from pymol import cmd

    current = cmd.get_model(snapshot.obj_name)
    current_indices = tuple(
        int(atom.index) for atom in (getattr(current, "atom", ()) or ())
    )
    if current_indices != snapshot.atom_indices:
        raise ValueError(
            f"Cannot restore {snapshot.obj_name}: its atom indices changed during painting."
        )
    max_index = max(snapshot.atom_indices, default=-1)
    b_values = np.zeros(max_index + 1, dtype=np.float32)
    colors = np.zeros(max_index + 1, dtype=np.int32)
    if snapshot.atom_indices:
        indices = np.asarray(snapshot.atom_indices, dtype=np.int64)
        b_values[indices] = snapshot.b_factors
        colors[indices] = snapshot.color_indices
    cmd.alter(
        snapshot.obj_name,
        "b = foldqc_restore_b[index]; color = foldqc_restore_color[index]",
        space={
            "foldqc_restore_b": b_values.tolist(),
            "foldqc_restore_color": colors.tolist(),
        },
    )
    cmd.recolor(snapshot.obj_name)


def rebuild() -> None:
    from pymol import cmd

    cmd.rebuild()


def refresh() -> None:
    from pymol import cmd

    cmd.refresh()


def _compact_integer_ranges(values: Iterable[int]) -> str:
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
    if not chain_id:
        return '""'
    if _SAFE_SELECTOR_NAME.fullmatch(chain_id):
        return chain_id
    return None


def _exact_token_selection(object_name: str, token: Any) -> str:
    if bool(token.is_hetatm) and token.atom_name is not None:
        return f"/{object_name}//{token.chain_id}/{token.resi}/{token.atom_name}"
    return f"/{object_name}//{token.chain_id}/{token.resi}/"


def compact_selection_expression(
    token_indices: Iterable[int],
    object_token_maps: Sequence[tuple[str, TokenMap]],
) -> str:
    """Build a compact PyMOL expression for tokens across viewer objects."""
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
        all_ligand_indices: dict[tuple[str, ResidueId, str], set[int]] = {}
        polymer_residues: dict[str, set[ResidueId]] = {}
        ligand_atoms: dict[tuple[str, ResidueId, str], list[str]] = {}
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
            key = (chain_id, token.residue_id, str(token.res_name or ""))
            all_ligand_indices.setdefault(key, set()).add(token_idx)

        for token_idx in indices:
            if token_idx not in selected_indices:
                continue
            token = token_map[token_idx]
            required_fields = (
                "token_idx",
                "chain_id",
                "residue_id",
                "res_name",
                "is_hetatm",
                "atom_name",
            )
            missing_fields = [
                field for field in required_fields if not hasattr(token, field)
            ]
            if missing_fields:
                raise ValueError(
                    f"Token {token_idx} for object {object_name!r} is missing "
                    f"required TokenInfo fields: {', '.join(missing_fields)}."
                )
            chain_id = str(token.chain_id)
            chain_selector = _chain_selector(chain_id)
            residue_id = token.residue_id

            if compact_object is None or chain_selector is None:
                fallback_selections.append(
                    _exact_token_selection(str(object_name), token)
                )
                continue
            if not bool(token.is_hetatm):
                polymer_residues.setdefault(chain_id, set()).add(residue_id)
                continue

            residue_name = str(token.res_name or "")
            atom_name = str(token.atom_name or "")
            if not (
                _SAFE_SELECTOR_NAME.fullmatch(residue_name)
                and _SAFE_SELECTOR_NAME.fullmatch(atom_name)
            ):
                fallback_selections.append(
                    _exact_token_selection(str(object_name), token)
                )
                continue
            ligand_atoms.setdefault((chain_id, residue_id, residue_name), []).append(
                atom_name
            )

        if compact_object is not None:
            for chain_id, residue_ids in polymer_residues.items():
                all_chain_indices = all_polymer_indices.get(chain_id, set())
                if all_chain_indices and all_chain_indices <= selected_indices:
                    add_clause(
                        f"%{compact_object} and polymer and chain "
                        f"{_chain_selector(chain_id)}"
                    )
                    continue
                plain_numbers = [
                    residue.number
                    for residue in residue_ids
                    if not residue.insertion_code
                ]
                if plain_numbers:
                    add_clause(
                        f"%{compact_object} and polymer and chain "
                        f"{_chain_selector(chain_id)} and resi "
                        f"{_compact_integer_ranges(plain_numbers)}"
                    )
                for residue in sorted(
                    (item for item in residue_ids if item.insertion_code),
                    key=lambda item: (item.number, item.insertion_code),
                ):
                    add_clause(
                        f"%{compact_object} and polymer and chain "
                        f"{_chain_selector(chain_id)} and resi {residue.resi}"
                    )

            for key, atom_names in ligand_atoms.items():
                chain_id, residue_id, residue_name = key
                all_residue_indices = all_ligand_indices.get(key, set())
                base = (
                    f"%{compact_object} and hetatm and chain "
                    f"{_chain_selector(chain_id)} and resi {residue_id.resi} "
                    f"and resn {residue_name}"
                )
                if all_residue_indices and all_residue_indices <= selected_indices:
                    add_clause(base)
                else:
                    names = ",".join(dict.fromkeys(atom_names))
                    add_clause(f"{base} and name {names}")

        for selection in fallback_selections:
            add_clause(selection)

    return " or ".join(clauses)


def _model_token_identities(model: Any) -> set[tuple[str, ResidueId, str, str | None]]:
    identities: set[tuple[str, ResidueId, str, str | None]] = set()
    for atom in getattr(model, "atom", []) or []:
        try:
            residue_id = ResidueId.parse(atom.resi)
        except (TypeError, ValueError):
            continue
        chain = str(getattr(atom, "chain", ""))
        resn = str(getattr(atom, "resn", ""))
        if bool(getattr(atom, "hetatm", False)):
            atom_name = str(getattr(atom, "name", ""))
            element = getattr(atom, "symbol", getattr(atom, "elem", ""))
            if is_hydrogen(element, atom_name):
                continue
            identities.add((chain, residue_id, resn, atom_name))
        else:
            identities.add((chain, residue_id, resn, None))
    return identities


def _overlap_summary_from_model(token_map: TokenMap, model: Any) -> TokenOverlapSummary:
    target_identities = _model_token_identities(model)
    matched_total = len(token_map.token_identities & target_identities)
    prediction_total = len(token_map.token_identities)
    target_total = len(target_identities)
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


def _atom_indices(model: Any) -> tuple[int, ...]:
    indices: list[int] = []
    for fallback, atom in enumerate(getattr(model, "atom", []) or [], start=1):
        index = int(getattr(atom, "index", fallback))
        if index <= 0:
            raise ValueError(f"PyMOL atom index must be positive, got {index}.")
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError("PyMOL object contains duplicate atom indices.")
    return tuple(indices)


def _paint_mapping_from_model(
    obj_name: str,
    token_map: TokenMap,
    model: Any,
) -> ObjectPaintMapping:
    indices = _atom_indices(model)
    max_index = max(indices, default=0)
    atom_token_indices = np.full(max_index + 1, -1, dtype=np.int32)
    for atom, index in zip(getattr(model, "atom", []) or [], indices):
        try:
            residue_id = ResidueId.parse(atom.resi)
        except (TypeError, ValueError):
            continue
        chain_id = str(getattr(atom, "chain", ""))
        residue_name = str(getattr(atom, "resn", ""))
        if bool(getattr(atom, "hetatm", False)):
            atom_name = str(getattr(atom, "name", ""))
            element = getattr(atom, "symbol", getattr(atom, "elem", ""))
            if is_hydrogen(element, atom_name):
                continue
            token_idx = token_map.hetatm_token_by_atom.get(
                (chain_id, residue_id, residue_name, atom_name), -1
            )
        else:
            token_idx = token_map.polymer_token_by_residue.get(
                (chain_id, residue_id, residue_name), -1
            )
        atom_token_indices[index] = token_idx
    return ObjectPaintMapping(
        obj_name=str(obj_name),
        atom_index_fingerprint=indices,
        atom_token_indices=atom_token_indices,
        atom_count=len(indices),
        max_atom_index=max_index,
        overlap=_overlap_summary_from_model(token_map, model),
    )


def prepare_object_paint_mapping(
    obj_name: str,
    token_map: TokenMap,
    *,
    model: Any | None = None,
) -> ObjectPaintMapping:
    """Inspect one object and prepare reusable atom-index painting metadata."""
    if model is None:
        from pymol import cmd

        model = cmd.get_model(obj_name)
    return _paint_mapping_from_model(obj_name, token_map, model)


def object_paint_mapping_is_valid(mapping: ObjectPaintMapping) -> bool:
    """Return whether an object's current atom-index order matches *mapping*."""
    from pymol import cmd

    try:
        current = tuple(int(index) for _model, index in cmd.index(mapping.obj_name))
    except Exception:
        return False
    return current == mapping.atom_index_fingerprint


def ensure_object_paint_mapping(
    obj_name: str,
    token_map: TokenMap,
    mapping: ObjectPaintMapping | None = None,
) -> tuple[ObjectPaintMapping, bool]:
    """Reuse a valid mapping or rebuild it, returning ``(mapping, rebuilt)``."""
    if (
        mapping is not None
        and mapping.obj_name == obj_name
        and object_paint_mapping_is_valid(mapping)
    ):
        return mapping, False
    return prepare_object_paint_mapping(obj_name, token_map), True


def compare_token_map_to_object(
    token_map: TokenMap, obj_name: str
) -> TokenOverlapSummary:
    """Compare prediction-token identities with a loaded viewer object."""
    from pymol import cmd

    return _overlap_summary_from_model(token_map, cmd.get_model(obj_name))


def selection_to_token_indices(
    token_map: TokenMap, selection: str, obj_name: str = "all"
) -> list[int]:
    """Resolve a viewer selection to sorted prediction-token indices."""
    from pymol import cmd

    model = cmd.get_model(f"({selection}) and {obj_name}")
    if model is None:
        return []
    polymer_residues: set[tuple[str, ResidueId, str]] = set()
    hetatm_atoms: set[tuple[str, ResidueId, str, str]] = set()
    for atom in model.atom:
        try:
            residue_id = ResidueId.parse(atom.resi)
        except ValueError:
            continue
        key = (str(atom.chain), residue_id, str(atom.resn))
        if atom.hetatm:
            if is_hydrogen(
                getattr(atom, "symbol", getattr(atom, "elem", "")), atom.name
            ):
                continue
            hetatm_atoms.add((*key, atom.name))
        else:
            polymer_residues.add(key)
    result = []
    for token in token_map:
        if token.is_hetatm:
            if (
                token.chain_id,
                token.residue_id,
                token.res_name,
                token.atom_name,
            ) in hetatm_atoms:
                result.append(token.token_idx)
        elif (token.chain_id, token.residue_id, token.res_name) in polymer_residues:
            result.append(token.token_idx)
    return sorted(result)


def tokens_within_distance(
    token_map: TokenMap,
    obj_name: str,
    reference_selection: str,
    cutoff: float,
) -> list[int]:
    """Return residue-expanded tokens with any atom near a scoped selection."""
    scoped_reference = f"(({reference_selection}) and ({obj_name}))"
    nearby = (
        f"byres (({obj_name}) within {float(cutoff):g} of {scoped_reference}) "
        f"and not {scoped_reference}"
    )
    return selection_to_token_indices(token_map, nearby, obj_name=obj_name)


def _resolve_color_range(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size else 0.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def token_bfactor_keys(token_map: TokenMap) -> list[tuple[str, str, str]]:
    return [
        (
            token.chain_id,
            token.resi,
            token.atom_name if token.is_hetatm and token.atom_name else "",
        )
        for token in token_map
    ]


def _expand_token_values_for_atoms(
    mapping: ObjectPaintMapping,
    token_values: np.ndarray,
    *,
    default: float | int,
) -> list[float] | list[int]:
    atom_values = np.full(
        mapping.max_atom_index + 1,
        default,
        dtype=token_values.dtype,
    )
    if mapping.atom_token_indices.size:
        atom_indices = np.flatnonzero(mapping.atom_token_indices >= 0)
        if atom_indices.size:
            token_indices = mapping.atom_token_indices[atom_indices]
            atom_values[atom_indices] = token_values[token_indices]
    return atom_values.tolist()


def _scaled_bfactor_values(values: np.ndarray, scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(arr), arr * float(scale), -1.0)


def _write_bfactors_bulk(
    obj_name: str,
    token_map: TokenMap,
    values: np.ndarray,
    *,
    scale: float = 1.0,
    mapping: ObjectPaintMapping | None = None,
) -> ObjectPaintMapping:
    from pymol import cmd

    if len(values) != len(token_map):
        raise ValueError(
            f"values length {len(values)} does not match token_map length "
            f"{len(token_map)}."
        )
    if mapping is None or mapping.obj_name != obj_name:
        mapping = prepare_object_paint_mapping(obj_name, token_map)
    atom_values = _expand_token_values_for_atoms(
        mapping,
        _scaled_bfactor_values(values, scale),
        default=-1.0,
    )
    cmd.alter(
        obj_name,
        "b = foldqc_atom_values[index]",
        space={"foldqc_atom_values": atom_values},
    )
    return mapping


def _reverse_color_list(colors: str) -> str:
    separator = " " if " " in colors else "_"
    return " ".join(reversed([part for part in colors.split(separator) if part]))


def _resolve_pymol_palette(key: str, reverse: bool = False) -> _PaletteResolution:
    spec = _PALETTE_SPECS_BY_KEY.get(key)
    if spec is not None and spec.rgb_stops:
        stops = tuple(reversed(spec.rgb_stops)) if reverse else spec.rgb_stops
        direction = "r" if reverse else "f"
        colors = tuple(
            _ColorDef(f"foldqc_{key}_{direction}_{index:02d}", rgb)
            for index, rgb in enumerate(stops)
        )
        return _PaletteResolution(" ".join(color.name for color in colors), colors)
    native = _PYMOL_PALETTES.get(key)
    if native is not None:
        forward, backward = native
        if reverse and backward:
            return _PaletteResolution(backward)
        return _PaletteResolution(_reverse_color_list(forward) if reverse else forward)
    return _PaletteResolution(_reverse_color_list(key) if reverse else key)


def paint_property(
    obj_name: str,
    token_map: TokenMap,
    values: np.ndarray,
    palette: str = "blue_white_red",
    reverse_palette: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    nan_color: str = NAN_COLOR_DEFAULT,
    mapping: ObjectPaintMapping | None = None,
) -> tuple[float, float]:
    return paint_property_bulk(
        obj_name,
        token_map,
        values,
        palette=palette,
        reverse_palette=reverse_palette,
        vmin=vmin,
        vmax=vmax,
        nan_color=nan_color,
        rebuild=True,
        mapping=mapping,
    )


def _object_union_selection(object_names: Sequence[str]) -> str:
    return " or ".join(f"({name})" for name in dict.fromkeys(object_names))


def _quantized_custom_colors(
    palette_key: str,
    colors: Sequence[_ColorDef],
    *,
    bins: int = 256,
) -> tuple[_ColorDef, ...]:
    stops = np.asarray([color.rgb for color in colors], dtype=np.float64)
    source_positions = np.linspace(0.0, 1.0, len(stops))
    target_positions = np.linspace(0.0, 1.0, bins)
    interpolated = np.column_stack(
        [
            np.interp(target_positions, source_positions, stops[:, channel])
            for channel in range(3)
        ]
    )
    digest = hashlib.sha256(stops.astype(np.float32).tobytes()).hexdigest()[:8]
    direction = "r" if colors and "_r_" in colors[0].name else "f"
    safe_key = re.sub(r"[^A-Za-z0-9_]", "_", palette_key)
    prefix = f"foldqc_{safe_key}_{direction}_q{bins}_{digest}"
    return tuple(
        _ColorDef(f"{prefix}_{index:03d}", tuple(float(v) for v in rgb))
        for index, rgb in enumerate(interpolated)
    )


def _ensure_color_indices(colors: Sequence[_ColorDef]) -> list[int]:
    from pymol import cmd

    indices = [int(cmd.get_color_index(color.name)) for color in colors]
    if any(index < 0 for index in indices):
        for color in colors:
            cmd.set_color(color.name, list(color.rgb))
        indices = [int(cmd.get_color_index(color.name)) for color in colors]
    if any(index < 0 for index in indices):
        raise ValueError("PyMOL did not register the FoldQC custom palette colors.")
    return indices


def _token_color_indices(
    values: np.ndarray,
    color_indices: Sequence[int],
    *,
    vmin: float,
    vmax: float,
    nan_color_index: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    result = np.full(arr.shape, int(nan_color_index), dtype=np.int64)
    finite = np.isfinite(arr)
    if finite.any():
        normalized = np.clip((arr[finite] - vmin) / (vmax - vmin), 0.0, 1.0)
        bins = np.rint(normalized * (len(color_indices) - 1)).astype(np.int64)
        palette_indices = np.asarray(color_indices, dtype=np.int64)
        result[finite] = palette_indices[bins]
    return result


def paint_properties_bulk(
    targets: Sequence[PaintTarget],
    palette: str = "blue_white_red",
    reverse_palette: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    nan_color: str = NAN_COLOR_DEFAULT,
    rebuild: bool = True,
) -> PaintBatchResult:
    """Paint one or more objects with one shared range and palette operation."""
    from pymol import cmd

    if not targets:
        raise ValueError("At least one paint target is required.")
    arrays = [np.asarray(target.values) for target in targets]
    combined = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
    used_vmin, used_vmax = _resolve_color_range(combined, vmin, vmax)

    resolved_targets: list[tuple[PaintTarget, ObjectPaintMapping]] = []
    for target in targets:
        if len(target.values) != len(target.token_map):
            raise ValueError(
                f"values length {len(target.values)} does not match token_map length "
                f"{len(target.token_map)}."
            )
        mapping = target.mapping
        if mapping is None or mapping.obj_name != target.obj_name:
            mapping = prepare_object_paint_mapping(target.obj_name, target.token_map)
        resolved_targets.append((target, mapping))

    resolved = _resolve_pymol_palette(palette, reverse_palette)
    selection = _object_union_selection(
        [target.obj_name for target, _mapping in resolved_targets]
    )
    if resolved.custom_colors:
        quantized = _quantized_custom_colors(palette, resolved.custom_colors)
        color_indices = _ensure_color_indices(quantized)
        nan_color_index = int(cmd.get_color_index(nan_color))
        if nan_color_index < 0:
            raise ValueError(f"Unknown PyMOL NaN color: {nan_color!r}.")
        for target, mapping in resolved_targets:
            atom_values = _expand_token_values_for_atoms(
                mapping,
                _scaled_bfactor_values(target.values, 1.0),
                default=-1.0,
            )
            token_colors = _token_color_indices(
                target.values,
                color_indices,
                vmin=used_vmin,
                vmax=used_vmax,
                nan_color_index=nan_color_index,
            )
            atom_colors = _expand_token_values_for_atoms(
                mapping, token_colors, default=nan_color_index
            )
            cmd.alter(
                target.obj_name,
                "b = foldqc_atom_values[index]; color = foldqc_atom_colors[index]",
                space={
                    "foldqc_atom_values": atom_values,
                    "foldqc_atom_colors": atom_colors,
                },
            )
        cmd.recolor(selection)
    else:
        for target, mapping in resolved_targets:
            _write_bfactors_bulk(
                target.obj_name,
                target.token_map,
                target.values,
                mapping=mapping,
            )
        cmd.spectrum(
            "b",
            resolved.palette,
            selection,
            minimum=used_vmin,
            maximum=used_vmax,
        )
        cmd.color(nan_color, f"({selection}) and b < 0")
    if rebuild:
        cmd.rebuild()
    return PaintBatchResult(
        used_vmin,
        used_vmax,
        tuple(mapping for _target, mapping in resolved_targets),
    )


def paint_property_bulk(
    obj_name: str,
    token_map: TokenMap,
    values: np.ndarray,
    palette: str = "blue_white_red",
    reverse_palette: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    nan_color: str = NAN_COLOR_DEFAULT,
    rebuild: bool = True,
    mapping: ObjectPaintMapping | None = None,
) -> tuple[float, float]:
    result = paint_properties_bulk(
        [PaintTarget(obj_name, token_map, values, mapping)],
        palette=palette,
        reverse_palette=reverse_palette,
        vmin=vmin,
        vmax=vmax,
        nan_color=nan_color,
        rebuild=rebuild,
    )
    return result.vmin, result.vmax


def _category_color_name(label: int) -> str:
    return (
        f"foldqc_category_{label:03d}"
        if label >= 0
        else f"foldqc_category_m{-label:03d}"
    )


def paint_categorical_labels_bulk(
    obj_name: str,
    token_map: TokenMap,
    values: np.ndarray,
    nan_color: str = NAN_COLOR_DEFAULT,
    rebuild: bool = True,
    mapping: ObjectPaintMapping | None = None,
) -> tuple[float, float]:
    result = paint_categorical_labels_batch(
        [PaintTarget(obj_name, token_map, values, mapping)],
        nan_color=nan_color,
        rebuild=rebuild,
    )
    return result.vmin, result.vmax


def paint_categorical_labels_batch(
    targets: Sequence[PaintTarget],
    nan_color: str = NAN_COLOR_DEFAULT,
    rebuild: bool = True,
) -> PaintBatchResult:
    """Paint integer labels across targets with shared viewer commands."""
    from pymol import cmd

    if not targets:
        raise ValueError("At least one paint target is required.")
    arrays = [np.asarray(target.values, dtype=np.float32) for target in targets]
    combined = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
    vmin, vmax = _resolve_color_range(combined)
    mappings: list[ObjectPaintMapping] = []
    for target, arr in zip(targets, arrays):
        mapping = _write_bfactors_bulk(
            target.obj_name,
            target.token_map,
            arr,
            mapping=target.mapping,
        )
        mappings.append(mapping)
    labels = sorted(
        {int(round(float(value))) for value in combined[np.isfinite(combined)]}
    )
    selection = _object_union_selection([target.obj_name for target in targets])
    for label in labels:
        color_name = _category_color_name(label)
        cmd.set_color(color_name, list(categorical_color(label)))
        cmd.color(color_name, f"({selection}) and b = {label:g}")
    cmd.color(nan_color, f"({selection}) and b < 0")
    if rebuild:
        cmd.rebuild()
    return PaintBatchResult(vmin, vmax, tuple(mappings))


def delete_colorbar(name: str = COLORBAR_OBJECT_NAME) -> None:
    from pymol import cmd

    cmd.delete(name)


def _split_palette_names(palette: str) -> list[str]:
    if " " in palette:
        return [part for part in palette.split() if part]
    return [part for part in palette.split("_") if part]


def show_colorbar(
    palette: str,
    reverse_palette: bool,
    vmin: float,
    vmax: float,
    *,
    object_names: list[str] | tuple[str, ...] | None = None,
    name: str = COLORBAR_OBJECT_NAME,
    segments: int = 64,
) -> None:
    del object_names, segments
    from pymol import cmd

    delete_colorbar(name)
    resolved = _resolve_pymol_palette(palette, reverse_palette)
    colors: list[str | list[float]]
    if resolved.custom_colors:
        colors = [list(color.rgb) for color in resolved.custom_colors]
    else:
        colors = _split_palette_names(resolved.palette)
    if len(colors) == 1:
        colors.append(colors[0])
    ranges = np.linspace(float(vmin), float(vmax), len(colors)).tolist()
    cmd.ramp_new(name, None, ranges, colors, quiet=1)


def reset_bfactors(obj_name: str, value: float = 100.0) -> None:
    from pymol import cmd

    cmd.alter(obj_name, f"b = {value:.2f}")
    cmd.rebuild()


def _representative_coords_from_model(model: Any, token_map: TokenMap) -> np.ndarray:
    coords = np.empty((len(token_map), 3), dtype=np.float32)
    atom_coords: dict[tuple[str, ResidueId, str, str], tuple[float, float, float]] = {}
    first_atom: dict[tuple[str, ResidueId, str], tuple[float, float, float]] = {}
    for atom in model.atom:
        try:
            residue_id = ResidueId.parse(atom.resi)
        except ValueError:
            continue
        residue_key = (str(atom.chain), residue_id, str(atom.resn))
        key = (*residue_key, str(atom.name))
        atom_coords[key] = tuple(atom.coord)
        first_atom.setdefault(residue_key, tuple(atom.coord))
    missing: list[str] = []
    for token in token_map:
        residue_key = (token.chain_id, token.residue_id, token.res_name)
        if token.is_hetatm and token.atom_name is not None:
            xyz = atom_coords.get((*residue_key, token.atom_name))
        else:
            xyz = next(
                (
                    atom_coords[(*residue_key, name)]
                    for name in ("CA", "C1'", "C1*")
                    if (*residue_key, name) in atom_coords
                ),
                first_atom.get(residue_key),
            )
        if xyz is None:
            atom_suffix = f"/{token.atom_name}" if token.atom_name else ""
            missing.append(
                f"{token.chain_id}:{token.res_name}{token.resi}{atom_suffix}"
            )
            continue
        coords[token.token_idx] = xyz
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise ValueError(
            f"Viewer object is missing canonical token coordinates: {preview}{suffix}."
        )
    return coords


def inspect_object_tokens(obj_name: str, token_map: TokenMap) -> ObjectTokenInspection:
    """Inspect one object once for reusable painting metadata and coordinates."""
    from pymol import cmd

    model = cmd.get_model(obj_name)
    return ObjectTokenInspection(
        paint_mapping=prepare_object_paint_mapping(obj_name, token_map, model=model),
        representative_coords=_representative_coords_from_model(model, token_map),
    )


def get_representative_coords(obj_name: str, token_map: TokenMap) -> np.ndarray:
    """Return one representative coordinate per prediction token."""
    return inspect_object_tokens(obj_name, token_map).representative_coords


def transform_object(
    obj_name: str, rotation: np.ndarray, translation: np.ndarray
) -> None:
    """Apply a rigid transform to every atom in one object."""
    from pymol import cmd, stored

    stored.foldqc_ensemble_rotation = rotation.tolist()
    stored.foldqc_ensemble_translation = translation.tolist()
    expression = (
        "x, y, z = ("
        "stored.foldqc_ensemble_rotation[0][0]*x + "
        "stored.foldqc_ensemble_rotation[0][1]*y + "
        "stored.foldqc_ensemble_rotation[0][2]*z + "
        "stored.foldqc_ensemble_translation[0], "
        "stored.foldqc_ensemble_rotation[1][0]*x + "
        "stored.foldqc_ensemble_rotation[1][1]*y + "
        "stored.foldqc_ensemble_rotation[1][2]*z + "
        "stored.foldqc_ensemble_translation[1], "
        "stored.foldqc_ensemble_rotation[2][0]*x + "
        "stored.foldqc_ensemble_rotation[2][1]*y + "
        "stored.foldqc_ensemble_rotation[2][2]*z + "
        "stored.foldqc_ensemble_translation[2])"
    )
    cmd.alter_state(1, obj_name, expression)


def _plddt_selection(color: Any) -> str:
    """Translate generic pLDDT bounds into valid PyMOL selection syntax."""
    if color.is_nan:
        return "(b<0)"
    minimum = color.minimum
    maximum = color.maximum
    if minimum is not None and maximum is not None:
        return f"((b<{maximum:g} and b>{minimum:g}) or b={minimum:g})"
    if minimum is not None:
        return f"(b>{minimum:g} or b={minimum:g})"
    if maximum is not None:
        return f"(b<{maximum:g})"
    raise ValueError(f"pLDDT class {color.key!r} has no bounds.")


def paint_plddt_class_coloring(
    obj_name: str,
    values: np.ndarray | None = None,
    token_map: TokenMap | None = None,
    rebuild: bool = True,
    mapping: ObjectPaintMapping | None = None,
) -> None:
    from pymol import cmd

    if values is not None and token_map is not None:
        paint_plddt_class_batch(
            [PaintTarget(obj_name, token_map, values, mapping)], rebuild=rebuild
        )
        return
    for color in PLDDT_CLASS_COLORS:
        color_name = color.key if color.key == "plddt_nan" else f"plddt_{color.key}"
        cmd.set_color(color_name, list(color.rgb))
        cmd.color(color_name, f"{obj_name} and {_plddt_selection(color)}")
    if rebuild:
        cmd.rebuild()


def paint_plddt_class_batch(
    targets: Sequence[PaintTarget],
    rebuild: bool = True,
) -> tuple[ObjectPaintMapping, ...]:
    """Write pLDDT B-factors and apply quality classes across targets."""
    from pymol import cmd

    if not targets:
        raise ValueError("At least one paint target is required.")
    mappings: list[ObjectPaintMapping] = []
    for target in targets:
        mapping = _write_bfactors_bulk(
            target.obj_name,
            target.token_map,
            target.values,
            scale=100.0,
            mapping=target.mapping,
        )
        mappings.append(mapping)
    selection = _object_union_selection([target.obj_name for target in targets])
    for color in PLDDT_CLASS_COLORS:
        color_name = color.key if color.key == "plddt_nan" else f"plddt_{color.key}"
        cmd.set_color(color_name, list(color.rgb))
        cmd.color(color_name, f"({selection}) and {_plddt_selection(color)}")
    if rebuild:
        cmd.rebuild()
    return tuple(mappings)


def update_token_selection(
    selection_name: str,
    token_indices: Iterable[int],
    object_token_maps: Sequence[tuple[str, TokenMap]],
    *,
    enable: bool = True,
    refresh_view: bool = True,
) -> None:
    """Create or replace a named selection from token indices."""
    from pymol import cmd

    expression = compact_selection_expression(token_indices, object_token_maps)
    cmd.select(selection_name, expression or "none")
    if enable:
        cmd.enable(selection_name)
    if refresh_view:
        cmd.refresh()


def show_token_selection(
    selection_name: str,
    token_indices: Iterable[int],
    object_token_maps: Sequence[tuple[str, TokenMap]],
) -> None:
    """Select tokens and display/zoom them as sticks."""
    from pymol import cmd

    expression = compact_selection_expression(token_indices, object_token_maps)
    cmd.select(selection_name, expression or "none")
    cmd.show("sticks", selection_name)
    cmd.enable(selection_name)
    cmd.zoom(selection_name)
    cmd.refresh()


def show_token_groups(
    selection_name: str,
    token_groups: Sequence[tuple[Iterable[int], str, TokenMap]],
) -> None:
    """Combine per-object token groups, then display and zoom the selection."""
    from pymol import cmd

    expressions = [
        compact_selection_expression(indices, [(object_name, token_map)])
        for indices, object_name, token_map in token_groups
    ]
    expression = " or ".join(part for part in expressions if part)
    cmd.select(selection_name, expression or "none")
    cmd.show("sticks", selection_name)
    cmd.enable(selection_name)
    cmd.zoom(selection_name)
    cmd.refresh()


def clear_selection(selection_name: str, *, refresh_view: bool = True) -> None:
    from pymol import cmd

    cmd.select(selection_name, "none")
    if refresh_view:
        cmd.refresh()


def clear_selections(
    selection_names: Iterable[str], *, refresh_view: bool = True
) -> None:
    from pymol import cmd

    for name in selection_names:
        cmd.select(str(name), "none")
    if refresh_view:
        cmd.refresh()


class PyMOLViewer:
    """Concrete object adapter implementing the application ``ViewerPort``."""

    def __init__(self) -> None:
        self.paint_mappings: dict[tuple[str, str], ObjectPaintMapping] = {}
        self._managed_colorbar = None

    def object_names(self, additional_names: Sequence[str] = ()) -> list[str]:
        return get_object_list(additional_names=additional_names)

    def ensure_structure_object(
        self, path, obj_name: str, *, zoom: bool = True
    ) -> bool:
        return ensure_structure_object(path, obj_name, zoom=zoom)

    def load_structure_object_if_missing(self, path, obj_name: str) -> bool:
        return load_structure_object_if_missing(path, obj_name)

    def delete_names(self, names: Sequence[str]) -> None:
        delete_viewer_names(names)

    def name_exists(self, name: str) -> bool:
        return viewer_name_exists(name)

    def group_members(self, group_name: str) -> tuple[str, ...]:
        return get_group_members(group_name)

    def add_to_group(self, group_name: str, names: Sequence[str]) -> None:
        add_objects_to_group(group_name, names)

    def remove_from_group(self, group_name: str, names: Sequence[str]) -> None:
        remove_objects_from_group(group_name, names)

    def inspect_tokens(self, obj_name: str, token_map: TokenMap):
        return inspect_object_tokens(obj_name, token_map)

    def transform(
        self, obj_name: str, rotation: np.ndarray, translation: np.ndarray
    ) -> None:
        transform_object(obj_name, rotation, translation)

    def selection_token_indices(
        self, token_map: TokenMap, selection: str, *, obj_name: str
    ) -> list[int]:
        return selection_to_token_indices(token_map, selection, obj_name=obj_name)

    def tokens_within_distance(
        self,
        token_map: TokenMap,
        obj_name: str,
        reference_selection: str,
        cutoff: float,
    ) -> list[int]:
        return tokens_within_distance(token_map, obj_name, reference_selection, cutoff)

    def snapshot_atom_visuals(self, obj_name: str):
        return snapshot_atom_visuals(obj_name)

    def restore_atom_visuals(self, snapshot) -> None:
        restore_atom_visuals(snapshot)

    def paint_continuous(
        self,
        targets: Sequence[PaintTarget],
        *,
        palette: str,
        reverse_palette: bool,
        vmin: float | None,
        vmax: float | None,
        rebuild: bool = False,
    ) -> PaintBatchResult:
        return paint_properties_bulk(
            targets,
            palette=palette,
            reverse_palette=reverse_palette,
            vmin=vmin,
            vmax=vmax,
            rebuild=rebuild,
        )

    def paint_categorical(
        self, targets: Sequence[PaintTarget], *, rebuild: bool = False
    ) -> PaintBatchResult:
        return paint_categorical_labels_batch(targets, rebuild=rebuild)

    def paint_plddt_classes(
        self, targets: Sequence[PaintTarget], *, rebuild: bool = False
    ) -> PaintBatchResult:
        mappings = paint_plddt_class_batch(targets, rebuild=rebuild)
        arrays = [np.asarray(target.values, dtype=np.float32) for target in targets]
        combined = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        vmin, vmax = _resolve_color_range(combined)
        return PaintBatchResult(vmin, vmax, tuple(mappings))

    def get_managed_colorbar(self):
        return self._managed_colorbar

    def replace_managed_colorbar(self, state) -> None:
        if state is None:
            delete_colorbar()
            self._managed_colorbar = None
            return
        show_colorbar(
            state.palette,
            state.reverse_palette,
            state.vmin,
            state.vmax,
            object_names=state.object_names,
        )
        self._managed_colorbar = state

    def rebuild(self) -> None:
        rebuild()

    def run_suspended(self, operation: Callable[[], T]) -> T:
        return run_with_updates_suspended(operation)

    def ensure_paint_mapping(
        self,
        obj_name: str,
        token_map: TokenMap,
        existing: ObjectPaintMapping | None,
    ) -> tuple[ObjectPaintMapping, bool]:
        return ensure_object_paint_mapping(obj_name, token_map, existing)
