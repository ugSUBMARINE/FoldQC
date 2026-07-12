"""PyMOL implementation of FoldQC's molecular-viewer boundary.

All viewer commands and PyMOL selection syntax live in this module. Other
modules exchange viewer-independent token metadata and call the focused
functions below. PyMOL imports stay lazy so the package remains importable in
plain Python and in tests with a fake viewer.
"""

from __future__ import annotations

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
from .token_map import TokenInfo, TokenOverlapSummary

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


def load_models_as_states(
    model_paths: list[tuple[int, Path]],
    obj_name: str = "foldqc_ensemble",
) -> str:
    """Load ranked models as consecutive states of one object."""
    from pymol import cmd

    for state_idx, (_rank, path) in enumerate(sorted(model_paths), start=1):
        cmd.load(str(path), obj_name, state=state_idx)
    return obj_name


def load_models_as_objects(
    model_paths: list[tuple[int, Path]],
    obj_prefix: str = "foldqc_model",
    group_name: str | None = None,
) -> list[tuple[int, str]]:
    """Load ranked models as separate objects and optionally group them."""
    from pymol import cmd

    current_objects = set(cmd.get_names("objects") or [])
    result: list[tuple[int, str]] = []
    try:
        try:
            cmd.set("suspend_updates", "on")
        except Exception:
            pass
        for rank, path in sorted(model_paths):
            obj_name = f"{obj_prefix}_{rank}"
            if obj_name not in current_objects:
                cmd.load(str(path), obj_name, quiet=1, zoom=0)
                current_objects.add(obj_name)
            if group_name:
                cmd.group(group_name, obj_name, "add")
            result.append((rank, obj_name))
    finally:
        try:
            cmd.set("suspend_updates", "off")
            cmd.rebuild()
        except Exception:
            pass
    return result


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
        return f"/{object_name}//{token.chain_id}/{token.res_num}/{token.atom_name}"
    return f"/{object_name}//{token.chain_id}/{token.res_num}/"


def compact_selection_expression(
    token_indices: Iterable[int],
    object_token_maps: Sequence[tuple[str, Sequence[Any]]],
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
            try:
                residue_number = int(token.res_num)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Token {token_idx} for object {object_name!r} has an invalid "
                    f"res_num: {token.res_num!r}."
                ) from None

            if compact_object is None or chain_selector is None:
                fallback_selections.append(
                    _exact_token_selection(str(object_name), token)
                )
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
                fallback_selections.append(
                    _exact_token_selection(str(object_name), token)
                )
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
                add_clause(
                    f"%{compact_object} and polymer and chain "
                    f"{_chain_selector(chain_id)} and resi "
                    f"{_compact_integer_ranges(residue_numbers)}"
                )

            for key, atom_names in ligand_atoms.items():
                chain_id, residue_number, residue_name = key
                all_residue_indices = all_ligand_indices.get(key, set())
                base = (
                    f"%{compact_object} and hetatm and chain "
                    f"{_chain_selector(chain_id)} and resi {residue_number} "
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


def _model_token_identities(model: Any) -> set[tuple[str, int, str, str | None]]:
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


def compare_token_map_to_object(
    token_map: list[TokenInfo], obj_name: str
) -> TokenOverlapSummary:
    """Compare prediction-token identities with a loaded viewer object."""
    from pymol import cmd

    prediction_identities = {
        (
            str(token.chain_id),
            int(token.res_num),
            str(token.res_name),
            str(token.atom_name or "") if token.is_hetatm else None,
        )
        for token in token_map
    }
    target_identities = _model_token_identities(cmd.get_model(obj_name))
    matched_total = len(prediction_identities & target_identities)
    prediction_total = len(prediction_identities)
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


def selection_to_token_indices(
    token_map: list[TokenInfo], selection: str, obj_name: str = "all"
) -> list[int]:
    """Resolve a viewer selection to sorted prediction-token indices."""
    from pymol import cmd

    model = cmd.get_model(f"({selection}) and {obj_name}")
    if model is None:
        return []
    polymer_residues: set[tuple[str, int]] = set()
    hetatm_atoms: set[tuple[str, int, str]] = set()
    for atom in model.atom:
        key = (atom.chain, int(atom.resi))
        if atom.hetatm:
            hetatm_atoms.add((*key, atom.name))
        else:
            polymer_residues.add(key)
    result = []
    for token in token_map:
        if token.is_hetatm:
            if (token.chain_id, token.res_num, token.atom_name) in hetatm_atoms:
                result.append(token.token_idx)
        elif (token.chain_id, token.res_num) in polymer_residues:
            result.append(token.token_idx)
    return sorted(result)


def tokens_within_distance(
    token_map: list[TokenInfo],
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


def token_bfactor_keys(token_map: list[TokenInfo]) -> list[tuple[str, str, str]]:
    return [
        (
            token.chain_id,
            str(token.res_num),
            token.atom_name if token.is_hetatm and token.atom_name else "",
        )
        for token in token_map
    ]


def _write_bfactors_bulk(
    obj_name: str,
    token_map: list[TokenInfo],
    values: np.ndarray,
    *,
    scale: float = 1.0,
) -> None:
    from pymol import cmd, stored

    if len(values) != len(token_map):
        raise ValueError(
            f"values length {len(values)} does not match token_map length "
            f"{len(token_map)}."
        )
    bmap: dict[tuple[str, str, str], float] = {}
    for token, key in zip(token_map, token_bfactor_keys(token_map)):
        value = float(values[token.token_idx])
        bmap[key] = value * scale if np.isfinite(value) else -1.0
    stored.foldqc_bmap = bmap
    cmd.alter(
        obj_name,
        "b = stored.foldqc_bmap.get((chain, resi, name), "
        "stored.foldqc_bmap.get((chain, resi, ''), -1.0))",
    )


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
    token_map: list[TokenInfo],
    values: np.ndarray,
    palette: str = "blue_white_red",
    reverse_palette: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    nan_color: str = NAN_COLOR_DEFAULT,
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
    )


def paint_property_bulk(
    obj_name: str,
    token_map: list[TokenInfo],
    values: np.ndarray,
    palette: str = "blue_white_red",
    reverse_palette: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    nan_color: str = NAN_COLOR_DEFAULT,
    rebuild: bool = True,
) -> tuple[float, float]:
    from pymol import cmd

    vmin, vmax = _resolve_color_range(values, vmin, vmax)
    _write_bfactors_bulk(obj_name, token_map, values)
    resolved = _resolve_pymol_palette(palette, reverse_palette)
    for color in resolved.custom_colors:
        cmd.set_color(color.name, list(color.rgb))
    cmd.spectrum("b", resolved.palette, obj_name, minimum=vmin, maximum=vmax)
    cmd.color(nan_color, f"{obj_name} and b < 0")
    if rebuild:
        cmd.rebuild()
    return vmin, vmax


def _category_color_name(label: int) -> str:
    return (
        f"foldqc_category_{label:03d}"
        if label >= 0
        else f"foldqc_category_m{-label:03d}"
    )


def paint_categorical_labels_bulk(
    obj_name: str,
    token_map: list[TokenInfo],
    values: np.ndarray,
    nan_color: str = NAN_COLOR_DEFAULT,
    rebuild: bool = True,
) -> tuple[float, float]:
    from pymol import cmd

    arr = np.asarray(values, dtype=np.float32)
    vmin, vmax = _resolve_color_range(arr)
    _write_bfactors_bulk(obj_name, token_map, arr)
    labels = sorted({int(round(float(value))) for value in arr[np.isfinite(arr)]})
    for label in labels:
        color_name = _category_color_name(label)
        cmd.set_color(color_name, list(categorical_color(label)))
        cmd.color(color_name, f"{obj_name} and b = {label:g}")
    cmd.color(nan_color, f"{obj_name} and b < 0")
    if rebuild:
        cmd.rebuild()
    return vmin, vmax


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


def get_representative_coords(obj_name: str, token_map: list[TokenInfo]) -> np.ndarray:
    """Return one representative coordinate per prediction token."""
    from pymol import cmd

    coords = np.zeros((len(token_map), 3), dtype=np.float32)
    model = cmd.get_model(obj_name)
    atom_coords: dict[tuple[str, int, str], tuple[float, float, float]] = {}
    first_atom: dict[tuple[str, int], tuple[float, float, float]] = {}
    for atom in model.atom:
        key = (atom.chain, int(atom.resi), atom.name)
        atom_coords[key] = tuple(atom.coord)
        first_atom.setdefault((atom.chain, int(atom.resi)), tuple(atom.coord))
    for token in token_map:
        residue_key = (token.chain_id, token.res_num)
        if token.is_hetatm and token.atom_name is not None:
            xyz = atom_coords.get(
                (*residue_key, token.atom_name), first_atom.get(residue_key, (0, 0, 0))
            )
        else:
            xyz = next(
                (
                    atom_coords[(*residue_key, name)]
                    for name in ("CA", "C1'", "C1*")
                    if (*residue_key, name) in atom_coords
                ),
                first_atom.get(residue_key, (0, 0, 0)),
            )
        coords[token.token_idx] = xyz
    return coords


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
    token_map: list[TokenInfo] | None = None,
    rebuild: bool = True,
) -> None:
    from pymol import cmd

    if values is not None and token_map is not None:
        _write_bfactors_bulk(obj_name, token_map, values, scale=100.0)
    for color in PLDDT_CLASS_COLORS:
        color_name = color.key if color.key == "plddt_nan" else f"plddt_{color.key}"
        cmd.set_color(color_name, list(color.rgb))
        cmd.color(color_name, f"{obj_name} and {_plddt_selection(color)}")
    if rebuild:
        cmd.rebuild()


def update_token_selection(
    selection_name: str,
    token_indices: Iterable[int],
    object_token_maps: Sequence[tuple[str, Sequence[Any]]],
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
    object_token_maps: Sequence[tuple[str, Sequence[Any]]],
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
    token_groups: Sequence[tuple[Iterable[int], str, Sequence[Any]]],
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
