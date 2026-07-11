"""
Painter
=======
Write per-token scalar values into PyMOL B-factors and apply spectrum
colouring.  All functions operate on a loaded PyMOL object via ``cmd``.
"""

from __future__ import annotations

import numpy as np

from .palettes import (
    BUILTIN_PALETTE_KEYS,
    PLDDT_CLASS_COLORS,
    categorical_color,
    resolve_pymol_palette,
)

# Backwards-compatible curated palette key list.
BUILTIN_PALETTES: list[str] = list(BUILTIN_PALETTE_KEYS)

# NaN tokens are painted with this PyMOL color
NAN_COLOR_DEFAULT = "grey70"

# Single viewport legend object used for continuous spectrum colouring.
COLORBAR_OBJECT_NAME = "foldqc_colorbar"


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


def token_bfactor_keys(token_map) -> list[tuple[str, str, str]]:
    """Return stable atom-property keys for a token map.

    Polymer residue tokens are keyed by ``(chain, resi, "")`` so every atom in
    the residue receives the token value. HETATM tokens are keyed by atom name.
    """
    return [
        (
            tok.chain_id,
            str(tok.res_num),
            tok.atom_name if tok.is_hetatm and tok.atom_name else "",
        )
        for tok in token_map
    ]


def _write_bfactors_bulk(
    obj_name: str,
    token_map,
    values: np.ndarray,
    *,
    scale: float = 1.0,
) -> None:
    """Write per-token values into B-factors with one PyMOL alter call."""
    from pymol import cmd, stored

    if len(values) != len(token_map):
        raise ValueError(
            f"values length {len(values)} does not match "
            f"token_map length {len(token_map)}."
        )

    bmap: dict[tuple[str, str, str], float] = {}
    for tok, key in zip(token_map, token_bfactor_keys(token_map)):
        v = float(values[tok.token_idx])
        bmap[key] = v * scale if np.isfinite(v) else -1.0

    stored.foldqc_bmap = bmap
    expr = (
        "b = stored.foldqc_bmap.get((chain, resi, name), "
        "stored.foldqc_bmap.get((chain, resi, ''), -1.0))"
    )
    cmd.alter(obj_name, expr)


def paint_property(
    obj_name: str,
    token_map,  # list[TokenInfo]
    values: np.ndarray,
    palette: str = "blue_white_red",
    reverse_palette: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    nan_color: str = NAN_COLOR_DEFAULT,
) -> tuple[float, float]:
    """Write *values* into B-factors of *obj_name* and apply spectrum coloring.

    Parameters
    ----------
    obj_name:
        PyMOL object name.
    token_map:
        Built by :func:`token_map.build_token_map`.
    values:
        Per-token scalar array, shape ``(N_tokens,)``.  ``np.nan`` entries are
        painted with *nan_color*.
    palette:
        Plugin palette key or PyMOL spectrum color name.
    reverse_palette:
        Reverse the palette direction. Native reversed PyMOL palettes are used
        when available.
    vmin, vmax:
        Explicit range for the color scale; auto-derived from finite values
        in *values* if ``None``.
    nan_color:
        PyMOL color name for tokens with ``np.nan`` value.

    Returns
    -------
    tuple[float, float]
        The ``(vmin, vmax)`` actually used for the color scale.
    """
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
    token_map,
    values: np.ndarray,
    palette: str = "blue_white_red",
    reverse_palette: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    nan_color: str = NAN_COLOR_DEFAULT,
    rebuild: bool = True,
) -> tuple[float, float]:
    """Bulk-write per-token B-factors, then apply spectrum coloring."""
    from pymol import cmd

    vmin, vmax = _resolve_color_range(values, vmin, vmax)
    _write_bfactors_bulk(obj_name, token_map, values)
    resolved = resolve_pymol_palette(palette, reverse=reverse_palette)
    for color in resolved.custom_colors:
        cmd.set_color(color.name, list(color.rgb))
    cmd.spectrum("b", resolved.palette, obj_name, minimum=vmin, maximum=vmax)
    cmd.color(nan_color, f"{obj_name} and b < 0")
    if rebuild:
        cmd.rebuild()
    return vmin, vmax


def _category_color_name(label: int) -> str:
    """Return the registered PyMOL color name for one category label."""
    if label >= 0:
        return f"foldqc_category_{label:03d}"
    return f"foldqc_category_m{-label:03d}"


def category_rgb(label: int) -> tuple[float, float, float]:
    """Return the categorical RGB triple used for label coloring."""
    return categorical_color(int(label))


def paint_categorical_labels_bulk(
    obj_name: str,
    token_map,
    values: np.ndarray,
    nan_color: str = NAN_COLOR_DEFAULT,
    rebuild: bool = True,
) -> tuple[float, float]:
    """Bulk-write integer labels, then apply categorical colors by label."""
    from pymol import cmd

    arr = np.asarray(values, dtype=np.float32)
    vmin, vmax = _resolve_color_range(arr)
    _write_bfactors_bulk(obj_name, token_map, arr)

    finite = arr[np.isfinite(arr)]
    labels = sorted({int(round(float(value))) for value in finite})
    for label in labels:
        color_name = _category_color_name(label)
        cmd.set_color(color_name, list(categorical_color(label)))
        cmd.color(color_name, f"{obj_name} and b = {label:g}")
    cmd.color(nan_color, f"{obj_name} and b < 0")
    if rebuild:
        cmd.rebuild()
    return vmin, vmax


def delete_colorbar(name: str = COLORBAR_OBJECT_NAME) -> None:
    """Remove the plugin's PyMOL colorbar object if it exists."""
    from pymol import cmd

    cmd.delete(name)


def _split_palette_names(palette: str) -> list[str]:
    """Split a PyMOL spectrum palette into colour names."""
    if " " in palette:
        return [part for part in palette.split() if part]
    return [part for part in palette.split("_") if part]


def _ramp_colors_for_palette(
    palette: str, reverse_palette: bool
) -> list[str | list[float]]:
    """Return explicit color stops for a PyMOL ramp object."""
    resolved = resolve_pymol_palette(palette, reverse=reverse_palette)
    if resolved.custom_colors:
        return [list(color.rgb) for color in resolved.custom_colors]

    colors = _split_palette_names(resolved.palette)
    if len(colors) == 1:
        colors.append(colors[0])
    return colors


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
    """Replace the continuous-colouring PyMOL ramp object."""
    from pymol import cmd

    delete_colorbar(name)
    colors = _ramp_colors_for_palette(palette, reverse_palette)
    range_values = np.linspace(float(vmin), float(vmax), len(colors)).tolist()
    cmd.ramp_new(name, None, range_values, colors, quiet=1)


def reset_bfactors(obj_name: str, value: float = 100.0) -> None:
    """Reset all B-factors of *obj_name* to *value*.

    Useful before loading a different property, or to restore the original
    pLDDT × 100 B-factors (pass ``value=100.0``).
    """
    from pymol import cmd

    cmd.alter(obj_name, f"b = {value:.2f}")
    cmd.rebuild()


def get_representative_coords(
    obj_name: str,
    token_map,  # list[TokenInfo]
) -> np.ndarray:
    """Return representative-atom coordinates for each token, shape ``(N, 3)``.

    - Polymer residues: Cα (protein) or C1′ (nucleotide), falling back to the
      first atom in the residue if neither is found.
    - Ligand atoms: the single atom of that token.
    """
    from pymol import cmd

    coords = np.zeros((len(token_map), 3), dtype=np.float32)

    model = cmd.get_model(obj_name)
    # Build a fast lookup: (chain, resi, name) → (x, y, z)
    atom_coords: dict[tuple[str, int, str], tuple[float, float, float]] = {}
    # Also store first atom per residue as fallback
    first_atom: dict[tuple[str, int], tuple[float, float, float]] = {}

    for atom in model.atom:
        key = (atom.chain, int(atom.resi), atom.name)
        atom_coords[key] = tuple(atom.coord)
        res_key = (atom.chain, int(atom.resi))
        if res_key not in first_atom:
            first_atom[res_key] = tuple(atom.coord)

    POLYMER_REPR = {"CA", "C1'", "C1*"}  # representative atom names

    for tok in token_map:
        c, r = tok.chain_id, tok.res_num
        if tok.is_hetatm and tok.atom_name is not None:
            key = (c, r, tok.atom_name)
            xyz = atom_coords.get(key, first_atom.get((c, r), (0.0, 0.0, 0.0)))
        else:
            xyz = None
            for repr_name in POLYMER_REPR:
                xyz = atom_coords.get((c, r, repr_name))
                if xyz is not None:
                    break
            if xyz is None:
                xyz = first_atom.get((c, r), (0.0, 0.0, 0.0))
        coords[tok.token_idx] = xyz

    return coords


# ---------------------------------------------------------------------------
# pLDDT quality-class colouring (AlphaFold colour scheme)
# ---------------------------------------------------------------------------


def paint_plddt_class_coloring(
    obj_name: str,
    values: np.ndarray | None = None,
    token_map=None,
    rebuild: bool = True,
) -> None:
    """Apply the 4-class AlphaFold pLDDT colour scheme to *obj_name*.

    Colours are applied to the B-factor column (pLDDT × 100, range 0–100).
    If *values* (per-token pLDDT in 0–1 range) and *token_map* are provided,
    values × 100 are written to B-factors first; otherwise the existing
    B-factor column is used directly.

    Colour key
    ----------
    Blue       (plddt >= 90) — very high confidence
    Light blue (70 <= plddt < 90) — high confidence
    Yellow     (50 <= plddt  < 70) — low confidence
    Orange     (plddt < 50) — very low confidence
    """
    from pymol import cmd

    if values is not None and token_map is not None:
        _write_bfactors_bulk(obj_name, token_map, values, scale=100.0)

    # Register the four colours with PyMOL
    for color in PLDDT_CLASS_COLORS:
        cmd.set_color(color.pymol_name, list(color.rgb))
        cmd.color(color.pymol_name, f"{obj_name} and {color.bfactor_selection}")
    if rebuild:
        cmd.rebuild()
