"""
Palette registry
================
Shared palette definitions for PyMOL colouring and Matplotlib plots.

The GUI intentionally exposes a curated subset.  Add a palette by appending a
single :class:`PaletteSpec` to ``PALETTE_SPECS``; painter and plots resolve it
through this module.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Sequence

ColorStop = tuple[float, float, float]
MplSpec = str | Sequence[str]


@dataclass(frozen=True)
class ColorDef:
    """One custom PyMOL colour that must be registered before ``spectrum``."""

    name: str
    rgb: ColorStop


@dataclass(frozen=True)
class PaletteResolution:
    """Resolved PyMOL palette string plus optional custom colour definitions."""

    palette: str
    custom_colors: tuple[ColorDef, ...] = ()


@dataclass(frozen=True)
class PaletteSpec:
    """Declarative palette definition."""

    key: str
    label: str
    pymol: str
    pymol_reverse: str | None = None
    mpl: MplSpec | None = None
    pymol_rgb_stops: tuple[ColorStop, ...] = ()
    curated: bool = True


@dataclass(frozen=True)
class PlddtClassColor:
    """One AlphaFold pLDDT quality-class color definition."""

    key: str
    label: str
    pymol_name: str
    rgb: ColorStop
    bfactor_selection: str


VIRIDIS_STOPS: tuple[ColorStop, ...] = (
    (0.267004, 0.004874, 0.329415),
    (0.275191, 0.194905, 0.496005),
    (0.212395, 0.359683, 0.551710),
    (0.153364, 0.497000, 0.557724),
    (0.122312, 0.633153, 0.530398),
    (0.288921, 0.758394, 0.428426),
    (0.626579, 0.854645, 0.223353),
    (0.993248, 0.906157, 0.143936),
)

MAGMA_STOPS: tuple[ColorStop, ...] = (
    (0.001462, 0.000466, 0.013866),
    (0.135053, 0.068391, 0.315000),
    (0.372116, 0.092816, 0.499053),
    (0.594508, 0.175701, 0.501241),
    (0.828886, 0.262229, 0.430644),
    (0.973381, 0.461520, 0.361965),
    (0.997341, 0.733545, 0.505167),
    (0.987053, 0.991438, 0.749504),
)

PLASMA_STOPS: tuple[ColorStop, ...] = (
    (0.050383, 0.029803, 0.527975),
    (0.325150, 0.006915, 0.639512),
    (0.546157, 0.038954, 0.647010),
    (0.723444, 0.196158, 0.538981),
    (0.859750, 0.360588, 0.406917),
    (0.955470, 0.533093, 0.285490),
    (0.994495, 0.740880, 0.166335),
    (0.940015, 0.975158, 0.131326),
)

INFERNO_STOPS: tuple[ColorStop, ...] = (
    (0.001462, 0.000466, 0.013866),
    (0.155850, 0.044559, 0.325338),
    (0.397674, 0.083257, 0.433183),
    (0.621685, 0.164184, 0.388781),
    (0.832299, 0.283913, 0.257383),
    (0.961293, 0.488716, 0.084289),
    (0.981173, 0.759135, 0.156863),
    (0.988362, 0.998364, 0.644924),
)

CIVIDIS_STOPS: tuple[ColorStop, ...] = (
    (0.000000, 0.135112, 0.304751),
    (0.130669, 0.231458, 0.432840),
    (0.298421, 0.332247, 0.423973),
    (0.425120, 0.431334, 0.447692),
    (0.555393, 0.537807, 0.471147),
    (0.695985, 0.648334, 0.440072),
    (0.849223, 0.771947, 0.359729),
    (0.995737, 0.909344, 0.217772),
)

CATEGORICAL_STOPS: tuple[ColorStop, ...] = (
    (0.1216, 0.4667, 0.7059),
    (1.0000, 0.4980, 0.0549),
    (0.1725, 0.6275, 0.1725),
    (0.8392, 0.1529, 0.1569),
    (0.5804, 0.4039, 0.7412),
    (0.5490, 0.3373, 0.2941),
    (0.8902, 0.4667, 0.7608),
    (0.4980, 0.4980, 0.4980),
    (0.7373, 0.7412, 0.1333),
    (0.0902, 0.7451, 0.8118),
    (0.6824, 0.7804, 0.9098),
    (1.0000, 0.7333, 0.4706),
    (0.5961, 0.8745, 0.5412),
    (1.0000, 0.5961, 0.5882),
    (0.7725, 0.6902, 0.8353),
    (0.7686, 0.6118, 0.5804),
    (0.9686, 0.7137, 0.8235),
    (0.7804, 0.7804, 0.7804),
    (0.8588, 0.8588, 0.5529),
    (0.6196, 0.8549, 0.8980),
)

PLDDT_CLASS_COLORS: tuple[PlddtClassColor, ...] = (
    PlddtClassColor(
        key="very_high",
        label="very high",
        pymol_name="plddt_very_high",
        rgb=(0.000, 0.326, 0.843),
        bfactor_selection="(b>90 or b=90)",
    ),
    PlddtClassColor(
        key="high",
        label="high",
        pymol_name="plddt_high",
        rgb=(0.341, 0.792, 0.976),
        bfactor_selection="((b<90 and b>70) or b=70)",
    ),
    PlddtClassColor(
        key="low",
        label="low",
        pymol_name="plddt_low",
        rgb=(1.000, 0.859, 0.071),
        bfactor_selection="((b<70 and b>50) or b=50)",
    ),
    PlddtClassColor(
        key="very_low",
        label="very low",
        pymol_name="plddt_very_low",
        rgb=(1.000, 0.494, 0.271),
        bfactor_selection="(b<50)",
    ),
)

PLDDT_CLASS_BAR_COLORS: tuple[ColorStop, ...] = tuple(
    color.rgb for color in reversed(PLDDT_CLASS_COLORS)
)

PALETTE_SPECS: tuple[PaletteSpec, ...] = (
    PaletteSpec(
        key="viridis",
        label="Viridis",
        pymol="",
        mpl="viridis",
        pymol_rgb_stops=VIRIDIS_STOPS,
        curated=True,
    ),
    PaletteSpec(
        key="magma",
        label="Magma",
        pymol="",
        mpl="magma",
        pymol_rgb_stops=MAGMA_STOPS,
        curated=False,  # The dark end of Magma is too dark for GUI display
    ),
    PaletteSpec(
        key="plasma",
        label="Plasma",
        pymol="",
        mpl="plasma",
        pymol_rgb_stops=PLASMA_STOPS,
        curated=True,
    ),
    PaletteSpec(
        key="inferno",
        label="Inferno",
        pymol="",
        mpl="inferno",
        pymol_rgb_stops=INFERNO_STOPS,
        curated=False,  # The dark end of Inferno is too dark for GUI display
    ),
    PaletteSpec(
        key="cividis",
        label="Cividis",
        pymol="",
        mpl="cividis",
        pymol_rgb_stops=CIVIDIS_STOPS,
        curated=True,
    ),
    PaletteSpec(
        key="white_blue",
        label="Blues",
        pymol="white_blue",
        pymol_reverse="blue_white",
        mpl="Blues",
    ),
    PaletteSpec(
        key="white_red",
        label="Reds",
        pymol="white_red",
        pymol_reverse="red_white",
        mpl="Reds",
    ),
    PaletteSpec(
        key="white_green",
        label="Greens",
        pymol="white_green",
        pymol_reverse="green_white",
        mpl="Greens",
    ),
    PaletteSpec(
        key="blue_white_red",
        label="Blue-white-red",
        pymol="blue_white_red",
        pymol_reverse="red_white_blue",
        mpl="coolwarm",
    ),
    PaletteSpec(
        key="green_white_red",
        label="Green-white-red",
        pymol="green_white_red",
        pymol_reverse="red_white_green",
        mpl=("green", "white", "red"),
        curated=False,
    ),
    PaletteSpec(
        key="cyan_white_magenta",
        label="Cyan-white-magenta",
        pymol="cyan_white_magenta",
        pymol_reverse="magenta_white_cyan",
        mpl=("cyan", "white", "magenta"),
        curated=False,
    ),
    PaletteSpec(
        key="yellow_white_magenta",
        label="Yellow-white-magenta",
        pymol="yellow_white_magenta",
        pymol_reverse="magenta_white_yellow",
        mpl=("yellow", "white", "magenta"),
        curated=False,
    ),
    PaletteSpec(
        key="rainbow",
        label="Rainbow",
        pymol="rainbow",
        pymol_reverse="rainbow_rev",
        mpl="rainbow",
        curated=False,  # Rainbow is not perceptually uniform, so not in the GUI
    ),
    PaletteSpec(
        key="rainbow2",
        label="Rainbow 2",
        pymol="rainbow2",
        pymol_reverse="rainbow2_rev",
        mpl="turbo",
        curated=False,  # Rainbow is not perceptually uniform, so not in the GUI
    ),
)

_SPECS_BY_KEY = {spec.key: spec for spec in PALETTE_SPECS}

BUILTIN_PALETTE_KEYS: list[str] = [spec.key for spec in PALETTE_SPECS if spec.curated]


def iter_gui_palettes() -> tuple[PaletteSpec, ...]:
    """Return curated palettes in GUI display order."""
    return tuple(spec for spec in PALETTE_SPECS if spec.curated)


def palette_keys() -> list[str]:
    """Return curated palette keys."""
    return list(BUILTIN_PALETTE_KEYS)


def categorical_color(label: int) -> ColorStop:
    """Return a deterministic categorical RGB triple for an integer label."""
    label = int(label)
    if 0 <= label < len(CATEGORICAL_STOPS):
        return CATEGORICAL_STOPS[label]
    hue = (0.61803398875 * float(label + 1)) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.65, 0.95)


def _resolve_spec(key: str, reverse: bool) -> tuple[PaletteSpec | None, bool]:
    """Return ``(spec, effective_reverse)`` for a palette key."""
    spec = _SPECS_BY_KEY.get(key)
    if spec is not None:
        return spec, reverse
    return None, reverse


def _reverse_color_list(colors: str) -> str:
    """Reverse a space or underscore separated colour list."""
    sep = " " if " " in colors else "_"
    parts = [part for part in colors.split(sep) if part]
    return " ".join(reversed(parts))


def _custom_color_defs(
    key: str,
    stops: tuple[ColorStop, ...],
    reverse: bool,
) -> tuple[ColorDef, ...]:
    ordered = tuple(reversed(stops)) if reverse else stops
    direction = "r" if reverse else "f"
    return tuple(
        ColorDef(name=f"foldqc_{key}_{direction}_{idx:02d}", rgb=rgb)
        for idx, rgb in enumerate(ordered)
    )


def resolve_pymol_palette(key: str, reverse: bool = False) -> PaletteResolution:
    """Resolve a palette key to a PyMOL ``cmd.spectrum`` palette string."""
    spec, effective_reverse = _resolve_spec(key, reverse)
    if spec is None:
        return PaletteResolution(_reverse_color_list(key) if reverse else key)

    if spec.pymol_rgb_stops:
        custom_colors = _custom_color_defs(
            spec.key, spec.pymol_rgb_stops, effective_reverse
        )
        return PaletteResolution(
            palette=" ".join(color.name for color in custom_colors),
            custom_colors=custom_colors,
        )

    if effective_reverse and spec.pymol_reverse:
        return PaletteResolution(spec.pymol_reverse)
    if effective_reverse:
        return PaletteResolution(_reverse_color_list(spec.pymol))
    return PaletteResolution(spec.pymol)


def resolve_matplotlib_cmap(key: str, reverse: bool = False):
    """Return ``(cmap, used_fallback)`` for a palette key.

    Matplotlib is imported lazily so PyMOL startup and painter usage do not
    acquire a plotting dependency.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    spec, effective_reverse = _resolve_spec(key, reverse)
    if spec is None:
        cmap_name = f"{key}_r" if reverse and not key.endswith("_r") else key
        try:
            return plt.get_cmap(cmap_name), False
        except ValueError:
            return plt.get_cmap("viridis"), True

    mpl_spec = spec.mpl
    if mpl_spec is None:
        mpl_spec = tuple(spec.pymol.replace("_", " ").split())

    if isinstance(mpl_spec, str):
        cmap_name = f"{mpl_spec}_r" if effective_reverse else mpl_spec
        try:
            return plt.get_cmap(cmap_name), False
        except ValueError:
            return plt.get_cmap("viridis"), True

    colors = list(reversed(mpl_spec)) if effective_reverse else list(mpl_spec)
    return (
        LinearSegmentedColormap.from_list(f"foldqc_{spec.key}", colors),
        False,
    )
