#!/usr/bin/env python3
"""Generate FoldQC palette registry snippets from Matplotlib colormaps."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence


def _key_from_cmap(cmap_name: str) -> str:
    key = re.sub(r"[^0-9A-Za-z]+", "_", cmap_name).strip("_").lower()
    return key or "palette"


def _constant_name(key: str) -> str:
    return f"{re.sub(r'[^0-9A-Za-z]+', '_', key).strip('_').upper()}_STOPS"


def _label_from_key(key: str) -> str:
    return key.replace("_", " ").title()


def _sample_positions(count: int) -> list[float]:
    if count < 2:
        raise ValueError("sample count must be at least 2")
    return [idx / (count - 1) for idx in range(count)]


def _format_rgb(rgb: Sequence[float]) -> str:
    red, green, blue = rgb[:3]
    return f"({red:.6f}, {green:.6f}, {blue:.6f})"


def _render_snippet(
    *,
    cmap_name: str,
    key: str,
    label: str,
    constant_name: str,
    stops: Sequence[Sequence[float]],
    curated: bool,
) -> str:
    lines = [f"{constant_name}: tuple[ColorStop, ...] = ("]
    lines.extend(f"    {_format_rgb(stop)}," for stop in stops)
    lines.extend(
        [
            ")",
            "",
            "PaletteSpec(",
            f'    key="{key}",',
            f'    label="{label}",',
            '    pymol="",',
            f'    mpl="{cmap_name}",',
            f"    pymol_rgb_stops={constant_name},",
        ]
    )
    if not curated:
        lines.append("    curated=False,")
    lines.append(")")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a Matplotlib colormap and print FoldQC ColorStop and "
            "PaletteSpec code for palettes.py."
        )
    )
    parser.add_argument(
        "cmap",
        help='Matplotlib colormap name, for example "viridis" or "magma".',
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=8,
        help="Number of RGB stops to sample, including both endpoints. Default: 8.",
    )
    parser.add_argument(
        "--key",
        help="PaletteSpec key. Defaults to a lowercase slug of the colormap name.",
    )
    parser.add_argument(
        "--label",
        help="PaletteSpec label. Defaults to a title-cased version of the key.",
    )
    parser.add_argument(
        "--constant",
        help="Name for the ColorStop tuple. Defaults to KEY_STOPS.",
    )
    parser.add_argument(
        "--not-curated",
        action="store_true",
        help="Emit curated=False for palettes that should not appear in the GUI.",
    )
    return parser


# def _ensure_mpl_configdir() -> None:
#     """Keep Matplotlib cache/config warnings out of generated snippets."""
#     if "MPLCONFIGDIR" in os.environ:
#         return

#     config_dir = os.path.join(tempfile.gettempdir(), "foldqc-matplotlib-cache")
#     os.makedirs(config_dir, exist_ok=True)
#     os.environ["MPLCONFIGDIR"] = config_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        # _ensure_mpl_configdir()
        import matplotlib
    except ImportError as exc:
        parser.exit(2, f"error: Matplotlib is required: {exc}\n")

    try:
        positions = _sample_positions(args.samples)
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")

    try:
        if hasattr(matplotlib, "colormaps"):
            cmap = matplotlib.colormaps[args.cmap]
        else:
            from matplotlib import cm

            cmap = cm.get_cmap(args.cmap)  # type: ignore[attr-defined]
    except (KeyError, ValueError) as exc:
        parser.exit(2, f"error: unknown Matplotlib colormap {args.cmap!r}: {exc}\n")

    key = args.key or _key_from_cmap(args.cmap)
    label = args.label or _label_from_key(key)
    constant_name = args.constant or _constant_name(key)
    stops = [cmap(position) for position in positions]

    print(
        _render_snippet(
            cmap_name=args.cmap,
            key=key,
            label=label,
            constant_name=constant_name,
            stops=stops,
            curated=not args.not_curated,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
