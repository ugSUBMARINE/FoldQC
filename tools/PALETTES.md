# Adding Palettes

Palette handling is centralized in `palettes.py`. The GUI, PyMOL painter, and
matrix plots all resolve palette definitions through that file, so a new
curated palette should normally require one registry edit plus tests.

## Files to modify

- `palettes.py` — required. Add the palette definition to `PALETTE_SPECS`.
- `tests/test_palettes.py` — required for registry behavior.
- `tests/test_painter_palettes.py` — required when PyMOL behavior changes, such
  as adding custom RGB colors or relying on a native reverse palette.
- `tests/test_plots.py` — required when Matplotlib/matrix-plot behavior changes.
- `gui.py`, `painter.py`, and `plots.py` — usually not modified. These consume
  the registry through helper functions.

## PaletteSpec fields

Each palette is one `PaletteSpec` entry:

```python
PaletteSpec(
    key="blue_white_red",
    label="Blue-white-red",
    pymol="blue_white_red",
    pymol_reverse="red_white_blue",
    mpl=("blue", "white", "red"),
)
```

- `key`: Stable internal identifier stored in the GUI combo box.
- `label`: User-facing GUI label.
- `pymol`: Native PyMOL palette name or a PyMOL color list accepted by
  `cmd.spectrum`, such as `"green white"`.
- `pymol_reverse`: Optional native PyMOL reversed palette. Use this when PyMOL
  already provides one, for example `red_white_blue` or `rainbow_rev`.
- `mpl`: Matplotlib colormap name, such as `"viridis"`, or a tuple/list of
  colors used to build a Matplotlib colormap.
- `pymol_rgb_stops`: Optional sampled RGB stops for palettes that PyMOL does not
  provide natively.
- `curated`: Defaults to `True`. Set `False` only for palettes that should be
  resolvable by code but hidden from the GUI.

## Add a native PyMOL palette

Use this when PyMOL already knows the palette name.

```python
PaletteSpec(
    key="blue_white_red",
    label="Blue-white-red",
    pymol="blue_white_red",
    pymol_reverse="red_white_blue",
    mpl=("blue", "white", "red"),
)
```

If PyMOL has a native reverse palette, put it in `pymol_reverse`. The GUI
`Reverse` checkbox will then use that native name instead of generating a color
list.

## Add a simple color-list palette

Use this when PyMOL can render the palette as a space-separated color list.

```python
PaletteSpec(
    key="green_white",
    label="Green-white",
    pymol="green white",
    mpl=("green", "white"),
)
```

If `pymol_reverse` is omitted, reverse mode automatically reverses the color
list, so `green white` becomes `white green`.

## Add a Matplotlib-like palette that PyMOL lacks

Use sampled RGB stops when PyMOL does not provide the palette. `viridis` is the
current example.

Generate the boilerplate from any Matplotlib colormap with:

```bash
python3 tools/mpl_palette_to_foldqc.py viridis --samples 8
```

```python
MY_STOPS: tuple[ColorStop, ...] = (
    (0.267004, 0.004874, 0.329415),
    (0.993248, 0.906157, 0.143936),
)

PaletteSpec(
    key="my_palette",
    label="My palette",
    pymol="",
    mpl="viridis",
    pymol_rgb_stops=MY_STOPS,
)
```

`painter.py` registers these stops lazily with `cmd.set_color` and passes the
generated color names to `cmd.spectrum`. Reverse mode uses the same sampled
colors in reverse order.

## Test checklist

1. In `tests/test_palettes.py`, verify the new key appears in
   `iter_gui_palettes()` if it is curated.
2. Test `resolve_pymol_palette(key, reverse=False)` and
   `resolve_pymol_palette(key, reverse=True)`.
3. In `tests/test_plots.py`, add a matrix-plot test if the Matplotlib mapping is
   new or non-obvious.
4. In `tests/test_painter_palettes.py`, add a painter test if custom PyMOL
   colors should be registered or a native reversed palette should be used.

Run:

```bash
uv run python -m pytest tests/test_palettes.py tests/test_painter_palettes.py tests/test_plots.py
```
