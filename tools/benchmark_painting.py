#!/usr/bin/env python3
"""Non-gating real-PyMOL benchmark for FoldQC ensemble painting.

Run from the repository root with a Python environment that can import PyMOL:

    uv run python tools/benchmark_painting.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pymol
from chempy import Atom
from chempy.models import Indexed
from pymol import cmd, stored

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.mol_viewer import (  # noqa: E402
    PaintTarget,
    ensure_object_paint_mapping,
    paint_properties_bulk,
    prepare_object_paint_mapping,
)
from FoldQC.palettes import VIRIDIS_STOPS  # noqa: E402
from FoldQC.token_map import TokenInfo, TokenMap  # noqa: E402


def _build_object(name: str, residues: int, atoms_per_residue: int) -> None:
    model = Indexed()
    atom_names = ("N", "CA", "C", "O", "CB", "CG", "CD", "CE")
    for residue in range(residues):
        for atom_offset in range(atoms_per_residue):
            atom = Atom()
            atom.name = atom_names[atom_offset % len(atom_names)]
            atom.resn = "ALA"
            atom.chain = "A"
            atom.resi = str(residue + 1)
            atom.hetatm = 0
            atom.coord = [float(residue), float(atom_offset), 0.0]
            model.atom.append(atom)
    cmd.load_model(model, name, quiet=1)


def _best_ms(func, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        timings.append((time.perf_counter() - start) * 1000.0)
    return min(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=int, default=5)
    parser.add_argument("--residues", type=int, default=1000)
    parser.add_argument("--atoms-per-residue", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    pymol.finish_launching(["pymol", "-cq"])
    token_map = TokenMap(
        tuple(
            TokenInfo(index, "A", index + 1, "ALA", False, None)
            for index in range(args.residues)
        )
    )
    names = [f"foldqc_benchmark_{index}" for index in range(args.models)]
    values = [
        np.linspace(0.0, 1.0, args.residues, dtype=np.float32)
        + index / max(args.models, 1) * 0.05
        for index in range(args.models)
    ]
    for name in names:
        _build_object(name, args.residues, args.atoms_per_residue)

    mappings = [prepare_object_paint_mapping(name, token_map) for name in names]
    legacy_colors = []
    for index, rgb in enumerate(VIRIDIS_STOPS):
        color_name = f"foldqc_benchmark_viridis_{index:02d}"
        cmd.set_color(color_name, list(rgb))
        legacy_colors.append(color_name)
    legacy_palette = " ".join(legacy_colors)

    def legacy_paint() -> None:
        for name, array in zip(names, values):
            stored.foldqc_benchmark_bmap = {
                ("A", str(index + 1), ""): float(value)
                for index, value in enumerate(array)
            }
            cmd.alter(
                name,
                "b = stored.foldqc_benchmark_bmap.get((chain, resi, name), "
                "stored.foldqc_benchmark_bmap.get((chain, resi, ''), -1.0))",
            )
            cmd.spectrum("b", legacy_palette, name, minimum=0.0, maximum=1.05)

    def optimized_paint() -> None:
        targets = []
        for index, (name, array, mapping) in enumerate(zip(names, values, mappings)):
            mapping, _rebuilt = ensure_object_paint_mapping(name, token_map, mapping)
            mappings[index] = mapping
            targets.append(PaintTarget(name, token_map, array, mapping))
        paint_properties_bulk(
            targets,
            palette="viridis",
            vmin=0.0,
            vmax=1.05,
            rebuild=False,
        )

    cmd.set("suspend_updates", "on")
    try:
        legacy_ms = _best_ms(legacy_paint, args.repeats)
        first_start = time.perf_counter()
        optimized_paint()
        first_ms = (time.perf_counter() - first_start) * 1000.0
        repeated_ms = _best_ms(optimized_paint, args.repeats)
    finally:
        cmd.set("suspend_updates", "off")
        cmd.rebuild()

    speedup = legacy_ms / repeated_ms if repeated_ms else float("inf")
    atom_count = args.models * args.residues * args.atoms_per_residue
    print(f"Objects: {args.models}; total atoms: {atom_count:,}")
    print(f"Legacy best: {legacy_ms:.2f} ms")
    print(f"Optimized first: {first_ms:.2f} ms")
    print(f"Optimized repeated best: {repeated_ms:.2f} ms")
    print(f"Repeated-paint speedup: {speedup:.2f}x")
    print("Target met" if speedup >= 2.0 else "Target not met")
    cmd.quit()


if __name__ == "__main__":
    main()
