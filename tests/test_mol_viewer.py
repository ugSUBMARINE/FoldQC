from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import numpy as np
from FoldQC import mol_viewer
from FoldQC.token_map import TokenInfo


def _token(
    idx: int,
    *,
    chain: str = "A",
    hetatm: bool = False,
    atom_name: str | None = None,
) -> TokenInfo:
    return TokenInfo(
        token_idx=idx,
        chain_id=chain,
        res_num=idx + 1,
        res_name="LIG" if hetatm else "ALA",
        is_hetatm=hetatm,
        atom_name=atom_name,
    )


def test_only_viewer_boundary_and_bootstrap_modules_import_pymol() -> None:
    root = Path(__file__).resolve().parents[1]
    allowed = {"__init__.py", "compat.py", "mol_viewer.py"}
    violations = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "pymol"
            ):
                if path.name not in allowed:
                    violations.append((path.name, node.lineno))
            elif isinstance(node, ast.Import):
                if any(alias.name.startswith("pymol") for alias in node.names):
                    if path.name not in allowed:
                        violations.append((path.name, node.lineno))
    assert violations == []


def test_object_discovery_and_ensure_structure_object(monkeypatch) -> None:
    cmd = types.SimpleNamespace(loads=[], enabled=[])

    def get_names(kind, *args, **kwargs):
        if kind == "all":
            return ["model_0", "ensemble"]
        if kwargs.get("enabled_only") or args:
            return []
        return ["model_0", mol_viewer.COLORBAR_OBJECT_NAME]

    cmd.get_names = get_names
    cmd.enable = lambda name: cmd.enabled.append(name)
    cmd.load = lambda *args, **kwargs: cmd.loads.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "pymol", types.SimpleNamespace(cmd=cmd))

    assert mol_viewer.get_object_list(additional_names=["ensemble"]) == [
        "ensemble",
        "model_0",
    ]
    assert not mol_viewer.ensure_structure_object("/tmp/model.cif", "model_0")
    assert cmd.enabled == ["model_0"]
    assert mol_viewer.ensure_structure_object("/tmp/other.cif", "model_1")
    assert cmd.loads == [(("/tmp/other.cif", "model_1"), {"quiet": 1, "zoom": 1})]


def test_tokens_within_distance_builds_backend_expression(monkeypatch) -> None:
    captured = []
    model = types.SimpleNamespace(
        atom=[types.SimpleNamespace(chain="A", resi="2", name="CA", hetatm=False)]
    )
    cmd = types.SimpleNamespace(
        get_model=lambda expression: captured.append(expression) or model
    )
    monkeypatch.setitem(sys.modules, "pymol", types.SimpleNamespace(cmd=cmd))
    token_map = [_token(0), _token(1)]

    assert mol_viewer.tokens_within_distance(
        token_map, "model_0", "resname LIG", 7.5
    ) == [1]
    assert "byres ((model_0) within 7.5" in captured[0]
    assert "not ((resname LIG) and (model_0))" in captured[0]


def test_coordinates_transform_and_plot_selection_are_viewer_operations(
    monkeypatch,
) -> None:
    atoms = [
        types.SimpleNamespace(chain="A", resi="1", name="CA", coord=(1, 2, 3)),
        types.SimpleNamespace(chain="L", resi="2", name="C1", coord=(4, 5, 6)),
    ]
    cmd = types.SimpleNamespace(altered=[], selections=[], enabled=[], refreshes=0)
    cmd.get_model = lambda _name: types.SimpleNamespace(atom=atoms)
    cmd.alter_state = lambda *args: cmd.altered.append(args)
    cmd.select = lambda *args: cmd.selections.append(args)
    cmd.enable = lambda name: cmd.enabled.append(name)
    cmd.refresh = lambda: setattr(cmd, "refreshes", cmd.refreshes + 1)
    stored = types.SimpleNamespace()
    monkeypatch.setitem(
        sys.modules, "pymol", types.SimpleNamespace(cmd=cmd, stored=stored)
    )
    token_map = [
        _token(0),
        _token(1, chain="L", hetatm=True, atom_name="C1"),
    ]

    np.testing.assert_array_equal(
        mol_viewer.get_representative_coords("model", token_map),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    )
    mol_viewer.transform_object("model", np.eye(3), np.array([1.0, 0.0, 0.0]))
    assert cmd.altered[0][0:2] == (1, "model")

    mol_viewer.update_token_selection(
        "foldqc_plot_selection", [0], [("model", token_map)]
    )
    assert cmd.selections == [
        ("foldqc_plot_selection", "(%model and polymer and chain A)")
    ]
    assert cmd.enabled == ["foldqc_plot_selection"]
    assert cmd.refreshes == 1
