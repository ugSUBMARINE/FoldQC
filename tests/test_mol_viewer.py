from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import numpy as np
from FoldQC import mol_viewer
from FoldQC.gui_services import ManagedColorbar
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap


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
        residue_id=ResidueId(idx + 1),
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


def test_incremental_ensemble_viewer_helpers_preserve_existing_objects(
    monkeypatch,
) -> None:
    objects = {"model_0"}
    all_names = {"model_0", "ensemble"}
    group_calls = []
    deleted = []
    loads = []

    def get_names(kind, *_args, **_kwargs):
        return sorted(all_names if kind == "all" else objects)

    def load(path, name, **kwargs):
        loads.append((path, name, kwargs))
        objects.add(name)
        all_names.add(name)

    def delete(name):
        deleted.append(name)
        objects.discard(name)
        all_names.discard(name)

    cmd = types.SimpleNamespace(
        get_names=get_names,
        get_object_list=lambda selection: (
            ["model_0"] if selection == "(ensemble)" else []
        ),
        load=load,
        group=lambda *args: group_calls.append(args),
        delete=delete,
    )
    monkeypatch.setitem(sys.modules, "pymol", types.SimpleNamespace(cmd=cmd))

    assert not mol_viewer.load_structure_object_if_missing(
        "/tmp/model_0.cif", "model_0"
    )
    assert mol_viewer.load_structure_object_if_missing("/tmp/model_1.cif", "model_1")
    assert loads == [("/tmp/model_1.cif", "model_1", {"quiet": 1, "zoom": 0})]
    assert mol_viewer.get_group_members("ensemble") == ("model_0",)

    mol_viewer.add_objects_to_group("ensemble", ("model_0", "model_1"))
    mol_viewer.remove_objects_from_group("ensemble", ("model_1",))
    assert group_calls == [
        ("ensemble", "model_0", "add"),
        ("ensemble", "model_1", "add"),
        ("ensemble", "model_1", "remove"),
    ]

    mol_viewer.delete_viewer_names(("missing", "model_1"))
    assert deleted == ["model_1"]


def test_managed_colorbar_replacement_never_renames_pymol_ramps(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        mol_viewer,
        "show_colorbar",
        lambda palette, reverse, vmin, vmax, **kwargs: calls.append(
            ("show", palette, reverse, vmin, vmax, kwargs["object_names"])
        ),
    )
    monkeypatch.setattr(
        mol_viewer, "delete_colorbar", lambda: calls.append(("delete",))
    )
    viewer = mol_viewer.PyMOLViewer()
    first = ManagedColorbar("white_blue", False, 0.0, 1.0, ("model_0",))
    second = ManagedColorbar("white_red", True, 1.0, 2.0, ("model_0",))

    viewer.replace_managed_colorbar(first)
    viewer.replace_managed_colorbar(second)
    viewer.replace_managed_colorbar(None)

    assert viewer.get_managed_colorbar() is None
    assert calls == [
        ("show", "white_blue", False, 0.0, 1.0, ("model_0",)),
        ("show", "white_red", True, 1.0, 2.0, ("model_0",)),
        ("delete",),
    ]
    assert not hasattr(viewer, "rename")


def test_object_paint_mapping_handles_polymer_ligand_order_and_sparse_indices(
    monkeypatch,
) -> None:
    atoms = [
        types.SimpleNamespace(
            index=4,
            chain="A",
            resi="1",
            resn="ALA",
            name="N",
            hetatm=False,
        ),
        types.SimpleNamespace(
            index=2,
            chain="L",
            resi="2",
            resn="LIG",
            name="C2",
            hetatm=True,
        ),
        types.SimpleNamespace(
            index=7,
            chain="A",
            resi="1",
            resn="ALA",
            name="CA",
            hetatm=False,
        ),
        types.SimpleNamespace(
            index=5,
            chain="L",
            resi="2",
            resn="LIG",
            name="C1",
            hetatm=True,
        ),
    ]
    cmd = types.SimpleNamespace(
        get_model=lambda _name: types.SimpleNamespace(atom=atoms),
        index=lambda _name: [("obj", atom.index) for atom in atoms],
    )
    monkeypatch.setitem(sys.modules, "pymol", types.SimpleNamespace(cmd=cmd))
    token_map = TokenMap(
        (
            _token(0),
            TokenInfo(1, "L", 2, "LIG", True, "C1"),
            TokenInfo(2, "L", 2, "LIG", True, "C2"),
        )
    )

    mapping = mol_viewer.prepare_object_paint_mapping("obj", token_map)

    assert mapping.atom_index_fingerprint == (4, 2, 7, 5)
    assert mapping.max_atom_index == 7
    assert mapping.atom_token_indices.tolist() == [-1, -1, 2, -1, 0, 1, -1, 0]
    assert mapping.overlap.matched_prediction_tokens == 3
    assert mol_viewer.object_paint_mapping_is_valid(mapping)


def test_paint_mapping_uses_insertion_code_residue_name_and_filters_hydrogen(
    monkeypatch,
) -> None:
    atoms = [
        types.SimpleNamespace(
            index=1,
            chain="A",
            resi="42A",
            resn="GLY",
            name="CA",
            hetatm=False,
        ),
        types.SimpleNamespace(
            index=2,
            chain="L",
            resi="42A",
            resn="LIG",
            name="C1",
            symbol="C",
            hetatm=True,
        ),
        types.SimpleNamespace(
            index=3,
            chain="L",
            resi="42A",
            resn="LIG",
            name="H1",
            symbol="H",
            hetatm=True,
        ),
    ]
    cmd = types.SimpleNamespace(
        get_model=lambda _name: types.SimpleNamespace(atom=atoms),
        index=lambda _name: [("obj", atom.index) for atom in atoms],
    )
    monkeypatch.setitem(sys.modules, "pymol", types.SimpleNamespace(cmd=cmd))
    token_map = TokenMap(
        (
            TokenInfo(0, "A", ResidueId(42, "A"), "GLY", False, None),
            TokenInfo(1, "L", ResidueId(42, "A"), "LIG", True, "C1"),
        )
    )

    mapping = mol_viewer.prepare_object_paint_mapping("obj", token_map)

    assert mapping.atom_token_indices.tolist() == [-1, 0, 1, -1]
    assert mapping.overlap.matched_prediction_tokens == 2


def test_ensure_object_paint_mapping_rebuilds_after_index_change(monkeypatch) -> None:
    models = [
        types.SimpleNamespace(
            atom=[
                types.SimpleNamespace(
                    index=1,
                    chain="A",
                    resi="1",
                    resn="ALA",
                    name="CA",
                    hetatm=False,
                )
            ]
        ),
        types.SimpleNamespace(
            atom=[
                types.SimpleNamespace(
                    index=2,
                    chain="A",
                    resi="1",
                    resn="ALA",
                    name="CA",
                    hetatm=False,
                )
            ]
        ),
    ]
    state = {"model": 0, "get_model_calls": 0}

    def get_model(_name):
        state["get_model_calls"] += 1
        return models[state["model"]]

    cmd = types.SimpleNamespace(
        get_model=get_model,
        index=lambda _name: [
            ("obj", atom.index) for atom in models[state["model"]].atom
        ],
    )
    monkeypatch.setitem(sys.modules, "pymol", types.SimpleNamespace(cmd=cmd))
    token_map = TokenMap((_token(0),))
    mapping = mol_viewer.prepare_object_paint_mapping("obj", token_map)

    same, rebuilt = mol_viewer.ensure_object_paint_mapping("obj", token_map, mapping)
    assert same is mapping
    assert rebuilt is False

    state["model"] = 1
    changed, rebuilt = mol_viewer.ensure_object_paint_mapping("obj", token_map, mapping)
    assert rebuilt is True
    assert changed.atom_index_fingerprint == (2,)
    assert state["get_model_calls"] == 2


def test_tokens_within_distance_builds_backend_expression(monkeypatch) -> None:
    captured = []
    model = types.SimpleNamespace(
        atom=[
            types.SimpleNamespace(
                chain="A", resi="2", resn="ALA", name="CA", hetatm=False
            )
        ]
    )
    cmd = types.SimpleNamespace(
        get_model=lambda expression: captured.append(expression) or model
    )
    monkeypatch.setitem(sys.modules, "pymol", types.SimpleNamespace(cmd=cmd))
    token_map = TokenMap((_token(0), _token(1)))

    assert mol_viewer.tokens_within_distance(
        token_map, "model_0", "resname LIG", 7.5
    ) == [1]
    assert "byres ((model_0) within 7.5" in captured[0]
    assert "not ((resname LIG) and (model_0))" in captured[0]


def test_coordinates_transform_and_plot_selection_are_viewer_operations(
    monkeypatch,
) -> None:
    atoms = [
        types.SimpleNamespace(
            chain="A", resi="1", resn="ALA", name="CA", coord=(1, 2, 3)
        ),
        types.SimpleNamespace(
            chain="L", resi="2", resn="LIG", name="C1", coord=(4, 5, 6)
        ),
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
    token_map = TokenMap(
        (
            _token(0),
            _token(1, chain="L", hetatm=True, atom_name="C1"),
        )
    )

    np.testing.assert_array_equal(
        mol_viewer.get_representative_coords("model", token_map),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    )
    mol_viewer.transform_object("model", np.eye(3), np.array([1.0, 0.0, 0.0]))
    assert cmd.altered[0][0:2] == (1, "model")

    mol_viewer.PyMOLViewer().update_token_selection(
        "foldqc_plot_selection", [0], [("model", token_map)]
    )
    assert cmd.selections == [
        ("foldqc_plot_selection", "(%model and polymer and chain A)")
    ]
    assert cmd.enabled == ["foldqc_plot_selection"]
    assert cmd.refreshes == 1
