from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC import dependencies


def test_standalone_installer_uses_pymol_script_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = Path(__file__).resolve().parents[1] / "tools" / "install_deps.py"
    simulated_path = tmp_path / "FoldQC" / "tools" / "install_deps.py"
    expected_parent = str(tmp_path)
    fake_dependencies = SimpleNamespace(
        DEPENDENCIES=(),
        missing_dependency_keys=lambda _keys: (),
    )
    original_import = builtins.__import__

    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "FoldQC":
            assert sys.path[0] == expected_parent
            return SimpleNamespace(dependencies=fake_dependencies)
        return original_import(name, globals, locals, fromlist, level)

    namespace = {
        "__builtins__": {**vars(builtins), "__import__": importing},
        "__file__": "/Applications/PyMOL.app/pymol/__init__.py",
        "__script__": str(simulated_path),
    }
    old_path = sys.path.copy()
    try:
        sys.path[:] = [entry for entry in sys.path if entry != expected_parent]
        source = source_path.read_text(encoding="utf-8")
        exec(compile(source, str(source_path), "exec"), namespace)
    finally:
        sys.path[:] = old_path

    assert "All optional FoldQC dependencies are installed." in capsys.readouterr().out


def test_dependency_keys_are_validated_deduplicated_and_ordered() -> None:
    assert dependencies.required_dependency_keys(
        ["sklearn", "matplotlib", "scipy", "matplotlib"]
    ) == ("matplotlib", "scipy", "sklearn")
    with pytest.raises(ValueError, match="Unknown dependency"):
        dependencies.required_dependency_keys(["plot"])


def test_missing_dependencies_check_import_names() -> None:
    seen = []

    def find_spec(name: str):
        seen.append(name)
        return object() if name == "scipy" else None

    assert dependencies.missing_dependency_keys(
        ("matplotlib", "scipy", "sklearn"), find_spec=find_spec
    ) == ("matplotlib", "sklearn")
    assert seen == ["matplotlib", "scipy", "sklearn"]


def test_pip_command_installs_only_requested_packages_without_user_site() -> None:
    assert dependencies.pip_install_args(("sklearn", "matplotlib")) == [
        "-m",
        "pip",
        "install",
        "--no-user",
        "--no-input",
        "--disable-pip-version-check",
        "--no-color",
        "--progress-bar",
        "off",
        "matplotlib",
        "scikit-learn",
    ]


def test_validation_command_uses_import_names() -> None:
    assert dependencies.validation_args(("scipy", "sklearn")) == [
        "-c",
        "import scipy; import sklearn",
    ]


def test_manual_instructions_quote_paths_and_optionally_include_user_site() -> None:
    instructions = dependencies.manual_install_instructions(
        ("matplotlib", "sklearn"),
        executable="/Applications/PyMOL App/python",
        prefix="/Applications/PyMOL App/env",
        user_site_enabled=True,
        platform_name="posix",
    )
    assert "conda install --prefix '/Applications/PyMOL App/env'" in instructions
    assert "'/Applications/PyMOL App/python' -m pip install --user" in instructions
    assert "matplotlib scikit-learn" in instructions

    no_user = dependencies.manual_install_instructions(
        ("scipy",), user_site_enabled=False
    )
    assert "--user" not in no_user


def test_environment_writability_checks_existing_parent(tmp_path: Path) -> None:
    checked = []

    def access(path: Path, mode: int) -> bool:
        checked.append((path, mode))
        return path.name != "blocked"

    assert dependencies.environment_is_writable(
        (tmp_path / "missing" / "site-packages",), access=access
    )
    assert checked[0][0] == tmp_path

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    assert not dependencies.environment_is_writable((blocked,), access=access)
