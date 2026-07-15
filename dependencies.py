"""Optional dependency discovery and installation command helpers.

This module is independent of Qt and PyMOL.  The GUI owns user interaction and
process lifecycle; the standalone installer and GUI share the registry and
command construction defined here.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import site
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DependencyKey = Literal["matplotlib", "scipy", "sklearn"]


@dataclass(frozen=True)
class DependencySpec:
    """One optional runtime dependency exposed through a FoldQC feature."""

    key: str
    import_name: str
    distribution_name: str
    display_name: str


DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec("matplotlib", "matplotlib", "matplotlib", "Matplotlib"),
    DependencySpec("scipy", "scipy", "scipy", "SciPy"),
    DependencySpec("sklearn", "sklearn", "scikit-learn", "scikit-learn"),
)
DEPENDENCY_BY_KEY: dict[str, DependencySpec] = {
    dependency.key: dependency for dependency in DEPENDENCIES
}


def required_dependency_keys(dependency_keys: Iterable[str]) -> tuple[str, ...]:
    """Validate and order explicit dependency keys in registry order."""
    requested = {str(key) for key in dependency_keys}
    unknown = requested - set(DEPENDENCY_BY_KEY)
    if unknown:
        raise ValueError(f"Unknown dependency keys: {sorted(unknown)!r}.")
    return tuple(
        dependency.key for dependency in DEPENDENCIES if dependency.key in requested
    )


def missing_dependency_keys(
    dependency_keys: Iterable[str],
    *,
    find_spec: Callable[[str], object | None] | None = None,
) -> tuple[str, ...]:
    """Return requested dependency keys whose import cannot be discovered."""
    if find_spec is None:
        find_spec = importlib.util.find_spec
    requested = set(dependency_keys)
    missing: list[str] = []
    for dependency in DEPENDENCIES:
        if dependency.key not in requested:
            continue
        try:
            available = find_spec(dependency.import_name) is not None
        except (ImportError, AttributeError, ValueError):
            available = False
        if not available:
            missing.append(dependency.key)
    return tuple(missing)


def dependency_specs(dependency_keys: Iterable[str]) -> tuple[DependencySpec, ...]:
    """Resolve dependency keys to specifications in registry order."""
    requested = set(dependency_keys)
    return tuple(
        dependency for dependency in DEPENDENCIES if dependency.key in requested
    )


def pip_install_args(dependency_keys: Iterable[str]) -> list[str]:
    """Return safe pip arguments for installation into the current environment."""
    packages = [
        dependency.distribution_name for dependency in dependency_specs(dependency_keys)
    ]
    return [
        "-m",
        "pip",
        "install",
        "--no-user",
        "--no-input",
        "--disable-pip-version-check",
        "--no-color",
        "--progress-bar",
        "off",
        *packages,
    ]


def validation_args(dependency_keys: Iterable[str]) -> list[str]:
    """Return interpreter arguments that import the requested dependencies."""
    import_names = [
        dependency.import_name for dependency in dependency_specs(dependency_keys)
    ]
    statement = "; ".join(f"import {name}" for name in import_names)
    return ["-c", statement]


def format_shell_command(
    arguments: Sequence[str], *, platform_name: str | None = None
) -> str:
    """Format *arguments* as a copyable command for the current platform."""
    platform_name = os.name if platform_name is None else platform_name
    if platform_name == "nt":
        return subprocess.list2cmdline(list(arguments))
    return shlex.join(arguments)


def manual_install_instructions(
    dependency_keys: Iterable[str],
    *,
    executable: str | None = None,
    prefix: str | None = None,
    user_site_enabled: bool | None = None,
    platform_name: str | None = None,
) -> str:
    """Return manual conda and optional user-site installation instructions."""
    specs = dependency_specs(dependency_keys)
    packages = [dependency.distribution_name for dependency in specs]
    executable = sys.executable if executable is None else executable
    prefix = sys.prefix if prefix is None else prefix
    if user_site_enabled is None:
        user_site_enabled = bool(site.ENABLE_USER_SITE)

    lines = [
        "Conda (preferred for conda-based PyMOL installations):",
        format_shell_command(
            ["conda", "install", "--prefix", prefix, *packages],
            platform_name=platform_name,
        ),
    ]
    if user_site_enabled:
        lines.extend(
            [
                "",
                "Pip user installation (may mix packages across conda environments):",
                format_shell_command(
                    [executable, "-m", "pip", "install", "--user", *packages],
                    platform_name=platform_name,
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Retry the feature after installing. A PyMOL restart may be necessary "
            "if it is not immediately available.",
        ]
    )
    return "\n".join(lines)


def environment_install_paths() -> tuple[Path, ...]:
    """Return environment directories that pip may need to write."""
    paths = []
    for name in ("purelib", "scripts"):
        value = sysconfig.get_path(name)
        if value:
            path = Path(value)
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def environment_is_writable(
    paths: Iterable[Path] | None = None,
    *,
    access: Callable[[Path, int], bool] = os.access,
) -> bool:
    """Best-effort check that pip's environment destinations are writable."""
    install_paths = environment_install_paths() if paths is None else tuple(paths)
    if not install_paths:
        return False
    for path in install_paths:
        candidate = Path(path)
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        if not candidate.is_dir() or not access(candidate, os.W_OK | os.X_OK):
            return False
    return True
