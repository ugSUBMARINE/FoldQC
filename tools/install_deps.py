"""Install all missing optional FoldQC dependencies from within PyMOL.

Usage (from the PyMOL command line or via File → Run Script…):
    run /path/to/FoldQC/tools/install_deps.py

The script installs packages into the same Python that PyMOL is running on
by using sys.executable, so no knowledge of PyMOL's Python path is required.
A restart may be necessary if the new packages are not immediately available.
"""

import subprocess
import sys
from pathlib import Path

# ``run /path/to/FoldQC/tools/install_deps.py`` does not establish package
# context, so make the directory containing the FoldQC package importable.
# PyMOL executes scripts in the ``pymol`` module namespace, where ``__file__``
# identifies pymol/__init__.py.  Its runner sets ``__script__`` to the actual
# script path before execution.
script_path = Path(globals().get("__script__") or __file__).resolve()
package_parent = str(script_path.parents[2])
if package_parent not in sys.path:
    sys.path.insert(0, package_parent)

from FoldQC import dependencies  # noqa: E402


def main() -> int:
    all_keys = tuple(dependency.key for dependency in dependencies.DEPENDENCIES)
    missing = dependencies.missing_dependency_keys(all_keys)

    print(f"Python interpreter: {sys.executable}")
    print(f"Environment prefix: {sys.prefix}")
    print()
    if not missing:
        print("All optional FoldQC dependencies are installed.")
        return 0

    names = ", ".join(
        dependency.distribution_name
        for dependency in dependencies.dependency_specs(missing)
    )
    print(f"Missing packages: {names}")
    if not dependencies.environment_is_writable():
        print("The PyMOL Python environment does not appear to be writable.")
        print()
        print(dependencies.manual_install_instructions(missing))
        return 1

    command = [sys.executable, *dependencies.pip_install_args(missing)]
    print(f"Running: {dependencies.format_shell_command(command)}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print()
        print("Installation failed. Manual alternatives:")
        print(dependencies.manual_install_instructions(missing))
        return result.returncode

    validation = subprocess.run(
        [sys.executable, *dependencies.validation_args(missing)], check=False
    )
    if validation.returncode != 0:
        print("The packages were installed, but import verification failed.")
        print(dependencies.manual_install_instructions(missing))
        return validation.returncode

    print()
    print("All optional FoldQC dependencies were installed and verified.")
    print("You can retry the FoldQC feature now.")
    print("A PyMOL restart may be necessary if it is not immediately available.")
    return 0


# PyMOL's ``run`` command need not assign the usual ``__main__`` module name.
main()
