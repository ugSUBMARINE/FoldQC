"""
install_deps.py — run once from PyMOL to install missing dependencies.

Usage (from the PyMOL command line or via File → Run Script…):
    run /path/to/FoldQC/tools/install_deps.py

The script installs packages into the same Python that PyMOL is running on
by using sys.executable, so no knowledge of PyMOL's Python path is required.
Restart PyMOL after running this script.
"""

import subprocess
import sys

REQUIRED: dict[str, str] = {
    # import_name  : pip_package_name
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "sklearn": "scikit-learn",  # import name differs from pip name
}

OPTIONAL: dict[str, str] = {}

print(f"Python interpreter: {sys.executable}")
print()

all_ok = True

for packages, label in [(REQUIRED, "[required]"), (OPTIONAL, "[optional]")]:
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
            print(f"  OK       {pip_name}")
        except ImportError:
            print(f"  Installing {pip_name} {label} ...", end="", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(" done.")
            else:
                print(" FAILED.")
                print(f"    stderr: {result.stderr.strip()}")
                if label == "[required]":
                    all_ok = False

print()
if all_ok:
    print("All required dependencies are installed.")
    print("Please restart PyMOL for the changes to take effect.")
else:
    print("Some required packages could not be installed.")
    print("Check the error messages above and install them manually.")
