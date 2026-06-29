"""
FoldQC
======================
Visualizes structure-prediction confidence metrics on loaded structures.

Registered properties include
------------------------------
- pLDDT               per-token predicted local distance difference test score
- PAE row/col mean    global positional uncertainty from PAE matrix
- PAE to selection    per-token PAE relative to a user-specified reference
- PDE mean            mean predicted distance error per token
- PDE to selection    PDE relative to a user-specified reference
- Chain ipTM          per-chain interface pTM from the confidence JSON
- Ensemble RMSD       per-token RMSD across multiple diffusion samples

Usage
-----
Install dependencies once (run from the PyMOL command line):
    run install_deps.py

Then open the plugin via Plugins → FoldQC…
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Karl Gruber"

# Singleton: keep the dialog alive so it survives garbage collection
_dialog_instance = None


def __init_plugin__(app=None):
    """Register the plugin in PyMOL's Plugin menu."""
    from pymol.plugins import addmenuitemqt

    addmenuitemqt("FoldQC\u2026", run_plugin_gui)


def run_plugin_gui():
    """Open (or raise) the main FoldQC plugin dialog."""
    global _dialog_instance

    # Re-use the existing window if it is still open
    if _dialog_instance is not None and _dialog_instance.isVisible():
        _dialog_instance.raise_()
        _dialog_instance.activateWindow()
        return

    try:
        from .gui import FoldQCPluginDialog
    except Exception as exc:
        from pymol.Qt import QtWidgets

        QtWidgets.QMessageBox.critical(
            None,
            "FoldQC - import error",
            (
                "Could not load the plugin GUI:\n\n"
                f"{exc}\n\n"
                "Run install_deps.py from PyMOL to install missing dependencies,\n"
                "then restart PyMOL."
            ),
        )
        return

    _dialog_instance = FoldQCPluginDialog()
    _dialog_instance.show()
