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

import logging

__version__ = "0.3.0"
__author__ = "Karl Gruber"

logger = logging.getLogger(__name__)

# Singleton: keep the dialog alive so it survives garbage collection
_dialog_instance = None


def __init_plugin__(app=None):
    """Register the plugin in PyMOL's Plugin menu."""
    from pymol.plugins import addmenuitemqt

    addmenuitemqt("FoldQC\u2026", run_plugin_gui)


def run_plugin_gui():
    """Open (or raise) the main FoldQC plugin dialog."""
    global _dialog_instance

    if _dialog_instance is None:
        try:
            from .gui import FoldQCPluginDialog
        except Exception as exc:
            logger.exception("Could not initialize the FoldQC GUI")
            from pymol.Qt import QtWidgets

            QtWidgets.QMessageBox.critical(
                None,
                "FoldQC - GUI initialization failed",
                (
                    "FoldQC could not initialize its dialog.\n\n"
                    f"{type(exc).__name__}: {exc}\n\n"
                    "See the PyMOL console for the full traceback. If the problem "
                    "persists after restarting PyMOL, include that traceback when "
                    "reporting the issue."
                ),
            )
            return

        _dialog_instance = FoldQCPluginDialog()

    _dialog_instance.show()
    _dialog_instance.raise_()
    _dialog_instance.activateWindow()
