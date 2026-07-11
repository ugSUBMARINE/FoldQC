from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class PluginRegistrationTests(unittest.TestCase):
    def test_pymol_menu_label_uses_foldqc_display_name(self) -> None:
        calls: list[tuple[str, object]] = []

        old_pymol = sys.modules.get("pymol")
        old_plugins = sys.modules.get("pymol.plugins")

        pymol_module = types.ModuleType("pymol")
        pymol_module.__path__ = []
        plugins_module = types.ModuleType("pymol.plugins")
        plugins_module.addmenuitemqt = lambda label, callback: calls.append(
            (label, callback)
        )
        sys.modules["pymol"] = pymol_module
        sys.modules["pymol.plugins"] = plugins_module
        try:
            plugin = importlib.import_module("FoldQC")
            plugin.__init_plugin__()
        finally:
            if old_pymol is None:
                sys.modules.pop("pymol", None)
            else:
                sys.modules["pymol"] = old_pymol
            if old_plugins is None:
                sys.modules.pop("pymol.plugins", None)
            else:
                sys.modules["pymol.plugins"] = old_plugins

        self.assertEqual(calls, [("FoldQC\u2026", plugin.run_plugin_gui)])


if __name__ == "__main__":
    unittest.main()
