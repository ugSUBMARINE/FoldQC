from __future__ import annotations

import ast
import importlib
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _dialog_lifecycle_harness():
    source_path = Path(__file__).resolve().parents[1] / "gui.py"
    tree = ast.parse(source_path.read_text())
    dialog = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FoldQCPluginDialog"
    )
    methods = [
        node
        for node in dialog.body
        if isinstance(node, ast.FunctionDef) and node.name in {"shutdown", "closeEvent"}
    ]
    harness = ast.ClassDef(
        name="DialogLifecycleHarness",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace: dict[str, object] = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["DialogLifecycleHarness"]


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

    def test_run_plugin_gui_reuses_hidden_dialog_instance(self) -> None:
        class FakeDialog:
            instances = 0

            def __init__(self) -> None:
                type(self).instances += 1
                self.show_calls = 0
                self.raise_calls = 0
                self.activate_calls = 0

            def show(self) -> None:
                self.show_calls += 1

            def raise_(self) -> None:
                self.raise_calls += 1

            def activateWindow(self) -> None:
                self.activate_calls += 1

        fake_gui = types.ModuleType("FoldQC.gui")
        fake_gui.FoldQCPluginDialog = FakeDialog
        old_gui = sys.modules.get("FoldQC.gui")
        plugin = importlib.import_module("FoldQC")
        old_dialog = plugin._dialog_instance
        sys.modules["FoldQC.gui"] = fake_gui
        plugin._dialog_instance = None
        try:
            plugin.run_plugin_gui()
            dialog = plugin._dialog_instance
            plugin.run_plugin_gui()
        finally:
            plugin._dialog_instance = old_dialog
            if old_gui is None:
                sys.modules.pop("FoldQC.gui", None)
            else:
                sys.modules["FoldQC.gui"] = old_gui

        self.assertEqual(FakeDialog.instances, 1)
        self.assertEqual(dialog.show_calls, 2)
        self.assertEqual(dialog.raise_calls, 2)
        self.assertEqual(dialog.activate_calls, 2)

    def test_run_plugin_gui_reports_unexpected_import_failure(self) -> None:
        class BrokenGui(types.ModuleType):
            def __getattr__(self, name: str):
                if name == "FoldQCPluginDialog":
                    raise RuntimeError("broken dialog import")
                raise AttributeError(name)

        calls: list[tuple[object, str, str]] = []
        qt_module = types.ModuleType("pymol.Qt")
        qt_module.QtWidgets = types.SimpleNamespace(
            QMessageBox=types.SimpleNamespace(
                critical=lambda parent, title, message: calls.append(
                    (parent, title, message)
                )
            )
        )
        pymol_module = types.ModuleType("pymol")
        pymol_module.__path__ = []

        plugin = importlib.import_module("FoldQC")
        old_dialog = plugin._dialog_instance
        old_gui = sys.modules.get("FoldQC.gui")
        old_pymol = sys.modules.get("pymol")
        old_qt = sys.modules.get("pymol.Qt")
        sys.modules["FoldQC.gui"] = BrokenGui("FoldQC.gui")
        sys.modules["pymol"] = pymol_module
        sys.modules["pymol.Qt"] = qt_module
        plugin._dialog_instance = None
        try:
            with self.assertLogs("FoldQC", level="ERROR") as logs:
                plugin.run_plugin_gui()
        finally:
            plugin._dialog_instance = old_dialog
            for name, previous in (
                ("FoldQC.gui", old_gui),
                ("pymol", old_pymol),
                ("pymol.Qt", old_qt),
            ):
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        self.assertEqual(len(calls), 1)
        _parent, title, message = calls[0]
        self.assertEqual(title, "FoldQC - GUI initialization failed")
        self.assertIn("RuntimeError: broken dialog import", message)
        self.assertIn("full traceback", message)
        self.assertNotIn("install_deps.py", message)
        self.assertIn("Could not initialize the FoldQC GUI", logs.output[0])

    def test_dialog_close_hides_without_shutting_down_services(self) -> None:
        harness = _dialog_lifecycle_harness()()
        calls: list[str] = []
        harness.hide = lambda: calls.append("hide")
        event = types.SimpleNamespace(ignore=lambda: calls.append("ignore"))

        harness.closeEvent(event)

        self.assertEqual(calls, ["ignore", "hide"])

    def test_dialog_shutdown_saves_and_closes_services_once(self) -> None:
        harness = _dialog_lifecycle_harness()()
        calls: list[str] = []
        harness._shutdown_complete = False
        harness._session = types.SimpleNamespace(
            save_geometry=lambda: calls.append("save")
        )
        harness.services = types.SimpleNamespace(
            close=lambda: calls.append("close"),
        )

        harness.shutdown()
        harness.shutdown()

        self.assertEqual(calls, ["save", "close"])


if __name__ == "__main__":
    unittest.main()
