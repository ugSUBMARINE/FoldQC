from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.gui_services import (
    AtomVisualSnapshot,
    ManagedColorbar,
    PaintBatchResult,
    PaintTarget,
)
from FoldQC.token_map import TokenMap
from FoldQC.viewer_transactions import (
    ColorbarChange,
    PaintTransaction,
    ViewerTransactionError,
)


class FakeViewer:
    def __init__(self) -> None:
        self.visuals = {
            "model_0": (
                np.array([10.0], dtype=np.float32),
                np.array([3], dtype=np.int32),
            ),
            "model_1": (
                np.array([20.0], dtype=np.float32),
                np.array([4], dtype=np.int32),
            ),
        }
        self.managed_colorbar = None
        self.colorbar_replacements = []
        self.restored: list[str] = []
        self.rebuilds = 0
        self.fail_colorbar = False
        self.events = []

    def snapshot_atom_visuals(self, obj_name):
        self.events.append(("snapshot", obj_name))
        b_factors, colors = self.visuals[obj_name]
        return AtomVisualSnapshot(obj_name, (0,), b_factors.copy(), colors.copy())

    def restore_atom_visuals(self, snapshot):
        self.visuals[snapshot.obj_name] = (
            snapshot.b_factors.copy(),
            snapshot.color_indices.copy(),
        )
        self.restored.append(snapshot.obj_name)

    def get_managed_colorbar(self):
        return self.managed_colorbar

    def replace_managed_colorbar(self, state):
        self.events.append(("colorbar", state))
        self.colorbar_replacements.append(state)
        if self.fail_colorbar:
            self.managed_colorbar = None
            self.fail_colorbar = False
            raise RuntimeError("colorbar failed")
        self.managed_colorbar = state

    def rebuild(self):
        self.rebuilds += 1

    def run_suspended(self, operation):
        self.events.append(("suspend", None))
        return operation()


def _targets():
    return (
        PaintTarget("model_0", TokenMap(()), np.array([], dtype=np.float32)),
        PaintTarget("model_1", TokenMap(()), np.array([], dtype=np.float32)),
    )


def test_successful_paint_commits_and_replaces_managed_colorbar() -> None:
    viewer = FakeViewer()
    previous = ManagedColorbar("white_blue", False, 0.0, 1.0, ("model_0",))
    viewer.managed_colorbar = previous
    transaction = PaintTransaction(
        viewer,
        _targets(),
        ColorbarChange("replace", "blue_white_red"),
    )

    def paint():
        viewer.visuals["model_0"] = (
            np.array([80.0], dtype=np.float32),
            np.array([8], dtype=np.int32),
        )
        return PaintBatchResult(0.0, 1.0, ())

    result = transaction.execute(paint)
    assert result.vmax == 1.0
    assert viewer.managed_colorbar == ManagedColorbar(
        "blue_white_red", False, 0.0, 1.0, ("model_0", "model_1")
    )
    assert viewer.restored == []
    assert viewer.events[:3] == [
        ("snapshot", "model_0"),
        ("snapshot", "model_1"),
        ("suspend", None),
    ]


def test_mid_paint_failure_restores_every_affected_object() -> None:
    viewer = FakeViewer()
    original = {
        name: (values[0].copy(), values[1].copy())
        for name, values in viewer.visuals.items()
    }
    transaction = PaintTransaction(viewer, _targets(), ColorbarChange("remove"))

    def paint():
        viewer.visuals["model_0"] = (
            np.array([99.0], dtype=np.float32),
            np.array([9], dtype=np.int32),
        )
        raise RuntimeError("second object failed")

    with pytest.raises(ViewerTransactionError, match="second object failed"):
        transaction.execute(paint)
    assert viewer.restored == ["model_1", "model_0"]
    for name, (b_factors, colors) in original.items():
        np.testing.assert_array_equal(viewer.visuals[name][0], b_factors)
        np.testing.assert_array_equal(viewer.visuals[name][1], colors)


def test_colorbar_failure_recreates_previous_managed_colorbar() -> None:
    viewer = FakeViewer()
    previous = ManagedColorbar("white_red", True, 0.2, 0.8, ("model_0",))
    viewer.managed_colorbar = previous
    viewer.fail_colorbar = True
    transaction = PaintTransaction(
        viewer,
        _targets()[:1],
        ColorbarChange("replace", "blue_white_red"),
    )
    with pytest.raises(ViewerTransactionError, match="colorbar failed"):
        transaction.execute(lambda: PaintBatchResult(0.0, 1.0, ()))
    assert viewer.managed_colorbar == previous
    assert viewer.colorbar_replacements[-1] == previous
    assert viewer.restored == ["model_0"]


def test_repeated_recolor_reuses_one_managed_colorbar_without_name_swaps() -> None:
    viewer = FakeViewer()
    for index in range(12):
        transaction = PaintTransaction(
            viewer,
            _targets()[:1],
            ColorbarChange("replace", "blue_white_red", bool(index % 2)),
        )
        transaction.execute(
            lambda index=index: PaintBatchResult(float(index), float(index + 1), ())
        )

    assert len(viewer.colorbar_replacements) == 12
    assert viewer.managed_colorbar.vmin == 11.0
    assert viewer.managed_colorbar.vmax == 12.0
    assert all(event[0] != "rename" for event in viewer.events)
