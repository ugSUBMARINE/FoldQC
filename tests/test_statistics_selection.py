from __future__ import annotations

import numpy as np
from FoldQC.gui_services import StatisticsSelectionTarget
from FoldQC.statistics_selection import (
    StatisticsSelectionService,
    threshold_selection_name,
)
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap


def _token_map(length: int) -> TokenMap:
    return TokenMap(
        tuple(
            TokenInfo(
                token_idx=index,
                chain_id="A",
                residue_id=ResidueId(index + 1),
                res_name="ALA",
                is_hetatm=False,
                atom_name=None,
            )
            for index in range(length)
        )
    )


class _Viewer:
    def __init__(self) -> None:
        self.calls = []

    def update_object_token_selection(self, name, targets) -> None:
        self.calls.append((name, tuple(targets)))


class _Presenter:
    def __init__(self) -> None:
        self.notices = []

    def present_notice(self, notice) -> None:
        self.notices.append(notice)


class _View:
    def __init__(self) -> None:
        self.states = []

    def set_statistics_selection(self, state) -> None:
        self.states.append(state)


def test_threshold_selection_uses_each_objects_stored_values() -> None:
    viewer = _Viewer()
    presenter = _Presenter()
    view = _View()
    service = StatisticsSelectionService(viewer, presenter, view)
    first_map = _token_map(3)
    second_map = _token_map(2)
    service.set_coloring_result(
        "plddt",
        (
            StatisticsSelectionTarget(
                "model_0", first_map, np.array([1.0, 3.0, np.nan])
            ),
            StatisticsSelectionTarget("model_1", second_map, np.array([2.0, 4.0])),
        ),
    )

    initial = view.states[-1]
    assert initial.enabled
    assert initial.minimum == 1.0
    assert initial.maximum == 4.0
    assert initial.threshold == 2.5

    assert service.select("ge", 2.5) == "foldqc_plddt_ge"
    name, targets = viewer.calls[-1]
    assert name == "foldqc_plddt_ge"
    assert [(target.obj_name, target.token_indices) for target in targets] == [
        ("model_0", (1,)),
        ("model_1", (1,)),
    ]
    assert targets[0].token_map is first_map
    assert targets[1].token_map is second_map
    assert "2 tokens ≥ 2.5" in view.states[-1].status_text
    assert presenter.notices == []


def test_threshold_selection_is_inclusive_and_excludes_nonfinite_values() -> None:
    viewer = _Viewer()
    view = _View()
    service = StatisticsSelectionService(viewer, _Presenter(), view)
    token_map = _token_map(4)
    service.set_coloring_result(
        "pae_row_mean",
        (
            StatisticsSelectionTarget(
                "model", token_map, np.array([2.0, 2.5, np.inf, np.nan])
            ),
        ),
    )

    assert service.select("le", 2.0) == "foldqc_pae_row_mean_le"
    assert viewer.calls[-1][1][0].token_indices == (0,)


def test_no_finite_values_disable_threshold_selection() -> None:
    viewer = _Viewer()
    view = _View()
    service = StatisticsSelectionService(viewer, _Presenter(), view)
    service.set_coloring_result(
        "plddt",
        (
            StatisticsSelectionTarget(
                "model", _token_map(2), np.array([np.nan, np.inf])
            ),
        ),
    )

    assert not view.states[-1].enabled
    assert service.select("ge", 0.0) is None
    assert viewer.calls == []


def test_threshold_selection_name_is_pymol_safe() -> None:
    assert threshold_selection_name("metric with/slash", "ge") == (
        "foldqc_metric_with_slash_ge"
    )
