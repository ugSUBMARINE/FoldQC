from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.analysis import (
    AnalysisPreflightError,
    AnalysisRequest,
    AnalysisResolver,
    ComputedMetric,
    build_data_load_plan,
)
from FoldQC.ensemble import EnsembleMember, EnsembleState
from FoldQC.gui_state import PluginState
from FoldQC.loader_models import ModelFiles, PredictionData, PredictionFiles
from FoldQC.model_state import ModelState
from FoldQC.providers.registry import BUILTIN_PROVIDERS
from FoldQC.structure_index import StructureIndex
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap


def _state(tmp_path: Path, rank: int, *, pae=None) -> ModelState:
    path = tmp_path / f"model_{rank}.cif"
    token_map = TokenMap((TokenInfo(0, "A", ResidueId(1), "ALA", False, None),))
    b_factors = np.zeros(1, dtype=np.float32)
    b_factors.setflags(write=False)
    index = StructureIndex(path, "cif", token_map, 1, (0,), b_factors)
    return ModelState(
        rank,
        PredictionData(
            "prediction",
            rank,
            path,
            BUILTIN_PROVIDERS.get("boltz").info,
            pae=pae,
        ),
        index,
    )


def _plugin_state(tmp_path: Path) -> PluginState:
    states = {rank: _state(tmp_path, rank) for rank in (0, 1)}
    files = PredictionFiles(
        "prediction",
        tmp_path,
        BUILTIN_PROVIDERS.get("boltz").info,
        models=[
            ModelFiles(
                0,
                tmp_path / "model_0.cif",
                "rank 0",
                "prediction_model_0",
                capabilities=frozenset({"plddt", "pae"}),
            ),
            ModelFiles(
                1,
                tmp_path / "model_1.cif",
                "rank 1",
                "prediction_model_1",
                capabilities=frozenset({"plddt"}),
            ),
        ],
    )
    members = (
        EnsembleMember(0, "prediction_model_0"),
        EnsembleMember(1, "prediction_model_1"),
    )
    ensemble = EnsembleState(
        "prediction_ensemble",
        members,
        False,
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )
    return PluginState(files, states, 0, ensemble=ensemble)


def test_request_validates_metric_cutoff_and_plot_contract() -> None:
    request = AnalysisRequest(
        "color", "prediction_model_0", "pae_row_mean", cutoff_angstrom=5.0
    )
    assert request.metric_key == "pae_row_mean"
    with pytest.raises(ValueError, match="requires a metric"):
        AnalysisRequest("line", "prediction_model_0")
    with pytest.raises(ValueError, match="greater than 0"):
        AnalysisRequest("color", "prediction_model_0", "plddt", cutoff_angstrom=0)


def test_resolver_uses_canonical_target_members_and_registry_requirements(
    tmp_path: Path,
) -> None:
    state = _plugin_state(tmp_path)
    resolved = AnalysisResolver().resolve(
        AnalysisRequest("matrix", "prediction_ensemble", "pae_row_mean", ui_revision=4),
        state,
    )
    assert resolved.target.kind == "ensemble_group"
    assert [member.rank for member in resolved.members] == [0, 1]
    assert resolved.required_capabilities == frozenset({"pae"})
    assert resolved.dependency_keys == ("matplotlib",)
    assert resolved.coverage_policy == "strict"
    assert resolved.validate_current(state, ui_revision=4)
    assert not resolved.validate_current(state, ui_revision=5)


def test_strict_load_plan_names_members_without_advertised_data(tmp_path: Path) -> None:
    resolved = AnalysisResolver().resolve(
        AnalysisRequest("color", "prediction_ensemble", "pae_row_mean"),
        _plugin_state(tmp_path),
    )
    with pytest.raises(AnalysisPreflightError) as caught:
        build_data_load_plan(resolved)
    assert caught.value.problem.affected_models == ("rank 1",)


def test_single_load_plan_is_deduplicated_and_converts_capabilities_at_boundary(
    tmp_path: Path,
) -> None:
    state = _plugin_state(tmp_path)
    resolved = AnalysisResolver().resolve(
        AnalysisRequest("color", "prediction_model_0", "pae_row_mean"), state
    )
    plan = build_data_load_plan(resolved)
    assert len(plan.requirements) == 1
    requirement = plan.requirements[0]
    assert requirement.rank == 0
    assert requirement.load_kwargs() == {
        "load_pae": True,
        "load_pde": False,
        "load_contact_probs": False,
        "load_token_plddt": False,
    }


def test_partial_summary_policy_does_not_require_a_selected_metric(
    tmp_path: Path,
) -> None:
    resolved = AnalysisResolver().resolve(
        AnalysisRequest("ensemble_site_summary", "prediction_ensemble"),
        _plugin_state(tmp_path),
    )
    assert resolved.metric_spec is None
    assert resolved.coverage_policy == "available"


def test_computed_metric_owns_readonly_float32_values(tmp_path: Path) -> None:
    state = _state(tmp_path, 0)
    result = ComputedMetric(
        0,
        "rank 0",
        "model_0",
        state,
        resolved_context := resolved_context_fixture(),
        [1, 2],
    )
    assert result.metric_context is resolved_context
    assert result.values.dtype == np.float32
    assert result.values.flags.c_contiguous
    assert not result.values.flags.writeable


def resolved_context_fixture():
    from FoldQC.gui_state import MetricContext

    return MetricContext(reference_selection="chain A", reference_indices=(0,))


def test_dialog_has_no_controller_or_state_mixin_bases() -> None:
    source = Path(__file__).resolve().parents[1].joinpath("gui.py").read_text()
    tree = ast.parse(source)
    dialog = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FoldQCPluginDialog"
    )
    assert [ast.unparse(base) for base in dialog.bases] == ["QtWidgets.QDialog"]
    assert "GuiStateBacked" not in source
