"""Chai-1 Discovery provider."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..loader_models import ModelFiles, PredictionData, PredictionFiles
from ..loader_utils import (
    STRUCTURE_SUFFIXES,
    _float_or_none,
    _load_json,
    _matrix_list_to_nested_dict,
    _normalise_confidence,
    _safe_object_name,
)
from .base import BaseProvider


@dataclass
class _ChaiCandidate:
    """One Chai-1 Discovery prediction candidate's discovered paths."""

    structure_path: Path
    model_idx: int | None
    explicit_rank: int | None
    confidence_path: Path | None
    pae_path: Path | None
    pde_path: Path | None
    aggregate_score: float | None


def _looks_like_chai(pred_dir: Path) -> bool:
    structures = [
        path
        for suffix in STRUCTURE_SUFFIXES
        for path in pred_dir.glob(f"pred*{suffix}")
        if _parse_chai_stem(path.stem) is not None
    ]
    if not structures:
        return False
    return bool(
        list(pred_dir.glob("scores*.json"))
        or list(pred_dir.glob("scores*.npz"))
        or list(pred_dir.glob("pae*.npy"))
    )


def _scan_chai_dir(pred_dir: Path) -> PredictionFiles:
    """Discover Chai-1 Discovery ranked output files."""
    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="chai1",
        input_path=pred_dir,
    )

    candidates: list[_ChaiCandidate] = []
    for structure_path in _chai_structure_paths(pred_dir):
        parsed = _parse_chai_stem(structure_path.stem)
        if parsed is None:
            continue
        model_idx, explicit_rank = parsed
        suffix = structure_path.stem.removeprefix("pred")
        confidence_path = _chai_companion_path(pred_dir, "scores", suffix)
        pae_path = _chai_companion_path(pred_dir, "pae", suffix)
        pde_path = _chai_companion_path(pred_dir, "pde", suffix)
        candidates.append(
            _ChaiCandidate(
                structure_path=structure_path,
                model_idx=model_idx,
                explicit_rank=explicit_rank,
                confidence_path=confidence_path,
                pae_path=pae_path,
                pde_path=pde_path,
                aggregate_score=_chai_aggregate_score(confidence_path),
            )
        )

    if not candidates:
        raise ValueError(
            f"No Chai-1 Discovery structure files found in {pred_dir}.\n"
            "Expected files named like 'pred.rank_<rank>.cif', "
            "'pred.model_idx_<idx>.rank_<rank>.cif', or "
            "'pred.model_idx_<idx>.cif'."
        )

    explicit = [item for item in candidates if item.explicit_rank is not None]
    raw = [item for item in candidates if item.explicit_rank is None]
    explicit.sort(
        key=lambda item: (
            item.explicit_rank,
            item.model_idx if item.model_idx is not None else -1,
            item.structure_path.name,
        )
    )
    raw.sort(
        key=lambda item: (
            item.aggregate_score is None,
            -float(item.aggregate_score or 0.0),
            item.model_idx if item.model_idx is not None else 0,
            item.structure_path.name,
        )
    )

    used_ranks: set[int] = set()
    for item in explicit:
        rank = int(item.explicit_rank)  # type: ignore[arg-type]  # explicit list is filtered
        if rank in used_ranks:
            continue
        used_ranks.add(rank)
        _append_chai_model(files, item, rank)

    next_rank = 0
    for item in raw:
        while next_rank in used_ranks:
            next_rank += 1
        used_ranks.add(next_rank)
        _append_chai_model(files, item, next_rank)

    files.models.sort(key=lambda model: model.rank)
    return files


def _append_chai_model(
    files: PredictionFiles,
    item: _ChaiCandidate,
    rank: int,
) -> None:
    metadata: dict[str, Any] = {}
    if item.model_idx is not None:
        metadata["model_idx"] = item.model_idx
    if item.aggregate_score is not None:
        metadata["aggregate_score"] = item.aggregate_score
    label = f"rank {rank}"
    if item.model_idx is not None:
        label += f" - model {item.model_idx}"
    files.models.append(
        ModelFiles(
            rank=rank,
            structure_path=item.structure_path,
            display_label=label,
            object_name=f"{_safe_object_name(files.name)}_model_{rank}",
            confidence_path=item.confidence_path,
            pae_path=item.pae_path,
            pde_path=item.pde_path,
            capabilities=frozenset(
                {"plddt"}
                | ({"pae"} if item.pae_path is not None else set())
                | ({"pde"} if item.pde_path is not None else set())
            ),
            metadata=metadata,
        )
    )


def _load_chai_model_data(
    model: ModelFiles,
    data: PredictionData,
    *,
    load_pae: bool,
    load_pde: bool,
) -> None:
    if model.confidence_path is not None:
        data.confidence = _normalise_confidence(
            _normalise_chai_confidence(_load_chai_scores(model.confidence_path))
        )

    if load_pae and model.pae_path is not None:
        data.pae = np.load(model.pae_path).astype(np.float32)

    if load_pde and model.pde_path is not None:
        data.pde = np.load(model.pde_path).astype(np.float32)


def _chai_structure_paths(pred_dir: Path) -> list[Path]:
    paths = [
        path
        for suffix in STRUCTURE_SUFFIXES
        for path in pred_dir.glob(f"pred*{suffix}")
    ]
    return sorted(paths, key=lambda path: path.name)


def _parse_chai_stem(stem: str) -> tuple[int | None, int | None] | None:
    match = re.fullmatch(r"pred(?:\.model_idx_(\d+))?(?:\.rank_(\d+))?", stem)
    if match is None:
        return None
    model_idx, rank = match.groups()
    if model_idx is None and rank is None:
        return None
    return (
        int(model_idx) if model_idx is not None else None,
        int(rank) if rank is not None else None,
    )


def _chai_companion_path(pred_dir: Path, stem: str, suffix: str) -> Path | None:
    if stem == "scores":
        json_path = pred_dir / f"{stem}{suffix}.json"
        if json_path.exists():
            return json_path
        npz_path = pred_dir / f"{stem}{suffix}.npz"
        if npz_path.exists():
            return npz_path
        return None
    path = pred_dir / f"{stem}{suffix}.npy"
    return path if path.exists() else None


def _chai_aggregate_score(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return _float_or_none(_load_chai_scores(path).get("aggregate_score"))
    except Exception:
        return None


def _load_chai_scores(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as scores:
            return {
                key: _numpy_score_to_python(key, scores[key]) for key in scores.files
            }
    raise ValueError(f"Unsupported Chai score file format: {path.suffix}")


def _numpy_score_to_python(key: str, value):
    array = np.asarray(value)
    scalar_keys = {
        "aggregate_score",
        "ptm",
        "iptm",
        "has_inter_chain_clashes",
    }
    if array.shape == ():
        return array.item()
    if key in scalar_keys and array.size == 1:
        return array.reshape(-1)[0].item()
    return array.tolist()


def _normalise_chai_confidence(confidence: dict | None) -> dict | None:
    if confidence is None:
        return None
    normalised = dict(confidence)

    if "aggregate_score" in normalised and "ranking_score" not in normalised:
        normalised["ranking_score"] = normalised["aggregate_score"]

    if (
        "per_chain_ptm" in normalised
        and "chains_ptm" not in normalised
        and "chain_ptm" not in normalised
    ):
        chain_ptm = _chai_first_sample(normalised["per_chain_ptm"])
        if isinstance(chain_ptm, list):
            normalised["chains_ptm"] = {
                str(idx): value for idx, value in enumerate(chain_ptm)
            }

    if (
        "per_chain_pair_iptm" in normalised
        and "pair_chains_iptm" not in normalised
        and "chain_pair_iptm" not in normalised
    ):
        matrix = _chai_first_sample(normalised["per_chain_pair_iptm"])
        if isinstance(matrix, list):
            normalised["pair_chains_iptm"] = _matrix_list_to_nested_dict(matrix)

    if "has_inter_chain_clashes" in normalised and "has_clash" not in normalised:
        normalised["has_clash"] = normalised["has_inter_chain_clashes"]

    return normalised


def _chai_first_sample(value):
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


class ChaiProvider(BaseProvider):
    key, label = "chai1", "Chai-1 Discovery"
    detect = staticmethod(_looks_like_chai)
    scan = staticmethod(_scan_chai_dir)

    def load_model_data(self, pred_files, model, data, options, *, structure_index):
        del structure_index
        _load_chai_model_data(
            model, data, load_pae=options.load_pae, load_pde=options.load_pde
        )
