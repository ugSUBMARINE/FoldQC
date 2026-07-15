"""Protenix provider."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..confidence import PROTENIX_CONFIDENCE_SUMMARY
from ..loader_models import ModelFiles, PredictionData, PredictionFiles
from ..loader_utils import (
    STRUCTURE_SUFFIXES,
    _first_present,
    _float_or_none,
    _load_json,
    _safe_object_name,
    _squeezed_float32_array,
)
from .base import BaseProvider, has_ancestor_candidate


@dataclass
class _ProtenixCandidate:
    """One Protenix prediction sample's discovered paths."""

    structure_path: Path
    summary_path: Path
    full_data_path: Path | None
    source_prefix: str
    sample_rank: int
    seed: int | None
    ranking_score: float | None


def _looks_like_protenix(pred_dir: Path) -> bool:
    for predictions_dir in _protenix_prediction_dirs(pred_dir):
        for structure_path in _protenix_structure_paths(predictions_dir):
            parsed = _parse_protenix_stem(structure_path.stem)
            if parsed is None:
                continue
            prefix, sample_rank = parsed
            summary_path = (
                predictions_dir
                / f"{prefix}_summary_confidence_sample_{sample_rank}.json"
            )
            if summary_path.exists():
                return True
    return False


def _scan_protenix_dir(pred_dir: Path, provider: BaseProvider) -> PredictionFiles:
    """Discover Protenix seed/predictions output files."""
    candidates = _protenix_prediction_candidates(pred_dir)
    if not candidates:
        raise ValueError(
            f"No Protenix prediction files found in {pred_dir}.\n"
            "Expected files named like '<sample>_sample_<rank>.cif' with "
            "matching '<sample>_summary_confidence_sample_<rank>.json' files "
            "under a 'seed_<seed>/predictions' directory."
        )

    name = _protenix_prediction_name(candidates, pred_dir)
    files = provider.prediction_files(name=name, pred_dir=pred_dir)

    candidates.sort(
        key=lambda item: (
            item.ranking_score is None,
            -float(item.ranking_score or 0.0),
            item.seed if item.seed is not None else 0,
            item.sample_rank,
            item.structure_path.name,
        )
    )

    for rank, item in enumerate(candidates):
        label = f"rank {rank}"
        if item.seed is not None:
            label += f" - seed {item.seed}"
        label += f" sample {item.sample_rank}"
        metadata: dict[str, Any] = {
            "sample_rank": item.sample_rank,
            "source_prefix": item.source_prefix,
        }
        if item.seed is not None:
            metadata["seed"] = item.seed
        if item.ranking_score is not None:
            metadata["ranking_score"] = item.ranking_score
        files.models.append(
            ModelFiles(
                rank=rank,
                structure_path=item.structure_path,
                display_label=label,
                object_name=f"{_safe_object_name(name)}_model_{rank}",
                confidence_path=item.full_data_path,
                summary_path=item.summary_path,
                capabilities=frozenset(
                    {"plddt", "pae", "pde", "contact_probs"}
                    if item.full_data_path is not None
                    else {"plddt"}
                ),
                metadata=metadata,
            )
        )

    return files


def _load_protenix_model_data(
    model: ModelFiles,
    data: PredictionData,
    *,
    load_pae: bool,
    load_pde: bool,
    load_contact_probs: bool,
    load_token_plddt: bool,
    structure_index,
) -> None:
    needs_full = load_pae or load_pde or load_contact_probs or load_token_plddt
    if not needs_full or model.confidence_path is None:
        return

    full = _load_json(model.confidence_path)
    if load_pae:
        pae = _first_present(full, ("token_pair_pae", "pae"))
        if pae is not None:
            data.pae = _squeezed_float32_array(pae)
    if load_pde:
        pde = _first_present(full, ("token_pair_pde", "pde"))
        if pde is not None:
            data.pde = _squeezed_float32_array(pde)
    if load_contact_probs and "contact_probs" in full:
        data.contact_probs = _squeezed_float32_array(full["contact_probs"])
    if load_token_plddt:
        atom_plddt = _first_present(full, ("atom_plddt", "atom_plddts"))
        if atom_plddt is not None:
            if structure_index is None:
                raise ValueError(
                    f"No StructureIndex available for {model.structure_path.name}."
                )
            data.token_plddt = structure_index.collapse_atom_plddt(
                np.asarray(atom_plddt, dtype=np.float32)
            )
            data.token_plddt_source = "provider_atom_mean"


def _protenix_prediction_candidates(pred_dir: Path) -> list[_ProtenixCandidate]:
    candidates: list[_ProtenixCandidate] = []
    for predictions_dir in _protenix_prediction_dirs(pred_dir):
        for structure_path in _protenix_structure_paths(predictions_dir):
            parsed = _parse_protenix_stem(structure_path.stem)
            if parsed is None:
                continue
            prefix, sample_rank = parsed
            summary_path = (
                predictions_dir
                / f"{prefix}_summary_confidence_sample_{sample_rank}.json"
            )
            if not summary_path.exists():
                continue
            full_data_path = (
                predictions_dir / f"{prefix}_full_data_sample_{sample_rank}.json"
            )
            summary = _load_json(summary_path)
            candidates.append(
                _ProtenixCandidate(
                    structure_path=structure_path,
                    summary_path=summary_path,
                    full_data_path=full_data_path if full_data_path.exists() else None,
                    source_prefix=prefix,
                    sample_rank=sample_rank,
                    seed=_parse_protenix_seed(predictions_dir),
                    ranking_score=_float_or_none(summary.get("ranking_score")),
                )
            )
    return candidates


def _protenix_prediction_dirs(pred_dir: Path) -> list[Path]:
    candidates = []
    if pred_dir.name == "predictions":
        candidates.append(pred_dir)
    direct = pred_dir / "predictions"
    if direct.is_dir():
        candidates.append(direct)
    candidates.extend(
        path for path in pred_dir.glob("seed_*/predictions") if path.is_dir()
    )
    return sorted(set(candidates))


def _protenix_structure_paths(predictions_dir: Path) -> list[Path]:
    paths = [
        path
        for suffix in STRUCTURE_SUFFIXES
        for path in predictions_dir.glob(f"*_sample_*{suffix}")
        if _parse_protenix_stem(path.stem) is not None
    ]
    return sorted(paths, key=lambda path: path.name)


def _parse_protenix_stem(stem: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"(.+)_sample_(\d+)", stem)
    if match is None:
        return None
    prefix, sample = match.groups()
    if prefix.endswith("_summary_confidence") or prefix.endswith("_full_data"):
        return None
    return prefix, int(sample)


def _parse_protenix_seed(predictions_dir: Path) -> int | None:
    for parent in [predictions_dir.parent, *predictions_dir.parents]:
        match = re.fullmatch(r"seed_(\d+)", parent.name)
        if match:
            return int(match.group(1))
    return None


def _protenix_prediction_name(
    candidates: list[_ProtenixCandidate], pred_dir: Path
) -> str:
    prefixes = {item.source_prefix for item in candidates}
    if len(prefixes) == 1:
        return next(iter(prefixes))
    return pred_dir.name


class ProtenixProvider(BaseProvider):
    key, label = "protenix", "Protenix"
    confidence_summary = PROTENIX_CONFIDENCE_SUMMARY
    detect = staticmethod(_looks_like_protenix)

    def scan(self, path: Path) -> PredictionFiles:
        return _scan_protenix_dir(path, self)

    def load_model_data(self, pred_files, model, data, options, *, structure_index):
        _load_protenix_model_data(
            model,
            data,
            load_pae=options.load_pae,
            load_pde=options.load_pde,
            load_contact_probs=options.load_contact_probs,
            load_token_plddt=options.load_token_plddt,
            structure_index=structure_index,
        )

    def is_internal_candidate(self, candidate, candidates):
        internal = (
            candidate.path.name == "predictions"
            or candidate.path.name.startswith("seed_")
        )
        return internal and has_ancestor_candidate(
            candidate, candidates, provider=self.key
        )
