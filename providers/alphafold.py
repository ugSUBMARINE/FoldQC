"""AlphaFold 3 local and Server providers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..loader_models import ModelFiles, PredictionData, PredictionFiles
from ..loader_utils import (
    _first,
    _load_json,
    _normalise_confidence,
    _safe_object_name,
)
from .base import BaseProvider, has_ancestor_candidate


@dataclass
class _AF3Sample:
    """One AF3 seed/sample directory's discovered paths and metadata."""

    model_path: Path
    confidence_path: Path | None
    summary_path: Path | None
    summary: dict
    ranking_score: float | None
    seed: int | None
    sample: int | None


def _looks_like_af3(pred_dir: Path) -> bool:
    if _af3_model_path(pred_dir) is not None and (
        _has_af3_metadata(pred_dir) or _af3_ranking_scores_path(pred_dir) is not None
    ):
        return True

    has_root_ranking_scores = _af3_ranking_scores_path(pred_dir) is not None
    for sample_dir in pred_dir.glob("seed-*_sample-*"):
        if not sample_dir.is_dir():
            continue
        if _af3_model_path(sample_dir) is not None and (
            _has_af3_metadata(sample_dir) or has_root_ranking_scores
        ):
            return True
    return False


def _has_af3_metadata(pred_dir: Path) -> bool:
    return (
        _first_confidence_json(pred_dir) is not None
        or _af3_summary_path(pred_dir) is not None
    )


def _looks_like_af3_server(pred_dir: Path) -> bool:
    return bool(
        list(pred_dir.glob("*_full_data_*.json"))
        or list(pred_dir.glob("full_data_*.json"))
    )


def _scan_af3_dir(pred_dir: Path) -> PredictionFiles:
    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="alphafold3",
        input_path=pred_dir,
        capabilities={"plddt"},
    )

    ranking_scores = _load_af3_ranking_scores(pred_dir)
    samples: list[_AF3Sample] = []
    for sample_dir in sorted(pred_dir.glob("seed-*_sample-*")):
        if not sample_dir.is_dir():
            continue
        model_path = _af3_model_path(sample_dir)
        if model_path is None:
            continue
        confidence_path = _first_confidence_json(sample_dir)
        summary_path = _af3_summary_path(sample_dir)
        summary = _load_json(summary_path) if summary_path is not None else {}
        seed, sample = _parse_af3_seed_sample(sample_dir.name)
        samples.append(
            _AF3Sample(
                model_path=model_path,
                confidence_path=confidence_path,
                summary_path=summary_path,
                summary=summary,
                ranking_score=ranking_scores.get((seed, sample)),
                seed=seed,
                sample=sample,
            )
        )

    if samples:
        samples.sort(
            key=lambda item: (
                _ranking_sort_key(item.summary, item.ranking_score),
                item.seed if item.seed is not None else 0,
                item.sample if item.sample is not None else 0,
            )
        )
        for rank, item in enumerate(samples):
            label = "rank " + str(rank)
            if item.seed is not None and item.sample is not None:
                label += f" - seed {item.seed} sample {item.sample}"
            files.models.append(
                ModelFiles(
                    rank=rank,
                    structure_path=item.model_path,
                    display_label=label,
                    object_name=f"{_safe_object_name(name)}_model_{rank}",
                    confidence_path=item.confidence_path,
                    summary_path=item.summary_path,
                    metadata={"seed": item.seed, "sample": item.sample},
                )
            )
    else:
        model_path = _af3_model_path(pred_dir)
        if model_path is None:
            raise ValueError(f"No AlphaFold 3 model CIF found in {pred_dir}.")
        stem = model_path.stem.removesuffix("_model")
        files.models.append(
            ModelFiles(
                rank=0,
                structure_path=model_path,
                display_label="rank 0",
                object_name=f"{_safe_object_name(stem)}_model_0",
                confidence_path=_first_confidence_json(pred_dir),
                summary_path=_af3_summary_path(pred_dir),
            )
        )

    if any(model.confidence_path is not None for model in files.models):
        files.capabilities.update({"pae", "contact_probs"})
    return files


def _scan_af3_server_dir(pred_dir: Path) -> PredictionFiles:
    """Discover ranked AlphaFold 3 Server output files."""
    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="af3_server",
        input_path=pred_dir,
        capabilities={"plddt"},
    )

    model_re = re.compile(r"(.+)_model_(\d+)$")
    ranked_models: list[tuple[int, str, Path]] = []
    for path in sorted(pred_dir.glob("*_model_*.cif")):
        match = model_re.fullmatch(path.stem)
        if match is None:
            continue
        prefix, rank_text = match.groups()
        ranked_models.append((int(rank_text), prefix, path))

    if not ranked_models:
        raise ValueError(f"No AlphaFold 3 Server model CIFs found in {pred_dir}.")

    for rank, prefix, structure_path in sorted(ranked_models):
        full_data_path = _ranked_json_path(
            pred_dir,
            prefix,
            "full_data",
            rank,
        )
        summary_path = _ranked_json_path(
            pred_dir,
            prefix,
            "summary_confidences",
            rank,
        )
        files.models.append(
            ModelFiles(
                rank=rank,
                structure_path=structure_path,
                display_label=f"rank {rank}",
                object_name=f"{_safe_object_name(name)}_model_{rank}",
                confidence_path=full_data_path,
                summary_path=summary_path,
                metadata={"source_prefix": prefix},
            )
        )

    if any(model.confidence_path is not None for model in files.models):
        files.capabilities.update({"pae", "contact_probs"})
    return files


def _load_af3_model_data(
    model: ModelFiles,
    data: PredictionData,
    *,
    load_pae: bool,
    load_contact_probs: bool,
    load_token_plddt: bool,
    structure_index,
) -> None:
    needs_full = load_pae or load_contact_probs or load_token_plddt
    if not needs_full or model.confidence_path is None:
        return

    full = _load_json(model.confidence_path)
    full_summary = {
        key: value
        for key, value in full.items()
        if key not in {"pae", "contact_probs", "atom_plddts"}
    }
    data.confidence = _normalise_confidence(
        {**(data.summary_confidence or {}), **full_summary}
    )

    if load_pae and "pae" in full:
        data.pae = np.asarray(full["pae"], dtype=np.float32)
    if load_contact_probs and "contact_probs" in full:
        data.contact_probs = np.asarray(full["contact_probs"], dtype=np.float32)
    if load_token_plddt and "atom_plddts" in full:
        if structure_index is None:
            raise ValueError(
                f"No StructureIndex available for {model.structure_path.name}."
            )
        data.token_plddt = structure_index.collapse_atom_plddt(
            np.asarray(full["atom_plddts"], dtype=np.float32)
        )
        data.token_plddt_source = "provider_atom_mean"


def _af3_model_path(path: Path) -> Path | None:
    """Return an AF3 model CIF using current or older truncated names."""
    return _first(path.glob("*_model.cif")) or (
        path / "model.cif" if (path / "model.cif").exists() else None
    )


def _af3_summary_path(path: Path) -> Path | None:
    """Return an AF3 summary JSON using current or older truncated names."""
    return _first(path.glob("*_summary_confidences.json")) or (
        path / "summary_confidences.json"
        if (path / "summary_confidences.json").exists()
        else None
    )


def _first_confidence_json(path: Path) -> Path | None:
    candidates = [
        candidate
        for candidate in path.glob("*_confidences.json")
        if candidate.name != "summary_confidences.json"
        and not candidate.name.endswith("_summary_confidences.json")
    ]
    return _first(candidates) or (
        path / "confidences.json" if (path / "confidences.json").exists() else None
    )


def _parse_af3_seed_sample(dirname: str) -> tuple[int | None, int | None]:
    match = re.fullmatch(r"seed-(\d+)_sample-(\d+)", dirname)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _load_af3_ranking_scores(
    pred_dir: Path,
) -> dict[tuple[int | None, int | None], float]:
    """Read AF3 ranking scores using (seed, sample) as the lookup key."""
    path = _af3_ranking_scores_path(pred_dir)
    if path is None:
        return {}

    scores: dict[tuple[int | None, int | None], float] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                seed = int(row["seed"])
                sample = int(row["sample"])
                score = float(row["ranking_score"])
            except (KeyError, TypeError, ValueError):
                continue
            scores[(seed, sample)] = score
    return scores


def _af3_ranking_scores_path(pred_dir: Path) -> Path | None:
    """Return current or older AF3 ranking-score CSV path."""
    prefixed = pred_dir / f"{pred_dir.name}_ranking_scores.csv"
    truncated = pred_dir / "ranking_scores.csv"
    if prefixed.exists():
        return prefixed
    if truncated.exists():
        return truncated
    return _first(pred_dir.glob("*_ranking_scores.csv"))


def _ranked_json_path(
    pred_dir: Path,
    prefix: str,
    stem: str,
    rank: int,
) -> Path | None:
    """Return prefixed or truncated ranked JSON path."""
    prefixed = pred_dir / f"{prefix}_{stem}_{rank}.json"
    truncated = pred_dir / f"{stem}_{rank}.json"
    if prefixed.exists():
        return prefixed
    if truncated.exists():
        return truncated
    return None


def _ranking_sort_key(summary: dict, ranking_score: float | None = None) -> float:
    value = ranking_score if ranking_score is not None else summary.get("ranking_score")
    try:
        return -float(value)  # type: ignore[arg-type]  # None/invalid caught below
    except (TypeError, ValueError):
        return float("inf")


class AlphaFold3Provider(BaseProvider):
    key, label = "alphafold3", "AlphaFold 3"
    detect = staticmethod(_looks_like_af3)
    scan = staticmethod(_scan_af3_dir)

    def load_model_data(self, pred_files, model, data, options, *, structure_index):
        _load_af3_model_data(
            model,
            data,
            load_pae=options.load_pae,
            load_contact_probs=options.load_contact_probs,
            load_token_plddt=options.load_token_plddt,
            structure_index=structure_index,
        )

    def is_internal_candidate(self, candidate, candidates):
        seed, sample = _parse_af3_seed_sample(candidate.path.name)
        return (
            seed is not None
            and sample is not None
            and has_ancestor_candidate(candidate, candidates, provider=self.key)
        )


class AF3ServerProvider(AlphaFold3Provider):
    key, label = "af3_server", "AlphaFold 3 Server"
    detect = staticmethod(_looks_like_af3_server)
    scan = staticmethod(_scan_af3_server_dir)
