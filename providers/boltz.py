"""Boltz, Boltz Lab, and Boltz API providers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..loader_models import (
    ModelFiles,
    PredictionData,
    PredictionFiles,
)
from ..loader_utils import (
    _float_or_none,
    _load_json,
    _load_optional_json,
    _normalise_confidence,
    _safe_object_name,
)
from .base import BaseProvider, has_ancestor_candidate


@dataclass
class _BoltzSampleCandidate:
    """One Boltz Lab / API prediction sample's discovered paths."""

    sample_index: int
    structure_path: Path
    pae_path: Path | None
    confidence: dict | None = None
    structure_confidence: float | None = None


def _looks_like_boltz(pred_dir: Path) -> bool:
    name = pred_dir.name
    return bool(
        list(pred_dir.glob(f"{name}_model_*.cif"))
        or list(pred_dir.glob(f"{name}_model_*.pdb"))
    )


def _looks_like_boltz_lab(pred_dir: Path) -> bool:
    metrics_path = pred_dir / "metrics.json"
    return metrics_path.exists() and bool(_boltz_sample_structure_paths(pred_dir))


def _looks_like_boltz_api(pred_dir: Path) -> bool:
    prediction_dir = _boltz_api_prediction_dir(pred_dir)
    if prediction_dir is None:
        return False
    metrics = _load_optional_json(prediction_dir / "metrics.json")
    if _is_boltz_api_metrics(metrics):
        return True
    run = _load_optional_json(pred_dir / "run.json")
    return _is_boltz_api_output(run.get("output") if isinstance(run, dict) else None)


def _scan_boltz_dir(pred_dir: Path) -> PredictionFiles:
    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="boltz",
        input_path=pred_dir,
        capabilities={"structure_plddt"},
    )

    rank_re = re.compile(rf"{re.escape(name)}_model_(\d+)")

    def rank_of(p: Path) -> int:
        match = rank_re.search(p.stem)
        return int(match.group(1)) if match else -1

    structures_by_rank: dict[int, Path] = {}
    for path in sorted(pred_dir.glob(f"{name}_model_*.pdb"), key=rank_of):
        rank = rank_of(path)
        if rank >= 0:
            structures_by_rank[rank] = path
    for path in sorted(pred_dir.glob(f"{name}_model_*.cif"), key=rank_of):
        rank = rank_of(path)
        if rank >= 0:
            structures_by_rank[rank] = path

    if not structures_by_rank:
        raise ValueError(
            f"No structure files found in {pred_dir}.\n"
            "Expected files named like '<name>_model_<rank>.cif' or '.pdb'."
        )

    files.models = [
        ModelFiles(
            rank=rank,
            structure_path=path,
            display_label=f"model_{rank}",
            object_name=f"{_safe_object_name(name)}_model_{rank}",
        )
        for rank, path in sorted(structures_by_rank.items())
    ]

    def by_rank(pattern: str) -> dict[int, Path]:
        result = {}
        for path in sorted(pred_dir.glob(pattern), key=rank_of):
            rank = rank_of(path)
            if rank >= 0:
                result[rank] = path
        return result

    confidence = by_rank(f"confidence_{name}_model_*.json")
    plddt = by_rank(f"plddt_{name}_model_*.npz")
    pae = by_rank(f"pae_{name}_model_*.npz")
    pde = by_rank(f"pde_{name}_model_*.npz")
    for model in files.models:
        model.confidence_path = confidence.get(model.rank)
        model.plddt_path = plddt.get(model.rank)
        model.pae_path = pae.get(model.rank)
        model.pde_path = pde.get(model.rank)

    if plddt:
        files.capabilities.add("plddt")
    if pae:
        files.capabilities.add("pae")
    if pde:
        files.capabilities.add("pde")

    affinity_path = pred_dir / f"affinity_{name}.json"
    if affinity_path.exists():
        files.affinity_file = affinity_path

    embeddings_path = pred_dir / f"embeddings_{name}.npz"
    if embeddings_path.exists():
        files.embeddings_file = embeddings_path

    return files


def _scan_boltz_lab_dir(pred_dir: Path) -> PredictionFiles:
    candidates = _boltz_sample_candidates(pred_dir)
    if not candidates:
        raise ValueError(
            f"No Boltz Lab prediction files found in {pred_dir}.\n"
            "Expected files named like 'sample_<n>_predicted_structure.cif' "
            "or '.pdb' with a metrics.json file."
        )

    metrics = _load_optional_json(pred_dir / "metrics.json")
    sample_results = (
        metrics.get("sample_results", {}) if isinstance(metrics, dict) else {}
    )
    if not isinstance(sample_results, dict):
        sample_results = {}

    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="boltz_lab",
        input_path=pred_dir,
        capabilities={"structure_plddt"},
    )

    for rank, item in enumerate(sorted(candidates, key=lambda item: item.sample_index)):
        sample_index = item.sample_index
        confidence = sample_results.get(f"sample_{sample_index}")
        files.models.append(
            ModelFiles(
                rank=rank,
                structure_path=item.structure_path,
                display_label=f"sample {sample_index}",
                object_name=f"{_safe_object_name(name)}_model_{rank}",
                pae_path=item.pae_path,
                metadata={
                    "sample_index": sample_index,
                    "confidence": confidence if isinstance(confidence, dict) else None,
                },
            )
        )

    if any(model.pae_path is not None for model in files.models):
        files.capabilities.add("pae")
    return files


def _scan_boltz_api_dir(pred_dir: Path) -> PredictionFiles:
    prediction_dir = _boltz_api_prediction_dir(pred_dir)
    if prediction_dir is None:
        raise ValueError(
            f"No Boltz API prediction files found in {pred_dir}.\n"
            "Expected a run root with 'outputs/files/prediction' or an extracted "
            "'prediction' folder."
        )

    candidates = _boltz_sample_candidates(prediction_dir)
    if not candidates:
        raise ValueError(
            f"No Boltz API prediction files found in {prediction_dir}.\n"
            "Expected files named like 'sample_<n>_predicted_structure.cif' "
            "or '.pdb'."
        )

    metrics_by_sample = _boltz_api_metrics_by_sample(pred_dir, prediction_dir)
    for item in candidates:
        item.confidence = metrics_by_sample.get(item.sample_index)
        item.structure_confidence = _float_or_none(
            (item.confidence or {}).get("structure_confidence")
        )

    candidates.sort(
        key=lambda item: (
            item.structure_confidence is None,
            -float(item.structure_confidence or 0.0),
            item.sample_index,
        )
    )

    name = pred_dir.name
    files = PredictionFiles(
        name=name,
        pred_dir=pred_dir,
        provider="boltz_api",
        input_path=pred_dir,
        capabilities={"structure_plddt"},
    )

    for rank, item in enumerate(candidates):
        metadata: dict[str, Any] = {
            "sample_index": item.sample_index,
            "confidence": item.confidence,
        }
        if item.structure_confidence is not None:
            metadata["structure_confidence"] = item.structure_confidence
        files.models.append(
            ModelFiles(
                rank=rank,
                structure_path=item.structure_path,
                display_label=f"rank {rank} - sample {item.sample_index}",
                object_name=f"{_safe_object_name(name)}_model_{rank}",
                pae_path=item.pae_path,
                metadata=metadata,
            )
        )

    if any(model.pae_path is not None for model in files.models):
        files.capabilities.add("pae")
    return files


def _load_boltz_model_data(
    pred_files: PredictionFiles,
    model: ModelFiles,
    data: PredictionData,
    *,
    load_pae: bool,
    load_pde: bool,
    load_embeddings: bool,
    load_plddt: bool,
) -> None:
    if load_plddt and model.plddt_path is not None:
        data.plddt = np.load(model.plddt_path)["plddt"]

    if load_pae and model.pae_path is not None:
        data.pae = np.load(model.pae_path)["pae"]

    if load_pde and model.pde_path is not None:
        data.pde = np.load(model.pde_path)["pde"]

    if model.confidence_path is not None:
        confidence = _load_json(model.confidence_path)
        data.confidence = _normalise_confidence(confidence)
    elif isinstance(model.metadata.get("confidence"), dict):
        data.confidence = _normalise_confidence(model.metadata["confidence"])

    if pred_files.affinity_file is not None:
        data.affinity = _load_json(pred_files.affinity_file)

    if load_embeddings and pred_files.embeddings_file is not None:
        emb = np.load(pred_files.embeddings_file)
        data.embeddings_s = emb["s"]
        data.embeddings_z = emb["z"]


def _boltz_api_prediction_dir(pred_dir: Path) -> Path | None:
    if _is_boltz_api_prediction_leaf(pred_dir):
        return pred_dir
    nested = pred_dir / "outputs" / "files" / "prediction"
    if _is_boltz_api_prediction_leaf(nested):
        return nested
    return None


def _is_boltz_api_prediction_leaf(pred_dir: Path) -> bool:
    return (
        pred_dir.is_dir()
        and (pred_dir / "metrics.json").exists()
        and bool(_boltz_sample_structure_paths(pred_dir))
    )


def _boltz_sample_structure_paths(pred_dir: Path) -> dict[int, Path]:
    structures_by_sample: dict[int, Path] = {}
    for suffix in (".pdb", ".cif"):
        for path in pred_dir.glob(f"sample_*_predicted_structure{suffix}"):
            sample_index = _parse_boltz_sample_structure_stem(path.stem)
            if sample_index is not None:
                structures_by_sample[sample_index] = path
    return structures_by_sample


def _parse_boltz_sample_structure_stem(stem: str) -> int | None:
    match = re.fullmatch(r"sample_(\d+)_predicted_structure", stem)
    return int(match.group(1)) if match else None


def _boltz_sample_candidates(pred_dir: Path) -> list[_BoltzSampleCandidate]:
    candidates = []
    for sample_index, structure_path in _boltz_sample_structure_paths(pred_dir).items():
        pae_path = pred_dir / f"sample_{sample_index}_pae.npz"
        candidates.append(
            _BoltzSampleCandidate(
                sample_index=sample_index,
                structure_path=structure_path,
                pae_path=pae_path if pae_path.exists() else None,
            )
        )
    return sorted(candidates, key=lambda item: item.sample_index)


def _is_boltz_api_metrics(metrics: dict | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    return "all_sample_results" in metrics or "best_sample" in metrics


def _is_boltz_api_output(output: dict | None) -> bool:
    if not isinstance(output, dict):
        return False
    return "all_sample_results" in output or "best_sample" in output


def _boltz_api_metrics_by_sample(
    pred_dir: Path,
    prediction_dir: Path,
) -> dict[int, dict]:
    metrics = _load_optional_json(prediction_dir / "metrics.json")
    results = None
    if isinstance(metrics, dict):
        results = metrics.get("all_sample_results")

    if not isinstance(results, list) or not results:
        run = _load_optional_json(pred_dir / "run.json")
        output = run.get("output") if isinstance(run, dict) else None
        if isinstance(output, dict):
            results = output.get("all_sample_results")

    if not isinstance(results, list):
        return {}

    by_sample = {}
    for sample_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        metrics_for_sample = result.get("metrics", result)
        if isinstance(metrics_for_sample, dict):
            by_sample[sample_index] = metrics_for_sample
    return by_sample


class BoltzProvider(BaseProvider):
    key, label = "boltz", "Boltz"
    detect = staticmethod(_looks_like_boltz)
    scan = staticmethod(_scan_boltz_dir)

    def load_model_data(self, pred_files, model, data, options):
        _load_boltz_model_data(
            pred_files,
            model,
            data,
            load_pae=options.load_pae,
            load_pde=options.load_pde,
            load_embeddings=options.load_embeddings,
            load_plddt=options.load_plddt,
        )


class BoltzLabProvider(BoltzProvider):
    key, label = "boltz_lab", "Boltz Lab"
    detect = staticmethod(_looks_like_boltz_lab)
    scan = staticmethod(_scan_boltz_lab_dir)


class BoltzAPIProvider(BoltzProvider):
    key, label = "boltz_api", "Boltz API"
    detect = staticmethod(_looks_like_boltz_api)
    scan = staticmethod(_scan_boltz_api_dir)

    def is_internal_candidate(self, candidate, candidates):
        return candidate.path.name == "prediction" and has_ancestor_candidate(
            candidate, candidates, provider=self.key
        )
