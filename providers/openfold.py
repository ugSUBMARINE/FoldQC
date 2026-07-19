"""OpenFold3 prediction provider."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..confidence import OPENFOLD3_CONFIDENCE_SUMMARY
from ..loader_models import ModelFiles, PredictionData, PredictionFiles
from ..loader_utils import (
    STRUCTURE_SUFFIXES,
    _float_or_none,
    _load_json,
    _safe_object_name,
)
from ..provider_errors import ProviderContractError
from ..structure_index import StructureIndex
from .base import BaseProvider, has_ancestor_candidate

_MODEL_STEM = re.compile(
    r"(?P<prefix>.+)_seed_(?P<seed>\d+)_sample_(?P<sample>\d+)_model"
)
_SEED_DIR = re.compile(r"seed_(\d+)")
_PAIR_KEY = re.compile(r"\(\s*(.+?)\s*,\s*(.+?)\s*\)")


@dataclass(frozen=True)
class _OpenFold3Sample:
    structure_path: Path
    summary_path: Path
    confidence_path: Path | None
    source_prefix: str
    seed: int
    sample: int
    ranking_score: float | None


def _openfold3_samples(
    pred_dir: Path, *, read_ranking_scores: bool = True
) -> list[_OpenFold3Sample]:
    samples: list[_OpenFold3Sample] = []
    for seed_dir in _openfold3_seed_dirs(pred_dir):
        for suffix in STRUCTURE_SUFFIXES:
            for structure_path in sorted(seed_dir.glob(f"*_model{suffix}")):
                match = _MODEL_STEM.fullmatch(structure_path.stem)
                if match is None:
                    continue
                prefix = match.group("prefix")
                seed = int(match.group("seed"))
                sample = int(match.group("sample"))
                seed_match = _SEED_DIR.fullmatch(seed_dir.name)
                if seed_match is not None and seed != int(seed_match.group(1)):
                    continue
                base = f"{prefix}_seed_{seed}_sample_{sample}"
                summary_path = seed_dir / f"{base}_confidences_aggregated.json"
                if not summary_path.is_file():
                    continue
                confidence_path = seed_dir / f"{base}_confidences.json"
                ranking_score = None
                if read_ranking_scores:
                    summary = _load_json(summary_path)
                    ranking_score = (
                        _float_or_none(summary.get("sample_ranking_score"))
                        if isinstance(summary, dict)
                        else None
                    )
                samples.append(
                    _OpenFold3Sample(
                        structure_path=structure_path,
                        summary_path=summary_path,
                        confidence_path=(
                            confidence_path if confidence_path.is_file() else None
                        ),
                        source_prefix=prefix,
                        seed=seed,
                        sample=sample,
                        ranking_score=ranking_score,
                    )
                )
    return samples


def _openfold3_seed_dirs(pred_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    if _SEED_DIR.fullmatch(pred_dir.name):
        candidates.append(pred_dir)
    candidates.extend(
        path
        for path in pred_dir.glob("seed_*")
        if path.is_dir() and _SEED_DIR.fullmatch(path.name)
    )
    return sorted(set(candidates))


def _looks_like_openfold3(pred_dir: Path) -> bool:
    return pred_dir.is_dir() and bool(
        _openfold3_samples(pred_dir, read_ranking_scores=False)
    )


def _scan_openfold3_dir(pred_dir: Path, provider: BaseProvider) -> PredictionFiles:
    samples = _openfold3_samples(pred_dir)
    if not samples:
        raise ValueError(
            f"No OpenFold3 prediction files found in {pred_dir}.\n"
            "Expected seed_<seed> directories containing matching "
            "'<query>_seed_<seed>_sample_<sample>_model.cif/.pdb' and "
            "'..._confidences_aggregated.json' files."
        )

    prefixes = {item.source_prefix for item in samples}
    name = next(iter(prefixes)) if len(prefixes) == 1 else pred_dir.name
    files = provider.prediction_files(name=name, pred_dir=pred_dir)
    samples.sort(
        key=lambda item: (
            item.ranking_score is None,
            -float(item.ranking_score or 0.0),
            item.seed,
            item.sample,
            item.structure_path.name,
        )
    )
    for rank, item in enumerate(samples):
        capabilities = {"plddt"}
        if item.confidence_path is not None:
            capabilities.update({"pae", "pde"})
        metadata: dict[str, Any] = {
            "seed": item.seed,
            "sample": item.sample,
            "source_prefix": item.source_prefix,
        }
        if item.ranking_score is not None:
            metadata["ranking_score"] = item.ranking_score
        files.models.append(
            ModelFiles(
                rank=rank,
                structure_path=item.structure_path,
                display_label=(f"rank {rank} - seed {item.seed} sample {item.sample}"),
                object_name=f"{_safe_object_name(name)}_model_{rank}",
                confidence_path=item.confidence_path,
                summary_path=item.summary_path,
                capabilities=frozenset(capabilities),
                metadata=metadata,
            )
        )
    return files


def _load_openfold3_model_data(
    model: ModelFiles,
    data: PredictionData,
    *,
    load_pae: bool,
    load_pde: bool,
    load_token_plddt: bool,
    structure_index: StructureIndex | None,
) -> None:
    if not (load_pae or load_pde or load_token_plddt):
        return
    if model.confidence_path is None:
        return

    full = _load_json(model.confidence_path)
    if load_pae and "pae" in full:
        data.pae = np.asarray(full["pae"], dtype=np.float32)
    if load_pde and "pde" in full:
        data.pde = np.asarray(full["pde"], dtype=np.float32)
    if load_token_plddt and "plddt" in full:
        if structure_index is None:
            raise ValueError(
                f"No StructureIndex available for {model.structure_path.name}."
            )
        data.token_plddt = structure_index.collapse_atom_plddt(
            np.asarray(full["plddt"], dtype=np.float32)
        )
        data.token_plddt_source = "provider_atom_mean"


def _normalize_scalar_confidence(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    if "ranking_score" not in normalized and "sample_ranking_score" in normalized:
        normalized["ranking_score"] = normalized["sample_ranking_score"]
    if "complex_plddt" not in normalized and "avg_plddt" in normalized:
        normalized["complex_plddt"] = normalized["avg_plddt"]
    raw_clash = normalized.get("has_clash")
    if isinstance(raw_clash, (int, float, np.integer, np.floating)) and not isinstance(
        raw_clash, (bool, np.bool_)
    ):
        numeric = float(raw_clash)
        if np.isfinite(numeric) and numeric in (0.0, 1.0):
            normalized["has_clash"] = bool(numeric)
    return normalized


def _normalize_chain_confidence(
    payload: dict,
    *,
    chain_order: tuple[str, ...],
    model: ModelFiles,
) -> dict:
    normalized = dict(payload)
    chain_indices = {chain: index for index, chain in enumerate(chain_order)}
    chain_ptm = normalized.get("chain_ptm")
    if isinstance(chain_ptm, dict):
        values: list[object | None] = [None] * len(chain_order)
        for chain, value in chain_ptm.items():
            index = _chain_index(chain, chain_indices, model, "chain_ptm")
            values[index] = value
        normalized["chain_ptm"] = values
    for field in ("chain_pair_iptm", "bespoke_iptm"):
        pair_values = normalized.get(field)
        if not isinstance(pair_values, dict) or any(
            isinstance(value, (dict, list)) for value in pair_values.values()
        ):
            continue
        matrix: list[list[object | None]] = [
            [None] * len(chain_order) for _ in chain_order
        ]
        for key, value in pair_values.items():
            match = _PAIR_KEY.fullmatch(str(key))
            if match is None:
                raise ProviderContractError(
                    f"{_model_context(model)}: {field} pair key {key!r} must "
                    "have the form '(chain_a, chain_b)'."
                )
            row = _chain_index(match.group(1), chain_indices, model, field)
            column = _chain_index(match.group(2), chain_indices, model, field)
            existing = matrix[row][column]
            reverse = matrix[column][row]
            if (existing is not None and existing != value) or (
                reverse is not None and reverse != value
            ):
                raise ProviderContractError(
                    f"{_model_context(model)}: {field} contains conflicting values "
                    f"for pair {key!r}."
                )
            matrix[row][column] = value
            matrix[column][row] = value
        normalized[field] = matrix
    return normalized


def _chain_index(
    chain: object,
    indices: dict[str, int],
    model: ModelFiles,
    field: str,
) -> int:
    label = str(chain).strip()
    try:
        return indices[label]
    except KeyError as exc:
        raise ProviderContractError(
            f"{_model_context(model)}: {field} references unknown chain {label!r}."
        ) from exc


def _model_context(model: ModelFiles) -> str:
    return (
        f"Provider 'openfold3', model {model.display_label!r}, "
        f"source {model.summary_path or '<unknown source>'}"
    )


class OpenFold3Provider(BaseProvider):
    key, label = "openfold3", "OpenFold3"
    confidence_summary = OPENFOLD3_CONFIDENCE_SUMMARY
    detect = staticmethod(_looks_like_openfold3)

    def scan(self, path: Path) -> PredictionFiles:
        return _scan_openfold3_dir(path, self)

    def normalize_confidence_payload(self, payload: dict | None) -> dict | None:
        return _normalize_scalar_confidence(payload)

    def normalize_model_confidence_payload(
        self,
        payload: dict | None,
        *,
        model: ModelFiles,
        structure_index: StructureIndex,
    ) -> dict | None:
        normalized = _normalize_scalar_confidence(payload)
        if not isinstance(normalized, dict):
            return normalized
        return _normalize_chain_confidence(
            normalized,
            chain_order=structure_index.token_map.chain_order,
            model=model,
        )

    def load_model_data(self, pred_files, model, data, options, *, structure_index):
        _load_openfold3_model_data(
            model,
            data,
            load_pae=options.load_pae,
            load_pde=options.load_pde,
            load_token_plddt=options.load_token_plddt,
            structure_index=structure_index,
        )

    def is_internal_candidate(self, candidate, candidates):
        return bool(
            _SEED_DIR.fullmatch(candidate.path.name)
        ) and has_ancestor_candidate(
            candidate,
            candidates,
            provider=self.key,
        )
