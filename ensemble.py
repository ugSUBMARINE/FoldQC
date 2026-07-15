"""
Ensemble
========
Utilities for working with multiple ranked prediction models.

When a provider produces N models, N structure files can be loaded as one
object-based ensemble ordered by rank. This viewer-independent module prepares
canonical model states and computes alignment plans, per-token RMSD, and metric
consensus (mean ± std). Viewer transactions live in :mod:`gui_loading`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .compute import plddt_values_for
from .model_state import ModelState

if TYPE_CHECKING:
    from .loader_models import PredictionData, PredictionFiles
    from .token_map import TokenMap


PhaseReporter = Callable[[str], None]


@dataclass(frozen=True)
class EnsembleMember:
    """Viewer metadata for one rank in an active object-based ensemble."""

    rank: int
    obj_name: str


@dataclass(frozen=True)
class EnsembleState:
    """All committed ensemble metadata, keyed to canonical model ranks."""

    group_name: str
    members: tuple[EnsembleMember, ...]
    aligned: bool
    rmsd: np.ndarray
    plddt_mean: np.ndarray
    plddt_std: np.ndarray

    def __post_init__(self) -> None:
        members = tuple(self.members)
        ranks = tuple(member.rank for member in members)
        if not ranks:
            raise ValueError("EnsembleState requires at least one member.")
        if len(ranks) != len(set(ranks)):
            raise ValueError("EnsembleState member ranks must be unique.")
        object.__setattr__(self, "members", members)
        for name in ("rmsd", "plddt_mean", "plddt_std"):
            values = np.array(getattr(self, name), dtype=np.float32, copy=True)
            if values.ndim != 1:
                raise ValueError(f"EnsembleState {name} must be one-dimensional.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(member.rank for member in self.members)


@dataclass(frozen=True)
class PreparedEnsembleMember:
    """One ensemble member prepared without accessing a molecular viewer."""

    model_label: str
    obj_name: str
    model_state: ModelState

    @property
    def rank(self) -> int:
        return self.model_state.rank

    @property
    def structure_path(self) -> Path:
        return Path(self.model_state.data.structure_path)

    @property
    def data(self) -> PredictionData:
        return self.model_state.data

    @property
    def token_map(self) -> TokenMap:
        return self.model_state.token_map


@dataclass(frozen=True)
class PreparedEnsemble:
    """Atomically prepared ensemble data borrowed from one prediction."""

    pred_files: PredictionFiles
    group_name: str
    members: tuple[PreparedEnsembleMember, ...]
    skip_alignment: bool
    reference_rank: int
    core_indices: tuple[int, ...]
    plddt_mean: np.ndarray
    plddt_std: np.ndarray


@dataclass(frozen=True)
class AlignmentTransform:
    """Rigid transform for one non-reference ensemble member."""

    rank: int
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class AlignmentPlan:
    """Pure alignment result ready to apply to viewer objects."""

    transforms: tuple[AlignmentTransform, ...]
    transformed_coords: tuple[np.ndarray, ...]
    rmsd: np.ndarray


def default_group_name(pred_files: PredictionFiles) -> str:
    """Return the default viewer group name for a prediction ensemble."""
    first_obj = pred_files.models[0].object_name
    obj_prefix = first_obj.rsplit("_", 1)[0]
    return f"{obj_prefix}_ensemble"


def _ordered_token_identities(token_map: TokenMap) -> tuple[tuple, ...]:
    return tuple(
        (
            token.chain_id,
            token.residue_id,
            token.res_name,
            token.atom_name if token.is_hetatm else None,
        )
        for token in token_map
    )


def validate_prepared_members(
    members: Sequence[PreparedEnsembleMember],
) -> None:
    """Validate token order and pLDDT lengths before touching the viewer."""
    if not members:
        raise ValueError("No ensemble models were loaded.")
    reference = members[0]
    reference_length = len(reference.token_map)
    reference_identities = _ordered_token_identities(reference.token_map)
    for member in members:
        if len(member.token_map) != reference_length:
            raise ValueError(
                f"Token count mismatch: {member.obj_name} maps to "
                f"{len(member.token_map)} tokens, but {reference.obj_name} maps "
                f"to {reference_length} tokens."
            )
        plddt = _member_plddt(member)
        if len(plddt) != reference_length:
            raise ValueError(
                f"pLDDT length mismatch for model_{member.rank}: "
                f"{len(plddt)} values for {reference_length} tokens."
            )
        if _ordered_token_identities(member.token_map) != reference_identities:
            raise ValueError(
                f"Token order mismatch for model_{member.rank}; ensemble models "
                "must contain the same ordered prediction tokens."
            )


def _data_load_flags(_existing: PredictionData | None) -> dict[str, bool]:
    """Return the minimal flags needed to add canonical pLDDT to a state."""
    return {
        "load_pae": False,
        "load_pde": False,
        "load_contact_probs": False,
        "load_token_plddt": True,
    }


def _has_plddt(data: PredictionData | None) -> bool:
    values, _source = plddt_values_for(data)
    return values is not None


def prepare_ensemble(
    pred_files: PredictionFiles,
    *,
    skip_alignment: bool,
    existing_states_by_rank: Mapping[int, ModelState] | None = None,
    report_phase: PhaseReporter | None = None,
) -> PreparedEnsemble:
    """Load and validate ensemble data without accessing Qt or PyMOL."""
    from .loader import load_prediction_data
    from .structure_index import StructureIndex

    report = report_phase or (lambda _label: None)
    models = tuple(pred_files.models)
    if not models:
        raise ValueError("No ensemble models were found.")
    existing_states_by_rank = existing_states_by_rank or {}
    prepared: list[PreparedEnsembleMember] = []
    total = len(models)
    for index, model in enumerate(models, start=1):
        report(f"Preparing {model.display_label} ensemble data… ({index}/{total})")
        existing = existing_states_by_rank.get(model.rank)
        structure_path = (
            pred_files.structure_path(model.rank)
            if hasattr(pred_files, "structure_path")
            else model.structure_path
        )
        structure_index = (
            existing.structure_index
            if existing is not None
            else StructureIndex.from_path(structure_path)
        )
        data = None if existing is None else existing.data
        if not _has_plddt(data):
            data = load_prediction_data(
                pred_files,
                model.rank,
                structure_index=structure_index,
                **_data_load_flags(data),
            )
        model_state = (
            existing
            if existing is not None and data is existing.data
            else ModelState(model.rank, data, structure_index)
        )
        prepared.append(
            PreparedEnsembleMember(
                model_label=model.display_label,
                obj_name=model.object_name,
                model_state=model_state,
            )
        )

    report("Validating ensemble token maps…")
    validate_prepared_members(prepared)
    reference = next((member for member in prepared if member.rank == 0), prepared[0])
    reference_plddt = _member_plddt(reference)
    core_indices: tuple[int, ...] = ()
    if not skip_alignment:
        core_indices = tuple(
            select_alignment_core(reference.token_map, reference_plddt)
        )
        if len(core_indices) < 3:
            raise ValueError(
                "Automatic ensemble alignment requires at least 3 polymer tokens."
            )

    plddt_mean, plddt_std = compute_metric_consensus(
        [_member_plddt(member) for member in prepared]
    )
    return PreparedEnsemble(
        pred_files=pred_files,
        group_name=default_group_name(pred_files),
        members=tuple(prepared),
        skip_alignment=skip_alignment,
        reference_rank=reference.rank,
        core_indices=core_indices,
        plddt_mean=plddt_mean,
        plddt_std=plddt_std,
    )


def _member_plddt(member: PreparedEnsembleMember) -> np.ndarray:
    plddt, _source = plddt_values_for(member.data)
    if plddt is None:
        raise ValueError(f"pLDDT data are not available for model_{member.rank}.")
    return plddt


def select_alignment_core(
    token_map: TokenMap,
    plddt: np.ndarray,
    threshold: float = 0.8,
    min_tokens: int = 3,
) -> list[int]:
    """Return polymer token indices to use as the ensemble alignment core.

    The default core is polymer tokens with pLDDT >= *threshold*. If that set
    is too small for a stable fit, all polymer tokens are returned instead.
    """
    if len(plddt) != len(token_map):
        raise ValueError(
            f"pLDDT length {len(plddt)} does not match token map length "
            f"{len(token_map)}."
        )

    polymer = [tok.token_idx for tok in token_map if not tok.is_hetatm]
    core = [
        idx for idx in polymer if np.isfinite(plddt[idx]) and plddt[idx] >= threshold
    ]
    return core if len(core) >= min_tokens else polymer


def kabsch_transform(
    mobile_coords: np.ndarray,
    target_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, t)`` so ``mobile @ R.T + t`` best fits *target_coords*."""
    if mobile_coords.shape != target_coords.shape:
        raise ValueError(
            f"Coordinate shape mismatch: {mobile_coords.shape} vs {target_coords.shape}."
        )
    if mobile_coords.ndim != 2 or mobile_coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape (N, 3).")
    if mobile_coords.shape[0] < 3:
        raise ValueError("At least 3 coordinates are required for Kabsch alignment.")

    mobile = np.asarray(mobile_coords, dtype=np.float64)
    target = np.asarray(target_coords, dtype=np.float64)
    mobile_centroid = mobile.mean(axis=0)
    target_centroid = target.mean(axis=0)
    mobile_centered = mobile - mobile_centroid
    target_centered = target - target_centroid

    covariance = mobile_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - mobile_centroid @ rotation.T
    return rotation.astype(np.float64), translation.astype(np.float64)


def invert_rigid_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the inverse of ``coords @ rotation.T + translation``."""
    inverse_rotation = np.asarray(rotation, dtype=np.float64).T
    inverse_translation = -np.asarray(translation, dtype=np.float64) @ np.asarray(
        rotation, dtype=np.float64
    )
    return inverse_rotation, inverse_translation


def calculate_alignment_plan(
    members: Sequence[PreparedEnsembleMember],
    coordinates_by_rank: Mapping[int, np.ndarray],
    *,
    reference_rank: int,
    core_indices: Sequence[int],
) -> AlignmentPlan:
    """Calculate transforms and RMSD from coordinates already in the viewer."""
    if len(core_indices) < 3:
        raise ValueError("At least 3 polymer tokens are required for alignment.")
    if reference_rank not in coordinates_by_rank:
        raise ValueError(f"Reference rank {reference_rank} is not loaded.")

    target_all = np.asarray(coordinates_by_rank[reference_rank], dtype=np.float32)
    target_core = target_all[list(core_indices)]
    transformed: list[np.ndarray] = []
    transforms: list[AlignmentTransform] = []
    for member in members:
        current = np.asarray(coordinates_by_rank[member.rank], dtype=np.float32)
        if current.shape != target_all.shape:
            raise ValueError(
                f"Coordinate shape mismatch for model_{member.rank}: "
                f"{current.shape} vs {target_all.shape}."
            )
        if member.rank == reference_rank:
            transformed.append(current)
            continue
        rotation, translation = kabsch_transform(
            current[list(core_indices)],
            target_core,
        )
        transforms.append(AlignmentTransform(member.rank, rotation, translation))
        transformed.append((current @ rotation.T + translation).astype(np.float32))

    transformed_tuple = tuple(transformed)
    return AlignmentPlan(
        transforms=tuple(transforms),
        transformed_coords=transformed_tuple,
        rmsd=compute_per_token_rmsd(list(transformed_tuple)),
    )


def compute_per_token_rmsd(
    coords_list: list[np.ndarray],
) -> np.ndarray:
    """Compute per-token RMSD across multiple coordinate sets.

    Each entry in *coords_list* is a ``(N_tokens, 3)`` array for one model.
    Returns shape ``(N_tokens,)`` with the per-token RMSD across all models,
    computed relative to the coordinate mean (no external alignment performed).

    Parameters
    ----------
    coords_list:
        Coordinate arrays for each diffusion sample, all with the same shape.
    """
    if not coords_list:
        raise ValueError("coords_list must not be empty.")
    if len(coords_list) == 1:
        return np.zeros(coords_list[0].shape[0], dtype=np.float32)

    stack = np.stack(coords_list, axis=0)  # (n_models, N_tokens, 3)
    mean = stack.mean(axis=0, keepdims=True)  # (1, N_tokens, 3)
    sq_diff = ((stack - mean) ** 2).sum(axis=2)  # (n_models, N_tokens)
    rmsd = np.sqrt(sq_diff.mean(axis=0))  # (N_tokens,)
    return rmsd.astype(np.float32)


def compute_metric_consensus(
    metric_arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation of a per-token metric across samples.

    Parameters
    ----------
    metric_arrays:
        List of per-token arrays, one per diffusion sample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mean, std)`` arrays, both shape ``(N_tokens,)``.
    """
    if not metric_arrays:
        raise ValueError("metric_arrays must not be empty.")
    stack = np.stack(metric_arrays, axis=0)  # (n_models, N_tokens)
    return stack.mean(axis=0).astype(np.float32), stack.std(axis=0).astype(np.float32)


def find_high_plddt_high_rmsd_tokens(
    plddt: np.ndarray,
    rmsd: np.ndarray,
    plddt_threshold: float = 0.7,
    rmsd_threshold: float = 2.0,
) -> list[int]:
    """Return token indices that have high pLDDT but high inter-sample RMSD.

    This combination is a red flag: the model is locally confident but the
    diffusion process does not converge — a sign of genuine pose ambiguity.

    Parameters
    ----------
    plddt_threshold:
        Minimum pLDDT (0–1) to be considered "locally confident".
    rmsd_threshold:
        Minimum RMSD (Å) to be considered "geometrically diverse".
    """
    mask = (plddt >= plddt_threshold) & (rmsd >= rmsd_threshold)
    return list(np.where(mask)[0])
