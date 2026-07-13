"""
Ensemble
========
Utilities for working with multiple ranked prediction models.

When a provider produces N models, N structure files can be loaded as one
object-based ensemble ordered by rank. This module provides tools to:
- Coordinate viewer-neutral ensemble loading through :mod:`mol_viewer`.
- Compute per-token RMSD across all samples.
- Compute per-metric consensus (mean ± std) across samples.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .mol_viewer import (
    ObjectPaintMapping,
    inspect_object_tokens,
    load_models_as_objects,
    rebuild,
    transform_object,
)

if TYPE_CHECKING:
    from .token_map import TokenMap


PhaseReporter = Callable[[str], None]


@dataclass
class EnsembleMember:
    """One loaded prediction model in an object-based ensemble."""

    rank: int
    obj_name: str
    data: Any
    token_map: TokenMap
    paint_mapping: ObjectPaintMapping | None = None


@dataclass(frozen=True)
class PreparedEnsembleMember:
    """One ensemble member prepared without accessing a molecular viewer."""

    rank: int
    model_label: str
    obj_name: str
    structure_path: Path
    data: Any
    token_map: TokenMap


@dataclass(frozen=True)
class PreparedEnsemble:
    """Atomically prepared ensemble data borrowed from one prediction."""

    pred_files: Any
    group_name: str
    members: tuple[PreparedEnsembleMember, ...]
    skip_alignment: bool
    reference_rank: int
    core_indices: tuple[int, ...]
    plddt_mean: np.ndarray
    plddt_std: np.ndarray
    _owns_prediction_files: bool = False


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


@dataclass(frozen=True)
class EnsembleMetrics:
    """Prepared ensemble-level metrics and display metadata."""

    aligned: bool
    rmsd: np.ndarray
    plddt_mean: np.ndarray
    plddt_std: np.ndarray
    mode_label: str


def default_group_name(pred_files) -> str:
    """Return the default viewer group name for a prediction ensemble."""
    first_obj = pred_files.models[0].object_name
    obj_prefix = first_obj.rsplit("_", 1)[0]
    return f"{obj_prefix}_ensemble"


def _ordered_token_identities(token_map: TokenMap) -> tuple[tuple, ...]:
    return tuple(
        (
            token.chain_id,
            token.res_num,
            token.res_name,
            token.atom_name if token.is_hetatm else None,
        )
        for token in token_map
    )


def validate_prepared_members(
    members: Sequence[PreparedEnsembleMember],
) -> None:
    """Validate token order and pLDDT lengths before touching the viewer."""
    validate_members(list(members))
    reference = members[0]
    reference_identities = _ordered_token_identities(reference.token_map)
    for member in members[1:]:
        if _ordered_token_identities(member.token_map) != reference_identities:
            raise ValueError(
                f"Token order mismatch for model_{member.rank}; ensemble models "
                "must contain the same ordered prediction tokens."
            )


def _data_load_flags(existing: Any | None) -> dict[str, bool]:
    """Return ensemble flags while preserving arrays already held by *existing*."""
    return {
        "load_pae": existing is not None and getattr(existing, "pae", None) is not None,
        "load_pde": existing is not None and getattr(existing, "pde", None) is not None,
        "load_contact_probs": existing is not None
        and getattr(existing, "contact_probs", None) is not None,
        "load_structure_plddt": True,
        "load_plddt": True,
    }


def _has_plddt(data: Any | None) -> bool:
    return data is not None and (
        getattr(data, "plddt", None) is not None
        or getattr(data, "structure_plddt", None) is not None
    )


def prepare_ensemble(
    pred_files: Any,
    *,
    skip_alignment: bool,
    existing_data_by_rank: Mapping[int, Any] | None = None,
    report_phase: PhaseReporter | None = None,
) -> PreparedEnsemble:
    """Load and validate ensemble data without accessing Qt or PyMOL."""
    from .loader import load_prediction_data
    from .token_map import build_token_map

    report = report_phase or (lambda _label: None)
    models = tuple(pred_files.models)
    if not models:
        raise ValueError("No ensemble models were found.")
    existing_data_by_rank = existing_data_by_rank or {}
    prepared: list[PreparedEnsembleMember] = []
    total = len(models)
    for index, model in enumerate(models, start=1):
        report(f"Preparing {model.display_label} ensemble data… ({index}/{total})")
        existing = existing_data_by_rank.get(model.rank)
        data = existing
        if not _has_plddt(existing):
            data = load_prediction_data(
                pred_files,
                model.rank,
                **_data_load_flags(existing),
            )
        token_map = build_token_map(data.structure_path)
        prepared.append(
            PreparedEnsembleMember(
                rank=model.rank,
                model_label=model.display_label,
                obj_name=model.object_name,
                structure_path=Path(data.structure_path),
                data=data,
                token_map=token_map,
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


def _member_plddt(member: Any) -> np.ndarray:
    data = member.data
    plddt = data.plddt if data.plddt is not None else data.structure_plddt
    if plddt is None:
        raise ValueError(f"pLDDT data are not available for model_{member.rank}.")
    return plddt


def build_members(
    pred_files,
    *,
    group_name: str | None = None,
) -> tuple[str, list[EnsembleMember]]:
    """Load/group all models and build per-object data plus token maps."""
    from .loader import load_prediction_data
    from .token_map import build_token_map

    first_obj = pred_files.models[0].object_name
    obj_prefix = first_obj.rsplit("_", 1)[0]
    group_name = group_name or default_group_name(pred_files)
    loaded = load_models_as_objects(
        [(m.rank, m.structure_path) for m in pred_files.models],
        obj_prefix=obj_prefix,
        group_name=group_name,
    )

    members = []
    reference_token_map = None
    for rank, obj_name in loaded:
        data = load_prediction_data(
            pred_files,
            rank,
            load_pae=False,
            load_pde=False,
            load_structure_plddt=True,
        )
        if reference_token_map is None:
            reference_token_map = build_token_map(data.structure_path)
            token_map = reference_token_map
        else:
            token_map = reference_token_map
        members.append(
            EnsembleMember(
                rank=rank,
                obj_name=obj_name,
                data=data,
                token_map=token_map,
            )
        )
    return group_name, members


def validate_members(members: list[EnsembleMember]) -> None:
    """Ensure all ensemble models have compatible token-indexed data."""
    if not members:
        raise ValueError("No ensemble models were loaded.")

    ref = members[0]
    ref_len = len(ref.token_map)
    for member in members:
        if len(member.token_map) != ref_len:
            raise ValueError(
                f"Token count mismatch: {member.obj_name} maps to "
                f"{len(member.token_map)} tokens, but {ref.obj_name} maps "
                f"to {ref_len} tokens."
            )
        if member.data.plddt is not None and len(member.data.plddt) != ref_len:
            raise ValueError(
                f"pLDDT length mismatch for model_{member.rank}: "
                f"{len(member.data.plddt)} values for {ref_len} tokens."
            )
        if (
            member.data.structure_plddt is not None
            and len(member.data.structure_plddt) != ref_len
        ):
            raise ValueError(
                f"Structure pLDDT length mismatch for model_{member.rank}: "
                f"{len(member.data.structure_plddt)} values for {ref_len} tokens."
            )


def prepare_metrics(
    members: list[EnsembleMember],
    *,
    skip_alignment: bool,
) -> EnsembleMetrics:
    """Prepare ensemble RMSD and pLDDT consensus arrays."""
    ref_member = next((m for m in members if m.rank == 0), members[0])
    aligned_coords = None
    if not skip_alignment:
        plddt = (
            ref_member.data.plddt
            if ref_member.data.plddt is not None
            else ref_member.data.structure_plddt
        )
        if plddt is None:
            raise ValueError("Automatic ensemble alignment requires pLDDT data.")
        core_indices = select_alignment_core(ref_member.token_map, plddt)
        aligned_coords = align_objects_to_reference(
            members, core_indices, reference_rank=ref_member.rank
        )

    rmsd = (
        compute_per_token_rmsd(aligned_coords)
        if aligned_coords is not None
        else compute_aligned_per_token_rmsd(members)
    )
    plddt_arrays = []
    for member in members:
        plddt = (
            member.data.plddt
            if member.data.plddt is not None
            else member.data.structure_plddt
        )
        if plddt is None:
            raise ValueError(f"pLDDT data are not available for model_{member.rank}.")
        plddt_arrays.append(plddt)
    plddt_mean, plddt_std = compute_metric_consensus(plddt_arrays)
    return EnsembleMetrics(
        aligned=not skip_alignment,
        rmsd=rmsd,
        plddt_mean=plddt_mean,
        plddt_std=plddt_std,
        mode_label="current coordinates"
        if skip_alignment
        else "automatic core alignment",
    )


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


def align_objects_to_reference(
    members: list[EnsembleMember],
    core_indices: list[int],
    reference_rank: int = 0,
) -> list[np.ndarray]:
    """Align every ensemble object to *reference_rank* using *core_indices*.

    The token maps for all members must be length-compatible, because the same
    token indices define matching residues in every object. Returns the
    transformed per-token coordinate arrays in member order.
    """
    if len(core_indices) < 3:
        raise ValueError("At least 3 polymer tokens are required for alignment.")

    ref = next((m for m in members if m.rank == reference_rank), None)
    if ref is None:
        raise ValueError(f"Reference rank {reference_rank} is not loaded.")

    target_inspection = inspect_object_tokens(ref.obj_name, ref.token_map)
    ref.paint_mapping = target_inspection.paint_mapping
    target_all_coords = target_inspection.representative_coords
    target_coords = target_all_coords[core_indices]
    aligned_coords: list[np.ndarray] = []
    for member in members:
        if member.rank == reference_rank:
            aligned_coords.append(target_all_coords)
            continue
        mobile_inspection = inspect_object_tokens(member.obj_name, member.token_map)
        member.paint_mapping = mobile_inspection.paint_mapping
        mobile_all_coords = mobile_inspection.representative_coords
        mobile_coords = mobile_all_coords[core_indices]
        rotation, translation = kabsch_transform(mobile_coords, target_coords)
        transform_object(member.obj_name, rotation, translation)
        aligned_coords.append(
            (mobile_all_coords @ rotation.T + translation).astype(np.float32)
        )

    rebuild()
    return aligned_coords


def compute_aligned_per_token_rmsd(
    members: list[EnsembleMember],
) -> np.ndarray:
    """Compute per-token RMSD from the current coordinates of ensemble objects."""
    coords_list = []
    for member in members:
        inspection = inspect_object_tokens(member.obj_name, member.token_map)
        member.paint_mapping = inspection.paint_mapping
        coords_list.append(inspection.representative_coords)
    return compute_per_token_rmsd(coords_list)


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
