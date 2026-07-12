"""
Ensemble
========
Utilities for working with multiple ranked prediction models.

When a provider produces N models, N structure files can be loaded as one
object-based ensemble ordered by rank. This module provides tools to:
- Load all models as separate PyMOL objects grouped as one ensemble.
- Compute per-token RMSD across all samples.
- Compute per-metric consensus (mean ± std) across samples.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EnsembleMember:
    """One loaded prediction model in an object-based ensemble."""

    rank: int
    obj_name: str
    data: Any
    token_map: list[Any]


@dataclass(frozen=True)
class EnsembleMetrics:
    """Prepared ensemble-level metrics and display metadata."""

    aligned: bool
    rmsd: np.ndarray
    plddt_mean: np.ndarray
    plddt_std: np.ndarray
    mode_label: str


def default_group_name(pred_files) -> str:
    """Return the default PyMOL group name for a prediction ensemble."""
    first_obj = pred_files.models[0].object_name
    obj_prefix = first_obj.rsplit("_", 1)[0]
    return f"{obj_prefix}_ensemble"


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
    loaded = load_as_objects(
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
            reference_token_map = build_token_map(obj_name, data.structure_path)
            token_map = reference_token_map
        else:
            token_map = clone_token_map_for_object(reference_token_map, obj_name)
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


def load_as_states(
    model_paths: list[tuple[int, Path]],
    obj_name: str = "foldqc_ensemble",
) -> str:
    """Load all models as consecutive states of one PyMOL object.

    Parameters
    ----------
    model_paths:
        List of ``(rank, path)`` pairs, e.g. from
        ``[(m.rank, m.structure_path) for m in pred_files.models]``.
    obj_name:
        Target PyMOL object name.

    Returns
    -------
    str
        The PyMOL object name (same as *obj_name*).
    """
    from pymol import cmd

    for state_idx, (rank, path) in enumerate(sorted(model_paths), start=1):
        cmd.load(str(path), obj_name, state=state_idx)

    return obj_name


def load_as_objects(
    model_paths: list[tuple[int, Path]],
    obj_prefix: str = "foldqc_model",
    group_name: str | None = None,
) -> list[tuple[int, str]]:
    """Load each model as a separate named PyMOL object and optionally group it.

    Existing objects with the target names are reused. Missing objects are
    loaded from *model_paths*.

    Returns a list of ``(rank, obj_name)`` pairs.
    """
    from pymol import cmd

    current_objects = set(cmd.get_names("objects") or [])
    result: list[tuple[int, str]] = []
    try:
        try:
            cmd.set("suspend_updates", "on")
        except Exception:
            pass
        for rank, path in sorted(model_paths):
            obj_name = f"{obj_prefix}_{rank}"
            if obj_name not in current_objects:
                cmd.load(str(path), obj_name, quiet=1, zoom=0)
                current_objects.add(obj_name)
            if group_name:
                cmd.group(group_name, obj_name, "add")
            result.append((rank, obj_name))
    finally:
        try:
            cmd.set("suspend_updates", "off")
            cmd.rebuild()
        except Exception:
            pass
    return result


def clone_token_map_for_object(token_map, obj_name: str) -> list[Any]:
    """Reuse a token topology with PyMOL selections targeting *obj_name*."""
    cloned = []
    for tok in token_map:
        if tok.is_hetatm and tok.atom_name is not None:
            sel = f"/{obj_name}//{tok.chain_id}/{tok.res_num}/{tok.atom_name}"
        else:
            sel = f"/{obj_name}//{tok.chain_id}/{tok.res_num}/"
        cloned.append(replace(tok, pymol_selection=sel))
    return cloned


def select_alignment_core(
    token_map,
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


def apply_transform_to_object(
    obj_name: str,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    """Apply a rigid transform to every atom in a PyMOL object."""
    from pymol import cmd, stored

    stored.foldqc_ensemble_rotation = rotation.tolist()
    stored.foldqc_ensemble_translation = translation.tolist()
    expr = (
        "x, y, z = ("
        "stored.foldqc_ensemble_rotation[0][0]*x + "
        "stored.foldqc_ensemble_rotation[0][1]*y + "
        "stored.foldqc_ensemble_rotation[0][2]*z + "
        "stored.foldqc_ensemble_translation[0], "
        "stored.foldqc_ensemble_rotation[1][0]*x + "
        "stored.foldqc_ensemble_rotation[1][1]*y + "
        "stored.foldqc_ensemble_rotation[1][2]*z + "
        "stored.foldqc_ensemble_translation[1], "
        "stored.foldqc_ensemble_rotation[2][0]*x + "
        "stored.foldqc_ensemble_rotation[2][1]*y + "
        "stored.foldqc_ensemble_rotation[2][2]*z + "
        "stored.foldqc_ensemble_translation[2])"
    )
    cmd.alter_state(1, obj_name, expr)


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

    target_all_coords = get_token_coords(ref.obj_name, ref.token_map)
    target_coords = target_all_coords[core_indices]
    aligned_coords: list[np.ndarray] = []
    for member in members:
        if member.rank == reference_rank:
            aligned_coords.append(target_all_coords)
            continue
        mobile_all_coords = get_token_coords(member.obj_name, member.token_map)
        mobile_coords = mobile_all_coords[core_indices]
        rotation, translation = kabsch_transform(mobile_coords, target_coords)
        apply_transform_to_object(member.obj_name, rotation, translation)
        aligned_coords.append(
            (mobile_all_coords @ rotation.T + translation).astype(np.float32)
        )

    from pymol import cmd

    cmd.rebuild()
    return aligned_coords


def compute_aligned_per_token_rmsd(
    members: list[EnsembleMember],
) -> np.ndarray:
    """Compute per-token RMSD from the current coordinates of ensemble objects."""
    coords_list = [get_token_coords(m.obj_name, m.token_map) for m in members]
    return compute_per_token_rmsd(coords_list)


def get_token_coords(
    obj_name: str,
    token_map,  # list[TokenInfo]
) -> np.ndarray:
    """Return representative-atom coordinates for a loaded object.

    Thin wrapper around :func:`painter.get_representative_coords`.
    Shape: ``(N_tokens, 3)``.
    """
    from .painter import get_representative_coords

    return get_representative_coords(obj_name, token_map)


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
