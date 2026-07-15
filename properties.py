"""
Properties
==========
Collapse PAE/PDE matrices and other prediction outputs into per-token scalar arrays
that can be painted onto a structure through :mod:`mol_viewer`.

All functions return a 1-D ``float32`` numpy array of shape ``(N_tokens,)``
unless otherwise noted.  Values of ``np.nan`` indicate tokens for which the
metric is undefined (e.g. no predicted contact within the cutoff distance).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .confidence import PredictionConfidence
from .token_map import TokenMap

# ---------------------------------------------------------------------------
# pLDDT
# ---------------------------------------------------------------------------


def plddt_values(plddt: np.ndarray) -> np.ndarray:
    """Return per-token pLDDT (0–1 range) as-is."""
    return plddt.astype(np.float32)


# ---------------------------------------------------------------------------
# PAE aggregations
# ---------------------------------------------------------------------------

_SPECTRAL_MAX_CLUSTERS = 12


def pae_row_mean(pae: np.ndarray) -> np.ndarray:
    """Mean PAE over each row: ``mean(PAE[i, :])`` for all ``i``.

    Interpretation: when aligned on token *i*, average uncertainty about
    all other tokens.  High values mark poorly anchored / disordered regions.
    """
    return pae.mean(axis=1).astype(np.float32)


def pae_col_mean(pae: np.ndarray) -> np.ndarray:
    """Mean PAE over each column: ``mean(PAE[:, j])`` for all ``j``.

    Interpretation: average positional uncertainty of token *j* regardless of
    the alignment frame.
    """
    return pae.mean(axis=0).astype(np.float32)


def _chain_index_groups(
    matrix: np.ndarray, token_map: TokenMap
) -> tuple[np.ndarray, Mapping[str, tuple[int, ...]]]:
    """Validate a token matrix and return token indices grouped by chain."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square.")
    n = arr.shape[0]
    if len(token_map) != n:
        raise ValueError(
            f"token_map length {len(token_map)} does not match matrix size {n}."
        )

    return arr, token_map.chain_to_indices


def pae_chain_summary(
    pae: np.ndarray,
    token_map: TokenMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-token PAE row/column means within and outside each chain.

    For token ``i`` the returned arrays are:
    ``mean(PAE[i, same_chain])``, ``mean(PAE[i, other_chains])``,
    ``mean(PAE[same_chain, i])``, and ``mean(PAE[other_chains, i])``.
    """
    arr, chain_to_indices = _chain_index_groups(pae, token_map)
    n = arr.shape[0]
    all_indices = np.arange(n, dtype=int)
    row_within = np.full(n, np.nan, dtype=np.float32)
    row_other = np.full(n, np.nan, dtype=np.float32)
    col_within = np.full(n, np.nan, dtype=np.float32)
    col_other = np.full(n, np.nan, dtype=np.float32)

    for tok in token_map:
        idx = int(tok.token_idx)
        same = np.array(chain_to_indices[str(tok.chain_id)], dtype=int)
        other = np.setdiff1d(all_indices, same, assume_unique=True)
        row_within[idx] = arr[idx, same].mean()
        col_within[idx] = arr[same, idx].mean()
        if other.size:
            row_other[idx] = arr[idx, other].mean()
            col_other[idx] = arr[other, idx].mean()

    return row_within, row_other, col_within, col_other


def pae_to_selection(
    pae: np.ndarray,
    ref_indices: list[int],
) -> np.ndarray:
    """Row-mean PAE restricted to a reference set of token indices.

    For each token ``i``: ``mean(PAE[i, ref_indices])``.

    Typical use: colour the protein by how confidently each residue is
    positioned relative to the ligand (set *ref_indices* to ligand tokens).
    Low values → residue is part of the confident binding pocket.
    """
    if not ref_indices:
        raise ValueError("ref_indices must not be empty.")
    ref = np.array(ref_indices, dtype=int)
    return pae[:, ref].mean(axis=1).astype(np.float32)


def pae_column_to_selection(
    pae: np.ndarray,
    ref_indices: list[int],
) -> np.ndarray:
    """Column-mean PAE restricted to a reference set of token indices.

    For each token ``j``: ``mean(PAE[ref_indices, j])``.

    This captures the opposite direction from :func:`pae_to_selection`, which is
    useful for asymmetric PAE matrices such as ligand-complex AF3 outputs.
    """
    if not ref_indices:
        raise ValueError("ref_indices must not be empty.")
    ref = np.array(ref_indices, dtype=int)
    return pae[ref, :].mean(axis=0).astype(np.float32)


def pae_symmetric_to_selection(
    pae: np.ndarray,
    ref_indices: list[int],
) -> np.ndarray:
    """Symmetric PAE to a reference selection.

    For non-reference token *i*:
        ``0.5 * (mean(PAE[i, ref]) + mean(PAE[ref, i]))``

    For reference token *j*:
        ``0.5 * (mean(PAE[j, non_ref]) + mean(PAE[non_ref, j]))``

    Merges both alignment directions to reduce asymmetry artefacts.
    """
    if not ref_indices:
        raise ValueError("ref_indices must not be empty.")

    n = pae.shape[0]
    ref = np.array(ref_indices, dtype=int)
    non_ref = np.setdiff1d(np.arange(n, dtype=int), ref, assume_unique=True)

    out = np.full(n, np.nan, dtype=np.float32)

    if non_ref.size:
        # Non-reference tokens: PAE relative to reference
        forward = pae[non_ref][:, ref].mean(axis=1)  # PAE[i, ref]
        backward = pae[ref][:, non_ref].mean(axis=0)  # PAE[ref, i] -> mean over ref
        out[non_ref] = 0.5 * (forward + backward)

    if ref.size:
        # Reference tokens: PAE relative to non-reference
        forward = pae[ref][:, non_ref].mean(axis=1)
        backward = pae[non_ref][:, ref].mean(axis=0)
        out[ref] = 0.5 * (forward + backward)

    return out


def pae_symmetric_mean_within_selection(
    pae: np.ndarray,
    ref_indices: list[int],
) -> np.ndarray:
    """Symmetric PAE within a selected set of token indices.

    For each selected token ``i``: ``mean(sym_pae[i, ref_indices])``, where
    ``sym_pae = 0.5 * (PAE + PAE.T)``. Tokens outside the selection are returned
    as ``np.nan`` so they are ignored by statistics and painting ranges.
    """
    if not ref_indices:
        raise ValueError("ref_indices must not be empty.")

    ref = np.array(ref_indices, dtype=int)
    sym_pae = 0.5 * (pae + pae.T)
    out = np.full(pae.shape[0], np.nan, dtype=np.float32)
    out[ref] = sym_pae[np.ix_(ref, ref)].mean(axis=1)
    return out


def pae_symmetric_to_selection_for_contacts(
    pae: np.ndarray,
    ref_indices: list[int],
    contact_indices: list[int],
) -> np.ndarray:
    """Symmetric mean PAE to reference, defined only for contact tokens.

    ``contact_indices`` is supplied by the caller so viewer-dependent contact
    shell construction stays outside this pure numeric module.
    """
    if not ref_indices:
        raise ValueError("ref_indices must not be empty.")

    n = pae.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)
    if not contact_indices:
        return out

    ref = np.array(ref_indices, dtype=int)
    contact = np.array(sorted(set(contact_indices)), dtype=int)
    forward = pae[contact][:, ref]
    backward = pae[ref][:, contact].T
    out[contact] = (0.5 * (forward + backward)).mean(axis=1).astype(np.float32)
    return out


def _pae_continuous_affinity(sym_pae: np.ndarray, threshold: float) -> np.ndarray:
    """Return a continuous affinity matrix from symmetric PAE distances."""
    sigma = max(float(threshold), 1.0e-6)
    distances = np.asarray(sym_pae, dtype=np.float64)
    affinity = np.exp(-((distances / sigma) ** 2))
    affinity[~np.isfinite(affinity)] = 0.0
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _spectral_cluster_count_from_eigengap(
    affinity: np.ndarray,
    max_clusters: int = _SPECTRAL_MAX_CLUSTERS,
) -> int:
    """Estimate spectral cluster count from the normalized-Laplacian eigengap."""
    n = affinity.shape[0]
    if n <= 1:
        return 1

    max_k = min(max(2, int(max_clusters)), n - 1)
    if max_k < 2:
        return 1

    from scipy.linalg import eigvalsh
    from scipy.sparse.csgraph import laplacian

    lap = laplacian(affinity, normed=True)
    eigenvalues = eigvalsh(lap, subset_by_index=[0, max_k])
    gaps = np.diff(eigenvalues)
    if gaps.size <= 1:
        return 1

    return int(np.argmax(gaps[1:]) + 2)


def pae_domain_labels(
    pae: np.ndarray,
    threshold: float = 5.0,
    method: str = "complete_linkage",
) -> np.ndarray:
    """Assign integer domain labels based on symmetric PAE.

    Complete linkage treats symmetric PAE as a distance and cuts the hierarchy
    at *threshold*, so every pair of tokens in a cluster must remain within the
    threshold.  Returns shape ``(N_tokens,)`` integer array.

    Parameters
    ----------
    threshold:
        PAE Å cutoff for "confident co-positioning".
    method:
        ``"complete_linkage"`` (scipy, strict mutual-domain clustering),
        ``"connected_components"`` (scipy, transitive graph clustering), or
        ``"spectral"`` (scikit-learn, finds soft clusters).
    """
    sym_pae = 0.5 * (pae + pae.T)

    if method == "complete_linkage":
        n = sym_pae.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float32)
        if n == 1:
            return np.zeros(1, dtype=np.float32)

        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        distances = np.asarray(sym_pae, dtype=np.float64).copy()
        finite = np.isfinite(distances)
        if not finite.all():
            finite_values = distances[finite]
            fill_value = (
                max(float(finite_values.max()), float(threshold))
                if finite_values.size
                else float(threshold)
            )
            distances[~finite] = fill_value
        np.fill_diagonal(distances, 0.0)
        condensed = squareform(distances, checks=False)
        tree = linkage(condensed, method="complete")
        labels = fcluster(tree, t=float(threshold), criterion="distance") - 1
        return labels.astype(np.float32)

    elif method == "connected_components":
        adjacency = (sym_pae < threshold).astype(np.float32)
        np.fill_diagonal(adjacency, 0.0)

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        _, labels = connected_components(csr_matrix(adjacency), directed=False)
        return labels.astype(np.float32)

    elif method == "spectral":
        from sklearn.cluster import SpectralClustering

        affinity = _pae_continuous_affinity(sym_pae, threshold)
        n_clusters = _spectral_cluster_count_from_eigengap(affinity)
        sc = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=42
        )
        labels = sc.fit_predict(affinity)
        return labels.astype(np.float32)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'complete_linkage', "
            "'connected_components', or 'spectral'."
        )


# ---------------------------------------------------------------------------
# PDE aggregations
# ---------------------------------------------------------------------------


def pde_mean(pde: np.ndarray) -> np.ndarray:
    """Mean PDE over each row: ``mean(PDE[i, :])`` for all ``i``.

    Interpretation: average distance-uncertainty of token *i* relative to
    all other tokens.
    """
    return pde.mean(axis=1).astype(np.float32)


def pde_mean_within_chain(
    pde: np.ndarray,
    token_map: TokenMap,
) -> np.ndarray:
    """Mean PDE for each token against tokens from the same chain.

    For each token ``i``: ``mean(PDE[i, chain_indices])``, where
    ``chain_indices`` are all token indices with the same ``chain_id`` as
    token ``i``.
    """
    n = pde.shape[0]
    if len(token_map) != n:
        raise ValueError(
            f"token_map length {len(token_map)} does not match PDE size {n}."
        )

    out = np.full(n, np.nan, dtype=np.float32)
    for tok in token_map:
        ref = np.array(token_map.chain_to_indices[tok.chain_id], dtype=int)
        out[tok.token_idx] = pde[tok.token_idx, ref].mean()

    return out


def pde_chain_summary(
    pde: np.ndarray,
    token_map: TokenMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-token PDE means against same-chain and other-chain tokens."""
    arr, chain_to_indices = _chain_index_groups(pde, token_map)
    n = arr.shape[0]
    all_indices = np.arange(n, dtype=int)
    within = np.full(n, np.nan, dtype=np.float32)
    other = np.full(n, np.nan, dtype=np.float32)

    for tok in token_map:
        idx = int(tok.token_idx)
        same = np.array(chain_to_indices[str(tok.chain_id)], dtype=int)
        other_indices = np.setdiff1d(all_indices, same, assume_unique=True)
        within[idx] = arr[idx, same].mean()
        if other_indices.size:
            other[idx] = arr[idx, other_indices].mean()

    return within, other


def pde_to_selection(
    pde: np.ndarray,
    ref_indices: list[int],
) -> np.ndarray:
    """Mean PDE to a reference set: ``mean(PDE[i, ref_indices])`` for all ``i``.

    Frame-independent analogue of :func:`pae_to_selection`.
    """
    if not ref_indices:
        raise ValueError("ref_indices must not be empty.")
    ref = np.array(ref_indices, dtype=int)
    return pde[:, ref].mean(axis=1).astype(np.float32)


def pde_mean_within_selection(
    pde: np.ndarray,
    ref_indices: list[int],
) -> np.ndarray:
    """Mean PDE within a selected set of token indices.

    For each selected token ``i``: ``mean(PDE[i, ref_indices])``.
    Tokens outside the selection are returned as ``np.nan`` so they are ignored
    by statistics and painting range calculations.
    """
    if not ref_indices:
        raise ValueError("ref_indices must not be empty.")

    ref = np.array(ref_indices, dtype=int)
    out = np.full(pde.shape[0], np.nan, dtype=np.float32)
    out[ref] = pde[np.ix_(ref, ref)].mean(axis=1)
    return out


def pde_contact_filtered(
    pde: np.ndarray,
    coords: np.ndarray,
    ref_indices: list[int],
    distance_cutoff: float = 5.0,
) -> np.ndarray:
    """Mean PDE to reference, restricted to predicted contacts.

    For each token *i*: average ``PDE[i, j]`` only over reference tokens *j*
    whose Euclidean distance to *i* is ≤ *distance_cutoff* (Å).
    Tokens with no contacts within range receive ``np.nan``.

    Parameters
    ----------
    coords:
        Representative-atom coordinates, shape ``(N_tokens, 3)``.
    distance_cutoff:
        Distance threshold in Å for a "contact".
    """
    if not ref_indices:
        raise ValueError("ref_indices must not be empty.")

    ref = np.array(ref_indices, dtype=int)
    n = pde.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)

    ref_coords = coords[ref]  # (n_ref, 3)
    dists = np.linalg.norm(
        coords[:, None, :] - ref_coords[None, :, :],
        axis=2,
    )
    mask = dists <= distance_cutoff
    counts = mask.sum(axis=1)
    valid = counts > 0
    if valid.any():
        pde_ref = pde[:, ref]
        masked = np.where(mask[valid], pde_ref[valid], 0.0)
        out[valid] = (masked.sum(axis=1) / counts[valid]).astype(np.float32)

    return out


def pde_to_selection_for_contacts(
    pde: np.ndarray,
    ref_indices: list[int],
    contact_indices: list[int],
) -> np.ndarray:
    """Mean PDE to reference, defined only for explicit contact tokens.

    ``contact_indices`` is supplied by the caller, allowing viewer adapter code to
    define residue contacts with all-atom selections while this module remains
    viewer-independent. Tokens outside ``contact_indices`` receive ``np.nan``.
    """
    if not ref_indices:
        raise ValueError("ref_indices must not be empty.")

    n = pde.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)
    if not contact_indices:
        return out

    ref = np.array(ref_indices, dtype=int)
    contact = np.array(sorted(set(contact_indices)), dtype=int)
    out[contact] = pde[contact][:, ref].mean(axis=1).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Interaction/contact probability aggregations
# ---------------------------------------------------------------------------


def contact_probability_mean(contact_probs: np.ndarray) -> np.ndarray:
    """Mean predicted contact probability for each token."""
    return contact_probs.mean(axis=1).astype(np.float32)


def contact_probability_to_selection(
    contact_probs: np.ndarray,
    ref_indices: list[int],
) -> np.ndarray:
    """Mean predicted contact probability to a reference token set.

    Reference tokens receive ``np.nan`` because their within-selection contact
    probabilities are often trivial and can dominate the displayed value range.
    """
    if not ref_indices:
        raise ValueError("ref_indices must not be empty.")
    ref = np.array(ref_indices, dtype=int)
    out = contact_probs[:, ref].mean(axis=1).astype(np.float32)
    out[ref] = np.nan
    return out


# ---------------------------------------------------------------------------
# Chain-level metrics from typed confidence
# ---------------------------------------------------------------------------


def _chain_order_from_token_map(token_map: TokenMap) -> list[str]:
    """Return chain IDs in token-map order, collapsing contiguous runs."""
    return [chain_id or "(blank)" for chain_id in token_map.chain_order]


def pair_chain_iptm_matrix(
    confidence: PredictionConfidence, token_map: TokenMap
) -> tuple[np.ndarray, list[str]]:
    """Return the pairwise chain ipTM matrix and chain labels.

    Missing cells are represented as ``np.nan``. Chain order follows the
    token map, which mirrors the prediction output chain order used by the
    provider's 0-based chain keys. Typed confidence normalization has already
    filled missing or zero-valued diagonal cells from per-chain pTM when
    possible.
    """
    labels = _chain_order_from_token_map(token_map)
    if not labels:
        raise ValueError("Token map contains no chains.")
    matrix = confidence.pair_chain_iptm
    if matrix is None:
        raise ValueError("Confidence data do not contain pairwise chain ipTM.")
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Pairwise chain ipTM does not match the token-map chains.")
    return matrix, labels


def chain_iptm_values(
    confidence: PredictionConfidence,
    token_map: TokenMap,
    ref_chain_key: str | None = None,
) -> np.ndarray:
    """Assign each token its chain's ipTM score.

    Parameters
    ----------
    confidence:
        Canonical typed confidence values in token-map chain order.
    token_map:
        Built by :meth:`structure_index.StructureIndex.from_path`.
    ref_chain_key:
        Chain index key (string) for the reference chain (e.g. ``"1"`` for the
        ligand).  If *None*, use the per-chain pTM (``chains_ptm`` diagonal).
    """
    n = len(token_map)
    out = np.full(n, np.nan, dtype=np.float32)

    if ref_chain_key is not None:
        if confidence.pair_chain_iptm is None:
            return out
        try:
            ref_index = int(ref_chain_key)
        except (TypeError, ValueError):
            return out
        if ref_index < 0 or ref_index >= confidence.pair_chain_iptm.shape[1]:
            return out
        chain_scores = confidence.pair_chain_iptm[:, ref_index]
    else:
        chain_scores = confidence.chain_iptm
        if chain_scores is None:
            chain_scores = confidence.chain_ptm
        if chain_scores is None:
            return out

    for tok in token_map:
        idx = token_map.chain_id_to_index.get(tok.chain_id)
        if idx is not None and idx < len(chain_scores):
            out[tok.token_idx] = float(chain_scores[idx])

    return out
