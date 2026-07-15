"""Canonical normalization and validation for provider prediction data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .loader_models import PredictionData


class ProviderContractError(ValueError):
    """An advertised provider field was absent from its source file."""


def _numeric_array(value: object, field: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must contain numeric values; got {array.dtype}.")
    return array


def _readonly_contiguous(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _reject_infinity(array: np.ndarray, field: str) -> None:
    if np.isinf(array).any():
        raise ValueError(f"{field} must not contain positive or negative infinity.")


def _validate_vector(value: object, field: str, token_count: int) -> np.ndarray:
    array = _numeric_array(value, field)
    if array.shape != (token_count,):
        raise ValueError(
            f"{field} must have shape ({token_count},); got {array.shape}."
        )
    result = (
        array
        if array.dtype == np.float32
        and array.flags.c_contiguous
        and not array.flags.writeable
        else np.array(array, dtype=np.float32, order="C", copy=True)
    )
    _reject_infinity(result, field)
    return result


def _validate_matrix(value: object, field: str, token_count: int) -> np.ndarray:
    array = _numeric_array(value, field)
    expected = (token_count, token_count)
    if array.shape != expected:
        raise ValueError(f"{field} must have shape {expected}; got {array.shape}.")
    result = (
        array
        if array.dtype == np.float32
        and array.flags.c_contiguous
        and not array.flags.writeable
        else np.array(array, dtype=np.float32, order="C", copy=True)
    )
    _reject_infinity(result, field)
    return _readonly_contiguous(result)


def normalize_and_validate_prediction_data(
    data: PredictionData,
    token_count: int,
) -> PredictionData:
    """Normalize all loaded fields and enforce the canonical token contract."""
    token_plddt = getattr(data, "token_plddt", None)
    source = getattr(data, "token_plddt_source", None)
    if token_plddt is not None and not hasattr(data, "token_plddt_source"):
        source = "provider_token"
        data.token_plddt_source = source
    if (token_plddt is None) != (source is None):
        raise ValueError(
            "token_plddt and token_plddt_source must be provided together."
        )
    if token_plddt is not None:
        if source not in {
            "structure_b_factor",
            "provider_token",
            "provider_atom_mean",
        }:
            raise ValueError(f"Unknown token_plddt_source provenance: {source!r}.")
        values = _validate_vector(token_plddt, "token_plddt", token_count)
        percentage = np.isfinite(values) & (values > 1.5)
        if percentage.any():
            if not values.flags.writeable:
                values = values.copy()
            values[percentage] /= 100.0
        finite = values[np.isfinite(values)]
        if finite.size and ((finite < 0.0).any() or (finite > 1.0).any()):
            raise ValueError("Finite token_plddt values must normalize to 0–1.")
        data.token_plddt = _readonly_contiguous(values)

    for field in ("pae", "pde", "contact_probs"):
        value = getattr(data, field, None)
        if value is None:
            continue
        matrix = _validate_matrix(value, field, token_count)
        if field == "contact_probs":
            finite = matrix[np.isfinite(matrix)]
            if finite.size and ((finite < 0.0).any() or (finite > 1.0).any()):
                raise ValueError("Finite contact_probs values must be within 0–1.")
        setattr(data, field, matrix)

    embeddings_s = getattr(data, "embeddings_s", None)
    embeddings_z = getattr(data, "embeddings_z", None)
    if (embeddings_s is None) != (embeddings_z is None):
        raise ValueError("embeddings_s and embeddings_z must be provided together.")
    if embeddings_s is not None and embeddings_z is not None:
        s = _numeric_array(embeddings_s, "embeddings_s")
        z = _numeric_array(embeddings_z, "embeddings_z")
        if s.ndim < 1 or s.shape[0] != token_count:
            raise ValueError(
                f"embeddings_s shape must begin with ({token_count},); got {s.shape}."
            )
        if z.ndim < 2 or z.shape[:2] != (token_count, token_count):
            raise ValueError(
                "embeddings_z shape must begin with "
                f"({token_count}, {token_count}); got {z.shape}."
            )
        _reject_infinity(s, "embeddings_s")
        _reject_infinity(z, "embeddings_z")
        data.embeddings_s = _readonly_contiguous(s)
        data.embeddings_z = _readonly_contiguous(z)

    for field in ("confidence", "summary_confidence", "affinity"):
        value = getattr(data, field, None)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{field} must be a dictionary or None.")
    return data


def require_advertised_fields(
    data: PredictionData,
    *,
    provider: str,
    model_label: str,
    requested: tuple[tuple[str, bool, Path | None], ...],
) -> None:
    """Raise a contextual error when requested advertised data is absent."""
    for field, advertised, source in requested:
        if advertised and getattr(data, field) is None:
            source_label = str(source) if source is not None else "<unknown source>"
            raise ProviderContractError(
                f"Provider {provider!r} advertised {field!r} for {model_label!r}, "
                f"but {source_label} did not provide it."
            )
