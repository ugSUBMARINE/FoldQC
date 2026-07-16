"""Typed provider-neutral confidence data and presentation schemas."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np

from .provider_errors import ProviderContractError


@dataclass(frozen=True)
class AffinityConfidence:
    predicted_value: float | None = None
    probability: float | None = None

    def __post_init__(self) -> None:
        for attribute in ("predicted_value", "probability"):
            value = getattr(self, attribute)
            if value is None:
                continue
            normalized = _domain_finite_float(value, f"affinity.{attribute}")
            object.__setattr__(self, attribute, normalized)


@dataclass(frozen=True, eq=False)
class PredictionConfidence:
    ranking_score: float | None = None
    confidence_score: float | None = None
    ptm: float | None = None
    iptm: float | None = None
    ligand_iptm: float | None = None
    protein_iptm: float | None = None
    complex_plddt: float | None = None
    complex_iplddt: float | None = None
    complex_pde: float | None = None
    complex_ipde: float | None = None
    fraction_disordered: float | None = None
    gpde: float | None = None
    has_clash: bool | None = None
    chain_ptm: np.ndarray | None = None
    chain_iptm: np.ndarray | None = None
    pair_chain_iptm: np.ndarray | None = None
    affinity: AffinityConfidence | None = None

    def __post_init__(self) -> None:
        """Validate directly constructed values and make chain arrays immutable."""
        scalar_attributes = tuple(_SCALAR_ALIASES)
        for attribute in scalar_attributes:
            value = getattr(self, attribute)
            if value is None:
                continue
            object.__setattr__(self, attribute, _domain_finite_float(value, attribute))
        if self.has_clash is not None and not isinstance(
            self.has_clash, (bool, np.bool_)
        ):
            raise ValueError("has_clash must be boolean or None.")
        if self.has_clash is not None:
            object.__setattr__(self, "has_clash", bool(self.has_clash))
        if self.affinity is not None and not isinstance(
            self.affinity, AffinityConfidence
        ):
            raise ValueError("affinity must be AffinityConfidence or None.")
        dimensions = {"chain_ptm": 1, "chain_iptm": 1, "pair_chain_iptm": 2}
        for attribute, ndim in dimensions.items():
            value = getattr(self, attribute)
            if value is None:
                continue
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.number) or array.ndim != ndim:
                raise ValueError(f"{attribute} must be a numeric {ndim}-D array.")
            if np.isinf(array).any():
                raise ValueError(f"{attribute} must not contain infinity.")
            normalized = np.ascontiguousarray(array, dtype=np.float32)
            normalized.setflags(write=False)
            object.__setattr__(self, attribute, normalized)
        vector_lengths = {
            len(value)
            for value in (self.chain_ptm, self.chain_iptm)
            if value is not None
        }
        if len(vector_lengths) > 1:
            raise ValueError("chain_ptm and chain_iptm must have the same length.")
        if self.pair_chain_iptm is not None:
            rows, columns = self.pair_chain_iptm.shape
            if rows != columns:
                raise ValueError("pair_chain_iptm must be a square matrix.")
            if vector_lengths and rows not in vector_lengths:
                raise ValueError(
                    "pair_chain_iptm dimensions must match the chain-vector length."
                )
        if self.pair_chain_iptm is not None and self.chain_ptm is not None:
            matrix = self.pair_chain_iptm.copy()
            changed = False
            for index, replacement in enumerate(self.chain_ptm):
                if np.isfinite(replacement) and (
                    np.isnan(matrix[index, index]) or matrix[index, index] == 0.0
                ):
                    matrix[index, index] = replacement
                    changed = True
            if changed:
                matrix.setflags(write=False)
                object.__setattr__(self, "pair_chain_iptm", matrix)

    @property
    def has_chain_iptm(self) -> bool:
        return any(
            value is not None and bool(np.isfinite(value).any())
            for value in (self.chain_iptm, self.chain_ptm, self.pair_chain_iptm)
        )

    @property
    def has_values(self) -> bool:
        return any(getattr(self, item.name) is not None for item in fields(self))


ConfidenceValueSource = Literal["confidence", "affinity"]


@dataclass(frozen=True)
class ConfidenceFieldSpec:
    attribute: str
    label: str
    precision: int = 4
    suffix: str = ""
    source: ConfidenceValueSource = "confidence"
    omit_when_missing: bool = False
    include_in_model_comparison: bool = True

    def __post_init__(self) -> None:
        if self.precision < 0:
            raise ValueError("Confidence field precision must not be negative.")
        if self.source == "confidence":
            allowed = {
                item.name
                for item in fields(PredictionConfidence)
                if item.name
                not in {"chain_ptm", "chain_iptm", "pair_chain_iptm", "affinity"}
            }
        elif self.source == "affinity":
            allowed = {item.name for item in fields(AffinityConfidence)}
        else:
            raise ValueError(f"Unknown confidence field source: {self.source!r}.")
        if self.attribute not in allowed:
            raise ValueError(
                f"Unknown {self.source} presentation field: {self.attribute!r}."
            )


@dataclass(frozen=True)
class ConfidenceSectionSpec:
    attribute: Literal["chain_ptm", "chain_iptm"]
    label: str

    def __post_init__(self) -> None:
        if self.attribute not in {"chain_ptm", "chain_iptm"}:
            raise ValueError(f"Unknown confidence section field: {self.attribute!r}.")


@dataclass(frozen=True)
class ConfidenceSummarySpec:
    fields: tuple[ConfidenceFieldSpec, ...] = ()
    sections: tuple[ConfidenceSectionSpec, ...] = ()
    informational_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "sections", tuple(self.sections))
        if self.informational_text is not None and (self.fields or self.sections):
            raise ValueError(
                "Informational confidence schemas cannot also define data fields."
            )


COMMON_CONFIDENCE_SUMMARY = ConfidenceSummarySpec(
    fields=(
        ConfidenceFieldSpec("ranking_score", "ranking_score"),
        ConfidenceFieldSpec("ptm", "ptm"),
        ConfidenceFieldSpec("iptm", "iptm"),
        ConfidenceFieldSpec("fraction_disordered", "fraction_disord."),
        ConfidenceFieldSpec("has_clash", "has_clash"),
    ),
    sections=(
        ConfidenceSectionSpec("chain_ptm", "chain_ptm"),
        ConfidenceSectionSpec("chain_iptm", "chain_iptm"),
    ),
)

CHAI_CONFIDENCE_SUMMARY = ConfidenceSummarySpec(
    fields=tuple(
        field
        for field in COMMON_CONFIDENCE_SUMMARY.fields
        if field.attribute != "fraction_disordered"
    ),
    sections=COMMON_CONFIDENCE_SUMMARY.sections,
)

PROTENIX_CONFIDENCE_SUMMARY = ConfidenceSummarySpec(
    fields=COMMON_CONFIDENCE_SUMMARY.fields
    + (ConfidenceFieldSpec("gpde", "gpde", omit_when_missing=True),),
    sections=COMMON_CONFIDENCE_SUMMARY.sections,
)

BOLTZ_CONFIDENCE_SUMMARY = ConfidenceSummarySpec(
    fields=(
        ConfidenceFieldSpec("confidence_score", "confidence_score"),
        ConfidenceFieldSpec("ptm", "ptm"),
        ConfidenceFieldSpec("iptm", "iptm"),
        ConfidenceFieldSpec("ligand_iptm", "ligand_iptm"),
        ConfidenceFieldSpec("protein_iptm", "protein_iptm"),
        ConfidenceFieldSpec("complex_plddt", "complex_plddt"),
        ConfidenceFieldSpec("complex_iplddt", "complex_iplddt"),
        ConfidenceFieldSpec("complex_pde", "complex_pde", suffix=" Å"),
        ConfidenceFieldSpec("complex_ipde", "complex_ipde", suffix=" Å"),
        ConfidenceFieldSpec(
            "predicted_value",
            "affinity_pred_value",
            precision=3,
            suffix="  (log₁₀[IC₅₀/μM])",
            source="affinity",
            omit_when_missing=True,
            include_in_model_comparison=False,
        ),
        ConfidenceFieldSpec(
            "probability",
            "affinity_probability",
            source="affinity",
            omit_when_missing=True,
            include_in_model_comparison=False,
        ),
    ),
    sections=(
        ConfidenceSectionSpec("chain_ptm", "chains_ptm"),
        ConfidenceSectionSpec("chain_iptm", "chains_iptm"),
    ),
)

STRUCTURE_CONFIDENCE_SUMMARY = ConfidenceSummarySpec(
    informational_text="Structure-only input: pLDDT read from B-factors."
)


_SCALAR_ALIASES: dict[str, tuple[str, ...]] = {
    "ranking_score": ("ranking_score", "aggregate_score"),
    "confidence_score": ("confidence_score", "structure_confidence"),
    "ptm": ("ptm",),
    "iptm": ("iptm",),
    "ligand_iptm": ("ligand_iptm",),
    "protein_iptm": ("protein_iptm",),
    "complex_plddt": ("complex_plddt",),
    "complex_iplddt": ("complex_iplddt",),
    "complex_pde": ("complex_pde",),
    "complex_ipde": ("complex_ipde",),
    "fraction_disordered": ("fraction_disordered", "disorder"),
    "gpde": ("gpde",),
}


def parse_prediction_confidence(
    payload: dict | None,
    *,
    chain_count: int,
    provider: str,
    model_label: str,
    source: Path | str | None,
    affinity_payload: dict | None = None,
) -> PredictionConfidence | None:
    """Convert transient provider JSON mappings into the canonical model."""
    context = _context(provider, model_label, source)
    if payload is not None and not isinstance(payload, dict):
        raise ProviderContractError(f"{context}: confidence data must be an object.")
    if affinity_payload is not None and not isinstance(affinity_payload, dict):
        raise ProviderContractError(f"{context}: affinity data must be an object.")
    payload = payload or {}
    values: dict[str, object] = {}
    for attribute, aliases in _SCALAR_ALIASES.items():
        raw, present = _first_present(payload, aliases)
        if present:
            values[attribute] = _optional_finite_float(raw, attribute, context)

    raw_clash, has_clash = _first_present(
        payload, ("has_clash", "has_inter_chain_clashes")
    )
    if has_clash:
        if raw_clash is not None and not isinstance(raw_clash, (bool, np.bool_)):
            raise ProviderContractError(
                f"{context}: has_clash must be boolean or null."
            )
        values["has_clash"] = None if raw_clash is None else bool(raw_clash)

    chain_ptm, has_chain_ptm = _first_present(payload, ("chains_ptm", "chain_ptm"))
    if has_chain_ptm:
        values["chain_ptm"] = _chain_vector(
            chain_ptm, "chain_ptm", chain_count, context
        )

    chain_iptm, has_chain_iptm = _first_present(payload, ("chains_iptm", "chain_iptm"))
    if has_chain_iptm:
        values["chain_iptm"] = _chain_vector(
            chain_iptm, "chain_iptm", chain_count, context
        )

    pair, has_pair = _first_present(
        payload,
        ("pair_chains_iptm", "chain_pair_iptm"),
    )
    if has_pair:
        values["pair_chain_iptm"] = _chain_matrix(
            pair, "pair_chain_iptm", chain_count, context
        )

    if affinity_payload is not None:
        predicted, has_predicted = _first_present(
            affinity_payload, ("affinity_pred_value",)
        )
        probability, has_probability = _first_present(
            affinity_payload,
            ("affinity_probability_binary", "affinity_probability"),
        )
        affinity = AffinityConfidence(
            predicted_value=(
                _optional_finite_float(predicted, "affinity_pred_value", context)
                if has_predicted
                else None
            ),
            probability=(
                _optional_finite_float(probability, "affinity_probability", context)
                if has_probability
                else None
            ),
        )
        if affinity.predicted_value is not None or affinity.probability is not None:
            values["affinity"] = affinity

    result = PredictionConfidence(**values)
    return result if result.has_values else None


def parse_prediction_confidence_summary(
    payload: dict | None,
    *,
    provider: str,
    model_label: str,
    source: Path | str | None,
    affinity_payload: dict | None = None,
) -> PredictionConfidence | None:
    """Parse scalar summary fields without requiring a structure index.

    Model comparison is available immediately after discovery, before the
    corresponding structures have been indexed.  Chain vectors and matrices
    therefore remain part of the full per-model load; this helper deliberately
    retains only recognized scalar and affinity fields.
    """
    if payload is not None and not isinstance(payload, dict):
        context = _context(provider, model_label, source)
        raise ProviderContractError(f"{context}: confidence data must be an object.")
    scalar_keys = {
        alias for aliases in _SCALAR_ALIASES.values() for alias in aliases
    } | {"has_clash", "has_inter_chain_clashes"}
    scalar_payload = (
        None
        if payload is None
        else {key: value for key, value in payload.items() if key in scalar_keys}
    )
    return parse_prediction_confidence(
        scalar_payload,
        chain_count=0,
        provider=provider,
        model_label=model_label,
        source=source,
        affinity_payload=affinity_payload,
    )


def merge_prediction_confidence(
    current: PredictionConfidence | None,
    incoming: PredictionConfidence | None,
    *,
    context: str = "prediction confidence",
) -> PredictionConfidence | None:
    """Monotonically fill missing fields and reject conflicting values."""
    if current is None:
        return incoming
    if incoming is None:
        return current
    merged: dict[str, object] = {}
    changed = False
    for item in fields(PredictionConfidence):
        old = getattr(current, item.name)
        new = getattr(incoming, item.name)
        if item.name == "affinity":
            affinity = _merge_affinity(old, new, context)
            merged[item.name] = affinity
            changed = changed or affinity is not old
            continue
        if old is None:
            merged[item.name] = new
            changed = changed or new is not None
        elif new is None:
            merged[item.name] = old
        elif _confidence_values_equal(old, new):
            merged[item.name] = old
        else:
            raise ValueError(f"Conflicting {context} field: {item.name}.")
    return PredictionConfidence(**merged) if changed else current


def validate_prediction_confidence(
    confidence: PredictionConfidence | None, chain_count: int
) -> PredictionConfidence | None:
    if confidence is None:
        return None
    if not isinstance(confidence, PredictionConfidence):
        raise ValueError("confidence must be PredictionConfidence or None.")
    for name in ("chain_ptm", "chain_iptm"):
        value = getattr(confidence, name)
        if value is not None and value.shape != (chain_count,):
            raise ValueError(
                f"confidence.{name} must have shape ({chain_count},); got {value.shape}."
            )
    pair = confidence.pair_chain_iptm
    if pair is not None and pair.shape != (chain_count, chain_count):
        raise ValueError(
            "confidence.pair_chain_iptm must have shape "
            f"({chain_count}, {chain_count}); got {pair.shape}."
        )
    return confidence


def _first_present(mapping: dict, keys: tuple[str, ...]) -> tuple[object, bool]:
    for key in keys:
        if key in mapping:
            return mapping[key], True
    return None, False


def _optional_finite_float(value: object, field: str, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ProviderContractError(f"{context}: {field} must be numeric or null.")
    result = float(value)
    if not np.isfinite(result):
        raise ProviderContractError(f"{context}: {field} must be finite or null.")
    return result


def _chain_vector(value: object, field: str, size: int, context: str) -> np.ndarray:
    result = np.full(size, np.nan, dtype=np.float32)
    if isinstance(value, list):
        if len(value) != size:
            raise ProviderContractError(
                f"{context}: {field} must contain {size} chain values; got {len(value)}."
            )
        items = enumerate(value)
    elif isinstance(value, dict):
        items = (
            (_chain_index(key, size, field, context), item)
            for key, item in value.items()
        )
    else:
        raise ProviderContractError(
            f"{context}: {field} must be a list or index-keyed dictionary."
        )
    for index, item in items:
        if item is not None:
            result[index] = _optional_finite_float(item, field, context)
    return _readonly_float32(result)


def _chain_matrix(value: object, field: str, size: int, context: str) -> np.ndarray:
    result = np.full((size, size), np.nan, dtype=np.float32)
    if isinstance(value, list):
        if len(value) != size or any(
            not isinstance(row, list) or len(row) != size for row in value
        ):
            raise ProviderContractError(
                f"{context}: {field} must have shape ({size}, {size})."
            )
        rows = enumerate(value)
    elif isinstance(value, dict):
        rows = (
            (_chain_index(key, size, field, context), row) for key, row in value.items()
        )
    else:
        raise ProviderContractError(
            f"{context}: {field} must be a matrix list or nested dictionary."
        )
    for row_index, row in rows:
        if isinstance(row, list):
            if len(row) != size:
                raise ProviderContractError(
                    f"{context}: {field} row {row_index} must contain {size} values."
                )
            columns = enumerate(row)
        elif isinstance(row, dict):
            columns = (
                (_chain_index(key, size, field, context), item)
                for key, item in row.items()
            )
        else:
            raise ProviderContractError(
                f"{context}: {field} row {row_index} must be a list or dictionary."
            )
        for column_index, item in columns:
            if item is not None:
                result[row_index, column_index] = _optional_finite_float(
                    item, field, context
                )
    return _readonly_float32(result)


def _chain_index(key: object, size: int, field: str, context: str) -> int:
    if isinstance(key, bool) or (
        not isinstance(key, (int, np.integer))
        and not (isinstance(key, str) and key.isdigit())
    ):
        raise ProviderContractError(
            f"{context}: {field} chain key {key!r} is not an integer index."
        )
    try:
        index = int(key)
    except (TypeError, ValueError) as exc:
        raise ProviderContractError(
            f"{context}: {field} chain key {key!r} is not an integer index."
        ) from exc
    if index < 0 or index >= size:
        raise ProviderContractError(
            f"{context}: {field} chain index {index} is outside 0..{size - 1}."
        )
    return index


def _readonly_float32(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    result.setflags(write=False)
    return result


def _domain_finite_float(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be numeric or None.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field} must be finite or None.")
    return result


def _confidence_values_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.shape == right.shape and bool(
            np.allclose(left, right, rtol=1e-6, atol=1e-8, equal_nan=True)
        )
    if isinstance(left, AffinityConfidence) and isinstance(right, AffinityConfidence):
        return left == right
    if isinstance(left, float) or isinstance(right, float):
        try:
            return bool(np.isclose(float(left), float(right), rtol=1e-6, atol=1e-8))
        except (TypeError, ValueError):
            return False
    return left == right


def _merge_affinity(
    current: AffinityConfidence | None,
    incoming: AffinityConfidence | None,
    context: str,
) -> AffinityConfidence | None:
    if current is None:
        return incoming
    if incoming is None:
        return current
    values: dict[str, float | None] = {}
    changed = False
    for attribute in ("predicted_value", "probability"):
        old = getattr(current, attribute)
        new = getattr(incoming, attribute)
        if old is None:
            values[attribute] = new
            changed = changed or new is not None
        elif new is None or _confidence_values_equal(old, new):
            values[attribute] = old
        else:
            raise ValueError(f"Conflicting {context} field: affinity.{attribute}.")
    return AffinityConfidence(**values) if changed else current


def _context(provider: str, model_label: str, source: Path | str | None) -> str:
    source_label = str(source) if source is not None else "<unknown source>"
    return f"Provider {provider!r}, model {model_label!r}, source {source_label}"
