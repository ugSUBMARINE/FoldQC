"""Published protein-interface calculations from canonical immutable inputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import replace
from itertools import combinations

import numpy as np

from . import metrics
from . import properties as P
from .confidence import PredictionConfidence
from .interface_analysis import (
    InterfaceAggregate,
    InterfaceAnalysisOptions,
    InterfaceContactDiagnostics,
    InterfaceModelResult,
    InterfaceScoreResult,
)
from .structure_index import StructureIndex, protein_token_indices


def interface_metric_specs() -> tuple[metrics.MetricSpec, ...]:
    return tuple(spec for spec in metrics.METRICS if spec.result_scope != "token")


_IMPLEMENTED = frozenset(
    {
        "interface_ipsae",
        "interface_lis_kim",
        "interface_lis_dunbrack",
        "interface_clis",
        "interface_ilis",
        "interface_pdockq",
        "interface_pdockq2_dunbrack",
        "interface_pdockq2_zhu",
        "interface_iptm",
    }
)


def validate_interface_dispatch() -> None:
    if {spec.key for spec in interface_metric_specs()} != _IMPLEMENTED:
        raise ValueError("Interface registry and numerical dispatch disagree.")


validate_interface_dispatch()


def _support(
    indices: tuple[int, ...], mask: np.ndarray, *, axis: int
) -> tuple[int, ...]:
    return tuple(
        index
        for index, present in zip(
            indices, np.asarray(mask.any(axis=axis)).reshape(-1), strict=True
        )
        if present
    )


def _transform_mean(pae: np.ndarray, contact: np.ndarray) -> float:
    return (
        float((1.0 / (1.0 + (pae[contact] / 10.0) ** 2)).mean())
        if contact.any()
        else 0.0
    )


def _plddt_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size and np.isfinite(values).all() else None


def _finite_pae(pae: np.ndarray | None) -> bool:
    return bool(pae is not None and np.isfinite(pae).all() and (pae >= 0).all())


def iter_interface_contacts(
    index: StructureIndex,
    diagnostic: InterfaceContactDiagnostics,
    check_cancel: Callable[[], None],
) -> Iterator[tuple[int, int, float]]:
    """Canonical endpoint indices and distances; no pairs retained in reports."""
    geometry = index.geometry
    if geometry is None or diagnostic.pair_count is None:
        raise ValueError("Contact geometry is unavailable.")
    if (
        diagnostic.cutoff_angstrom != P.INTERFACE_CONTACT_CUTOFF
        or diagnostic.comparison != "<="
    ):
        raise ValueError("Unsupported contact definition.")
    tm = index.token_map
    eligible = protein_token_indices(tm, geometry)
    ia = tuple(i for i in eligible if tm[i].chain_id == diagnostic.chain_a)
    ib = tuple(i for i in eligible if tm[i].chain_id in diagnostic.partners)
    coords = geometry.ca if diagnostic.convention == "ca" else geometry.cb
    first, second = coords[list(ia)], coords[list(ib)]
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Required representative-atom coordinates are unavailable.")
    count = 0
    support_a: set[int] = set()
    support_b: set[int] = set()
    for start, mask, squared in P.interface_contact_tiles(first, second, check_cancel):
        rows, columns = np.nonzero(mask)
        for row, column in zip(rows, columns, strict=True):
            a, b = ia[start + int(row)], ib[int(column)]
            count += 1
            support_a.add(a)
            support_b.add(b)
            yield a, b, float(np.sqrt(squared[row, column]))
    check_cancel()
    if (count, len(support_a), len(support_b)) != (
        diagnostic.pair_count,
        diagnostic.residues_a,
        diagnostic.residues_b,
    ):
        raise ValueError("Contact audit does not match the captured diagnostic.")


def compute_interface_model(
    index: StructureIndex,
    pae: np.ndarray | None,
    confidence: PredictionConfidence | None,
    options: InterfaceAnalysisOptions,
    *,
    rank: int,
    label: str,
    object_name: str,
    check_cancel: Callable[[], None] = lambda: None,
) -> InterfaceModelResult:
    """No provider parsing, file access, viewer access or mutation occurs here."""
    tm, geometry = index.token_map, index.geometry
    if pae is not None and pae.shape != (len(tm), len(tm)):
        raise ValueError("Interface PAE must match the canonical token map.")
    eligible = frozenset(protein_token_indices(tm, geometry))
    groups = {
        chain: tuple(i for i in tm.chain_to_indices[chain] if i in eligible)
        for chain in tm.chain_order
        if any(i in eligible for i in tm.chain_to_indices[chain])
    }
    scores: list[InterfaceScoreResult] = []
    contacts: list[InterfaceContactDiagnostics] = []
    ca_contacts: dict[tuple[str, str], np.ndarray | None] = {}
    # Store CA contact masks for the original chain-versus-rest calculation.
    # PAE chain blocks are released after each pair (never cached in reports).
    for a, b in combinations(groups, 2):
        check_cancel()
        ia, ib = groups[a], groups[b]
        ab = None if pae is None else np.asarray(pae[np.ix_(ia, ib)], dtype=np.float64)
        ba = None if pae is None else np.asarray(pae[np.ix_(ib, ia)], dtype=np.float64)
        cb = ca = None
        if geometry is not None:
            for name in ("ca", "cb"):
                coords = getattr(geometry, name)
                if (
                    np.isfinite(coords[list(ia)]).all()
                    and np.isfinite(coords[list(ib)]).all()
                ):
                    contact = P.interface_contact_mask(
                        coords[list(ia)], coords[list(ib)], check_cancel
                    )
                    if name == "ca":
                        ca = contact
                    else:
                        cb = contact
        ca_contacts[a, b] = ca
        ca_contacts[b, a] = None if ca is None else ca.T
        contacts.append(
            InterfaceContactDiagnostics(
                "chain_pair",
                a,
                (b,),
                "cb_ca_gly",
                None if cb is None else int(cb.sum()),
                None if cb is None else len(_support(ia, cb, axis=1)),
                None if cb is None else len(_support(ib, cb, axis=0)),
                "Missing or ambiguous Cβ/Cα-glycine coordinates." if cb is None else "",
            )
        )
        valid_pae = _finite_pae(ab) and _finite_pae(ba)
        pae_reason = (
            "PAE unavailable."
            if pae is None
            else "PAE contains undefined or negative values."
        )
        cb_reason = "Missing or ambiguous Cβ/Cα-glycine coordinates."
        for spec in interface_metric_specs():
            if spec.result_scope != "chain_pair":
                continue
            definition = spec.interface
            assert definition is not None
            params = definition.parameters
            if spec.key == "interface_ipsae":
                params = (
                    ("pae_cutoff_angstrom", str(options.ipsae_cutoff)),
                    ("pae_comparison", "<"),
                )
            result = InterfaceScoreResult(spec.key, a, (b,), None, parameters=params)
            key = spec.key
            if key == "interface_iptm":
                matrix = None if confidence is None else confidence.pair_chain_iptm
                if matrix is not None:
                    ai, bi = tm.chain_id_to_index[a], tm.chain_id_to_index[b]
                    forward, reverse = float(matrix[ai, bi]), float(matrix[bi, ai])
                    result = replace(
                        result,
                        value=max(forward, reverse)
                        if np.isfinite([forward, reverse]).all()
                        else None,
                        forward=forward if np.isfinite(forward) else None,
                        reverse=reverse if np.isfinite(reverse) else None,
                        confidence_source="provider normalized pair_chain_iptm",
                    )
                if result.value is None:
                    result = replace(
                        result,
                        reason="One or both provider pair ipTM directions unavailable; no global or pTM fallback.",
                    )
                scores.append(result)
                continue
            if key != "interface_pdockq" and not valid_pae:
                scores.append(replace(result, reason=pae_reason))
                continue
            if definition.support in {"cb", "cb_pae"} and cb is None:
                scores.append(replace(result, reason=cb_reason))
                continue
            if key == "interface_ipsae":
                assert ab is not None and ba is not None
                forward, profile_a = P.interface_ipsae(ab, options.ipsae_cutoff)
                reverse, profile_b = P.interface_ipsae(ba, options.ipsae_cutoff)
                mask_a, mask_b = ab < options.ipsae_cutoff, ba < options.ipsae_cutoff
                union = mask_a | mask_b.T
                result = replace(
                    result,
                    value=max(forward, reverse),
                    forward=forward,
                    reverse=reverse,
                    support_a=_support(ia, union, axis=1),
                    support_b=_support(ib, union, axis=0),
                    counts=(
                        ("qualifying_pairs_ab", int(mask_a.sum())),
                        ("qualifying_pairs_ba", int(mask_b.sum())),
                    ),
                    residue_profile=tuple(zip(ia, map(float, profile_a), strict=True))
                    + tuple(zip(ib, map(float, profile_b), strict=True)),
                )
            elif key in {
                "interface_lis_kim",
                "interface_lis_dunbrack",
                "interface_clis",
                "interface_ilis",
            }:
                assert ab is not None and ba is not None
                strict = key == "interface_lis_dunbrack"
                mask_a, mask_b = (ab < 12, ba < 12) if strict else (ab <= 12, ba <= 12)
                lis_a, lis_b = P.interface_lis(ab, mask_a), P.interface_lis(ba, mask_b)
                lia_a, lia_b = int(mask_a.sum()), int(mask_b.sum())
                counts: tuple[tuple[str, int], ...] = (
                    ("LIA_ab", lia_a),
                    ("LIA_ba", lia_b),
                    ("LIA_combined", lia_a + lia_b),
                )
                if key in {"interface_clis", "interface_ilis"}:
                    assert cb is not None
                    mask_a, mask_b = mask_a & cb, mask_b & cb.T
                    clis_a, clis_b = (
                        P.interface_lis(ab, mask_a),
                        P.interface_lis(ba, mask_b),
                    )
                    counts += (
                        ("cLIA_ab", int(mask_a.sum())),
                        ("cLIA_ba", int(mask_b.sum())),
                        ("cLIA_combined", int(mask_a.sum() + mask_b.sum())),
                        ("cb_contact_pairs", int(cb.sum())),
                    )
                    forward, reverse = clis_a, clis_b
                    value = (clis_a + clis_b) / 2
                    if key == "interface_ilis":
                        value = float(np.sqrt((lis_a + lis_b) / 2 * value))
                else:
                    forward, reverse = lis_a, lis_b
                    value = (lis_a + lis_b) / 2
                union = mask_a | mask_b.T
                result = replace(
                    result,
                    value=value,
                    forward=None if key == "interface_ilis" else forward,
                    reverse=None if key == "interface_ilis" else reverse,
                    counts=counts,
                    support_a=_support(ia, union, axis=1),
                    support_b=_support(ib, union, axis=0),
                )
            else:
                assert cb is not None and geometry is not None
                sa, sb = _support(ia, cb, axis=1), _support(ib, cb, axis=0)
                n = int(cb.sum())
                plddt = _plddt_mean(geometry.cb_plddt[list(sa + sb)]) if n else 0.0
                result = replace(
                    result,
                    support_a=sa,
                    support_b=sb,
                    counts=(("cb_contact_pairs", n),),
                    interface_plddt=plddt,
                    confidence_source="original structure Cβ (Cα for glycine), 0–100",
                )
                if plddt is None:
                    result = replace(
                        result,
                        reason="Required representative-atom pLDDT is unavailable.",
                    )
                elif key == "interface_pdockq":
                    result = replace(result, value=P.interface_pdockq(plddt, n))
                else:
                    assert ab is not None and ba is not None
                    forward = (
                        P.interface_pdockq2(plddt, _transform_mean(ab, cb))
                        if n
                        else 0.0
                    )
                    reverse = (
                        P.interface_pdockq2(plddt, _transform_mean(ba, cb.T))
                        if n
                        else 0.0
                    )
                    result = replace(
                        result,
                        value=max(forward, reverse),
                        forward=forward,
                        reverse=reverse,
                        reason="No contacts: reference implementation returns zero."
                        if not n
                        else "",
                        parameters=params + (("pae_scale_angstrom", "10"),),
                    )
            scores.append(result)
    # Score each chain against all other proteins, accumulating CA contact pairs.
    for a, ia in groups.items():
        check_cancel()
        partners = tuple(b for b in groups if b != a)
        if not partners:
            continue
        contact_count = 0
        contact_a: set[int] = set()
        contact_b: set[int] = set()
        contact_reason = ""
        for b in partners:
            check_cancel()
            mask = ca_contacts[a, b]
            if mask is None:
                contact_reason = "Missing or ambiguous Cα coordinates."
                break
            contact_count += int(mask.sum())
            contact_a.update(_support(ia, mask, axis=1))
            contact_b.update(_support(groups[b], mask, axis=0))
        contacts.append(
            InterfaceContactDiagnostics(
                "chain_rest",
                a,
                partners,
                "ca",
                None if contact_reason else contact_count,
                None if contact_reason else len(contact_a),
                None if contact_reason else len(contact_b),
                contact_reason,
            )
        )
        n, pae_sum, plddt_sum = 0, 0.0, 0.0
        support_a: set[int] = set()
        support_b: set[int] = set()
        reason = ""
        for b in partners:
            rest_contact = ca_contacts[a, b]
            if rest_contact is None or geometry is None:
                reason = "Missing or ambiguous Cα coordinates."
                break
            ib = groups[b]
            if pae is None:
                reason = "PAE unavailable."
                break
            block = np.asarray(pae[np.ix_(ia, ib)], dtype=np.float64)
            if not _finite_pae(block):
                reason = "PAE contains undefined or negative values."
                break
            multiplicity = rest_contact.sum(axis=1)
            active = multiplicity > 0
            conf = geometry.ca_plddt[list(ia)][active]
            if not np.isfinite(conf).all():
                reason = "Required Cα pLDDT is unavailable."
                break
            n += int(rest_contact.sum())
            plddt_sum += float(np.dot(conf, multiplicity[active]))
            pae_sum += float((1 / (1 + (block[rest_contact] / 10) ** 2)).sum())
            support_a.update(_support(ia, rest_contact, axis=1))
            support_b.update(_support(ib, rest_contact, axis=0))
        plddt = plddt_sum / n if n else 0.0
        spec = metrics.METRICS.require("interface_pdockq2_zhu")
        assert spec.interface is not None
        scores.append(
            InterfaceScoreResult(
                spec.key,
                a,
                partners,
                None
                if reason
                else P.interface_pdockq2(plddt, pae_sum / n if n else 0, original=True),
                reason=reason
                or (
                    "No contacts: original sigmoid baseline is retained."
                    if not n
                    else ""
                ),
                parameters=spec.interface.parameters + (("pae_scale_angstrom", "10"),),
                counts=(("ca_contact_pairs", n),),
                support_a=tuple(sorted(support_a)),
                support_b=tuple(sorted(support_b)),
                interface_plddt=None if reason else plddt,
                confidence_source="original structure Cα, scored chain weighted by contact count, 0–100",
            )
        )
    excluded = sum(
        token.is_hetatm and token.res_name.upper() in {"MSE", "SEP", "TPO", "PTR"}
        for token in tm
    )
    diagnostics: tuple[str, ...] = (
        (
            f"Excluded {excluded} modified-residue HETATM atom tokens; canonical tokenization preserved.",
        )
        if excluded
        else ()
    )
    if len(groups) < 2:
        diagnostics += ("Fewer than two protein chains; no protein interfaces.",)
    return InterfaceModelResult(
        rank,
        label,
        object_name,
        str(index.path),
        tm,
        tuple(groups),
        tuple(scores),
        diagnostics,
        tuple(sorted(eligible)),
        tuple(contacts),
    )


def aggregate_interface_models(
    models: tuple[InterfaceModelResult, ...],
) -> tuple[InterfaceAggregate, ...]:
    chain_order = tuple(dict.fromkeys(c for model in models for c in model.chain_order))
    chain_position = {chain: i for i, chain in enumerate(chain_order)}
    groups: dict[
        tuple[str, str, tuple[str, ...], tuple[tuple[str, str, str], ...]],
        list[tuple[int, InterfaceScoreResult]],
    ] = defaultdict(list)
    for model in models:
        protein = frozenset(
            model.protein_indices or protein_token_indices(model.token_map)
        )
        for score in model.scores:
            if score.value is None:
                continue
            a, partners = score.chain_a, score.partners
            if metrics.METRICS.require(score.metric_key).result_scope == "chain_pair":
                a, b = sorted((a, *partners), key=chain_position.__getitem__)
                partners = (b,)
                if a != score.chain_a:
                    score = replace(score, forward=score.reverse, reverse=score.forward)
            else:
                partners = tuple(sorted(partners, key=chain_position.__getitem__))
            identities = tuple(
                sorted(
                    (t.chain_id, t.resi, t.res_name)
                    for t in model.token_map
                    if not t.is_hetatm
                    and t.chain_id in (a, *partners)
                    and t.token_idx in protein
                )
            )
            groups[score.metric_key, a, partners, identities].append(
                (model.rank, score)
            )
    result = []
    for (key, a, partners, identities), members in groups.items():
        values = np.array([score.value for _, score in members], dtype=np.float64)
        forwards = [score.forward for _, score in members]
        reverses = [score.reverse for _, score in members]
        result.append(
            InterfaceAggregate(
                key,
                a,
                partners,
                identities,
                tuple(rank for rank, _ in members),
                len(members),
                float(values.mean()),
                float(values.std(ddof=0)),
                float(values.min()),
                float(values.max()),
                float(np.mean([x for x in forwards if x is not None]))
                if all(x is not None for x in forwards)
                else None,
                float(np.mean([x for x in reverses if x is not None]))
                if all(x is not None for x in reverses)
                else None,
            )
        )
    return tuple(result)
