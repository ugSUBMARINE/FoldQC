"""Regression coverage for frozen-record behavior across supported Pythons."""

from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import numpy as np
import pytest
from FoldQC.structure_index import StructureIndex, _AtomRecord
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap


@pytest.fixture(params=("residue", "token", "map", "atom", "index"))
def frozen_record(request):
    residue = ResidueId(42, "A")
    token = TokenInfo(0, "A", residue, "ALA", False, None)
    token_map = TokenMap((token,))
    return {
        "residue": residue,
        "token": token,
        "map": token_map,
        "atom": _AtomRecord(False, "CA", "ALA", residue, "A", 90.0, "C"),
        "index": StructureIndex(
            Path("model.cif"), "cif", token_map, 1, (0,), np.array([0.9])
        ),
    }[request.param]


def test_frozen_records_reject_field_property_and_unknown_attribute_mutation(
    frozen_record,
):
    # Property and unknown-name assignments triggered CPython's broken
    # frozen/slotted setter even though stored fields rejected writes correctly.
    field_names = [field.name for field in fields(frozen_record)]
    names = list(field_names)
    names.extend(
        name
        for name in ("res_num", "insertion_code", "resi")
        if isinstance(getattr(type(frozen_record), name, None), property)
    )
    names.append("unexpected_attribute")
    for name in names:
        original = getattr(frozen_record, name, None)
        with pytest.raises(FrozenInstanceError):
            setattr(frozen_record, name, None)
        with pytest.raises(FrozenInstanceError):
            delattr(frozen_record, name)
        if name in field_names:
            assert getattr(frozen_record, name) is original
        else:
            assert getattr(frozen_record, name, None) == original
    assert not hasattr(frozen_record, "unexpected_attribute")


def test_residue_updates_create_new_immutable_token_identities():
    residue = ResidueId(42, "A")
    token = TokenInfo(0, "A", residue, "ALA", False, None)
    original_hash = hash(token)
    with pytest.raises(FrozenInstanceError):
        token.residue_id.number = 43
    with pytest.raises(FrozenInstanceError):
        token.residue_id.insertion_code = "B"

    updated = replace(token, residue_id=replace(residue, insertion_code="B"))
    assert token.resi == "42A"
    assert hash(token) == original_hash
    assert updated.resi == "42B"
    assert updated != token
    assert TokenMap((token,)) != TokenMap((updated,))
