"""
Token map
=========
Store viewer-independent prediction-token metadata created by
``StructureIndex`` so per-token arrays can be mapped onto a loaded structure by
a viewer adapter.

Token definition
----------------
- Polymer chains (protein, DNA, RNA): **one token per residue**.
  All atoms belonging to a residue share that token index; the token’s
  representative coordinate is the Cα (protein) or C1′ (nucleotide).
- Ligand chains (HETATM): **one token per heavy atom**.

The token order follows the chain order in the CIF/PDB file, which in turn
reflects the provider's original output order.

Critical implementation note
-----------------------------
Molecular viewers may reorder HETATM atoms after loading a structure file.
Token indices are therefore derived exclusively from the original CIF/PDB file
and ligand tokens retain atom names for stable viewer-side lookup.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import overload


@dataclass(frozen=True, slots=True)
class TokenInfo:
    """Viewer-independent identity and metadata for one prediction token."""

    token_idx: int  # 0-based prediction token index
    chain_id: str  # structure-file chain ID, e.g. "A"
    res_num: int  # residue sequence number from the structure file
    res_name: str  # residue or ligand name, e.g. "ALA", "SAH", "LIG"
    is_hetatm: bool  # True for ligand atoms, False for polymer residues

    # For HETATM tokens only: individual atom name (e.g. "C1", "N3").
    # None for polymer tokens (all atoms of the residue share the token).
    atom_name: str | None


TokenIdentity = tuple[str, int, str, str | None]
PolymerTokenKey = tuple[str, int]
HetatmTokenKey = tuple[str, int, str]


@dataclass(frozen=True, slots=True)
class TokenMap(Sequence[TokenInfo]):
    """Immutable prediction-token sequence with reusable derived metadata.

    Equality and hashing depend only on :attr:`tokens`. Hashing therefore
    scales with the number of tokens and is intended only for small, bounded
    caches rather than process-wide memoization.
    """

    tokens: tuple[TokenInfo, ...]
    chain_order: tuple[str, ...] = field(
        init=False, repr=False, compare=False, hash=False
    )
    chain_to_indices: Mapping[str, tuple[int, ...]] = field(
        init=False, repr=False, compare=False, hash=False
    )
    chain_id_to_index: Mapping[str, int] = field(
        init=False, repr=False, compare=False, hash=False
    )
    polymer_indices: tuple[int, ...] = field(
        init=False, repr=False, compare=False, hash=False
    )
    polymer_token_by_residue: Mapping[PolymerTokenKey, int] = field(
        init=False, repr=False, compare=False, hash=False
    )
    hetatm_token_by_atom: Mapping[HetatmTokenKey, int] = field(
        init=False, repr=False, compare=False, hash=False
    )
    token_identities: frozenset[TokenIdentity] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        tokens = tuple(self.tokens)
        object.__setattr__(self, "tokens", tokens)

        chain_order: list[str] = []
        chain_to_indices: dict[str, list[int]] = {}
        polymer_indices: list[int] = []
        polymer_tokens: dict[PolymerTokenKey, int] = {}
        hetatm_tokens: dict[HetatmTokenKey, int] = {}
        identities: set[TokenIdentity] = set()
        last_chain: str | None = None

        for position, token in enumerate(tokens):
            if token.token_idx != position:
                raise ValueError(
                    "TokenMap requires dense token indices matching tuple positions; "
                    f"position {position} contains token_idx {token.token_idx}."
                )

            chain_id = str(token.chain_id)
            residue_number = token.res_num
            residue_name = str(token.res_name)
            token_idx = int(token.token_idx)

            if chain_id != last_chain:
                chain_order.append(chain_id)
                last_chain = chain_id
            chain_to_indices.setdefault(chain_id, []).append(token_idx)

            if token.is_hetatm:
                atom_name = str(token.atom_name or "")
                hetatm_tokens[(chain_id, residue_number, atom_name)] = token_idx
                identity_atom_name: str | None = atom_name
            else:
                polymer_indices.append(token_idx)
                polymer_tokens[(chain_id, residue_number)] = token_idx
                identity_atom_name = None
            identities.add((chain_id, residue_number, residue_name, identity_atom_name))

        chain_indices = {
            chain_id: tuple(indices) for chain_id, indices in chain_to_indices.items()
        }
        chain_id_to_index = {
            chain_id: chain_idx for chain_idx, chain_id in enumerate(chain_order)
        }
        object.__setattr__(self, "chain_order", tuple(chain_order))
        object.__setattr__(self, "chain_to_indices", MappingProxyType(chain_indices))
        object.__setattr__(
            self, "chain_id_to_index", MappingProxyType(chain_id_to_index)
        )
        object.__setattr__(self, "polymer_indices", tuple(polymer_indices))
        object.__setattr__(
            self, "polymer_token_by_residue", MappingProxyType(polymer_tokens)
        )
        object.__setattr__(
            self, "hetatm_token_by_atom", MappingProxyType(hetatm_tokens)
        )
        object.__setattr__(self, "token_identities", frozenset(identities))

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[TokenInfo]:
        return iter(self.tokens)

    @overload
    def __getitem__(self, index: int) -> TokenInfo: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TokenInfo, ...]: ...

    def __getitem__(self, index: int | slice) -> TokenInfo | tuple[TokenInfo, ...]:
        return self.tokens[index]


@dataclass(frozen=True)
class TokenOverlapSummary:
    """Identity overlap between a prediction token map and a viewer object."""

    prediction_tokens: int
    target_tokens: int
    matched_prediction_tokens: int
    matched_target_tokens: int
    target_coverage: float
    prediction_coverage: float
