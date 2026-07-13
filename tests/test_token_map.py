from __future__ import annotations

import sys
import tempfile
import types
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.mol_viewer import (  # noqa: E402
    compact_selection_expression,
    compare_token_map_to_object,
)
from FoldQC.token_map import (  # noqa: E402
    TokenInfo,
    TokenMap,
    build_token_map,
    extract_structure_plddt,
    parse_structure_atoms,
)

QUOTED_ATOM_CIF = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
HETATM 2832 C "C1'" . FAD B 2 . -3.128 2.675 2.545 1.00 98.60
HETATM 2833 O "O5'" . FAD B 2 . 0.702 -1.532 4.883 1.00 98.42
#
"""


class TokenMapTests(unittest.TestCase):
    @staticmethod
    def _token(
        token_idx: int,
        *,
        chain_id: str = "A",
        res_num: int = 1,
        res_name: str = "ALA",
        is_hetatm: bool = False,
        atom_name: str | None = None,
        obj_name: str = "model",
    ) -> TokenInfo:
        del obj_name
        return TokenInfo(
            token_idx=token_idx,
            chain_id=chain_id,
            res_num=res_num,
            res_name=res_name,
            is_hetatm=is_hetatm,
            atom_name=atom_name,
        )

    def test_token_info_is_frozen(self) -> None:
        token = self._token(0)

        with self.assertRaises(FrozenInstanceError):
            token.res_num = 2  # type: ignore[misc]

    def test_token_map_sequence_hash_and_metadata(self) -> None:
        tokens = (
            self._token(0, chain_id="A", res_num=1),
            self._token(1, chain_id="A", res_num=2),
            self._token(
                2,
                chain_id="L",
                res_num=3,
                res_name="LIG",
                is_hetatm=True,
                atom_name="C1",
            ),
        )
        token_map = TokenMap(tokens)

        self.assertEqual(len(token_map), 3)
        self.assertIs(token_map[1], tokens[1])
        self.assertEqual(token_map[1:], tokens[1:])
        self.assertEqual(tuple(token_map), tokens)
        self.assertEqual(token_map, TokenMap(tokens))
        self.assertEqual(hash(token_map), hash(TokenMap(tokens)))
        self.assertEqual(token_map.chain_order, ("A", "L"))
        self.assertEqual(token_map.chain_to_indices, {"A": (0, 1), "L": (2,)})
        self.assertEqual(token_map.chain_id_to_index, {"A": 0, "L": 1})
        self.assertEqual(token_map.polymer_indices, (0, 1))
        self.assertEqual(token_map.polymer_token_by_residue, {("A", 1): 0, ("A", 2): 1})
        self.assertEqual(token_map.hetatm_token_by_atom, {("L", 3, "C1"): 2})
        self.assertEqual(
            token_map.token_identities,
            frozenset(
                {
                    ("A", 1, "ALA", None),
                    ("A", 2, "ALA", None),
                    ("L", 3, "LIG", "C1"),
                }
            ),
        )
        with self.assertRaises(TypeError):
            token_map.chain_to_indices["B"] = (3,)  # type: ignore[index]

    def test_token_map_rejects_non_dense_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "dense token indices"):
            TokenMap((self._token(1),))

    def test_cif_parser_unquotes_atom_names_with_apostrophes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quoted.cif"
            path.write_text(QUOTED_ATOM_CIF)

            atoms = parse_structure_atoms(path)
            token_map = build_token_map(path)
            plddt = extract_structure_plddt(path)

        self.assertEqual([atom["name"] for atom in atoms], ["C1'", "O5'"])
        self.assertIsInstance(token_map, TokenMap)
        self.assertEqual([tok.atom_name for tok in token_map], ["C1'", "O5'"])
        self.assertFalse(hasattr(token_map[0], "pymol_selection"))
        np.testing.assert_allclose(plddt, np.array([0.986, 0.9842], dtype=np.float32))

    def _compare_overlap(self, token_map, atoms):
        old_pymol = sys.modules.get("pymol")
        cmd = types.SimpleNamespace(
            get_model=lambda _obj_name: types.SimpleNamespace(atom=atoms)
        )
        sys.modules["pymol"] = types.SimpleNamespace(cmd=cmd)
        try:
            return compare_token_map_to_object(token_map, "target")
        finally:
            if old_pymol is None:
                sys.modules.pop("pymol", None)
            else:
                sys.modules["pymol"] = old_pymol

    @staticmethod
    def _atom(
        chain: str,
        resi: int | str,
        resn: str,
        *,
        hetatm: bool = False,
        name: str = "CA",
    ):
        return types.SimpleNamespace(
            chain=chain,
            resi=str(resi),
            resn=resn,
            hetatm=hetatm,
            name=name,
        )

    def test_token_overlap_full_match(self) -> None:
        token_map = TokenMap(
            (
                self._token(0, chain_id="A", res_num=1, res_name="ALA"),
                self._token(1, chain_id="A", res_num=2, res_name="GLY"),
            )
        )

        overlap = self._compare_overlap(
            token_map,
            [
                self._atom("A", 1, "ALA", name="N"),
                self._atom("A", 1, "ALA", name="CA"),
                self._atom("A", 2, "GLY", name="CA"),
            ],
        )

        self.assertEqual(overlap.prediction_tokens, 2)
        self.assertEqual(overlap.target_tokens, 2)
        self.assertEqual(overlap.matched_prediction_tokens, 2)
        self.assertEqual(overlap.matched_target_tokens, 2)
        self.assertEqual(overlap.target_coverage, 1.0)
        self.assertEqual(overlap.prediction_coverage, 1.0)

    def test_token_overlap_target_subset_keeps_full_target_coverage(self) -> None:
        token_map = TokenMap(
            (
                self._token(0, chain_id="A", res_num=1, res_name="ALA"),
                self._token(1, chain_id="A", res_num=2, res_name="GLY"),
            )
        )

        overlap = self._compare_overlap(
            token_map,
            [self._atom("A", 1, "ALA", name="CA")],
        )

        self.assertEqual(overlap.target_tokens, 1)
        self.assertEqual(overlap.matched_target_tokens, 1)
        self.assertEqual(overlap.target_coverage, 1.0)
        self.assertEqual(overlap.prediction_coverage, 0.5)

    def test_token_overlap_residue_name_mismatch_does_not_match(self) -> None:
        token_map = TokenMap((self._token(0, chain_id="A", res_num=1, res_name="ALA"),))

        overlap = self._compare_overlap(
            token_map,
            [self._atom("A", 1, "ASP", name="CA")],
        )

        self.assertEqual(overlap.target_tokens, 1)
        self.assertEqual(overlap.matched_target_tokens, 0)
        self.assertEqual(overlap.target_coverage, 0.0)

    def test_token_overlap_hetatm_requires_residue_and_atom_name(self) -> None:
        token_map = TokenMap(
            (
                self._token(
                    0,
                    chain_id="L",
                    res_num=1,
                    res_name="LIG",
                    is_hetatm=True,
                    atom_name="C1",
                ),
            )
        )

        wrong_atom = self._compare_overlap(
            token_map,
            [self._atom("L", 1, "LIG", hetatm=True, name="C2")],
        )
        wrong_resn = self._compare_overlap(
            token_map,
            [self._atom("L", 1, "DRG", hetatm=True, name="C1")],
        )
        matched = self._compare_overlap(
            token_map,
            [self._atom("L", 1, "LIG", hetatm=True, name="C1")],
        )

        self.assertEqual(wrong_atom.matched_target_tokens, 0)
        self.assertEqual(wrong_resn.matched_target_tokens, 0)
        self.assertEqual(matched.matched_target_tokens, 1)

    def test_compact_selection_collapses_unordered_residue_ranges(self) -> None:
        token_map = TokenMap(
            (
                self._token(0, res_num=-2),
                self._token(1, res_num=-1),
                self._token(2, res_num=0),
                self._token(3, res_num=2),
                self._token(4, res_num=3),
                self._token(5, res_num=5),
            )
        )

        expression = compact_selection_expression(
            [5, 3, 0, 4, 2, 1, 3, -1, 99],
            [("model", token_map)],
        )

        self.assertEqual(
            expression,
            "(%model and polymer and chain A)",
        )

    def test_compact_selection_groups_objects_chains_and_ligand_residues(
        self,
    ) -> None:
        first_map = TokenMap(
            (
                self._token(0, chain_id="A", res_num=10, obj_name="model_0"),
                self._token(1, chain_id="A", res_num=11, obj_name="model_0"),
                self._token(2, chain_id="B", res_num=4, obj_name="model_0"),
                self._token(
                    3,
                    chain_id="X",
                    res_num=1,
                    res_name="LIG",
                    is_hetatm=True,
                    atom_name="C1",
                    obj_name="model_0",
                ),
                self._token(
                    4,
                    chain_id="X",
                    res_num=1,
                    res_name="LIG",
                    is_hetatm=True,
                    atom_name="N3",
                    obj_name="model_0",
                ),
            )
        )
        second_map = TokenMap(
            (
                self._token(0, chain_id="A", res_num=10, obj_name="model_1"),
                self._token(1, chain_id="A", res_num=11, obj_name="model_1"),
                self._token(2, chain_id="B", res_num=4, obj_name="model_1"),
                self._token(
                    3,
                    chain_id="X",
                    res_num=1,
                    res_name="LIG",
                    is_hetatm=True,
                    atom_name="C1",
                    obj_name="model_1",
                ),
                self._token(
                    4,
                    chain_id="X",
                    res_num=1,
                    res_name="LIG",
                    is_hetatm=True,
                    atom_name="N3",
                    obj_name="model_1",
                ),
            )
        )

        expression = compact_selection_expression(
            [0, 1, 2, 3, 4],
            [("model_0", first_map), ("model_1", second_map)],
        )

        self.assertEqual(
            expression,
            "(%model_0 and polymer and chain A) or "
            "(%model_0 and polymer and chain B) or "
            "(%model_0 and hetatm and chain X and resi 1 and resn LIG) or "
            "(%model_1 and polymer and chain A) or "
            "(%model_1 and polymer and chain B) or "
            "(%model_1 and hetatm and chain X and resi 1 and resn LIG)",
        )

    def test_compact_selection_supports_blank_chain_and_apostrophe_atom_names(
        self,
    ) -> None:
        token_map = TokenMap(
            (
                self._token(0, chain_id="", res_num=7),
                self._token(
                    1,
                    chain_id="X",
                    res_num=2,
                    res_name="FAD",
                    is_hetatm=True,
                    atom_name="C1'",
                ),
                self._token(
                    2,
                    chain_id="X",
                    res_num=2,
                    res_name="FAD",
                    is_hetatm=True,
                    atom_name="O5'",
                ),
            )
        )

        expression = compact_selection_expression([0, 1, 2], [("model", token_map)])

        self.assertEqual(
            expression,
            '(%model and polymer and chain "") or '
            "(%model and hetatm and chain X and resi 2 and resn FAD)",
        )

    def test_compact_selection_keeps_partial_chain_and_ligand_details(self) -> None:
        token_map = TokenMap(
            (
                self._token(0, res_num=1),
                self._token(1, res_num=2),
                self._token(2, res_num=3),
                self._token(
                    3,
                    chain_id="X",
                    res_num=1,
                    res_name="LIG",
                    is_hetatm=True,
                    atom_name="C1",
                ),
                self._token(
                    4,
                    chain_id="X",
                    res_num=1,
                    res_name="LIG",
                    is_hetatm=True,
                    atom_name="N3",
                ),
            )
        )

        expression = compact_selection_expression([0, 1, 3], [("model", token_map)])

        self.assertEqual(
            expression,
            "(%model and polymer and chain A and resi 1-2) or "
            "(%model and hetatm and chain X and resi 1 and resn LIG and name C1)",
        )

    def test_compact_selection_falls_back_for_unsafe_identifiers(self) -> None:
        token_map = TokenMap(
            (
                self._token(0, res_num=1),
                self._token(
                    1,
                    chain_id="X",
                    res_num=2,
                    res_name="LIG",
                    is_hetatm=True,
                    atom_name="C,1",
                ),
            )
        )

        expression = compact_selection_expression(
            [0, 1], [("unsafe-object", token_map)]
        )

        self.assertEqual(
            expression,
            "(/unsafe-object//A/1/) or (/unsafe-object//X/2/C,1)",
        )

    def test_compact_selection_requires_object_name_and_tokeninfo_fields(self) -> None:
        token_map = TokenMap((self._token(0),))

        with self.assertRaisesRegex(ValueError, "non-empty object name"):
            compact_selection_expression([0], [(None, token_map)])

        with self.assertRaisesRegex(ValueError, "required TokenInfo fields"):
            compact_selection_expression(
                [0],
                [("model", [type("IncompleteToken", (), {"token_idx": 0})()])],
            )

    def test_compact_selection_rejects_invalid_structured_values(self) -> None:
        token = TokenInfo(
            token_idx=0,
            chain_id="A",
            res_num="not-a-residue-number",  # type: ignore[arg-type]
            res_name="ALA",
            is_hetatm=False,
            atom_name=None,
        )

        with self.assertRaisesRegex(ValueError, "invalid res_num"):
            compact_selection_expression([0], [("model", TokenMap((token,)))])


if __name__ == "__main__":
    unittest.main()
