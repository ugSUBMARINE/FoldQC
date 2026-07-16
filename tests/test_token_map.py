from __future__ import annotations

import sys
import tempfile
import types
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.mol_viewer import (  # noqa: E402
    compact_selection_expression,
    compare_token_map_to_object,
)
from FoldQC.structure_index import StructureIndex  # noqa: E402
from FoldQC.token_map import ResidueId, TokenInfo, TokenMap  # noqa: E402

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
        insertion_code: str = "",
        res_name: str = "ALA",
        is_hetatm: bool = False,
        atom_name: str | None = None,
        obj_name: str = "model",
    ) -> TokenInfo:
        del obj_name
        return TokenInfo(
            token_idx=token_idx,
            chain_id=chain_id,
            residue_id=ResidueId(res_num, insertion_code),
            res_name=res_name,
            is_hetatm=is_hetatm,
            atom_name=atom_name,
        )

    def test_token_info_is_frozen(self) -> None:
        token = self._token(0)

        with self.assertRaises(FrozenInstanceError):
            token.res_num = 2  # type: ignore[misc]

    def test_residue_id_normalizes_and_parses_canonical_labels(self) -> None:
        self.assertEqual(ResidueId(42, "."), ResidueId(42))
        self.assertEqual(ResidueId(42, "?"), ResidueId(42))
        self.assertEqual(ResidueId.parse("-3A"), ResidueId(-3, "A"))
        self.assertEqual(str(ResidueId(42, "A")), "42A")
        with self.assertRaisesRegex(ValueError, "single alphanumeric"):
            ResidueId(42, "AB")

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
        self.assertEqual(
            token_map.polymer_token_by_residue,
            {("A", ResidueId(1), "ALA"): 0, ("A", ResidueId(2), "ALA"): 1},
        )
        self.assertEqual(
            token_map.hetatm_token_by_atom,
            {("L", ResidueId(3), "LIG", "C1"): 2},
        )
        self.assertEqual(
            token_map.token_identities,
            frozenset(
                {
                    ("A", ResidueId(1), "ALA", None),
                    ("A", ResidueId(2), "ALA", None),
                    ("L", ResidueId(3), "LIG", "C1"),
                }
            ),
        )
        with self.assertRaises(TypeError):
            token_map.chain_to_indices["B"] = (3,)  # type: ignore[index]

    def test_token_map_rejects_non_dense_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "dense token indices"):
            TokenMap((self._token(1),))

    def test_chain_order_is_unique_by_first_appearance(self) -> None:
        token_map = TokenMap(
            (
                self._token(0, chain_id="A", res_num=1),
                self._token(1, chain_id="B", res_num=1),
                self._token(2, chain_id="A", res_num=2),
            )
        )
        self.assertEqual(token_map.chain_order, ("A", "B"))

    def test_cif_insertion_codes_and_ligand_hydrogens_are_lossless(self) -> None:
        cif = """data_test
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.auth_comp_id
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_ins_code
_atom_site.B_iso_or_equiv
ATOM C CA ALA 42 A A 90
HETATM C C1 LIG 42 L . 70
HETATM H H1 LIG 42 L . 60
HETATM D D1 LIG 42 L . 50
HETATM T T1 LIG 42 L . 40
HETATM O O1 LIG 42 L A 30
#
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "insertions.cif"
            path.write_text(cif)
            index = StructureIndex.from_path(path)

        self.assertEqual(index.atom_count, 6)
        self.assertEqual(index.atom_to_token, (0, 1, None, None, None, 2))
        self.assertEqual(
            [token.residue_id for token in index.token_map],
            [ResidueId(42, "A"), ResidueId(42), ResidueId(42, "A")],
        )
        np.testing.assert_allclose(
            index.collapse_atom_plddt(np.array([90, 70, 1, 2, 3, 30])),
            [0.9, 0.7, 0.3],
        )

    def test_pdb_insertion_code_and_element_columns_are_parsed(self) -> None:
        pdb = (
            "ATOM      1  CA  GLY A  42A     0.000   0.000   0.000  1.00 80.00           C  \n"
            "HETATM    2  H1  LIG L  42A     1.000   0.000   0.000  1.00 70.00           H  \n"
            "HETATM    3  C1  LIG L  42A     2.000   0.000   0.000  1.00 60.00           C  \n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "insertions.pdb"
            path.write_text(pdb)
            index = StructureIndex.from_path(path)

        self.assertEqual(index.atom_to_token, (0, None, 1))
        self.assertEqual(index.token_map[0].resi, "42A")
        self.assertEqual(index.token_map[1].resi, "42A")

    def test_structure_rejects_conflicts_duplicates_and_only_hydrogens(self) -> None:
        cases = {
            "conflict.pdb": (
                "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 80.00           C  \n"
                "ATOM      2  CA  GLY A   1      1.000   0.000   0.000  1.00 70.00           C  \n",
                "Conflicting residue names",
            ),
            "duplicate.pdb": (
                "HETATM    1  C1  LIG L   1      0.000   0.000   0.000  1.00 80.00           C  \n"
                "HETATM    2  C1  LIG L   1      1.000   0.000   0.000  1.00 70.00           C  \n",
                "Duplicate HETATM",
            ),
            "ligand_conflict.pdb": (
                "HETATM    1  C1  LIG L   1      0.000   0.000   0.000  1.00 80.00           C  \n"
                "HETATM    2  C2  DRG L   1      1.000   0.000   0.000  1.00 70.00           C  \n",
                "Conflicting residue names",
            ),
            "hydrogen.pdb": (
                "HETATM    1  H1  LIG L   1      0.000   0.000   0.000  1.00 80.00           H  \n",
                "no supported tokens",
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, (text, message) in cases.items():
                path = Path(tmp) / name
                path.write_text(text)
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    StructureIndex.from_path(path)

    def test_cif_parser_unquotes_atom_names_with_apostrophes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quoted.cif"
            path.write_text(QUOTED_ATOM_CIF)

            original_open = Path.open
            with mock.patch.object(
                Path,
                "open",
                autospec=True,
                side_effect=original_open,
            ) as open_mock:
                index = StructureIndex.from_path(path)

        self.assertEqual(open_mock.call_count, 1)
        token_map = index.token_map
        self.assertIsInstance(token_map, TokenMap)
        self.assertEqual([tok.atom_name for tok in token_map], ["C1'", "O5'"])
        self.assertFalse(hasattr(token_map[0], "pymol_selection"))
        self.assertEqual(index.atom_count, 2)
        self.assertEqual(index.atom_to_token, (0, 1))
        np.testing.assert_allclose(
            index.structure_plddt,
            np.array([0.986, 0.9842], dtype=np.float32),
        )
        self.assertFalse(index.structure_plddt.flags.writeable)
        with self.assertRaises(FrozenInstanceError):
            index.path = Path("other.cif")  # type: ignore[misc]

    def test_atom_plddt_sources_average_polymer_atoms_in_token_order(self) -> None:
        pdb = """\
ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ALA A   1      1.000   0.000   0.000  1.00 80.00           C
HETATM    3  C1  LIG L   2      2.000   0.000   0.000  1.00 70.00           C
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pdb"
            path.write_text(pdb)
            index = StructureIndex.from_path(path)

        self.assertEqual(index.format, "pdb")
        self.assertEqual(index.atom_to_token, (0, 0, 1))
        np.testing.assert_allclose(index.structure_plddt, [0.85, 0.7])
        np.testing.assert_allclose(
            index.collapse_atom_plddt(np.array([70.0, 90.0, 60.0])),
            [0.8, 0.6],
        )
        with self.assertRaisesRegex(ValueError, "does not match 3 atoms"):
            index.collapse_atom_plddt(np.ones(2, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "must not contain infinity"):
            index.collapse_atom_plddt(np.array([0.8, np.inf, 0.6]))

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

    def test_compact_selection_emits_exact_insertion_code_clauses(self) -> None:
        token_map = TokenMap(
            (
                self._token(0, res_num=41),
                self._token(1, res_num=42, insertion_code="A"),
                self._token(2, res_num=43),
            )
        )

        expression = compact_selection_expression([0, 1], [("model", token_map)])

        self.assertIn("resi 41", expression)
        self.assertIn("resi 42A", expression)

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
        with self.assertRaisesRegex(ValueError, "Invalid residue identifier"):
            ResidueId.parse("not-a-residue-number")


if __name__ == "__main__":
    unittest.main()
