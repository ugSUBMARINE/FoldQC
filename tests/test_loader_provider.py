from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.loader import (  # noqa: E402
    discover_prediction_candidates,
    load_prediction_data,
    scan_prediction_path,
)


CIF_TEXT = """data_test
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.auth_comp_id
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.B_iso_or_equiv
ATOM N ALA 1 A 80.0
ATOM CA ALA 1 A 100.0
HETATM C1 LIG 1 L 40.0
#
"""


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def _write_af3_server_job(root: Path) -> None:
    root.mkdir(parents=True)
    (root / f"{root.name}_model_0.cif").write_text(CIF_TEXT)
    _write_json(
        root / f"{root.name}_summary_confidences_0.json", {"ranking_score": 1.0}
    )
    _write_json(root / f"{root.name}_full_data_0.json", {})


def _write_protenix_job(root: Path) -> None:
    predictions = root / "seed_1" / "predictions"
    predictions.mkdir(parents=True)
    (predictions / f"{root.name}_sample_0.cif").write_text(CIF_TEXT)
    _write_json(
        predictions / f"{root.name}_summary_confidence_sample_0.json",
        {"ranking_score": 1.0},
    )


def _tar_write_mode(path: Path) -> str:
    return "w:gz" if path.name.endswith((".tar.gz", ".tgz")) else "w"


def _add_tar_text(tf: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode()
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _add_tar_json(tf: tarfile.TarFile, name: str, payload: dict) -> None:
    _add_tar_text(tf, name, json.dumps(payload))


def _add_tar_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _npz_bytes(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


class LoaderProviderTests(unittest.TestCase):
    def test_single_structure_file_scans_as_structure_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            structure = Path(tmp) / "model.cif"
            structure.write_text(CIF_TEXT)

            files = scan_prediction_path(structure)
            data = load_prediction_data(files)

        self.assertEqual(files.provider, "structure_only")
        self.assertEqual(files.n_models, 1)
        self.assertFalse(files.supports_ensemble)
        self.assertEqual(files.structure_path(0).name, "model.cif")
        np.testing.assert_allclose(
            data.structure_plddt,
            np.array([0.8, 0.4], dtype=np.float32),
        )

    def test_boltz_directory_scans_with_legacy_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = Path(tmp) / "target"
            pred_dir.mkdir()
            (pred_dir / "target_model_0.cif").write_text(CIF_TEXT)
            _write_json(pred_dir / "confidence_target_model_0.json", {"ptm": 0.8})
            np.savez(pred_dir / "plddt_target_model_0.npz", plddt=np.array([0.7, 0.6]))

            files = scan_prediction_path(pred_dir)
            data = load_prediction_data(files)

        self.assertEqual(files.provider, "boltz")
        self.assertEqual(files.structure_files[0][0], 0)
        self.assertTrue(files.has_plddt)
        np.testing.assert_allclose(data.plddt, np.array([0.7, 0.6]))

    def test_boltz_lab_directory_scans_sample_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = Path(tmp) / "lab_job"
            pred_dir.mkdir()
            (pred_dir / "sample_0_predicted_structure.cif").write_text(CIF_TEXT)
            np.savez(pred_dir / "sample_0_pae.npz", pae=np.array([[0.0, 1.0]]))
            _write_json(
                pred_dir / "metrics.json",
                {
                    "sample_results": {
                        "sample_0": {
                            "structure_confidence": 0.91,
                            "ptm": 0.82,
                            "complex_pde": 1.2,
                        }
                    }
                },
            )

            files = scan_prediction_path(pred_dir)
            data = load_prediction_data(files, rank=0, load_pae=True)

        self.assertEqual(files.provider, "boltz_lab")
        self.assertEqual(files.provider_label, "Boltz Lab")
        self.assertEqual(files.name, "lab_job")
        self.assertEqual(files.n_models, 1)
        self.assertTrue(files.has_pae)
        self.assertEqual(files.models[0].display_label, "sample 0")
        self.assertEqual(files.models[0].metadata["sample_index"], 0)
        self.assertEqual(
            files.structure_path(0).name, "sample_0_predicted_structure.cif"
        )
        np.testing.assert_allclose(data.pae, np.array([[0.0, 1.0]]))
        self.assertEqual(data.confidence["confidence_score"], 0.91)
        self.assertEqual(data.confidence["structure_confidence"], 0.91)

    def test_boltz_api_run_root_scans_samples_best_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "api_job"
            prediction = root / "outputs" / "files" / "prediction"
            prediction.mkdir(parents=True)
            (root / ".boltz-run.json").write_text("{}")
            (prediction / "sample_0_predicted_structure.cif").write_text(CIF_TEXT)
            (prediction / "sample_1_predicted_structure.cif").write_text(CIF_TEXT)
            np.savez(prediction / "sample_1_pae.npz", pae=np.array([[0.0, 2.0]]))
            results = [
                {"metrics": {"structure_confidence": 0.25, "ptm": 0.7}},
                {"metrics": {"structure_confidence": 0.95, "ptm": 0.9}},
            ]
            _write_json(
                prediction / "metrics.json",
                {"best_sample": results[1], "all_sample_results": results},
            )
            _write_json(root / "run.json", {"output": {"all_sample_results": results}})

            files = scan_prediction_path(root)
            data = load_prediction_data(files, rank=0, load_pae=True)

        self.assertEqual(files.provider, "boltz_api")
        self.assertEqual(files.provider_label, "Boltz API")
        self.assertEqual(files.name, "api_job")
        self.assertEqual(files.n_models, 2)
        self.assertTrue(files.has_pae)
        self.assertEqual(files.models[0].display_label, "rank 0 - sample 1")
        self.assertEqual(files.models[0].metadata["sample_index"], 1)
        self.assertEqual(files.models[1].display_label, "rank 1 - sample 0")
        self.assertEqual(
            files.structure_path(0).name, "sample_1_predicted_structure.cif"
        )
        np.testing.assert_allclose(data.pae, np.array([[0.0, 2.0]]))
        self.assertEqual(data.confidence["confidence_score"], 0.95)
        self.assertEqual(data.confidence["ptm"], 0.9)

    def test_boltz_api_run_root_falls_back_to_run_json_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "api_job"
            prediction = root / "outputs" / "files" / "prediction"
            prediction.mkdir(parents=True)
            (prediction / "sample_0_predicted_structure.cif").write_text(CIF_TEXT)
            _write_json(prediction / "metrics.json", {"all_sample_results": []})
            _write_json(
                root / "run.json",
                {
                    "output": {
                        "all_sample_results": [
                            {"metrics": {"structure_confidence": 0.81, "iptm": 0.72}}
                        ]
                    }
                },
            )

            files = scan_prediction_path(root)
            data = load_prediction_data(files, rank=0)

        self.assertEqual(files.provider, "boltz_api")
        self.assertEqual(data.confidence["confidence_score"], 0.81)
        self.assertEqual(data.confidence["iptm"], 0.72)

    def test_boltz_api_extracted_prediction_folder_scans_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prediction = Path(tmp) / "prediction"
            prediction.mkdir()
            (prediction / "sample_0_predicted_structure.cif").write_text(CIF_TEXT)
            _write_json(
                prediction / "metrics.json",
                {
                    "all_sample_results": [
                        {"metrics": {"structure_confidence": 0.88, "ptm": 0.77}}
                    ]
                },
            )

            files = scan_prediction_path(prediction)
            data = load_prediction_data(files, rank=0)

        self.assertEqual(files.provider, "boltz_api")
        self.assertEqual(files.name, "prediction")
        self.assertEqual(files.n_models, 1)
        self.assertEqual(data.confidence["confidence_score"], 0.88)

    def test_boltz_lab_and_api_parent_discovery_reports_top_level_candidates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            lab = parent / "lab_job"
            lab.mkdir()
            (lab / "sample_0_predicted_structure.cif").write_text(CIF_TEXT)
            _write_json(
                lab / "metrics.json",
                {"sample_results": {"sample_0": {"structure_confidence": 0.5}}},
            )

            api = parent / "api_job"
            prediction = api / "outputs" / "files" / "prediction"
            prediction.mkdir(parents=True)
            (prediction / "sample_0_predicted_structure.cif").write_text(CIF_TEXT)
            _write_json(
                prediction / "metrics.json",
                {"all_sample_results": [{"metrics": {"structure_confidence": 0.9}}]},
            )
            _write_json(api / "run.json", {"output": {"all_sample_results": []}})

            discovery = discover_prediction_candidates(parent)

        self.assertEqual(
            [
                (candidate.provider, candidate.relative_path)
                for candidate in discovery.candidates
            ],
            [("boltz_api", "api_job"), ("boltz_lab", "lab_job")],
        )

    def test_chai_ranked_directory_scans_json_scores_and_pae(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "chai_job"
            root.mkdir()
            (root / "pred.rank_0.cif").write_text(CIF_TEXT)
            _write_json(
                root / "scores.rank_0.json",
                {
                    "aggregate_score": 0.91,
                    "ptm": 0.82,
                    "iptm": 0.73,
                    "per_chain_ptm": [[0.8, 0.7]],
                    "per_chain_pair_iptm": [[[0.8, 0.4], [0.3, 0.7]]],
                    "has_inter_chain_clashes": False,
                },
            )
            np.save(root / "pae.rank_0.npy", np.array([[0.0, 1.0], [2.0, 0.0]]))

            files = scan_prediction_path(root)
            data = load_prediction_data(files, rank=0, load_pae=True)

        self.assertEqual(files.provider, "chai1")
        self.assertEqual(files.provider_label, "Chai-1 Discovery")
        self.assertEqual(files.n_models, 1)
        self.assertTrue(files.has_pae)
        self.assertFalse(files.has_contact_probs)
        self.assertEqual(files.models[0].display_label, "rank 0")
        np.testing.assert_allclose(data.pae, np.array([[0.0, 1.0], [2.0, 0.0]]))
        self.assertEqual(data.confidence["ranking_score"], 0.91)
        self.assertEqual(data.confidence["chains_ptm"], {"0": 0.8, "1": 0.7})
        self.assertEqual(
            data.confidence["pair_chains_iptm"],
            {"0": {"0": 0.8, "1": 0.4}, "1": {"0": 0.3, "1": 0.7}},
        )
        self.assertIs(data.confidence["has_clash"], False)

    def test_chai_older_ranked_names_support_model_idx_and_pde(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "older_chai"
            root.mkdir()
            (root / "pred.model_idx_3.rank_0.cif").write_text(CIF_TEXT)
            _write_json(
                root / "scores.model_idx_3.rank_0.json",
                {"aggregate_score": 0.5},
            )
            np.save(
                root / "pde.model_idx_3.rank_0.npy",
                np.array([[0.0, 3.0], [3.0, 0.0]], dtype=np.float32),
            )

            files = scan_prediction_path(root)
            lazy = load_prediction_data(files, rank=0, load_pde=False)
            data = load_prediction_data(files, rank=0, load_pde=True)

        self.assertEqual(files.provider, "chai1")
        self.assertTrue(files.has_pde)
        self.assertEqual(files.models[0].metadata["model_idx"], 3)
        self.assertIn("model 3", files.models[0].display_label)
        self.assertIsNone(lazy.pde)
        np.testing.assert_allclose(data.pde, np.array([[0.0, 3.0], [3.0, 0.0]]))

    def test_chai_raw_model_idx_files_sort_by_npz_aggregate_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw_chai"
            root.mkdir()
            for model_idx, score in [(0, 0.25), (1, 0.95)]:
                (root / f"pred.model_idx_{model_idx}.cif").write_text(CIF_TEXT)
                np.savez(
                    root / f"scores.model_idx_{model_idx}.npz",
                    aggregate_score=np.array([score], dtype=np.float32),
                    ptm=np.array([0.8], dtype=np.float32),
                    iptm=np.array([0.7], dtype=np.float32),
                    per_chain_ptm=np.array([[0.6, 0.5]], dtype=np.float32),
                    per_chain_pair_iptm=np.array(
                        [[[0.6, 0.2], [0.1, 0.5]]],
                        dtype=np.float32,
                    ),
                    has_inter_chain_clashes=np.array([False]),
                )

            files = scan_prediction_path(root)
            data = load_prediction_data(files, rank=0)

        self.assertEqual(files.provider, "chai1")
        self.assertEqual(
            [model.metadata["model_idx"] for model in files.models],
            [1, 0],
        )
        self.assertEqual([model.rank for model in files.models], [0, 1])
        self.assertAlmostEqual(data.confidence["aggregate_score"], 0.95, places=6)
        self.assertAlmostEqual(data.confidence["ranking_score"], 0.95, places=6)
        self.assertEqual(set(data.confidence["pair_chains_iptm"]), {"0", "1"})

    def test_protenix_summary_only_server_folder_scans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "protenix_job"
            predictions = root / "job_a" / "seed_123" / "predictions"
            predictions.mkdir(parents=True)
            for sample, score in [(0, 0.7), (1, 0.9)]:
                (predictions / f"job_a_sample_{sample}.cif").write_text(CIF_TEXT)
                _write_json(
                    predictions / f"job_a_summary_confidence_sample_{sample}.json",
                    {
                        "ranking_score": score,
                        "ptm": 0.8,
                        "iptm": 0.75,
                        "gpde": 0.5,
                        "chain_ptm": [0.6, 0.5],
                        "chain_iptm": [0.4, 0.3],
                        "chain_pair_iptm": [[0.0, 0.4], [0.4, 0.0]],
                        "has_clash": False,
                    },
                )

            files = scan_prediction_path(root)
            data = load_prediction_data(files, rank=0, load_pae=True)

        self.assertEqual(files.provider, "protenix")
        self.assertEqual(files.provider_label, "Protenix")
        self.assertEqual(files.name, "job_a")
        self.assertEqual(files.n_models, 2)
        self.assertTrue(files.supports_ensemble)
        self.assertEqual(files.models[0].metadata["sample_rank"], 1)
        self.assertEqual(files.models[0].metadata["seed"], 123)
        self.assertFalse(files.has_pae)
        self.assertFalse(files.has_pde)
        self.assertFalse(files.has_contact_probs)
        self.assertEqual(data.confidence["ranking_score"], 0.9)
        self.assertEqual(data.confidence["chains_ptm"], {"0": 0.6, "1": 0.5})
        self.assertEqual(
            data.confidence["pair_chains_iptm"],
            {"0": {"0": 0.0, "1": 0.4}, "1": {"0": 0.4, "1": 0.0}},
        )
        self.assertIsNone(data.pae)

    def test_protenix_full_data_loads_lazily(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "protenix_full"
            predictions = root / "seed_7" / "predictions"
            predictions.mkdir(parents=True)
            (predictions / "target_sample_0.cif").write_text(CIF_TEXT)
            _write_json(
                predictions / "target_summary_confidence_sample_0.json",
                {"ranking_score": 0.95, "ptm": 0.8, "iptm": 0.75},
            )
            _write_json(
                predictions / "target_full_data_sample_0.json",
                {
                    "atom_plddt": [0.8, 1.0, 0.4],
                    "token_pair_pae": [[0.0, 1.0], [2.0, 0.0]],
                    "token_pair_pde": [[0.0, 3.0], [3.5, 0.0]],
                    "contact_probs": [[1.0, 0.25], [0.25, 1.0]],
                },
            )

            files = scan_prediction_path(root)
            lazy = load_prediction_data(
                files,
                rank=0,
                load_pae=False,
                load_pde=False,
                load_contact_probs=False,
                load_plddt=False,
            )
            data = load_prediction_data(
                files,
                rank=0,
                load_pae=True,
                load_pde=True,
                load_contact_probs=True,
                load_plddt=True,
            )

        self.assertEqual(files.provider, "protenix")
        self.assertTrue(files.has_pae)
        self.assertTrue(files.has_pde)
        self.assertTrue(files.has_contact_probs)
        self.assertTrue(files.has_plddt)
        self.assertIsNone(lazy.pae)
        self.assertIsNone(lazy.pde)
        self.assertIsNone(lazy.contact_probs)
        self.assertIsNone(lazy.plddt)
        np.testing.assert_allclose(data.plddt, np.array([0.9, 0.4], dtype=np.float32))
        np.testing.assert_allclose(data.pae, np.array([[0.0, 1.0], [2.0, 0.0]]))
        np.testing.assert_allclose(data.pde, np.array([[0.0, 3.0], [3.5, 0.0]]))
        np.testing.assert_allclose(
            data.contact_probs, np.array([[1.0, 0.25], [0.25, 1.0]])
        )

    def test_af3_samples_sort_by_ranking_score_without_root_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "hello_fold"
            root.mkdir()
            (root / "hello_fold_model.cif").write_text(CIF_TEXT)
            _write_json(
                root / "hello_fold_summary_confidences.json", {"ranking_score": 9.0}
            )
            for sample, score in [(0, 0.1), (1, 0.9)]:
                sample_dir = root / f"seed-1234_sample-{sample}"
                sample_dir.mkdir()
                stem = f"hello_fold_seed-1234_sample-{sample}"
                (sample_dir / f"{stem}_model.cif").write_text(CIF_TEXT)
                _write_json(
                    sample_dir / f"{stem}_summary_confidences.json",
                    {"ranking_score": score},
                )
                _write_json(sample_dir / f"{stem}_confidences.json", {})

            files = scan_prediction_path(root)

        self.assertEqual(files.provider, "alphafold3")
        self.assertEqual(files.n_models, 2)
        self.assertEqual(files.models[0].metadata["sample"], 1)
        self.assertEqual(files.models[1].metadata["sample"], 0)
        self.assertIn("seed 1234 sample 1", files.models[0].display_label)

    def test_af3_samples_accept_truncated_older_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "older_af3"
            root.mkdir()
            (root / "older_af3_model.cif").write_text(CIF_TEXT)
            _write_json(
                root / "older_af3_summary_confidences.json", {"ranking_score": 9.0}
            )
            for sample, score in [(0, 0.2), (1, 0.8)]:
                sample_dir = root / f"seed-999_sample-{sample}"
                sample_dir.mkdir()
                (sample_dir / "model.cif").write_text(CIF_TEXT)
                _write_json(
                    sample_dir / "summary_confidences.json",
                    {"ranking_score": score},
                )
                _write_json(sample_dir / "confidences.json", {})

            files = scan_prediction_path(root)

        self.assertEqual(files.provider, "alphafold3")
        self.assertEqual(files.n_models, 2)
        self.assertEqual(files.models[0].metadata["sample"], 1)
        self.assertEqual(files.models[0].structure_path.name, "model.cif")
        self.assertEqual(files.models[0].confidence_path.name, "confidences.json")
        self.assertEqual(files.models[0].summary_path.name, "summary_confidences.json")

    def test_af3_samples_prefer_ranking_scores_csv_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "multi_seed"
            root.mkdir()
            (root / "multi_seed_model.cif").write_text(CIF_TEXT)
            (root / "ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n111,0,0.75\n222,0,0.95\n111,1,0.85\n"
            )
            for seed, sample in [(111, 0), (222, 0), (111, 1)]:
                sample_dir = root / f"seed-{seed}_sample-{sample}"
                sample_dir.mkdir()
                (sample_dir / "model.cif").write_text(CIF_TEXT)
                _write_json(
                    sample_dir / "summary_confidences.json",
                    {"ranking_score": 0.5},
                )
                _write_json(sample_dir / "confidences.json", {})

            files = scan_prediction_path(root)

        self.assertEqual(
            [
                (model.metadata["seed"], model.metadata["sample"])
                for model in files.models
            ],
            [(222, 0), (111, 1), (111, 0)],
        )

    def test_af3_samples_accept_prefixed_ranking_scores_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "prefixed"
            root.mkdir()
            (root / "prefixed_model.cif").write_text(CIF_TEXT)
            (root / "prefixed_ranking_scores.csv").write_text(
                "seed,sample,ranking_score\n1,0,0.1\n1,1,0.9\n"
            )
            for sample in [0, 1]:
                sample_dir = root / f"seed-1_sample-{sample}"
                sample_dir.mkdir()
                (sample_dir / "model.cif").write_text(CIF_TEXT)
                _write_json(
                    sample_dir / "summary_confidences.json",
                    {"ranking_score": 0.5},
                )
                _write_json(sample_dir / "confidences.json", {})

            files = scan_prediction_path(root)

        self.assertEqual(
            [
                (model.metadata["seed"], model.metadata["sample"])
                for model in files.models
            ],
            [(1, 1), (1, 0)],
        )

    def test_af3_atom_plddts_collapse_to_token_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "af3"
            root.mkdir()
            (root / "af3_model.cif").write_text(CIF_TEXT)
            _write_json(root / "af3_summary_confidences.json", {"ptm": 0.5})
            _write_json(
                root / "af3_confidences.json",
                {
                    "atom_plddts": [80.0, 100.0, 40.0],
                    "pae": [[0.0, 1.0], [2.0, 0.0]],
                    "contact_probs": [[1.0, 0.25], [0.25, 1.0]],
                },
            )

            files = scan_prediction_path(root)
            data = load_prediction_data(files, load_pae=True, load_contact_probs=True)

        self.assertEqual(files.provider, "alphafold3")
        np.testing.assert_allclose(data.plddt, np.array([0.9, 0.4], dtype=np.float32))
        np.testing.assert_allclose(data.pae, np.array([[0.0, 1.0], [2.0, 0.0]]))
        np.testing.assert_allclose(
            data.contact_probs, np.array([[1.0, 0.25], [0.25, 1.0]])
        )

    def test_af3_server_flat_ranked_files_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "server_job"
            root.mkdir()
            for rank in [0, 1]:
                (root / f"server_job_model_{rank}.cif").write_text(CIF_TEXT)
                _write_json(
                    root / f"server_job_summary_confidences_{rank}.json",
                    {"ranking_score": 1.0 - rank, "chain_iptm": [0.5]},
                )
                _write_json(
                    root / f"server_job_full_data_{rank}.json",
                    {
                        "atom_plddts": [80.0, 100.0, 40.0],
                        "pae": [[0.0, 1.0], [2.0, 0.0]],
                        "contact_probs": [[1.0, 0.25], [0.25, 1.0]],
                    },
                )

            files = scan_prediction_path(root)
            data = load_prediction_data(
                files, rank=0, load_pae=True, load_contact_probs=True
            )

        self.assertEqual(files.provider, "af3_server")
        self.assertEqual(files.n_models, 2)
        self.assertTrue(files.supports_ensemble)
        self.assertEqual(files.models[0].display_label, "rank 0")
        self.assertEqual(files.models[0].object_name, "server_job_model_0")
        np.testing.assert_allclose(data.plddt, np.array([0.9, 0.4], dtype=np.float32))
        np.testing.assert_allclose(data.pae, np.array([[0.0, 1.0], [2.0, 0.0]]))
        np.testing.assert_allclose(
            data.contact_probs, np.array([[1.0, 0.25], [0.25, 1.0]])
        )

    def test_af3_server_accepts_truncated_ranked_json_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "server_job"
            root.mkdir()
            (root / "server_job_model_0.cif").write_text(CIF_TEXT)
            _write_json(root / "summary_confidences_0.json", {"ranking_score": 1.0})
            _write_json(root / "full_data_0.json", {})

            files = scan_prediction_path(root)

        self.assertEqual(files.provider, "af3_server")
        self.assertEqual(
            files.models[0].summary_path.name, "summary_confidences_0.json"
        )
        self.assertEqual(files.models[0].confidence_path.name, "full_data_0.json")

    def test_wrapped_prediction_zip_scans_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "server_job"
            root.mkdir()
            (root / "server_job_model_0.cif").write_text(CIF_TEXT)
            _write_json(
                root / "server_job_summary_confidences_0.json",
                {"ranking_score": 1.0},
            )
            _write_json(
                root / "server_job_full_data_0.json",
                {
                    "atom_plddts": [80.0, 100.0, 40.0],
                    "pae": [[0.0, 1.0], [2.0, 0.0]],
                },
            )
            archive = Path(tmp) / "wrapped.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                for path in root.iterdir():
                    zf.write(path, Path(root.name) / path.name)

            files = scan_prediction_path(archive)
            data = load_prediction_data(files, rank=0, load_pae=True)

        self.assertEqual(files.provider, "af3_server")
        self.assertEqual(files.input_path.name, "wrapped.zip")
        self.assertEqual(files.pred_dir.name, "server_job")
        self.assertEqual(files.n_models, 1)
        np.testing.assert_allclose(data.plddt, np.array([0.9, 0.4], dtype=np.float32))
        np.testing.assert_allclose(data.pae, np.array([[0.0, 1.0], [2.0, 0.0]]))

    def test_boltz_api_tar_gz_scans_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive.tar.gz"
            with tarfile.open(archive, _tar_write_mode(archive)) as tf:
                _add_tar_text(
                    tf,
                    "prediction/sample_0_predicted_structure.cif",
                    CIF_TEXT,
                )
                _add_tar_bytes(
                    tf,
                    "prediction/sample_0_pae.npz",
                    _npz_bytes(pae=np.array([[0.0, 1.0], [1.0, 0.0]])),
                )
                _add_tar_json(
                    tf,
                    "prediction/metrics.json",
                    {
                        "all_sample_results": [
                            {"metrics": {"structure_confidence": 0.91, "ptm": 0.8}}
                        ]
                    },
                )

            files = scan_prediction_path(archive)
            data = load_prediction_data(files, rank=0, load_pae=True)

        self.assertEqual(files.provider, "boltz_api")
        self.assertEqual(files.input_path.name, "archive.tar.gz")
        self.assertEqual(files.name, "prediction")
        self.assertEqual(files.pred_dir.name, "prediction")
        np.testing.assert_allclose(data.pae, np.array([[0.0, 1.0], [1.0, 0.0]]))
        self.assertEqual(data.confidence["confidence_score"], 0.91)

    def test_tgz_prediction_archive_scans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "server_job.tgz"
            with tarfile.open(archive, _tar_write_mode(archive)) as tf:
                _add_tar_text(tf, "server_job/server_job_model_0.cif", CIF_TEXT)
                _add_tar_json(
                    tf,
                    "server_job/server_job_summary_confidences_0.json",
                    {"ranking_score": 1.0},
                )
                _add_tar_json(tf, "server_job/server_job_full_data_0.json", {})

            files = scan_prediction_path(archive)

        self.assertEqual(files.provider, "af3_server")
        self.assertEqual(files.input_path.name, "server_job.tgz")
        self.assertEqual(files.pred_dir.name, "server_job")

    def test_plain_tar_prediction_archive_scans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "server_job.tar"
            with tarfile.open(archive, _tar_write_mode(archive)) as tf:
                _add_tar_text(tf, "server_job_model_0.cif", CIF_TEXT)
                _add_tar_json(
                    tf,
                    "server_job_summary_confidences_0.json",
                    {"ranking_score": 1.0},
                )
                _add_tar_json(tf, "server_job_full_data_0.json", {})

            files = scan_prediction_path(archive)

        self.assertEqual(files.provider, "af3_server")
        self.assertEqual(files.input_path.name, "server_job.tar")
        self.assertEqual(files.name, "server_job")
        self.assertEqual(files.models[0].object_name, "server_job_model_0")

    def test_wrapped_prediction_folder_scans_child_provider_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp) / "download"
            root = wrapper / "server_job"
            root.mkdir(parents=True)
            (root / "server_job_model_0.cif").write_text(CIF_TEXT)
            _write_json(
                root / "server_job_summary_confidences_0.json",
                {"ranking_score": 1.0},
            )
            _write_json(root / "server_job_full_data_0.json", {})

            files = scan_prediction_path(wrapper)

        self.assertEqual(files.provider, "af3_server")
        self.assertEqual(files.input_path.name, "download")
        self.assertEqual(files.pred_dir.name, "server_job")
        self.assertEqual(files.n_models, 1)

    def test_discover_wrapped_prediction_folder_returns_one_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp) / "download"
            _write_af3_server_job(wrapper / "server_job")

            discovery = discover_prediction_candidates(wrapper)
            files = discovery.scan(discovery.candidates[0])

        self.assertEqual(
            [c.relative_path for c in discovery.candidates], ["server_job"]
        )
        self.assertEqual(discovery.candidates[0].provider, "af3_server")
        self.assertEqual(files.provider, "af3_server")
        self.assertEqual(files.input_path.name, "download")
        self.assertEqual(files.pred_dir.name, "server_job")

    def test_discover_multiple_prediction_folders_sorts_by_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp) / "download"
            _write_af3_server_job(wrapper / "b_job")
            _write_af3_server_job(wrapper / "a_job")

            discovery = discover_prediction_candidates(wrapper)

        self.assertEqual(
            [c.relative_path for c in discovery.candidates], ["a_job", "b_job"]
        )
        self.assertEqual(
            [c.provider for c in discovery.candidates], ["af3_server", "af3_server"]
        )

    def test_discover_protenix_parent_aggregate_prefers_child_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp) / "protenix_jobs"
            _write_protenix_job(wrapper / "b_job")
            _write_protenix_job(wrapper / "a_job")

            discovery = discover_prediction_candidates(wrapper)
            files = discovery.scan(discovery.candidates[1])

        self.assertEqual(
            [c.relative_path for c in discovery.candidates], ["a_job", "b_job"]
        )
        self.assertEqual(
            [c.provider for c in discovery.candidates], ["protenix", "protenix"]
        )
        self.assertEqual(files.provider, "protenix")
        self.assertEqual(files.name, "b_job")
        self.assertEqual(files.input_path.name, "protenix_jobs")

    def test_discover_protenix_nested_job_hides_duplicate_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp) / "jobs"
            _write_protenix_job(
                wrapper / "SbPZS_Protenix" / "protenix_prediction_2a4d8576"
            )

            discovery = discover_prediction_candidates(wrapper)
            files = discovery.scan(discovery.candidates[0])

        self.assertEqual(
            [c.relative_path for c in discovery.candidates],
            ["SbPZS_Protenix/protenix_prediction_2a4d8576"],
        )
        self.assertEqual(discovery.candidates[0].provider, "protenix")
        self.assertEqual(files.provider, "protenix")
        self.assertEqual(files.pred_dir.name, "protenix_prediction_2a4d8576")

    def test_discover_multiple_prediction_zip_transfers_extraction_lifetime(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "jobs.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                for job in ["b_job", "a_job"]:
                    zf.writestr(f"{job}/{job}_model_0.cif", CIF_TEXT)
                    zf.writestr(
                        f"{job}/{job}_summary_confidences_0.json",
                        json.dumps({"ranking_score": 1.0}),
                    )
                    zf.writestr(f"{job}/{job}_full_data_0.json", json.dumps({}))

            discovery = discover_prediction_candidates(archive)
            files = discovery.scan(discovery.candidates[0])
            structure_path = files.structure_path(0)

            self.assertEqual(
                [c.relative_path for c in discovery.candidates],
                ["a_job", "b_job"],
            )
            self.assertIsNotNone(files._temporary_directory)
            self.assertIsNone(discovery._temporary_directory)
            self.assertTrue(structure_path.exists())

    def test_discover_multiple_prediction_tar_transfers_extraction_lifetime(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "jobs.tar"
            with tarfile.open(archive, _tar_write_mode(archive)) as tf:
                for job in ["b_job", "a_job"]:
                    _add_tar_text(tf, f"{job}/{job}_model_0.cif", CIF_TEXT)
                    _add_tar_json(
                        tf,
                        f"{job}/{job}_summary_confidences_0.json",
                        {"ranking_score": 1.0},
                    )
                    _add_tar_json(tf, f"{job}/{job}_full_data_0.json", {})

            discovery = discover_prediction_candidates(archive)
            files = discovery.scan(discovery.candidates[0])
            structure_path = files.structure_path(0)

            self.assertEqual(
                [c.relative_path for c in discovery.candidates],
                ["a_job", "b_job"],
            )
            self.assertIsNotNone(files._temporary_directory)
            self.assertIsNone(discovery._temporary_directory)
            self.assertTrue(structure_path.exists())

    def test_invalid_tar_archive_reports_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "invalid.tar"
            archive.write_text("not a tar archive")

            with self.assertRaisesRegex(ValueError, "Invalid archive"):
                scan_prediction_path(archive)

    def test_tar_archive_rejects_unsafe_paths(self) -> None:
        unsafe_names = [
            "../evil.cif",
            "/evil.cif",
            "C:/evil.cif",
            r"\\server\share\evil.cif",
        ]
        for unsafe_name in unsafe_names:
            with self.subTest(unsafe_name=unsafe_name):
                with tempfile.TemporaryDirectory() as tmp:
                    archive = Path(tmp) / "unsafe.tar"
                    with tarfile.open(archive, _tar_write_mode(archive)) as tf:
                        _add_tar_text(tf, unsafe_name, CIF_TEXT)

                    with self.assertRaisesRegex(ValueError, "Unsafe path"):
                        scan_prediction_path(archive)

    def test_tar_archive_rejects_links_and_special_members(self) -> None:
        unsafe_members = [
            ("link", tarfile.SYMTYPE, "prediction/sample_0_predicted_structure.cif"),
            (
                "hardlink",
                tarfile.LNKTYPE,
                "prediction/sample_0_predicted_structure.cif",
            ),
            ("fifo", tarfile.FIFOTYPE, ""),
        ]
        for member_name, member_type, linkname in unsafe_members:
            with self.subTest(member_name=member_name):
                with tempfile.TemporaryDirectory() as tmp:
                    archive = Path(tmp) / "unsafe.tar"
                    with tarfile.open(archive, _tar_write_mode(archive)) as tf:
                        info = tarfile.TarInfo(member_name)
                        info.type = member_type
                        info.linkname = linkname
                        tf.addfile(info)

                    with self.assertRaisesRegex(
                        ValueError,
                        "Refusing to extract link|Unsupported member type",
                    ):
                        scan_prediction_path(archive)

    def test_flat_prediction_zip_uses_zip_stem_as_provider_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "fold_ss_fusion.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("fold_ss_fusion_model_0.cif", CIF_TEXT)
                zf.writestr(
                    "fold_ss_fusion_summary_confidences_0.json",
                    json.dumps({"ranking_score": 1.0}),
                )
                zf.writestr("fold_ss_fusion_full_data_0.json", json.dumps({}))

            files = scan_prediction_path(archive)

        self.assertEqual(files.provider, "af3_server")
        self.assertEqual(files.name, "fold_ss_fusion")
        self.assertEqual(files.models[0].object_name, "fold_ss_fusion_model_0")
        self.assertEqual(
            files.models[0].structure_path.name,
            "fold_ss_fusion_model_0.cif",
        )

    def test_chai_prediction_zip_scans_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "chai_job.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("chai_job/pred.rank_0.cif", CIF_TEXT)
                zf.writestr(
                    "chai_job/scores.rank_0.json",
                    json.dumps({"aggregate_score": 1.0}),
                )
                with tempfile.NamedTemporaryFile(suffix=".npy") as fh:
                    np.save(fh, np.array([[0.0, 1.0], [1.0, 0.0]]))
                    fh.seek(0)
                    zf.writestr("chai_job/pae.rank_0.npy", fh.read())

            files = scan_prediction_path(archive)
            data = load_prediction_data(files, rank=0, load_pae=True)

        self.assertEqual(files.provider, "chai1")
        self.assertEqual(files.input_path.name, "chai_job.zip")
        self.assertEqual(files.pred_dir.name, "chai_job")
        np.testing.assert_allclose(data.pae, np.array([[0.0, 1.0], [1.0, 0.0]]))

    def test_protenix_prediction_zip_scans_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "protenix.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(
                    "bundle/seed_1/predictions/target_sample_0.cif",
                    CIF_TEXT,
                )
                zf.writestr(
                    "bundle/seed_1/predictions/target_summary_confidence_sample_0.json",
                    json.dumps({"ranking_score": 1.0, "chain_ptm": [0.9]}),
                )

            files = scan_prediction_path(archive)
            data = load_prediction_data(files, rank=0)

        self.assertEqual(files.provider, "protenix")
        self.assertEqual(files.input_path.name, "protenix.zip")
        self.assertEqual(files.name, "target")
        self.assertEqual(data.confidence["chains_ptm"], {"0": 0.9})

    def test_af3_full_confidence_json_is_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "af3"
            root.mkdir()
            (root / "af3_model.cif").write_text(CIF_TEXT)
            _write_json(root / "af3_summary_confidences.json", {"ptm": 0.5})
            (root / "af3_confidences.json").write_text("{not-json")

            files = scan_prediction_path(root)
            data = load_prediction_data(
                files,
                load_pae=False,
                load_contact_probs=False,
                load_plddt=False,
            )

            self.assertEqual(data.summary_confidence["ptm"], 0.5)
            with self.assertRaises(json.JSONDecodeError):
                load_prediction_data(files, load_pae=True, load_plddt=False)


if __name__ == "__main__":
    unittest.main()
