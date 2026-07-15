"""Single-structure provider."""

from pathlib import Path

from ..confidence import STRUCTURE_CONFIDENCE_SUMMARY
from ..loader_models import ModelFiles, PredictionFiles
from ..loader_utils import STRUCTURE_SUFFIXES, _safe_object_name
from .base import BaseProvider


class StructureProvider(BaseProvider):
    key, label = "structure_only", "Structure-only"
    supports_ensemble = False
    confidence_summary = STRUCTURE_CONFIDENCE_SUMMARY

    def detect(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES

    def scan(self, path: Path) -> PredictionFiles:
        if path.suffix.lower() not in STRUCTURE_SUFFIXES:
            raise ValueError(f"Unsupported structure file format: {path.suffix}")
        name = path.stem
        files = self.prediction_files(name=name, pred_dir=path.parent)
        files.input_path = path
        files.models = [
            ModelFiles(
                rank=0,
                structure_path=path,
                display_label=path.name,
                object_name=_safe_object_name(name),
                capabilities=frozenset({"plddt"}),
            )
        ]
        return files
