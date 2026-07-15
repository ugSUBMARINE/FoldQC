"""Single-structure provider."""

from pathlib import Path

from ..loader_models import ModelFiles, PredictionFiles
from ..loader_utils import STRUCTURE_SUFFIXES, _safe_object_name
from .base import BaseProvider


class StructureProvider(BaseProvider):
    key, label = "structure_only", "Structure-only"
    supports_ensemble = False

    def detect(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES

    def scan(self, path: Path) -> PredictionFiles:
        if path.suffix.lower() not in STRUCTURE_SUFFIXES:
            raise ValueError(f"Unsupported structure file format: {path.suffix}")
        name = path.stem
        return PredictionFiles(
            name=name,
            pred_dir=path.parent,
            provider=self.key,
            input_path=path,
            models=[
                ModelFiles(
                    rank=0,
                    structure_path=path,
                    display_label=path.name,
                    object_name=_safe_object_name(name),
                    capabilities=frozenset({"plddt"}),
                )
            ],
        )
