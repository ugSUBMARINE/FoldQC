from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.loader_models import (  # noqa: E402
    ModelFiles,
    PredictionCandidate,
    PredictionData,
    PredictionDiscovery,
    PredictionFiles,
)
from FoldQC.providers.base import BaseProvider, LoadOptions  # noqa: E402
from FoldQC.providers.registry import (  # noqa: E402
    BUILTIN_PROVIDERS,
    ProviderRegistry,
)


class _FakeProvider(BaseProvider):
    supports_ensemble = True

    def __init__(self, key: str, label: str, matches: bool) -> None:
        self.key = key
        self.label = label
        self.matches = matches

    def detect(self, path: Path) -> bool:
        return self.matches

    def scan(self, path: Path) -> PredictionFiles:
        return PredictionFiles(path.name, path, provider=self.key)


def test_builtin_registry_preserves_detection_precedence() -> None:
    assert [provider.key for provider in BUILTIN_PROVIDERS.providers] == [
        "af3_server",
        "alphafold3",
        "chai1",
        "protenix",
        "boltz_api",
        "boltz_lab",
        "boltz",
        "structure_only",
    ]


def test_registry_uses_first_matching_provider() -> None:
    first = _FakeProvider("first", "First", True)
    second = _FakeProvider("second", "Second", True)
    registry = ProviderRegistry((first, second))

    assert registry.detect(Path("prediction")) is first


def test_registry_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="unique"):
        ProviderRegistry(
            (
                _FakeProvider("duplicate", "First", False),
                _FakeProvider("duplicate", "Second", False),
            )
        )


def test_registry_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        BUILTIN_PROVIDERS.get("missing")


def test_model_types_live_in_loader_models() -> None:
    assert ModelFiles.__module__ == "FoldQC.loader_models"
    assert PredictionFiles.__module__ == "FoldQC.loader_models"
    assert PredictionData.__module__ == "FoldQC.loader_models"
    assert PredictionCandidate.__module__ == "FoldQC.loader_models"
    assert PredictionDiscovery.__module__ == "FoldQC.loader_models"


def test_provider_contracts_live_in_internal_provider_modules() -> None:
    assert LoadOptions.__module__ == "FoldQC.providers.base"
    assert ProviderRegistry.__module__ == "FoldQC.providers.registry"
