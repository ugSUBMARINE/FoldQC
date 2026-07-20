"""Deterministic registry of FoldQC's built-in providers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .alphafold import AF3ServerProvider, AlphaFold3Provider
from .alphafold_database import AlphaFoldDatabaseProvider
from .base import BaseProvider
from .boltz import BoltzAPIProvider, BoltzLabProvider, BoltzProvider
from .chai import ChaiProvider
from .openfold import OpenFold3Provider
from .protenix import ProtenixProvider
from .structure import StructureProvider


class ProviderRegistry:
    def __init__(self, providers: Iterable[BaseProvider]) -> None:
        self.providers = tuple(providers)
        keys = [provider.key for provider in self.providers]
        if len(keys) != len(set(keys)):
            raise ValueError("Provider keys must be unique.")
        self._by_key = {provider.key: provider for provider in self.providers}

    def get(self, key: str) -> BaseProvider:
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise ValueError(f"Unsupported provider: {key}") from exc

    def detect(self, path: Path) -> BaseProvider | None:
        for provider in self.providers:
            if provider.detect(path):
                return provider
        return None

    @property
    def directory_labels(self) -> tuple[str, ...]:
        return tuple(p.label for p in self.providers if p.key != "structure_only")


BUILTIN_PROVIDERS = ProviderRegistry(
    (
        AlphaFoldDatabaseProvider(),
        AF3ServerProvider(),
        AlphaFold3Provider(),
        OpenFold3Provider(),
        ChaiProvider(),
        ProtenixProvider(),
        BoltzAPIProvider(),
        BoltzLabProvider(),
        BoltzProvider(),
        StructureProvider(),
    )
)
