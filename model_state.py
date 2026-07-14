"""Canonical per-rank prediction state independent of Qt and PyMOL."""

from __future__ import annotations

from dataclasses import dataclass

from .loader_models import PredictionData
from .token_map import TokenMap


@dataclass
class ModelState:
    """Loaded prediction data and immutable token map for one model rank."""

    rank: int
    data: PredictionData
    token_map: TokenMap

    def __post_init__(self) -> None:
        self._validate_data(self.data)

    def _validate_data(self, data: PredictionData) -> None:
        if int(data.rank) != self.rank:
            raise ValueError(
                f"ModelState rank {self.rank} cannot contain rank {data.rank} data."
            )

    def replace_data(self, data: PredictionData) -> None:
        """Replace lazily loaded data while retaining this state's token map."""
        self._validate_data(data)
        self.data = data

    def replace(self, data: PredictionData, token_map: TokenMap) -> None:
        """Replace both loaded data and token map for a freshly loaded rank."""
        self._validate_data(data)
        self.data = data
        self.token_map = token_map
