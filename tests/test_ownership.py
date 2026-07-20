from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FoldQC.loader_discovery import scan_prediction_candidate
from FoldQC.loader_models import (
    PredictionCandidate,
    PredictionDiscovery,
    PredictionFiles,
    ProviderInfo,
)
from FoldQC.ownership import TemporaryDirectoryOwner
from FoldQC.providers.registry import BUILTIN_PROVIDERS


class _Owner:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _Provider:
    def __init__(self, files: PredictionFiles, *, fail: bool = False) -> None:
        self.files = files
        self.fail = fail

    def detect(self, _path: Path) -> bool:
        return True

    def scan(self, _path: Path) -> PredictionFiles:
        if self.fail:
            raise ValueError("scan failed")
        return self.files


def _discovery(owner: _Owner):
    info = ProviderInfo("test", "Test provider")
    candidate = PredictionCandidate(Path("/tmp/prediction"), info, ".")
    return (
        PredictionDiscovery(Path("/tmp/input.zip"), (candidate,), owner),
        candidate,
    )


def test_prediction_files_close_is_idempotent() -> None:
    owner = _Owner()
    files = PredictionFiles(
        "prediction", Path("/tmp/prediction"), ProviderInfo("test", "Test")
    )
    files.adopt_resource_owner(owner)

    files.close()
    files.close()

    assert owner.close_count == 1


def test_temporary_directory_owner_rejects_non_temporary_roots() -> None:
    with pytest.raises(ValueError, match="system temporary directory"):
        TemporaryDirectoryOwner(Path("/"))


def test_discovery_transfers_owner_only_after_successful_scan() -> None:
    owner = _Owner()
    discovery, candidate = _discovery(owner)
    files = PredictionFiles("prediction", candidate.path, provider=candidate.provider)
    with mock.patch.object(BUILTIN_PROVIDERS, "get", return_value=_Provider(files)):
        result = scan_prediction_candidate(discovery, candidate)

    assert result is files
    assert result.input_path == discovery.input_path
    assert discovery._resource_owner is None
    result.close()
    assert owner.close_count == 1


def test_failed_scan_leaves_discovery_owner_intact() -> None:
    owner = _Owner()
    discovery, candidate = _discovery(owner)
    files = PredictionFiles("prediction", candidate.path, provider=candidate.provider)
    with (
        mock.patch.object(
            BUILTIN_PROVIDERS, "get", return_value=_Provider(files, fail=True)
        ),
        pytest.raises(ValueError, match="scan failed"),
    ):
        scan_prediction_candidate(discovery, candidate)

    assert discovery._resource_owner is owner
    discovery.close()
    discovery.close()
    assert owner.close_count == 1


def test_unknown_candidate_does_not_transfer_owner() -> None:
    owner = _Owner()
    discovery, candidate = _discovery(owner)
    other = PredictionCandidate(Path("/tmp/other"), candidate.provider, "other")
    with pytest.raises(ValueError, match="Unknown prediction candidate"):
        scan_prediction_candidate(discovery, other)
    assert discovery._resource_owner is owner
