"""Small explicit resource-ownership contracts."""

from __future__ import annotations

import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Closeable(Protocol):
    """An object whose owned external resources can be released explicitly."""

    def close(self) -> None: ...


class TemporaryDirectoryOwner:
    """Idempotently own and remove one temporary directory tree."""

    def __init__(self, root: Path) -> None:
        resolved = root.resolve()
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if resolved == temporary_root or temporary_root not in resolved.parents:
            raise ValueError(
                "TemporaryDirectoryOwner requires a child of the system "
                "temporary directory."
            )
        self.root = resolved
        self._finalizer = weakref.finalize(
            self,
            shutil.rmtree,
            resolved,
            ignore_errors=True,
        )

    def close(self) -> None:
        self._finalizer()
