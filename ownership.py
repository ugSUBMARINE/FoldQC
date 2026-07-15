"""Small explicit resource-ownership contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Closeable(Protocol):
    """An object whose owned external resources can be released explicitly."""

    def close(self) -> None: ...
