"""Qt-independent contracts for presenting FoldQC workflow outcomes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

NoticeSeverity = Literal["information", "warning", "error"]
ChoiceRole = Literal["accept", "reject", "destructive", "help"]


@dataclass(frozen=True)
class Notice:
    """One user-facing workflow notice with stable diagnostic identity."""

    code: str
    message: str
    severity: NoticeSeverity = "warning"
    title: str = "FoldQC"
    affected_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChoiceOption:
    key: str
    label: str
    role: ChoiceRole = "accept"


@dataclass(frozen=True)
class ChoiceRequest:
    code: str
    title: str
    message: str
    options: tuple[ChoiceOption, ...]
    default_key: str | None = None

    def __post_init__(self) -> None:
        keys = tuple(option.key for option in self.options)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Choice requests require unique, non-empty options.")
        if self.default_key is not None and self.default_key not in keys:
            raise ValueError("Choice request default must name an option.")


@dataclass(frozen=True)
class SelectionItem:
    key: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class SelectionRequest:
    code: str
    title: str
    message: str
    items: tuple[SelectionItem, ...]
    default_key: str | None = None

    def __post_init__(self) -> None:
        keys = tuple(item.key for item in self.items)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Selection requests require unique, non-empty items.")
        if self.default_key is not None and self.default_key not in keys:
            raise ValueError("Selection request default must name an item.")


@dataclass(frozen=True)
class ProgressRequest:
    operation_id: str
    title: str
    label: str
    delay_ms: int = 0
    cancellable: bool = False

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("Progress requests require an operation id.")
        if self.delay_ms < 0:
            raise ValueError("Progress delay cannot be negative.")


@dataclass(frozen=True)
class PreparedPlot:
    """A fully prepared figure whose ownership can transfer to a presenter."""

    figure: object
    title: str


@runtime_checkable
class PresentationPort(Protocol):
    def present_notice(self, notice: Notice) -> None: ...

    def choose(self, request: ChoiceRequest) -> str | None: ...

    def select_item(self, request: SelectionRequest) -> str | None: ...

    def start_progress(
        self,
        request: ProgressRequest,
        on_cancel: Callable[[], None] | None = None,
    ) -> None: ...

    def update_progress(self, operation_id: str, label: str) -> None: ...

    def finish_progress(self, operation_id: str) -> None: ...

    def set_window_title(self, title: str) -> None: ...

    def show_statistics(self, text: str) -> None: ...

    def show_plot(self, prepared: PreparedPlot) -> None: ...
