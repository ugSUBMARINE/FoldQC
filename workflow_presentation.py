"""Small typed presentation helpers shared by GUI-neutral workflows."""

from __future__ import annotations

from .presentation import ChoiceOption, ChoiceRequest, Notice


def _present(owner, severity: str, title: str, message: str) -> None:
    owner._presenter.present_notice(
        Notice(
            "workflow_notice",
            str(message),
            severity=severity,
            title=str(title),
        )
    )


def present_information(owner, title: str, message: str) -> None:
    _present(owner, "information", title, message)


def present_warning(owner, title: str, message: str) -> None:
    _present(owner, "warning", title, message)


def present_error(owner, title: str, message: str) -> None:
    _present(owner, "error", title, message)


def confirm(owner, title: str, message: str) -> bool:
    return (
        owner._presenter.choose(
            ChoiceRequest(
                "workflow_confirmation",
                str(title),
                str(message),
                (
                    ChoiceOption("yes", "Yes"),
                    ChoiceOption("cancel", "Cancel", "reject"),
                ),
                default_key="cancel",
            )
        )
        == "yes"
    )
