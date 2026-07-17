"""Threshold selections derived from the most recently colored metric arrays."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np

from .gui_services import (
    DialogViewPort,
    ObjectTokenSelection,
    StatisticsSelectionTarget,
    StatisticsSelectionViewState,
    ViewerPort,
)
from .presentation import Notice, PresentationPort

ThresholdComparison = Literal["ge", "le"]


def threshold_selection_name(metric_key: str, comparison: ThresholdComparison) -> str:
    """Return a stable PyMOL-safe name for one metric/comparison pair."""
    safe_metric = re.sub(r"[^A-Za-z0-9_]", "_", metric_key).strip("_")
    if not safe_metric:
        safe_metric = "metric"
    return f"foldqc_{safe_metric}_{comparison}"


class StatisticsSelectionService:
    """Retain colored values and turn thresholds into named viewer selections."""

    def __init__(
        self,
        viewer: ViewerPort,
        presenter: PresentationPort,
        view: DialogViewPort,
    ) -> None:
        self._viewer = viewer
        self._presenter = presenter
        self._view = view
        self._metric_key: str | None = None
        self._targets: tuple[StatisticsSelectionTarget, ...] = ()
        self._view_state = StatisticsSelectionViewState()

    def set_coloring_result(
        self,
        metric_key: str,
        targets: tuple[StatisticsSelectionTarget, ...],
    ) -> None:
        """Capture the exact values and scope represented by the statistics panel."""
        finite = [target.values[np.isfinite(target.values)] for target in targets]
        pooled = (
            np.concatenate([values for values in finite if values.size])
            if any(values.size for values in finite)
            else np.array([], dtype=np.float32)
        )
        if not targets or pooled.size == 0:
            self.clear("No finite metric values are available for selection.")
            return

        self._metric_key = metric_key
        self._targets = targets
        minimum = float(np.min(pooled))
        maximum = float(np.max(pooled))
        self._view_state = StatisticsSelectionViewState(
            enabled=True,
            threshold=float(np.median(pooled)),
            minimum=minimum,
            maximum=maximum,
            status_text=f"Range: {minimum:g} to {maximum:g}",
        )
        self._view.set_statistics_selection(self._view_state)

    def clear(self, message: str = "Apply a metric coloring first.") -> None:
        self._metric_key = None
        self._targets = ()
        self._view_state = StatisticsSelectionViewState(status_text=message)
        self._view.set_statistics_selection(self._view_state)

    def select(self, comparison: ThresholdComparison, threshold: float) -> str | None:
        """Create a consolidated selection from canonical token identities."""
        if comparison not in {"ge", "le"}:
            raise ValueError(f"Unsupported threshold comparison: {comparison!r}")
        if self._metric_key is None or not self._targets:
            return None
        threshold = float(threshold)
        if not np.isfinite(threshold):
            self._presenter.present_notice(
                Notice(
                    "statistics_threshold_invalid",
                    "The selection threshold must be a finite number.",
                    severity="error",
                    title="FoldQC - invalid threshold",
                )
            )
            return None

        selections: list[ObjectTokenSelection] = []
        selected_count = 0
        for target in self._targets:
            finite = np.isfinite(target.values)
            mask = (
                finite & (target.values >= threshold)
                if comparison == "ge"
                else finite & (target.values <= threshold)
            )
            indices = tuple(int(index) for index in np.flatnonzero(mask))
            selected_count += len(indices)
            selections.append(
                ObjectTokenSelection(target.obj_name, target.token_map, indices)
            )

        selection_name = threshold_selection_name(self._metric_key, comparison)
        try:
            self._viewer.update_object_token_selection(selection_name, selections)
        except Exception as exc:
            self._presenter.present_notice(
                Notice(
                    "statistics_selection_failed",
                    f"Could not create the threshold selection.\n\n{exc}",
                    severity="error",
                    title="FoldQC - selection error",
                )
            )
            return None

        operator = "≥" if comparison == "ge" else "≤"
        self._view_state = StatisticsSelectionViewState(
            enabled=True,
            threshold=threshold,
            minimum=self._view_state.minimum,
            maximum=self._view_state.maximum,
            status_text=(
                f"{selection_name}: {selected_count} token"
                f"{'s' if selected_count != 1 else ''} {operator} {threshold:g}"
            ),
        )
        self._view.set_statistics_selection(self._view_state)
        return selection_name
