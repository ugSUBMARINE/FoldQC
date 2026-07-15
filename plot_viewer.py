"""
Qt plot viewer
==============
Embeds Matplotlib figures in the host viewer's Qt event loop.

Qt imports go through :mod:`compat`; Matplotlib's Qt backend is imported lazily
so plugin startup and non-GUI tests do not depend on QtAgg availability.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .compat import QAction, QtCore, QtGui, QtWidgets
from .mol_viewer import (
    clear_selections,
    disable_object,
    enable_object,
    refresh,
    show_token_groups,
    update_token_selection,
)

if TYPE_CHECKING:
    from .token_map import TokenMap

MAX_INITIAL_CANVAS_DIMENSION = 900
SELECTION_COLOR = "#ff8c00"
SELECTION_BAR_EDGE_COLOR = "#666666"
SELECTION_SPAN_ALPHA = 0.16
SELECTION_OUTLINE_WIDTH = 2.0
SELECTION_HINT = "Click/drag: replace selection · Cmd/Ctrl/Alt-click/drag: add"
CLEAR_ICON_SIZE = 24
CLEAR_ICON_INSET = 1


def _make_clear_selection_icon() -> Any:
    """Return a dark monochrome trash icon matching Matplotlib's toolbar."""

    def draw_pixmap(color: str):
        size = CLEAR_ICON_SIZE
        left = CLEAR_ICON_INSET
        right = size - CLEAR_ICON_INSET
        top = CLEAR_ICON_INSET
        bottom = size - CLEAR_ICON_INSET
        center = size // 2
        pixmap = QtGui.QPixmap(size, size)
        transparent = getattr(
            getattr(QtCore.Qt, "GlobalColor", QtCore.Qt), "transparent"
        )
        pixmap.fill(transparent)
        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(QtGui.QColor(color))
        pen.setWidthF(3.0)
        painter.setPen(pen)
        painter.drawLine(left, top + 4, right, top + 4)
        painter.drawLine(center - 3, top, center + 3, top)
        painter.drawLine(center - 5, top + 2, center + 5, top + 2)
        painter.drawLine(left + 1, top + 7, left + 2, bottom)
        painter.drawLine(right - 1, top + 7, right - 2, bottom)
        painter.drawLine(left + 2, bottom, right - 2, bottom)
        painter.drawLine(center - 3, top + 8, center - 3, bottom - 3)
        painter.drawLine(center + 3, top + 8, center + 3, bottom - 3)
        painter.end()
        return pixmap

    icon = QtGui.QIcon(draw_pixmap("#202020"))
    icon_mode = getattr(QtGui.QIcon, "Mode", QtGui.QIcon)
    icon_state = getattr(QtGui.QIcon, "State", QtGui.QIcon)
    icon.addPixmap(
        draw_pixmap("#555555"),
        icon_mode.Disabled,
        icon_state.Off,
    )
    return icon


def _load_qt_backend() -> tuple[type, type]:
    """Return Matplotlib's Qt canvas and toolbar classes.

    ``backend_qtagg`` is version-agnostic across Qt5/Qt6 in newer Matplotlib
    releases.  ``backend_qt5agg`` remains useful for older installations and
    works across the supported Qt5/Qt6 host environments.
    """
    try:
        from matplotlib.backends.backend_qtagg import (  # type: ignore[import-not-found]
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
    except Exception:
        from matplotlib.backends.backend_qt5agg import (  # type: ignore[import-not-found]
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )

    return FigureCanvasQTAgg, NavigationToolbar2QT


class PlotDialog(QtWidgets.QDialog):
    """Dialog containing one Matplotlib figure canvas and toolbar."""

    def __init__(
        self,
        figure: Any,
        title: str,
        parent: QtWidgets.QWidget | None = None,
        on_close: Callable[[PlotDialog], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._figure = figure
        self._on_close = on_close
        self._layout_pending = False
        self._selection_metadata: dict[str, Any] | None = None
        self._selection_ax: Any | None = None
        self._selection_axes: list[Any] = []
        self._selector: Any | None = None
        self._selectors: list[Any] = []
        self._mpl_cids: list[int] = []
        self._selected_token_indices: list[int] = []
        self._selected_row_indices: list[int] = []
        self._selected_col_indices: list[int] = []
        self._selected_ensemble_member_indices: list[int] = []
        self._selected_bar_group_indices: list[int] = []
        self._matrix_selection_mask: np.ndarray | None = None
        self._highlight_artists: list[Any] = []
        self._matrix_boundary_artist: Any | None = None
        self._bar_original_styles: list[tuple[Any, Any, float]] = []
        self._clear_selection_action: Any | None = None

        self.setWindowTitle(title)

        FigureCanvas, NavigationToolbar = _load_qt_backend()
        self._canvas = FigureCanvas(figure)
        self._canvas.setParent(self)
        self._canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self._canvas.updateGeometry()
        self._toolbar = NavigationToolbar(self._canvas, self, coordinates=True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)
        self._selection_hint = QtWidgets.QLabel(SELECTION_HINT)
        self._selection_hint.setVisible(False)
        layout.addWidget(self._selection_hint)

        self._resize_to_figure_size()
        self._install_viewer_selection_bridge()
        self._schedule_fit_figure_to_canvas()

    def _target_canvas_size(self) -> tuple[int, int]:
        """Return an initial Qt canvas size that preserves figure proportions."""
        try:
            width_in, height_in = self._figure.get_size_inches()
            dpi = float(self._figure.get_dpi())
        except Exception:
            return 900, 650

        width = max(1, int(round(float(width_in) * dpi)))
        height = max(1, int(round(float(height_in) * dpi)))
        max_width = MAX_INITIAL_CANVAS_DIMENSION
        max_height = MAX_INITIAL_CANVAS_DIMENSION

        try:
            screen = self.screen() or QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                max_width = min(max_width, max(1, int(available.width() * 0.9)))
                max_height = min(max_height, max(1, int(available.height() * 0.85)))
        except Exception:
            pass

        scale = min(1.0, max_width / width, max_height / height)
        return max(1, int(round(width * scale))), max(1, int(round(height * scale)))

    def _resize_to_figure_size(self) -> None:
        """Size the dialog so the initial canvas follows Matplotlib figsize."""
        canvas_w, canvas_h = self._target_canvas_size()
        self._canvas.resize(canvas_w, canvas_h)

        margins = self.layout().contentsMargins()
        spacing = max(0, int(self.layout().spacing()))
        toolbar_h = self._toolbar.sizeHint().height()
        dialog_w = canvas_w + margins.left() + margins.right()
        dialog_h = canvas_h + toolbar_h + spacing + margins.top() + margins.bottom()
        self.resize(dialog_w, dialog_h)

    def _schedule_fit_figure_to_canvas(self) -> None:
        """Run tight layout after Qt has finished the current resize/show step."""
        if self._layout_pending:
            return
        self._layout_pending = True
        QtCore.QTimer.singleShot(0, self._fit_figure_to_canvas)

    def _fit_figure_to_canvas(self) -> None:
        """Recompute Matplotlib layout for the current Qt canvas size."""
        self._layout_pending = False
        try:
            self._figure.set_tight_layout(True)
        except Exception:
            pass
        try:
            # A real renderer is needed for text/colorbar extents. This mirrors
            # what Matplotlib's toolbar does when its "tight layout" button works.
            self._canvas.draw()
            self._figure.tight_layout()
        except Exception:
            pass
        self._canvas.draw_idle()

    def resizeEvent(self, event) -> None:  # noqa: N802  Qt override
        super().resizeEvent(event)
        self._schedule_fit_figure_to_canvas()

    def _install_viewer_selection_bridge(self) -> None:
        """Connect token-aware Matplotlib figures to viewer selections."""
        metadata = getattr(self._figure, "_foldqc_viewer_selection", None)
        if not isinstance(metadata, dict):
            return
        if metadata.get("kind") not in {
            "line",
            "bars",
            "matrix",
            "ensemble_site_summary",
        }:
            return
        if metadata.get("kind") == "ensemble_site_summary":
            if not metadata.get("member_obj_names"):
                return
        elif not metadata.get("token_map") or not metadata.get("obj_name"):
            return
        if not self._figure.axes:
            return

        self._selection_metadata = metadata
        if metadata.get("kind") in {"bars", "ensemble_site_summary"}:
            self._selection_axes = list(self._figure.axes)
        else:
            self._selection_axes = [self._figure.axes[0]]
        self._selection_ax = self._selection_axes[0]
        self._install_toolbar_formatters()
        self._install_selection_controls()
        self._mpl_cids.append(
            self._canvas.mpl_connect("button_press_event", self._on_plot_click)
        )

        try:
            from matplotlib.widgets import RectangleSelector

            minspany = 0.5 if metadata.get("kind") == "matrix" else 0.0
            for ax in self._selection_axes:
                selector = RectangleSelector(
                    ax,
                    self._on_rectangle_select,
                    useblit=False,
                    button=[1],
                    minspanx=0.5,
                    minspany=minspany,
                    spancoords="data",
                    interactive=False,
                )
                self._selectors.append(selector)
            self._selector = self._selectors[0] if self._selectors else None
        except TypeError:
            # Older Matplotlib releases do not accept ``interactive``.
            for ax in self._selection_axes:
                selector = RectangleSelector(
                    ax,
                    self._on_rectangle_select,
                    useblit=False,
                    button=[1],
                    minspanx=0.5,
                    minspany=0.5 if metadata.get("kind") == "matrix" else 0.0,
                    spancoords="data",
                )
                self._selectors.append(selector)
            self._selector = self._selectors[0] if self._selectors else None
        except Exception:
            self._selector = None

    def _install_selection_controls(self) -> None:
        """Show interaction help and add an icon-only Clear toolbar action."""
        self._selection_hint.setVisible(True)
        action = QAction(_make_clear_selection_icon(), "", self)
        action.setToolTip("Clear plot selection")
        action.setEnabled(False)
        action.triggered.connect(self._clear_plot_selection)
        self._toolbar.addSeparator()
        self._toolbar.addAction(action)
        self._clear_selection_action = action

    def _set_clear_action_enabled(self, enabled: bool) -> None:
        action = getattr(self, "_clear_selection_action", None)
        if action is not None:
            action.setEnabled(bool(enabled))

    @staticmethod
    def _format_token_identity(token: Any) -> str:
        """Return a toolbar label for one prediction token."""
        token_idx = int(token.token_idx)
        chain_id = str(token.chain_id)
        residue_id = token.resi
        res_name = str(token.res_name)
        if bool(token.is_hetatm):
            residue = f"{chain_id}:{res_name.upper()}{residue_id}"
            atom_name = getattr(token, "atom_name", None)
            if atom_name:
                residue = f"{residue}/{atom_name}"
        else:
            residue = f"{chain_id}:{res_name.title()}{residue_id}"
        return f"token {token_idx} · {residue}"

    @staticmethod
    def _make_position_token_lookup(
        token_indices: list[int],
        positions: list[float],
        *,
        half_width: float = 0.5,
    ) -> Callable[[float | None], int | None] | None:
        """Return a cached lookup for discrete token display positions."""
        if not token_indices or len(token_indices) != len(positions):
            return None
        try:
            position_array = np.asarray(positions, dtype=np.float64)
            index_array = np.asarray(token_indices, dtype=np.int64)
        except (TypeError, ValueError, OverflowError):
            return None
        finite = np.isfinite(position_array)
        if not np.any(finite):
            return None
        position_array = position_array[finite]
        index_array = index_array[finite]
        order = np.argsort(position_array, kind="stable")
        position_array = position_array[order]
        index_array = index_array[order]
        tolerance = abs(float(half_width))

        def lookup(value: float | None) -> int | None:
            if value is None:
                return None
            try:
                coordinate = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if not np.isfinite(coordinate):
                return None
            insertion = int(np.searchsorted(position_array, coordinate, side="left"))
            candidates = [
                idx
                for idx in (insertion - 1, insertion)
                if 0 <= idx < position_array.size
            ]
            if not candidates:
                return None
            nearest = min(
                candidates,
                key=lambda idx: (abs(float(position_array[idx]) - coordinate), idx),
            )
            if abs(float(position_array[nearest]) - coordinate) > tolerance:
                return None
            return int(index_array[nearest])

        return lookup

    @staticmethod
    def _token_identity_for_index(token_map: Any, token_idx: int | None) -> str | None:
        """Return one token identity, or None for invalid toolbar metadata."""
        if token_idx is None or token_idx < 0:
            return None
        try:
            token = token_map[token_idx]
            return PlotDialog._format_token_identity(token)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _reflow_matrix_toolbar_message(message: str) -> str:
        """Move matrix token identities beside Matplotlib's image value."""
        lines = str(message).splitlines()
        if not lines:
            return str(message)
        marker = " | row: "
        if marker not in lines[0]:
            return str(message)
        coordinates, identities = lines[0].split(marker, 1)
        identity_line = f"row: {identities}".replace(" | col: ", ", col: ", 1)
        if len(lines) > 1:
            identity_line = f"{identity_line} {' '.join(lines[1:])}"
        return f"{coordinates}\n{identity_line}"

    def _install_matrix_toolbar_message_layout(self) -> None:
        """Keep matrix coordinates and token/value details on separate lines."""
        toolbar = getattr(self, "_toolbar", None)
        if toolbar is None or getattr(toolbar, "_foldqc_matrix_message_layout", False):
            return
        original_set_message = getattr(toolbar, "set_message", None)
        if not callable(original_set_message):
            return

        def set_message(message: str) -> None:
            original_set_message(self._reflow_matrix_toolbar_message(message))

        toolbar.set_message = set_message
        toolbar._foldqc_matrix_message_layout = True

    def _install_toolbar_formatters(self) -> None:
        """Append token identities to Matplotlib's native toolbar readout."""
        metadata = self._selection_metadata or {}
        kind = metadata.get("kind")
        if kind == "ensemble_site_summary":
            return
        if kind == "bars" and "bar_token_indices" in metadata:
            return

        token_map = metadata.get("token_map")
        if token_map is None:
            return

        identity_for_coordinates: Callable[[float | None, float | None], str | None]
        if kind == "matrix":
            try:
                row_indices = self._metadata_indices("row_indices")
                col_indices = self._metadata_indices("col_indices")
            except (TypeError, ValueError, OverflowError):
                return
            row_lookup = self._make_position_token_lookup(
                row_indices,
                [float(i) for i in range(len(row_indices))],
            )
            col_lookup = self._make_position_token_lookup(
                col_indices,
                [float(i) for i in range(len(col_indices))],
            )
            if row_lookup is None or col_lookup is None:
                return

            def matrix_identity(x: float | None, y: float | None) -> str | None:
                row_identity = self._token_identity_for_index(token_map, row_lookup(y))
                col_identity = self._token_identity_for_index(token_map, col_lookup(x))
                if row_identity is None or col_identity is None:
                    return None
                return f"row: {row_identity} | col: {col_identity}"

            identity_for_coordinates = matrix_identity
        elif kind in {"line", "bars"}:
            try:
                token_indices = self._metadata_indices("token_indices")
            except (TypeError, ValueError, OverflowError):
                return
            raw_positions = metadata.get("x_positions")
            if raw_positions is None:
                positions = [float(i) for i in range(len(token_indices))]
            else:
                try:
                    positions = [float(position) for position in raw_positions]
                except (TypeError, ValueError, OverflowError):
                    return
            token_lookup = self._make_position_token_lookup(token_indices, positions)
            if token_lookup is None:
                return

            def token_identity(x: float | None, _y: float | None) -> str | None:
                return self._token_identity_for_index(token_map, token_lookup(x))

            identity_for_coordinates = token_identity
        else:
            return

        installed = False
        for axis in self._selection_axes:
            original_formatter = getattr(axis, "format_coord", None)
            if not callable(original_formatter):
                continue

            def format_coord(
                x,
                y,
                *,
                original=original_formatter,
                identity_formatter=identity_for_coordinates,
            ):
                base = original(x, y)
                try:
                    identity = identity_formatter(x, y)
                except Exception:
                    identity = None
                if not identity:
                    return base
                return f"{base} | {identity}"

            axis.format_coord = format_coord
            installed = True
        if kind == "matrix" and installed:
            self._install_matrix_toolbar_message_layout()

    def _toolbar_is_active(self) -> bool:
        """Return True while Matplotlib's pan/zoom toolbar mode is active."""
        mode = getattr(self._toolbar, "mode", "")
        name = getattr(mode, "name", None)
        if name is not None:
            return str(name).upper() != "NONE"
        return bool(str(mode))

    @staticmethod
    def _event_is_additive(event) -> bool:
        """Return whether a mouse event requests additive selection."""
        additive_modifiers = {"cmd", "super", "ctrl", "control", "alt"}
        modifiers = {
            str(modifier).lower()
            for modifier in (getattr(event, "modifiers", None) or [])
        }
        if modifiers & additive_modifiers:
            return True

        # Matplotlib before 3.8 does not expose MouseEvent.modifiers.
        key = str(getattr(event, "key", "") or "").lower()
        return bool(set(key.split("+")) & additive_modifiers)

    def _metadata_indices(self, key: str) -> list[int]:
        metadata = self._selection_metadata or {}
        return [int(i) for i in metadata.get(key, [])]

    def _metadata_positions(self, key: str, count: int) -> list[float]:
        metadata = self._selection_metadata or {}
        values = metadata.get(key)
        if values is None:
            return [float(i) for i in range(count)]
        positions = [float(x) for x in values]
        if len(positions) != count:
            return [float(i) for i in range(count)]
        return positions

    def _metadata_bar_groups(self) -> list[list[int]]:
        metadata = self._selection_metadata or {}
        groups = metadata.get("bar_token_indices")
        if not groups:
            return []
        return [[int(i) for i in group] for group in groups]

    def _metadata_bar_positions_widths(
        self, count: int
    ) -> tuple[list[float], list[float]]:
        metadata = self._selection_metadata or {}
        positions = metadata.get("bar_x_positions")
        widths = metadata.get("bar_widths")
        if positions is None or len(positions) != count:
            positions = [float(i) for i in range(count)]
        else:
            positions = [float(x) for x in positions]
        if widths is None or len(widths) != count:
            widths = [0.8 for _ in range(count)]
        else:
            widths = [float(w) for w in widths]
        return positions, widths

    def _metadata_member_positions_widths(
        self, count: int
    ) -> tuple[list[float], list[float]]:
        metadata = self._selection_metadata or {}
        positions = metadata.get("member_x_positions")
        widths = metadata.get("member_widths")
        if positions is None or len(positions) != count:
            positions = [float(i) for i in range(count)]
        else:
            positions = [float(x) for x in positions]
        if widths is None or len(widths) != count:
            widths = [0.9 for _ in range(count)]
        else:
            widths = [float(w) for w in widths]
        return positions, widths

    @staticmethod
    def _unique_tokens(*groups: list[int]) -> list[int]:
        seen: set[int] = set()
        result: list[int] = []
        for group in groups:
            for token_idx in group:
                if token_idx in seen:
                    continue
                seen.add(token_idx)
                result.append(token_idx)
        return result

    @staticmethod
    def _nearest_token(
        value: float | None,
        token_indices: list[int],
        positions: list[float],
    ) -> list[int]:
        if value is None or not token_indices or not positions:
            return []
        pos = min(range(len(positions)), key=lambda i: abs(positions[i] - value))
        return [token_indices[pos]]

    @staticmethod
    def _tokens_in_interval(
        start: float | None,
        end: float | None,
        token_indices: list[int],
        positions: list[float],
    ) -> list[int]:
        if start is None or end is None or not token_indices or not positions:
            return []
        lo, hi = sorted((float(start), float(end)))
        return [
            token_idx
            for token_idx, pos in zip(token_indices, positions)
            if lo <= pos <= hi
        ]

    def _bar_groups_in_interval(
        self,
        start: float | None,
        end: float | None,
        *,
        point: bool = False,
    ) -> list[int]:
        groups = self._metadata_bar_groups()
        if start is None or end is None or not groups:
            return []
        lo, hi = sorted((float(start), float(end)))
        positions, widths = self._metadata_bar_positions_widths(len(groups))

        selected: list[int] = []
        for group, pos, width in zip(groups, positions, widths):
            half_width = abs(float(width)) / 2.0
            left = float(pos) - half_width
            right = float(pos) + half_width
            if point:
                overlaps = left <= lo <= right
            else:
                overlaps = right >= lo and left <= hi
            if overlaps:
                selected.extend(group)
        return selected

    def _bar_group_indices_in_interval(
        self,
        start: float | None,
        end: float | None,
        *,
        point: bool = False,
    ) -> list[int]:
        groups = self._metadata_bar_groups()
        if start is None or end is None or not groups:
            return []
        lo, hi = sorted((float(start), float(end)))
        positions, widths = self._metadata_bar_positions_widths(len(groups))
        selected: list[int] = []
        for idx, (pos, width) in enumerate(zip(positions, widths)):
            half_width = abs(float(width)) / 2.0
            left = float(pos) - half_width
            right = float(pos) + half_width
            overlaps = left <= lo <= right if point else right >= lo and left <= hi
            if overlaps:
                selected.append(idx)
        return selected

    def _members_in_interval(
        self,
        start: float | None,
        end: float | None,
        *,
        point: bool = False,
    ) -> list[int]:
        metadata = self._selection_metadata or {}
        members = metadata.get("member_obj_names") or []
        if start is None or end is None or not members:
            return []
        lo, hi = sorted((float(start), float(end)))
        positions, widths = self._metadata_member_positions_widths(len(members))
        selected: list[int] = []
        for idx, (pos, width) in enumerate(zip(positions, widths)):
            half_width = abs(float(width)) / 2.0
            left = float(pos) - half_width
            right = float(pos) + half_width
            if point:
                overlaps = left <= lo <= right
            else:
                overlaps = right >= lo and left <= hi
            if overlaps:
                selected.append(idx)
        return selected

    @staticmethod
    def _merged_position_intervals(
        positions: list[float], half_width: float = 0.5
    ) -> list[tuple[float, float]]:
        """Return merged display intervals for selected x positions."""
        intervals = sorted(
            (float(position) - half_width, float(position) + half_width)
            for position in set(positions)
        )
        merged: list[tuple[float, float]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _remove_highlight_artists(self) -> None:
        for artist in getattr(self, "_highlight_artists", []):
            try:
                artist.remove()
            except Exception:
                pass
        self._highlight_artists = []
        boundary = getattr(self, "_matrix_boundary_artist", None)
        if boundary is not None:
            try:
                boundary.remove()
            except Exception:
                pass
        self._matrix_boundary_artist = None

    def _restore_bar_styles(self) -> None:
        for patch, edgecolor, linewidth in getattr(self, "_bar_original_styles", []):
            patch.set_edgecolor(edgecolor)
            patch.set_linewidth(linewidth)

    def _capture_bar_styles(self) -> None:
        if getattr(self, "_bar_original_styles", []):
            return
        styles = []
        for ax in self._selection_axes:
            for patch in getattr(ax, "patches", []):
                styles.append((patch, patch.get_edgecolor(), patch.get_linewidth()))
        self._bar_original_styles = styles

    def _render_span_highlights(self) -> None:
        self._remove_highlight_artists()
        token_indices = self._metadata_indices("token_indices")
        positions = self._metadata_positions("x_positions", len(token_indices))
        position_by_token = dict(zip(token_indices, positions))
        selected_positions = [
            position_by_token[index]
            for index in self._selected_token_indices
            if index in position_by_token
        ]
        for start, end in self._merged_position_intervals(selected_positions):
            for ax in self._selection_axes:
                self._highlight_artists.append(
                    ax.axvspan(
                        start,
                        end,
                        color=SELECTION_COLOR,
                        alpha=SELECTION_SPAN_ALPHA,
                        linewidth=0.0,
                        zorder=4.0,
                    )
                )

    def _render_bar_highlights(self, group_indices: list[int]) -> None:
        self._remove_highlight_artists()
        self._capture_bar_styles()
        self._restore_bar_styles()
        if not group_indices:
            return
        metadata = self._selection_metadata or {}
        if metadata.get("kind") == "ensemble_site_summary":
            count = len(metadata.get("member_obj_names") or [])
            positions, widths = self._metadata_member_positions_widths(count)
        else:
            count = len(self._metadata_bar_groups())
            positions, widths = self._metadata_bar_positions_widths(count)
        selected_intervals = [
            (
                positions[index] - abs(widths[index]) / 2.0,
                positions[index] + abs(widths[index]) / 2.0,
            )
            for index in group_indices
            if 0 <= index < count
        ]
        for patch, _edgecolor, _linewidth in self._bar_original_styles:
            center = float(patch.get_x()) + float(patch.get_width()) / 2.0
            if any(left <= center <= right for left, right in selected_intervals):
                patch.set_edgecolor(SELECTION_BAR_EDGE_COLOR)
                patch.set_linewidth(SELECTION_OUTLINE_WIDTH)

    def _render_group_span_highlights(self, group_indices: list[int]) -> None:
        """Highlight selected ensemble-member x groups with translucent spans."""
        self._remove_highlight_artists()
        metadata = self._selection_metadata or {}
        count = len(metadata.get("member_obj_names") or [])
        positions, widths = self._metadata_member_positions_widths(count)
        for index in group_indices:
            if index < 0 or index >= count:
                continue
            half_width = abs(widths[index]) / 2.0
            for ax in self._selection_axes:
                self._highlight_artists.append(
                    ax.axvspan(
                        positions[index] - half_width,
                        positions[index] + half_width,
                        color=SELECTION_COLOR,
                        alpha=SELECTION_SPAN_ALPHA,
                        linewidth=0.0,
                        zorder=4.0,
                    )
                )

    @staticmethod
    def _matrix_cells_in_rectangle(
        x0: float | None,
        y0: float | None,
        x1: float | None,
        y1: float | None,
        shape: tuple[int, int],
    ) -> set[tuple[int, int]]:
        if None in {x0, y0, x1, y1}:
            return set()
        n_rows, n_cols = shape
        x_lo, x_hi = sorted((float(x0), float(x1)))
        y_lo, y_hi = sorted((float(y0), float(y1)))
        rows = [row for row in range(n_rows) if y_lo <= row <= y_hi]
        cols = [col for col in range(n_cols) if x_lo <= col <= x_hi]
        return {(row, col) for row in rows for col in cols}

    def _update_matrix_mask(
        self, cells: set[tuple[int, int]], *, additive: bool
    ) -> None:
        metadata = self._selection_metadata or {}
        shape = (
            len(metadata.get("row_indices") or []),
            len(metadata.get("col_indices") or []),
        )
        current = getattr(self, "_matrix_selection_mask", None)
        if current is None or current.shape != shape or not additive:
            current = np.zeros(shape, dtype=bool)
        for row, col in cells:
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                current[row, col] = True
        self._matrix_selection_mask = current

    @staticmethod
    def _matrix_boundary_segments(mask: np.ndarray) -> list[list[tuple[float, float]]]:
        """Return outer cell-boundary segments for a boolean matrix mask."""
        segments: list[list[tuple[float, float]]] = []
        n_rows, n_cols = mask.shape
        for row, col in np.argwhere(mask):
            left, right = col - 0.5, col + 0.5
            top, bottom = row - 0.5, row + 0.5
            if row == 0 or not mask[row - 1, col]:
                segments.append([(left, top), (right, top)])
            if row == n_rows - 1 or not mask[row + 1, col]:
                segments.append([(left, bottom), (right, bottom)])
            if col == 0 or not mask[row, col - 1]:
                segments.append([(left, top), (left, bottom)])
            if col == n_cols - 1 or not mask[row, col + 1]:
                segments.append([(right, top), (right, bottom)])
        return segments

    def _render_matrix_highlight(self) -> None:
        self._remove_highlight_artists()
        mask = getattr(self, "_matrix_selection_mask", None)
        if mask is None or not np.any(mask):
            return
        from matplotlib.collections import LineCollection

        boundary = LineCollection(
            self._matrix_boundary_segments(mask),
            colors=[SELECTION_COLOR],
            linewidths=[SELECTION_OUTLINE_WIDTH],
            zorder=10,
        )
        self._selection_ax.add_collection(boundary)
        self._matrix_boundary_artist = boundary

    def _update_click_highlight(self, event, *, additive: bool) -> None:
        metadata = self._selection_metadata or {}
        kind = metadata.get("kind")
        if kind == "matrix":
            row = self._nearest_token(
                event.ydata,
                list(range(len(metadata.get("row_indices") or []))),
                [float(i) for i in range(len(metadata.get("row_indices") or []))],
            )
            col = self._nearest_token(
                event.xdata,
                list(range(len(metadata.get("col_indices") or []))),
                [float(i) for i in range(len(metadata.get("col_indices") or []))],
            )
            cells = {(row[0], col[0])} if row and col else set()
            self._update_matrix_mask(cells, additive=additive)
            self._render_matrix_highlight()
        elif kind == "ensemble_site_summary":
            self._render_group_span_highlights(self._selected_ensemble_member_indices)
        elif kind == "bars" and metadata.get("bar_token_indices"):
            groups = self._bar_group_indices_in_interval(
                event.xdata, event.xdata, point=True
            )
            self._selected_bar_group_indices = (
                self._unique_tokens(self._selected_bar_group_indices, groups)
                if additive
                else groups
            )
            self._render_bar_highlights(self._selected_bar_group_indices)
        else:
            self._render_span_highlights()
        self._canvas.draw_idle()

    def _update_rectangle_highlight(self, eclick, erelease, *, additive: bool) -> None:
        metadata = self._selection_metadata or {}
        kind = metadata.get("kind")
        if kind == "matrix":
            shape = (
                len(metadata.get("row_indices") or []),
                len(metadata.get("col_indices") or []),
            )
            cells = self._matrix_cells_in_rectangle(
                eclick.xdata,
                eclick.ydata,
                erelease.xdata,
                erelease.ydata,
                shape,
            )
            self._update_matrix_mask(cells, additive=additive)
            self._render_matrix_highlight()
        elif kind == "ensemble_site_summary":
            self._render_group_span_highlights(self._selected_ensemble_member_indices)
        elif kind == "bars" and metadata.get("bar_token_indices"):
            groups = self._bar_group_indices_in_interval(eclick.xdata, erelease.xdata)
            self._selected_bar_group_indices = (
                self._unique_tokens(self._selected_bar_group_indices, groups)
                if additive
                else groups
            )
            self._render_bar_highlights(self._selected_bar_group_indices)
        else:
            self._render_span_highlights()
        self._canvas.draw_idle()

    def _tokens_for_click(self, xdata: float | None, ydata: float | None):
        metadata = self._selection_metadata or {}
        kind = metadata.get("kind")
        if kind == "matrix":
            row_indices = self._metadata_indices("row_indices")
            col_indices = self._metadata_indices("col_indices")
            row_positions = [float(i) for i in range(len(row_indices))]
            col_positions = [float(i) for i in range(len(col_indices))]
            rows = self._nearest_token(ydata, row_indices, row_positions)
            cols = self._nearest_token(xdata, col_indices, col_positions)
            return self._unique_tokens(rows, cols), rows, cols

        if kind == "bars" and metadata.get("bar_token_indices"):
            tokens = self._bar_groups_in_interval(xdata, xdata, point=True)
            return self._unique_tokens(tokens), [], []

        token_indices = self._metadata_indices("token_indices")
        positions = self._metadata_positions("x_positions", len(token_indices))
        tokens = self._nearest_token(xdata, token_indices, positions)
        return tokens, [], []

    def _tokens_for_rectangle(
        self,
        x0: float | None,
        y0: float | None,
        x1: float | None,
        y1: float | None,
    ):
        metadata = self._selection_metadata or {}
        kind = metadata.get("kind")
        if kind == "matrix":
            row_indices = self._metadata_indices("row_indices")
            col_indices = self._metadata_indices("col_indices")
            row_positions = [float(i) for i in range(len(row_indices))]
            col_positions = [float(i) for i in range(len(col_indices))]
            rows = self._tokens_in_interval(y0, y1, row_indices, row_positions)
            cols = self._tokens_in_interval(x0, x1, col_indices, col_positions)
            return self._unique_tokens(rows, cols), rows, cols

        if kind == "bars" and metadata.get("bar_token_indices"):
            tokens = self._bar_groups_in_interval(x0, x1)
            return self._unique_tokens(tokens), [], []

        token_indices = self._metadata_indices("token_indices")
        positions = self._metadata_positions("x_positions", len(token_indices))
        tokens = self._tokens_in_interval(x0, x1, token_indices, positions)
        return tokens, [], []

    def _selection_name(self, suffix: str) -> str:
        metadata = self._selection_metadata or {}
        prefix = str(metadata.get("selection_prefix") or "foldqc_plot")
        prefix = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in prefix)
        if not prefix:
            prefix = "foldqc_plot"
        return f"{prefix}_{suffix}"

    def _selection_object_token_maps(self) -> list[tuple[str, TokenMap]]:
        metadata = self._selection_metadata or {}
        token_maps = metadata.get("token_maps")
        if token_maps is not None:
            object_names = metadata.get("token_map_obj_names")
            if object_names is None:
                raise ValueError(
                    "Plot selection metadata with token_maps requires "
                    "token_map_obj_names."
                )
            if len(object_names) != len(token_maps):
                raise ValueError(
                    "Plot selection token_map_obj_names must correspond "
                    "one-to-one with token_maps."
                )
            object_token_maps = list(zip(object_names, token_maps))
        else:
            object_name = metadata.get("obj_name")
            token_map = metadata.get("token_map")
            if not object_name or token_map is None:
                raise ValueError(
                    "Plot selection metadata requires an object name and token map."
                )
            object_token_maps = [
                (
                    object_name,
                    token_map,
                )
            ]
        return object_token_maps

    def _selection_groups_for_member_sites(
        self, member_indices: list[int]
    ) -> list[tuple[list[int], str, TokenMap]]:
        metadata = self._selection_metadata or {}
        object_names = metadata.get("member_obj_names") or []
        token_maps = metadata.get("member_token_maps") or []
        site_indices = metadata.get("member_site_indices") or []
        if len(object_names) != len(token_maps):
            raise ValueError(
                "Ensemble site member_obj_names must correspond one-to-one "
                "with member_token_maps."
            )
        if len(site_indices) != len(token_maps):
            raise ValueError(
                "Ensemble site member_site_indices must correspond one-to-one "
                "with member_token_maps."
            )
        groups: list[tuple[list[int], str, TokenMap]] = []
        for member_idx in member_indices:
            if member_idx < 0 or member_idx >= len(token_maps):
                continue
            token_map = token_maps[member_idx]
            indices = site_indices[member_idx]
            object_name = object_names[member_idx]
            groups.append((indices, object_name, token_map))
        return groups

    def _apply_ensemble_site_selection(
        self, member_indices: list[int], *, additive: bool = False
    ) -> None:
        """Enable selected ensemble members and show their binding-site selection."""
        if not member_indices:
            return
        metadata = self._selection_metadata or {}
        obj_names = [str(name) for name in metadata.get("member_obj_names") or []]
        if not obj_names:
            return
        if additive:
            member_indices = self._unique_tokens(
                getattr(self, "_selected_ensemble_member_indices", []),
                member_indices,
            )
        else:
            member_indices = self._unique_tokens(member_indices)
        self._selected_ensemble_member_indices = member_indices
        selected = set(member_indices)
        selection_name = str(metadata.get("selection_name") or "foldqc_ensemble_site")
        selection_name = "".join(
            ch if ch.isalnum() or ch == "_" else "_" for ch in selection_name
        )
        if not selection_name:
            selection_name = "foldqc_ensemble_site"
        try:
            for idx, obj_name in enumerate(obj_names):
                if idx in selected:
                    enable_object(obj_name)
                else:
                    disable_object(obj_name)
            groups = self._selection_groups_for_member_sites(sorted(selected))
            show_token_groups(selection_name, groups)
            self._set_clear_action_enabled(True)
        except Exception as exc:
            print(f"FoldQC ensemble site selection failed: {exc}")

    def _apply_viewer_selection(
        self,
        token_indices: list[int],
        row_indices: list[int] | None = None,
        col_indices: list[int] | None = None,
        *,
        additive: bool = False,
    ) -> None:
        """Create or update named viewer selections for selected plot tokens."""
        if not token_indices:
            return
        if additive:
            token_indices = self._unique_tokens(
                getattr(self, "_selected_token_indices", []), token_indices
            )
            row_indices = self._unique_tokens(
                getattr(self, "_selected_row_indices", []), row_indices or []
            )
            col_indices = self._unique_tokens(
                getattr(self, "_selected_col_indices", []), col_indices or []
            )
        else:
            token_indices = self._unique_tokens(token_indices)
            row_indices = self._unique_tokens(row_indices or [])
            col_indices = self._unique_tokens(col_indices or [])
        self._selected_token_indices = token_indices
        self._selected_row_indices = row_indices
        self._selected_col_indices = col_indices
        try:
            object_token_maps = self._selection_object_token_maps()
            update_token_selection(
                self._selection_name("selection"),
                token_indices,
                object_token_maps,
                refresh_view=False,
            )
            if (self._selection_metadata or {}).get("kind") == "matrix":
                update_token_selection(
                    self._selection_name("rows"),
                    row_indices,
                    object_token_maps,
                    refresh_view=False,
                )
                update_token_selection(
                    self._selection_name("cols"),
                    col_indices,
                    object_token_maps,
                    refresh_view=False,
                )
            refresh()
            self._set_clear_action_enabled(True)
        except Exception as exc:
            print(f"FoldQC plot selection failed: {exc}")

    def _clear_plot_selection(self) -> None:
        """Clear plot highlights, accumulated state, and named viewer selections."""
        self._remove_highlight_artists()
        self._restore_bar_styles()
        self._selected_token_indices = []
        self._selected_row_indices = []
        self._selected_col_indices = []
        self._selected_ensemble_member_indices = []
        self._selected_bar_group_indices = []
        mask = getattr(self, "_matrix_selection_mask", None)
        if mask is not None:
            mask.fill(False)
        metadata = self._selection_metadata or {}
        try:
            if metadata.get("kind") == "ensemble_site_summary":
                selection_name = str(
                    metadata.get("selection_name") or "foldqc_ensemble_site"
                )
                clear_selections([selection_name], refresh_view=False)
                for obj_name in metadata.get("member_obj_names") or []:
                    enable_object(str(obj_name))
                refresh()
            else:
                names = [self._selection_name("selection")]
                if metadata.get("kind") == "matrix":
                    names.extend(
                        [self._selection_name("rows"), self._selection_name("cols")]
                    )
                clear_selections(names)
        except Exception as exc:
            print(f"FoldQC plot selection clear failed: {exc}")
        self._set_clear_action_enabled(False)
        self._canvas.draw_idle()

    def _on_plot_click(self, event) -> None:
        if event.inaxes not in self._selection_axes:
            return
        if event.button != 1 or self._toolbar_is_active():
            return
        additive = self._event_is_additive(event)
        if (self._selection_metadata or {}).get("kind") == "ensemble_site_summary":
            member_indices = self._members_in_interval(
                event.xdata, event.xdata, point=True
            )
            self._apply_ensemble_site_selection(member_indices, additive=additive)
            if member_indices:
                self._update_click_highlight(event, additive=additive)
            return
        token_indices, rows, cols = self._tokens_for_click(event.xdata, event.ydata)
        self._apply_viewer_selection(
            token_indices,
            rows,
            cols,
            additive=additive,
        )
        if token_indices:
            self._update_click_highlight(event, additive=additive)

    def _on_rectangle_select(self, eclick, erelease) -> None:
        if self._toolbar_is_active():
            return
        additive = self._event_is_additive(eclick) or self._event_is_additive(erelease)
        if (self._selection_metadata or {}).get("kind") == "ensemble_site_summary":
            member_indices = self._members_in_interval(eclick.xdata, erelease.xdata)
            self._apply_ensemble_site_selection(
                member_indices,
                additive=additive,
            )
            if member_indices:
                self._update_rectangle_highlight(eclick, erelease, additive=additive)
            return
        token_indices, rows, cols = self._tokens_for_rectangle(
            eclick.xdata,
            eclick.ydata,
            erelease.xdata,
            erelease.ydata,
        )
        self._apply_viewer_selection(
            token_indices,
            rows,
            cols,
            additive=additive,
        )
        if token_indices:
            self._update_rectangle_highlight(eclick, erelease, additive=additive)

    def closeEvent(self, event) -> None:  # noqa: N802  Qt override
        try:
            for cid in self._mpl_cids:
                self._canvas.mpl_disconnect(cid)
            self._mpl_cids.clear()
            selectors = self._selectors or (
                [self._selector] if self._selector is not None else []
            )
            for selector in selectors:
                try:
                    selector.set_active(False)
                except Exception:
                    pass
            self._selectors.clear()
            from matplotlib import pyplot as plt

            plt.close(self._figure)
        finally:
            if self._on_close is not None:
                self._on_close(self)
            super().closeEvent(event)


def show_figure(
    figure: Any,
    title: str,
    parent: QtWidgets.QWidget | None = None,
    on_close: Callable[[PlotDialog], None] | None = None,
) -> PlotDialog:
    """Create, show, and return a Qt dialog for *figure*."""
    dialog = PlotDialog(figure, title=title, parent=parent, on_close=on_close)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    dialog._schedule_fit_figure_to_canvas()
    return dialog
