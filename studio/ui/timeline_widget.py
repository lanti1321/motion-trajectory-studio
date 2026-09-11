from __future__ import annotations

import bisect

import numpy as np
from PySide6.QtCore import QEvent, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent, QNativeGestureEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGestureEvent, QPinchGesture, QSizePolicy, QWidget

from studio.core.project import TimelineLabel, TimelineSegment
from studio.ui.timeline_motion import motion_spans
from studio.ui.timeline_paint import TimelinePaintMixin


def _zoom_factor_from_wheel(angle_y: int, pixel_y: int) -> float | None:
    """Map a wheel/trackpad delta to a multiplicative zoom factor."""
    if angle_y != 0 and (pixel_y == 0 or abs(angle_y) >= 120):
        return 1.15 if angle_y > 0 else 1.0 / 1.15
    if pixel_y != 0:
        return 1.15 ** (pixel_y / 120.0)
    if angle_y != 0:
        return 1.15 if angle_y > 0 else 1.0 / 1.15
    return None


class TimelineWidget(TimelinePaintMixin, QWidget):
    seekRequested = Signal(float)
    labelSelectionChanged = Signal(list)
    rangeSelectionChanged = Signal(object)
    segmentSelectionChanged = Signal(str)
    visibleWindowChanged = Signal(float, float, float)
    clipMoveRequested = Signal(float, float, float)

    HEADER_WIDTH = 130
    RULER_HEIGHT = 26
    SEGMENT_LANE_HEIGHT = 24
    MIN_ROW_HEIGHT = 18
    MAX_ROW_HEIGHT = 46

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.duration = 0.0
        self.current_time = 0.0
        self.tracks: list[str] = []
        self.labels: list[TimelineLabel] = []
        self.segments: list[TimelineSegment] = []
        self.overridden_ranges: list[tuple[str, float, float]] = []
        self.selected_label_ids: set[str] = set()
        self.selected_segment_id: str | None = None
        self._last_selected_label_id: str | None = None
        self.selected_range: tuple[float, float] | None = None
        self.highlighted_tracks: set[str] = set()
        self.frame_interval = 0.001
        self.frame_times: list[float] = []
        self._channel_times: np.ndarray | None = None
        self._channel_values: dict[str, np.ndarray] = {}
        self._motion_spans: dict[str, list[tuple[float, float]]] = {}
        self._motion_cache: dict[tuple, list[tuple[float, float]]] = {}
        self._range_drag_mode: str | None = None
        self._range_drag_anchor = 0.0
        self._range_press_x = 0.0
        self._range_before_drag: tuple[float, float] | None = None
        self._move_preview: tuple[float, float] | None = None
        self._move_pointer_offset = 0.0
        self.zoom = 1.0
        self.pan_offset = 0.0
        self._data_version = 0
        self._static_layer: QPixmap | None = None
        self._static_key: tuple | None = None
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.grabGesture(Qt.GestureType.PinchGesture)
        self._update_content_height()

    # ------------------------------------------------------------------ data

    def set_data(
        self,
        duration: float,
        tracks: list[str],
        labels: list[TimelineLabel],
        frame_interval: float | None = None,
        frame_times: list[float] | None = None,
        segments: list[TimelineSegment] | None = None,
        overridden_ranges: list[tuple[str, float, float]] | None = None,
        channel_times: np.ndarray | None = None,
        channel_values: dict[str, np.ndarray] | None = None,
        highlighted_tracks: set[str] | None = None,
    ) -> None:
        self.duration = max(0.0, duration)
        self.tracks = tracks
        self.labels = labels
        self.segments = segments or []
        self.overridden_ranges = list(overridden_ranges or [])
        self.highlighted_tracks = set(highlighted_tracks or [])
        existing_ids = {label.id for label in labels}
        self.selected_label_ids.intersection_update(existing_ids)
        segment_ids = {segment.id for segment in self.segments}
        if self.selected_segment_id not in segment_ids:
            self.selected_segment_id = None
        if frame_interval is not None and frame_interval > 0:
            self.frame_interval = float(frame_interval)
        self.frame_times = (
            [float(value) for value in frame_times]
            if frame_times is not None
            else []
        )
        self._channel_times = channel_times
        self._channel_values = dict(channel_values or {})
        self._recompute_motion_spans()
        if self.selected_range is not None:
            start, end = self.selected_range
            self.selected_range = (
                self._snap_time(min(start, self.duration)),
                self._snap_time(min(end, self.duration)),
            )
            if self.selected_range[1] <= self.selected_range[0]:
                self.selected_range = None
        self._data_version += 1
        self._update_content_height()
        self.update()
        start, end = self._visible_window()
        self.visibleWindowChanged.emit(start, end, self.duration)

    def set_highlighted_tracks(self, tracks: set[str] | None) -> None:
        """Update the highlight without touching the (expensive) channel data."""
        highlighted = set(tracks or [])
        if highlighted == self.highlighted_tracks:
            return
        self.highlighted_tracks = highlighted
        self.update()

    def _recompute_motion_spans(self) -> None:
        times = self._channel_times
        self._motion_spans = {}
        if times is None:
            self._motion_cache.clear()
            return
        fresh: dict[tuple, list[tuple[float, float]]] = {}
        for name, values in self._channel_values.items():
            key = self._channel_fingerprint(name, times, values)
            spans = self._motion_cache.get(key)
            if spans is None:
                spans = motion_spans(times, values)
            fresh[key] = spans
            self._motion_spans[name] = spans
        self._motion_cache = fresh

    @staticmethod
    def _channel_fingerprint(
        name: str, times: np.ndarray, values: np.ndarray
    ) -> tuple:
        """Identify the concrete channel buffer used by the motion-span cache.

        Looking only at the first/last sample is not sufficient: moving a clip
        changes samples in the middle while normally preserving both endpoints,
        the sample count, and the duration.  In that case the old key kept the
        pre-move blue motion spans on screen.  Timeline renders create fresh
        channel arrays, so the data-buffer address is a cheap, reliable revision
        token without hashing hundreds of thousands of samples on every refresh.
        """
        if values.size == 0:
            return (name, 0, 0, 0.0)
        return (
            name,
            int(values.__array_interface__["data"][0]),
            int(values.size),
            float(times[-1]),
        )

    # ---------------------------------------------------------------- layout

    def _row_height(self) -> float:
        track_count = max(1, len(self.tracks))
        available = (
            self.height()
            - self.RULER_HEIGHT
            - self.SEGMENT_LANE_HEIGHT
        )
        return float(
            min(
                self.MAX_ROW_HEIGHT,
                max(self.MIN_ROW_HEIGHT, available / track_count),
            )
        )

    def _update_content_height(self) -> None:
        track_count = max(1, len(self.tracks))
        self.setMinimumHeight(
            self.RULER_HEIGHT
            + self.SEGMENT_LANE_HEIGHT
            + track_count * self.MIN_ROW_HEIGHT
        )
        self.updateGeometry()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        # Keep the splitter's preferred height compact. Extra joint rows
        # scroll inside the timeline pane instead of shoving the 3D view.
        visible_rows = min(max(1, len(self.tracks)), 8)
        return QSize(
            640,
            self.RULER_HEIGHT
            + self.SEGMENT_LANE_HEIGHT
            + int(visible_rows * min(self.MAX_ROW_HEIGHT, 30)),
        )

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._static_layer = None

    def _track_top(self, row: int) -> float:
        return (
            self.RULER_HEIGHT
            + self.SEGMENT_LANE_HEIGHT
            + row * self._row_height()
        )

    def _row_at_y(self, y: float) -> int | None:
        top = (
            self.RULER_HEIGHT
            + self.SEGMENT_LANE_HEIGHT
        )
        if y < top:
            return None
        row = int((y - top) // self._row_height())
        return row if 0 <= row < len(self.tracks) else None

    def _segment_lane_rect(self) -> QRectF:
        return QRectF(
            self.HEADER_WIDTH,
            self.RULER_HEIGHT,
            max(1, self.width() - self.HEADER_WIDTH - 4),
            self.SEGMENT_LANE_HEIGHT - 2,
        )

    # ------------------------------------------------------------- selection

    def selected_labels(self) -> list[TimelineLabel]:
        return sorted(
            (
                label
                for label in self.labels
                if label.id in self.selected_label_ids
            ),
            key=lambda label: label.time,
        )

    def clear_label_selection(self) -> None:
        if self.selected_label_ids:
            self.selected_label_ids.clear()
            self._last_selected_label_id = None
            self.labelSelectionChanged.emit([])
            self.update()

    def set_selected_range(
        self, start: float, end: float, emit: bool = True
    ) -> None:
        first, last = sorted((self._snap_time(start), self._snap_time(end)))
        if last <= first:
            self.clear_selected_range(emit)
            return
        self.selected_range = (first, last)
        if emit:
            self.rangeSelectionChanged.emit(self.selected_range)
        self.update()

    def clear_selected_range(self, emit: bool = True) -> None:
        if self.selected_range is None:
            return
        self.selected_range = None
        if emit:
            self.rangeSelectionChanged.emit(None)
        self.update()

    def _snap_time(self, seconds: float) -> float:
        clipped = max(0.0, min(self.duration, float(seconds)))
        if self.frame_times:
            index = bisect.bisect_left(self.frame_times, clipped)
            if index <= 0:
                return self.frame_times[0]
            if index >= len(self.frame_times):
                return self.frame_times[-1]
            previous = self.frame_times[index - 1]
            following = self.frame_times[index]
            return (
                previous
                if clipped - previous <= following - clipped
                else following
            )
        return min(
            self.duration,
            round(clipped / self.frame_interval) * self.frame_interval,
        )

    def set_current_time(self, seconds: float) -> None:
        clamped = max(0.0, min(self.duration, seconds))
        if clamped == self.current_time:
            return
        self.current_time = clamped
        self.update()
        start, end = self._visible_window()
        self.visibleWindowChanged.emit(start, end, self.duration)

    def set_zoom(self, zoom: float, anchor_time: float | None = None) -> None:
        old_start, old_end = self._visible_window()
        self.zoom = max(1.0, min(200.0, float(zoom)))
        if anchor_time is not None and self.duration > 0:
            old_span = max(1e-12, old_end - old_start)
            fraction = (float(anchor_time) - old_start) / old_span
            fraction = max(0.0, min(1.0, fraction))
            new_span = self.duration / self.zoom
            new_start = float(anchor_time) - fraction * new_span
            new_start = max(
                0.0, min(max(0.0, self.duration - new_span), new_start)
            )
            self.pan_offset = new_start + new_span / 2.0 - self.current_time
        self.update()
        start, end = self._visible_window()
        self.visibleWindowChanged.emit(start, end, self.duration)

    def set_view_fraction(self, fraction: float) -> None:
        if self.duration <= 0 or self.zoom <= 1.0:
            self.pan_offset = 0.0
        else:
            span = self.duration / self.zoom
            start = np.clip(float(fraction), 0.0, 1.0) * (self.duration - span)
            self.pan_offset = float(start + span / 2.0 - self.current_time)
        self.update()
        start, end = self._visible_window()
        self.visibleWindowChanged.emit(start, end, self.duration)

    def _visible_window(self) -> tuple[float, float]:
        if self.duration <= 0:
            return 0.0, 0.0
        span = self.duration / self.zoom
        center = self.current_time + self.pan_offset
        start = max(0.0, min(self.duration - span, center - span / 2))
        return start, start + span

    # -------------------------------------------------------------- painting

    def mousePressEvent(self, event: QMouseEvent) -> None:
        header_width = self.HEADER_WIDTH
        if self.duration > 0 and event.position().x() >= header_width:
            if event.button() == Qt.MouseButton.RightButton:
                clicked_time = self._x_to_time(
                    event.position().x(), header_width
                )
                if (
                    self.selected_range is not None
                    and self.selected_range[0]
                    <= clicked_time
                    <= self.selected_range[1]
                ):
                    self.clear_selected_range(emit=True)
                    self.selected_segment_id = None
                    event.accept()
                    return
                super().mousePressEvent(event)
                return
            if (
                self.RULER_HEIGHT
                <= event.position().y()
                <= self.RULER_HEIGHT + self.SEGMENT_LANE_HEIGHT
            ):
                segment = self._segment_at_x(event.position().x(), header_width)
                if segment is not None:
                    self.selected_segment_id = segment.id
                    self.set_selected_range(segment.start, segment.end, emit=True)
                    self.segmentSelectionChanged.emit(segment.id)
                    self.update()
                    return
            clicked_label = self._label_at_x(event.position().x(), header_width)
            if clicked_label is not None:
                modifiers = event.modifiers()
                if (
                    modifiers & Qt.KeyboardModifier.ShiftModifier
                    and self._last_selected_label_id
                ):
                    anchor = next(
                        (
                            label
                            for label in self.labels
                            if label.id == self._last_selected_label_id
                        ),
                        clicked_label,
                    )
                    low, high = sorted((anchor.time, clicked_label.time))
                    self.selected_label_ids.update(
                        label.id
                        for label in self.labels
                        if low <= label.time <= high
                    )
                elif modifiers & Qt.KeyboardModifier.ControlModifier:
                    if clicked_label.id in self.selected_label_ids:
                        self.selected_label_ids.remove(clicked_label.id)
                    else:
                        self.selected_label_ids.add(clicked_label.id)
                    self._last_selected_label_id = clicked_label.id
                else:
                    self.selected_label_ids = {clicked_label.id}
                    self._last_selected_label_id = clicked_label.id
                self.labelSelectionChanged.emit(
                    [label.id for label in self.selected_labels()]
                )
                self.seekRequested.emit(clicked_label.time)
                self.update()
                return
            clicked_time = self._x_to_time(event.position().x(), header_width)
            self._range_press_x = event.position().x()
            self._range_drag_anchor = clicked_time
            self._range_before_drag = self.selected_range
            handle = self._range_handle_at_x(
                event.position().x(), header_width
            )
            if (
                handle is None
                and self.selected_range is not None
                and self._row_at_y(event.position().y()) is not None
                and self.selected_range[0] < clicked_time < self.selected_range[1]
            ):
                self._range_drag_mode = "move"
                self._move_pointer_offset = clicked_time - self.selected_range[0]
                self._move_preview = self.selected_range
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self._range_drag_mode = handle or "new"
            if self._range_drag_mode == "new":
                self.selected_range = (clicked_time, clicked_time)
                self.selected_segment_id = None
            self.grabMouse()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._range_drag_mode is None:
            return
        header_width = self.HEADER_WIDTH
        current = self._x_to_time(event.position().x(), header_width)
        if self._range_drag_mode == "move" and self.selected_range:
            span = self.selected_range[1] - self.selected_range[0]
            start = self._snap_time(current - self._move_pointer_offset)
            start = min(max(0.0, start), max(0.0, self.duration - span))
            self._move_preview = (start, start + span)
            self.update()
        elif self._range_drag_mode == "start" and self.selected_range:
            self.set_selected_range(current, self.selected_range[1], emit=False)
        elif self._range_drag_mode == "end" and self.selected_range:
            self.set_selected_range(self.selected_range[0], current, emit=False)
        else:
            self.set_selected_range(
                self._range_drag_anchor, current, emit=False
            )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._range_drag_mode is None:
            return
        moved = abs(event.position().x() - self._range_press_x)
        self.releaseMouse()
        drag_mode = self._range_drag_mode
        self._range_drag_mode = None
        self.unsetCursor()
        preview = self._move_preview
        self._move_preview = None
        if drag_mode == "move":
            source = self._range_before_drag
            self._range_before_drag = None
            self.update()
            if moved >= 3 and source is not None and preview is not None:
                if abs(preview[0] - source[0]) >= self.frame_interval * 0.5:
                    self.clipMoveRequested.emit(source[0], source[1], preview[0])
            elif moved < 3:
                clicked = self._x_to_time(event.position().x(), self.HEADER_WIDTH)
                self.seekRequested.emit(clicked)
            return
        if moved < 3:
            clicked = self._x_to_time(event.position().x(), self.HEADER_WIDTH)
            self.selected_range = self._range_before_drag
            self._range_before_drag = None
            self.seekRequested.emit(clicked)
            self.update()
            return
        self._range_before_drag = None
        if self.selected_range is not None:
            self.set_selected_range(*self.selected_range)

    def event(self, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Gesture:
            return self._handle_pinch_gesture(event)
        if event.type() == QEvent.Type.NativeGesture:
            return self._handle_native_zoom(event)
        return super().event(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        pos_x = event.position().x()
        over_tracks = pos_x >= self.HEADER_WIDTH
        modifiers = event.modifiers()
        angle = event.angleDelta()
        pixel = event.pixelDelta()
        horizontal = abs(angle.x()) > abs(angle.y()) or abs(pixel.x()) > abs(
            pixel.y()
        )
        if modifiers & Qt.KeyboardModifier.ShiftModifier or (
            over_tracks and horizontal
        ):
            delta = angle.x() if horizontal and angle.x() != 0 else angle.y()
            if delta == 0:
                delta = pixel.x() if horizontal and pixel.x() != 0 else pixel.y()
            if delta != 0:
                self._pan_time(0.08 * (-1 if delta > 0 else 1))
                event.accept()
                return
        want_zoom = over_tracks or bool(
            modifiers & Qt.KeyboardModifier.ControlModifier
        )
        if want_zoom:
            factor = _zoom_factor_from_wheel(angle.y(), pixel.y())
            if factor is not None:
                anchor = (
                    self._x_to_time(pos_x, self.HEADER_WIDTH)
                    if over_tracks and self.duration > 0
                    else None
                )
                self.set_zoom(self.zoom * factor, anchor_time=anchor)
                event.accept()
                return
        super().wheelEvent(event)

    def _handle_pinch_gesture(self, event: QEvent) -> bool:
        if not isinstance(event, QGestureEvent):
            return False
        pinch = event.gesture(Qt.GestureType.PinchGesture)
        if not isinstance(pinch, QPinchGesture):
            return False
        if pinch.changeFlags() & QPinchGesture.ChangeFlag.ScaleFactorChanged:
            center = pinch.centerPoint()
            anchor = (
                self._x_to_time(center.x(), self.HEADER_WIDTH)
                if center.x() >= self.HEADER_WIDTH and self.duration > 0
                else None
            )
            self.set_zoom(
                self.zoom * max(0.5, min(2.0, float(pinch.scaleFactor()))),
                anchor_time=anchor,
            )
            event.accept()
            return True
        return False

    def _handle_native_zoom(self, event: QEvent) -> bool:
        if not isinstance(event, QNativeGestureEvent):
            return False
        if event.gestureType() != Qt.NativeGestureType.ZoomNativeGesture:
            return False
        value = float(event.value())
        if value == 0.0:
            return False
        factor = 1.0 + value
        if factor <= 0:
            return False
        pos = event.position()
        anchor = (
            self._x_to_time(pos.x(), self.HEADER_WIDTH)
            if pos.x() >= self.HEADER_WIDTH and self.duration > 0
            else None
        )
        self.set_zoom(self.zoom * factor, anchor_time=anchor)
        event.accept()
        return True

    def _pan_time(self, span_ratio: float) -> None:
        visible_start, visible_end = self._visible_window()
        span = visible_end - visible_start
        self.pan_offset += span * span_ratio
        self.update()
        start, end = self._visible_window()
        self.visibleWindowChanged.emit(start, end, self.duration)

    # ----------------------------------------------------------- hit testing

    def _x_to_time(self, mouse_x: float, header_width: int) -> float:
        visible_start, visible_end = self._visible_window()
        ratio = (mouse_x - header_width) / max(1, self.width() - header_width)
        return self._snap_time(
            visible_start
            + max(0.0, min(1.0, ratio)) * (visible_end - visible_start)
        )

    def _time_to_x(self, seconds: float, header_width: int) -> float:
        visible_start, visible_end = self._visible_window()
        visible_duration = max(1e-12, visible_end - visible_start)
        return header_width + (self.width() - header_width) * (
            seconds - visible_start
        ) / visible_duration

    def _range_handle_at_x(
        self, mouse_x: float, header_width: int
    ) -> str | None:
        if self.selected_range is None:
            return None
        start_x = self._time_to_x(self.selected_range[0], header_width)
        end_x = self._time_to_x(self.selected_range[1], header_width)
        if abs(mouse_x - start_x) <= 8:
            return "start"
        if abs(mouse_x - end_x) <= 8:
            return "end"
        return None

    def _label_at_x(
        self, mouse_x: float, header_width: int
    ) -> TimelineLabel | None:
        visible_start, visible_end = self._visible_window()
        if visible_end - visible_start <= 0:
            return None
        candidates: list[tuple[float, TimelineLabel]] = []
        for label in self.labels:
            if not visible_start <= label.time <= visible_end:
                continue
            distance = abs(mouse_x - self._time_to_x(label.time, header_width))
            if distance <= 8:
                candidates.append((distance, label))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _segment_at_x(
        self, mouse_x: float, header_width: int
    ) -> TimelineSegment | None:
        visible_start, visible_end = self._visible_window()
        clicked_time = self._x_to_time(mouse_x, header_width)
        if not visible_start <= clicked_time <= visible_end:
            return None
        for segment in reversed(self.segments):
            if segment.start <= clicked_time <= segment.end:
                return segment
        return None

    @staticmethod
    def label_display_text(label: TimelineLabel) -> str:
        title = label.title.strip()
        return title if title else label.id
