from __future__ import annotations

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap

from studio.ui.i18n import ui_text
from studio.ui.timeline_render import column_envelope, spans_to_columns

TRACK_COLORS = ["#4fc3f7"]


class TimelinePaintMixin:
    """Cached timeline background, segments, curves, labels, and overlays."""

    def _static_cache_key(self) -> tuple:
        visible_start, visible_end = self._visible_window()
        return (
            self.width(),
            self.height(),
            self._data_version,
            round(visible_start, 6),
            round(visible_end, 6),
            tuple(sorted(self.highlighted_tracks)),
            tuple(sorted(self.selected_label_ids)),
            self.selected_segment_id,
            tuple(
                (segment.id, segment.start, segment.end, segment.title,
                 segment.color)
                for segment in self.segments
            ),
            tuple(self.overridden_ranges),
            tuple((label.id, label.time, label.title) for label in self.labels),
        )

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        key = self._static_cache_key()
        if self._static_layer is None or self._static_key != key:
            self._static_layer = self._render_static_layer()
            self._static_key = key
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._static_layer)
        if self.duration > 0:
            self._paint_overlay(painter)

    def _render_static_layer(self) -> QPixmap:
        ratio = self.devicePixelRatioF()
        pixmap = QPixmap(
            max(1, int(self.width() * ratio)),
            max(1, int(self.height() * ratio)),
        )
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QColor("#202124"))
        painter = QPainter(pixmap)
        try:
            self._paint_static(painter)
        finally:
            painter.end()
        return pixmap

    def _paint_static(self, painter: QPainter) -> None:
        header_width = self.HEADER_WIDTH
        ruler_height = self.RULER_HEIGHT
        painter.setPen(QColor("#9aa0a6"))
        painter.drawText(8, 16, ui_text("时间"))
        painter.drawText(8, ruler_height + 16, ui_text("分段"))
        lane_rect = self._segment_lane_rect()
        painter.fillRect(lane_rect, QColor("#2b3140"))
        painter.setPen(QColor("#59697d"))
        painter.drawRect(lane_rect)

        if self.duration <= 0:
            return
        visible_start, visible_end = self._visible_window()
        visible_duration = max(1e-12, visible_end - visible_start)
        timeline_width = max(1, self.width() - header_width)

        for segment in self.segments:
            left_time = max(visible_start, segment.start)
            right_time = min(visible_end, segment.end)
            if right_time <= left_time:
                continue
            left_x = self._time_to_x(left_time, header_width)
            right_x = self._time_to_x(right_time, header_width)
            color = QColor(segment.color)
            color.setAlpha(210 if segment.id == self.selected_segment_id else 150)
            background = QColor(segment.color)
            background.setAlpha(38 if segment.id == self.selected_segment_id else 22)
            painter.fillRect(
                QRectF(
                    left_x,
                    self.RULER_HEIGHT + self.SEGMENT_LANE_HEIGHT,
                    right_x - left_x,
                    max(
                        1.0,
                        self.height()
                        - self.RULER_HEIGHT
                        - self.SEGMENT_LANE_HEIGHT,
                    ),
                ),
                background,
            )
            painter.fillRect(
                QRectF(left_x, lane_rect.top(), right_x - left_x,
                       lane_rect.height()),
                color,
            )
            painter.setPen(QColor("#eceff1"))
            painter.drawText(
                int(left_x + 4),
                int(lane_rect.top() + lane_rect.height() - 6),
                segment.title.strip() or segment.id,
            )

        row_height = self._row_height()
        for row, name in enumerate(self.tracks or [ui_text("轨迹")]):
            top = self._track_top(row)
            if top > self.height():
                break
            highlighted = name in self.highlighted_tracks
            painter.setPen(
                QColor("#ffffff") if highlighted else QColor("#d5d7da")
            )
            painter.drawText(8, int(top + row_height * 0.7), name)
            track_rect = QRectF(
                header_width,
                top + 1,
                timeline_width - 4,
                row_height - 2,
            )
            painter.fillRect(track_rect, QColor("#252b34"))
            self._paint_track_content(
                painter, name, row, track_rect, visible_start, visible_end
            )
            for channel, start, end in self.overridden_ranges:
                if channel != name:
                    continue
                left_time = max(visible_start, start)
                right_time = min(visible_end, end)
                if right_time <= left_time:
                    continue
                left_x = self._time_to_x(left_time, header_width)
                right_x = self._time_to_x(right_time, header_width)
                override = QColor("#ffb300")
                override.setAlpha(105)
                override_rect = QRectF(
                    left_x,
                    track_rect.top(),
                    max(2.0, right_x - left_x),
                    track_rect.height(),
                )
                painter.fillRect(override_rect, override)
                painter.setPen(QPen(QColor("#ffd54f"), 2))
                painter.drawLine(
                    int(left_x),
                    int(track_rect.top() + 1),
                    int(right_x),
                    int(track_rect.top() + 1),
                )
            painter.setPen(
                QPen(
                    QColor("#90caf9") if highlighted else QColor("#59697d"),
                    2 if highlighted else 1,
                )
            )
            painter.drawRect(track_rect)

        tick_count = 10
        painter.setPen(QColor("#9aa0a6"))
        for tick in range(tick_count + 1):
            x = header_width + timeline_width * tick / tick_count
            seconds = visible_start + visible_duration * tick / tick_count
            painter.drawLine(int(x), 18, int(x), self.height())
            painter.drawText(int(x + 2), 13, f"{seconds:.2f}s")

        for label in self.labels:
            if not visible_start <= label.time <= visible_end:
                continue
            x = self._time_to_x(label.time, header_width)
            selected = label.id in self.selected_label_ids
            painter.setPen(
                QPen(
                    QColor("#ff6d00") if selected else QColor("#fbbc04"),
                    4 if selected else 2,
                )
            )
            painter.drawLine(int(x), 0, int(x), self.height())
            painter.setPen(QColor("#fbbc04"))
            painter.drawText(
                int(x + 3),
                ruler_height + 12,
                self.label_display_text(label),
            )
            if selected:
                painter.setBrush(QColor("#ff6d00"))
                painter.drawEllipse(QPointF(x, 5), 5, 5)
                painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_overlay(self, painter: QPainter) -> None:
        header_width = self.HEADER_WIDTH
        ruler_height = self.RULER_HEIGHT
        if self.selected_range is not None:
            start, end = self.selected_range
            visible_start, visible_end = self._visible_window()
            left_time = max(visible_start, start)
            right_time = min(visible_end, end)
            if right_time > left_time:
                left_x = self._time_to_x(left_time, header_width)
                right_x = self._time_to_x(right_time, header_width)
                painter.setPen(QPen(QColor("#ff8f00"), 2))
                painter.setBrush(QColor(255, 143, 0, 45))
                painter.drawRect(
                    QRectF(
                        left_x,
                        ruler_height,
                        right_x - left_x,
                        self.height() - ruler_height,
                    )
                )
                painter.setBrush(QColor("#ff8f00"))
                painter.drawRect(QRectF(left_x - 4, ruler_height, 8, 18))
                painter.drawRect(QRectF(right_x - 4, ruler_height, 8, 18))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QColor("#fff3e0"))
                painter.drawText(
                    int(left_x + 5),
                    self.height() - 6,
                    f"{start:.3f}s – {end:.3f}s  ({end - start:.3f}s)",
                )
        if self._move_preview is not None:
            start, end = self._move_preview
            left_x = self._time_to_x(start, header_width)
            right_x = self._time_to_x(end, header_width)
            pen = QPen(QColor("#66ff99"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(102, 255, 153, 35))
            painter.drawRect(
                QRectF(
                    left_x,
                    ruler_height + self.SEGMENT_LANE_HEIGHT,
                    right_x - left_x,
                    self.height() - ruler_height - self.SEGMENT_LANE_HEIGHT,
                )
            )
        x = self._time_to_x(self.current_time, header_width)
        painter.setPen(QPen(QColor("#ff5252"), 2))
        painter.drawLine(int(x), 0, int(x), self.height())

    def _paint_track_content(
        self,
        painter: QPainter,
        name: str,
        row: int,
        track_rect: QRectF,
        visible_start: float,
        visible_end: float,
    ) -> None:
        columns = max(1, int(track_rect.width()))
        left = track_rect.left()
        color = QColor(TRACK_COLORS[row % len(TRACK_COLORS)])
        block = QColor(color)
        block.setAlpha(185)
        for first, last in spans_to_columns(
            self._motion_spans.get(name, []), visible_start, visible_end, columns
        ):
            painter.fillRect(
                QRectF(
                    left + first,
                    track_rect.top() + 1,
                    max(1.0, float(last - first)),
                    track_rect.height() - 2,
                ),
                block,
            )

        values = self._channel_values.get(name)
        envelope = column_envelope(
            self._channel_times, values, visible_start, visible_end, columns
        )
        if envelope is None:
            return
        column_index, minima, maxima = envelope
        low = float(np.min(minima))
        high = float(np.max(maxima))
        baseline = track_rect.bottom() - 2
        usable = max(1.0, track_rect.height() - 4)
        if high - low < 1e-6:
            # A channel that holds still inside the visible window reads better
            # as a centred line than as one pinned to the bottom of the row.
            top_y = np.full(column_index.size, baseline - usable / 2)
            bottom_y = top_y
        else:
            span = high - low
            top_y = baseline - (maxima - low) / span * usable
            bottom_y = baseline - (minima - low) / span * usable
        xs = left + column_index
        lines = [
            QLineF(x, float(y0), x, float(max(y1, y0 + 0.8)))
            for x, y0, y1 in zip(xs, top_y, bottom_y)
        ]
        pen = QPen(QColor("#eceff1"), 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLines(lines)

    # ---------------------------------------------------------------- events

