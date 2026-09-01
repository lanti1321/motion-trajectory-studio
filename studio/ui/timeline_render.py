"""Vectorised helpers that keep timeline painting independent of sample count.

A recorded trajectory easily reaches hundreds of thousands of samples, while the
timeline never has more horizontal pixels than the widget is wide.  Everything in
this module reduces a channel to at most one value pair per pixel column so that
painting cost depends on widget size instead of trajectory length.
"""

from __future__ import annotations

import numpy as np

Envelope = tuple[np.ndarray, np.ndarray, np.ndarray]


def box_filter(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average with edge padding, O(n) regardless of window size."""
    if window <= 1 or values.size < 2:
        return np.asarray(values, dtype=float)
    window = min(int(window), values.size)
    if window <= 1:
        return np.asarray(values, dtype=float)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(np.asarray(values, dtype=float), (left, right), mode="edge")
    cumulative = np.cumsum(np.insert(padded, 0, 0.0))
    return (cumulative[window:] - cumulative[:-window]) / window


def boolean_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return half-open [start, stop) index pairs of contiguous True runs."""
    if mask.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    padded = np.r_[False, mask.astype(bool), False].astype(np.int8)
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def column_envelope(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
    columns: int,
) -> Envelope | None:
    """Reduce a channel to per-pixel-column minima and maxima.

    Returns the column indices that contain at least one sample together with the
    minimum and maximum value inside each of those columns, or ``None`` when the
    requested window holds no samples.
    """
    columns = max(1, int(columns))
    if times is None or values is None:
        return None
    if times.size < 2 or values.size != times.size or end <= start:
        return None

    edges = np.searchsorted(
        times, np.linspace(start, end, columns + 1), side="left"
    )
    span_starts = edges[:-1]
    span_stops = edges[1:]
    populated = span_stops > span_starts
    if not np.any(populated):
        return None

    column_index = np.flatnonzero(populated)
    reduce_at = span_starts[populated]
    # ``reduceat`` folds the final segment to the end of the array, so trim the
    # view to the last populated column instead of the full channel.
    window = values[: int(span_stops[populated][-1])]
    minima = np.minimum.reduceat(window, reduce_at)
    maxima = np.maximum.reduceat(window, reduce_at)
    return column_index, minima, maxima


def spans_to_columns(
    spans: list[tuple[float, float]],
    start: float,
    end: float,
    columns: int,
    minimum_gap_columns: int = 3,
) -> list[tuple[int, int]]:
    """Clip time spans to the visible window and convert them to column ranges.

    Pauses thinner than ``minimum_gap_columns`` are absorbed into the surrounding
    block: at a zoomed-out view they would render as single-pixel slivers that
    read as noise rather than as a real pause.  Zooming in makes them reappear.
    """
    columns = max(1, int(columns))
    if end <= start or not spans:
        return []
    scale = columns / (end - start)
    result: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        left = max(start, span_start)
        right = min(end, span_end)
        if right <= left:
            continue
        first = max(0, int((left - start) * scale))
        last = min(columns, max(first + 1, int(np.ceil((right - start) * scale))))
        if result and first - result[-1][1] < minimum_gap_columns:
            result[-1] = (result[-1][0], max(result[-1][1], last))
        else:
            result.append((first, last))
    return result
