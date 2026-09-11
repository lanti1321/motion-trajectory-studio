"""Detection of the intervals where a joint channel is actually moving.

Raw teleoperation data is noisy enough that a naive per-sample threshold breaks a
single gesture into hundreds of fragments.  The detector below smooths the signal,
applies a hysteresis threshold, and merges short gaps so that one physical motion
becomes one block on the timeline.
"""

from __future__ import annotations

import numpy as np

from studio.ui.timeline_render import boolean_runs, box_filter


def motion_spans(
    times: np.ndarray,
    values: np.ndarray,
    *,
    velocity_fraction: float = 0.06,
    absolute_floor: float = 1e-4,
    smoothing_seconds: float = 0.05,
    merge_gap_seconds: float = 0.20,
    minimum_duration: float = 0.08,
    maximum_spans: int = 400,
) -> list[tuple[float, float]]:
    """Return time intervals where the channel is actively moving."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 3 or values.size != times.size:
        return []

    sample_interval = float(np.median(np.diff(times)))
    if not np.isfinite(sample_interval) or sample_interval <= 0:
        return []

    window = max(1, int(round(smoothing_seconds / sample_interval)))
    smoothed = box_filter(values, window)
    velocity = box_filter(
        np.abs(np.gradient(smoothed, times, edge_order=1)), window
    )

    # Estimate the active-motion scale from samples that are actually above the
    # stationary noise floor.  Using the percentile of the whole recording made
    # sparse channels (especially grippers) disappear: a short open/close action
    # in a long recording leaves the global 95th percentile at exactly zero.
    active_velocity = velocity[
        np.isfinite(velocity) & (velocity > absolute_floor)
    ]
    if active_velocity.size == 0:
        return []
    reference = float(np.percentile(active_velocity, 90))
    enter_threshold = max(absolute_floor, reference * velocity_fraction)
    exit_threshold = enter_threshold * 0.5

    candidate = velocity >= exit_threshold
    starts, stops = boolean_runs(candidate)
    if starts.size == 0:
        return []

    # Keep only runs that peak above the enter threshold, so noise riding just
    # over the exit threshold never becomes a block on its own.
    seeds = np.r_[0, np.cumsum(velocity >= enter_threshold)]
    confirmed = (seeds[stops] - seeds[starts]) > 0
    starts = starts[confirmed]
    stops = stops[confirmed]
    if starts.size == 0:
        return []

    spans = [
        (float(times[start]), float(times[min(stop, times.size - 1)]))
        for start, stop in zip(starts, stops)
    ]
    spans = _merge_and_filter(spans, merge_gap_seconds, minimum_duration)

    # Very fragmented channels stay readable by progressively coarsening the
    # merge gap rather than by dropping information entirely.
    gap = merge_gap_seconds
    while len(spans) > maximum_spans and gap < times[-1] - times[0]:
        gap *= 2.0
        spans = _merge_and_filter(spans, gap, minimum_duration)
    return spans


def _merge_and_filter(
    spans: list[tuple[float, float]],
    merge_gap: float,
    minimum_duration: float,
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [
        (start, end)
        for start, end in merged
        if end - start >= minimum_duration
    ]
