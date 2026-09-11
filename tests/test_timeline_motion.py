from __future__ import annotations

import numpy as np
import pytest

from studio.ui.timeline_motion import motion_spans
from studio.ui.timeline_render import column_envelope, spans_to_columns


def test_motion_spans_detects_active_interval() -> None:
    times = np.linspace(0.0, 4.0, 401)
    values = np.zeros_like(times)
    values[100:201] = np.linspace(0.0, 1.0, 101)
    spans = motion_spans(times, values)
    assert len(spans) == 1
    # Smoothing widens the detected edges slightly on either side.
    assert spans[0][0] == pytest.approx(1.0, abs=0.1)
    assert spans[0][1] == pytest.approx(2.0, abs=0.1)


def test_motion_spans_ignores_static_channel() -> None:
    times = np.linspace(0.0, 2.0, 201)
    values = np.ones_like(times) * 0.5
    assert motion_spans(times, values) == []


def test_motion_spans_detects_sparse_gripper_actions_in_long_recording() -> None:
    times = np.linspace(0.0, 130.0, 130_001)
    values = np.zeros_like(times)
    values[40_000:40_501] = np.linspace(0.0, 0.8, 501)
    values[40_501:80_000] = 0.8
    values[80_000:80_501] = np.linspace(0.8, 0.0, 501)

    spans = motion_spans(times, values)

    assert len(spans) == 2
    assert spans[0][0] == pytest.approx(40.0, abs=0.15)
    assert spans[0][1] == pytest.approx(40.5, abs=0.15)
    assert spans[1][0] == pytest.approx(80.0, abs=0.15)
    assert spans[1][1] == pytest.approx(80.5, abs=0.15)


def test_motion_spans_does_not_fragment_a_noisy_gesture() -> None:
    generator = np.random.default_rng(0)
    times = np.linspace(0.0, 20.0, 20_000)
    values = np.zeros_like(times)
    values[5_000:15_000] = np.linspace(0.0, 1.2, 10_000)
    values += generator.normal(0.0, 2e-4, values.size)
    spans = motion_spans(times, values)
    assert 1 <= len(spans) <= 3
    assert spans[0][0] < 5.5
    assert spans[-1][1] > 14.5


def test_column_envelope_is_bounded_by_column_count() -> None:
    times = np.linspace(0.0, 100.0, 160_000)
    values = np.sin(times)
    envelope = column_envelope(times, values, 0.0, 100.0, 800)
    assert envelope is not None
    index, minima, maxima = envelope
    assert index.size <= 800
    assert np.all(minima <= maxima)
    assert minima.min() == pytest.approx(values.min(), abs=1e-6)
    assert maxima.max() == pytest.approx(values.max(), abs=1e-6)


def test_column_envelope_respects_the_visible_window() -> None:
    times = np.linspace(0.0, 10.0, 1_001)
    values = times.copy()
    envelope = column_envelope(times, values, 4.0, 6.0, 200)
    assert envelope is not None
    _, minima, maxima = envelope
    assert minima.min() == pytest.approx(4.0, abs=0.05)
    assert maxima.max() == pytest.approx(6.0, abs=0.05)


def test_column_envelope_returns_none_outside_the_data() -> None:
    times = np.linspace(0.0, 1.0, 100)
    values = np.zeros_like(times)
    assert column_envelope(times, values, 5.0, 6.0, 100) is None


def test_spans_to_columns_clips_to_the_window() -> None:
    columns = spans_to_columns([(-1.0, 0.5), (1.5, 9.0)], 0.0, 2.0, 100)
    assert columns[0][0] == 0
    assert columns[0][1] == 25
    assert columns[-1][1] == 100


def test_spans_to_columns_absorbs_subpixel_gaps() -> None:
    spans = [(0.0, 1.0), (1.01, 2.0), (5.0, 6.0)]
    merged = spans_to_columns(spans, 0.0, 100.0, 100)
    assert merged == [(0, 2), (5, 6)]


def test_spans_to_columns_keeps_gaps_when_zoomed_in() -> None:
    spans = [(0.0, 1.0), (1.05, 2.0)]
    assert len(spans_to_columns(spans, 0.0, 100.0, 100)) == 1
    assert len(spans_to_columns(spans, 0.0, 4.0, 800)) == 2
