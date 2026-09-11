"""Measure timeline paint cost on a real trajectory.

Usage: python scripts/benchmark_timeline.py [csv_path]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from studio.io.trajectory_io import load_trajectory  # noqa: E402
from studio.ui.timeline_widget import TimelineWidget  # noqa: E402


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "../action_end5.csv"
    application = QApplication.instance() or QApplication([])
    data = load_trajectory(path)
    tracks = list(data.position_channels)
    print(f"{path}: {data.frame_count} samples x {len(tracks)} channels, "
          f"{data.duration:.1f}s")

    timeline = TimelineWidget()
    timeline.resize(1200, 560)

    started = time.perf_counter()
    timeline.set_data(
        data.duration,
        tracks,
        [],
        data.frequency.median_dt,
        data.times.tolist(),
        channel_times=data.times,
        channel_values={name: data.channels[name] for name in tracks},
    )
    print(f"set_data (motion detection): {time.perf_counter() - started:.3f}s")
    blocks = sum(len(spans) for spans in timeline._motion_spans.values())
    print(f"motion blocks across all channels: {blocks}")

    started = time.perf_counter()
    timeline._render_static_layer()
    print(f"first full paint: {time.perf_counter() - started:.3f}s")

    target = QPixmap(1200, 560)
    started = time.perf_counter()
    for _ in range(60):
        timeline.render(target)
    elapsed = time.perf_counter() - started
    print(f"60 cached repaints: {elapsed:.3f}s ({elapsed / 60 * 1000:.2f}ms each)")

    started = time.perf_counter()
    for step in range(60):
        timeline.set_current_time(data.duration * step / 60)
        timeline.render(target)
    elapsed = time.perf_counter() - started
    print(f"60 playhead frames: {elapsed:.3f}s ({elapsed / 60 * 1000:.2f}ms each)")


if __name__ == "__main__":
    main()
