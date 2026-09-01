from __future__ import annotations

import numpy as np

from studio.core.trajectory import TrajectoryData

MAX_RESIZED_SCALAR_SAMPLES = 12_000_000


def _resize_segment(
    data: TrajectoryData,
    first: int,
    last: int,
    output_duration: float,
) -> tuple[TrajectoryData, int]:
    if output_duration <= 0:
        raise ValueError("output duration must be positive")
    new_count = max(3, int(round(output_duration * data.frequency.hz)) + 1)
    channel_count = max(1, len(data.channels))
    current_samples = data.frame_count * channel_count
    sample_budget = max(
        MAX_RESIZED_SCALAR_SAMPLES, int(current_samples * 1.25)
    )
    projected_frames = data.frame_count - (last - first + 1) + new_count
    projected_samples = projected_frames * channel_count
    if projected_samples > sample_budget:
        safe_count = max(
            3,
            sample_budget // channel_count
            - (data.frame_count - (last - first + 1)),
        )
        safe_duration = (safe_count - 1) / data.frequency.hz
        raise ValueError(
            "平滑后的轨迹过大，可能耗尽内存："
            f"预计 {projected_frames:,} 帧 × {channel_count} 通道。"
            f"当前工程建议该选区最长约 {safe_duration:.3f}s；"
            "请缩短时长或先降低轨迹采样率。"
        )
    source_phase = np.linspace(first, last, new_count)
    rebuilt = {
        name: np.r_[
            values[:first],
            np.interp(source_phase, np.arange(data.frame_count), values),
            values[last + 1 :],
        ]
        for name, values in data.channels.items()
    }
    # The caller rewrites its target channels and recomputes velocities once at
    # the end. Avoid a full redundant gradient pass over every large channel.
    return _from_channels(data, rebuilt, recompute_velocities=False), first + new_count - 1


def max_safe_segment_duration(
    data: TrajectoryData, start: float, end: float
) -> float:
    """Largest resize duration allowed by the editor's peak-memory budget."""
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    channel_count = max(1, len(data.channels))
    current_samples = data.frame_count * channel_count
    budget = max(MAX_RESIZED_SCALAR_SAMPLES, int(current_samples * 1.25))
    outside_frames = data.frame_count - (last - first + 1)
    allowed_frames = max(3, budget // channel_count - outside_frames)
    return max(2 * data.frequency.median_dt, (allowed_frames - 1) / data.frequency.hz)


def _take(data: TrajectoryData, indices: np.ndarray) -> TrajectoryData:
    channels = {name: values[indices] for name, values in data.channels.items()}
    return _from_channels(data, channels)


def _insert_channels(
    data: TrajectoryData,
    after_frame: int,
    inserted: dict[str, np.ndarray],
    count: int,
) -> TrajectoryData:
    channels = {
        name: np.r_[values[: after_frame + 1], inserted[name], values[after_frame + 1 :]]
        for name, values in data.channels.items()
    }
    expected = data.frame_count + count
    if any(len(values) != expected for values in channels.values()):
        raise RuntimeError("insert operation produced inconsistent channel lengths")
    return _from_channels(data, channels)


def _insert_selected_channels(
    data: TrajectoryData,
    after_frame: int,
    inserted: dict[str, np.ndarray],
    count: int,
    selected: set[str],
) -> TrajectoryData:
    """Insert into selected lanes while unselected lanes keep their timing."""
    channels: dict[str, np.ndarray] = {}
    for name, values in data.channels.items():
        if name in selected:
            channels[name] = np.r_[
                values[: after_frame + 1],
                inserted[name],
                values[after_frame + 1 :],
            ]
        else:
            channels[name] = np.r_[values, np.full(count, values[-1])]
    return _from_channels(data, channels)


def _from_channels(
    data: TrajectoryData,
    channels: dict[str, np.ndarray],
    *,
    recompute_velocities: bool = True,
) -> TrajectoryData:
    count = len(next(iter(channels.values())))
    times = np.arange(count, dtype=np.float64) / data.frequency.hz
    result = TrajectoryData(
        times,
        channels,
        list(data.position_channels),
        dict(data.velocity_channels),
        dict(data.effort_channels),
        dict(data.metadata),
    )
    if recompute_velocities:
        result.recompute_velocities()
    return result


