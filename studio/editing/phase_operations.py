from __future__ import annotations

import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.editing.operation_data import _insert_channels, _insert_selected_channels


def phase_aligned_copy_segment(
    data: TrajectoryData,
    start: float,
    end: float,
    insert_at: float,
    channels: list[str] | None = None,
) -> TrajectoryData:
    """Insert a periodic clip with one shared phase and short no-slow crossfades."""
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    destination = data.nearest_frame(insert_at)
    count = last - first
    if count < 12:
        raise ValueError("高频相位拼接片段至少需要 12 帧")
    selected = set(channels or data.position_channels)
    names = [name for name in data.position_channels if name in data.channels and name in selected]
    if not names:
        raise ValueError("高频相位拼接没有位置通道")
    clip = np.column_stack(
        [data.channels[name][first + 1 : last + 1] for name in names]
    )
    scale = np.maximum(np.ptp(clip, axis=0), 1e-6)
    entry_position = np.asarray(
        [data.channels[name][destination] for name in names]
    )
    entry_velocity = np.asarray(
        [
            data.channels[name][destination]
            - data.channels[name][max(0, destination - 1)]
            for name in names
        ]
    )
    candidate_velocity = clip - np.roll(clip, 1, axis=0)
    entry_score = np.mean(((clip - entry_position) / scale) ** 2, axis=1)
    entry_score += 0.35 * np.mean(
        ((candidate_velocity - entry_velocity) / scale) ** 2, axis=1
    )
    if destination + 1 < data.frame_count:
        exit_position = np.asarray(
            [data.channels[name][destination + 1] for name in names]
        )
        rolled_last = np.roll(clip, 1, axis=0)
        entry_score += 0.5 * np.mean(
            ((rolled_last - exit_position) / scale) ** 2, axis=1
        )
    offset = int(np.argmin(entry_score))
    inserted: dict[str, np.ndarray] = {}
    # A common offset is essential: independently aligning joints destroys the
    # bimanual relative pose even when every individual curve looks smooth.
    for name in names:
        values = data.channels[name]
        raw = values[first + 1 : last + 1].copy()
        inserted[name] = np.roll(raw, -offset)

    representative = np.mean((clip - np.mean(clip, axis=0)) / scale, axis=1)
    spectrum = np.abs(np.fft.rfft(representative))
    minimum_bin = max(1, int(np.ceil(count * data.frequency.median_dt / 1.0)))
    if len(spectrum) > minimum_bin:
        peak_bin = minimum_bin + int(np.argmax(spectrum[minimum_bin:]))
        period = max(8, int(round(count / max(1, peak_bin))))
    else:
        period = max(8, count // 4)
    blend_count = max(
        6,
        min(count // 4, period // 4, int(round(0.08 * data.frequency.hz))),
    )
    blend_count = min(blend_count, count // 3)
    u = np.linspace(0.0, 1.0, blend_count)
    smooth = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
    for name in names:
        values = inserted[name]
        entry_delta = data.channels[name][destination] - values[0]
        values[:blend_count] += entry_delta * (1.0 - smooth)
        if destination + 1 < data.frame_count:
            exit_delta = data.channels[name][destination + 1] - values[-1]
            values[-blend_count:] += exit_delta * smooth
    if selected.issuperset(data.position_channels):
        complete = {
            name: inserted.get(name, values[first + 1 : last + 1].copy())
            for name, values in data.channels.items()
        }
        return _insert_channels(data, destination, complete, count)
    return _insert_selected_channels(data, destination, inserted, count, selected)


def high_speed_phase_smooth(
    data: TrajectoryData,
    start: float,
    end: float,
    channels: list[str] | None = None,
) -> TrajectoryData:
    """Repair a high-speed interval by shared-period continuation and crossfade."""
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    count = last - first + 1
    selected = list(channels or data.position_channels)
    selected = [name for name in selected if name in data.channels]
    if count < 8 or not selected:
        raise ValueError("高速相位平滑选区至少需要 8 帧和一个位置通道")
    available = min(first, data.frame_count - last - 1)
    maximum_period = min(available, int(round(data.frequency.hz)))
    if maximum_period < 8:
        raise ValueError("选区前后没有足够上下文用于高速相位分析")
    context = min(available, max(count * 4, maximum_period))
    left_matrix = np.column_stack(
        [data.channels[name][first - context : first] for name in selected]
    )
    right_matrix = np.column_stack(
        [data.channels[name][last + 1 : last + 1 + context] for name in selected]
    )
    power = np.zeros(context // 2 + 1)
    for matrix in (left_matrix, right_matrix):
        centered = matrix - np.mean(matrix, axis=0)
        scale = np.maximum(np.std(centered, axis=0), 1e-8)
        spectrum = np.fft.rfft(centered / scale, axis=0)
        power += np.sum(np.abs(spectrum) ** 2, axis=1)
    minimum_frequency = max(1, int(np.floor(context / maximum_period)))
    maximum_frequency = min(len(power) - 1, context // 8)
    if maximum_frequency < minimum_frequency:
        raise ValueError("上下文过短，无法估计高速运动周期")
    peak_bin = minimum_frequency + int(
        np.argmax(power[minimum_frequency : maximum_frequency + 1])
    )
    period = int(round(context / peak_bin))
    period = int(np.clip(period, 8, maximum_period))
    indices = np.arange(count)
    left_indices = first - period + (indices % period)
    right_indices = last + 1 + ((indices - count) % period)
    u = np.linspace(0.0, 1.0, count)
    weight = 0.5 - 0.5 * np.cos(np.pi * u)
    result = data.copy()
    for name in selected:
        left_prediction = data.channels[name][left_indices]
        right_prediction = data.channels[name][right_indices]
        result.channels[name][first : last + 1] = (
            (1.0 - weight) * left_prediction + weight * right_prediction
        )
    result.metadata = dict(result.metadata)
    result.metadata["last_high_speed_period_frames"] = period
    result.recompute_velocities()
    return result


