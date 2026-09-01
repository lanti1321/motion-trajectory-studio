from __future__ import annotations

import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.editing.operation_data import _from_channels, _insert_channels, _insert_selected_channels, _take


def trim(data: TrajectoryData, start: float, end: float, channels: list[str] | None = None) -> TrajectoryData:
    if not 0 <= start < end <= data.duration:
        raise ValueError("trim range is outside the trajectory")
    start_frame = data.nearest_frame(start)
    end_frame = data.nearest_frame(end)
    selected = set(channels or data.position_channels)
    if selected.issuperset(data.position_channels):
        return _take(data, np.arange(start_frame, end_frame + 1))
    result = data.copy()
    for name in selected:
        if name in result.channels:
            result.channels[name][:start_frame] = result.channels[name][start_frame]
            result.channels[name][end_frame + 1 :] = result.channels[name][end_frame]
    result.recompute_velocities()
    return result


def delete_range(data: TrajectoryData, start: float, end: float, channels: list[str] | None = None) -> TrajectoryData:
    if not 0 <= start < end <= data.duration:
        raise ValueError("delete range is outside the trajectory")
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    selected = set(channels or data.position_channels)
    if selected.issuperset(data.position_channels):
        indices = np.r_[np.arange(first), np.arange(last + 1, data.frame_count)]
        return _take(data, indices)
    removed = last - first + 1
    rebuilt = {name: values.copy() for name, values in data.channels.items()}
    for name in selected:
        if name not in rebuilt:
            continue
        values = data.channels[name]
        rebuilt[name] = np.r_[
            values[:first], values[last + 1 :], np.full(removed, values[-1])
        ]
    return _from_channels(data, rebuilt)


def hold(data: TrajectoryData, at: float, duration: float, channels: list[str] | None = None) -> TrajectoryData:
    if duration <= 0:
        raise ValueError("hold duration must be positive")
    frame = data.nearest_frame(at)
    count = max(1, int(round(duration * data.frequency.hz)))
    selected = set(channels or data.channels)
    inserted = {name: np.full(count, data.channels[name][frame]) for name in selected if name in data.channels}
    if selected.issuperset(data.position_channels):
        complete = {
            name: inserted.get(name, np.full(count, values[frame]))
            for name, values in data.channels.items()
        }
        return _insert_channels(data, frame, complete, count)
    return _insert_selected_channels(data, frame, inserted, count, selected)


def loop(
    data: TrajectoryData,
    start: float,
    end: float,
    repetitions: int | None = None,
    added_duration: float | None = None,
    channels: list[str] | None = None,
) -> TrajectoryData:
    if not 0 <= start < end <= data.duration:
        raise ValueError("loop range is outside the trajectory")
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    if last <= first:
        raise ValueError("loop must contain at least two frames")
    segment_length = last - first
    if repetitions is None:
        if added_duration is None or added_duration <= 0:
            raise ValueError("provide repetitions or added_duration")
        repetitions = max(1, int(round(added_duration / (segment_length / data.frequency.hz))))
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    selected = set(channels or data.position_channels)
    inserted = {
        name: np.tile(data.channels[name][first + 1 : last + 1], repetitions)
        for name in selected if name in data.channels
    }
    count = segment_length * repetitions
    if selected.issuperset(data.position_channels):
        complete = {
            name: inserted.get(name, np.tile(values[first + 1 : last + 1], repetitions))
            for name, values in data.channels.items()
        }
        return _insert_channels(data, first, complete, count)
    return _insert_selected_channels(data, first, inserted, count, selected)


def copy_segment(
    data: TrajectoryData,
    start: float,
    end: float,
    insert_at: float,
    channels: list[str] | None = None,
) -> TrajectoryData:
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    destination = data.nearest_frame(insert_at)
    if last <= first:
        raise ValueError("copied segment must contain at least two frames")
    selected = set(channels or data.position_channels)
    inserted = {
        name: data.channels[name][first + 1 : last + 1].copy()
        for name in selected if name in data.channels
    }
    count = last - first
    if selected.issuperset(data.position_channels):
        complete = {
            name: inserted.get(name, values[first + 1 : last + 1].copy())
            for name, values in data.channels.items()
        }
        return _insert_channels(data, destination, complete, count)
    return _insert_selected_channels(data, destination, inserted, count, selected)


def move_segment(
    data: TrajectoryData,
    start: float,
    end: float,
    destination: float,
    channels: list[str] | None = None,
) -> TrajectoryData:
    """Move selected joint samples to a stationary interval without resizing time.

    This deliberately behaves like moving a video clip into an empty lane: the
    source becomes a hold and the destination is overwritten.  It never shifts
    unrelated joints or silently blends an incompatible destination.
    """
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    target_first = data.nearest_frame(destination)
    count = last - first + 1
    target_last = target_first + count - 1
    selected = list(channels or data.position_channels)
    if last <= first:
        raise ValueError("移动片段至少需要包含两个采样帧")
    if not selected or any(name not in data.position_channels for name in selected):
        raise ValueError("移动片段包含不存在或非位置类型的关节通道")
    if target_first < 0 or target_last >= data.frame_count:
        raise ValueError("目标片段超出轨迹末尾，请向左拖动后重试")
    if target_first == first:
        return data.copy()

    # A destination is considered empty when its selected joints remain nearly
    # stationary.  Use a noise-aware velocity threshold so recorded encoder
    # jitter does not make every hold interval look occupied.
    # Overlap is valid and is the normal way to nudge a gesture.  Only inspect
    # the newly occupied portion; the intersection already belongs to the
    # source clip and must not be mistaken for destination activity.
    target_indices = np.arange(target_first, target_last + 1)
    new_target_indices = target_indices[
        (target_indices < first) | (target_indices > last)
    ]
    occupied: list[str] = []
    for name in selected:
        if new_target_indices.size < 2:
            continue
        values = data.channels[name][new_target_indices]
        times = data.times[new_target_indices]
        # The newly occupied range is contiguous on either side for equal-size
        # moves.  A defensive split avoids taking a gradient across a gap.
        breaks = np.flatnonzero(np.diff(new_target_indices) > 1) + 1
        chunks = np.split(np.arange(new_target_indices.size), breaks)
        moving = False
        for chunk in chunks:
            if chunk.size < 2:
                continue
            velocity = np.abs(
                np.gradient(values[chunk], times[chunk], edge_order=1)
            )
            if float(np.percentile(velocity, 95)) > 0.02:
                moving = True
                break
        if moving:
            occupied.append(name)
    if occupied:
        shown = "、".join(occupied[:6])
        suffix = "…" if len(occupied) > 6 else ""
        raise ValueError(
            f"目标区间并非空白：{shown}{suffix} 仍在运动。"
            "请选择这些关节保持静止的时间段。"
        )

    result = data.copy()
    snapshots = {
        name: data.channels[name][first : last + 1].copy()
        for name in selected
    }
    for name in selected:
        values = result.channels[name]
        hold_value = values[first] if target_first > last else values[last]
        values[first : last + 1] = hold_value
        values[target_first : target_last + 1] = snapshots[name]
    result.recompute_velocities()
    return result
