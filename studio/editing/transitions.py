from __future__ import annotations

import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.editing.operation_data import _from_channels, _resize_segment


def _align_inserted_channels(
    destination: TrajectoryData,
    inserted: TrajectoryData,
    hold_frame: int,
) -> TrajectoryData:
    """Pad/drop channels so `inserted` matches the destination trajectory.

    Overlapping channels keep the inserted motion. Destination-only channels
    hold the pose at ``hold_frame``. Extra channels in the inserted file are
    ignored so a clip from another trajectory can be dropped in directly.
    """
    if not any(name in inserted.channels for name in destination.channels):
        raise ValueError("拼接轨迹没有可重叠的通道")
    hold_frame = int(np.clip(hold_frame, 0, destination.frame_count - 1))
    channels: dict[str, np.ndarray] = {}
    for name, values in destination.channels.items():
        if name in inserted.channels:
            channels[name] = inserted.channels[name]
        else:
            channels[name] = np.full(
                inserted.frame_count, float(values[hold_frame])
            )
    return TrajectoryData(
        inserted.times.copy(),
        channels,
        list(destination.position_channels),
        {
            src: dst
            for src, dst in destination.velocity_channels.items()
            if src in channels and dst in channels
        },
        {
            src: dst
            for src, dst in destination.effort_channels.items()
            if src in channels and dst in channels
        },
        dict(inserted.metadata),
    )


def concatenate(
    first: TrajectoryData,
    second: TrajectoryData,
    transition_duration: float = 0.0,
    order: int = 5,
) -> TrajectoryData:
    if abs(first.frequency.hz - second.frequency.hz) > first.frequency.hz * 1e-6:
        second = second.resample(first.frequency.hz)
    second = _align_inserted_channels(
        first, second, hold_frame=first.frame_count - 1
    )
    bridge_count = max(0, int(round(transition_duration * first.frequency.hz)))
    duration = transition_duration if bridge_count else 0.0
    first_last = first.frame_count - 1
    channels: dict[str, np.ndarray] = {}
    for name in first.channels:
        parts = [first.channels[name]]
        if bridge_count:
            parts.append(
                _bridge_between_trajectories(
                    first,
                    second,
                    name,
                    first_last,
                    0,
                    bridge_count,
                    duration,
                    order,
                )
            )
        parts.append(second.channels[name])
        channels[name] = np.concatenate(parts)
    return _from_channels(first, channels)


def splice_at(
    data: TrajectoryData,
    inserted: TrajectoryData,
    at: float,
    replace_until: float | None = None,
    transition_duration: float = 0.0,
    order: int = 5,
) -> TrajectoryData:
    if abs(data.frequency.hz - inserted.frequency.hz) > data.frequency.hz * 1e-6:
        inserted = inserted.resample(data.frequency.hz)
    first = data.nearest_frame(at)
    last = data.nearest_frame(replace_until) if replace_until is not None else first
    if last < first:
        raise ValueError("splice replacement end precedes insertion point")
    inserted = _align_inserted_channels(data, inserted, hold_frame=first)
    bridge_count = max(0, int(round(transition_duration * data.frequency.hz)))
    duration = transition_duration if bridge_count else 0.0
    inserted_last = inserted.frame_count - 1
    channels: dict[str, np.ndarray] = {}
    for name, values in data.channels.items():
        prefix = values[: first + 1]
        suffix = values[last + 1 :]
        pieces = [prefix]
        if bridge_count:
            pieces.append(
                _bridge_between_trajectories(
                    data,
                    inserted,
                    name,
                    first,
                    0,
                    bridge_count,
                    duration,
                    order,
                )
            )
        pieces.append(inserted.channels[name])
        if bridge_count and len(suffix):
            pieces.append(
                _bridge_between_values(
                    data,
                    inserted,
                    name,
                    inserted_last,
                    last,
                    bridge_count,
                    duration,
                    order,
                )
            )
        pieces.append(suffix)
        channels[name] = np.concatenate(pieces)
    return _from_channels(data, channels)


def polynomial_transition(
    data: TrajectoryData,
    start: float,
    end: float,
    channels: list[str],
    order: int = 5,
    preserve_endpoint_derivatives: bool = False,
    output_duration: float | None = None,
    *,
    boundary_source: TrajectoryData | None = None,
    boundary_frames: tuple[int, int] | None = None,
    preserve_source_motion: bool = False,
) -> TrajectoryData:
    if order not in {1, 3, 5}:
        raise ValueError("transition order must be 1, 3, or 5")
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    if last <= first:
        raise ValueError("transition range must contain at least two frames")
    if output_duration is not None:
        resized, resized_last = _resize_segment(
            data, first, last, output_duration
        )
        return polynomial_transition(
            resized,
            float(resized.times[first]),
            float(resized.times[resized_last]),
            channels,
            order,
            preserve_endpoint_derivatives,
            None,
            boundary_source=data,
            boundary_frames=(first, last),
            preserve_source_motion=preserve_source_motion,
        )
    u = np.linspace(0.0, 1.0, last - first + 1)
    duration = float(data.times[last] - data.times[first])
    source = boundary_source or data
    src_first, src_last = boundary_frames or (first, last)
    use_splice_velocity = boundary_source is not None
    result = data.copy()
    for name in channels:
        if name not in result.channels:
            raise ValueError(f"transition references missing channel: {name}")
        values = result.channels[name]
        start_position = float(source.channels[name][src_first])
        end_position = float(source.channels[name][src_last])
        if preserve_source_motion and order == 5:
            outer_v0, outer_a0 = _outer_endpoint_state(
                source, name, src_first, "left"
            )
            outer_v1, outer_a1 = _outer_endpoint_state(
                source, name, src_last, "right"
            )
            inner_v0, inner_a0 = _outer_endpoint_state(
                result, name, first, "right"
            )
            inner_v1, inner_a1 = _outer_endpoint_state(
                result, name, last, "left"
            )
            coefficients = np.empty(6)
            coefficients[:3] = [
                0.0,
                duration * (outer_v0 - inner_v0),
                0.5 * duration**2 * (outer_a0 - inner_a0),
            ]
            target = np.asarray(
                [
                    -np.sum(coefficients[:3]),
                    duration * (outer_v1 - inner_v1)
                    - coefficients[1]
                    - 2.0 * coefficients[2],
                    duration**2 * (outer_a1 - inner_a1)
                    - 2.0 * coefficients[2],
                ]
            )
            matrix = np.asarray(
                [[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=float
            )
            coefficients[3:] = np.linalg.solve(matrix, target)
            correction = sum(
                coefficient * u**power
                for power, coefficient in enumerate(coefficients)
            )
            values[first : last + 1] += correction
            continue
        if not preserve_endpoint_derivatives or order == 1:
            start_velocity = end_velocity = 0.0
            start_acceleration = end_acceleration = 0.0
        else:
            delta_q = end_position - start_position
            start_velocity, start_acceleration = _splice_boundary_state(
                source,
                name,
                src_first,
                "left",
                duration,
                delta_q,
                use_splice_velocity=use_splice_velocity,
            )
            end_velocity, end_acceleration = _splice_boundary_state(
                source,
                name,
                src_last,
                "right",
                duration,
                delta_q,
                use_splice_velocity=use_splice_velocity,
            )
        if order == 1:
            segment = start_position + (end_position - start_position) * u
        elif order == 3:
            segment = (
                (2 * u**3 - 3 * u**2 + 1) * start_position
                + (u**3 - 2 * u**2 + u) * duration * start_velocity
                + (-2 * u**3 + 3 * u**2) * end_position
                + (u**3 - u**2) * duration * end_velocity
            )
        else:
            coefficients = np.empty(6)
            coefficients[:3] = [
                start_position,
                duration * start_velocity,
                0.5 * duration**2 * start_acceleration,
            ]
            matrix = np.asarray([[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=float)
            target = np.asarray(
                [
                    end_position - np.sum(coefficients[:3]),
                    duration * end_velocity - coefficients[1] - 2 * coefficients[2],
                    duration**2 * end_acceleration - 2 * coefficients[2],
                ]
            )
            coefficients[3:] = np.linalg.solve(matrix, target)
            segment = sum(coefficient * u**power for power, coefficient in enumerate(coefficients))
        values[first : last + 1] = segment
    result.recompute_velocities()
    return result


def sample_patch(
    data: TrajectoryData,
    start: float,
    end: float,
    channel_values: dict[str, list[float] | np.ndarray],
    output_duration: float | None = None,
) -> TrajectoryData:
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    if last < first:
        raise ValueError("sample patch range is invalid")
    source_count = last - first + 1
    for name in channel_values:
        if name not in data.channels:
            raise ValueError(f"sample patch references missing channel: {name}")
    if output_duration is not None:
        result, resized_last = _resize_segment(
            data, first, last, output_duration
        )
        count = resized_last - first + 1
    else:
        result = data.copy()
        count = source_count
        resized_last = last
    for name, raw_values in channel_values.items():
        values = np.asarray(raw_values, dtype=float)
        if len(values) != count:
            values = np.interp(
                np.linspace(0, 1, count),
                np.linspace(0, 1, len(values)),
                values,
            )
        result.channels[name][first : resized_last + 1] = values
    patched = list(channel_values)
    if output_duration is not None and patched and resized_last - first >= 6:
        # Keep the actual Bezier/IK samples through the middle.  Only replace a
        # short entry and exit window with quintic Hermite bridges.  The old
        # implementation mixed just one endpoint by 15%, which cannot match
        # velocity and produced the C1 errors reported by the editor.
        count = resized_last - first + 1
        edge_intervals = min(
            max(3, int(round(0.25 / result.frequency.median_dt))),
            max(3, (count - 1) // 3),
        )
        entry_join = first + edge_intervals
        exit_join = resized_last - edge_intervals
        dt = result.frequency.median_dt
        for name in patched:
            values = result.channels[name]
            start_position = float(data.channels[name][first])
            end_position = float(data.channels[name][last])
            start_velocity = _splice_endpoint_velocity(data, name, first, "left")
            end_velocity = _splice_endpoint_velocity(data, name, last, "right")
            entry_velocity = float(
                (values[entry_join + 1] - values[entry_join]) / dt
            )
            exit_velocity = float(
                (values[exit_join] - values[exit_join - 1]) / dt
            )
            entry_duration = float(
                result.times[entry_join] - result.times[first]
            )
            exit_duration = float(
                result.times[resized_last] - result.times[exit_join]
            )
            values[first : entry_join + 1] = _hermite_segment_values(
                start_position,
                float(values[entry_join]),
                entry_join - first + 1,
                entry_duration,
                start_velocity,
                entry_velocity,
                0.0,
                0.0,
                order=5,
            )
            values[exit_join : resized_last + 1] = _hermite_segment_values(
                float(values[exit_join]),
                end_position,
                resized_last - exit_join + 1,
                exit_duration,
                exit_velocity,
                end_velocity,
                0.0,
                0.0,
                order=5,
            )
    result.recompute_velocities()
    return result


def quintic_keyframe_patch(
    data: TrajectoryData,
    start: float,
    key_time: float,
    end: float,
    targets: dict[str, float],
    output_duration: float | None = None,
) -> TrajectoryData:
    first = data.nearest_frame(start)
    key = data.nearest_frame(key_time)
    last = data.nearest_frame(end)
    if not first < key < last:
        raise ValueError("keyframe smoothing requires start < keyframe < end")
    if output_duration is not None:
        key_phase = (key - first) / (last - first)
        boundary_source = data
        boundary_frames = (first, last)
        resized, resized_last = _resize_segment(
            data, first, last, output_duration
        )
        resized_key = first + int(
            round(key_phase * (resized_last - first))
        )
        resized_key = max(first + 1, min(resized_last - 1, resized_key))
        result = quintic_keyframe_patch(
            resized,
            float(resized.times[first]),
            float(resized.times[resized_key]),
            float(resized.times[resized_last]),
            targets,
            None,
        )
        # Resizing changes the time scale of the selected source segment.  A
        # zero-slope correction alone therefore preserves the *resized* edge
        # velocity, which no longer matches the untouched prefix/suffix.  Build
        # each target channel as two C2-matched quintics through the dragged
        # keyframe so both outer joins retain their original velocity and
        # acceleration.  Using zero acceleration here caused a visible snap at
        # the edge of a moving/high-frequency source segment.
        for name, target in targets.items():
            start_position = float(boundary_source.channels[name][boundary_frames[0]])
            end_position = float(boundary_source.channels[name][boundary_frames[1]])
            start_velocity = _splice_endpoint_velocity(
                boundary_source, name, boundary_frames[0], "left"
            )
            end_velocity = _splice_endpoint_velocity(
                boundary_source, name, boundary_frames[1], "right"
            )
            _, start_acceleration = _outer_endpoint_state(
                boundary_source, name, boundary_frames[0], "left"
            )
            _, end_acceleration = _outer_endpoint_state(
                boundary_source, name, boundary_frames[1], "right"
            )
            baseline = result.channels[name].copy()
            baseline_velocity = np.gradient(
                baseline, result.times, edge_order=min(2, result.frame_count - 1)
            )
            baseline_acceleration = np.gradient(
                baseline_velocity,
                result.times,
                edge_order=min(2, result.frame_count - 1),
            )
            key_velocity = float(baseline_velocity[resized_key])
            key_acceleration = float(baseline_acceleration[resized_key])
            left_duration = float(result.times[resized_key] - result.times[first])
            right_duration = float(result.times[resized_last] - result.times[resized_key])
            left_values = _hermite_segment_values(
                start_position, float(target), resized_key - first + 1,
                left_duration, start_velocity, key_velocity,
                start_acceleration, key_acceleration, order=5,
            )
            right_values = _hermite_segment_values(
                float(target), end_position, resized_last - resized_key + 1,
                right_duration, key_velocity, end_velocity,
                key_acceleration, end_acceleration, order=5,
            )
            result.channels[name][first : resized_key + 1] = left_values
            result.channels[name][resized_key : resized_last + 1] = right_values
        non_targets = [
            name
            for name in data.position_channels
            if name not in targets and name in result.channels
        ]
        if non_targets:
            result = polynomial_transition(
                result,
                float(result.times[first]),
                float(result.times[resized_last]),
                non_targets,
                order=5,
                preserve_endpoint_derivatives=True,
                output_duration=None,
                boundary_source=boundary_source,
                boundary_frames=boundary_frames,
            )
        result.recompute_velocities()
        return result
    result = data.copy()
    left_u = np.linspace(0, 1, key - first + 1)
    right_u = np.linspace(0, 1, last - key + 1)
    left_weight = 10 * left_u**3 - 15 * left_u**4 + 6 * left_u**5
    right_weight = 1 - (10 * right_u**3 - 15 * right_u**4 + 6 * right_u**5)
    for name, target in targets.items():
        if name not in result.channels:
            raise ValueError(f"keyframe patch references missing channel: {name}")
        values = result.channels[name]
        correction = float(target) - values[key]
        values[first : key + 1] += correction * left_weight
        values[key + 1 : last + 1] += correction * right_weight[1:]
    result.recompute_velocities()
    return result



def _transition_values(
    start: float, end: float, count: int, order: int
) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=float)
    duration = max(count, 1) / max(1.0, count)
    return _hermite_segment_values(
        start,
        end,
        count,
        float(count),
        0.0,
        0.0,
        0.0,
        0.0,
        order=order,
    )


def _hermite_segment_values(
    start_pos: float,
    end_pos: float,
    frame_count: int,
    duration: float,
    start_velocity: float,
    end_velocity: float,
    start_acceleration: float,
    end_acceleration: float,
    order: int = 5,
) -> np.ndarray:
    if frame_count <= 0:
        return np.array([], dtype=float)
    if duration <= 0.0:
        duration = float(max(frame_count - 1, 1)) * 1e-3
    u = np.linspace(0.0, 1.0, frame_count)
    if order == 1:
        return start_pos + (end_pos - start_pos) * u
    if order == 3:
        return (
            (2 * u**3 - 3 * u**2 + 1) * start_pos
            + (u**3 - 2 * u**2 + u) * duration * start_velocity
            + (-2 * u**3 + 3 * u**2) * end_pos
            + (u**3 - u**2) * duration * end_velocity
        )
    if order != 5:
        raise ValueError("transition order must be 1, 3, or 5")
    coefficients = np.empty(6)
    coefficients[:3] = [
        start_pos,
        duration * start_velocity,
        0.5 * duration**2 * start_acceleration,
    ]
    matrix = np.asarray([[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=float)
    target = np.asarray(
        [
            end_pos - np.sum(coefficients[:3]),
            duration * end_velocity - coefficients[1] - 2 * coefficients[2],
            duration**2 * end_acceleration - 2 * coefficients[2],
        ]
    )
    coefficients[3:] = np.linalg.solve(matrix, target)
    return sum(
        coefficient * u**power
        for power, coefficient in enumerate(coefficients)
    )


def _bridge_between_trajectories(
    left: TrajectoryData,
    right: TrajectoryData,
    channel: str,
    left_frame: int,
    right_frame: int,
    bridge_count: int,
    duration: float,
    order: int,
) -> np.ndarray:
    start_pos = float(left.channels[channel][left_frame])
    end_pos = float(right.channels[channel][right_frame])
    delta_q = end_pos - start_pos
    start_velocity, start_acceleration = _splice_boundary_state(
        left,
        channel,
        left_frame,
        "left",
        duration,
        delta_q,
        use_splice_velocity=True,
    )
    end_velocity, end_acceleration = _splice_boundary_state(
        right,
        channel,
        right_frame,
        "left",
        duration,
        delta_q,
        use_splice_velocity=True,
    )
    return _hermite_segment_values(
        start_pos,
        end_pos,
        bridge_count,
        duration,
        start_velocity,
        end_velocity,
        start_acceleration,
        end_acceleration,
        order=order,
    )


def _bridge_between_values(
    data: TrajectoryData,
    inserted: TrajectoryData,
    channel: str,
    inserted_last: int,
    data_last: int,
    bridge_count: int,
    duration: float,
    order: int,
) -> np.ndarray:
    start_pos = float(inserted.channels[channel][inserted_last])
    end_pos = float(data.channels[channel][data_last + 1])
    delta_q = end_pos - start_pos
    if inserted.frame_count > 1 and inserted_last > 0:
        dt = inserted.times[inserted_last] - inserted.times[inserted_last - 1]
        start_velocity = float(
            (
                inserted.channels[channel][inserted_last]
                - inserted.channels[channel][inserted_last - 1]
            )
            / dt
        )
    else:
        start_velocity = 0.0
    start_acceleration = _cap_transition_acceleration(
        _outer_endpoint_state(inserted, channel, inserted_last, "left")[1],
        delta_q,
        duration,
    )
    end_velocity, end_acceleration = _splice_boundary_state(
        data,
        channel,
        data_last,
        "right",
        duration,
        delta_q,
        use_splice_velocity=True,
    )
    return _hermite_segment_values(
        start_pos,
        end_pos,
        bridge_count,
        duration,
        start_velocity,
        end_velocity,
        start_acceleration,
        end_acceleration,
        order=order,
    )


def _outer_endpoint_state(
    data: TrajectoryData,
    channel: str,
    frame: int,
    side: str,
) -> tuple[float, float]:
    """Fit V/A on the outside of the splice using a short local window.

    Uses ~50 ms of samples *outside* the selection so high-rate gradient
    noise on a single frame does not dominate the quintic coefficients.
    """
    window_count = max(
        5,
        min(
            51,
            int(round(0.05 / max(data.frequency.median_dt, 1e-6))) + 1,
        ),
    )
    if side == "left":
        indices = np.arange(frame - window_count + 1, frame + 1)
    elif side == "right":
        indices = np.arange(frame, frame + window_count)
    else:
        raise ValueError(f"unsupported endpoint side: {side}")
    if indices[0] < 0 or indices[-1] >= data.frame_count:
        # Near trajectory ends: fall back to a one-sided local fit.
        if side == "left":
            start = max(0, frame - window_count + 1)
            indices = np.arange(start, frame + 1)
        else:
            stop = min(data.frame_count, frame + window_count)
            indices = np.arange(frame, stop)
        if len(indices) < 3:
            return 0.0, 0.0
    relative_times = data.times[indices] - data.times[frame]
    values = data.channels[channel][indices]
    if np.allclose(relative_times, 0.0):
        return 0.0, 0.0
    coefficients = np.polyfit(relative_times, values, min(2, len(indices) - 1))
    if len(coefficients) == 1:
        return 0.0, 0.0
    if len(coefficients) == 2:
        return float(coefficients[0]), 0.0
    velocity = float(coefficients[1])
    acceleration = float(2.0 * coefficients[0])
    return velocity, acceleration


def _cap_transition_acceleration(
    acceleration: float, delta_q: float, duration: float
) -> float:
    """Keep ½ T²·a from dominating a tiny endpoint displacement."""
    if duration <= 0.0:
        return 0.0
    cap = 6.0 * abs(delta_q) / (duration * duration)
    if cap < 1e-9:
        return 0.0
    return float(np.clip(acceleration, -cap, cap))


def _splice_endpoint_velocity(
    data: TrajectoryData,
    channel: str,
    frame: int,
    side: str,
) -> float:
    """One-sided velocity from adjacent frames so C1 matches prefix/suffix."""
    if side == "left":
        if frame <= 0:
            return 0.0
        dt = data.times[frame] - data.times[frame - 1]
        if dt <= 0.0:
            return 0.0
        return float(
            (data.channels[channel][frame] - data.channels[channel][frame - 1])
            / dt
        )
    if frame >= data.frame_count - 1:
        return 0.0
    dt = data.times[frame + 1] - data.times[frame]
    if dt <= 0.0:
        return 0.0
    return float(
        (data.channels[channel][frame + 1] - data.channels[channel][frame])
        / dt
    )


def _splice_boundary_state(
    data: TrajectoryData,
    channel: str,
    frame: int,
    side: str,
    duration: float,
    delta_q: float,
    *,
    use_splice_velocity: bool = True,
) -> tuple[float, float]:
    if use_splice_velocity:
        velocity = _splice_endpoint_velocity(data, channel, frame, side)
        _, acceleration = _outer_endpoint_state(data, channel, frame, side)
        acceleration = _cap_transition_acceleration(
            acceleration, delta_q, duration
        )
        return velocity, acceleration
    velocity, acceleration = _outer_endpoint_state(
        data, channel, frame, side
    )
    return velocity, acceleration


def recommend_splice_transition_duration(
    base: TrajectoryData,
    inserted: TrajectoryData,
    at: float,
    channels: list[str] | None = None,
    *,
    v_ref: float = 0.4,
    a_ref: float = 1.0,
    max_duration: float = 60.0,
) -> float:
    """Suggest bridge duration between an existing trajectory and a splice insert."""
    selected = channels or list(base.position_channels)
    first = base.nearest_frame(at)
    floor = max(0.2, 3.0 * base.frequency.median_dt)
    needed = floor
    min_speed = v_ref * 0.05
    for name in selected:
        if name not in base.channels or name not in inserted.channels:
            continue
        delta_q = float(
            inserted.channels[name][0] - base.channels[name][first]
        )
        abs_dq = abs(delta_q)
        v0 = _splice_endpoint_velocity(base, name, first, "left")
        v1 = _splice_endpoint_velocity(inserted, name, 0, "right")
        v_max = max(abs(v0), abs(v1))
        channel_terms = [abs_dq / v_ref]
        if v_max >= min_speed:
            channel_terms.append(v_max / a_ref)
            channel_terms.append(abs(v0 - v1) / a_ref)
        if abs_dq >= 1e-6:
            channel_terms.append(float(np.sqrt(6.0 * abs_dq / a_ref)))
        needed = max(needed, max(channel_terms))
    capped = min(max(floor, needed * 1.15), max_duration)
    return float(np.ceil(capped * 1000.0) / 1000.0)


def recommend_transition_duration(
    data: TrajectoryData,
    start: float,
    end: float,
    channels: list[str] | None = None,
    *,
    v_ref: float = 0.4,
    a_ref: float = 1.0,
    max_duration: float = 60.0,
    channel_limits: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Suggest a lower-bound transition duration from splice V/A and Δq.

    Uses conservative heuristics per channel and returns the max. Values are
    clamped to a practical upper bound so near-static gripper channels with
    tiny numerical velocity do not explode the recommendation.
    """
    selected = channels or list(data.position_channels)
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    if last <= first:
        return max(0.2, 3.0 * data.frequency.median_dt)
    floor = max(0.2, 3.0 * data.frequency.median_dt)
    needed = floor
    min_speed = v_ref * 0.05
    for name in selected:
        if name not in data.channels:
            continue
        delta_q = float(
            data.channels[name][last] - data.channels[name][first]
        )
        abs_dq = abs(delta_q)
        v0 = _splice_endpoint_velocity(data, name, first, "left")
        v1 = _splice_endpoint_velocity(data, name, last, "right")
        v_max = max(abs(v0), abs(v1))
        channel_terms = [abs_dq / v_ref]
        if v_max >= min_speed:
            channel_terms.append(v_max / a_ref)
            channel_terms.append(abs(v0 - v1) / a_ref)
        if abs_dq >= 1e-6:
            channel_terms.append(float(np.sqrt(6.0 * abs_dq / a_ref)))
        needed = max(needed, max(channel_terms))
    preferred = min(max(floor, needed * 1.15), max_duration)
    if channel_limits:
        # With inherited endpoint velocities, a longer quintic can overshoot
        # *more* because the velocity terms scale with duration. Search the
        # feasible duration interval instead of treating the heuristic as a
        # monotonic lower bound.
        candidates = np.unique(
            np.r_[
                np.geomspace(floor, max_duration, 96),
                preferred,
                float(data.times[last] - data.times[first]),
            ]
        )
        feasible = [
            float(duration)
            for duration in candidates
            if _transition_stays_within_limits(
                data, first, last, selected, float(duration), channel_limits
            )
        ]
        if feasible:
            # Prefer a feasible value at/above the dynamics estimate; if that
            # interval does not exist, choose the closest feasible value.
            above = [value for value in feasible if value >= preferred]
            preferred = min(above) if above else min(
                feasible, key=lambda value: abs(np.log(value / preferred))
            )
    return float(np.ceil(preferred * 1000.0) / 1000.0)


def _transition_stays_within_limits(
    data: TrajectoryData,
    first: int,
    last: int,
    channels: list[str],
    duration: float,
    channel_limits: dict[str, tuple[float, float]],
) -> bool:
    for name in channels:
        if name not in data.channels or name not in channel_limits:
            continue
        start = float(data.channels[name][first])
        end = float(data.channels[name][last])
        delta = end - start
        v0, a0 = _splice_boundary_state(
            data, name, first, "left", duration, delta,
            use_splice_velocity=True,
        )
        v1, a1 = _splice_boundary_state(
            data, name, last, "right", duration, delta,
            use_splice_velocity=True,
        )
        values = _hermite_segment_values(
            start, end, 257, duration, v0, v1, a0, a1, order=5
        )
        lower, upper = channel_limits[name]
        tolerance = 1e-9 * max(1.0, abs(lower), abs(upper))
        if float(np.min(values)) < lower - tolerance or float(np.max(values)) > upper + tolerance:
            return False
    return True



