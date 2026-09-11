from __future__ import annotations

import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.editing.operation_data import _from_channels


def speed_segment(
    data: TrajectoryData,
    start: float,
    end: float,
    factor: float,
    channels: list[str] | None = None,
) -> TrajectoryData:
    if factor <= 0:
        raise ValueError("speed factor must be positive")
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    old_count = last - first
    new_count = max(1, int(round(old_count / factor)))
    source_phase = np.linspace(first, last, new_count + 1)
    selected = set(channels or data.channels)
    rebuilt: dict[str, np.ndarray] = {}
    for name, values in data.channels.items():
        if name in selected:
            segment = np.interp(source_phase, np.arange(data.frame_count), values)
        else:
            segment = np.linspace(values[first], values[last], new_count + 1)
        rebuilt[name] = np.r_[values[:first], segment, values[last + 1 :]]
    return _from_channels(data, rebuilt)


def speed_ramp_segment(
    data: TrajectoryData,
    start: float,
    end: float,
    peak_factor: float,
    channels: list[str] | None = None,
    ramp_duration: float | None = None,
) -> TrajectoryData:
    """Change speed with minimum-duration quintic blends at both boundaries."""
    if peak_factor <= 0:
        raise ValueError("peak speed factor must be positive")
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    old_count = last - first
    if old_count < 2:
        raise ValueError("speed ramp range is too short")
    selected = set(channels or data.position_channels)
    global_scope = not channels or selected.issuperset(data.position_channels)
    if not global_scope and peak_factor < 1:
        raise ValueError(
            "slowing only part of the motors cannot fit the original interval; "
            "select all motors or use a factor of at least 1"
        )
    if not global_scope:
        new_count = max(2, int(round(old_count / peak_factor)))
        u = np.linspace(0.0, 1.0, new_count + 1)
        phase = _quintic_phase(u, 1.0 / peak_factor, 0.0)
        source_phase = first + phase * old_count
        result = data.copy()
        accelerated_end = first + new_count
        for name in selected:
            if name not in result.channels:
                continue
            accelerated = np.interp(
                source_phase, np.arange(data.frame_count), data.channels[name]
            )
            result.channels[name][first : accelerated_end + 1] = accelerated
            if accelerated_end < last:
                result.channels[name][accelerated_end : last + 1] = accelerated[-1]
        result.recompute_velocities()
        return result
    if peak_factor < 0.5:
        raise ValueError(
            "smooth global speed factor must be at least 0.5"
        )
    if ramp_duration is None:
        new_count = max(2, int(round(old_count / peak_factor)))
        if new_count < 6:
            raise ValueError(
                "变速后的片段至少需要保留 7 帧才能维持 C2 连续；"
                "请降低速度倍数或扩大选区"
            )
        u = np.linspace(0.0, 1.0, new_count + 1)
        rate = 1.0 + 2.0 * (peak_factor - 1.0) * np.sin(np.pi * u) ** 2
        cumulative = np.r_[
            0.0, np.cumsum((rate[:-1] + rate[1:]) * 0.5)
        ]
        cumulative /= cumulative[-1]
        source_phase = first + cumulative * old_count
        boundary_indices = np.asarray(
            [0, 1, 2, new_count - 2, new_count - 1, new_count]
        )
        source_phase[boundary_indices] = np.asarray(
            [first, first + 1, first + 2, last - 2, last - 1, last],
            dtype=float,
        )
        rebuilt = {
            name: np.r_[
                values[:first],
                np.interp(source_phase, np.arange(data.frame_count), values),
                values[last + 1 :],
            ]
            for name, values in data.channels.items()
        }
        return _from_channels(data, rebuilt)

    requested_ramp = float(ramp_duration)
    source_phase, output_duration = quintic_speed_time_warp(
        old_count,
        peak_factor,
        requested_ramp,
        data.frequency.median_dt,
    )
    absolute_phase = first + source_phase * old_count
    rebuilt: dict[str, np.ndarray] = {}
    for name, values in data.channels.items():
        segment = np.interp(
            absolute_phase, np.arange(data.frame_count), values
        )
        rebuilt[name] = np.r_[values[:first], segment, values[last + 1 :]]
    result = _from_channels(data, rebuilt)
    expected_count = max(6, int(round(output_duration / data.frequency.median_dt)))
    if len(source_phase) != expected_count + 1:
        raise RuntimeError("五次时间映射输出采样数不一致")
    return result


def quintic_speed_time_warp(
    source_count: int,
    factor: float,
    ramp_duration: float,
    dt: float,
) -> tuple[np.ndarray, float]:
    """Build one monotonic C2 time map with a constant-speed middle section."""
    if source_count < 2 or factor <= 0 or ramp_duration <= 0 or dt <= 0:
        raise ValueError("五次变速参数无效")
    source_duration = source_count * dt
    maximum_ramp = source_duration / (1.0 + factor)
    effective_ramp = min(ramp_duration, maximum_ramp * 0.95)
    plateau_duration = (
        source_duration - effective_ramp * (1.0 + factor)
    ) / factor
    output_duration = 2.0 * effective_ramp + max(0.0, plateau_duration)
    output_count = max(6, int(round(output_duration / dt)))
    output_time = np.linspace(0.0, output_duration, output_count + 1)
    plateau_end = output_duration - effective_ramp
    rate = np.full_like(output_time, factor)
    left = output_time < effective_ramp
    right = output_time > plateau_end
    left_u = np.clip(output_time[left] / effective_ramp, 0.0, 1.0)
    right_u = np.clip(
        (output_time[right] - plateau_end) / effective_ramp, 0.0, 1.0
    )
    left_s = left_u**3 * (10.0 - 15.0 * left_u + 6.0 * left_u**2)
    right_s = right_u**3 * (10.0 - 15.0 * right_u + 6.0 * right_u**2)
    rate[left] = 1.0 + (factor - 1.0) * left_s
    rate[right] = factor + (1.0 - factor) * right_s
    if not np.isfinite(rate).all() or np.any(rate <= 0):
        raise ValueError("五次时间映射产生了无效速度倍率")
    cumulative = np.r_[
        0.0,
        np.cumsum((rate[:-1] + rate[1:]) * 0.5 * np.diff(output_time)),
    ]
    cumulative /= cumulative[-1]
    # Preserve the exact discrete neighbourhood at the two outer joins.  This
    # makes C1 equal to the source even for noisy 1000 Hz recordings.
    if output_count >= 6 and source_count >= 6:
        cumulative[[0, 1, 2, -3, -2, -1]] = np.asarray(
            [0, 1, 2, source_count - 2, source_count - 1, source_count],
            dtype=float,
        ) / source_count
    if np.any(np.diff(cumulative) <= 0):
        raise ValueError("当前选区过短或倍速过高，无法生成单调五次时间映射")
    return cumulative, output_duration


def recommend_speed_ramp_duration(
    data: TrajectoryData,
    start: float,
    end: float,
    factor: float,
    channels: list[str] | None = None,
    *,
    acceleration_budget: float = 1.0,
) -> float:
    """Return a conservative minimum duration for each quintic speed blend."""
    first = data.nearest_frame(start)
    last = data.nearest_frame(end)
    span = float(data.times[last] - data.times[first])
    if span <= 0 or factor <= 0:
        raise ValueError("变速选区或速度倍数无效")
    compressed_span = span / factor
    floor = max(6.0 * data.frequency.median_dt, 0.05)
    # A short local repair window: normally 2% of the compressed clip, capped
    # so long/noisy recordings cannot create enormous recommendations.
    preferred = min(0.25, max(floor, compressed_span * 0.02))
    preferred = min(preferred, max(floor, compressed_span * 0.25))
    return float(np.ceil(preferred * 1000.0) / 1000.0)


def _quintic_phase(
    u: np.ndarray, start_slope: float, end_slope: float
) -> np.ndarray:
    coefficients = np.empty(6)
    coefficients[:3] = [0.0, start_slope, 0.0]
    matrix = np.asarray([[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=float)
    target = np.asarray(
        [
            1.0 - coefficients[1],
            end_slope - coefficients[1],
            0.0,
        ]
    )
    coefficients[3:] = np.linalg.solve(matrix, target)
    return sum(
        coefficient * u**power
        for power, coefficient in enumerate(coefficients)
    )



