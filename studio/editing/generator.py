"""Blank trajectory files and a small keyframe-based Trajectory Generator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.models.pose import hold_pose_trajectory


def blank_trajectory(
    channel_values: Mapping[str, float],
    duration: float = 2.0,
    hz: float = 50.0,
) -> TrajectoryData:
    """Create a hold-pose trajectory that can later receive generator keyframes."""
    data = hold_pose_trajectory(channel_values, duration=duration, hz=hz)
    data.metadata = dict(data.metadata)
    data.metadata["generator"] = "blank"
    return data


def generate_from_keyframes(
    keyframes: Sequence[tuple[float, Mapping[str, float]]],
    hz: float = 50.0,
) -> TrajectoryData:
    """Build a trajectory from time/pose keyframes.

    One keyframe becomes a short hold. Two or more are linearly interpolated;
    the editor can later overlay quintic keyframe patches without knowing a
    skill algorithm.
    """
    if not keyframes:
        raise ValueError("Trajectory Generator needs at least one keyframe")
    ordered = sorted(
        ((float(time), dict(pose)) for time, pose in keyframes),
        key=lambda item: item[0],
    )
    names: list[str] = []
    seen: set[str] = set()
    for _time, pose in ordered:
        for name in pose:
            channel = str(name)
            if channel not in seen:
                seen.add(channel)
                names.append(channel)
    if not names:
        raise ValueError("Trajectory Generator keyframes have no channels")
    start = ordered[0][0]
    times = [time - start for time, _pose in ordered]
    if times[0] < 0:
        raise ValueError("keyframe times must be increasing")
    duration = max(times[-1], 1.0 / max(hz, 1e-6))
    if len(ordered) == 1:
        return blank_trajectory(ordered[0][1], duration=max(duration, 2.0), hz=hz)

    count = max(2, int(round(duration * hz)) + 1)
    sample_times = np.linspace(0.0, duration, count)
    knot_times = np.asarray(times, dtype=np.float64)
    channels: dict[str, np.ndarray] = {}
    for name in names:
        values = np.asarray(
            [float(pose.get(name, 0.0)) for _time, pose in ordered],
            dtype=np.float64,
        )
        channels[name] = np.interp(sample_times, knot_times, values)
    data = TrajectoryData(sample_times, channels, names)
    data.metadata["generator"] = "keyframes"
    data.metadata["keyframe_count"] = len(ordered)
    return data
