from __future__ import annotations

import numpy as np

from studio.core.trajectory import TrajectoryData


def prepare_execution_trajectory(
    trajectory: TrajectoryData,
    start_time: float,
    current_positions: dict[str, float],
    transition_duration: float = 3.0,
) -> TrajectoryData:
    """Build a safe execution clip starting with a zero-velocity C2 bridge.

    Values in ``current_positions`` and the returned trajectory use trajectory
    channel units. A hardware driver is responsible for mapping them to motors.
    """
    if transition_duration <= 0:
        raise ValueError("transition_duration must be positive")
    missing = set(trajectory.position_channels) - set(current_positions)
    if missing:
        raise ValueError(
            "current robot state is missing channels: " + ", ".join(sorted(missing))
        )
    start = trajectory.nearest_frame(start_time)
    hz = trajectory.frequency.hz
    bridge_count = max(2, int(round(transition_duration * hz)) + 1)
    bridge_times = np.linspace(0.0, transition_duration, bridge_count)
    source_times = trajectory.times[start:] - trajectory.times[start]
    # The bridge already contains the selected start pose as its last sample.
    output_times = np.r_[bridge_times, transition_duration + source_times[1:]]
    channels: dict[str, np.ndarray] = {}
    for name, values in trajectory.channels.items():
        if name in trajectory.position_channels:
            initial = float(current_positions[name])
            target = float(values[start])
            if trajectory.frame_count > 1:
                velocity = np.gradient(
                    values,
                    trajectory.times,
                    edge_order=min(2, trajectory.frame_count - 1),
                )
                target_velocity = float(velocity[start])
            else:
                velocity = np.zeros(1)
                target_velocity = 0.0
            if trajectory.frame_count > 2:
                acceleration = np.gradient(
                    velocity,
                    trajectory.times,
                    edge_order=min(2, trajectory.frame_count - 1),
                )
                target_acceleration = float(acceleration[start])
            else:
                target_acceleration = 0.0
            duration = transition_duration
            coefficients = np.linalg.solve(
                np.asarray(
                    [
                        [duration**3, duration**4, duration**5],
                        [3 * duration**2, 4 * duration**3, 5 * duration**4],
                        [6 * duration, 12 * duration**2, 20 * duration**3],
                    ]
                ),
                np.asarray(
                    [
                        target - initial,
                        target_velocity,
                        target_acceleration,
                    ]
                ),
            )
            bridge = initial + sum(
                coefficients[power - 3] * bridge_times**power
                for power in range(3, 6)
            )
        else:
            bridge = np.full(bridge_count, float(values[start]))
        channels[name] = np.r_[bridge, values[start + 1 :]]
    result = TrajectoryData(
        output_times,
        channels,
        list(trajectory.position_channels),
        dict(trajectory.velocity_channels),
        dict(trajectory.effort_channels),
        dict(trajectory.metadata),
    )
    result.metadata.update(
        {
            "execution_source_start_time": float(trajectory.times[start]),
            "execution_transition_duration": float(transition_duration),
        }
    )
    result.recompute_velocities()
    return result
