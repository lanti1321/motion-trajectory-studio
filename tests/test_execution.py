from __future__ import annotations

import numpy as np
import pytest

from studio.core.trajectory import TrajectoryData
from studio.execution import prepare_execution_trajectory


def test_execution_clip_adds_three_second_quintic_bridge() -> None:
    times = np.arange(11, dtype=float) / 10
    source = TrajectoryData(times, {"q0": times.copy()}, ["q0"])
    result = prepare_execution_trajectory(
        source, start_time=0.5, current_positions={"q0": -0.5}
    )
    assert result.times[0] == 0.0
    assert result.channels["q0"][0] == pytest.approx(-0.5)
    bridge_end = result.nearest_frame(3.0)
    assert result.times[bridge_end] == pytest.approx(3.0)
    assert result.channels["q0"][bridge_end] == pytest.approx(0.5)
    assert result.channels["q0"][-1] == pytest.approx(1.0)
    # The bridge starts at rest and matches the source's velocity at the join.
    bridge = result.channels["q0"][: bridge_end + 1]
    bridge_velocity = np.gradient(bridge, result.times[: bridge_end + 1])
    assert abs(bridge_velocity[0]) < 0.02
    assert bridge_velocity[-1] == pytest.approx(1.0, abs=0.08)


def test_execution_clip_requires_complete_current_state() -> None:
    source = TrajectoryData(
        np.asarray([0.0, 0.1]),
        {"q0": np.zeros(2), "q1": np.zeros(2)},
        ["q0", "q1"],
    )
    with pytest.raises(ValueError, match="q1"):
        prepare_execution_trajectory(source, 0.0, {"q0": 0.0})
