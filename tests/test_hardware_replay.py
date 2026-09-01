from __future__ import annotations

import pytest
import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.hardware.openarm_replay import (
    build_dual_replay_clip,
    build_load_command,
    is_motor_fault_code,
    motor_fault_text,
    parse_temperature_packet,
)


def test_build_independent_controller_load_command(tmp_path) -> None:
    path = tmp_path / "timeline.csv"
    path.touch()
    assert build_load_command(path, 54321) == f"load|{path.resolve()}|54321"


def test_dual_replay_clip_has_openarm_interleaved_shape() -> None:
    times = np.asarray([0.0, 0.1, 0.2])
    positions = {f"q{i}": times + i for i in range(16)}
    source = TrajectoryData(times, positions, list(positions))
    clip = build_dual_replay_clip(source, 1)
    assert list(clip.channels) == [
        name
        for index in range(16)
        for name in (f"q{index}", f"dq{index}")
    ]
    assert len(clip.channels) == 32
    assert clip.times.tolist() == pytest.approx([0.0, 0.1])
    for index in range(16):
        np.testing.assert_allclose(clip.channels[f"dq{index}"], 1.0)


def test_dual_replay_clip_prefers_maintained_velocity_channels() -> None:
    times = np.asarray([0.0, 0.1, 0.2])
    positions = {f"q{i}": times + i for i in range(16)}
    channels = dict(positions)
    velocity_channels = {}
    for index in range(16):
        velocity_name = f"measured_dq{index}"
        channels[velocity_name] = np.full(3, 0.25 + index)
        velocity_channels[f"q{index}"] = velocity_name
    source = TrajectoryData(
        times,
        channels,
        list(positions),
        velocity_channels,
    )
    clip = build_dual_replay_clip(source, 1)
    for index in range(16):
        np.testing.assert_allclose(clip.channels[f"dq{index}"], 0.25 + index)


def test_parse_extended_temperature_telemetry() -> None:
    fields = ["1.25", "0", "0", "0", "0"]
    fields += [str(index / 10) for index in range(16)]
    fields += ["0"]
    fields += [str(index) for index in range(16)]
    fields += [str(30 + index) for index in range(16)]
    fields += [str(40 + index) for index in range(16)]
    fields += [str(index / 10) for index in range(16)]
    sample = parse_temperature_packet(",".join(fields))
    assert sample is not None
    assert sample.time == pytest.approx(1.25)
    assert sample.mos == tuple(range(30, 46))
    assert sample.rotor == tuple(range(40, 56))
    assert sample.errors == tuple(range(16))
    assert sample.positions == pytest.approx(tuple(index / 10 for index in range(16)))


def test_legacy_telemetry_has_no_temperature_sample() -> None:
    assert parse_temperature_packet(",".join(["0"] * 22)) is None


def test_damiao_operating_state_is_not_reported_as_motor_fault() -> None:
    assert not is_motor_fault_code(0)
    assert not is_motor_fault_code(1)
    assert is_motor_fault_code(8)
    assert is_motor_fault_code(14)
    assert not is_motor_fault_code(15)
    assert motor_fault_text(10) == "过流"
