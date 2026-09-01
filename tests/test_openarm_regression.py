from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from studio.core.project import TimelineLabel
from studio.core.timeline import TimelineEngine
from studio.io.trajectory_io import load_trajectory
from studio.editing.operations import polynomial_transition
from studio.models.adapter import MuJoCoModelAdapter, auto_map_channels
from studio.validation.checks import sample_collisions, validate_model_mapping


ROOT = Path(__file__).resolve().parents[2]
TRAJECTORY = ROOT / "action_end5.csv"
MODEL = (
    Path(__file__).resolve().parents[1]
    / "model_library"
    / "OpenArm_V1"
    / "openarm_bimanual.xml"
)


@pytest.mark.skipif(
    not TRAJECTORY.is_file() or not MODEL.is_file(),
    reason="real OpenArm regression assets are unavailable",
)
def test_real_openarm_partial_speed_regression() -> None:
    source = load_trajectory(TRAJECTORY)
    adapter = MuJoCoModelAdapter(MODEL)
    mapping = auto_map_channels(source.position_channels, adapter.info)
    assert len(mapping) == len(source.position_channels)
    preset = json.loads(
        (MODEL.parent / "model.json").read_text(encoding="utf-8")
    )
    transforms = {
        channel: {
            key: float(value)
            for key, value in settings.items()
            if key in {"scale", "offset", "minimum", "maximum"}
        }
        for channel, settings in preset["default_mapping"].items()
    }
    long_transition = polynomial_transition(
        source,
        100.173,
        142.593,
        ["q0"],
        order=5,
        preserve_endpoint_derivatives=True,
        output_duration=5.0,
    )
    start_frame = long_transition.nearest_frame(100.173)
    end_frame = long_transition.nearest_frame(105.173)
    assert long_transition.channels["q0"][start_frame] == pytest.approx(
        source.channels["q0"][source.nearest_frame(100.173)]
    )
    assert long_transition.channels["q0"][end_frame] == pytest.approx(
        source.channels["q0"][source.nearest_frame(142.593)]
    )
    assert np.isfinite(
        long_transition.channels["q0"][start_frame : end_frame + 1]
    ).all()

    start, end = 10.0, 12.0
    first = source.nearest_frame(start)
    last = source.nearest_frame(end)
    engine = TimelineEngine(
        source,
        labels=[
            TimelineLabel("before", 5.0),
            TimelineLabel("inside", 11.0),
            TimelineLabel("after", 15.0),
        ],
    )
    engine.add_operation(
        {
            "type": "speed_ramp",
            "start": start,
            "end": end,
            "factor": 2.0,
            "channels": ["q0", "q1"],
            "changes_duration": False,
        }
    )
    output = engine.render()

    assert output.frame_count == source.frame_count
    assert output.duration == pytest.approx(source.duration)
    np.testing.assert_array_equal(
        output.channels["q0"][:first], source.channels["q0"][:first]
    )
    np.testing.assert_array_equal(
        output.channels["q0"][last + 1 :],
        source.channels["q0"][last + 1 :],
    )
    assert output.channels["q0"][last] == pytest.approx(
        source.channels["q0"][last]
    )
    assert [label.time for label in engine.labels] == [5.0, 11.0, 15.0]

    expected_velocity = np.gradient(
        output.channels["q0"], output.times, edge_order=2
    )
    np.testing.assert_allclose(output.channels["dq0"], expected_velocity)
    assert np.isfinite(expected_velocity).all()
    acceleration = np.gradient(
        expected_velocity, output.times, edge_order=2
    )
    assert np.isfinite(acceleration).all()

    source_mapping_report = validate_model_mapping(
        source, adapter, mapping, transforms
    )
    mapping_report = validate_model_mapping(
        output, adapter, mapping, transforms
    )
    assert [
        (issue.code, issue.channel, issue.frame)
        for issue in mapping_report.issues
    ] == [
        (issue.code, issue.channel, issue.frame)
        for issue in source_mapping_report.issues
    ]
    collision_report = sample_collisions(
        output,
        adapter,
        mapping,
        max_samples=100,
        transforms=transforms,
    )
    assert collision_report.metrics["sampled_contact_frames"] >= 0
