from __future__ import annotations

import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.models.adapter import MuJoCoModelAdapter
from studio.validation.checks import (
    continuity_metrics,
    detect_stalls,
    sample_collisions,
    validate_model_mapping,
)


MODEL = """<mujoco>
  <worldbody>
    <body name="a">
      <joint name="a_joint" type="slide" axis="1 0 0" range="-1 1"/>
      <geom type="sphere" size=".1"/>
    </body>
    <body name="b">
      <joint name="b_joint" type="slide" axis="0 1 0" range="-1 1"/>
      <geom type="sphere" size=".1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_stall_and_continuity_detection() -> None:
    times = np.arange(1001) / 100
    q = np.sin(times)
    q[300:401] = q[300]
    data = TrajectoryData(times, {"q": q}, ["q"])
    stalls = detect_stalls(data, speed_threshold=1e-3, minimum_duration=0.5)
    assert stalls
    metrics = continuity_metrics(data, 3.0)
    assert "velocity_jump" in metrics["q"]


def test_mapping_transform_is_applied_before_limit_check(tmp_path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL, encoding="utf-8")
    adapter = MuJoCoModelAdapter(model_path)
    times = np.arange(3) / 100
    data = TrajectoryData(times, {"motor": np.asarray([-10.0, 0.0, 10.0])}, ["motor"])
    report = validate_model_mapping(
        data,
        adapter,
        {"motor": "a_joint"},
        {"motor": {"scale": 0.05, "offset": 0.0}},
    )
    assert report.ok


def test_collision_sampling(tmp_path) -> None:
    model_path = tmp_path / "collision.xml"
    model_path.write_text(MODEL, encoding="utf-8")
    adapter = MuJoCoModelAdapter(model_path)
    times = np.arange(3) / 100
    data = TrajectoryData(
        times,
        {"a": np.zeros(3), "b": np.zeros(3)},
        ["a", "b"],
    )
    report = sample_collisions(
        data, adapter, {"a": "a_joint", "b": "b_joint"}
    )
    assert report.metrics["sampled_contact_frames"] > 0
    contacts = [issue for issue in report.issues if issue.code == "model_contact"]
    assert contacts
    assert "frame" in contacts[0].message
    assert "↔" in contacts[0].message


def test_collision_sampling_applies_transform_clamps(tmp_path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL, encoding="utf-8")
    adapter = MuJoCoModelAdapter(model_path)
    times = np.arange(3) / 100
    data = TrajectoryData(
        times,
        {"a": np.full(3, 100.0)},
        ["a"],
    )
    sample_collisions(
        data,
        adapter,
        {"a": "a_joint"},
        transforms={"a": {"scale": 2.0, "maximum": 0.5}},
    )
    joint = next(
        value for value in adapter.info.joints if value.name == "a_joint"
    )
    assert adapter.data.qpos[joint.qpos_address] == 0.5
