from __future__ import annotations

import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.editing.kinematics import KinematicsEngine
from studio.models.adapter import MuJoCoModelAdapter


MODEL = """<mujoco model="kinematics_test">
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <joint name="joint0" type="hinge" axis="0 0 1" range="-2 2"/>
      <geom type="capsule" size=".02" fromto="0 0 0 .2 0 0"/>
      <body name="tool" pos=".2 0 0">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="sphere" size=".03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

SIX_DOF_MODEL = """<mujoco model="six_dof">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <joint name="x" type="slide" axis="1 0 0" range="-.2 .2"/>
      <geom type="sphere" size=".01" mass=".01"/>
      <body>
        <joint name="y" type="slide" axis="0 1 0" range="-.2 .2"/>
        <geom type="sphere" size=".01" mass=".01"/>
        <body>
          <joint name="z" type="slide" axis="0 0 1" range="-.2 .2"/>
          <geom type="sphere" size=".01" mass=".01"/>
          <body>
            <joint name="rx" type="hinge" axis="1 0 0" range="-1 1"/>
            <geom type="sphere" size=".01" mass=".01"/>
            <body>
              <joint name="ry" type="hinge" axis="0 1 0" range="-1 1"/>
              <geom type="sphere" size=".01" mass=".01"/>
              <body name="tool">
                <joint name="rz" type="hinge" axis="0 0 1" range="-1 1"/>
                <geom type="sphere" size=".01" mass=".01"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_generic_batch_ik_preserves_existing_pose(tmp_path) -> None:
    model_path = tmp_path / "model.xml"
    model_path.write_text(MODEL, encoding="utf-8")
    adapter = MuJoCoModelAdapter(model_path)
    times = np.arange(21) / 100
    trajectory = TrajectoryData(
        times,
        {
            "a": np.linspace(0, 0.2, len(times)),
            "b": np.linspace(0, -0.1, len(times)),
        },
        ["a", "b"],
    )
    engine = KinematicsEngine(adapter, {"a": "joint0", "b": "joint1"})
    frames = np.arange(trajectory.frame_count)
    positions, rotations = engine.sample_body_poses(trajectory, "tool", frames)
    solved, report = engine.batch_ik(
        trajectory, frames, "tool", positions, rotations
    )
    np.testing.assert_allclose(solved, trajectory.matrix(["a", "b"]), atol=1e-5)
    assert not report.failed_frames
    assert report.max_position_error < 2e-6


def test_tcp_offset_and_bezier_are_model_independent(tmp_path) -> None:
    model_path = tmp_path / "six.xml"
    model_path.write_text(SIX_DOF_MODEL, encoding="utf-8")
    adapter = MuJoCoModelAdapter(model_path)
    times = np.arange(21) / 100
    names = ["x", "y", "z", "rx", "ry", "rz"]
    trajectory = TrajectoryData(
        times,
        {name: np.zeros(len(times)) for name in names},
        names,
    )
    engine = KinematicsEngine(
        adapter,
        {name: name for name in names},
        base_body_name="world",
    )
    offset = np.asarray([0.01, -0.015, 0.02])
    shifted, report = engine.cartesian_offset(
        trajectory, "tool", 0.0, offset, 0.1
    )
    assert not report.failed_frames
    positions, _ = engine.sample_body_poses(shifted, "tool")
    np.testing.assert_allclose(positions[-1], offset, atol=5e-5)

    curved, curve_report = engine.cubic_bezier(
        trajectory,
        "tool",
        0.0,
        0.2,
        np.asarray([0.02, 0.01, 0.0]),
        np.asarray([0.02, -0.01, 0.0]),
    )
    assert not curve_report.failed_frames
    curve_positions, _ = engine.sample_body_poses(curved, "tool")
    assert np.max(curve_positions[:, 0]) > 0.01
    np.testing.assert_allclose(curve_positions[[0, -1]], 0.0, atol=5e-5)


def test_position_only_ik_allows_translation_without_orientation_control(tmp_path) -> None:
    model_path = tmp_path / "six.xml"
    model_path.write_text(SIX_DOF_MODEL, encoding="utf-8")
    adapter = MuJoCoModelAdapter(model_path)
    names = ["x", "y", "z"]
    trajectory = TrajectoryData(
        np.asarray([0.0]), {name: np.zeros(1) for name in names}, names
    )
    engine = KinematicsEngine(adapter, {name: name for name in names})
    solved, report = engine.batch_ik(
        trajectory,
        np.asarray([0]),
        "tool",
        np.asarray([[0.02, -0.01, 0.03]]),
        np.asarray([np.eye(3)]),
        position_only=True,
    )
    assert not report.failed_frames
    np.testing.assert_allclose(solved[0], [0.02, -0.01, 0.03], atol=5e-5)
