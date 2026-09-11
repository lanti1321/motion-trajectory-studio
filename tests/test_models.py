from __future__ import annotations

import numpy as np
import pytest

from studio.core.trajectory import TrajectoryData
from studio.models.adapter import MuJoCoModelAdapter, auto_map_channels
from studio.validation.checks import validate_model_mapping


MJCF = """<mujoco model="two_link">
  <worldbody>
    <camera name="overview" pos="1 0 1"/>
    <body name="base">
      <joint name="joint0" type="hinge" range="-1 1"/>
      <geom type="capsule" size=".03" fromto="0 0 0 0 0 .3"/>
      <body name="tool" pos="0 0 .3">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .2 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

URDF = """<robot name="simple">
  <link name="base"/>
  <link name="tip">
    <inertial><mass value="1"/><origin xyz="0 0 0"/>
      <inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size=".1 .1 .1"/></geometry></visual>
  </link>
  <joint name="elbow" type="revolute">
    <parent link="base"/><child link="tip"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="2"/>
  </joint>
</robot>
"""

SIBLING_MODEL = """<mujoco>
  <worldbody>
    <body name="left_root" pos="0 .3 0">
      <joint name="left_joint"/><geom type="sphere" size=".01"/>
    </body>
    <body name="right_root" pos="0 -.3 0">
      <joint name="right_joint"/><geom type="sphere" size=".01"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_mjcf_adapter_and_mapping(tmp_path) -> None:
    path = tmp_path / "model.xml"
    path.write_text(MJCF, encoding="utf-8")
    adapter = MuJoCoModelAdapter(path)
    assert adapter.info.format == "mjcf"
    assert [joint.name for joint in adapter.info.joints] == ["joint0", "joint1"]
    assert "tool" in adapter.info.body_names
    assert adapter.info.camera_names == ("overview",)
    assert auto_map_channels(["q0", "q1"], adapter.info) == {
        "q0": "joint0",
        "q1": "joint1",
    }
    adapter.apply_positions({"joint0": 0.2, "joint1": -0.3})
    position, rotation = adapter.body_pose("tool")
    assert position.shape == (3,)
    assert rotation.shape == (3, 3)


def test_urdf_adapter(tmp_path) -> None:
    path = tmp_path / "simple.urdf"
    path.write_text(URDF, encoding="utf-8")
    adapter = MuJoCoModelAdapter(path)
    assert adapter.info.format == "urdf"
    assert adapter.info.joints[0].name == "elbow"
    assert adapter.info.joints[0].lower == pytest.approx(-1.5)


def test_mapping_validation_checks_joint_limits(tmp_path) -> None:
    path = tmp_path / "model.xml"
    path.write_text(MJCF, encoding="utf-8")
    adapter = MuJoCoModelAdapter(path)
    trajectory = TrajectoryData(
        np.arange(3) / 100,
        {"q0": np.asarray([0.0, 2.0, 0.0]), "q1": np.zeros(3)},
        ["q0", "q1"],
    )
    report = validate_model_mapping(
        trajectory, adapter, {"q0": "joint0", "q1": "joint1"}
    )
    assert not report.ok
    assert any(issue.code == "joint_limit" for issue in report.issues)


def test_generic_joint_center_distance_adjustment(tmp_path) -> None:
    path = tmp_path / "siblings.xml"
    path.write_text(SIBLING_MODEL, encoding="utf-8")
    adapter = MuJoCoModelAdapter(path)
    actual = adapter.set_joint_center_distance(
        "left_joint",
        "right_joint",
        "left_root",
        "right_root",
        0.422,
    )
    assert actual == pytest.approx(0.422, abs=1e-7)
