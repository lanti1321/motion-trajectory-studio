from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.models.adapter import MuJoCoModelAdapter
from studio.models.library import ModelLibrary
from studio.models.pose import (
    channel_mapping_for_joints,
    hold_pose_trajectory,
    independent_scalar_joints,
    part_modules_from_joints,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model_library" / "OpenArm_O6"
SCENE = MODEL_DIR / "scene.xml"

ARM_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]
HAND_JOINTS = [
    "lh_thumb_cmc_yaw",
    "lh_thumb_cmc_pitch",
    "lh_thumb_ip",
    "lh_index_mcp_pitch",
    "lh_index_dip",
    "lh_middle_mcp_pitch",
    "lh_middle_dip",
    "lh_ring_mcp_pitch",
    "lh_ring_dip",
    "lh_pinky_mcp_pitch",
    "lh_pinky_dip",
    "rh_thumb_cmc_yaw",
    "rh_thumb_cmc_pitch",
    "rh_thumb_ip",
    "rh_index_mcp_pitch",
    "rh_index_dip",
    "rh_middle_mcp_pitch",
    "rh_middle_dip",
    "rh_ring_mcp_pitch",
    "rh_ring_dip",
    "rh_pinky_mcp_pitch",
    "rh_pinky_dip",
]
ACTUATED_HAND_JOINTS = [
    "lh_thumb_cmc_yaw",
    "lh_thumb_cmc_pitch",
    "lh_index_mcp_pitch",
    "lh_middle_mcp_pitch",
    "lh_ring_mcp_pitch",
    "lh_pinky_mcp_pitch",
    "rh_thumb_cmc_yaw",
    "rh_thumb_cmc_pitch",
    "rh_index_mcp_pitch",
    "rh_middle_mcp_pitch",
    "rh_ring_mcp_pitch",
    "rh_pinky_mcp_pitch",
]
COUPLED_HAND_JOINTS = [
    "lh_thumb_ip",
    "lh_index_dip",
    "lh_middle_dip",
    "lh_ring_dip",
    "lh_pinky_dip",
    "rh_thumb_ip",
    "rh_index_dip",
    "rh_middle_dip",
    "rh_ring_dip",
    "rh_pinky_dip",
]


@pytest.mark.skipif(not SCENE.is_file(), reason="OpenArm_O6 model is unavailable")
def test_library_discovers_openarm_o6() -> None:
    entries = {entry.name: entry for entry in ModelLibrary(ROOT / "model_library").scan()}
    o6 = entries["OpenArm_O6"]
    assert o6.loadable
    assert o6.entry_file == SCENE
    assert "default_mapping" in o6.metadata


@pytest.mark.skipif(not SCENE.is_file(), reason="OpenArm_O6 model is unavailable")
def test_openarm_o6_loads_without_damiao_jaws() -> None:
    adapter = MuJoCoModelAdapter(SCENE)
    names = [joint.name for joint in adapter.info.joints]
    independent = [joint.name for joint in adapter.info.joints if joint.independent]
    dependent = [joint.name for joint in adapter.info.joints if not joint.independent]

    assert "openarm_left_finger_joint1" not in names
    assert "openarm_left_finger_joint2" not in names
    assert "openarm_right_finger_joint1" not in names
    assert "openarm_right_finger_joint2" not in names
    assert "openarm_left_hand_tcp" in adapter.info.body_names
    assert "openarm_right_hand_tcp" in adapter.info.body_names
    assert set(names) == set(ARM_JOINTS + HAND_JOINTS)
    assert set(independent) == set(ARM_JOINTS + ACTUATED_HAND_JOINTS)
    assert set(dependent) == set(COUPLED_HAND_JOINTS)
    assert len(names) == 36
    assert len(independent) == 26


@pytest.mark.skipif(not SCENE.is_file(), reason="OpenArm_O6 model is unavailable")
def test_openarm_o6_default_mapping_and_equalities() -> None:
    adapter = MuJoCoModelAdapter(SCENE)
    preset = json.loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))
    mapping = preset["default_mapping"]
    assert "q7" not in mapping
    assert "q15" not in mapping
    named = {joint.name: joint for joint in adapter.info.joints}
    for channel, settings in mapping.items():
        joint_name = settings["joint"]
        assert joint_name in named, channel
        assert named[joint_name].independent, joint_name
    for coupled in COUPLED_HAND_JOINTS:
        assert coupled not in mapping
        assert not named[coupled].independent

    adapter.apply_positions(
        {
            "lh_index_mcp_pitch": 1.0,
            "rh_index_mcp_pitch": 0.8,
            "lh_thumb_cmc_pitch": 0.4,
            "rh_thumb_cmc_pitch": 0.4,
        }
    )
    dip_left = named["lh_index_dip"]
    dip_right = named["rh_index_dip"]
    assert adapter.data.qpos[named["lh_index_mcp_pitch"].qpos_address] == pytest.approx(1.0)
    assert adapter.data.qpos[named["rh_index_mcp_pitch"].qpos_address] == pytest.approx(0.8)
    assert adapter.data.qpos[dip_left.qpos_address] == pytest.approx(0.89)
    assert adapter.data.qpos[dip_right.qpos_address] == pytest.approx(0.712)
    assert adapter.data.qpos[named["lh_thumb_ip"].qpos_address] == pytest.approx(0.916)
    assert adapter.data.qpos[named["rh_thumb_ip"].qpos_address] == pytest.approx(0.744)


@pytest.mark.skipif(not SCENE.is_file(), reason="OpenArm_O6 model is unavailable")
def test_openarm_o6_rest_pose_includes_independent_fingers() -> None:
    adapter = MuJoCoModelAdapter(SCENE)
    preset = json.loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))
    joint_names = [joint.name for joint in independent_scalar_joints(adapter.info)]
    mapping = channel_mapping_for_joints(
        joint_names,
        {
            channel: str(settings["joint"])
            for channel, settings in preset["default_mapping"].items()
        },
    )
    assert mapping["q0"] == "openarm_left_joint1"
    assert mapping["lh_index_mcp_pitch"] == "lh_index_mcp_pitch"
    assert mapping["rh_thumb_cmc_yaw"] == "rh_thumb_cmc_yaw"
    pose = hold_pose_trajectory({channel: 0.0 for channel in mapping})
    assert len(pose.position_channels) == 26
    assert "lh_pinky_mcp_pitch" in pose.channels
    assert "lh_index_dip" not in pose.channels
    assert "lh_thumb_ip" not in pose.channels
    assert "openarm_left_finger_joint2" not in pose.channels


@pytest.mark.skipif(not SCENE.is_file(), reason="OpenArm_O6 model is unavailable")
def test_openarm_o6_part_modules_expose_six_actuators_per_hand() -> None:
    adapter = MuJoCoModelAdapter(SCENE)
    modules = {
        str(module["id"]): module
        for module in part_modules_from_joints(
            independent_scalar_joints(adapter.info)
        )
    }
    assert modules["left_hand"]["joint_names"] == [
        name for name in ACTUATED_HAND_JOINTS if name.startswith("lh_")
    ]
    assert modules["right_hand"]["joint_names"] == [
        name for name in ACTUATED_HAND_JOINTS if name.startswith("rh_")
    ]
    assert "lh_index_dip" not in modules["left_hand"]["joint_names"]
    assert "rh_thumb_ip" not in modules["right_hand"]["joint_names"]
