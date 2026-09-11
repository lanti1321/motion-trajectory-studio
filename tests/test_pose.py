from __future__ import annotations

import numpy as np
import pytest

from studio.editing.operations import apply_operation
from studio.models.pose import (
    GROUP_LEFT_ARM,
    GROUP_LEFT_HAND,
    GROUP_RIGHT_ARM,
    GROUP_RIGHT_HAND,
    channel_mapping_for_joints,
    channel_value_from_joint,
    classify_joint_group,
    default_joint_groups,
    extend_trajectory_with_constants,
    finger_subgroups,
    hold_pose_trajectory,
    joint_display_name,
    joint_pose_patch_operation,
    joint_value_from_channel,
    part_modules_from_joints,
)


def test_o6_joints_group_into_hands_and_arms() -> None:
    assert classify_joint_group("openarm_left_joint3") == GROUP_LEFT_ARM
    assert classify_joint_group("openarm_right_joint7") == GROUP_RIGHT_ARM
    assert classify_joint_group("lh_index_mcp_pitch") == GROUP_LEFT_HAND
    assert classify_joint_group("rh_thumb_cmc_yaw") == GROUP_RIGHT_HAND
    groups = default_joint_groups(
        {
            "q0": "openarm_left_joint1",
            "lh_index_mcp_pitch": "lh_index_mcp_pitch",
            "q8": "openarm_right_joint1",
            "rh_pinky_mcp_pitch": "rh_pinky_mcp_pitch",
        }
    )
    assert groups[GROUP_LEFT_ARM] == ["q0"]
    assert groups[GROUP_LEFT_HAND] == ["lh_index_mcp_pitch"]
    assert groups[GROUP_RIGHT_ARM] == ["q8"]
    assert groups[GROUP_RIGHT_HAND] == ["rh_pinky_mcp_pitch"]


def test_hold_pose_and_channel_mapping_keep_openarm_preset_names() -> None:
    mapping = channel_mapping_for_joints(
        ["openarm_left_joint1", "lh_index_mcp_pitch"],
        {"q0": "openarm_left_joint1"},
    )
    assert mapping == {
        "q0": "openarm_left_joint1",
        "lh_index_mcp_pitch": "lh_index_mcp_pitch",
    }
    pose = hold_pose_trajectory({"q0": 0.1, "lh_index_mcp_pitch": 0.4}, duration=1.0, hz=10.0)
    assert pose.frame_count == 11
    assert pose.position_channels == ["q0", "lh_index_mcp_pitch"]
    np.testing.assert_allclose(pose.channels["lh_index_mcp_pitch"], 0.4)


def test_extend_trajectory_adds_missing_finger_channels() -> None:
    base = hold_pose_trajectory({"q0": 0.0}, duration=0.1, hz=10.0)
    extended = extend_trajectory_with_constants(base, {"q0": 1.0, "lh_index_mcp_pitch": 0.2})
    assert "lh_index_mcp_pitch" in extended.position_channels
    np.testing.assert_allclose(extended.channels["q0"], 0.0)
    np.testing.assert_allclose(extended.channels["lh_index_mcp_pitch"], 0.2)


def test_mapping_transforms_round_trip() -> None:
    transform = {"scale": -2.0, "offset": 0.5}
    joint = joint_value_from_channel(0.25, transform)
    assert joint == pytest.approx(0.0)
    assert channel_value_from_joint(joint, transform) == pytest.approx(0.25)


def test_static_pose_patch_writes_all_frames() -> None:
    pose = hold_pose_trajectory({"lh_index_mcp_pitch": 0.0}, duration=0.2, hz=10.0)
    operation = joint_pose_patch_operation(pose, "lh_index_mcp_pitch", 0.7)
    assert operation["start"] == pytest.approx(0.0)
    assert operation["end"] == pytest.approx(0.2)
    assert operation["channel_values"]["lh_index_mcp_pitch"] == [0.7] * pose.frame_count
    patched = apply_operation(pose, operation)
    np.testing.assert_allclose(patched.channels["lh_index_mcp_pitch"], 0.7)


def test_moving_channel_patch_writes_current_frame_only() -> None:
    pose = hold_pose_trajectory({"lh_index_mcp_pitch": 0.0}, duration=0.2, hz=10.0)
    pose.channels["lh_index_mcp_pitch"][-1] = 0.3
    operation = joint_pose_patch_operation(
        pose, "lh_index_mcp_pitch", 0.9, frame=0
    )
    assert operation["start"] == pytest.approx(0.0)
    assert operation["end"] == pytest.approx(0.0)
    assert operation["channel_values"]["lh_index_mcp_pitch"] == [0.9]


def test_part_module_patch_writes_constant_value_across_selected_range() -> None:
    pose = hold_pose_trajectory(
        {"lh_index_mcp_pitch": 0.0}, duration=1.0, hz=10.0
    )
    pose.channels["lh_index_mcp_pitch"] = np.linspace(
        0.0, 1.0, pose.frame_count
    )
    operation = joint_pose_patch_operation(
        pose,
        "lh_index_mcp_pitch",
        0.45,
        frame=5,
        selected_range=(0.2, 0.7),
    )
    assert operation["start"] == pytest.approx(0.2)
    assert operation["end"] == pytest.approx(0.7)
    assert operation["channel_values"]["lh_index_mcp_pitch"] == [0.45] * 6
    patched = apply_operation(pose, operation)
    np.testing.assert_allclose(
        patched.channels["lh_index_mcp_pitch"][2:8], 0.45
    )
    assert patched.channels["lh_index_mcp_pitch"][1] == pytest.approx(0.1)
    assert patched.channels["lh_index_mcp_pitch"][8] == pytest.approx(0.8)


LEFT_HAND_JOINTS = [
    "lh_thumb_cmc_yaw",
    "lh_thumb_cmc_pitch",
    "lh_index_mcp_pitch",
    "lh_middle_mcp_pitch",
    "lh_ring_mcp_pitch",
    "lh_pinky_mcp_pitch",
]


def test_part_modules_from_o6_style_names() -> None:
    names = [
        "openarm_left_joint1",
        "openarm_left_joint7",
        "openarm_right_joint1",
        "openarm_right_joint7",
        *LEFT_HAND_JOINTS,
        "rh_thumb_cmc_yaw",
        "rh_index_mcp_pitch",
        "rh_pinky_mcp_pitch",
    ]
    modules = {str(module["id"]): module for module in part_modules_from_joints(names)}
    assert list(modules) == ["left_hand", "right_hand", "left_arm", "right_arm"]
    assert modules["left_hand"]["title"] == GROUP_LEFT_HAND
    assert modules["left_hand"]["joint_names"] == LEFT_HAND_JOINTS
    assert modules["left_arm"]["joint_names"] == [
        "openarm_left_joint1",
        "openarm_left_joint7",
    ]
    assert modules["right_arm"]["joint_names"] == [
        "openarm_right_joint1",
        "openarm_right_joint7",
    ]


def test_finger_subgroups_split_left_hand() -> None:
    groups = finger_subgroups(LEFT_HAND_JOINTS)
    assert [group["title"] for group in groups] == [
        "拇指",
        "食指",
        "中指",
        "无名指",
        "小指",
    ]
    assert groups[0]["joint_names"] == [
        "lh_thumb_cmc_yaw",
        "lh_thumb_cmc_pitch",
    ]
    assert groups[1]["joint_names"] == ["lh_index_mcp_pitch"]
    assert joint_display_name("lh_index_mcp_pitch") == "食指 MCP 俯仰"
    assert joint_display_name("openarm_left_joint3") == "关节 3"


def test_arm_only_model_has_no_empty_hand_module() -> None:
    modules = part_modules_from_joints(
        ["openarm_left_joint1", "openarm_right_joint2"]
    )
    assert [module["id"] for module in modules] == ["left_arm", "right_arm"]


def test_damiao_gripper_is_not_a_hand_module() -> None:
    modules = {
        str(module["id"]): module
        for module in part_modules_from_joints(
            [
                "openarm_left_joint1",
                "openarm_left_finger_joint1",
                "openarm_right_finger_joint1",
            ]
        )
    }
    assert "left_hand" not in modules
    assert modules["left_arm"]["joint_names"] == ["openarm_left_joint1"]
    assert modules["left_gripper"]["joint_names"] == ["openarm_left_finger_joint1"]
    assert modules["right_gripper"]["joint_names"] == ["openarm_right_finger_joint1"]
