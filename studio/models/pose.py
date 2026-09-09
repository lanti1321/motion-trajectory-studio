from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.models.adapter import JointInfo, ModelAdapter, ModelInfo


HAND_TOKENS = (
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
    "finger",
    "mcp",
    "cmc",
    "dip",
    "pip",
    "ip",
)

GROUP_LEFT_ARM = "左臂"
GROUP_RIGHT_ARM = "右臂"
GROUP_LEFT_HAND = "左手"
GROUP_RIGHT_HAND = "右手"
GROUP_LEFT_GRIPPER = "左夹爪"
GROUP_RIGHT_GRIPPER = "右夹爪"
GROUP_OTHER = "其他关节"

MODULE_ORDER = (
    ("left_hand", GROUP_LEFT_HAND),
    ("right_hand", GROUP_RIGHT_HAND),
    ("left_arm", GROUP_LEFT_ARM),
    ("right_arm", GROUP_RIGHT_ARM),
    ("left_gripper", GROUP_LEFT_GRIPPER),
    ("right_gripper", GROUP_RIGHT_GRIPPER),
    ("other", GROUP_OTHER),
)

FINGER_ORDER = (
    ("thumb", "拇指"),
    ("index", "食指"),
    ("middle", "中指"),
    ("ring", "无名指"),
    ("pinky", "小指"),
)

_GRIPPER_TOKENS = ("finger_joint", "gripper", "jaw")
_ARM_JOINT_RE = re.compile(r"(?:^|_)joint(\d+)$", re.IGNORECASE)
_GRIPPER_JOINT_RE = re.compile(r"finger_joint(\d+)", re.IGNORECASE)
_AXIS_LABELS = (
    ("cmc_yaw", "CMC 偏航"),
    ("cmc_pitch", "CMC 俯仰"),
    ("mcp_pitch", "MCP 俯仰"),
    ("mcp_yaw", "MCP 偏航"),
    ("mcp", "MCP"),
    ("pip", "PIP"),
    ("dip", "DIP"),
    ("ip", "IP"),
)


def scalar_joints(model: ModelInfo) -> list[JointInfo]:
    return [joint for joint in model.joints if joint.qpos_width == 1]


def independent_scalar_joints(model: ModelInfo) -> list[JointInfo]:
    return [joint for joint in scalar_joints(model) if joint.independent]


def scalar_joint_positions(adapter: ModelAdapter) -> dict[str, float]:
    info = adapter.info
    values: dict[str, float] = {}
    qpos = getattr(adapter, "data", None)
    qpos_array = None if qpos is None else getattr(qpos, "qpos", None)
    for joint in scalar_joints(info):
        if qpos_array is None:
            values[joint.name] = 0.0
            continue
        values[joint.name] = float(qpos_array[joint.qpos_address])
    return values


def independent_joint_positions(adapter: ModelAdapter) -> dict[str, float]:
    names = {joint.name for joint in independent_scalar_joints(adapter.info)}
    return {
        name: value
        for name, value in scalar_joint_positions(adapter).items()
        if name in names
    }


def channel_mapping_for_joints(
    joint_names: Sequence[str],
    preset_mapping: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Prefer preset channel names when they already target a joint."""
    joint_to_channel = {
        str(joint): str(channel)
        for channel, joint in dict(preset_mapping or {}).items()
        if str(joint) in set(joint_names)
    }
    mapping: dict[str, str] = {}
    used_channels: set[str] = set()
    for joint_name in joint_names:
        channel = joint_to_channel.get(joint_name, joint_name)
        if channel in used_channels:
            channel = joint_name
        used_channels.add(channel)
        mapping[channel] = joint_name
    return mapping


def _is_gripper_joint(key: str) -> bool:
    return any(token in key for token in _GRIPPER_TOKENS) and not key.startswith(
        ("lh_", "rh_")
    )


def classify_joint_group(joint_name: str) -> str:
    key = joint_name.casefold()
    is_left = key.startswith("lh_") or "left" in key
    is_right = key.startswith("rh_") or "right" in key
    if _is_gripper_joint(key):
        if is_left:
            return GROUP_LEFT_GRIPPER
        if is_right:
            return GROUP_RIGHT_GRIPPER
        return GROUP_OTHER
    is_hand = key.startswith(("lh_", "rh_")) or any(
        token in key for token in HAND_TOKENS
    )
    if is_hand and is_left:
        return GROUP_LEFT_HAND
    if is_hand and is_right:
        return GROUP_RIGHT_HAND
    if is_left:
        return GROUP_LEFT_ARM
    if is_right:
        return GROUP_RIGHT_ARM
    return GROUP_OTHER


def default_joint_groups(channel_to_joint: Mapping[str, str]) -> dict[str, list[str]]:
    buckets = {
        GROUP_LEFT_ARM: [],
        GROUP_RIGHT_ARM: [],
        GROUP_LEFT_HAND: [],
        GROUP_RIGHT_HAND: [],
        GROUP_LEFT_GRIPPER: [],
        GROUP_RIGHT_GRIPPER: [],
        GROUP_OTHER: [],
    }
    for channel, joint_name in channel_to_joint.items():
        buckets[classify_joint_group(joint_name)].append(str(channel))
    return {name: channels for name, channels in buckets.items() if channels}


def _joint_name(joint: JointInfo | str) -> str:
    return joint.name if isinstance(joint, JointInfo) else str(joint)


def part_modules_from_joints(
    joints: Sequence[JointInfo | str],
) -> list[dict[str, object]]:
    buckets: dict[str, list[str]] = {module_id: [] for module_id, _title in MODULE_ORDER}
    for joint in joints:
        name = _joint_name(joint)
        group = classify_joint_group(name)
        for module_id, title in MODULE_ORDER:
            if title == group:
                buckets[module_id].append(name)
                break
    return [
        {
            "id": module_id,
            "title": title,
            "joint_names": buckets[module_id],
        }
        for module_id, title in MODULE_ORDER
        if buckets[module_id]
    ]


def _finger_key(joint_name: str) -> str | None:
    key = joint_name.casefold()
    for token, _title in FINGER_ORDER:
        if token in key:
            return token
    return None


def finger_subgroups(joint_names: Sequence[str]) -> list[dict[str, object]]:
    names = [str(name) for name in joint_names]
    if not names:
        return []
    if not any(
        classify_joint_group(name) in {GROUP_LEFT_HAND, GROUP_RIGHT_HAND}
        for name in names
    ):
        return [{"id": "joints", "title": "", "joint_names": names}]
    buckets: dict[str, list[str]] = {token: [] for token, _title in FINGER_ORDER}
    other: list[str] = []
    for name in names:
        finger = _finger_key(name)
        if finger is None:
            other.append(name)
        else:
            buckets[finger].append(name)
    groups = [
        {"id": token, "title": title, "joint_names": buckets[token]}
        for token, title in FINGER_ORDER
        if buckets[token]
    ]
    if other:
        groups.append({"id": "other", "title": "其他", "joint_names": other})
    return groups


def joint_display_name(joint_name: str) -> str:
    key = joint_name.casefold()
    stripped = key.removeprefix("lh_").removeprefix("rh_")
    finger = next(
        (title for token, title in FINGER_ORDER if token in stripped),
        None,
    )
    axis = next((title for token, title in _AXIS_LABELS if token in stripped), None)
    if finger and axis:
        return f"{finger} {axis}"
    if axis:
        return axis
    if finger:
        return finger
    arm = _ARM_JOINT_RE.search(joint_name)
    if arm:
        return f"关节 {arm.group(1)}"
    gripper = _GRIPPER_JOINT_RE.search(joint_name)
    if gripper:
        return f"夹爪 {gripper.group(1)}"
    return joint_name


def hold_pose_trajectory(
    channel_values: Mapping[str, float],
    duration: float = 2.0,
    hz: float = 50.0,
) -> TrajectoryData:
    if not channel_values:
        raise ValueError("hold pose needs at least one joint channel")
    if duration <= 0 or hz <= 0:
        raise ValueError("hold pose duration and hz must be positive")
    count = max(2, int(round(duration * hz)) + 1)
    times = np.arange(count, dtype=np.float64) / hz
    times[-1] = duration
    channels = {
        str(name): np.full(count, float(value), dtype=np.float64)
        for name, value in channel_values.items()
    }
    return TrajectoryData(times, channels, list(channels))


def extend_trajectory_with_constants(
    data: TrajectoryData,
    channel_values: Mapping[str, float],
) -> TrajectoryData:
    channels = {name: values.copy() for name, values in data.channels.items()}
    positions = list(data.position_channels)
    added = False
    for name, value in channel_values.items():
        channel = str(name)
        if channel in channels:
            continue
        channels[channel] = np.full(data.frame_count, float(value), dtype=np.float64)
        positions.append(channel)
        added = True
    if not added:
        return data
    return TrajectoryData(
        data.times.copy(),
        channels,
        positions,
        dict(data.velocity_channels),
        dict(data.effort_channels),
        dict(data.metadata),
    )


def joint_pose_patch_operation(
    data: TrajectoryData,
    channel: str,
    channel_value: float,
    frame: int | None = None,
) -> dict[str, object]:
    if channel not in data.channels:
        raise KeyError(f"trajectory has no channel named {channel!r}")
    values = data.channels[channel]
    static = float(np.ptp(values)) < 1e-9
    if static:
        return {
            "type": "sample_patch",
            "start": float(data.times[0]),
            "end": float(data.times[-1]),
            "channel_values": {
                channel: np.full(data.frame_count, float(channel_value)).tolist()
            },
        }
    index = 0 if frame is None else int(np.clip(frame, 0, data.frame_count - 1))
    time_s = float(data.times[index])
    return {
        "type": "sample_patch",
        "start": time_s,
        "end": time_s,
        "channel_values": {channel: [float(channel_value)]},
    }


def joint_value_from_channel(
    channel_value: float,
    transform: Mapping[str, float] | None = None,
) -> float:
    scale = float((transform or {}).get("scale", 1.0))
    offset = float((transform or {}).get("offset", 0.0))
    value = float(channel_value) * scale + offset
    if transform and "minimum" in transform:
        value = max(value, float(transform["minimum"]))
    if transform and "maximum" in transform:
        value = min(value, float(transform["maximum"]))
    return value


def channel_value_from_joint(
    joint_value: float,
    transform: Mapping[str, float] | None = None,
) -> float:
    scale = float((transform or {}).get("scale", 1.0))
    offset = float((transform or {}).get("offset", 0.0))
    if abs(scale) < 1e-12:
        raise ValueError("mapping scale must be non-zero")
    return (float(joint_value) - offset) / scale
