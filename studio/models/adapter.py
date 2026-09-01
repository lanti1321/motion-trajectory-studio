from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class JointInfo:
    name: str
    joint_type: str
    qpos_address: int
    dof_address: int
    qpos_width: int
    limited: bool
    lower: float | None
    upper: float | None
    independent: bool = True


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    format: str
    joints: tuple[JointInfo, ...]
    body_names: tuple[str, ...]
    camera_names: tuple[str, ...] = ()


class ModelAdapter(ABC):
    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def apply_positions(self, values: dict[str, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class MuJoCoModelAdapter(ModelAdapter):
    """Loads both MJCF/XML and MuJoCo-compatible URDF files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"model file not found: {self.path}")
        if self.path.suffix.lower() not in {".xml", ".mjcf", ".urdf"}:
            raise ValueError("model must be an MJCF/XML or URDF file")
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        self._joints = self._read_joints()
        self._joint_by_name = {joint.name: joint for joint in self._joints}
        self._body_names = tuple(
            name
            for index in range(self.model.nbody)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, index))
        )
        self._camera_names = tuple(
            name
            for index in range(self.model.ncam)
            if (name := mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, index
            ))
        )
        mujoco.mj_forward(self.model, self.data)

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            self.path,
            "urdf" if self.path.suffix.lower() == ".urdf" else "mjcf",
            tuple(self._joints),
            self._body_names,
            self._camera_names,
        )

    def _read_joints(self) -> list[JointInfo]:
        widths = {
            int(mujoco.mjtJoint.mjJNT_FREE): 7,
            int(mujoco.mjtJoint.mjJNT_BALL): 4,
            int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
            int(mujoco.mjtJoint.mjJNT_HINGE): 1,
        }
        type_names = {
            int(mujoco.mjtJoint.mjJNT_FREE): "free",
            int(mujoco.mjtJoint.mjJNT_BALL): "ball",
            int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
            int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
        }
        dependent_joint_ids = {
            int(self.model.eq_obj1id[index])
            for index in range(self.model.neq)
            if int(self.model.eq_type[index]) == int(mujoco.mjtEq.mjEQ_JOINT)
            and int(self.model.eq_obj2id[index]) >= 0
        }
        result: list[JointInfo] = []
        for index in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            if not name:
                name = f"joint_{index}"
            joint_type = int(self.model.jnt_type[index])
            limited = bool(self.model.jnt_limited[index])
            lower, upper = (float(v) for v in self.model.jnt_range[index])
            result.append(
                JointInfo(
                    name,
                    type_names[joint_type],
                    int(self.model.jnt_qposadr[index]),
                    int(self.model.jnt_dofadr[index]),
                    widths[joint_type],
                    limited,
                    lower if limited else None,
                    upper if limited else None,
                    index not in dependent_joint_ids,
                )
            )
        return result

    def apply_positions(self, values: dict[str, float]) -> None:
        for joint_name, value in values.items():
            joint = self._joint_by_name.get(joint_name)
            if joint is None:
                raise KeyError(f"model has no joint named {joint_name!r}")
            if joint.qpos_width != 1:
                raise ValueError(
                    f"scalar channel cannot directly drive {joint.joint_type} joint {joint_name!r}"
                )
            self.data.qpos[joint.qpos_address] = float(value)
        self._apply_joint_equalities()
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _apply_joint_equalities(self) -> None:
        for index in range(self.model.neq):
            if int(self.model.eq_type[index]) != int(mujoco.mjtEq.mjEQ_JOINT):
                continue
            first_id = int(self.model.eq_obj1id[index])
            second_id = int(self.model.eq_obj2id[index])
            if first_id < 0 or second_id < 0:
                continue
            first_address = int(self.model.jnt_qposadr[first_id])
            second_address = int(self.model.jnt_qposadr[second_id])
            value = float(self.data.qpos[second_address])
            coefficients = self.model.eq_data[index, :5]
            self.data.qpos[first_address] = sum(
                float(coefficient) * value**power
                for power, coefficient in enumerate(coefficients)
            )

    def body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise KeyError(f"model has no body named {body_name!r}")
        return self.data.xpos[body_id].copy(), self.data.xmat[body_id].reshape(3, 3).copy()

    def set_joint_center_distance(
        self,
        joint_a: str,
        joint_b: str,
        body_a: str,
        body_b: str,
        target_distance: float,
    ) -> float:
        """Moves two sibling body roots symmetrically to set joint-center spacing."""
        if target_distance <= 0:
            raise ValueError("target joint-center distance must be positive")
        joint_a_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_a
        )
        joint_b_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_b
        )
        body_a_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_a
        )
        body_b_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_b
        )
        if min(joint_a_id, joint_b_id, body_a_id, body_b_id) < 0:
            raise ValueError("joint-center adjustment references a missing model object")
        if self.model.body_parentid[body_a_id] != self.model.body_parentid[body_b_id]:
            raise ValueError("adjusted body roots must share the same parent")
        mujoco.mj_forward(self.model, self.data)
        separation = self.data.xanchor[joint_a_id] - self.data.xanchor[joint_b_id]
        current_distance = float(np.linalg.norm(separation))
        if current_distance <= 1e-12:
            raise ValueError("joint centers overlap and have no separation direction")
        world_shift = (
            separation / current_distance * ((target_distance - current_distance) / 2)
        )
        parent_id = int(self.model.body_parentid[body_a_id])
        parent_rotation = self.data.xmat[parent_id].reshape(3, 3)
        local_shift = parent_rotation.T @ world_shift
        self.model.body_pos[body_a_id] += local_shift
        self.model.body_pos[body_b_id] -= local_shift
        mujoco.mj_forward(self.model, self.data)
        actual_distance = float(
            np.linalg.norm(
                self.data.xanchor[joint_a_id] - self.data.xanchor[joint_b_id]
            )
        )
        if abs(actual_distance - target_distance) > 1e-6:
            raise RuntimeError(
                f"joint-center distance adjustment failed: {actual_distance:.9f}m"
            )
        return actual_distance

    def joint_limits(self, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for name in joint_names:
            joint = self._joint_by_name[name]
            lower.append(joint.lower if joint.lower is not None else -np.inf)
            upper.append(joint.upper if joint.upper is not None else np.inf)
        return np.asarray(lower), np.asarray(upper)

    def ancestor_joint_names(self, body_name: str) -> list[str]:
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise KeyError(f"model has no body named {body_name!r}")
        body_ids: set[int] = set()
        current = int(body_id)
        while current > 0:
            body_ids.add(current)
            current = int(self.model.body_parentid[current])
        return [
            joint.name
            for index, joint in enumerate(self._joints)
            if int(self.model.jnt_bodyid[index]) in body_ids and joint.qpos_width == 1
        ]

    def render(self, width: int = 960, height: int = 540) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data)
            return renderer.render().copy()
        finally:
            renderer.close()


def auto_map_channels(channel_names: list[str], model: ModelInfo) -> dict[str, str]:
    """Returns channel -> joint mappings for confident exact/normalized matches."""
    joint_names = [
        joint.name
        for joint in model.joints
        if joint.qpos_width == 1 and joint.independent
    ]
    exact = {name: name for name in joint_names}
    normalized: dict[str, list[str]] = {}
    for name in joint_names:
        normalized.setdefault(_normalize(name), []).append(name)
    result: dict[str, str] = {}
    for channel in channel_names:
        if channel in exact:
            result[channel] = exact[channel]
            continue
        candidates = normalized.get(_normalize(channel), [])
        if len(candidates) == 1:
            result[channel] = candidates[0]
    if not result and len(channel_names) == len(joint_names):
        result.update(zip(channel_names, joint_names))
    return result


def _normalize(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum()).removeprefix("joint").removeprefix("q")
