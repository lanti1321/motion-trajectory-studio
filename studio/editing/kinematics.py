from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from studio.core.trajectory import TrajectoryData
from studio.models.adapter import MuJoCoModelAdapter

ProgressCallback = Callable[[int, int, str], bool]


@dataclass(frozen=True)
class IKFailure:
    frame: int
    position_error: float
    orientation_error: float
    limit_channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class IKReport:
    max_position_error: float
    max_orientation_error: float
    max_iterations: int
    failed_frames: tuple[int, ...]
    failures: tuple[IKFailure, ...] = ()


class KinematicsEngine:
    def __init__(
        self,
        adapter: MuJoCoModelAdapter,
        channel_to_joint: dict[str, str],
        transforms: dict[str, dict[str, float]] | None = None,
        base_body_name: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.mapping = dict(channel_to_joint)
        self.base_body_name = base_body_name
        joint_lookup = {joint.name: joint for joint in adapter.info.joints}
        invalid = [
            joint_name
            for joint_name in self.mapping.values()
            if joint_name not in joint_lookup or joint_lookup[joint_name].qpos_width != 1
        ]
        if invalid:
            raise ValueError(f"IK mapping contains unsupported joints: {invalid}")
        self.channels = list(self.mapping)
        self.joints = [joint_lookup[self.mapping[channel]] for channel in self.channels]
        self.qpos_addresses = np.asarray([joint.qpos_address for joint in self.joints], dtype=int)
        self.dof_addresses = np.asarray([joint.dof_address for joint in self.joints], dtype=int)
        self.lower = np.asarray([joint.lower if joint.lower is not None else -np.inf for joint in self.joints])
        self.upper = np.asarray([joint.upper if joint.upper is not None else np.inf for joint in self.joints])
        self.scales = np.asarray(
            [
                float((transforms or {}).get(channel, {}).get("scale", 1.0))
                for channel in self.channels
            ]
        )
        self.offsets = np.asarray(
            [
                float((transforms or {}).get(channel, {}).get("offset", 0.0))
                for channel in self.channels
            ]
        )
        self.minimums = np.asarray(
            [
                float((transforms or {}).get(channel, {}).get("minimum", -np.inf))
                for channel in self.channels
            ]
        )
        self.maximums = np.asarray(
            [
                float((transforms or {}).get(channel, {}).get("maximum", np.inf))
                for channel in self.channels
            ]
        )
        if np.any(self.scales == 0):
            raise ValueError("IK mapping scales must be non-zero")

    def apply_frame(self, trajectory: TrajectoryData, frame: int) -> None:
        self.adapter.apply_positions(
            {
                self.mapping[channel]: float(
                    np.clip(
                        trajectory.channels[channel][frame] * self.scales[index]
                        + self.offsets[index],
                        self.minimums[index],
                        self.maximums[index],
                    )
                )
                for index, channel in enumerate(self.channels)
            }
        )

    def sample_body_poses(
        self,
        trajectory: TrajectoryData,
        body_name: str,
        frames: np.ndarray | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = np.arange(trajectory.frame_count) if frames is None else np.asarray(frames, dtype=int)
        positions = np.empty((len(selected), 3))
        rotations = np.empty((len(selected), 3, 3))
        for output_index, frame in enumerate(selected):
            if progress and not progress(
                output_index, len(selected), "采样末端姿态"
            ):
                raise InterruptedError("operation canceled")
            self.apply_frame(trajectory, int(frame))
            position, rotation = self.adapter.body_pose(body_name)
            if self.base_body_name:
                base_position, base_rotation = self.adapter.body_pose(self.base_body_name)
                position = base_rotation.T @ (position - base_position)
                rotation = base_rotation.T @ rotation
            positions[output_index], rotations[output_index] = position, rotation
        if progress and not progress(
            len(selected), len(selected), "采样末端姿态"
        ):
            raise InterruptedError("operation canceled")
        return positions, rotations

    def batch_ik(
        self,
        trajectory: TrajectoryData,
        frames: np.ndarray,
        body_name: str,
        target_positions: np.ndarray,
        target_rotations: np.ndarray,
        max_iterations: int = 80,
        initial_q: np.ndarray | None = None,
        position_only: bool = False,
        progress: ProgressCallback | None = None,
    ) -> tuple[np.ndarray, IKReport]:
        body_id = mujoco.mj_name2id(
            self.adapter.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise KeyError(f"model has no body named {body_name!r}")
        frames = np.asarray(frames, dtype=int)
        if initial_q is not None and np.asarray(initial_q).shape != (len(self.channels),):
            raise ValueError("initial_q width does not match IK joint mapping")
        solved = np.empty((len(frames), len(self.channels)))
        jacobian_position = np.zeros((3, self.adapter.model.nv))
        jacobian_rotation = np.zeros((3, self.adapter.model.nv))
        max_position_error = 0.0
        max_orientation_error = 0.0
        max_used_iterations = 0
        failed: list[int] = []
        failure_details: list[IKFailure] = []
        previous: np.ndarray | None = None
        source_positions = np.clip(
            trajectory.matrix(self.channels) * self.scales + self.offsets,
            self.minimums,
            self.maximums,
        )
        for output_index, frame in enumerate(frames):
            if progress and not progress(
                output_index, len(frames), "批量逆运动学"
            ):
                raise InterruptedError("operation canceled")
            reference = source_positions[frame]
            world_target_position = target_positions[output_index]
            world_target_rotation = target_rotations[output_index]
            if self.base_body_name:
                self.apply_frame(trajectory, int(frame))
                base_position, base_rotation = self.adapter.body_pose(self.base_body_name)
                world_target_position = (
                    base_position + base_rotation @ world_target_position
                )
                world_target_rotation = base_rotation @ world_target_rotation
            if previous is not None:
                q = previous.copy()
            elif initial_q is not None:
                q = np.asarray(initial_q, dtype=float).copy()
            else:
                q = reference.copy()
            for iteration in range(max_iterations):
                self.adapter.data.qpos[self.qpos_addresses] = q
                mujoco.mj_forward(self.adapter.model, self.adapter.data)
                position_error = world_target_position - self.adapter.data.xpos[body_id]
                current_rotation = self.adapter.data.xmat[body_id].reshape(3, 3)
                orientation_error = Rotation.from_matrix(
                    world_target_rotation @ current_rotation.T
                ).as_rotvec()
                if np.linalg.norm(position_error) < 2e-6 and (
                    position_only or np.linalg.norm(orientation_error) < 2e-5
                ):
                    break
                mujoco.mj_jacBody(
                    self.adapter.model,
                    self.adapter.data,
                    jacobian_position,
                    jacobian_rotation,
                    body_id,
                )
                if position_only:
                    jacobian = jacobian_position[:, self.dof_addresses]
                    error = position_error
                else:
                    jacobian = np.vstack(
                        (
                            jacobian_position[:, self.dof_addresses],
                            jacobian_rotation[:, self.dof_addresses],
                        )
                    )
                    error = np.r_[position_error, orientation_error]
                system = jacobian @ jacobian.T + 1e-8 * np.eye(len(error))
                pseudo_inverse = np.linalg.solve(system, jacobian).T
                nullspace = np.eye(len(q)) - pseudo_inverse @ jacobian
                step = pseudo_inverse @ error + 0.02 * nullspace @ (reference - q)
                norm = np.linalg.norm(step)
                if norm > 0.05:
                    step *= 0.05 / norm
                q = np.clip(q + step, self.lower, self.upper)
            position_norm = float(np.linalg.norm(position_error))
            orientation_norm = float(np.linalg.norm(orientation_error))
            max_position_error = max(max_position_error, position_norm)
            max_orientation_error = max(max_orientation_error, orientation_norm)
            max_used_iterations = max(max_used_iterations, iteration + 1)
            if position_norm > 5e-4 or (
                not position_only and orientation_norm > np.deg2rad(0.2)
            ):
                failed.append(int(frame))
                limit_tolerance = 1e-5
                at_limit = tuple(
                    self.channels[index]
                    for index in range(len(q))
                    if (
                        np.isfinite(self.lower[index])
                        and q[index] <= self.lower[index] + limit_tolerance
                    )
                    or (
                        np.isfinite(self.upper[index])
                        and q[index] >= self.upper[index] - limit_tolerance
                    )
                )
                failure_details.append(
                    IKFailure(
                        int(frame), position_norm, orientation_norm, at_limit
                    )
                )
            solved[output_index] = q
            previous = q
        if progress and not progress(
            len(frames), len(frames), "批量逆运动学"
        ):
            raise InterruptedError("operation canceled")
        return solved, IKReport(
            max_position_error,
            max_orientation_error,
            max_used_iterations,
            tuple(failed),
            tuple(failure_details),
        )

    def cartesian_offset(
        self,
        trajectory: TrajectoryData,
        body_name: str,
        start_time: float,
        offset_xyz: np.ndarray,
        transition_duration: float,
    ) -> tuple[TrajectoryData, IKReport]:
        start = trajectory.nearest_frame(start_time)
        transition_end = trajectory.nearest_frame(
            min(trajectory.duration, start_time + transition_duration)
        )
        frames = np.arange(start, trajectory.frame_count)
        positions, rotations = self.sample_body_poses(trajectory, body_name, frames)
        target_positions = positions.copy()
        transition_count = transition_end - start
        if transition_count > 0:
            u = np.arange(transition_count + 1) / transition_count
            weight = 10 * u**3 - 15 * u**4 + 6 * u**5
            target_positions[: transition_count + 1] += weight[:, None] * np.asarray(offset_xyz)
        target_positions[transition_count + 1 :] += np.asarray(offset_xyz)
        solved, report = self.batch_ik(
            trajectory, frames, body_name, target_positions, rotations
        )
        result = trajectory.copy()
        for index, channel in enumerate(self.channels):
            result.channels[channel][frames] = (
                solved[:, index] - self.offsets[index]
            ) / self.scales[index]
        result.recompute_velocities()
        return result, report

    def cartesian_offset_window(
        self,
        trajectory: TrajectoryData,
        body_name: str,
        start_time: float,
        key_time: float,
        end_time: float,
        offset_xyz: np.ndarray,
        rotation_delta: np.ndarray,
        exit_duration: float = 0.5,
        progress: ProgressCallback | None = None,
    ) -> tuple[TrajectoryData, IKReport, int, int]:
        """Apply an end-body offset from the playhead through a selected range.

        The offset ramps in from selection start to the playhead, remains fully
        applied through selection end, then ramps out in a short extra tail.
        """
        first = trajectory.nearest_frame(start_time)
        key = trajectory.nearest_frame(key_time)
        selected_last = trajectory.nearest_frame(end_time)
        last = trajectory.nearest_frame(
            min(trajectory.duration, end_time + max(0.0, exit_duration))
        )
        if not first < key <= selected_last <= last:
            raise ValueError("末端偏移要求选区起点 < 红色播放头 <= 选区终点")
        all_frames = np.arange(first, last + 1)
        solve_hz = float(os.getenv("MOTION_STUDIO_CONSTRAINT_IK_HZ", "200"))
        stride = max(1, int(round(trajectory.frequency.hz / solve_hz)))
        solve_frames = np.unique(
            np.r_[all_frames[::stride], key, selected_last, all_frames[-1]]
        ).astype(int)
        positions, rotations = self.sample_body_poses(
            trajectory, body_name, solve_frames, progress
        )
        weights = np.ones(len(solve_frames), dtype=float)
        before = solve_frames <= key
        entry_u = (solve_frames[before] - first) / max(1, key - first)
        weights[before] = entry_u**3 * (10.0 - 15.0 * entry_u + 6.0 * entry_u**2)
        if last > selected_last:
            after = solve_frames > selected_last
            exit_u = (solve_frames[after] - selected_last) / (last - selected_last)
            weights[after] = 1.0 - exit_u**3 * (
                10.0 - 15.0 * exit_u + 6.0 * exit_u**2
            )
        target_positions = positions + weights[:, None] * np.asarray(offset_xyz)
        rotation_vector = Rotation.from_matrix(rotation_delta).as_rotvec()
        delta_rotations = Rotation.from_rotvec(
            weights[:, None] * rotation_vector
        ).as_matrix()
        target_rotations = np.einsum("nij,njk->nik", delta_rotations, rotations)
        solved, report = self.batch_ik(
            trajectory, solve_frames, body_name,
            target_positions, target_rotations, progress=progress,
        )
        dense_weights = np.ones(len(all_frames), dtype=float)
        dense_before = all_frames <= key
        dense_entry_u = (all_frames[dense_before] - first) / max(1, key - first)
        dense_weights[dense_before] = dense_entry_u**3 * (
            10.0 - 15.0 * dense_entry_u + 6.0 * dense_entry_u**2
        )
        if last > selected_last:
            dense_after = all_frames > selected_last
            dense_exit_u = (
                (all_frames[dense_after] - selected_last) / (last - selected_last)
            )
            dense_weights[dense_after] = 1.0 - dense_exit_u**3 * (
                10.0 - 15.0 * dense_exit_u + 6.0 * dense_exit_u**2
            )
        result = trajectory.copy()
        for index, channel in enumerate(self.channels):
            dense_solution = np.interp(
                all_frames, solve_frames, solved[:, index]
            )
            source_model_q = (
                trajectory.channels[channel][all_frames] * self.scales[index]
                + self.offsets[index]
            )
            # Sparse IK interpolation is piecewise linear and otherwise leaves
            # a non-zero correction slope at the splice.  Envelope the joint
            # correction with the same analytic minimum-jerk weight so its
            # position, velocity and acceleration all vanish at both joins.
            dense = source_model_q + dense_weights * (
                dense_solution - source_model_q
            )
            result.channels[channel][all_frames] = (
                dense - self.offsets[index]
            ) / self.scales[index]
        result.recompute_velocities()
        return result, report, first, last

    def constrain_body_pose(
        self,
        trajectory: TrajectoryData,
        body_name: str,
        start_time: float,
        end_time: float,
        *,
        fixed_position: bool = False,
        progress: ProgressCallback | None = None,
    ) -> tuple[TrajectoryData, IKReport]:
        """Keep body orientation, or its full pose, fixed relative to the base."""
        first = trajectory.nearest_frame(start_time)
        last = trajectory.nearest_frame(end_time)
        if last - first < 2:
            raise ValueError("末端约束区间过短")
        all_frames = np.arange(first, last + 1)
        solve_hz = float(os.getenv("MOTION_STUDIO_CONSTRAINT_IK_HZ", "200"))
        stride = max(1, int(round(trajectory.frequency.hz / solve_hz)))
        solve_frames = np.unique(
            np.r_[all_frames[::stride], all_frames[-1]]
        ).astype(int)
        positions, rotations = self.sample_body_poses(
            trajectory, body_name, solve_frames, progress
        )
        count = len(solve_frames)
        edge = max(2, min(count // 2, int(round(count * 0.1))))
        weight = np.ones(count)
        edge_u = np.linspace(0.0, 1.0, edge)
        smooth = edge_u**3 * (10.0 - 15.0 * edge_u + 6.0 * edge_u**2)
        weight[:edge] = smooth
        weight[-edge:] = smooth[::-1]
        target_positions = positions.copy()
        if fixed_position:
            target_positions += weight[:, None] * (positions[0] - positions)
        target_rotations = np.empty_like(rotations)
        fixed_rotation = rotations[0]
        for index, (rotation, blend) in enumerate(zip(rotations, weight)):
            correction = Rotation.from_matrix(
                fixed_rotation @ rotation.T
            ).as_rotvec()
            target_rotations[index] = (
                Rotation.from_rotvec(blend * correction).as_matrix() @ rotation
            )
        solved, report = self.batch_ik(
            trajectory,
            solve_frames,
            body_name,
            target_positions,
            target_rotations,
            progress=progress,
        )
        result = trajectory.copy()
        for index, channel in enumerate(self.channels):
            dense = np.interp(all_frames, solve_frames, solved[:, index])
            result.channels[channel][all_frames] = (
                dense - self.offsets[index]
            ) / self.scales[index]
        result.recompute_velocities()
        return result, report

    def constrain_relative_body_pose(
        self,
        trajectory: TrajectoryData,
        reference_engine: "KinematicsEngine",
        primary_body: str,
        follower_body: str,
        start_time: float,
        end_time: float,
        progress: ProgressCallback | None = None,
    ) -> tuple[TrajectoryData, IKReport]:
        """Keep follower pose fixed relative to a moving primary end body."""
        first = trajectory.nearest_frame(start_time)
        last = trajectory.nearest_frame(end_time)
        if last - first < 2:
            raise ValueError("双末端相对位姿约束区间过短")
        all_frames = np.arange(first, last + 1)
        solve_hz = float(os.getenv("MOTION_STUDIO_CONSTRAINT_IK_HZ", "200"))
        stride = max(1, int(round(trajectory.frequency.hz / solve_hz)))
        solve_frames = np.unique(
            np.r_[all_frames[::stride], all_frames[-1]]
        ).astype(int)
        primary_positions, primary_rotations = reference_engine.sample_body_poses(
            trajectory, primary_body, solve_frames, progress
        )
        follower_positions, follower_rotations = reference_engine.sample_body_poses(
            trajectory, follower_body, solve_frames, progress
        )
        relative_position = primary_rotations[0].T @ (
            follower_positions[0] - primary_positions[0]
        )
        relative_rotation = primary_rotations[0].T @ follower_rotations[0]
        target_positions = primary_positions + np.einsum(
            "nij,j->ni", primary_rotations, relative_position
        )
        target_rotations = np.einsum(
            "nij,jk->nik", primary_rotations, relative_rotation
        )
        solved, report = self.batch_ik(
            trajectory,
            solve_frames,
            follower_body,
            target_positions,
            target_rotations,
            progress=progress,
        )
        result = trajectory.copy()
        for index, channel in enumerate(self.channels):
            dense = np.interp(all_frames, solve_frames, solved[:, index])
            result.channels[channel][all_frames] = (
                dense - self.offsets[index]
            ) / self.scales[index]
        result.recompute_velocities()
        return result, report

    def cubic_bezier(
        self,
        trajectory: TrajectoryData,
        body_name: str,
        start_time: float,
        end_time: float,
        control_point_1: np.ndarray,
        control_point_2: np.ndarray,
        preserve_orientation_schedule: bool = True,
    ) -> tuple[TrajectoryData, IKReport]:
        start = trajectory.nearest_frame(start_time)
        end = trajectory.nearest_frame(end_time)
        frames = np.arange(start, end + 1)
        positions, rotations = self.sample_body_poses(trajectory, body_name, frames)
        u = np.linspace(0.0, 1.0, len(frames))[:, None]
        target_positions = (
            (1 - u) ** 3 * positions[0]
            + 3 * (1 - u) ** 2 * u * np.asarray(control_point_1)
            + 3 * (1 - u) * u**2 * np.asarray(control_point_2)
            + u**3 * positions[-1]
        )
        if preserve_orientation_schedule:
            target_rotations = rotations
        else:
            key_rotations = Rotation.from_matrix(np.stack([rotations[0], rotations[-1]]))
            target_rotations = Slerp([0.0, 1.0], key_rotations)(u[:, 0]).as_matrix()
        solved, report = self.batch_ik(
            trajectory, frames, body_name, target_positions, target_rotations
        )
        result = trajectory.copy()
        for index, channel in enumerate(self.channels):
            result.channels[channel][frames] = (
                solved[:, index] - self.offsets[index]
            ) / self.scales[index]
        result.recompute_velocities()
        return result, report
