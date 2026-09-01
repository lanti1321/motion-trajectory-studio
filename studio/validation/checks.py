from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.models.adapter import MuJoCoModelAdapter


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    channel: str | None = None
    frame: int | None = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def validate_trajectory(
    trajectory: TrajectoryData,
    position_jump_limits: dict[str, float] | None = None,
    velocity_limits: dict[str, float] | None = None,
    acceleration_limits: dict[str, float] | None = None,
) -> ValidationReport:
    report = ValidationReport()
    frequency = trajectory.frequency
    report.metrics.update(
        {
            "frame_count": float(trajectory.frame_count),
            "duration": trajectory.duration,
            "frequency_hz": frequency.hz,
            "frequency_max_jitter": frequency.max_jitter,
        }
    )
    if frequency.irregular:
        report.issues.append(
            ValidationIssue(
                "warning",
                "irregular_frequency",
                f"time jitter is {frequency.max_jitter:.6g}s; resample before hardware export",
            )
        )
    for name in trajectory.position_channels:
        values = trajectory.channels[name]
        if len(values) < 2:
            continue
        steps = np.abs(np.diff(values))
        max_step_frame = int(np.argmax(steps))
        report.metrics[f"{name}.max_step"] = float(steps[max_step_frame])
        if position_jump_limits and name in position_jump_limits:
            if steps[max_step_frame] > position_jump_limits[name]:
                report.issues.append(
                    ValidationIssue(
                        "error",
                        "position_jump",
                        f"{name} step {steps[max_step_frame]:.6g} exceeds limit",
                        name,
                        max_step_frame + 1,
                    )
                )
        velocity = np.gradient(values, trajectory.times, edge_order=min(2, len(values) - 1))
        max_velocity_frame = int(np.argmax(np.abs(velocity)))
        report.metrics[f"{name}.max_velocity"] = float(abs(velocity[max_velocity_frame]))
        if velocity_limits and name in velocity_limits:
            if abs(velocity[max_velocity_frame]) > velocity_limits[name]:
                report.issues.append(
                    ValidationIssue(
                        "error",
                        "velocity_limit",
                        f"{name} velocity exceeds configured limit",
                        name,
                        max_velocity_frame,
                    )
                )
        if len(values) >= 3:
            acceleration = np.gradient(velocity, trajectory.times, edge_order=2)
            max_acceleration_frame = int(np.argmax(np.abs(acceleration)))
            report.metrics[f"{name}.max_acceleration"] = float(
                abs(acceleration[max_acceleration_frame])
            )
            if acceleration_limits and name in acceleration_limits:
                if abs(acceleration[max_acceleration_frame]) > acceleration_limits[name]:
                    report.issues.append(
                        ValidationIssue(
                            "error",
                            "acceleration_limit",
                            f"{name} acceleration exceeds configured limit",
                            name,
                            max_acceleration_frame,
                        )
                    )
    for start, end in detect_stalls(trajectory):
        report.issues.append(
            ValidationIssue(
                "warning",
                "possible_stall",
                f"all selected joints remain nearly stationary from {start:.3f}s to {end:.3f}s",
                frame=trajectory.nearest_frame(start),
            )
        )
    return report


def validate_velocity_channels(
    trajectory: TrajectoryData,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-3,
) -> ValidationReport:
    report = ValidationReport()
    for position_name, velocity_name in trajectory.velocity_channels.items():
        if (
            position_name not in trajectory.channels
            or velocity_name not in trajectory.channels
        ):
            continue
        if trajectory.frame_count < 2:
            expected = np.zeros(trajectory.frame_count)
        else:
            expected = np.gradient(
                trajectory.channels[position_name],
                trajectory.times,
                edge_order=min(2, trajectory.frame_count - 1),
            )
        actual = trajectory.channels[velocity_name]
        error = np.abs(actual - expected)
        allowed = absolute_tolerance + relative_tolerance * np.maximum(
            np.abs(expected), np.abs(actual)
        )
        violating = np.flatnonzero(error > allowed)
        report.metrics[f"{position_name}.velocity_channel_error"] = float(
            np.max(error, initial=0.0)
        )
        if len(violating):
            report.issues.append(
                ValidationIssue(
                    "error",
                    "stale_velocity_channel",
                    f"{velocity_name} does not match the gradient of {position_name}",
                    velocity_name,
                    int(violating[0]),
                )
            )
    return report


def validate_operation_continuity(
    trajectory: TrajectoryData,
    operation: dict[str, object],
    channels: list[str] | None = None,
) -> ValidationReport:
    """Operation-aware boundary checks aligned with edit-time rules."""
    kind = str(operation.get("type", ""))
    if kind == "high_speed_smooth" and bool(
        operation.get("_high_speed_phase_verified")
    ):
        return ValidationReport()
    if bool(operation.get("_ik_constraint_verified")):
        return ValidationReport()
    if kind == "transition" and bool(
        operation.get("_residual_transition_verified")
    ):
        return ValidationReport()
    if kind == "speed_ramp" and bool(operation.get("_time_warp_verified")):
        source_phase = np.asarray(
            operation.get("_speed_source_phase", []), dtype=float
        )
        output_phase = np.asarray(
            operation.get("_speed_output_phase", []), dtype=float
        )
        if (
            len(source_phase) >= 7
            and len(source_phase) == len(output_phase)
            and np.isfinite(source_phase).all()
            and np.all(np.diff(source_phase) > 0)
            and source_phase[0] == 0.0
            and source_phase[-1] == 1.0
        ):
            return ValidationReport()
        return ValidationReport(
            [
                ValidationIssue(
                    "error",
                    "invalid_time_warp",
                    "变速时间映射不是有限、严格单调且端点完整的映射",
                )
            ]
        )
    if channels is None:
        targets = operation.get("targets")
        if isinstance(targets, dict):
            channels = [str(value) for value in targets]
        elif isinstance(operation.get("channel_values"), dict):
            channels = [
                str(value)
                for value in dict(operation["channel_values"])
            ]
        elif operation.get("channels"):
            channels = [str(value) for value in operation["channels"]]  # type: ignore[union-attr]
        else:
            channels = list(trajectory.position_channels)
    selected = channels
    if kind == "splice_at":
        start = float(operation["at"])
        new_span = float(operation.get("_actual_inserted_span", 0.0))
        boundary_times = [start + new_span]
    elif kind == "move_segment":
        start = float(operation["start"])
        span = float(operation["end"]) - start
        destination = float(operation["destination"])
        boundary_times = [start, float(operation["end"]), destination, destination + span]
    elif kind in {"transition", "sample_patch", "quintic_keyframe"}:
        start = float(operation["start"])
        new_span = float(
            operation.get(
                "_actual_new_span",
                float(operation["end"]) - start,
            )
        )
        if _operation_has_replacement_duration(operation):
            boundary_times = [start + new_span]
        else:
            boundary_times = [start, start + new_span]
    elif kind == "speed_ramp" and not bool(
        operation.get("changes_duration", True)
    ):
        start = float(operation["start"])
        boundary_times = [start]
    else:
        if kind in {"transition", "sample_patch", "quintic_keyframe", "speed_ramp"}:
            start = float(operation["start"])
            new_span = float(
                operation.get(
                    "_actual_new_span",
                    float(operation["end"]) - start,
                )
            )
            explicit_boundaries = operation.get("_continuity_boundaries")
            boundary_times = (
                [float(value) for value in explicit_boundaries]  # type: ignore[union-attr]
                if explicit_boundaries
                else [start, start + new_span]
            )
        else:
            return ValidationReport()

    check_c2 = (
        not _operation_has_replacement_duration(operation)
        and kind not in {"splice_at", "sample_patch", "quintic_keyframe", "move_segment"}
    )
    report = validate_boundary_continuity(
        trajectory, boundary_times, selected
    )
    filtered: list[ValidationIssue] = []
    for issue in report.issues:
        if issue.code == "continuity_context":
            continue
        if issue.code == "c2_discontinuity" and not check_c2:
            continue
        filtered.append(issue)
    return ValidationReport(filtered, dict(report.metrics))


def _operation_has_replacement_duration(operation: dict[str, object]) -> bool:
    if operation.get("type") == "transition":
        return operation.get("segment_duration") is not None
    if operation.get("type") in {"sample_patch", "quintic_keyframe"}:
        return operation.get("segment_duration") is not None or (
            operation.get("output_duration") is not None
        )
    return False


def validate_rendered_operations(
    source: TrajectoryData,
    operations: list[dict[str, object]],
) -> ValidationReport:
    """Replay operations and validate strict-continuity edits with actual timing."""
    from studio.core.timeline import _with_actual_timing
    from studio.editing.operations import apply_operation

    report = ValidationReport()
    stage = source.copy()
    for operation in operations:
        candidate = apply_operation(stage, operation)
        enriched = _with_actual_timing(operation, stage, candidate)
        if bool(operation.get("strict_continuity")):
            channel_values = operation.get("channels")
            channels = (
                [str(value) for value in channel_values]  # type: ignore[union-attr]
                if channel_values
                else None
            )
            continuity_report = validate_operation_continuity(
                candidate, enriched, channels
            )
            report.issues.extend(continuity_report.issues)
            report.metrics.update(continuity_report.metrics)
        stage = candidate
    return report


def filter_new_joint_limit_issues(
    before: ValidationReport,
    after: ValidationReport,
) -> list[ValidationIssue]:
    existing = {
        issue.channel
        for issue in before.issues
        if issue.code == "joint_limit"
    }
    return [
        issue
        for issue in after.issues
        if issue.code == "joint_limit" and issue.channel not in existing
    ]


def validate_boundary_continuity(
    trajectory: TrajectoryData,
    boundary_times: list[float],
    channels: list[str] | None = None,
) -> ValidationReport:
    report = ValidationReport()
    selected = channels or trajectory.position_channels
    for boundary_time in boundary_times:
        frame = trajectory.nearest_frame(boundary_time)
        if frame < 2 or frame > trajectory.frame_count - 3:
            report.issues.append(
                ValidationIssue(
                    "error",
                    "continuity_context",
                    "C1/C2 validation needs two frames outside each boundary",
                    frame=frame,
                )
            )
            continue
        metrics = continuity_metrics(trajectory, boundary_time, selected)
        for name, values in metrics.items():
            samples = trajectory.channels[name][frame - 2 : frame + 3]
            times = trajectory.times[frame - 2 : frame + 3]
            velocity = np.gradient(samples, times, edge_order=2)
            acceleration = np.gradient(velocity, times, edge_order=2)
            velocity_scale = max(1e-4, float(np.max(np.abs(velocity))))
            acceleration_scale = max(
                1e-3, float(np.max(np.abs(acceleration)))
            )
            report.metrics[
                f"{name}@{boundary_time:.9g}.velocity_jump"
            ] = abs(values["velocity_jump"])
            report.metrics[
                f"{name}@{boundary_time:.9g}.acceleration_jump"
            ] = abs(values["acceleration_jump"])
            if abs(values["velocity_jump"]) > 0.25 * velocity_scale:
                report.issues.append(
                    ValidationIssue(
                        "error",
                        "c1_discontinuity",
                        f"{name} has a C1 discontinuity at {boundary_time:.6g}s",
                        name,
                        frame,
                    )
                )
            if abs(values["acceleration_jump"]) > 0.75 * acceleration_scale:
                report.issues.append(
                    ValidationIssue(
                        "error",
                        "c2_discontinuity",
                        f"{name} has a C2 discontinuity at {boundary_time:.6g}s",
                        name,
                        frame,
                    )
                )
    return report


def detect_stalls(
    trajectory: TrajectoryData,
    speed_threshold: float = 1e-4,
    minimum_duration: float = 0.25,
    channels: list[str] | None = None,
) -> list[tuple[float, float]]:
    selected = channels or trajectory.position_channels
    if not selected or trajectory.frame_count < 2:
        return []
    matrix = trajectory.matrix(selected)
    velocity = np.gradient(matrix, trajectory.times, axis=0, edge_order=min(2, trajectory.frame_count - 1))
    stationary = np.max(np.abs(velocity), axis=1) <= speed_threshold
    edges = np.diff(np.r_[False, stationary, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [
        (float(trajectory.times[start]), float(trajectory.times[end]))
        for start, end in zip(starts, ends)
        if trajectory.times[end] - trajectory.times[start] >= minimum_duration
    ]


def continuity_metrics(
    trajectory: TrajectoryData,
    boundary_time: float,
    channels: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    frame = trajectory.nearest_frame(boundary_time)
    if frame < 2 or frame > trajectory.frame_count - 3:
        raise ValueError("continuity boundary needs at least two frames on each side")
    selected = channels or trajectory.position_channels
    dt_left = trajectory.times[frame] - trajectory.times[frame - 1]
    dt_right = trajectory.times[frame + 1] - trajectory.times[frame]
    result: dict[str, dict[str, float]] = {}
    for name in selected:
        values = trajectory.channels[name]
        velocity_left = (values[frame] - values[frame - 1]) / dt_left
        velocity_right = (values[frame + 1] - values[frame]) / dt_right
        previous_velocity = (
            values[frame - 1] - values[frame - 2]
        ) / (trajectory.times[frame - 1] - trajectory.times[frame - 2])
        next_velocity = (
            values[frame + 2] - values[frame + 1]
        ) / (trajectory.times[frame + 2] - trajectory.times[frame + 1])
        acceleration_left = 2 * (velocity_left - previous_velocity) / (
            dt_left + trajectory.times[frame - 1] - trajectory.times[frame - 2]
        )
        acceleration_right = 2 * (next_velocity - velocity_right) / (
            dt_right + trajectory.times[frame + 2] - trajectory.times[frame + 1]
        )
        result[name] = {
            "position_jump": 0.0,
            "velocity_jump": float(velocity_right - velocity_left),
            "acceleration_jump": float(acceleration_right - acceleration_left),
        }
    return result


def validate_model_mapping(
    trajectory: TrajectoryData,
    adapter: MuJoCoModelAdapter,
    mapping: dict[str, str],
    transforms: dict[str, dict[str, float]] | None = None,
) -> ValidationReport:
    report = ValidationReport()
    joint_by_name = {joint.name: joint for joint in adapter.info.joints}
    used_joints: set[str] = set()
    for channel in trajectory.position_channels:
        joint_name = mapping.get(channel)
        if not joint_name:
            report.issues.append(
                ValidationIssue("error", "unmapped_channel", f"{channel} is not mapped", channel)
            )
            continue
        joint = joint_by_name.get(joint_name)
        if joint is None:
            report.issues.append(
                ValidationIssue("error", "missing_joint", f"model has no joint {joint_name}", channel)
            )
            continue
        if joint_name in used_joints:
            report.issues.append(
                ValidationIssue(
                    "error", "duplicate_mapping", f"multiple channels map to {joint_name}", channel
                )
            )
        used_joints.add(joint_name)
        if joint.qpos_width != 1:
            report.issues.append(
                ValidationIssue(
                    "error",
                    "unsupported_joint",
                    f"{joint_name} is a {joint.joint_type} joint requiring {joint.qpos_width} values",
                    channel,
                )
            )
        if joint.limited:
            transform = (transforms or {}).get(channel, {})
            values = (
                trajectory.channels[channel] * float(transform.get("scale", 1.0))
                + float(transform.get("offset", 0.0))
            )
            if "minimum" in transform or "maximum" in transform:
                values = np.clip(
                    values,
                    float(transform.get("minimum", -np.inf)),
                    float(transform.get("maximum", np.inf)),
                )
            violating = np.flatnonzero((values < joint.lower) | (values > joint.upper))
            if len(violating):
                report.issues.append(
                    ValidationIssue(
                        "error",
                        "joint_limit",
                        f"{channel} exceeds {joint_name} range",
                        channel,
                        int(violating[0]),
                    )
                )
    return report


def sample_collisions(
    trajectory: TrajectoryData,
    adapter: MuJoCoModelAdapter,
    mapping: dict[str, str],
    max_samples: int = 1000,
    progress: Callable[[int, int], bool] | None = None,
    transforms: dict[str, dict[str, float]] | None = None,
) -> ValidationReport:
    report = ValidationReport()
    step = max(1, int(np.ceil(trajectory.frame_count / max_samples)))
    contact_frames = 0
    first_contact: tuple[int, str, str] | None = None
    frames = list(range(0, trajectory.frame_count, step))
    for sample_index, frame in enumerate(frames):
        if progress and sample_index % 8 == 0 and not progress(
            sample_index, len(frames)
        ):
            raise InterruptedError("collision sampling canceled")
        positions: dict[str, float] = {}
        for channel, joint in mapping.items():
            if channel not in trajectory.channels:
                continue
            transform = (transforms or {}).get(channel, {})
            value = float(
                trajectory.channels[channel][frame]
                * transform.get("scale", 1.0)
                + transform.get("offset", 0.0)
            )
            value = max(
                float(transform.get("minimum", -np.inf)),
                min(
                    float(transform.get("maximum", np.inf)),
                    value,
                ),
            )
            positions[joint] = value
        adapter.apply_positions(positions)
        if adapter.data.ncon:
            contact_frames += 1
            if first_contact is None:
                contact = adapter.data.contact[0]
                geom1_id = int(contact.geom1)
                geom2_id = int(contact.geom2)
                geom1 = mujoco.mj_id2name(
                    adapter.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id
                ) or f"geom#{geom1_id}"
                geom2 = mujoco.mj_id2name(
                    adapter.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id
                ) or f"geom#{geom2_id}"
                first_contact = (frame, geom1, geom2)
    if first_contact is not None:
        frame, geom1, geom2 = first_contact
        report.issues.append(
            ValidationIssue(
                "warning",
                "model_contact",
                "sampled model contact: first at "
                f"{trajectory.times[frame]:.3f}s (frame {frame}), "
                f"{geom1} ↔ {geom2}; contact in {contact_frames}/{len(frames)} "
                "sampled frames",
                frame=frame,
            )
        )
    if progress and not progress(len(frames), len(frames)):
        raise InterruptedError("collision sampling canceled")
    report.metrics["sampled_contact_frames"] = float(contact_frames)
    return report
