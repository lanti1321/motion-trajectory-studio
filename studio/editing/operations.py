from __future__ import annotations

import numpy as np

from studio.core.trajectory import TrajectoryData
from studio.editing.operation_data import max_safe_segment_duration
from studio.editing.basic_operations import (
    copy_segment,
    delete_range,
    hold,
    loop,
    move_segment,
    trim,
)
from studio.editing.phase_operations import (
    high_speed_phase_smooth,
    phase_aligned_copy_segment,
)
from studio.editing.retiming import (
    quintic_speed_time_warp,
    recommend_speed_ramp_duration,
    speed_ramp_segment,
    speed_segment,
)
from studio.editing.transitions import (
    concatenate,
    polynomial_transition,
    quintic_keyframe_patch,
    recommend_splice_transition_duration,
    recommend_transition_duration,
    sample_patch,
    splice_at,
)

__all__ = [
    "apply_operation",
    "concatenate",
    "copy_segment",
    "delete_range",
    "high_speed_phase_smooth",
    "hold",
    "loop",
    "move_segment",
    "max_safe_segment_duration",
    "phase_aligned_copy_segment",
    "polynomial_transition",
    "quintic_keyframe_patch",
    "quintic_speed_time_warp",
    "recommend_speed_ramp_duration",
    "recommend_splice_transition_duration",
    "recommend_transition_duration",
    "sample_patch",
    "speed_ramp_segment",
    "speed_segment",
    "splice_at",
    "trim",
]


def apply_operation(data: TrajectoryData, operation: dict[str, object]) -> TrajectoryData:
    kind = str(operation["type"])
    if kind in {"label_edit", "joint_group_edit"}:
        return data.copy()
    if kind == "trim":
        return trim(data, float(operation["start"]), float(operation["end"]), _string_list(operation.get("channels")))
    if kind == "delete":
        return delete_range(data, float(operation["start"]), float(operation["end"]), _string_list(operation.get("channels")))
    if kind == "hold":
        return hold(
            data,
            float(operation["at"]),
            float(operation["duration"]),
            _string_list(operation.get("channels")),
        )
    if kind == "loop":
        return loop(
            data,
            float(operation["start"]),
            float(operation["end"]),
            int(operation["repetitions"]) if "repetitions" in operation else None,
            float(operation["added_duration"]) if "added_duration" in operation else None,
            _string_list(operation.get("channels")),
        )
    if kind == "copy":
        copy_function = (
            phase_aligned_copy_segment
            if operation.get("phase_align")
            else copy_segment
        )
        return copy_function(
            data,
            float(operation["start"]),
            float(operation["end"]),
            float(operation["insert_at"]),
            _string_list(operation.get("channels")),
        )
    if kind == "move_segment":
        return move_segment(
            data,
            float(operation["start"]),
            float(operation["end"]),
            float(operation["destination"]),
            _string_list(operation.get("channels")),
        )
    if kind == "speed":
        return speed_segment(
            data,
            float(operation["start"]),
            float(operation["end"]),
            float(operation["factor"]),
            _string_list(operation.get("channels")),
        )
    if kind == "high_speed_smooth":
        result = high_speed_phase_smooth(
            data,
            float(operation["start"]),
            float(operation["end"]),
            _string_list(operation.get("channels")),
        )
        operation["_high_speed_phase_verified"] = True
        operation["detected_period_frames"] = int(
            result.metadata.get("last_high_speed_period_frames", 0)
        )
        return result
    if kind == "speed_ramp":
        result = speed_ramp_segment(
            data,
            float(operation["start"]),
            float(operation["end"]),
            float(operation["factor"]),
            _string_list(operation.get("channels")),
            float(operation["ramp_duration"])
            if operation.get("ramp_duration") is not None
            else None,
        )
        if operation.get("ramp_duration") is not None and bool(
            operation.get("changes_duration", True)
        ):
            first = data.nearest_frame(float(operation["start"]))
            last = data.nearest_frame(float(operation["end"]))
            phase, _duration = quintic_speed_time_warp(
                last - first,
                float(operation["factor"]),
                float(operation["ramp_duration"]),
                data.frequency.median_dt,
            )
            operation["_speed_source_phase"] = phase.tolist()
            operation["_speed_output_phase"] = np.linspace(
                0.0, 1.0, len(phase)
            ).tolist()
            operation["_time_warp_verified"] = True
        return result
    if kind == "splice":
        from studio.io.trajectory_io import load_trajectory

        second = load_trajectory(
            str(operation["path"]),
            float(operation["hz"]) if "hz" in operation else None,
        )
        return concatenate(
            data,
            second,
            float(operation.get("transition_duration", 0.0)),
            int(operation.get("order", 5)),
        )
    if kind == "splice_at":
        from studio.io.trajectory_io import load_trajectory

        second = load_trajectory(
            str(operation["path"]),
            float(operation["hz"]) if "hz" in operation else None,
        )
        return splice_at(
            data,
            second,
            float(operation["at"]),
            float(operation["replace_until"])
            if "replace_until" in operation
            else None,
            float(operation.get("transition_duration", 0.0)),
            int(operation.get("order", 5)),
        )
    if kind == "transition":
        result = polynomial_transition(
            data,
            float(operation["start"]),
            float(operation["end"]),
            _string_list(operation.get("channels")) or list(data.position_channels),
            int(operation.get("order", 5)),
            bool(operation.get("preserve_endpoint_derivatives", False)),
            _operation_segment_duration(operation),
            preserve_source_motion=bool(
                operation.get("preserve_source_motion", False)
            ),
        )
        if operation.get("preserve_source_motion"):
            operation["_residual_transition_verified"] = True
        return result
    if kind == "sample_patch":
        return sample_patch(
            data,
            float(operation["start"]),
            float(operation["end"]),
            {
                str(name): list(values)  # type: ignore[arg-type]
                for name, values in dict(operation["channel_values"]).items()  # type: ignore[arg-type]
            },
            _operation_segment_duration(operation),
        )
    if kind == "quintic_keyframe":
        return quintic_keyframe_patch(
            data,
            float(operation["start"]),
            float(operation["key_time"]),
            float(operation["end"]),
            {
                str(name): float(value)
                for name, value in dict(operation["targets"]).items()  # type: ignore[arg-type]
            },
            _operation_segment_duration(operation),
        )
    if kind == "insert_module":
        return _apply_legacy_clip_insert(data, operation)
    raise ValueError(f"unsupported operation type: {kind}")


def _apply_legacy_clip_insert(
    data: TrajectoryData, operation: dict[str, object]
) -> TrajectoryData:
    """Open old clip-insert operations that still point at a trajectory file."""
    path = operation.get("path")
    if not path:
        raise ValueError(
            "This open-source edition does not include action modules. "
            "Skill and event insert operations cannot be applied."
        )
    from studio.io.trajectory_io import load_trajectory

    clip = load_trajectory(str(path))
    mapping = {
        str(source): str(destination)
        for source, destination in dict(operation.get("motor_mapping") or {}).items()
    }
    if mapping:
        channels = {
            mapping.get(name, name): values
            for name, values in clip.channels.items()
        }
        names = [mapping.get(name, name) for name in clip.position_channels]
        clip = TrajectoryData(clip.times, channels, names, metadata=dict(clip.metadata))
    return splice_at(
        data,
        clip,
        float(operation["at"]),
        float(operation["replace_until"])
        if "replace_until" in operation
        else None,
        float(operation.get("transition_duration", 0.0)),
        int(operation.get("order", 5)),
    )


def _operation_segment_duration(
    operation: dict[str, object],
) -> float | None:
    value = operation.get("segment_duration", operation.get("output_duration"))
    return float(value) if value is not None else None


def _string_list(value: object) -> list[str] | None:
    if value is None:
        return None
    return [str(item) for item in value]  # type: ignore[arg-type]
