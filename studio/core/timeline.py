from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import numpy as np

from studio.core.project import TimelineLabel
from studio.core.trajectory import TrajectoryData
from studio.editing.operations import apply_operation


@dataclass(frozen=True)
class TimelineEdit:
    operation: dict[str, object]
    duration_before: float
    duration_after: float
    mapping_operation: dict[str, object]

    def map_time(self, seconds: float) -> float:
        return remap_time(
            seconds,
            self.mapping_operation,
            self.duration_before,
            self.duration_after,
        )

    def unmap_time(self, seconds: float) -> float:
        return unmap_time(
            seconds,
            self.mapping_operation,
            self.duration_before,
            self.duration_after,
        )


class TimelineEngine:
    """Non-destructive operation stack with deterministic undo/redo."""

    def __init__(
        self,
        source: TrajectoryData,
        operations: list[dict[str, object]] | None = None,
        labels: list[TimelineLabel] | None = None,
        validator: Callable[[TrajectoryData, dict[str, object]], None] | None = None,
    ) -> None:
        self.source = source
        self.operations = list(operations or [])
        self.labels = [replace(label) for label in labels or []]
        self._label_history: list[list[TimelineLabel]] = []
        self._redo: list[tuple[dict[str, object], list[TimelineLabel]]] = []
        self._cache: TrajectoryData | None = None
        self._validator = validator
        self.last_edit: TimelineEdit | None = None
        self.last_change_operation: dict[str, object] | None = None

    def render(self) -> TrajectoryData:
        if self._cache is None:
            result = self.source.copy()
            for operation in self.operations:
                result = apply_operation(result, operation)
            self._cache = result
        return self._cache

    def set_validator(
        self,
        validator: Callable[[TrajectoryData, dict[str, object]], None] | None,
    ) -> None:
        self._validator = validator

    def rebase_operation_paths(
        self, replacements: dict[str, str]
    ) -> None:
        def update(operation: dict[str, object]) -> None:
            if operation.get("path"):
                current = str(operation["path"])
                if current in replacements:
                    operation["path"] = replacements[current]

        for operation in self.operations:
            update(operation)
        for operation, _labels in self._redo:
            update(operation)
        self._cache = None

    def add_operation(self, operation: dict[str, object]) -> TimelineEdit:
        before = self.render()
        duration_before = before.duration
        stored_operation = dict(operation)
        stored_operation["_labels_before"] = [asdict(label) for label in self.labels]

        # Render and validate without mutating the operation stack.  Failed edits
        # must not consume an undo slot or alter labels.
        candidate = apply_operation(before.copy(), stored_operation)
        if candidate.frame_count < 1 or candidate.duration < 0:
            raise ValueError("operation produced an invalid trajectory")
        mapping_operation = _with_actual_timing(
            stored_operation, before, candidate
        )
        if self._validator is not None:
            self._validator(candidate, mapping_operation)
        labels_after = remap_labels(
            self.labels,
            mapping_operation,
            duration_before,
            candidate.duration,
        )
        if str(stored_operation["type"]) == "label_edit":
            labels_after = [
                TimelineLabel(**payload)
                for payload in stored_operation.get("labels_after", [])  # type: ignore[arg-type]
            ]
        self._label_history.append([replace(label) for label in self.labels])
        self.operations.append(stored_operation)
        self.labels = labels_after
        self._redo.clear()
        self._cache = candidate
        self.last_edit = TimelineEdit(
            dict(stored_operation),
            duration_before,
            candidate.duration,
            mapping_operation,
        )
        self.last_change_operation = stored_operation
        return self.last_edit

    def replace_labels(self, labels: list[TimelineLabel]) -> TimelineEdit:
        return self.add_operation(
            {
                "type": "label_edit",
                "labels_after": [asdict(label) for label in labels],
            }
        )

    def undo(self) -> bool:
        if not self.operations:
            return False
        edited = self.render()
        operation = self.operations.pop()
        self._redo.append((operation, [replace(label) for label in self.labels]))
        if self._label_history:
            self.labels = self._label_history.pop()
        elif "_labels_before" in operation:
            self.labels = [
                TimelineLabel(**payload)
                for payload in operation["_labels_before"]  # type: ignore[union-attr]
            ]
        self._cache = None
        original = self.render()
        mapping_operation = _with_actual_timing(
            operation, original, edited
        )
        self.last_edit = TimelineEdit(
            dict(operation),
            original.duration,
            edited.duration,
            mapping_operation,
        )
        self.last_change_operation = operation
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        before = self.render()
        operation, labels_after = self._redo[-1]
        after = apply_operation(before.copy(), operation)
        mapping_operation = _with_actual_timing(
            operation, before, after
        )
        if self._validator is not None:
            self._validator(after, mapping_operation)
        self._redo.pop()
        self._label_history.append([replace(label) for label in self.labels])
        self.operations.append(operation)
        self.labels = labels_after
        self._cache = after
        self.last_edit = TimelineEdit(
            dict(operation),
            before.duration,
            after.duration,
            mapping_operation,
        )
        self.last_change_operation = operation
        return True


def remap_labels(
    labels: list[TimelineLabel],
    operation: dict[str, object],
    duration_before: float,
    duration_after: float | None = None,
) -> list[TimelineLabel]:
    kind = str(operation["type"])
    result: list[TimelineLabel] = []
    if kind in {"trim", "delete"} and not bool(
        operation.get("changes_duration", True)
    ):
        return [replace(label) for label in labels]
    if kind == "trim":
        start = float(operation["start"])
        end = float(operation["end"])
        return [
            replace(label, time=label.time - start)
            for label in labels
            if start <= label.time <= end
        ]
    if kind == "delete":
        start = float(operation["start"])
        end = float(operation["end"])
        removed = float(
            operation.get("_actual_removed_duration", end - start)
        )
        for label in labels:
            if label.time < start:
                result.append(replace(label))
            elif label.time > end:
                result.append(replace(label, time=label.time - removed))
        return result
    if kind == "hold":
        at = float(operation["at"])
        delta = float(
            operation.get(
                "_actual_added_duration", operation["duration"]
            )
        )
        return [replace(label, time=label.time + delta if label.time > at else label.time) for label in labels]
    if kind == "loop":
        start = float(operation["start"])
        end = float(operation["end"])
        period = float(
            operation.get("_actual_source_span", end - start)
        )
        if "repetitions" in operation:
            repetitions = int(operation["repetitions"])
            delta = float(
                operation.get("_actual_added_duration", period * repetitions)
            )
        else:
            repetitions = max(
                1, int(round(float(operation["added_duration"]) / period))
            )
            delta = float(
                operation.get(
                    "_actual_added_duration",
                    operation["added_duration"],
                )
            )
        shifted = [
            replace(label, time=label.time + delta if label.time > start else label.time)
            for label in labels
        ]
        used_ids = {label.id for label in shifted}
        duplicates: list[TimelineLabel] = []
        for repetition in range(repetitions):
            for label in labels:
                if not start < label.time <= end:
                    continue
                duplicate_id = _unique_label_id(
                    f"{label.id}_loop{repetition + 1}", used_ids
                )
                duplicates.append(
                    replace(
                        label,
                        id=duplicate_id,
                        time=start
                        + repetition * period
                        + (label.time - start),
                    )
                )
        return sorted(shifted + duplicates, key=lambda item: item.time)
    if kind == "copy":
        start = float(operation["start"])
        end = float(operation["end"])
        insert_at = float(operation["insert_at"])
        span = float(
            operation.get("_actual_added_duration", end - start)
        )
        shifted = [
            replace(label, time=label.time + span if label.time > insert_at else label.time)
            for label in labels
        ]
        used_ids = {label.id for label in shifted}
        duplicates = []
        for label in labels:
            if not start <= label.time <= end:
                continue
            duplicates.append(
                replace(
                    label,
                    id=_unique_label_id(f"{label.id}_copy", used_ids),
                    time=insert_at + (label.time - start),
                )
            )
        return sorted(shifted + duplicates, key=lambda item: item.time)
    if kind in {"speed", "speed_ramp"}:
        if kind == "speed_ramp" and not bool(
            operation.get("changes_duration", True)
        ):
            return [replace(label) for label in labels]
        start = float(operation["start"])
        end = float(operation["end"])
        factor = float(operation["factor"])
        new_span = float(
            operation.get("_actual_new_span", (end - start) / factor)
        )
        shift = new_span - (end - start)
        for label in labels:
            if label.time <= start:
                result.append(replace(label))
            elif label.time <= end:
                old_span = end - start
                old_phase = (label.time - start) / old_span
                if (
                    kind == "speed_ramp"
                    and operation.get("_speed_source_phase")
                    and operation.get("_speed_output_phase")
                ):
                    output_phase = float(
                        np.interp(
                            old_phase,
                            np.asarray(
                                operation["_speed_source_phase"],
                                dtype=float,
                            ),
                            np.asarray(
                                operation["_speed_output_phase"],
                                dtype=float,
                            ),
                        )
                    )
                else:
                    output_phase = old_phase
                result.append(
                    replace(
                        label,
                        time=start + output_phase * new_span,
                    )
                )
            else:
                result.append(replace(label, time=label.time + shift))
        return result
    if kind == "splice_at":
        start = float(operation["at"])
        end = float(operation.get("replace_until", start))
        inserted_duration = float(operation.get("inserted_duration", 0.0))
        transition_duration = float(operation.get("transition_duration", 0.0))
        inserted_span = float(
            operation.get(
                "_actual_inserted_span",
                inserted_duration + 2 * transition_duration,
            )
        )
        shift = inserted_span - (end - start)
        for label in labels:
            if label.time < start:
                result.append(replace(label))
            elif label.time == start:
                result.append(replace(label))
            elif end > start and label.time < end:
                continue
            elif end > start and label.time == end:
                result.append(replace(label, time=start + inserted_span))
            else:
                result.append(replace(label, time=label.time + shift))
        return sorted(result, key=lambda item: item.time)
    segment_duration = operation.get(
        "segment_duration", operation.get("output_duration")
    )
    if (
        kind in {"transition", "sample_patch", "quintic_keyframe"}
        and segment_duration is not None
    ):
        start = float(operation["start"])
        end = float(operation["end"])
        old_span = end - start
        new_span = float(
            operation.get("_actual_new_span", segment_duration)
        )
        if old_span <= 0:
            return [replace(label) for label in labels]
        scale = new_span / old_span
        shift = new_span - old_span
        for label in labels:
            if label.time <= start:
                result.append(replace(label))
            elif label.time <= end:
                result.append(
                    replace(
                        label,
                        time=start + (label.time - start) * scale,
                    )
                )
            else:
                result.append(replace(label, time=label.time + shift))
        return result
    return [replace(label) for label in labels]


def remap_time(
    seconds: float,
    operation: dict[str, object],
    duration_before: float,
    duration_after: float | None = None,
) -> float:
    marker = TimelineLabel("__cursor__", float(seconds))
    mapped = remap_labels(
        [marker], operation, duration_before, duration_after
    )
    if mapped:
        return max(0.0, mapped[0].time)
    kind = str(operation["type"])
    if kind == "trim":
        return 0.0 if seconds < float(operation["start"]) else (
            duration_after or 0.0
        )
    if kind == "delete":
        return float(operation["start"])
    if kind == "splice_at":
        return float(operation["at"])
    return min(float(seconds), duration_after or duration_before)


def unmap_time(
    seconds: float,
    operation: dict[str, object],
    duration_before: float,
    duration_after: float,
) -> float:
    kind = str(operation["type"])
    value = float(seconds)
    if kind == "trim":
        return min(
            duration_before, value + float(operation["start"])
        )
    if kind == "delete":
        start = float(operation["start"])
        removed = float(operation.get("_actual_removed_duration", 0.0))
        return value if value <= start else value + removed
    if kind == "hold":
        at = float(operation["at"])
        delta = float(operation.get("_actual_added_duration", 0.0))
        if value <= at:
            return value
        if value <= at + delta:
            return at
        return value - delta
    if kind in {"speed", "speed_ramp", "transition", "sample_patch", "quintic_keyframe"}:
        if (
            kind == "speed_ramp"
            and not bool(operation.get("changes_duration", True))
        ):
            return value
        start = float(operation["start"])
        old_span = float(operation.get("_actual_source_span", 0.0))
        new_span = float(operation.get("_actual_new_span", old_span))
        if value <= start:
            return value
        if value <= start + new_span and new_span > 0:
            output_phase = (value - start) / new_span
            if (
                kind == "speed_ramp"
                and operation.get("_speed_source_phase")
                and operation.get("_speed_output_phase")
            ):
                source_phase = float(
                    np.interp(
                        output_phase,
                        np.asarray(
                            operation["_speed_output_phase"],
                            dtype=float,
                        ),
                        np.asarray(
                            operation["_speed_source_phase"],
                            dtype=float,
                        ),
                    )
                )
            else:
                source_phase = output_phase
            return start + source_phase * old_span
        return value - (new_span - old_span)
    if kind in {"loop", "copy"}:
        point = float(
            operation["start"]
            if kind == "loop"
            else operation["insert_at"]
        )
        delta = float(operation.get("_actual_added_duration", 0.0))
        if value <= point:
            return value
        if value <= point + delta:
            if kind == "copy":
                return float(operation["start"]) + (value - point)
            period = float(operation.get("_actual_source_span", 0.0))
            return point + (
                (value - point) % period if period > 0 else 0.0
            )
        return value - delta
    if kind == "splice_at":
        start = float(operation["at"])
        inserted = float(operation.get("_actual_inserted_span", 0.0))
        replaced_end = float(operation.get("replace_until", start))
        shift = inserted - (replaced_end - start)
        if value <= start:
            return value
        if value <= start + inserted:
            return start
        return value - shift
    return min(value, duration_before)


def _with_actual_timing(
    operation: dict[str, object],
    before: TrajectoryData,
    after: TrajectoryData,
) -> dict[str, object]:
    mapped = dict(operation)
    kind = str(operation["type"])
    delta = after.duration - before.duration
    mapped["_actual_added_duration"] = max(0.0, delta)
    mapped["_actual_removed_duration"] = max(0.0, -delta)
    if "start" in operation and "end" in operation:
        first = before.nearest_frame(float(operation["start"]))
        last = before.nearest_frame(float(operation["end"]))
        start = float(before.times[first])
        end = float(before.times[last])
        mapped["start"] = start
        mapped["end"] = end
        mapped["_actual_source_span"] = end - start
        mapped["_actual_new_span"] = max(0.0, end - start + delta)
        if (
            kind == "speed_ramp"
            and bool(operation.get("changes_duration", True))
        ):
            old_count = last - first
            factor = float(operation["factor"])
            if operation.get("ramp_duration") is not None:
                return mapped
            new_count = max(2, int(round(old_count / factor)))
            output_phase = np.linspace(0.0, 1.0, new_count + 1)
            rate = (
                1.0
                + 2.0
                * (factor - 1.0)
                * np.sin(np.pi * output_phase) ** 2
            )
            source_phase = np.r_[
                0.0,
                np.cumsum((rate[:-1] + rate[1:]) * 0.5),
            ]
            source_phase /= source_phase[-1]
            if new_count >= 6:
                boundary_indices = np.asarray(
                    [0, 1, 2, new_count - 2, new_count - 1, new_count]
                )
                boundary_values = np.asarray(
                    [0, 1, 2, old_count - 2, old_count - 1, old_count],
                    dtype=float,
                ) / old_count
                source_phase[boundary_indices] = boundary_values
            mapped["_speed_source_phase"] = source_phase.tolist()
            mapped["_speed_output_phase"] = output_phase.tolist()
    if kind == "hold":
        frame = before.nearest_frame(float(operation["at"]))
        mapped["at"] = float(before.times[frame])
    if kind == "copy":
        frame = before.nearest_frame(float(operation["insert_at"]))
        mapped["insert_at"] = float(before.times[frame])
    if kind == "move_segment":
        frame = before.nearest_frame(float(operation["destination"]))
        mapped["destination"] = float(before.times[frame])
    if kind == "splice_at":
        first = before.nearest_frame(float(operation["at"]))
        start = float(before.times[first])
        end = start
        if operation.get("replace_until") is not None:
            last = before.nearest_frame(float(operation["replace_until"]))
            end = float(before.times[last])
        mapped["at"] = start
        mapped["replace_until"] = end
        mapped["_actual_inserted_span"] = max(0.0, end - start + delta)
    return mapped


def _unique_label_id(base: str, used_ids: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate
