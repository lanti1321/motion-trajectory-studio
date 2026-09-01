from __future__ import annotations

import numpy as np
import pytest

from studio.core.project import TimelineLabel
from studio.core.timeline import TimelineEngine, remap_labels
from studio.core.trajectory import TrajectoryData
from studio.editing.operations import (
    concatenate,
    copy_segment,
    delete_range,
    hold,
    high_speed_phase_smooth,
    loop,
    move_segment,
    polynomial_transition,
    phase_aligned_copy_segment,
    quintic_keyframe_patch,
    sample_patch,
    speed_ramp_segment,
    speed_segment,
    splice_at,
    trim,
)


@pytest.fixture
def trajectory() -> TrajectoryData:
    times = np.arange(1001) / 100
    return TrajectoryData(
        times,
        {
            "q0": np.sin(times),
            "q1": np.cos(times),
            "sensor": times.copy(),
        },
        ["q0", "q1"],
    )


def test_quintic_transition_is_minimum_jerk_without_overshoot(trajectory: TrajectoryData) -> None:
    result = polynomial_transition(trajectory, 2, 4, ["q0"], order=5)
    first = result.nearest_frame(2)
    last = result.nearest_frame(4)
    assert result.channels["q0"][first] == pytest.approx(trajectory.channels["q0"][first])
    assert result.channels["q0"][last] == pytest.approx(trajectory.channels["q0"][last])
    segment = result.channels["q0"][first : last + 1]
    lower, upper = sorted(
        (trajectory.channels["q0"][first], trajectory.channels["q0"][last])
    )
    assert segment.min() >= lower - 1e-12
    assert segment.max() <= upper + 1e-12
    assert np.all(np.diff(segment) <= 1e-12)
    np.testing.assert_array_equal(result.channels["q1"], trajectory.channels["q1"])


def test_quintic_residual_mode_preserves_high_frequency_motion() -> None:
    times = np.arange(4001) / 1000.0
    values = 0.2 * times + 0.04 * np.sin(2.0 * np.pi * 35.0 * times)
    data = TrajectoryData(times, {"q0": values}, ["q0"])
    preserved = polynomial_transition(
        data, 0.5, 3.5, ["q0"], order=5,
        preserve_endpoint_derivatives=True,
        preserve_source_motion=True,
    )
    replaced = polynomial_transition(
        data, 0.5, 3.5, ["q0"], order=5,
        preserve_endpoint_derivatives=True,
    )
    first = data.nearest_frame(1.0)
    last = data.nearest_frame(3.0)

    def high_frequency_energy(samples: np.ndarray) -> float:
        return float(np.std(np.diff(samples, n=2)))

    source_energy = high_frequency_energy(data.channels["q0"][first:last])
    preserved_energy = high_frequency_energy(
        preserved.channels["q0"][first:last]
    )
    replaced_energy = high_frequency_energy(
        replaced.channels["q0"][first:last]
    )
    assert preserved_energy == pytest.approx(source_energy, rel=0.05)
    assert replaced_energy < source_energy * 0.1


def test_resized_quintic_keyframe_preserves_c2_boundaries(
    trajectory: TrajectoryData,
) -> None:
    from studio.validation.checks import validate_operation_continuity

    operation = {
        "type": "quintic_keyframe",
        "start": 2.0,
        "key_time": 3.0,
        "end": 4.0,
        "targets": {"q0": 0.25},
        "segment_duration": 3.0,
        "strict_continuity": True,
    }
    result = quintic_keyframe_patch(
        trajectory, 2.0, 3.0, 4.0, {"q0": 0.25}, 3.0
    )
    operation["_actual_new_span"] = 3.0
    report = validate_operation_continuity(result, operation)
    assert not [issue for issue in report.issues if issue.code == "c1_discontinuity"]
    assert not [issue for issue in report.issues if issue.code == "c2_discontinuity"]
    resized_key = result.nearest_frame(3.5)
    assert result.channels["q0"][resized_key] == pytest.approx(0.25)


def test_speed_segment_shortens_duration(trajectory: TrajectoryData) -> None:
    result = speed_segment(trajectory, 2, 6, 2.0)
    assert result.duration == pytest.approx(8.0, abs=0.02)
    assert result.channels["q0"][result.nearest_frame(2)] == pytest.approx(
        trajectory.channels["q0"][trajectory.nearest_frame(2)]
    )


def test_loop_and_hold_extend_duration(trajectory: TrajectoryData) -> None:
    looped = loop(trajectory, 1, 2, repetitions=2)
    assert looped.duration == pytest.approx(12, abs=0.02)
    held = hold(trajectory, 3, 1)
    assert held.duration == pytest.approx(11, abs=0.02)


def test_partial_delete_advances_only_selected_lane() -> None:
    times = np.arange(101) / 10.0
    left = times.copy()
    right = 2.0 * times
    data = TrajectoryData(times, {"left": left, "right": right}, ["left", "right"])
    result = delete_range(data, 2.0, 4.0, ["left"])
    assert result.duration == pytest.approx(data.duration)
    np.testing.assert_array_equal(result.channels["right"], right)
    assert result.channels["left"][20] == pytest.approx(left[41])
    assert np.ptp(result.channels["left"][-21:]) == 0.0


def test_partial_hold_delays_only_selected_lane() -> None:
    times = np.arange(51) / 10.0
    left = times.copy()
    right = 2.0 * times
    data = TrajectoryData(times, {"left": left, "right": right}, ["left", "right"])
    result = hold(data, 2.0, 1.0, ["left"])
    assert result.duration == pytest.approx(6.0)
    assert np.ptp(result.channels["left"][20:31]) == 0.0
    np.testing.assert_array_equal(result.channels["right"][:51], right)
    assert np.ptp(result.channels["right"][51:]) == 0.0


def test_partial_loop_and_copy_leave_other_lane_timing_intact() -> None:
    times = np.arange(101) / 10.0
    data = TrajectoryData(
        times, {"left": np.sin(times), "right": np.cos(times)}, ["left", "right"]
    )
    looped = loop(data, 1.0, 2.0, repetitions=1, channels=["left"])
    copied = copy_segment(data, 1.0, 2.0, 5.0, ["left"])
    for result in (looped, copied):
        np.testing.assert_array_equal(
            result.channels["right"][: data.frame_count], data.channels["right"]
        )
        assert np.ptp(result.channels["right"][data.frame_count - 1 :]) == 0.0


def test_partial_trim_keeps_duration_and_holds_selected_outside_range() -> None:
    times = np.arange(101) / 10.0
    data = TrajectoryData(
        times, {"left": times.copy(), "right": np.sin(times)}, ["left", "right"]
    )
    result = trim(data, 2.0, 6.0, ["left"])
    assert result.duration == data.duration
    assert np.ptp(result.channels["left"][:21]) == 0.0
    assert np.ptp(result.channels["left"][60:]) == 0.0
    np.testing.assert_array_equal(result.channels["right"], data.channels["right"])


def test_copy_and_splice(trajectory: TrajectoryData) -> None:
    copied = copy_segment(trajectory, 1, 2, 5)
    assert copied.duration == pytest.approx(11, abs=0.02)
    second = TrajectoryData(
        np.arange(101) / 100,
        {name: values[:101] for name, values in trajectory.channels.items()},
        trajectory.position_channels,
    )
    spliced = concatenate(trajectory, second, transition_duration=0.2)
    assert spliced.duration == pytest.approx(11.21, abs=0.02)


def test_move_segment_moves_only_selected_joint_into_stationary_space() -> None:
    times = np.arange(101) / 10.0
    left = np.zeros(101)
    left[11:20] = np.linspace(0.0, 0.8, 9)
    left[20:31] = 0.8
    left[31:40] = np.linspace(0.8, 0.0, 9)
    right = np.sin(times)
    data = TrajectoryData(times, {"left": left, "right": right}, ["left", "right"])
    result = move_segment(data, 1.0, 4.0, 6.0, ["left"])
    first, last, destination = 10, 40, 60
    np.testing.assert_allclose(
        result.channels["left"][destination : destination + last - first + 1],
        data.channels["left"][first : last + 1],
    )
    assert np.ptp(result.channels["left"][first : last + 1]) == 0.0
    np.testing.assert_array_equal(result.channels["right"], right)
    assert result.duration == data.duration


def test_move_segment_rejects_occupied_target_and_out_of_bounds() -> None:
    times = np.arange(101) / 10.0
    values = np.sin(times * 4.0)
    data = TrajectoryData(times, {"q0": values}, ["q0"])
    with pytest.raises(ValueError, match="并非空白"):
        move_segment(data, 1.0, 2.0, 5.0, ["q0"])
    with pytest.raises(ValueError, match="轨迹末尾"):
        move_segment(data, 1.0, 3.0, 9.0, ["q0"])


def test_move_segment_allows_overlap_for_small_timing_nudge() -> None:
    times = np.arange(101) / 10.0
    values = np.zeros(101)
    values[31:50] = np.sin(np.linspace(0.0, np.pi, 19))
    data = TrajectoryData(times, {"q0": values}, ["q0"])

    moved = move_segment(data, 3.0, 5.0, 2.5, ["q0"])
    np.testing.assert_allclose(
        moved.channels["q0"][25:46], data.channels["q0"][30:51]
    )
    # Only the tail vacated by the shift becomes a hold; the overlap contains
    # the original snapshot shifted intact.
    assert np.ptp(moved.channels["q0"][46:51]) == 0.0


def test_phase_aligned_copy_keeps_speed_and_shared_bimanual_phase() -> None:
    times = np.arange(2001) / 1000.0
    wave = np.sin(2.0 * np.pi * 12.0 * times)
    data = TrajectoryData(
        times,
        {"left": wave, "right": wave + 0.4},
        ["left", "right"],
    )
    insert_at = 1.137
    plain = copy_segment(data, 0.2, 0.7, insert_at)
    aligned = phase_aligned_copy_segment(data, 0.2, 0.7, insert_at)
    seam = data.nearest_frame(insert_at)
    plain_jump = abs(plain.channels["left"][seam + 1] - plain.channels["left"][seam])
    aligned_jump = abs(
        aligned.channels["left"][seam + 1] - aligned.channels["left"][seam]
    )
    assert aligned_jump < plain_jump
    inserted = slice(seam + 1, seam + 1 + data.nearest_frame(0.5))
    np.testing.assert_allclose(
        aligned.channels["right"][inserted] - aligned.channels["left"][inserted],
        0.4,
        atol=1e-10,
    )


def test_high_speed_phase_smooth_repairs_without_slowing_or_desynchronizing() -> None:
    times = np.arange(3001) / 1000.0
    wave = np.sin(2.0 * np.pi * 20.0 * times)
    damaged = wave.copy()
    first, last = 1200, 1349
    damaged[first : last + 1] = 0.35
    data = TrajectoryData(
        times,
        {"left": damaged, "right": damaged + 0.25},
        ["left", "right"],
    )
    output = high_speed_phase_smooth(data, times[first], times[last])
    assert output.frame_count == data.frame_count
    assert output.duration == pytest.approx(data.duration)
    before_error = float(np.mean((damaged[first : last + 1] - wave[first : last + 1]) ** 2))
    after_error = float(
        np.mean((output.channels["left"][first : last + 1] - wave[first : last + 1]) ** 2)
    )
    assert after_error < before_error * 0.1
    np.testing.assert_allclose(
        output.channels["right"][first : last + 1]
        - output.channels["left"][first : last + 1],
        0.25,
        atol=1e-10,
    )


def test_speed_ramp_matches_normal_speed_at_boundaries(trajectory: TrajectoryData) -> None:
    output = speed_ramp_segment(trajectory, 2, 6, 2)
    assert output.duration < trajectory.duration
    start = output.nearest_frame(2)
    before = output.channels["q0"][start] - output.channels["q0"][start - 1]
    after = output.channels["q0"][start + 1] - output.channels["q0"][start]
    assert after == pytest.approx(before, rel=0.15, abs=1e-3)
    end = output.nearest_frame(4)
    np.testing.assert_allclose(
        output.channels["q0"][start - 2 : start + 3],
        trajectory.channels["q0"][trajectory.nearest_frame(2) - 2 : trajectory.nearest_frame(2) + 3],
    )
    np.testing.assert_allclose(
        output.channels["q0"][end - 2 : end + 3],
        trajectory.channels["q0"][trajectory.nearest_frame(6) - 2 : trajectory.nearest_frame(6) + 3],
    )


def test_speed_ramp_rejects_too_few_frames_for_smooth_boundary(
    trajectory: TrajectoryData,
) -> None:
    with pytest.raises(ValueError, match="至少需要保留 7 帧"):
        speed_ramp_segment(trajectory, 2, 2.1, 100)


def test_explicit_quintic_speed_blends_repair_compressed_boundaries(
    trajectory: TrajectoryData,
) -> None:
    engine = TimelineEngine(
        trajectory,
        labels=[
            TimelineLabel("before_blend", 1.0),
            TimelineLabel("core_start", 2.0),
            TimelineLabel("core_middle", 4.0),
            TimelineLabel("core_end", 6.0),
            TimelineLabel("after_blend", 7.0),
        ],
    )
    edit = engine.add_operation(
        {
            "type": "speed_ramp",
            "start": 2.0,
            "end": 6.0,
            "factor": 2.0,
            "ramp_duration": 0.2,
            "changes_duration": True,
        }
    )
    assert edit.mapping_operation["start"] == pytest.approx(2.0)
    assert edit.mapping_operation["end"] == pytest.approx(6.0)
    labels = {label.id: label.time for label in engine.labels}
    assert labels["before_blend"] == pytest.approx(1.0)
    assert labels["core_start"] == pytest.approx(2.0)
    assert labels["core_middle"] == pytest.approx(3.05, abs=0.02)
    assert labels["core_end"] == pytest.approx(4.10, abs=0.02)
    assert labels["after_blend"] == pytest.approx(5.10, abs=0.02)


def test_label_driven_splice_replaces_selected_range(
    trajectory: TrajectoryData,
) -> None:
    inserted = TrajectoryData(
        np.arange(51) / 100,
        {
            name: values[:51].copy()
            for name, values in trajectory.channels.items()
        },
        trajectory.position_channels,
    )
    output = splice_at(
        trajectory,
        inserted,
        at=2.0,
        replace_until=4.0,
        transition_duration=0.1,
    )
    assert output.duration == pytest.approx(8.71, abs=0.02)
    assert np.all(np.diff(output.times) > 0)


def test_quintic_keyframe_patch_is_exact_and_c2_at_bounds(
    trajectory: TrajectoryData,
) -> None:
    original = trajectory.channels["q0"].copy()
    key = trajectory.nearest_frame(5.0)
    output = quintic_keyframe_patch(
        trajectory, 4.0, 5.0, 6.0, {"q0": original[key] + 0.2}
    )
    assert output.channels["q0"][key] == pytest.approx(original[key] + 0.2)
    for time_value in (4.0, 6.0):
        frame = trajectory.nearest_frame(time_value)
        assert output.channels["q0"][frame] == pytest.approx(original[frame])
    np.testing.assert_array_equal(output.channels["q1"], trajectory.channels["q1"])


def test_sample_patch_targets_only_selected_channels(
    trajectory: TrajectoryData,
) -> None:
    first = trajectory.nearest_frame(1.0)
    last = trajectory.nearest_frame(1.2)
    replacement = np.linspace(-1, 1, last - first + 1)
    output = sample_patch(
        trajectory, 1.0, 1.2, {"q0": replacement.tolist()}
    )
    np.testing.assert_allclose(output.channels["q0"][first : last + 1], replacement)
    np.testing.assert_array_equal(output.channels["q1"], trajectory.channels["q1"])


def test_resized_sample_patch_preserves_bezier_middle_and_c1_boundaries(
    trajectory: TrajectoryData,
) -> None:
    from studio.validation.checks import validate_operation_continuity

    first = trajectory.nearest_frame(2.0)
    last = trajectory.nearest_frame(4.0)
    source = trajectory.channels["q0"][first : last + 1]
    curved = source + 0.4 * np.sin(np.linspace(0.0, np.pi, len(source)))
    output = sample_patch(
        trajectory, 2.0, 4.0, {"q0": curved}, output_duration=3.0
    )
    operation = {
        "type": "sample_patch",
        "start": 2.0,
        "end": 4.0,
        "channel_values": {"q0": curved.tolist()},
        "segment_duration": 3.0,
        "_actual_new_span": 3.0,
        "strict_continuity": True,
    }
    report = validate_operation_continuity(output, operation)
    assert not [issue for issue in report.issues if issue.code == "c1_discontinuity"]
    middle = output.nearest_frame(3.5)
    assert output.channels["q0"][middle] > 0.2


def test_quintic_can_shorten_total_duration(
    trajectory: TrajectoryData,
) -> None:
    output = polynomial_transition(
        trajectory,
        2.0,
        6.0,
        ["q0"],
        order=5,
        output_duration=2.0,
    )
    assert output.duration == pytest.approx(8.0, abs=0.02)
    start = output.nearest_frame(2.0)
    end = output.nearest_frame(4.0)
    assert output.channels["q0"][start] == pytest.approx(
        trajectory.channels["q0"][trajectory.nearest_frame(2.0)]
    )
    assert output.channels["q0"][end] == pytest.approx(
        trajectory.channels["q0"][trajectory.nearest_frame(6.0)]
    )


def test_partial_motor_acceleration_keeps_global_duration(
    trajectory: TrajectoryData,
) -> None:
    output = speed_ramp_segment(
        trajectory, 2.0, 6.0, 2.0, channels=["q0"]
    )
    assert output.duration == pytest.approx(trajectory.duration)
    np.testing.assert_array_equal(output.channels["q1"], trajectory.channels["q1"])
    accelerated_end = output.nearest_frame(4.0)
    original_end = trajectory.channels["q0"][trajectory.nearest_frame(6.0)]
    assert output.channels["q0"][accelerated_end] == pytest.approx(
        original_end, abs=1e-8
    )
    assert np.allclose(
        output.channels["q0"][accelerated_end : output.nearest_frame(6.0) + 1],
        original_end,
    )


def test_delete_shortens_and_is_contiguous(trajectory: TrajectoryData) -> None:
    output = delete_range(trajectory, 3, 5)
    assert output.duration == pytest.approx(8, abs=0.02)
    assert np.all(np.diff(output.times) > 0)


def test_timeline_is_non_destructive_and_undoable(trajectory: TrajectoryData) -> None:
    original = trajectory.channels["q0"].copy()
    engine = TimelineEngine(trajectory)
    engine.add_operation({"type": "speed", "start": 2.0, "end": 6.0, "factor": 2.0})
    assert engine.render().duration < trajectory.duration
    np.testing.assert_array_equal(trajectory.channels["q0"], original)
    assert engine.undo()
    assert engine.render().duration == pytest.approx(trajectory.duration)
    assert engine.redo()
    assert engine.render().duration < trajectory.duration


def test_label_remapping_for_speed_and_delete() -> None:
    labels = [TimelineLabel("A", 1), TimelineLabel("B", 4), TimelineLabel("C", 8)]
    speed = remap_labels(labels, {"type": "speed", "start": 2, "end": 6, "factor": 2}, 10)
    assert [label.time for label in speed] == pytest.approx([1, 3, 6])
    deleted = remap_labels(labels, {"type": "delete", "start": 2, "end": 6}, 10)
    assert [label.id for label in deleted] == ["A", "C"]
    assert deleted[-1].time == pytest.approx(4)


def test_label_remapping_for_splice_replacement() -> None:
    labels = [
        TimelineLabel("A", 1),
        TimelineLabel("B", 2),
        TimelineLabel("M", 3),
        TimelineLabel("C", 4),
        TimelineLabel("D", 8),
    ]
    mapped = remap_labels(
        labels,
        {
            "type": "splice_at",
            "at": 2.0,
            "replace_until": 4.0,
            "inserted_duration": 1.0,
            "transition_duration": 0.1,
        },
        10,
    )
    assert [label.id for label in mapped] == ["A", "B", "C", "D"]
    assert [label.time for label in mapped] == pytest.approx([1, 2, 3.2, 7.2])


def test_loop_and_copy_duplicate_internal_labels() -> None:
    labels = [
        TimelineLabel("A", 1),
        TimelineLabel("M", 1.5),
        TimelineLabel("B", 2),
        TimelineLabel("C", 4),
    ]
    looped = remap_labels(
        labels,
        {"type": "loop", "start": 1.0, "end": 2.0, "repetitions": 2},
        5,
    )
    assert [label.time for label in looped] == pytest.approx(
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0]
    )
    assert len({label.id for label in looped}) == len(looped)

    copied = remap_labels(
        labels,
        {"type": "copy", "start": 1.0, "end": 2.0, "insert_at": 4.0},
        5,
    )
    assert len(copied) == 7
    assert len({label.id for label in copied}) == len(copied)


def test_smoothing_duration_remaps_internal_and_following_labels() -> None:
    labels = [
        TimelineLabel("A", 2),
        TimelineLabel("M", 4),
        TimelineLabel("B", 6),
        TimelineLabel("C", 8),
    ]
    mapped = remap_labels(
        labels,
        {
            "type": "transition",
            "start": 2.0,
            "end": 6.0,
            "output_duration": 2.0,
        },
        10,
    )
    assert [label.time for label in mapped] == pytest.approx([2, 3, 4, 6])


def test_failed_operation_is_fully_transactional(
    trajectory: TrajectoryData,
) -> None:
    labels = [TimelineLabel("A", 2.0)]

    def reject(
        _candidate: TrajectoryData, _operation: dict[str, object]
    ) -> None:
        raise ValueError("rejected")

    engine = TimelineEngine(trajectory, labels=labels, validator=reject)
    with pytest.raises(ValueError, match="rejected"):
        engine.add_operation(
            {"type": "delete", "start": 2.0, "end": 4.0}
        )
    assert engine.operations == []
    assert engine.labels == labels
    assert not engine.undo()
    assert engine.render().duration == pytest.approx(trajectory.duration)


def test_actual_rounded_segment_duration_drives_time_mapping(
    trajectory: TrajectoryData,
) -> None:
    engine = TimelineEngine(
        trajectory,
        labels=[
            TimelineLabel("inside", 4.0),
            TimelineLabel("after", 8.0),
        ],
    )
    edit = engine.add_operation(
        {
            "type": "transition",
            "start": 2.0,
            "end": 6.0,
            "channels": ["q0"],
            "segment_duration": 1.234,
        }
    )
    assert engine.render().duration == pytest.approx(7.23)
    assert edit.map_time(6.0) == pytest.approx(3.23)
    assert [label.time for label in engine.labels] == pytest.approx(
        [2.615, 5.23]
    )
    assert engine.undo()
    assert engine.last_edit is not None
    assert engine.last_edit.unmap_time(5.23) == pytest.approx(8.0)
    assert engine.redo()
    assert engine.last_edit is not None
    assert engine.last_edit.map_time(8.0) == pytest.approx(5.23)


def test_structural_edits_recompute_mapped_velocity_channels() -> None:
    times = np.arange(1001) / 100
    data = TrajectoryData(
        times,
        {"q0": times**2, "dq0": np.full_like(times, -123.0)},
        ["q0"],
        {"q0": "dq0"},
    )
    output = delete_range(data, 3.0, 5.0)
    expected = np.gradient(output.channels["q0"], output.times, edge_order=2)
    np.testing.assert_allclose(output.channels["dq0"], expected)


def test_label_changes_share_operation_undo_redo(
    trajectory: TrajectoryData,
) -> None:
    engine = TimelineEngine(
        trajectory, labels=[TimelineLabel("A", 1.0)]
    )
    engine.replace_labels(
        [TimelineLabel("A", 1.0), TimelineLabel("B", 2.0)]
    )
    assert [label.id for label in engine.labels] == ["A", "B"]
    assert engine.undo()
    assert [label.id for label in engine.labels] == ["A"]
    assert engine.redo()
    assert [label.id for label in engine.labels] == ["A", "B"]


def test_quintic_preserves_outer_c1_c2_for_quadratic_motion() -> None:
    times = np.arange(1001) / 100
    data = TrajectoryData(times, {"q0": times**2}, ["q0"])
    output = polynomial_transition(
        data,
        2.0,
        6.0,
        ["q0"],
        order=5,
        preserve_endpoint_derivatives=True,
    )
    np.testing.assert_allclose(
        output.channels["q0"],
        data.channels["q0"],
        rtol=0,
        atol=1e-10,
    )


def test_quintic_transition_ignores_original_envelope() -> None:
    times = np.arange(101) / 100
    values = np.zeros_like(times)
    values[28:31] = [-2.0, -1.0, 0.0]
    values[70:73] = [0.0, 1.0, 2.0]
    data = TrajectoryData(times, {"q0": values}, ["q0"])
    output = polynomial_transition(
        data,
        0.30,
        0.70,
        ["q0"],
        order=5,
        preserve_endpoint_derivatives=True,
        output_duration=0.5,
    )
    first = output.nearest_frame(0.30)
    last = output.nearest_frame(0.80)
    assert output.channels["q0"][first] == pytest.approx(
        data.channels["q0"][data.nearest_frame(0.30)]
    )
    assert output.channels["q0"][last] == pytest.approx(
        data.channels["q0"][data.nearest_frame(0.70)]
    )
    assert np.isfinite(output.channels["q0"][first : last + 1]).all()


def test_boundary_velocity_cap_prevents_tiny_delta_fling() -> None:
    times = np.arange(1001) / 1000.0
    values = np.full_like(times, -1.1)
    values[:400] = -1.0 + (-0.9) * times[:400]
    values[400:] = -1.36
    values[900:] = -1.40
    data = TrajectoryData(times, {"q0": values}, ["q0"])
    start, end = 0.40, 0.90
    short = polynomial_transition(
        data,
        start,
        end,
        ["q0"],
        order=5,
        preserve_endpoint_derivatives=True,
        output_duration=0.5,
    )
    first = short.nearest_frame(start)
    last = short.nearest_frame(start + 0.5)
    segment = short.channels["q0"][first : last + 1]
    q0 = data.channels["q0"][data.nearest_frame(start)]
    q1 = data.channels["q0"][data.nearest_frame(end)]
    assert segment.min() >= min(q0, q1) - 0.5
    assert segment.max() <= max(q0, q1) + 0.5


def test_transition_from_zero_start_passes_end_c1() -> None:
    from studio.validation.checks import validate_boundary_continuity

    times = np.arange(2001) / 1000.0
    data = TrajectoryData(
        times,
        {
            "q0": np.linspace(0.0, 1.0, len(times)),
            "q8": np.linspace(0.0, 0.2, len(times)),
        },
        ["q0", "q8"],
    )
    duration = 0.5
    output = polynomial_transition(
        data,
        0.0,
        1.0,
        ["q0", "q8"],
        order=5,
        preserve_endpoint_derivatives=True,
        output_duration=duration,
    )
    report = validate_boundary_continuity(
        output, [duration], ["q0", "q8"]
    )
    assert not any(
        issue.code == "c1_discontinuity" for issue in report.issues
    )


def test_recommend_transition_duration_scales_with_boundary_speed() -> None:
    from studio.editing.operations import recommend_transition_duration

    times = np.arange(1001) / 100.0
    slow = TrajectoryData(times, {"q0": 0.1 * times}, ["q0"])
    fast = TrajectoryData(times, {"q0": 2.0 * times}, ["q0"])
    slow_t = recommend_transition_duration(slow, 2.0, 6.0, ["q0"])
    fast_t = recommend_transition_duration(fast, 2.0, 6.0, ["q0"])
    assert fast_t > slow_t
    assert slow_t >= 0.2


def test_recommended_transition_duration_avoids_joint_limit_overshoot() -> None:
    from studio.editing.operations import recommend_transition_duration

    times = np.arange(401) / 100.0
    values = np.full_like(times, 0.9)
    values[:101] = np.linspace(-0.1, 0.9, 101)
    values[200:] = np.linspace(0.9, -1.1, 201)
    data = TrajectoryData(times, {"q0": values}, ["q0"])
    duration = recommend_transition_duration(
        data, 1.0, 2.0, ["q0"],
        channel_limits={"q0": (-1.2, 1.0)},
    )
    result = polynomial_transition(
        data, 1.0, 2.0, ["q0"], order=5,
        preserve_endpoint_derivatives=True, output_duration=duration,
    )
    first = result.nearest_frame(1.0)
    last = result.nearest_frame(1.0 + duration)
    assert np.max(result.channels["q0"][first : last + 1]) <= 1.0 + 1e-9


def test_resize_rejects_memory_exhausting_duration() -> None:
    data = TrajectoryData(
        np.arange(101) / 100.0,
        {f"q{i}": np.zeros(101) for i in range(16)},
        [f"q{i}" for i in range(16)],
    )
    with pytest.raises(ValueError, match="可能耗尽内存"):
        polynomial_transition(
            data, 0.2, 0.8, list(data.position_channels),
            order=5, preserve_endpoint_derivatives=True,
            output_duration=100_000.0,
        )


def test_recommend_duration_stays_bounded_for_static_gripper() -> None:
    from studio.editing.operations import recommend_transition_duration

    times = np.arange(2001) / 1000.0
    values = np.zeros(len(times))
    values[-1] = 0.1
    data = TrajectoryData(
        times,
        {"q0": np.linspace(0, 1, len(times)), "q15": values},
        ["q0", "q15"],
    )
    duration = recommend_transition_duration(data, 0.0, 2.0, ["q0", "q15"])
    assert duration <= 60.0
    assert duration < 100.0


def test_recommend_splice_transition_duration() -> None:
    from studio.editing.operations import (
        recommend_splice_transition_duration,
    )

    base_times = np.arange(1001) / 100.0
    base = TrajectoryData(base_times, {"q0": 0.1 * base_times}, ["q0"])
    inserted_times = np.arange(501) / 100.0
    inserted = TrajectoryData(
        inserted_times,
        {"q0": 1.0 + 0.05 * inserted_times},
        ["q0"],
    )
    duration = recommend_splice_transition_duration(base, inserted, 2.0, ["q0"])
    assert 0.2 <= duration <= 60.0


def test_validate_rendered_operations_uses_actual_timing(
    trajectory: TrajectoryData,
) -> None:
    from studio.core.timeline import TimelineEngine
    from studio.validation.checks import validate_rendered_operations

    engine = TimelineEngine(trajectory)
    engine.add_operation(
        {
            "type": "transition",
            "start": 2.0,
            "end": 4.0,
            "order": 5,
            "preserve_endpoint_derivatives": True,
            "strict_continuity": True,
            "segment_duration": 1.0,
            "channels": ["q0"],
        }
    )
    report = validate_rendered_operations(trajectory, engine.operations)
    assert not any(
        issue.code == "continuity_context" for issue in report.issues
    )


def test_failed_redo_keeps_redo_transaction_intact(
    trajectory: TrajectoryData, tmp_path
) -> None:
    inserted = tmp_path / "inserted.csv"
    inserted.write_text(
        "time,q0,q1,sensor\n0,0,1,0\n.01,.01,.99,.01\n",
        encoding="utf-8",
    )
    engine = TimelineEngine(trajectory)
    engine.add_operation(
        {
            "type": "splice_at",
            "path": str(inserted),
            "at": 2.0,
        }
    )
    assert engine.undo()
    inserted.unlink()
    with pytest.raises(Exception):
        engine.redo()
    assert engine.operations == []
    assert len(engine._redo) == 1
    assert engine.render().duration == pytest.approx(trajectory.duration)


def test_reloaded_label_edit_can_undo_to_empty_labels(
    trajectory: TrajectoryData,
) -> None:
    engine = TimelineEngine(
        trajectory,
        operations=[
            {
                "type": "label_edit",
                "labels_after": [
                    {
                        "id": "A",
                        "time": 1.0,
                        "title": "",
                        "note": "",
                    }
                ],
                "_labels_before": [],
            }
        ],
        labels=[TimelineLabel("A", 1.0)],
    )
    assert engine.undo()
    assert engine.labels == []


def test_off_grid_speed_mapping_uses_actual_frame_span(
    trajectory: TrajectoryData,
) -> None:
    engine = TimelineEngine(
        trajectory, labels=[TimelineLabel("end", 6.0)]
    )
    edit = engine.add_operation(
        {
            "type": "speed",
            "start": 2.0,
            "end": 6.0,
            "factor": 3.0,
        }
    )
    assert edit.map_time(6.0) == pytest.approx(3.33)
    assert engine.labels[0].time == pytest.approx(3.33)


def test_speed_ramp_labels_follow_nonlinear_source_phase(
    trajectory: TrajectoryData,
) -> None:
    engine = TimelineEngine(
        trajectory, labels=[TimelineLabel("quarter", 3.0)]
    )
    edit = engine.add_operation(
        {
            "type": "speed_ramp",
            "start": 2.0,
            "end": 6.0,
            "factor": 2.0,
            "changes_duration": True,
        }
    )
    source_phase = np.asarray(
        edit.mapping_operation["_speed_source_phase"]
    )
    output_phase = np.asarray(
        edit.mapping_operation["_speed_output_phase"]
    )
    expected = 2.0 + 2.0 * np.interp(
        0.25, source_phase, output_phase
    )
    assert engine.labels[0].time == pytest.approx(expected)
    assert engine.labels[0].time != pytest.approx(2.5, abs=0.05)
    assert edit.unmap_time(engine.labels[0].time) == pytest.approx(3.0)
