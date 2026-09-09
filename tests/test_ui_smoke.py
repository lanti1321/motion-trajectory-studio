from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QAbstractButton, QApplication, QDockWidget, QGroupBox
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from studio.core.project import ProjectDocument, TimelineLabel
from studio.core.timeline import TimelineEngine
from studio.core.trajectory import TrajectoryData
from studio.io.trajectory_io import load_trajectory
from studio.ui.main_window import MainWindow
import studio.ui.main_window as main_window_module
from studio.ui.segment_dialog import SegmentEditDialog
from studio.ui.timeline_widget import TimelineWidget


def test_main_window_starts() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.windowTitle() == "Motion Trajectory Studio"
    assert window.timeline_widget is not None
    assert window.model_library.root.name == "model_library"
    assert window.model_library_list is not None
    window.close()


def test_main_window_minimum_size_fits_laptop_screen() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.timeline_widget.set_data(10.0, [f"q{i}" for i in range(36)], [])
    application.processEvents()
    hint = window.minimumSizeHint()
    assert hint.width() <= 1100
    assert hint.height() <= 700
    window.show()
    window._fit_to_available_screen()
    application.processEvents()
    window.showMaximized()
    application.processEvents()
    window._refresh_viewports_for_size()
    assert window.minimumSizeHint().height() <= 700
    window.close()
    application.processEvents()


def test_operation_validator_builds_baseline_report(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Exercise the delayed edit-validation path, not only window startup."""
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.adapter = object()  # type: ignore[assignment]
    window.project.joint_mapping = {"q0": "joint0"}
    window.timeline_engine = None
    monkeypatch.setattr(
        main_window_module,
        "validate_model_mapping",
        lambda *_args, **_kwargs: main_window_module.ValidationReport(),
    )
    monkeypatch.setattr(
        main_window_module,
        "filter_new_joint_limit_issues",
        lambda *_args, **_kwargs: [],
    )

    window._operation_validator(object(), {})  # type: ignore[arg-type]
    window.close()
    application.processEvents()


def test_timeline_supports_ctrl_multi_label_selection() -> None:
    application = QApplication.instance() or QApplication([])
    timeline = TimelineWidget()
    timeline.resize(1000, 200)
    timeline.set_data(
        10.0,
        ["track"],
        [
            TimelineLabel("A", 2.0),
            TimelineLabel("B", 5.0),
            TimelineLabel("C", 8.0),
        ],
    )
    timeline.show()
    application.processEvents()
    QTest.mouseClick(
        timeline, Qt.MouseButton.LeftButton, pos=QPoint(304, 10)
    )
    QTest.mouseClick(
        timeline,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        QPoint(826, 10),
    )
    assert [label.id for label in timeline.selected_labels()] == ["A", "C"]
    timeline.close()


def test_timeline_drag_selects_and_adjusts_frame_snapped_range() -> None:
    application = QApplication.instance() or QApplication([])
    timeline = TimelineWidget()
    timeline.resize(1000, 200)
    timeline.set_data(10.0, ["track"], [], frame_interval=0.01)
    timeline.show()
    application.processEvents()

    QTest.mousePress(timeline, Qt.MouseButton.LeftButton, pos=QPoint(304, 80))
    QTest.mouseMove(timeline, QPoint(652, 80))
    QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, pos=QPoint(652, 80))

    assert timeline.selected_range is not None
    assert timeline.selected_range == pytest.approx((2.0, 6.0), abs=0.02)

    end_x = 130 + int(870 * timeline.selected_range[1] / 10.0)
    QTest.mousePress(
        timeline, Qt.MouseButton.LeftButton, pos=QPoint(end_x, 80)
    )
    QTest.mouseMove(timeline, QPoint(739, 80))
    QTest.mouseRelease(
        timeline, Qt.MouseButton.LeftButton, pos=QPoint(739, 80)
    )
    assert timeline.selected_range[1] == pytest.approx(7.0, abs=0.02)
    timeline.close()


def test_timeline_dragging_selected_body_requests_clip_move() -> None:
    application = QApplication.instance() or QApplication([])
    timeline = TimelineWidget()
    timeline.resize(1000, 200)
    timeline.set_data(10.0, ["track"], [], frame_interval=0.01)
    timeline.set_selected_range(2.0, 4.0)
    requested = []
    timeline.clipMoveRequested.connect(
        lambda start, end, destination: requested.append((start, end, destination))
    )
    timeline.show()
    application.processEvents()

    # Drag the range body from about 3 seconds to about 7 seconds.
    QTest.mousePress(timeline, Qt.MouseButton.LeftButton, pos=QPoint(391, 80))
    QTest.mouseMove(timeline, QPoint(739, 80))
    QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, pos=QPoint(739, 80))

    assert requested
    assert requested[0][:2] == pytest.approx((2.0, 4.0), abs=0.02)
    assert requested[0][2] == pytest.approx(6.0, abs=0.03)
    timeline.close()


def test_click_seek_and_label_selection_do_not_replace_time_range() -> None:
    application = QApplication.instance() or QApplication([])
    timeline = TimelineWidget()
    timeline.resize(1000, 200)
    timeline.set_data(
        10.0,
        ["track"],
        [TimelineLabel("M", 5.0)],
        frame_interval=0.01,
    )
    timeline.set_selected_range(2.0, 7.0)
    timeline.show()
    application.processEvents()

    QTest.mouseClick(
        timeline, Qt.MouseButton.LeftButton, pos=QPoint(478, 80)
    )
    assert timeline.selected_range == pytest.approx((2.0, 7.0))

    QTest.mouseClick(
        timeline, Qt.MouseButton.LeftButton, pos=QPoint(565, 10)
    )
    assert [label.id for label in timeline.selected_labels()] == ["M"]
    assert timeline.selected_range == pytest.approx((2.0, 7.0))
    timeline.close()


def test_right_click_inside_timeline_range_clears_selection() -> None:
    application = QApplication.instance() or QApplication([])
    timeline = TimelineWidget()
    timeline.resize(1000, 200)
    timeline.set_data(10.0, ["track"], [], frame_interval=0.01)
    timeline.set_selected_range(2.0, 7.0)
    timeline.show()
    application.processEvents()

    # x=565 maps to approximately 5 seconds and lies inside the range.
    QTest.mouseClick(
        timeline, Qt.MouseButton.RightButton, pos=QPoint(565, 80)
    )
    assert timeline.selected_range is None
    timeline.close()


def test_timeline_first_and_last_buttons_seek_endpoint_frames() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    data = TrajectoryData(
        np.arange(11) / 10.0,
        {"q0": np.arange(11, dtype=float)},
        ["q0"],
    )
    window.source = data
    window.timeline_engine = TimelineEngine(data)
    window.frame = 5

    window.timeline_first_button.click()
    assert window.frame == 0
    window.timeline_last_button.click()
    assert window.frame == data.frame_count - 1
    window.close()


def test_joint_group_slot_replaces_channel_selection() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    for name in ("left_1", "left_2", "right_1"):
        window.channel_list.addItem(name)
    window.project.joint_groups = {
        "左臂": ["left_1", "left_2"],
        "右臂": ["right_1"],
    }
    window._refresh_joint_group_controls()

    assert window._joint_group_actions[0].shortcut().toString() == "Alt+1"
    assert window._joint_group_actions[1].shortcut().toString() == "Alt+2"
    window.select_joint_group_slot(0)
    assert {item.text() for item in window.channel_list.selectedItems()} == {
        "left_1", "left_2"
    }
    window._joint_group_actions[1].trigger()
    assert [item.text() for item in window.channel_list.selectedItems()] == [
        "right_1"
    ]
    window.close()


def test_zoomed_drag_uses_visible_time_coordinates() -> None:
    application = QApplication.instance() or QApplication([])
    timeline = TimelineWidget()
    timeline.resize(1000, 200)
    timeline.set_data(10.0, ["track"], [], frame_interval=0.01)
    timeline.set_current_time(5.0)
    timeline.set_zoom(2.0)
    timeline.show()
    application.processEvents()

    QTest.mousePress(
        timeline, Qt.MouseButton.LeftButton, pos=QPoint(304, 80)
    )
    QTest.mouseMove(timeline, QPoint(652, 80))
    QTest.mouseRelease(
        timeline, Qt.MouseButton.LeftButton, pos=QPoint(652, 80)
    )
    assert timeline.selected_range == pytest.approx(
        (3.5, 5.5), abs=0.02
    )
    timeline.close()


def test_import_trajectory_starts_clean_edit_session(
    tmp_path, monkeypatch
) -> None:
    application = QApplication.instance() or QApplication([])
    path = tmp_path / "new.csv"
    path.write_text("time,q0\n0,0\n.01,.1\n", encoding="utf-8")
    window = MainWindow()
    old = TrajectoryData(
        np.arange(11) / 100,
        {"q0": np.arange(11, dtype=float)},
        ["q0"],
    )
    window.source = old
    window.project.operations = [
        {"type": "delete", "start": 0.01, "end": 0.02}
    ]
    window.project.labels = [TimelineLabel("old", 0.01)]
    window.timeline_engine = TimelineEngine(
        old, window.project.operations, window.project.labels
    )
    window._set_dirty(False)
    monkeypatch.setattr(
        "studio.ui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(path), ""),
    )

    window.import_trajectory()

    assert window.timeline_engine is not None
    assert window.timeline_engine.operations == []
    assert window.project.labels == []
    window._set_dirty(False)
    window.close()
    application.processEvents()


def test_failed_project_open_keeps_current_session(
    tmp_path, monkeypatch
) -> None:
    application = QApplication.instance() or QApplication([])
    broken = tmp_path / "broken.motionproj"
    broken.write_text("{", encoding="utf-8")
    window = MainWindow()
    source = TrajectoryData(
        np.arange(11) / 100,
        {"q0": np.arange(11, dtype=float)},
        ["q0"],
    )
    project = ProjectDocument("current")
    engine = TimelineEngine(source)
    window.project = project
    window.source = source
    window.timeline_engine = engine
    window._set_dirty(False)
    monkeypatch.setattr(
        "studio.ui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(broken), ""),
    )
    monkeypatch.setattr(
        "studio.ui.main_window.QMessageBox.critical",
        lambda *args, **kwargs: None,
    )

    window.open_project()

    assert window.project is project
    assert window.source is source
    assert window.timeline_engine is engine
    window.close()
    application.processEvents()


def test_portable_save_rebases_runtime_splice_paths(
    tmp_path, monkeypatch
) -> None:
    application = QApplication.instance() or QApplication([])
    source_path = tmp_path / "source.csv"
    splice_path = tmp_path / "splice.csv"
    content = "time,q0\n0,0\n.01,.1\n"
    source_path.write_text(content, encoding="utf-8")
    splice_path.write_text(content, encoding="utf-8")
    destination = tmp_path / "portable" / "demo.motionproj"
    destination.parent.mkdir()

    source = load_trajectory(source_path)
    operation = {
        "type": "splice_at",
        "path": str(splice_path.resolve()),
        "at": 0.0,
    }
    window = MainWindow()
    window.source = source
    window.project = ProjectDocument(
        "demo", trajectory_path=str(source_path.resolve())
    )
    window.timeline_engine = TimelineEngine(
        source, operations=[operation]
    )
    monkeypatch.setattr(
        "studio.ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(destination), ""),
    )

    window.save_portable_project()

    runtime_path = Path(
        str(window.timeline_engine.operations[0]["path"])
    )
    assert runtime_path.is_absolute()
    assert runtime_path.is_file()
    assert runtime_path.parent == destination.parent / "media"
    assert not Path(str(window.project.operations[0]["path"])).is_absolute()
    window._set_dirty(False)
    window.close()
    application.processEvents()


def test_irregular_timeline_snaps_to_real_frame_times() -> None:
    application = QApplication.instance() or QApplication([])
    timeline = TimelineWidget()
    timeline.set_data(
        0.04,
        ["track"],
        [],
        frame_interval=0.01,
        frame_times=[0.0, 0.01, 0.025, 0.04],
    )
    timeline.set_selected_range(0.012, 0.038)
    assert timeline.selected_range == pytest.approx((0.01, 0.04))
    timeline.close()
    application.processEvents()


def test_segment_dialog_shows_total_delta_and_all_motor_scope() -> None:
    application = QApplication.instance() or QApplication([])
    dialog = SegmentEditDialog(
        "平滑",
        2.0,
        4.0,
        10.0,
        ["q0", "q1"],
        "平滑段新时长 (s)",
        1.0,
        0.01,
        20.0,
        resulting_total=lambda value: 8.0 + value,
        all_channels=True,
    )
    assert dialog.scope_label.text() == "全部位置通道"
    assert dialog.total_label.text() == "9.000s"
    assert dialog.delta_label.text() == "-1.000s"
    dialog.parameter.setValue(3.0)
    assert dialog.total_label.text() == "11.000s"
    assert dialog.delta_label.text() == "+1.000s"
    dialog.close()
    application.processEvents()


def test_new_project_creates_file_and_first_import_keeps_project(
    tmp_path, monkeypatch
) -> None:
    application = QApplication.instance() or QApplication([])
    trajectory = tmp_path / "motion.csv"
    trajectory.write_text(
        "time,q0\n0,0\n.01,.1\n", encoding="utf-8"
    )
    requested = tmp_path / "demo"
    window = MainWindow()
    monkeypatch.setattr(
        "studio.ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(requested), ""),
    )
    window.new_project()
    expected_project = tmp_path / "demo.motionproj"
    assert expected_project.is_file()
    assert window.project.path == expected_project.resolve()
    assert window.source is None
    assert window.timeline_engine is None

    window._set_dirty(True)
    monkeypatch.setattr(
        window,
        "_confirm_discard_or_save",
        lambda: pytest.fail(
            "first trajectory import must not discard a model-only project"
        ),
    )
    monkeypatch.setattr(
        "studio.ui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(trajectory), ""),
    )
    window.import_trajectory()
    assert window.source is not None
    assert window.project.path == expected_project.resolve()
    assert window.project.trajectory_path == str(trajectory.resolve())
    assert window.project.operations == []
    assert window.project.labels == []
    assert window.save_project()
    reopened = ProjectDocument.load(expected_project)
    assert reopened.resolve(reopened.trajectory_path) == trajectory.resolve()
    window._set_dirty(False)
    monkeypatch.setattr(
        window, "_confirm_discard_or_save", lambda: True
    )
    window.close()
    application.processEvents()


def test_timeline_scroll_keeps_all_joint_rows() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    tracks = [f"q{i}" for i in range(16)]
    window.timeline_widget.set_data(
        10.0,
        tracks,
        [],
        frame_interval=0.01,
    )
    application.processEvents()
    expected = (
        TimelineWidget.RULER_HEIGHT
        + TimelineWidget.SEGMENT_LANE_HEIGHT
        + 16 * TimelineWidget.MIN_ROW_HEIGHT
    )
    assert window.timeline_widget.minimumHeight() >= expected
    assert window.timeline_scroll.widgetResizable() is True
    window.close()
    application.processEvents()


def test_timeline_draws_motion_blocks() -> None:
    application = QApplication.instance() or QApplication([])
    timeline = TimelineWidget()
    times = np.linspace(0.0, 2.0, 201)
    values = np.zeros(201)
    values[50:150] = np.linspace(0.0, 1.0, 100)
    timeline.resize(900, 300)
    timeline.set_data(
        2.0,
        ["q0"],
        [],
        frame_interval=0.01,
        frame_times=times.tolist(),
        channel_times=times,
        channel_values={"q0": values},
    )
    timeline.show()
    application.processEvents()
    assert timeline._motion_spans["q0"]
    timeline.close()
    application.processEvents()


def test_selected_labels_can_define_the_edit_range() -> None:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    labels = [
        TimelineLabel("A", 1.0),
        TimelineLabel("B", 3.0),
        TimelineLabel("C", 5.0),
    ]
    window.timeline_widget.set_data(6.0, ["q0"], labels)
    window.timeline_widget.selected_label_ids = {"A", "C"}
    assert window._set_range_from_selected_labels()
    assert window.timeline_widget.selected_range == pytest.approx(
        (1.0, 5.0)
    )
    window.close()
    application.processEvents()


def test_timeline_displays_label_title_instead_of_internal_id() -> None:
    label = TimelineLabel("T001", 1.0, "抓取开始")
    assert TimelineWidget.label_display_text(label) == "抓取开始"
    assert (
        TimelineWidget.label_display_text(
            TimelineLabel("T002", 2.0, "")
        )
        == "T002"
    )


def test_loading_model_creates_part_module_inspectors(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("MOTION_STUDIO_LANG", raising=False)
    application = QApplication.instance() or QApplication([])
    model = tmp_path / "hand.xml"
    model.write_text(
        """<mujoco model="hand">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <joint name="openarm_left_joint1" type="hinge" limited="true" range="-1 1"/>
      <geom type="sphere" size=".02"/>
      <body name="finger" pos="0 0 .05">
        <joint name="lh_index_mcp_pitch" type="hinge" limited="true" range="0 1.6"/>
        <geom type="capsule" size=".01" fromto="0 0 0 .05 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    window = MainWindow()
    window._load_model_path(model)
    assert window.source is not None
    assert window.timeline_engine is not None
    assert "lh_index_mcp_pitch" in window.source.position_channels
    assert window.project.joint_mapping["lh_index_mcp_pitch"] == "lh_index_mcp_pitch"
    modules = {str(module["id"]): module for module in window.part_module_host.modules}
    assert "left_hand" in modules
    assert "left_arm" in modules
    dock_titles = [dock.windowTitle() for dock in window.findChildren(QDockWidget)]
    assert "部件模块" in dock_titles
    button_texts = [obj.text() for obj in window.findChildren(QAbstractButton)]
    assert "打开" in button_texts
    first = window.open_part_module("left_hand")
    application.processEvents()
    assert first is not None
    assert first.windowTitle() == "左手"
    assert "lh_index_mcp_pitch" in first.panel._joints
    assert "openarm_left_joint1" not in first.panel._joints
    group_titles = [box.title() for box in first.findChildren(QGroupBox)]
    assert "食指" in group_titles
    second = window.open_part_module("left_hand")
    assert second is first
    assert len(window.part_module_host.open_windows()) == 1
    window._commit_joint_pose("lh_index_mcp_pitch", 0.7)
    rendered = window.timeline_engine.render()
    np.testing.assert_allclose(rendered.channels["lh_index_mcp_pitch"], 0.7)
    window._set_dirty(False)
    window.close()
    application.processEvents()
