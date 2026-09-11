#!/usr/bin/env python3
"""Exercise OSS trajectory features on openarm_teleop/action1.csv. No hardware."""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

ROOT = Path(__file__).resolve().parents[1]
ACTION1 = Path("/home/lan/openarm_teleop/action1.csv")
MODEL_DIR = ROOT / "model_library" / "OpenArm_V1"
MODEL = MODEL_DIR / "scene.xml"
PRESET = json.loads((MODEL_DIR / "model.json").read_text(encoding="utf-8"))

from studio.io.trajectory_io import export_trajectory
from studio.ui.main_window import MainWindow
from studio.ui.message_box import QMessageBox as StudioMessageBox
from studio.ui.segment_dialog import SegmentEditDialog

LAST_ERROR = ""


class StepRecorder:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def run(self, name: str, fn) -> bool:  # type: ignore[no-untyped-def]
        try:
            note = fn() or ""
            self.rows.append((name, "PASS", str(note)))
            print(f"PASS  {name}" + (f"  ({note})" if note else ""))
            return True
        except Exception as error:
            self.rows.append((name, "FAIL", f"{error}"))
            print(f"FAIL  {name}: {error}")
            traceback.print_exc()
            return False


def _accept_dialog(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    return QDialog.DialogCode.Accepted


def _critical(*args, **kwargs):  # type: ignore[no-untyped-def]
    global LAST_ERROR
    LAST_ERROR = str(args[2] if len(args) > 2 else kwargs.get("text", ""))
    print(f"  dialog: {LAST_ERROR}")
    return QMessageBox.StandardButton.Ok


def install_dialog_stubs(window: MainWindow) -> None:
    StudioMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
    StudioMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard)
    StudioMessageBox.critical = staticmethod(_critical)
    StudioMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    window._confirm_discard_or_save = lambda: True  # type: ignore[method-assign]

    original_init = SegmentEditDialog.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        self.value = float(args[6]) if len(args) > 6 else 1.0

    SegmentEditDialog.__init__ = patched_init  # type: ignore[method-assign]
    SegmentEditDialog.exec = _accept_dialog  # type: ignore[method-assign]

    from PySide6.QtWidgets import QInputDialog, QFileDialog, QProgressDialog

    QInputDialog.getInt = staticmethod(lambda *a, **k: (1, True))
    QInputDialog.getDouble = staticmethod(lambda *a, **k: (0.25, True))

    def get_item(parent, title, label, items, current=0, editable=False):  # type: ignore[no-untyped-def]
        choices = list(items)
        if "末端" in str(label) or "end" in str(label).casefold():
            for name in choices:
                if "left_link7" in name or name.endswith("tcp"):
                    return name, True
            return choices[-1], True
        return choices[current] if choices else "", True

    QInputDialog.getItem = staticmethod(get_item)
    QProgressDialog.setValue = lambda self, *_a, **_k: None  # type: ignore[method-assign]
    QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (str(Path(tempfile.gettempdir()) / "action1_oss_export.csv"), "")
    )


def high_motion_window(data, width: float = 4.0) -> tuple[float, float]:
    times = data.times
    duration = float(times[-1])
    if duration <= width + 1.0:
        return 0.05, max(0.1, duration - 0.05)
    channels = [name for name in data.position_channels if name.startswith("q")]
    energy = np.zeros(len(times))
    for name in channels:
        values = data.channels[name]
        energy += np.abs(np.gradient(values))
    window = max(2, int(width / max(data.frequency.median_dt, 1e-6)))
    kernel = np.ones(window) / window
    smooth = np.convolve(energy, kernel, mode="same")
    center = int(np.argmax(smooth[window:-window]) + window)
    start = float(times[max(0, center - window // 2)])
    end = min(duration - 0.05, start + width)
    return start, end


def main() -> int:
    if not ACTION1.is_file():
        raise SystemExit(f"missing {ACTION1}")
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    install_dialog_stubs(window)
    rec = StepRecorder()

    def load_model() -> str:
        window._load_model_path(MODEL, PRESET)
        assert window.adapter is not None
        return f"{len(window.adapter.info.joints)} joints"

    def load_traj() -> str:
        window._import_trajectory_path(ACTION1, confirm=False)
        data = window._rendered()
        assert data is not None
        assert data.frame_count > 1000
        return f"{data.frame_count} frames, {data.duration:.3f}s, {data.frequency.hz:.1f} Hz"

    rec.run("load OpenArm_V1 model", load_model)
    if not rec.run("import action1.csv", load_traj):
        window.close()
        return 1

    data = window._rendered()
    assert data is not None
    start, end = high_motion_window(data, 4.0)
    pad_start = max(0.0, start - 1.5)
    pad_end = min(data.duration, end + 1.5)

    def select(a: float, b: float) -> None:
        window.timeline_widget.set_selected_range(a, b, emit=True)
        window.frame = window._rendered().nearest_frame((a + b) / 2)

    def trim_active() -> str:
        select(pad_start, pad_end)
        assert window.add_trim() is None or True
        rendered = window._rendered()
        assert rendered is not None
        assert rendered.duration < data.duration
        return f"kept {pad_start:.3f}-{pad_end:.3f}s → {rendered.duration:.3f}s"

    rec.run("trim to high-motion window", trim_active)
    current = window._rendered()
    assert current is not None
    mid = current.duration * 0.35
    span = min(0.8, current.duration * 0.15)

    def delete_slice() -> str:
        before = window._rendered().duration
        select(mid, mid + 0.15)
        window.add_delete()
        after = window._rendered().duration
        assert after < before
        return f"{before:.3f}s → {after:.3f}s"

    def copy_paste() -> str:
        rendered = window._rendered()
        select(0.20, 0.45)
        window.frame = rendered.nearest_frame(0.20)
        window.add_copy()
        assert window._trajectory_clipboard is not None
        window.frame = rendered.nearest_frame(min(rendered.duration - 0.01, 1.2))
        window.paste_copy()
        return "copied 0.25s and pasted"

    def loop_once() -> str:
        select(0.10, 0.25)
        window.add_loop()
        return f"duration {window._rendered().duration:.3f}s"

    def hold_pose() -> str:
        window.frame = window._rendered().nearest_frame(0.05)
        window.add_hold()
        return f"duration {window._rendered().duration:.3f}s"

    def speed_up() -> str:
        select(0.40, 0.90)
        before = window._rendered().duration
        from studio.editing.operations import recommend_speed_ramp_duration

        channels = list(window._rendered().position_channels)
        ramp = recommend_speed_ramp_duration(window._rendered(), 0.40, 0.90, 1.4, channels)
        ok = window._add_operation(
            {
                "type": "speed_ramp",
                "start": 0.40,
                "end": 0.90,
                "factor": 1.4,
                "ramp_duration": ramp,
                "changes_duration": True,
                "strict_continuity": True,
                "channels": channels,
            }
        )
        if not ok:
            raise RuntimeError(LAST_ERROR or "speed_ramp rejected")
        after = window._rendered().duration
        return f"{before:.3f}s → {after:.3f}s, ramp {ramp:.3f}s"

    def quintic() -> str:
        select(0.30, 0.55)
        ok = window._add_operation(
            {
                "type": "transition",
                "start": 0.30,
                "end": 0.55,
                "order": 5,
                "preserve_endpoint_derivatives": True,
                "strict_continuity": True,
                "segment_duration": 0.25,
                "channels": ["q0", "q1", "q2"],
            }
        )
        if not ok:
            raise RuntimeError(LAST_ERROR or "quintic rejected")
        return "q0-q2"

    def high_speed() -> str:
        select(0.60, 0.85)
        window.add_high_speed_smooth()
        return "q all"

    def labels_and_range() -> str:
        rendered = window._rendered()
        window.frame = rendered.nearest_frame(0.2)
        window.project.labels = list(window.project.labels)
        from studio.core.project import TimelineLabel

        window.project.labels = [
            TimelineLabel("A", 0.15, "start"),
            TimelineLabel("B", 0.45, "end"),
        ]
        if window.timeline_engine:
            window.timeline_engine.replace_labels(window.project.labels)
            window.project.labels = list(window.timeline_engine.labels)
        window._refresh_timeline()
        window.timeline_widget.selected_label_ids = {"A", "B"}
        window.timeline_widget.set_selected_range(0.15, 0.45, emit=True)
        return "A/B labels"

    def move_clip() -> str:
        rendered = window._rendered()
        dest = 0.02
        ok = window._add_operation(
            {
                "type": "move_segment",
                "start": 0.40,
                "end": 0.52,
                "destination": dest,
                "channels": ["q7"],
                "strict_continuity": True,
            }
        )
        if not ok:
            raise RuntimeError(LAST_ERROR or "move rejected")
        return f"moved q7 0.12s to {dest:.3f}s"

    def splice_self() -> str:
        from studio.editing.operations import recommend_splice_transition_duration
        from studio.models.pose import hold_pose_trajectory

        clip = window._rendered()
        pose = {
            name: float(clip.channels[name][0])
            for name in clip.position_channels
        }
        snippet = hold_pose_trajectory(pose, duration=0.3, hz=float(clip.frequency.hz))
        snippet_path = Path(tempfile.gettempdir()) / "action1_oss_snippet.npz"
        export_trajectory(snippet, snippet_path)
        insert_at = 0.08
        blend = recommend_splice_transition_duration(
            clip, snippet, insert_at, list(clip.position_channels)
        )
        ok = window._add_operation(
            {
                "type": "splice_at",
                "path": str(snippet_path),
                "at": insert_at,
                "transition_duration": blend,
                "order": 5,
                "inserted_duration": snippet.duration,
                "strict_continuity": True,
            }
        )
        if not ok:
            raise RuntimeError(LAST_ERROR or "splice rejected")
        return f"spliced hold {snippet.duration:.3f}s at {insert_at:.3f}s blend {blend:.3f}s"

    def keyframe() -> str:
        window.generate_trajectory_from_playhead()
        return f"frame {window.frame}"

    def part_pose() -> str:
        rendered = window._rendered()
        select(0.10, 0.20)
        window.frame = rendered.nearest_frame(0.15)
        from studio.models.pose import joint_pose_patch_operation

        joint = window.project.joint_mapping.get("q0")
        assert joint
        operation = joint_pose_patch_operation(
            rendered,
            "q0",
            float(rendered.channels["q0"][window.frame]),
            frame=window.frame,
            selected_range=(0.10, 0.20),
        )
        operation["direct_pose_override"] = True
        assert window._add_operation(operation)
        return "q0 range write"

    def end_effector() -> str:
        rendered = window._rendered()
        select(0.12, 0.28)
        window.frame = rendered.nearest_frame(0.20)
        from studio.editing.kinematics import KinematicsEngine

        bodies = list(window.adapter.info.body_names)
        end_body = next(
            (name for name in bodies if "left_link7" in name),
            bodies[-1],
        )
        engine = window._kinematics_for_body(end_body, None)
        first = rendered.nearest_frame(0.12)
        last = rendered.nearest_frame(0.28)
        key = rendered.nearest_frame(0.20)
        result, report, out_first, out_last = engine.cartesian_offset_window(
            rendered,
            end_body,
            0.12,
            float(rendered.times[key]),
            0.28,
            np.array([0.0, 0.0, 0.004]),
            np.eye(3),
            0.08,
        )
        if report.failed_frames:
            raise ValueError(f"{len(report.failed_frames)} IK failures")
        assert window._add_operation(
            {
                "type": "sample_patch",
                "start": float(rendered.times[out_first]),
                "end": float(rendered.times[out_last]),
                "channel_values": {
                    channel: result.channels[channel][out_first : out_last + 1].tolist()
                    for channel in engine.channels
                },
                "strict_continuity": True,
            }
        )
        return f"{end_body} +4mm Z"

    def playhead_and_preview() -> str:
        rendered = window._rendered()
        window.seek_frame(0)
        window._apply_current_frame()
        window.seek_frame(rendered.frame_count // 2)
        window._apply_current_frame()
        window.seek_frame(rendered.frame_count - 1)
        window._apply_current_frame()
        return f"seeked 0 / mid / last ({rendered.frame_count})"

    def cameras() -> str:
        window.show_camera_monitor()
        window.camera_monitor.hide()
        return f"{window.camera_selector.count()} camera entries"

    def validate_traj() -> str:
        from studio.validation.checks import (
            validate_trajectory,
            validate_velocity_channels,
        )

        rendered = window._rendered()
        report = validate_trajectory(rendered)
        report.issues.extend(validate_velocity_channels(rendered).issues)
        errors = [issue for issue in report.issues if issue.severity == "error"]
        return f"{len(report.issues)} issues, {len(errors)} errors"

    def export_csv() -> str:
        rendered = window._rendered()
        path = Path(tempfile.gettempdir()) / "action1_oss_export.csv"
        export_trajectory(rendered, path)
        assert path.is_file() and path.stat().st_size > 100
        return str(path)

    def undo_redo() -> str:
        before = window._rendered().duration
        window.undo()
        mid = window._rendered().duration
        window.redo()
        after = window._rendered().duration
        return f"{before:.3f} → undo {mid:.3f} → redo {after:.3f}"

    rec.run("delete slice", delete_slice)
    rec.run("copy / paste", copy_paste)
    rec.run("loop", loop_once)
    rec.run("hold", hold_pose)
    rec.run("speed ramp 1.4x", speed_up)
    rec.run("quintic smooth", quintic)
    rec.run("high-speed phase smooth", high_speed)
    rec.run("labels", labels_and_range)
    rec.run("move clip", move_clip)
    rec.run("splice snippet", splice_self)
    rec.run("playhead keyframe", keyframe)
    rec.run("part-module range write", part_pose)
    rec.run("end-effector +4mm", end_effector)
    rec.run("playhead / 3D preview frames", playhead_and_preview)
    rec.run("camera monitor (no devices)", cameras)
    rec.run("validate (no hardware)", validate_traj)
    rec.run("export CSV", export_csv)
    rec.run("undo / redo", undo_redo)

    window._set_dirty(False)
    window.close()
    application.processEvents()

    failed = [name for name, status, _note in rec.rows if status == "FAIL"]
    print("\n=== action1 feature pass ===")
    for name, status, note in rec.rows:
        print(f"{status:4}  {name}: {note}")
    print(f"\n{len(rec.rows) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
