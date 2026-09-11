from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QInputDialog, QProgressDialog

from studio.core.trajectory import TrajectoryData
from studio.editing.kinematics import KinematicsEngine
from studio.ui.jog_teach_dialog import JogTeachDialog
from studio.ui.message_box import QMessageBox


class EndEffectorControlMixin:
    """End-effector selection, drag/jog preview, IK diagnostics, and commit."""

    def _select_kinematic_bodies(
        self, title: str
    ) -> tuple[str | None, str] | None:
        if self.adapter is None:
            QMessageBox.information(self, title, "请先加载模型")
            return None
        bodies = list(self.adapter.info.body_names)
        if not bodies:
            QMessageBox.information(self, title, "模型中没有可选body")
            return None
        base_choices = ["world"] + [name for name in bodies if name != "world"]
        base, ok = QInputDialog.getItem(
            self, title, "选择基座body", base_choices, 0, False
        )
        if not ok:
            return None
        preferred = next(
            (
                index
                for index, name in enumerate(bodies)
                if any(
                    token in name.casefold()
                    for token in ("tcp", "end_effector", "tool")
                )
            ),
            len(bodies) - 1,
        )
        end_body, ok = QInputDialog.getItem(
            self, title, "选择要编辑的末端body", bodies, preferred, False
        )
        if not ok:
            return None
        return (None if base == "world" else base), end_body

    def _select_relative_kinematic_bodies(
        self, title: str
    ) -> tuple[str | None, str, str] | None:
        if self.adapter is None:
            QMessageBox.information(self, title, "请先加载模型")
            return None
        bodies = list(self.adapter.info.body_names)
        if len(bodies) < 2:
            QMessageBox.information(self, title, "模型中没有两个可选末端 body")
            return None
        base_choices = ["world"] + [name for name in bodies if name != "world"]
        base, ok = QInputDialog.getItem(
            self, title, "选择两个末端共同参考的基座 body", base_choices, 0, False
        )
        if not ok:
            return None
        likely_ends = [
            name
            for name in bodies
            if any(
                token in name.casefold()
                for token in ("tcp", "end_effector", "tool", "gripper")
            )
        ] or bodies
        primary, ok = QInputDialog.getItem(
            self,
            title,
            "选择主末端（完整保留其原始高频运动）",
            likely_ends,
            0,
            False,
        )
        if not ok:
            return None
        follower_choices = [name for name in likely_ends if name != primary]
        if not follower_choices:
            follower_choices = [name for name in bodies if name != primary]
        follower, ok = QInputDialog.getItem(
            self,
            title,
            "选择从末端（通过 IK 保持相对位姿）",
            follower_choices,
            0,
            False,
        )
        if not ok:
            return None
        return (None if base == "world" else base), primary, follower

    def _kinematics_for_body(
        self, body_name: str, base_body_name: str | None
    ) -> KinematicsEngine:
        if self.adapter is None:
            raise ValueError("model is not loaded")
        ancestors = set(self.adapter.ancestor_joint_names(body_name))
        selected_channels = {
            item.text() for item in self.channel_list.selectedItems()
        }
        mapping = {
            channel: joint
            for channel, joint in self.project.joint_mapping.items()
            if joint in ancestors
            and (not selected_channels or channel in selected_channels)
        }
        if not mapping:
            raise ValueError("末端祖先关节没有对应的轨迹通道，请先完成关节映射")
        transforms = {
            channel: self.project.mapping_transforms.get(channel, {})
            for channel in mapping
        }
        return KinematicsEngine(
            self.adapter, mapping, transforms, base_body_name=base_body_name
        )

    def _format_ik_failure_report(self, report, data: TrajectoryData) -> str:  # type: ignore[no-untyped-def]
        details = list(getattr(report, "failures", ()))
        if not details:
            return "没有收到可定位的 IK 失败采样详情"
        max_gap = max(1, int(round(data.frequency.hz * 0.05)))
        groups: list[list[object]] = []
        for detail in details:
            if not groups or detail.frame - groups[-1][-1].frame > max_gap:
                groups.append([detail])
            else:
                groups[-1].append(detail)

        lines = [
            f"最早失败时间：{float(data.times[details[0].frame]):.3f}s",
            f"共 {len(details)} 个失败采样点，合并为 {len(groups)} 个连续问题区间：",
        ]
        for number, group in enumerate(groups[:12], 1):
            start = float(data.times[group[0].frame])
            end = float(data.times[group[-1].frame])
            position_error = max(item.position_error for item in group) * 1000.0
            orientation_error = np.rad2deg(
                max(item.orientation_error for item in group)
            )
            limit_channels = sorted(
                {channel for item in group for channel in item.limit_channels}
            )
            segment_names = [
                segment.title or segment.id
                for segment in self.project.segments
                if segment.end >= start and segment.start <= end
            ]
            label_names = [
                label.title or label.id
                for label in self.project.labels
                if start - 0.05 <= label.time <= end + 0.05
            ]
            location = f"{start:.3f}s" if abs(end - start) < 1e-9 else f"{start:.3f}–{end:.3f}s"
            line = (
                f"{number}. {location}：位置误差 {position_error:.2f}mm，"
                f"姿态误差 {orientation_error:.2f}°"
            )
            if segment_names:
                line += "；动作段=" + ", ".join(segment_names)
            if label_names:
                line += "；附近标签=" + ", ".join(label_names)
            if limit_channels:
                line += "；触及关节限位=" + ", ".join(limit_channels)
            lines.append(line)
        if len(groups) > 12:
            lines.append(f"其余 {len(groups) - 12} 个区间已省略，请先处理最早问题区间。")
        return "\n".join(lines)

    def start_end_effector_drag(self) -> None:
        data = self._rendered()
        if data is None or self.adapter is None:
            QMessageBox.information(self, "末端拖拽", "请先加载模型和轨迹")
            return
        bodies = self._select_kinematic_bodies("末端拖拽")
        if bodies is None:
            return
        base_body, end_body = bodies
        mode, accepted = QInputDialog.getItem(
            self,
            "末端示教模式",
            "选择操作方式",
            ["鼠标拖拽", "6D 按钮点动示教"],
            0,
            False,
        )
        if not accepted:
            return
        try:
            engine = self._kinematics_for_body(end_body, base_body)
            key_time = float(data.times[self.frame])
            selected_range = self.timeline_widget.selected_range
            if selected_range is None:
                raise ValueError("请先框选末端拖拽自动平滑的时间段")
            start, end = selected_range
            if not start <= key_time <= end:
                raise ValueError(
                    "当前播放帧必须位于框选时间段内"
                )
            exit_duration, duration_ok = QInputDialog.getDouble(
                self,
                "末端偏移范围",
                "选区结束后额外退出平滑时长（秒）",
                0.5, 0.1, 5.0, 2,
            )
            if not duration_ok:
                return
            position, rotation = engine.sample_body_poses(
                data, end_body, np.asarray([self.frame])
            )
            base_rotation = np.eye(3)
            if base_body:
                _base_position, base_rotation = self.adapter.body_pose(base_body)
            model_q = np.clip(
                data.matrix(engine.channels)[self.frame] * engine.scales
                + engine.offsets,
                engine.minimums,
                engine.maximums,
            )
            self._end_effector_drag_context = {
                "data": data,
                "engine": engine,
                "body": end_body,
                "base_body": base_body,
                "frame": self.frame,
                "key_time": key_time,
                "start": start,
                "end": end,
                "exit_duration": exit_duration,
                "position": position[0],
                "current_position": position[0].copy(),
                "rotation": rotation[0],
                "current_rotation": rotation[0].copy(),
                "base_rotation": base_rotation,
                "current_model_q": model_q,
                "accumulated_offset": np.zeros(3),
                "accumulated_rotation": np.zeros(3),
                "last_targets": None,
                "last_report": None,
            }
            self._pending_drag_delta[:] = 0.0
            self._last_drag_update_at = 0.0
            mouse_mode = mode == "鼠标拖拽"
            self.viewport.set_end_effector_drag_enabled(mouse_mode)
            if not mouse_mode:
                dialog = JogTeachDialog(self)
                dialog.jog.connect(self._handle_end_effector_jog)
                dialog.commitRequested.connect(
                    lambda: self._finish_jog_teach(dialog)
                )
                dialog.cancelRequested.connect(self._cancel_end_effector_drag)
                self._jog_teach_dialog = dialog
                dialog.show()
            if np.isclose(key_time, start):
                range_description = (
                    f"从首帧 {key_time:.3f}s 到 {end:.3f}s 应用偏移"
                )
            elif np.isclose(key_time, end):
                range_description = (
                    f"从 {start:.3f}s 平滑调整到末帧 {key_time:.3f}s"
                )
            else:
                range_description = (
                    f"{start:.3f}s 平滑进入，从红标 {key_time:.3f}s "
                    f"到 {end:.3f}s 保持偏移"
                )
            self.statusBar().showMessage(
                f"末端拖拽：{end_body}，{range_description}"
            )
        except Exception as error:
            QMessageBox.critical(self, "末端拖拽启动失败", str(error))

    def _handle_end_effector_drag(
        self, screen_x: float, screen_y: float, depth: float, final: bool
    ) -> None:
        context = self._end_effector_drag_context
        if context is None or self.adapter is None:
            return
        try:
            if final and np.linalg.norm(self._pending_drag_delta) > 0:
                pending = self._pending_drag_delta.copy()
                self._pending_drag_delta[:] = 0.0
                self._last_drag_update_at = 0.0
                self._handle_end_effector_drag(
                    float(pending[0]),
                    float(pending[1]),
                    float(pending[2]),
                    False,
                )
                context = self._end_effector_drag_context
                if context is None:
                    return
            elif not final:
                self._pending_drag_delta += np.asarray(
                    [screen_x, screen_y, depth]
                )
                now = time.monotonic()
                if now - self._last_drag_update_at < 0.02:
                    return
                screen_x, screen_y, depth = (
                    float(value) for value in self._pending_drag_delta
                )
                self._pending_drag_delta[:] = 0.0
                self._last_drag_update_at = now
            engine = context["engine"]
            assert isinstance(engine, KinematicsEngine)
            data = context["data"]
            assert isinstance(data, TrajectoryData)
            if not final:
                right, up, forward = self.viewport.camera_axes()
                motion_scale = max(
                    0.02, float(self.viewport.camera.distance) * 0.12
                )
                world_step = motion_scale * (
                    right * screen_x - up * screen_y + forward * depth
                )
                step = np.asarray(context["base_rotation"]).T @ world_step
                step_norm = float(np.linalg.norm(step))
                max_step = 0.0015
                if step_norm > max_step:
                    step *= max_step / step_norm
                if np.linalg.norm(step) < 1e-10:
                    return
                target_position = np.asarray(context["current_position"]) + step
                target_rotation = np.asarray(context["rotation"])
                frame = int(context["frame"])
                current_model_q = np.asarray(context["current_model_q"])
                solved, report = engine.batch_ik(
                    data,
                    np.asarray([frame]),
                    str(context["body"]),
                    target_position[None, :],
                    target_rotation[None, :, :],
                    max_iterations=30,
                    initial_q=current_model_q,
                    position_only=True,
                )
                if report.max_position_error > 0.0005:
                    self.statusBar().showMessage(
                        f"增量目标不可达，位置误差={report.max_position_error * 1000:.3f}mm"
                    )
                    return
                joint_step = float(np.max(np.abs(solved[0] - current_model_q)))
                if joint_step > 0.15:
                    self.statusBar().showMessage(
                        f"IK分支发生跳变，已拒绝本次增量（最大关节变化={joint_step:.3f}）"
                    )
                    return
                raw_targets = {
                    channel: float(
                        (solved[0, index] - engine.offsets[index])
                        / engine.scales[index]
                    )
                    for index, channel in enumerate(engine.channels)
                }
                context["current_position"] = target_position
                context["current_model_q"] = solved[0].copy()
                context["accumulated_offset"] = (
                    np.asarray(context["accumulated_offset"]) + step
                )
                context["last_targets"] = raw_targets
                context["last_report"] = report
                self.adapter.apply_positions(
                    {
                        engine.mapping[channel]: float(solved[0, index])
                        for index, channel in enumerate(engine.channels)
                    }
                )
                self.viewport.refresh()
                if self.camera_preview.isVisible():
                    self.camera_preview.refresh()
                offset = np.asarray(context["accumulated_offset"])
                self.statusBar().showMessage(
                    f"增量拖拽：累计 X={offset[0] * 1000:.1f}mm "
                    f"Y={offset[1] * 1000:.1f}mm Z={offset[2] * 1000:.1f}mm "
                    f"本次步长={np.linalg.norm(step) * 1000:.2f}mm"
                )
                return
            raw_targets = context.get("last_targets")
            report = context.get("last_report")
            if not isinstance(raw_targets, dict) or report is None:
                self._cancel_end_effector_drag()
                return
            offset = np.asarray(context["accumulated_offset"])
            if report.failed_frames and report.max_position_error > 0.002:
                frame_time = float(data.times[int(context["frame"])])
                raise ValueError(
                    f"红标 {frame_time:.3f}s 的拖拽目标不可达："
                    f"位置误差={report.max_position_error * 1000:.3f}mm，"
                    f"姿态误差={np.rad2deg(report.max_orientation_error):.3f}°"
                )
            accumulated_rotation = np.asarray(
                context.get("accumulated_rotation", np.zeros(3))
            )
            if (
                np.linalg.norm(offset) < 1e-8
                and np.linalg.norm(accumulated_rotation) < 1e-8
            ):
                self._cancel_end_effector_drag()
                return
            engine = context["engine"]
            data = context["data"]
            assert isinstance(engine, KinematicsEngine)
            assert isinstance(data, TrajectoryData)
            original_rotation = np.asarray(context["rotation"])
            current_rotation = np.asarray(context["current_rotation"])
            rotation_delta = current_rotation @ original_rotation.T
            progress = QProgressDialog(
                "正在将末端偏移应用到框选轨迹…", "取消", 0, 100, self
            )
            progress.setWindowTitle("末端范围偏移")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)

            def update_progress(completed: int, total: int, phase: str) -> bool:
                progress.setLabelText(phase)
                progress.setValue(int(100 * completed / max(1, total)))
                QApplication.processEvents()
                return not progress.wasCanceled()

            try:
                result, range_report, first, last = engine.cartesian_offset_window(
                    data,
                    str(context["body"]),
                    float(context["start"]),
                    float(context["key_time"]),
                    float(context["end"]),
                    offset,
                    rotation_delta,
                    float(context["exit_duration"]),
                    progress=update_progress,
                )
            finally:
                progress.close()
            if range_report.failed_frames:
                raise ValueError(
                    f"偏移范围内有 {len(range_report.failed_frames)} 个 IK 采样点未收敛；"
                    f"最大位置误差 {range_report.max_position_error * 1000:.2f}mm，"
                    f"最大姿态误差 {np.rad2deg(range_report.max_orientation_error):.2f}°\n\n"
                    + self._format_ik_failure_report(range_report, data)
                )
            operation = {
                "type": "sample_patch",
                "start": float(data.times[first]),
                "end": float(data.times[last]),
                "channel_values": {
                    channel: result.channels[channel][first : last + 1].tolist()
                    for channel in engine.channels
                },
                "strict_continuity": True,
                "kinematics": {
                    "kind": "end_effector_offset_window",
                    "base_body": context["base_body"],
                    "end_body": context["body"],
                    "offset": offset.tolist(),
                    "rotation_radians": accumulated_rotation.tolist(),
                    "key_time": float(context["key_time"]),
                    "selected_end": float(context["end"]),
                    "exit_duration": float(context["exit_duration"]),
                    "max_position_error": range_report.max_position_error,
                    "max_orientation_error": range_report.max_orientation_error,
                },
            }
            preview_start = float(data.times[first])
            preview_end = float(data.times[last])
            self.viewport.set_end_effector_drag_enabled(False)
            self._end_effector_drag_context = None
            if self._add_operation(operation):
                rendered = self._rendered()
                if rendered is not None:
                    preview_end = min(rendered.duration, preview_end)
                    self.frame = rendered.nearest_frame(preview_start)
                    self.play_started_time = float(rendered.times[self.frame])
                    self.play_started_at = time.monotonic()
                    self._preview_stop_time = preview_end
                    self.playing = preview_end > self.play_started_time
                    self._apply_current_frame()
                    self.statusBar().showMessage(
                        f"示教已提交，正在平滑预览 {preview_start:.3f}s → {preview_end:.3f}s"
                    )
        except Exception as error:
            self._cancel_end_effector_drag()
            QMessageBox.critical(self, "末端拖拽失败", str(error))

    def _handle_end_effector_jog(
        self, dx: float, dy: float, dz: float,
        rx_degrees: float, ry_degrees: float, rz_degrees: float,
    ) -> None:
        context = self._end_effector_drag_context
        if context is None or self.adapter is None:
            return
        try:
            engine = context["engine"]
            data = context["data"]
            assert isinstance(engine, KinematicsEngine)
            assert isinstance(data, TrajectoryData)
            translation = np.asarray([dx, dy, dz], dtype=float)
            rotation_step = np.deg2rad(
                np.asarray([rx_degrees, ry_degrees, rz_degrees], dtype=float)
            )
            target_position = np.asarray(context["current_position"]) + translation
            target_rotation = (
                Rotation.from_rotvec(rotation_step).as_matrix()
                @ np.asarray(context["current_rotation"])
            )
            frame = int(context["frame"])
            current_model_q = np.asarray(context["current_model_q"])
            solved, report = engine.batch_ik(
                data, np.asarray([frame]), str(context["body"]),
                target_position[None, :], target_rotation[None, :, :],
                max_iterations=40, initial_q=current_model_q,
                position_only=False,
            )
            if (
                report.max_position_error > 0.0005
                or report.max_orientation_error > np.deg2rad(0.2)
            ):
                self.statusBar().showMessage(
                    "6D 点动目标不可达："
                    f"位置误差 {report.max_position_error * 1000:.2f}mm，"
                    f"姿态误差 {np.rad2deg(report.max_orientation_error):.2f}°"
                )
                return
            if float(np.max(np.abs(solved[0] - current_model_q))) > 0.15:
                self.statusBar().showMessage("6D 点动触发 IK 分支跳变保护，已拒绝")
                return
            context["current_position"] = target_position
            context["current_rotation"] = target_rotation
            context["current_model_q"] = solved[0].copy()
            context["accumulated_offset"] = (
                np.asarray(context["accumulated_offset"]) + translation
            )
            context["accumulated_rotation"] = (
                np.asarray(context["accumulated_rotation"]) + rotation_step
            )
            context["last_targets"] = {
                channel: float(
                    (solved[0, index] - engine.offsets[index]) / engine.scales[index]
                )
                for index, channel in enumerate(engine.channels)
            }
            context["last_report"] = report
            self.adapter.apply_positions(
                {
                    engine.mapping[channel]: float(solved[0, index])
                    for index, channel in enumerate(engine.channels)
                }
            )
            self.viewport.refresh()
            offset = np.asarray(context["accumulated_offset"]) * 1000.0
            angles = np.rad2deg(np.asarray(context["accumulated_rotation"]))
            self.statusBar().showMessage(
                f"6D示教 XYZ=({offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f})mm；"
                f"Rxyz=({angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f})°"
            )
        except Exception as error:
            QMessageBox.critical(self, "6D 按钮点动失败", str(error))

    def _finish_jog_teach(self, dialog: JogTeachDialog) -> None:
        self._handle_end_effector_drag(0.0, 0.0, 0.0, True)
        if self._jog_teach_dialog is dialog:
            self._jog_teach_dialog = None
        dialog.accept()

    def _cancel_end_effector_drag(self) -> None:
        self.viewport.set_end_effector_drag_enabled(False)
        dialog = self._jog_teach_dialog
        self._jog_teach_dialog = None
        if dialog is not None and dialog.isVisible():
            dialog.reject()
        self._end_effector_drag_context = None
        self._pending_drag_delta[:] = 0.0
        self._last_drag_update_at = 0.0
        self._apply_current_frame()
        self.statusBar().showMessage("末端拖拽已取消")
