from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QDialog, QFileDialog, QInputDialog

from studio.core.trajectory import TrajectoryData
from studio.io.trajectory_io import load_trajectory
from studio.ui.segment_dialog import SegmentEditDialog
from studio.ui.message_box import QMessageBox


class EditingActionsMixin:
    """Timeline selection, copy/paste, retiming, smoothing, and splice actions."""

    def _ask_range(self, title: str) -> tuple[float, float] | None:
        selected = self.timeline_widget.selected_range
        if selected is None:
            QMessageBox.information(
                self,
                title,
                "请先在时间轨道空白处按住左键拖动，框选要编辑的时间段。",
            )
            return None
        return selected

    def _move_selected_timeline_clip(
        self, start: float, end: float, destination: float
    ) -> None:
        data = self._rendered()
        if data is None:
            return
        channels = self._selected_motor_channels(data)
        operation = {
            "type": "move_segment",
            "start": start,
            "end": end,
            "destination": destination,
            "channels": channels,
            "strict_continuity": True,
        }
        if self._add_operation(operation):
            span = end - start
            self.timeline_widget.set_selected_range(
                destination, destination + span, emit=True
            )
            self.statusBar().showMessage(
                f"已移动 {len(channels)} 个关节的片段："
                f"{start:.3f}s → {destination:.3f}s"
            )

    def _ask_segment_duration(
        self,
        title: str,
        selected: tuple[float, float],
        data: TrajectoryData,
        channels: list[str],
        *,
        recommend: bool = False,
    ) -> float | None:
        from studio.editing.operations import (
            max_safe_segment_duration,
            recommend_transition_duration,
        )

        original_duration = selected[1] - selected[0]
        recommended: float | None = None
        note = (
            "该数值仅表示框选平滑段编辑后的时长，不是轨迹总时长。"
            "改变时长会重采样选区内全部通道并移动后续标签。"
        )
        default_duration = original_duration
        min_duration = data.frequency.median_dt * 2
        safe_max_duration = max_safe_segment_duration(
            data, selected[0], selected[1]
        )
        if recommend:
            channel_limits: dict[str, tuple[float, float]] = {}
            if self.adapter is not None:
                joint_lookup = {joint.name: joint for joint in self.adapter.info.joints}
                for channel in channels:
                    joint = joint_lookup.get(self.project.joint_mapping.get(channel, ""))
                    if joint is None or joint.lower is None or joint.upper is None:
                        continue
                    transform = self.project.mapping_transforms.get(channel, {})
                    scale = float(transform.get("scale", 1.0))
                    offset = float(transform.get("offset", 0.0))
                    if scale == 0.0:
                        continue
                    raw_limits = (
                        (float(joint.lower) - offset) / scale,
                        (float(joint.upper) - offset) / scale,
                    )
                    channel_limits[channel] = (
                        min(raw_limits), max(raw_limits)
                    )
            recommended = recommend_transition_duration(
                data, selected[0], selected[1], channels,
                channel_limits=channel_limits,
            )
            if recommended > safe_max_duration:
                QMessageBox.warning(
                    self,
                    title,
                    f"满足当前衔接条件的建议时长为 {recommended:.3f}s，但按当前"
                    f" {data.frequency.hz:.1f}Hz、{len(data.channels)} 个通道的内存预算，"
                    f"该选区最多允许 {safe_max_duration:.3f}s。\n"
                    "请扩大原始选区、降低采样率，或选择速度更接近的位置作为边界。",
                )
                return None
            default_duration = recommended
            note = (
                f"按衔接点速度/加速度、端点位移和关节限位搜索，建议时长约 "
                f"{recommended:.3f}s（已填入下方）。"
                "端点速度非零时，过长或过短都可能造成五次曲线过冲；"
                "建议值不是强制最小时长。"
                "时长只改过渡段本身，会移动后续标签。"
            )
        dialog = SegmentEditDialog(
            title,
            selected[0],
            selected[1],
            data.duration,
            channels,
            "平滑段新时长 (s)",
            default_duration,
            min_duration,
            safe_max_duration,
            resulting_total=lambda value: data.duration
            - original_duration
            + value,
            note=note,
            all_channels=set(channels).issuperset(
                data.position_channels
            ),
            recommended_minimum=None,
            parent=self,
        )
        return (
            dialog.value
            if dialog.exec() == QDialog.DialogCode.Accepted
            else None
        )

    def add_trim(self) -> None:
        selected = self._ask_range("裁剪保留")
        data = self._rendered()
        if selected and data is not None:
            channels = self._selected_motor_channels(data)
            self._add_operation({
                "type": "trim", "start": selected[0], "end": selected[1],
                "channels": channels,
                "changes_duration": set(channels).issuperset(data.position_channels),
            })

    def add_delete(self) -> None:
        selected = self._ask_range("删除片段")
        data = self._rendered()
        if selected and data is not None:
            channels = self._selected_motor_channels(data)
            self._add_operation({
                "type": "delete", "start": selected[0], "end": selected[1],
                "channels": channels,
                "changes_duration": set(channels).issuperset(data.position_channels),
            })

    def add_copy(self) -> None:
        selected = self._ask_range("复制片段")
        data = self._rendered()
        if not selected or data is None:
            return
        channels = tuple(self._selected_motor_channels(data))
        self._trajectory_clipboard = (id(data), selected[0], selected[1], channels)
        self.statusBar().showMessage(
            f"已复制片段 {selected[0]:.3f}s → {selected[1]:.3f}s "
            f"（{selected[1] - selected[0]:.3f}s）；移动播放头后按 Ctrl+V 粘贴"
        )

    def paste_copy(self) -> None:
        data = self._rendered()
        if data is None:
            return
        if self._trajectory_clipboard is None:
            QMessageBox.information(
                self, "粘贴片段", "剪贴板为空。请先框选时间段并点击“复制片段”或按 Ctrl+C。"
            )
            return
        data_id, start, end, channels = self._trajectory_clipboard
        if data_id != id(data):
            QMessageBox.information(
                self,
                "粘贴片段",
                "复制后轨迹已经发生编辑。为避免粘贴到错误帧，请重新框选并复制。",
            )
            return
        insert_at = float(data.times[self.frame])
        self._add_operation(
            {
                "type": "copy",
                "start": start,
                "end": end,
                "insert_at": insert_at,
                "phase_align": True,
                "channels": list(channels),
            }
        )
        self.statusBar().showMessage(
            f"已将复制片段以统一相位、无减速短交叉淡化方式粘贴到 {insert_at:.3f}s"
        )

    def add_loop(self) -> None:
        selected = self._ask_range("循环")
        if not selected:
            return
        data = self._rendered()
        if data is None:
            return
        repetitions, ok = QInputDialog.getInt(self, "循环", "重复次数", 1, 1, 1000)
        if ok:
            self._add_operation(
                {
                    "type": "loop",
                    "start": selected[0],
                    "end": selected[1],
                    "repetitions": repetitions,
                    "channels": self._selected_motor_channels(data),
                }
            )

    def add_hold(self) -> None:
        data = self._rendered()
        if data is None:
            return
        labels = self.timeline_widget.selected_labels()
        at = labels[0].time if labels else float(data.times[self.frame])
        duration, ok = QInputDialog.getDouble(
            self, "增加空白等待", "维持当前姿态的持续时间(s)", 1, 0.001, 3600, 3
        )
        if ok:
            operation: dict[str, object] = {
                "type": "hold", "at": at, "duration": duration,
                "channels": self._selected_motor_channels(data),
            }
            self._add_operation(operation)

    def add_speed(self) -> None:
        from studio.editing.operations import recommend_speed_ramp_duration

        data = self._rendered()
        if data is None:
            return
        selected = self._ask_range("片段变速")
        if not selected:
            return
        channels = self._selected_motor_channels(data)
        changes_duration = set(channels).issuperset(data.position_channels)
        original_span = selected[1] - selected[0]

        def ramp_duration(factor: float) -> float:
            return recommend_speed_ramp_duration(
                data, selected[0], selected[1], factor, channels
            )

        def speed_total(factor: float) -> float:
            return (
                data.duration
                - original_span
                + original_span / max(factor, 1e-9)
            )

        def speed_summary(factor: float) -> str:
            blend = ramp_duration(factor)
            compressed_end = selected[0] + original_span / factor
            front = blend if selected[0] > data.times[0] + blend * 0.5 else 0.0
            rear = blend if compressed_end < data.duration - blend * 0.5 else 0.0
            return (
                f"自动五次平滑：前置 {front:.3f}s、后置 {rear:.3f}s\n"
                f"主体先压缩为 {original_span / factor:.3f}s；"
                f"随后只在新边界附近使用最长 {blend:.3f}s 的局部五次衔接。"
            )

        dialog = SegmentEditDialog(
            "变速",
            selected[0],
            selected[1],
            data.duration,
            channels,
            "速度倍数",
            1.0,
            0.5,
            100.0,
            resulting_total=speed_total
            if changes_duration
            else lambda _factor: data.duration,
            note=(
                "框选区间先直接压缩到目标倍速，再在压缩后片段的新首尾边界"
                "附近生成短五次多项式衔接；不会按关节速度放大平滑时长。"
            ),
            all_channels=changes_duration,
            dynamic_summary=speed_summary if changes_duration else None,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            factor = dialog.value
            operation: dict[str, object] = {
                "type": "speed_ramp",
                "start": selected[0],
                "end": selected[1],
                "factor": factor,
                "label_ids": [
                    label.id for label in self.timeline_widget.selected_labels()
                ],
                "changes_duration": changes_duration,
                "strict_continuity": True,
            }
            if changes_duration:
                operation["ramp_duration"] = ramp_duration(factor)
            if channels:
                operation["channels"] = channels
            self._add_operation(operation)

    def add_transition(self) -> None:
        data = self._rendered()
        if data is None:
            return
        selected = self._ask_range("标签间五次平滑")
        if not selected:
            return
        labels = self.timeline_widget.selected_labels()
        channels = self._selected_motor_channels(data)
        output_duration = self._ask_segment_duration(
            "五次平滑参数", selected, data, channels, recommend=True
        )
        if output_duration is None:
            return
        self._add_operation(
            {
                "type": "transition",
                "start": selected[0],
                "end": selected[1],
                "order": 5,
                "preserve_endpoint_derivatives": True,
                "strict_continuity": True,
                "segment_duration": output_duration,
                "channels": channels,
                "label_ids": [label.id for label in labels],
            }
        )

    def add_high_speed_smooth(self) -> None:
        data = self._rendered()
        if data is None:
            return
        selected = self._ask_range("高速相位平滑")
        if selected is None:
            return
        channels = self._selected_motor_channels(data)
        answer = QMessageBox.question(
            self,
            "高速相位平滑（不减速）",
            f"将保持 {selected[0]:.3f}s → {selected[1]:.3f}s 的原时长和采样率，"
            "从选区前后估计统一周期并进行余弦交叉淡化。\n"
            "所有选中关节共用同一个周期，适合周期性高速动作的接缝或短时抽动。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._add_operation(
            {
                "type": "high_speed_smooth",
                "start": selected[0],
                "end": selected[1],
                "channels": channels,
                "strict_continuity": True,
            }
        )

    def add_transition_between_labels(self) -> None:
        if self._set_range_from_selected_labels():
            self.add_transition()

    def add_splice(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择要拼接的轨迹", "", "Trajectories (*.csv *.npz)"
        )
        if not path:
            return
        data = self._rendered()
        if data is None:
            return
        try:
            inserted = load_trajectory(path, data.frequency.hz)
        except Exception as error:
            QMessageBox.critical(self, "轨迹拼接", str(error))
            return
        selected_range = self.timeline_widget.selected_range
        insert_at = (
            selected_range[0]
            if selected_range
            else float(data.times[self.frame])
        )
        from studio.editing.operations import recommend_splice_transition_duration

        channels = self._selected_motor_channels(data) or list(
            data.position_channels
        )
        recommended = recommend_splice_transition_duration(
            data, inserted, insert_at, channels
        )
        duration, ok = QInputDialog.getDouble(
            self,
            "轨迹拼接",
            f"过渡段时长(s)\n推荐最低时长约 {recommended:.3f}s",
            recommended,
            recommended,
            60.0,
            3,
        )
        if ok:
            operation: dict[str, object] = {
                "type": "splice_at",
                "path": str(Path(path).resolve()),
                "at": insert_at,
                "transition_duration": duration,
                "order": 5,
                "inserted_duration": inserted.duration,
                "strict_continuity": True,
            }
            if selected_range is not None:
                answer = QMessageBox.question(
                    self,
                    "轨迹拼接",
                    f"是否替换框选的 {selected_range[0]:.3f}s → "
                    f"{selected_range[1]:.3f}s 原片段？\n"
                    "选择“否”将在选区起点插入，不删除原片段。",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if answer == QMessageBox.StandardButton.Yes:
                    operation["replace_until"] = selected_range[1]
            self._add_operation(
                operation
            )
