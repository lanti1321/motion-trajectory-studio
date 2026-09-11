from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import QDialog, QInputDialog

from studio.core.project import TimelineLabel, TimelineSegment
from studio.core.trajectory import TrajectoryData
from studio.ui.segment_annotate_dialog import SegmentAnnotateDialog
from studio.ui.message_box import QMessageBox


class TimelineActionsMixin:
    """Labels, ranges, motor scope, undo/redo, and timeline refresh actions."""

    def add_label(self) -> None:
        data = self._rendered()
        if data is None:
            return
        title, ok = QInputDialog.getText(self, "添加标签", "标签标题")
        if not ok:
            return
        label_id = f"T{len(self.project.labels) + 1:03d}"
        labels = [
            *self.project.labels,
            TimelineLabel(
                label_id,
                float(data.times[self.frame]),
                title or label_id,
            ),
        ]
        if self.timeline_engine:
            self.timeline_engine.replace_labels(labels)
            self.project.operations = list(self.timeline_engine.operations)
            self.project.labels = list(self.timeline_engine.labels)
        else:
            self.project.labels = labels
        self._refresh_timeline()
        self._set_dirty(True)

    def rename_selected_label(self) -> None:
        selected = self.timeline_widget.selected_labels()
        if len(selected) != 1:
            QMessageBox.information(
                self,
                "重命名标签",
                "请只选择一个要重命名的标签。",
            )
            return
        label = selected[0]
        title, ok = QInputDialog.getText(
            self,
            "重命名标签",
            "标签名称",
            text=label.title or label.id,
        )
        if not ok or not title.strip():
            return
        labels = [
            replace(item, title=title.strip())
            if item.id == label.id
            else replace(item)
            for item in self.project.labels
        ]
        if self.timeline_engine is not None:
            self.timeline_engine.replace_labels(labels)
            self.project.operations = list(
                self.timeline_engine.operations
            )
            self.project.labels = list(self.timeline_engine.labels)
        else:
            self.project.labels = labels
        self._refresh_timeline()
        self._set_dirty(True)

    def _on_label_selection_changed(self, label_ids: list[str]) -> None:
        self._update_selection_status(label_ids)

    def select_range_from_labels(self) -> None:
        self._set_range_from_selected_labels()

    def _set_range_from_selected_labels(self) -> bool:
        labels = self.timeline_widget.selected_labels()
        if len(labels) < 2:
            QMessageBox.information(
                self,
                "按标签设置选区",
                "请至少选择两个标签；将使用最早和最晚标签作为选区边界。",
            )
            return False
        start = labels[0].time
        end = labels[-1].time
        self.timeline_widget.set_selected_range(start, end)
        self.statusBar().showMessage(
            f"已按标签设置选区：{labels[0].id} → {labels[-1].id}"
        )
        return True

    def _on_range_selection_changed(
        self, _selected_range: tuple[float, float] | None
    ) -> None:
        self._update_selection_status(
            [label.id for label in self.timeline_widget.selected_labels()]
        )

    def _update_selection_status(self, _label_ids: list[str]) -> None:
        selected = self.timeline_widget.selected_labels()
        lines: list[str] = []
        if self.timeline_widget.selected_range:
            start, end = self.timeline_widget.selected_range
            lines.append(
                f"时间选区：{start:.3f}s → {end:.3f}s（{end - start:.3f}s）"
            )
        else:
            lines.append("未框选时间段：在关节轨道空白处按住左键拖动")
        segment_id = self.timeline_widget.selected_segment_id
        if segment_id:
            segment = next(
                (
                    item
                    for item in self.project.segments
                    if item.id == segment_id
                ),
                None,
            )
            if segment is not None:
                lines.append(
                    f"动作分段：{segment.title or segment.id} "
                    f"({segment.start:.3f}s → {segment.end:.3f}s)"
                )
        elif self.project.segments:
            lines.append(
                f"已标注 {len(self.project.segments)} 个动作分段（点击分段栏色块可选中）"
            )
        if selected:
            lines.append(
                f"参考标签（{len(selected)}）："
                + "、".join(
                    label.title.strip() or label.id
                    for label in selected
                )
            )
        else:
            lines.append("未选择参考标签（Ctrl/Shift+点击标签可多选）")
        self.label_selection_status.setText("\n".join(lines))

    def _on_channel_selection_changed(self) -> None:
        # Highlighting must not rebuild the timeline: the channel arrays hold
        # hundreds of thousands of samples and re-deriving motion spans here
        # would stall the UI on every click.
        self.timeline_widget.set_highlighted_tracks(
            {item.text() for item in self.channel_list.selectedItems()}
        )
        self._update_motor_scope_status()

    def _update_motor_scope_status(self) -> None:
        channels = [item.text() for item in self.channel_list.selectedItems()]
        if channels:
            self.motor_scope_status.setText(
                f"当前操作电机/通道（{len(channels)}）：{'、'.join(channels)}"
            )
        else:
            self.motor_scope_status.setText(
                "未单独选择电机：操作作用于全部位置通道"
            )

    def _selected_motor_channels(self, data: TrajectoryData) -> list[str]:
        selected = [
            item.text()
            for item in self.channel_list.selectedItems()
            if item.text() in data.position_channels
        ]
        return selected or list(data.position_channels)

    def save_joint_group(self) -> None:
        channels = [item.text() for item in self.channel_list.selectedItems()]
        if not channels:
            QMessageBox.information(self, "保存关节组", "请先选择一个或多个通道")
            return
        current_name = str(self.joint_group_combo.currentData() or "")
        name, ok = QInputDialog.getText(
            self, "保存关节组", "组名称", text=current_name
        )
        if ok and name.strip():
            before = {
                key: list(value)
                for key, value in self.project.joint_groups.items()
            }
            after = {
                **before,
                name.strip(): list(channels),
            }
            self._commit_joint_groups(before, after)
            self._select_joint_group_by_name(name.strip())

    def _commit_joint_groups(
        self,
        before: dict[str, list[str]],
        after: dict[str, list[str]],
    ) -> None:
        if self.timeline_engine is not None:
            self.timeline_engine.add_operation(
                {
                    "type": "joint_group_edit",
                    "groups_before": before,
                    "groups_after": after,
                }
            )
            self.project.operations = list(self.timeline_engine.operations)
        self.project.joint_groups = after
        self._refresh_joint_group_controls()
        self._set_dirty(True)

    def _refresh_joint_group_controls(self) -> None:
        groups = list(self.project.joint_groups.items())
        combo = getattr(self, "joint_group_combo", None)
        if combo is not None:
            previous = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for slot, (name, channels) in enumerate(groups):
                shortcut = f"Alt+{slot + 1}" if slot < 9 else "—"
                combo.addItem(
                    f"{shortcut}  {name}  ({len(channels)})", name
                )
            if previous is not None:
                index = combo.findData(previous)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)
        for slot, action in enumerate(
            getattr(self, "_joint_group_actions", [])
        ):
            if slot < len(groups):
                name, channels = groups[slot]
                action.setText(
                    f"关节组 {slot + 1}：{name}（{len(channels)} 个关节）"
                )
                action.setEnabled(True)
            else:
                action.setText(f"关节组 {slot + 1}：空")
                action.setEnabled(False)

    def _select_joint_group_combo_index(self, index: int) -> None:
        name = self.joint_group_combo.itemData(index)
        if name is not None:
            self._select_joint_group_by_name(str(name))

    def select_joint_group_slot(self, slot: int) -> None:
        groups = list(self.project.joint_groups)
        if 0 <= slot < len(groups):
            self._select_joint_group_by_name(groups[slot])

    def _select_joint_group_by_name(self, name: str) -> None:
        requested = set(self.project.joint_groups.get(name, []))
        if not requested:
            return
        self.channel_list.blockSignals(True)
        self.channel_list.clearSelection()
        available: set[str] = set()
        for index in range(self.channel_list.count()):
            item = self.channel_list.item(index)
            if item.text() in requested:
                item.setSelected(True)
                available.add(item.text())
        self.channel_list.blockSignals(False)
        self._on_channel_selection_changed()
        missing = sorted(requested - available)
        message = f"已选择关节组“{name}”：{len(available)} 个关节"
        if missing:
            message += f"；当前轨迹缺少 {len(missing)} 个保存通道"
        self.statusBar().showMessage(message)

    def rename_joint_group(self) -> None:
        old_name = self.joint_group_combo.currentData()
        if old_name is None:
            QMessageBox.information(self, "重命名关节组", "当前没有可重命名的关节组")
            return
        new_name, ok = QInputDialog.getText(
            self, "重命名关节组", "新名称", text=str(old_name)
        )
        new_name = new_name.strip()
        if not ok or not new_name or new_name == old_name:
            return
        if new_name in self.project.joint_groups:
            QMessageBox.warning(self, "重命名关节组", "该名称已经存在")
            return
        before = {key: list(value) for key, value in self.project.joint_groups.items()}
        after = {
            (new_name if key == old_name else key): list(value)
            for key, value in before.items()
        }
        self._commit_joint_groups(before, after)
        self._select_joint_group_by_name(new_name)

    def delete_joint_group(self) -> None:
        name = self.joint_group_combo.currentData()
        if name is None:
            QMessageBox.information(self, "删除关节组", "当前没有可删除的关节组")
            return
        answer = QMessageBox.question(
            self,
            "删除关节组",
            f"确定删除关节组“{name}”吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        before = {key: list(value) for key, value in self.project.joint_groups.items()}
        after = {key: value for key, value in before.items() if key != name}
        self._commit_joint_groups(before, after)

    def _add_operation(self, operation: dict[str, object]) -> bool:
        if self.timeline_engine is None:
            return False
        try:
            before = self.timeline_engine.render()
            playhead_time = float(before.times[self.frame])
            selected_range = self.timeline_widget.selected_range
            edit = self.timeline_engine.add_operation(operation)
            self.project.operations = list(self.timeline_engine.operations)
            self.project.labels = list(self.timeline_engine.labels)
            rendered = self.timeline_engine.render()
            self.frame = rendered.nearest_frame(edit.map_time(playhead_time))
            if self._trajectory_clipboard is not None:
                _old_id, copied_start, copied_end, copied_channels = self._trajectory_clipboard
                mapped_copy_start = edit.map_time(copied_start)
                mapped_copy_end = edit.map_time(copied_end)
                if mapped_copy_end > mapped_copy_start:
                    self._trajectory_clipboard = (
                        id(rendered),
                        mapped_copy_start,
                        mapped_copy_end,
                        copied_channels,
                    )
                else:
                    self._trajectory_clipboard = None
            self._remap_segments(edit)
            self._refresh_timeline()
            if selected_range is not None:
                mapped_start = edit.map_time(selected_range[0])
                mapped_end = edit.map_time(selected_range[1])
                if mapped_end > mapped_start:
                    self.timeline_widget.set_selected_range(
                        mapped_start, mapped_end, emit=True
                    )
            self._apply_current_frame()
            self._set_dirty(True)
            return True
        except Exception as error:
            QMessageBox.critical(self, "编辑失败", str(error))
            return False

    def undo(self) -> None:
        if self.timeline_engine:
            current = self.timeline_engine.render()
            playhead_time = float(current.times[self.frame])
            selected_range = self.timeline_widget.selected_range
        else:
            return
        if self.timeline_engine.undo():
            operation = self.timeline_engine.last_change_operation or {}
            if operation.get("type") == "joint_group_edit":
                self.project.joint_groups = {
                    str(name): [str(value) for value in values]
                    for name, values in dict(
                        operation.get("groups_before", {})
                    ).items()
                }
            self.project.operations = list(self.timeline_engine.operations)
            self.project.labels = list(self.timeline_engine.labels)
            edit = self.timeline_engine.last_edit
            rendered = self.timeline_engine.render()
            if edit is not None:
                self.frame = rendered.nearest_frame(
                    edit.unmap_time(playhead_time)
                )
                self._remap_segments_with(edit.unmap_time, rendered.duration)
                if selected_range is not None:
                    self.timeline_widget.set_selected_range(
                        edit.unmap_time(selected_range[0]),
                        edit.unmap_time(selected_range[1]),
                        emit=False,
                    )
            self._refresh_timeline()
            self._apply_current_frame()
            self._set_dirty(True)

    def redo(self) -> None:
        if self.timeline_engine:
            current = self.timeline_engine.render()
            playhead_time = float(current.times[self.frame])
            selected_range = self.timeline_widget.selected_range
        else:
            return
        try:
            changed = self.timeline_engine.redo()
        except Exception as error:
            QMessageBox.critical(self, "重做失败", str(error))
            return
        if changed:
            operation = self.timeline_engine.last_change_operation or {}
            if operation.get("type") == "joint_group_edit":
                self.project.joint_groups = {
                    str(name): [str(value) for value in values]
                    for name, values in dict(
                        operation.get("groups_after", {})
                    ).items()
                }
            self.project.operations = list(self.timeline_engine.operations)
            self.project.labels = list(self.timeline_engine.labels)
            edit = self.timeline_engine.last_edit
            rendered = self.timeline_engine.render()
            if edit is not None:
                self.frame = rendered.nearest_frame(
                    edit.map_time(playhead_time)
                )
                self._remap_segments(edit)
                if selected_range is not None:
                    self.timeline_widget.set_selected_range(
                        edit.map_time(selected_range[0]),
                        edit.map_time(selected_range[1]),
                        emit=False,
                    )
            self._refresh_timeline()
            self._apply_current_frame()
            self._set_dirty(True)

    def _refresh_timeline(self) -> None:
        self._refresh_joint_group_controls()
        data = self._rendered()
        if data is None:
            return
        tracks = list(data.position_channels)
        highlighted = {
            item.text()
            for item in self.channel_list.selectedItems()
            if item.text() in tracks
        }
        channel_values = {
            name: data.channels[name]
            for name in tracks
            if name in data.channels
        }
        overridden_ranges: list[tuple[str, float, float]] = []
        if self.timeline_engine is not None:
            for operation in self.timeline_engine.operations:
                if not operation.get("direct_pose_override"):
                    continue
                values = operation.get("channel_values")
                if not isinstance(values, dict):
                    continue
                start = float(operation.get("start") or 0.0)
                end = float(operation.get("end", start))
                overridden_ranges.extend(
                    (str(channel), start, end)
                    for channel in values
                    if str(channel) in tracks
                )
        self.timeline_widget.set_data(
            data.duration,
            tracks,
            self.project.labels,
            data.frequency.median_dt,
            data.times.tolist(),
            segments=list(self.project.segments),
            overridden_ranges=overridden_ranges,
            channel_times=data.times,
            channel_values=channel_values,
            highlighted_tracks=highlighted,
        )
        self._on_label_selection_changed(
            [label.id for label in self.timeline_widget.selected_labels()]
        )

    def _remap_segments(self, edit) -> None:  # type: ignore[no-untyped-def]
        data = self._rendered()
        if data is None:
            return
        self._remap_segments_with(edit.map_time, data.duration)

    def _remap_segments_with(
        self, mapper, duration: float  # type: ignore[no-untyped-def]
    ) -> None:
        remapped: list[TimelineSegment] = []
        for segment in self.project.segments:
            start = float(mapper(segment.start))
            end = float(mapper(segment.end))
            if end <= start or start >= duration:
                continue
            remapped.append(
                replace(
                    segment,
                    start=max(0.0, start),
                    end=min(duration, end),
                )
            )
        self.project.segments = remapped

    def annotate_selected_segment(self) -> None:
        selected = self.timeline_widget.selected_range
        if selected is None:
            QMessageBox.information(
                self,
                "标注动作分段",
                "请先在时间轴上框选要标注的时间段。",
            )
            return
        start, end = selected
        dialog = SegmentAnnotateDialog(start, end, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        title = dialog.segment_title or f"S{len(self.project.segments) + 1:03d}"
        segment_id = f"S{len(self.project.segments) + 1:03d}"
        used = {segment.id for segment in self.project.segments}
        suffix = 2
        while segment_id in used:
            segment_id = f"S{len(self.project.segments) + suffix:03d}"
            suffix += 1
        self.project.segments.append(
            TimelineSegment(
                segment_id,
                start,
                end,
                title,
                dialog.segment_color,
            )
        )
        self.timeline_widget.selected_segment_id = segment_id
        self._refresh_timeline()
        self._set_dirty(True)
        self.statusBar().showMessage(
            f"已标注动作分段：{title} ({start:.3f}s → {end:.3f}s)"
        )

    def edit_selected_segment(self) -> None:
        segment_id = self.timeline_widget.selected_segment_id
        if not segment_id:
            QMessageBox.information(self, "编辑动作分段", "请先点击分段栏中的色块。")
            return
        segment = next(
            (item for item in self.project.segments if item.id == segment_id),
            None,
        )
        if segment is None:
            return
        dialog = SegmentAnnotateDialog(
            segment.start,
            segment.end,
            segment.title,
            segment.color,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        title = dialog.segment_title or segment.title or segment.id
        self.project.segments = [
            replace(
                item,
                title=title,
                color=dialog.segment_color,
            )
            if item.id == segment_id
            else item
            for item in self.project.segments
        ]
        self._refresh_timeline()
        self._set_dirty(True)

    def delete_selected_segment(self) -> None:
        segment_id = self.timeline_widget.selected_segment_id
        if not segment_id:
            QMessageBox.information(self, "删除动作分段", "请先点击分段栏中的色块。")
            return
        self.project.segments = [
            segment
            for segment in self.project.segments
            if segment.id != segment_id
        ]
        self.timeline_widget.selected_segment_id = None
        self._refresh_timeline()
        self._set_dirty(True)
