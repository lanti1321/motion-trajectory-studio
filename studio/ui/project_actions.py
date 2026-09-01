from __future__ import annotations

import copy
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QProgressDialog

from studio.core.project import ProjectDocument
from studio.core.timeline import TimelineEngine
from studio.core.trajectory import TrajectoryData
from studio.io.bundle import pack_project
from studio.io.trajectory_io import export_trajectory, load_trajectory
from studio.models.adapter import MuJoCoModelAdapter
from studio.ui.message_box import QMessageBox
from studio.validation.checks import (
    ValidationReport,
    filter_new_joint_limit_issues,
    sample_collisions,
    validate_model_mapping,
    validate_rendered_operations,
    validate_trajectory,
    validate_velocity_channels,
)


class ProjectActionsMixin:
    """Project persistence, export, packaging, and full validation actions."""

    def _default_save_dialog_path(self) -> str:
        if self.project.path is not None:
            return str(self.project.path.parent / f"{self.project.name}.motionproj")
        return os.path.join(os.path.expanduser("~"), f"{self.project.name}.motionproj")

    def _ask_save_project_path(self, title: str) -> Path | None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            self._default_save_dialog_path(),
            "Motion project (*.motionproj)",
        )
        if not path:
            return None
        return self._motion_project_path(path, default_name=self.project.name)

    def save_project(self) -> bool:
        if self.project.path is None:
            destination = self._ask_save_project_path("保存工程")
            if destination is None:
                return False
        else:
            destination = self.project.path
        return self._save_project_to(destination)

    def save_project_as(self) -> bool:
        destination = self._ask_save_project_path("工程另存为")
        return self._save_project_to(destination) if destination else False

    def _serialized_operations(
        self,
        destination: Path,
        operations: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        serialized: list[dict[str, object]] = []
        for operation in operations:
            item = dict(operation)
            if (
                item.get("type") in {"splice", "splice_at"}
                and item.get("path")
            ):
                candidate = Path(str(item["path"])).expanduser().resolve()
                try:
                    item["path"] = candidate.relative_to(
                        destination.parent
                    ).as_posix()
                except ValueError:
                    item["path"] = str(candidate)
            serialized.append(item)
        return serialized

    def _save_project_to(self, destination: Path) -> bool:
        previous_model_path = self.project.model_path
        previous_trajectory_path = self.project.trajectory_path
        previous_operations = list(self.project.operations)
        try:
            model_path = self.project.resolve(self.project.model_path)
            trajectory_path = self.project.resolve(
                self.project.trajectory_path
            )
            self.project.model_path = self._path_for_destination(
                model_path, destination
            )
            self.project.trajectory_path = self._path_for_destination(
                trajectory_path, destination
            )
            operations = (
                list(self.timeline_engine.operations)
                if self.timeline_engine
                else list(self.project.operations)
            )
            self.project.operations = self._serialized_operations(
                destination, operations
            )
            self.project.labels = (
                list(self.timeline_engine.labels)
                if self.timeline_engine
                else list(self.project.labels)
            )
            self.project.segments = list(self.project.segments)
            self.project.save(destination)
            self._set_dirty(False)
            self.statusBar().showMessage(f"已保存 {self.project.path}")
            return True
        except Exception as error:
            self.project.model_path = previous_model_path
            self.project.trajectory_path = previous_trajectory_path
            self.project.operations = previous_operations
            QMessageBox.critical(self, "工程保存失败", str(error))
            return False

    @staticmethod
    def _path_for_destination(
        asset: Path | None, destination: Path
    ) -> str | None:
        if asset is None:
            return None
        resolved = asset.expanduser().resolve()
        try:
            return resolved.relative_to(destination.parent).as_posix()
        except ValueError:
            return str(resolved)

    @staticmethod
    def _motion_project_path(
        path: str | Path, default_name: str = "project"
    ) -> Path:
        destination = Path(path).expanduser()
        if destination.exists() and destination.is_dir():
            destination = destination / f"{default_name}.motionproj"
        elif destination.suffix.lower() != ".motionproj":
            destination = destination.with_suffix(".motionproj")
        return destination.resolve()

    def save_portable_project(self) -> None:
        destination = self._ask_save_project_path("保存可迁移工程")
        if destination is None:
            return
        previous_project = copy.deepcopy(self.project)
        try:
            trajectory_source = (
                Path(str(self.source.metadata["source_path"])).resolve()
                if self.source
                and self.source.metadata.get("source_path")
                else self.project.resolve(self.project.trajectory_path)
            )
            runtime_operations = (
                list(self.timeline_engine.operations)
                if self.timeline_engine
                else list(self.project.operations)
            )
            self.project.path = destination
            if self.adapter:
                self.project.model_path = self.project.import_model_bundle(
                    self.adapter.path
                )
            if trajectory_source:
                self.project.trajectory_path = self.project.import_trajectory(
                    trajectory_source
                )
            portable_operations: list[dict[str, object]] = []
            path_replacements: dict[str, str] = {}
            for operation in runtime_operations:
                item = dict(operation)
                if (
                    item.get("type") in {"splice", "splice_at"}
                    and item.get("path")
                ):
                    old_path = str(item["path"])
                    source_path = Path(old_path).expanduser()
                    if not source_path.is_absolute():
                        resolved = previous_project.resolve(old_path)
                        if resolved is None:
                            raise ValueError("拼接轨迹路径无效")
                        source_path = resolved
                    relative_path = self.project.import_trajectory(
                        source_path.resolve()
                    )
                    item["path"] = relative_path
                    path_replacements[old_path] = str(
                        (destination.parent / relative_path).resolve()
                    )
                portable_operations.append(item)
            self.project.operations = portable_operations
            self.project.save()
            if self.timeline_engine is not None:
                self.timeline_engine.rebase_operation_paths(
                    path_replacements
                )
            if (
                self.source is not None
                and self.project.trajectory_path is not None
            ):
                self.source.metadata["source_path"] = str(
                    (
                        destination.parent
                        / self.project.trajectory_path
                    ).resolve()
                )
            self._set_dirty(False)
            self.statusBar().showMessage("可迁移工程已保存")
        except Exception as error:
            self.project = previous_project
            QMessageBox.critical(self, "可迁移工程保存失败", str(error))

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", "Motion project (*.motionproj)")
        if not path:
            return
        if not self._confirm_discard_or_save():
            return
        try:
            project = ProjectDocument.load(path)
            model_path = project.resolve(project.model_path)
            trajectory_path = project.resolve(project.trajectory_path)
            adapter = MuJoCoModelAdapter(model_path) if model_path else None
            if adapter is not None:
                for adjustment in project.model_adjustments:
                    if adjustment.get("type") == "joint_center_distance":
                        adapter.set_joint_center_distance(
                            str(adjustment["joint_a"]),
                            str(adjustment["joint_b"]),
                            str(adjustment["body_a"]),
                            str(adjustment["body_b"]),
                            float(adjustment["distance_m"]),
                        )
            source = load_trajectory(trajectory_path) if trajectory_path else None
            timeline_engine = None
            if source is not None:
                operations = []
                for operation in project.operations:
                    resolved = dict(operation)
                    if resolved.get("type") in {"splice", "splice_at"} and resolved.get("path"):
                        resolved_path = project.resolve(str(resolved["path"]))
                        if resolved_path is None:
                            raise ValueError("拼接轨迹路径无效")
                        resolved["path"] = str(resolved_path)
                    operations.append(resolved)
                timeline_engine = TimelineEngine(
                    source, operations, project.labels
                )
                timeline_engine.render()

            self.project = project
            self.adapter = adapter
            self.source = source
            self.timeline_engine = timeline_engine
            self.frame = 0
            self.timeline_widget.clear_selected_range(emit=False)
            self.timeline_widget.clear_label_selection()
            if self.adapter is not None:
                self.viewport.set_adapter(self.adapter)
                self.camera_preview.set_adapter(self.adapter)
            else:
                self.viewport.set_adapter(None)
                self.camera_preview.set_adapter(None)
            self._refresh_camera_selector()
            self._install_timeline_validator()
            self.channel_list.clear()
            if self.source is not None:
                self.channel_list.addItems(self.source.position_channels)
                self._update_motor_scope_status()
                self._refresh_timeline()
                self._apply_current_frame()
            self._set_dirty(False)
        except Exception as error:
            QMessageBox.critical(self, "工程打开失败", str(error))

    def export(self) -> None:
        data = self._rendered()
        if data is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出轨迹", "", "CSV (*.csv);;NPZ (*.npz)")
        if not path:
            return
        try:
            export_data = data.copy()
            export_data.recompute_velocities()
            report = self._full_validation_report(export_data)
            errors = [
                issue
                for issue in report.issues
                if issue.severity == "error"
            ]
            warnings = [
                issue
                for issue in report.issues
                if issue.severity == "warning"
            ]
            if errors:
                raise ValueError(
                    "导出验证失败：\n"
                    + "\n".join(issue.message for issue in errors[:20])
                )
            if warnings:
                answer = QMessageBox.warning(
                    self,
                    "导出验证警告",
                    "\n".join(issue.message for issue in warnings[:20])
                    + "\n\n仍要导出吗？",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            export_trajectory(
                export_data,
                path,
                recompute_velocities=True,
            )
            self.statusBar().showMessage(f"已导出 {path}")
        except InterruptedError:
            self._apply_current_frame()
            self.statusBar().showMessage("导出前验证已取消")
        except Exception as error:
            QMessageBox.critical(self, "导出失败", str(error))

    def validate(self) -> None:
        data = self._rendered()
        if data is None:
            return
        try:
            report = self._full_validation_report(data)
        except InterruptedError:
            self._apply_current_frame()
            self.statusBar().showMessage("验证已取消")
            return
        text = "验证通过" if report.ok else "\n".join(
            f"[{issue.severity}] {issue.message}" for issue in report.issues[:30]
        )
        QMessageBox.information(self, "轨迹验证", text)

    def _full_validation_report(self, data: TrajectoryData):  # type: ignore[no-untyped-def]
        report = validate_trajectory(data)
        velocity_report = validate_velocity_channels(data)
        report.issues.extend(velocity_report.issues)
        report.metrics.update(velocity_report.metrics)
        source_mapping = ValidationReport()
        if self.timeline_engine is not None:
            if self.adapter is not None and self.project.joint_mapping:
                source_mapping = validate_model_mapping(
                    self.timeline_engine.source,
                    self.adapter,
                    self.project.joint_mapping,
                    self.project.mapping_transforms,
                )
            continuity_report = validate_rendered_operations(
                self.timeline_engine.source,
                self.timeline_engine.operations,
            )
            report.issues.extend(continuity_report.issues)
            report.metrics.update(continuity_report.metrics)
        if self.adapter:
            mapping_report = validate_model_mapping(
                data,
                self.adapter,
                self.project.joint_mapping,
                self.project.mapping_transforms,
            )
            report.issues.extend(
                filter_new_joint_limit_issues(source_mapping, mapping_report)
            )
            progress = QProgressDialog(
                "正在抽样碰撞…", "取消", 0, 100, self
            )
            progress.setWindowTitle("轨迹验证")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(250)

            def update_collision_progress(
                completed: int, total: int
            ) -> bool:
                progress.setValue(
                    int(100 * completed / max(1, total))
                )
                QApplication.processEvents()
                return not progress.wasCanceled()

            try:
                collision_report = sample_collisions(
                    data,
                    self.adapter,
                    self.project.joint_mapping,
                    progress=update_collision_progress,
                    transforms=self.project.mapping_transforms,
                )
            finally:
                progress.close()
            report.issues.extend(collision_report.issues)
            report.metrics.update(collision_report.metrics)
            self._apply_current_frame()
        return report

    def pack(self) -> None:
        if self.project.path is None:
            QMessageBox.information(self, "打包工程", "请先保存可迁移工程")
            return
        path, _ = QFileDialog.getSaveFileName(self, "打包工程", "", "Tar archive (*.tar.gz)")
        if path:
            try:
                pack_project(self.project.path, path)
                self.statusBar().showMessage(f"工程包已生成：{path}")
            except Exception as error:
                QMessageBox.critical(self, "打包失败", str(error))

