from __future__ import annotations

from collections.abc import Callable, Sequence
import uuid

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from studio.models.adapter import JointInfo
from studio.models.pose import part_modules_from_joints
from studio.ui.i18n import ui_text
from studio.ui.joint_pose_panel import JointPosePanel


class PartModuleCatalog(QWidget):
    openRequested = Signal(str)
    createRequested = Signal()
    deleteRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._hint = QLabel(ui_text("拖动滑条可预览；松开后写入当前轨迹。"))
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        self._create = QPushButton(ui_text("新建自定义模块"))
        self._create.clicked.connect(self.createRequested.emit)
        layout.addWidget(self._create)
        self._list = QWidget()
        self._list_layout = QVBoxLayout(self._list)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)
        layout.addWidget(self._list)
        self._empty = QLabel(ui_text("请先加载模型"))
        self._empty.setWordWrap(True)
        layout.addWidget(self._empty)
        layout.addStretch(1)
        self._modules: list[dict[str, object]] = []

    def set_modules(self, modules: Sequence[dict[str, object]]) -> None:
        self._modules = [dict(module) for module in modules]
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._empty.setVisible(not self._modules)
        for module in self._modules:
            module_id = str(module["id"])
            title = ui_text(str(module["title"]))
            count = len(list(module["joint_names"]))  # type: ignore[arg-type]
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            label = QPushButton(f"{title} ({count})")
            label.setFlat(True)
            label.setStyleSheet("text-align: left; padding-left: 0;")
            button = QPushButton(ui_text("打开"))
            button.setObjectName(f"open_module_{module_id}")
            for widget in (label, button):
                widget.clicked.connect(
                    lambda _checked=False, mid=module_id: self.openRequested.emit(mid)
                )
            row_layout.addWidget(label, 1)
            row_layout.addWidget(button)
            if bool(module.get("custom")):
                delete = QPushButton(ui_text("删除"))
                delete.setObjectName(f"delete_module_{module_id}")
                delete.clicked.connect(
                    lambda _checked=False, mid=module_id:
                    self.deleteRequested.emit(mid)
                )
                row_layout.addWidget(delete)
            self._list_layout.addWidget(row)


class CustomPartModuleDialog(QDialog):
    def __init__(
        self,
        joints: Sequence[JointInfo],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(ui_text("新建自定义模块"))
        self.resize(420, 560)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(ui_text("模块名称")))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(ui_text("例如：双臂肩关节"))
        layout.addWidget(self.name_edit)
        layout.addWidget(QLabel(ui_text("选择模块包含的电机")))
        self.joint_list = QListWidget()
        for joint in joints:
            item = QListWidgetItem(joint.name)
            item.setData(Qt.ItemDataRole.UserRole, joint.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.joint_list.addItem(item)
        layout.addWidget(self.joint_list, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_joint_names(self) -> list[str]:
        return [
            str(item.data(Qt.ItemDataRole.UserRole))
            for index in range(self.joint_list.count())
            if (
                (item := self.joint_list.item(index)).checkState()
                == Qt.CheckState.Checked
            )
        ]

    def accept(self) -> None:
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, ui_text("新建自定义模块"), ui_text("请输入模块名称"))
            return
        if not self.selected_joint_names():
            QMessageBox.warning(
                self,
                ui_text("新建自定义模块"),
                ui_text("请至少选择一个电机"),
            )
            return
        super().accept()

    def module_definition(self) -> dict[str, object]:
        return {
            "id": f"custom_{uuid.uuid4().hex}",
            "title": self.name_edit.text().strip(),
            "joint_names": self.selected_joint_names(),
        }


class PartInspectorWindow(QDialog):
    closedModule = Signal(str)

    def __init__(
        self,
        module: dict[str, object],
        joints: list[JointInfo],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.module_id = str(module["id"])
        self.setWindowTitle(ui_text(str(module["title"])))
        self.setModal(False)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        joint_count = max(1, len(joints))
        self.setMinimumWidth(360)
        self.setMaximumWidth(420)
        self.resize(400, min(720, 90 + 54 * joint_count))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.panel = JointPosePanel(self, show_hint=True)
        layout.addWidget(self.panel)
        self.panel.set_joints(joints)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.closedModule.emit(self.module_id)
        super().closeEvent(event)


class PartModuleHost:
    def __init__(
        self,
        parent: QWidget,
        preview: Callable[[str, float], None],
        commit: Callable[[str, float], None],
    ) -> None:
        self._parent = parent
        self._preview = preview
        self._commit = commit
        self.catalog = PartModuleCatalog(parent)
        self.catalog.openRequested.connect(self.open_module)
        self._joints: dict[str, JointInfo] = {}
        self._modules: list[dict[str, object]] = []
        self._windows: dict[str, PartInspectorWindow] = {}

    @property
    def modules(self) -> list[dict[str, object]]:
        return list(self._modules)

    def window_for(self, module_id: str) -> PartInspectorWindow | None:
        return self._windows.get(module_id)

    def open_windows(self) -> list[PartInspectorWindow]:
        return list(self._windows.values())

    def clear(self) -> None:
        self.set_joints([])

    def set_joints(
        self,
        joints: Sequence[JointInfo],
        custom_modules: Sequence[dict[str, object]] = (),
    ) -> None:
        self._joints = {joint.name: joint for joint in joints}
        self._modules = part_modules_from_joints(list(joints))
        known_ids = {str(module["id"]) for module in self._modules}
        for custom in custom_modules:
            module_id = str(custom.get("id") or "")
            title = str(custom.get("title") or "").strip()
            joint_names = [
                str(name)
                for name in custom.get("joint_names", [])  # type: ignore[union-attr]
                if str(name) in self._joints
            ]
            if not module_id or not title or not joint_names or module_id in known_ids:
                continue
            self._modules.append(
                {
                    "id": module_id,
                    "title": title,
                    "joint_names": joint_names,
                    "custom": True,
                }
            )
            known_ids.add(module_id)
        self.catalog.set_modules(self._modules)
        known = {str(module["id"]) for module in self._modules}
        for module_id in list(self._windows):
            if module_id not in known:
                window = self._windows.pop(module_id)
                window.close()
                continue
            module = next(item for item in self._modules if item["id"] == module_id)
            self._windows[module_id].panel.set_joints(self._module_joints(module))

    def open_module(self, module_id: str) -> PartInspectorWindow | None:
        existing = self._windows.get(module_id)
        if existing is not None:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing
        module = next(
            (item for item in self._modules if item["id"] == module_id),
            None,
        )
        if module is None:
            return None
        window = PartInspectorWindow(
            module, self._module_joints(module), self._parent
        )
        window.panel.previewChanged.connect(self._preview)
        window.panel.poseCommitted.connect(self._commit)
        window.closedModule.connect(self._forget_window)
        self._windows[module_id] = window
        window.show()
        window.raise_()
        window.activateWindow()
        return window

    def set_values(
        self,
        values: dict[str, float],
        skip: str | None = None,
    ) -> None:
        for window in self._windows.values():
            window.panel.set_values(values, skip=skip)

    def close_all(self) -> None:
        for window in list(self._windows.values()):
            window.close()
        self._windows.clear()

    def _module_joints(self, module: dict[str, object]) -> list[JointInfo]:
        return [
            self._joints[name]
            for name in list(module["joint_names"])  # type: ignore[arg-type]
            if name in self._joints
        ]

    def _forget_window(self, module_id: str) -> None:
        self._windows.pop(module_id, None)
