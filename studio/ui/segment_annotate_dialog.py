from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

SEGMENT_COLOR_PRESETS = [
    ("蓝", "#4fc3f7"),
    ("绿", "#81c784"),
    ("橙", "#ffb74d"),
    ("紫", "#ba68c8"),
    ("红", "#e57373"),
    ("青", "#4dd0e1"),
    ("黄", "#fff176"),
    ("粉", "#f06292"),
]


class SegmentAnnotateDialog(QDialog):
    def __init__(
        self,
        start: float,
        end: float,
        title: str = "",
        color: str = SEGMENT_COLOR_PRESETS[0][1],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("动作分段标注")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("起点", QLabel(f"{start:.3f}s"))
        form.addRow("终点", QLabel(f"{end:.3f}s"))
        form.addRow("时长", QLabel(f"{end - start:.3f}s"))
        self.title = QLineEdit(title)
        self.title.setPlaceholderText("例如：抓取、放置、回零")
        form.addRow("分段名称", self.title)
        self.color = QComboBox()
        for label, value in SEGMENT_COLOR_PRESETS:
            self.color.addItem(label, value)
        index = next(
            (
                index
                for index, (_, value) in enumerate(SEGMENT_COLOR_PRESETS)
                if value.lower() == color.lower()
            ),
            -1,
        )
        if index < 0:
            self.color.addItem("自定义", color)
            index = self.color.count() - 1
        self.color.setCurrentIndex(index)
        form.addRow("颜色", self.color)
        custom_color = QPushButton("选择自定义颜色…")
        custom_color.clicked.connect(self._choose_custom_color)
        form.addRow("", custom_color)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def segment_title(self) -> str:
        return self.title.text().strip()

    @property
    def segment_color(self) -> str:
        return str(self.color.currentData())

    def preview_color(self) -> QColor:
        return QColor(self.segment_color)

    def _choose_custom_color(self) -> None:
        selected = QColorDialog.getColor(
            QColor(self.segment_color), self, "选择区间颜色"
        )
        if not selected.isValid():
            return
        value = selected.name()
        custom_index = next(
            (
                index
                for index in range(self.color.count())
                if self.color.itemText(index) == "自定义"
            ),
            -1,
        )
        if custom_index < 0:
            self.color.addItem("自定义", value)
            custom_index = self.color.count() - 1
        else:
            self.color.setItemData(custom_index, value)
        self.color.setCurrentIndex(custom_index)
