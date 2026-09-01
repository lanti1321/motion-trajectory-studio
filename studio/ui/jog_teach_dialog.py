from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class JogTeachDialog(QDialog):
    jog = Signal(float, float, float, float, float, float)
    commitRequested = Signal()
    cancelRequested = Signal()

    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("末端 6D 按钮示教")
        self.setMinimumWidth(430)
        self._active_delta: tuple[float, ...] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._repeat)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.translation_step = QDoubleSpinBox()
        self.translation_step.setRange(0.1, 20.0)
        self.translation_step.setValue(1.0)
        self.translation_step.setSuffix(" mm")
        self.rotation_step = QDoubleSpinBox()
        self.rotation_step.setRange(0.1, 10.0)
        self.rotation_step.setValue(1.0)
        self.rotation_step.setSuffix(" °")
        form.addRow("平移步长", self.translation_step)
        form.addRow("旋转步长", self.rotation_step)
        layout.addLayout(form)
        layout.addWidget(QLabel("按一下单步点动；按住按钮连续点动。坐标相对所选基座。"))
        translation_group = QGroupBox("平移 XYZ")
        translation_grid = QGridLayout(translation_group)
        rotation_group = QGroupBox("旋转 Rx / Ry / Rz")
        rotation_grid = QGridLayout(rotation_group)

        def add_row(grid: QGridLayout, row: int, axis: str,
                    negative: tuple[float, ...], positive: tuple[float, ...],
                    negative_text: str, positive_text: str) -> None:
            grid.addWidget(QLabel(axis), row, 1)
            for column, text, delta in (
                (0, negative_text, negative), (2, positive_text, positive)
            ):
                button = QPushButton(text)
                button.pressed.connect(lambda value=delta: self._start(value))
                button.released.connect(self._stop)
                grid.addWidget(button, row, column)

        add_row(translation_grid, 0, "X", (-1, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0), "后 −X", "前 +X")
        add_row(translation_grid, 1, "Y", (0, -1, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0), "右 −Y", "左 +Y")
        add_row(translation_grid, 2, "Z", (0, 0, -1, 0, 0, 0),
                (0, 0, 1, 0, 0, 0), "下 −Z", "上 +Z")
        add_row(rotation_grid, 0, "Rx", (0, 0, 0, -1, 0, 0),
                (0, 0, 0, 1, 0, 0), "Rx −", "Rx +")
        add_row(rotation_grid, 1, "Ry", (0, 0, 0, 0, -1, 0),
                (0, 0, 0, 0, 1, 0), "Ry −", "Ry +")
        add_row(rotation_grid, 2, "Rz", (0, 0, 0, 0, 0, -1),
                (0, 0, 0, 0, 0, 1), "Rz −", "Rz +")
        for group in (translation_group, rotation_group):
            layout.addWidget(group)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("提交示教")
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.commitRequested.emit
        )
        buttons.rejected.connect(self.cancelRequested.emit)
        layout.addWidget(buttons)

    def _scaled(self, delta: tuple[float, ...]) -> tuple[float, ...]:
        translation = self.translation_step.value() / 1000.0
        rotation = self.rotation_step.value()
        return tuple(
            value * (translation if index < 3 else rotation)
            for index, value in enumerate(delta)
        )

    def _start(self, delta: tuple[float, ...]) -> None:
        self._active_delta = delta
        self.jog.emit(*self._scaled(delta))
        self._timer.start()

    def _repeat(self) -> None:
        if self._active_delta is not None:
            self.jog.emit(*self._scaled(self._active_delta))

    def _stop(self) -> None:
        self._timer.stop()
        self._active_delta = None

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._stop()
        self.cancelRequested.emit()
        super().closeEvent(event)
