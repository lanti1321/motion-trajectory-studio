from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from studio.hardware.openarm_replay import (
    MotorTemperatureSample,
    is_motor_fault_code,
    motor_fault_text,
)
from studio.ui.i18n import ui_text

MOTOR_NAMES = [
    *(f"左臂 J{i}" for i in range(1, 8)),
    "左夹爪",
    *(f"右臂 J{i}" for i in range(1, 8)),
    "右夹爪",
]


class TemperatureMonitor(QWidget):
    def __init__(self, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.detection_mode = QCheckBox("温度检测模式（强制从头完整执行并记录曲线）")
        self.high_dynamics_mode = QCheckBox(
            "高动态真机模式（摇酒等高速动作：3.5 rad/s，60 rad/s²）"
        )
        self.status = QLabel("等待真机温度遥测")
        self.table = QTableWidget(16, 4)
        self.table.setHorizontalHeaderLabels(
            ["关节", "整体/MOS ℃", "转子 ℃", "状态"]
        )
        self.table.verticalHeader().setVisible(False)
        for row, name in enumerate(MOTOR_NAMES):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem("--"))
            self.table.setItem(row, 2, QTableWidgetItem("--"))
            self.table.setItem(row, 3, QTableWidgetItem("--"))
        self.table.resizeColumnsToContents()
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.detection_mode)
        layout.addWidget(self.high_dynamics_mode)
        layout.addWidget(self.status)
        layout.addWidget(self.table, 1)

    def update_sample(self, sample: MotorTemperatureSample) -> None:
        hottest = max(max(sample.mos), max(sample.rotor))
        for row in range(16):
            self.table.item(row, 1).setText(str(sample.mos[row]))
            self.table.item(row, 2).setText(str(sample.rotor[row]))
            error = sample.errors[row]
            fault = is_motor_fault_code(error)
            self.table.item(row, 3).setText(
                f"故障 {error}（{motor_fault_text(error)}）" if fault else "正常"
            )
            color = QColor(
                "#d32f2f"
                if fault or max(sample.mos[row], sample.rotor[row]) >= 70
                else "#202020"
            )
            for column in range(1, 4):
                self.table.item(row, column).setForeground(color)
        self.status.setText(f"实时遥测 t={sample.time:.2f}s，最高温度 {hottest}℃")


class TemperatureCurveWidget(QWidget):
    def __init__(self, samples: list[MotorTemperatureSample], parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.samples = samples
        self.setMinimumSize(1200, 820)

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        if len(self.samples) < 2:
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                ui_text("没有足够的温度样本"),
            )
            return
        columns, rows, margin = 4, 4, 8.0
        cell_width = self.width() / columns
        cell_height = self.height() / rows
        times = [sample.time for sample in self.samples]
        start, span = times[0], max(times[-1] - times[0], 1e-6)
        for motor in range(16):
            column, row = motor % columns, motor // columns
            left = column * cell_width + margin
            top = row * cell_height + margin
            right = (column + 1) * cell_width - margin
            bottom = (row + 1) * cell_height - margin
            painter.setPen(QPen(QColor("#bbbbbb"), 1))
            painter.drawRect(int(left), int(top), int(right - left), int(bottom - top))
            mos = [sample.mos[motor] for sample in self.samples]
            rotor = [sample.rotor[motor] for sample in self.samples]
            low = min(mos + rotor) - 2
            high = max(max(mos + rotor) + 2, low + 5)
            plot_top, plot_bottom = top + 28, bottom - 20
            painter.setPen(QColor("#202020"))
            painter.drawText(
                int(left + 5), int(top + 16),
                ui_text(f"{MOTOR_NAMES[motor]}  MOS {mos[-1]}℃ / 转子 {rotor[-1]}℃"),
            )
            for values, color in ((mos, "#e65100"), (rotor, "#1565c0")):
                points = [
                    QPointF(
                        left + 34 + (times[index] - start) / span * (right - left - 42),
                        plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top),
                    )
                    for index, value in enumerate(values)
                ]
                painter.setPen(QPen(QColor(color), 1.5))
                painter.drawPolyline(points)
            painter.setPen(QColor("#666666"))
            painter.drawText(int(left + 4), int(plot_top + 4), f"{high}℃")
            painter.drawText(int(left + 4), int(plot_bottom), f"{low}℃")


class TemperatureCurveDialog(QDialog):
    def __init__(self, samples: list[MotorTemperatureSample], parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("完整轨迹温度曲线（橙色 MOS / 蓝色转子）")
        self.resize(1250, 900)
        layout = QVBoxLayout(self)
        layout.addWidget(TemperatureCurveWidget(samples, self))
