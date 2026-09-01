from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class SegmentEditDialog(QDialog):
    """Shared, explicit summary for operations that act on a time selection."""

    def __init__(
        self,
        title: str,
        start: float,
        end: float,
        total_duration: float,
        channels: list[str],
        parameter_label: str,
        parameter_value: float,
        minimum: float,
        maximum: float,
        decimals: int = 3,
        resulting_total: Callable[[float], float] | None = None,
        note: str = "",
        all_channels: bool = False,
        recommended_minimum: float | None = None,
        dynamic_summary: Callable[[float], str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self._resulting_total = resulting_total
        self._dynamic_summary = dynamic_summary

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("框选起点", QLabel(f"{start:.3f}s"))
        form.addRow("框选终点", QLabel(f"{end:.3f}s"))
        form.addRow("选区原时长", QLabel(f"{end - start:.3f}s"))
        scope = (
            "全部位置通道"
            if all_channels or not channels
            else "、".join(channels)
        )
        self.scope_label = QLabel(scope)
        self.scope_label.setWordWrap(True)
        form.addRow("本次作用电机", self.scope_label)
        if recommended_minimum is not None:
            form.addRow(
                "推荐最低时长",
                QLabel(f"{recommended_minimum:.3f}s"),
            )
        self.parameter = QDoubleSpinBox()
        self.parameter.setDecimals(decimals)
        effective_minimum = minimum
        if recommended_minimum is not None:
            effective_minimum = max(minimum, recommended_minimum)
        self.parameter.setRange(effective_minimum, maximum)
        self.parameter.setValue(max(parameter_value, effective_minimum))
        form.addRow(parameter_label, self.parameter)
        self.total_label = QLabel()
        form.addRow("预计轨迹总时长", self.total_label)
        self.delta_label = QLabel()
        form.addRow("预计总轨迹时长变化", self.delta_label)
        layout.addLayout(form)

        self.dynamic_summary_label = QLabel()
        self.dynamic_summary_label.setWordWrap(True)
        if dynamic_summary is not None:
            layout.addWidget(self.dynamic_summary_label)

        if note:
            note_label = QLabel(note)
            note_label.setWordWrap(True)
            layout.addWidget(note_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.parameter.valueChanged.connect(self._refresh_total)
        self._total_duration = total_duration
        self._refresh_total(self.parameter.value())

    def _refresh_total(self, value: float) -> None:
        total = (
            self._resulting_total(value)
            if self._resulting_total is not None
            else self._total_duration
        )
        self.total_label.setText(f"{max(0.0, total):.3f}s")
        delta = total - self._total_duration
        self.delta_label.setText(f"{delta:+.3f}s")
        if self._dynamic_summary is not None:
            self.dynamic_summary_label.setText(self._dynamic_summary(value))

    @property
    def value(self) -> float:
        return float(self.parameter.value())
