from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from studio.models.adapter import JointInfo
from studio.models.pose import finger_subgroups, joint_display_name
from studio.ui.i18n import ui_text


SLIDER_STEPS = 1000


class JointPosePanel(QWidget):
    """Sliders for a joint subset; emits radians."""

    previewChanged = Signal(str, float)
    poseCommitted = Signal(str, float)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        show_hint: bool = False,
    ) -> None:
        super().__init__(parent)
        self._joints: dict[str, JointInfo] = {}
        self._sliders: dict[str, QSlider] = {}
        self._spinboxes: dict[str, QDoubleSpinBox] = {}
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._hint = QLabel(
            ui_text("拖动滑条可预览；松开后写入当前轨迹。")
        )
        self._hint.setWordWrap(True)
        self._hint.setVisible(show_hint)
        layout.addWidget(self._hint)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._body)
        self._empty = QLabel(ui_text("请先加载模型"))
        self._empty.setWordWrap(True)
        self._body_layout.addWidget(self._empty)
        self._body_layout.addStretch(1)

    def clear(self) -> None:
        self.set_joints([])

    def set_joints(self, joints: list[JointInfo]) -> None:
        self._joints = {joint.name: joint for joint in joints}
        self._sliders.clear()
        self._spinboxes.clear()
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not joints:
            self._empty = QLabel(ui_text("请先加载模型"))
            self._empty.setWordWrap(True)
            self._body_layout.addWidget(self._empty)
            self._body_layout.addStretch(1)
            return
        named = {joint.name: joint for joint in joints}
        for group in finger_subgroups([joint.name for joint in joints]):
            members = [
                named[name]
                for name in group["joint_names"]  # type: ignore[index]
                if name in named
            ]
            if not members:
                continue
            title = str(group.get("title") or "")
            host: QWidget
            if title:
                host = QGroupBox(ui_text(title))
            else:
                host = QWidget()
            form = QFormLayout(host)
            form.setContentsMargins(6, 8, 6, 6)
            for joint in members:
                form.addRow(self._joint_label(joint), self._make_row(joint))
            self._body_layout.addWidget(host)
        self._body_layout.addStretch(1)

    def set_values(
        self,
        values: dict[str, float],
        skip: str | None = None,
    ) -> None:
        self._updating = True
        try:
            for name, radians in values.items():
                if name == skip or name not in self._joints:
                    continue
                self._set_row(name, radians)
        finally:
            self._updating = False

    def current_values(self) -> dict[str, float]:
        return {
            name: self._spin_to_radians(name, spin.value())
            for name, spin in self._spinboxes.items()
        }

    def _joint_label(self, joint: JointInfo) -> QLabel:
        label = QLabel(ui_text(joint_display_name(joint.name)))
        label.setToolTip(joint.name)
        label.setWordWrap(True)
        return label

    def _make_row(self, joint: JointInfo) -> QWidget:
        lower, upper = self._joint_range(joint)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, SLIDER_STEPS)
        slider.setSingleStep(1)
        slider.setPageStep(max(1, SLIDER_STEPS // 20))
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        if joint.joint_type == "hinge":
            spin.setRange(math.degrees(lower), math.degrees(upper))
            spin.setSuffix(" °")
            spin.setSingleStep(1.0)
        else:
            spin.setRange(lower, upper)
            spin.setSuffix(" m" if joint.joint_type == "slide" else "")
            spin.setSingleStep(0.001)
        slider.valueChanged.connect(
            lambda ticks, name=joint.name: self._slider_moved(name, ticks)
        )
        slider.sliderReleased.connect(
            lambda name=joint.name: self._commit(name)
        )
        spin.valueChanged.connect(
            lambda displayed, name=joint.name: self._spin_moved(name, displayed)
        )
        spin.editingFinished.connect(lambda name=joint.name: self._commit(name))
        self._sliders[joint.name] = slider
        self._spinboxes[joint.name] = spin
        layout.addWidget(slider, 1)
        layout.addWidget(spin)
        self._updating = True
        try:
            self._set_row(joint.name, 0.0)
        finally:
            self._updating = False
        return row

    def _slider_moved(self, joint_name: str, ticks: int) -> None:
        if self._updating:
            return
        radians = self._ticks_to_radians(joint_name, ticks)
        self._updating = True
        try:
            self._spinboxes[joint_name].setValue(
                self._radians_to_spin(joint_name, radians)
            )
        finally:
            self._updating = False
        self.previewChanged.emit(joint_name, radians)

    def _spin_moved(self, joint_name: str, displayed: float) -> None:
        if self._updating:
            return
        radians = self._spin_to_radians(joint_name, displayed)
        self._updating = True
        try:
            self._sliders[joint_name].setValue(
                self._radians_to_ticks(joint_name, radians)
            )
        finally:
            self._updating = False
        self.previewChanged.emit(joint_name, radians)

    def _commit(self, joint_name: str) -> None:
        if joint_name not in self._spinboxes:
            return
        self.poseCommitted.emit(
            joint_name,
            self._spin_to_radians(joint_name, self._spinboxes[joint_name].value()),
        )

    def _set_row(self, joint_name: str, radians: float) -> None:
        slider = self._sliders.get(joint_name)
        spin = self._spinboxes.get(joint_name)
        if slider is None or spin is None:
            return
        slider.setValue(self._radians_to_ticks(joint_name, radians))
        spin.setValue(self._radians_to_spin(joint_name, radians))

    def _joint_range(self, joint: JointInfo) -> tuple[float, float]:
        lower = joint.lower if joint.lower is not None else -math.pi
        upper = joint.upper if joint.upper is not None else math.pi
        if upper <= lower:
            upper = lower + 1e-6
        return lower, upper

    def _ticks_to_radians(self, joint_name: str, ticks: int) -> float:
        lower, upper = self._joint_range(self._joints[joint_name])
        ratio = max(0.0, min(1.0, ticks / SLIDER_STEPS))
        return lower + ratio * (upper - lower)

    def _radians_to_ticks(self, joint_name: str, radians: float) -> int:
        lower, upper = self._joint_range(self._joints[joint_name])
        ratio = (radians - lower) / (upper - lower)
        return int(round(max(0.0, min(1.0, ratio)) * SLIDER_STEPS))

    def _radians_to_spin(self, joint_name: str, radians: float) -> float:
        joint = self._joints[joint_name]
        if joint.joint_type == "hinge":
            return math.degrees(radians)
        return radians

    def _spin_to_radians(self, joint_name: str, displayed: float) -> float:
        joint = self._joints[joint_name]
        if joint.joint_type == "hinge":
            return math.radians(displayed)
        return displayed
