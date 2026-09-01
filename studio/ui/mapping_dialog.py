from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from studio.models.adapter import ModelInfo, auto_map_channels


class JointMappingDialog(QDialog):
    def __init__(
        self,
        channels: list[str],
        model: ModelInfo,
        existing: dict[str, str] | None = None,
        transforms: dict[str, dict[str, float]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("轨迹通道映射")
        self.resize(620, 560)
        self._combos: dict[str, QComboBox] = {}
        self._scales: dict[str, QDoubleSpinBox] = {}
        self._offsets: dict[str, QDoubleSpinBox] = {}
        self._limits: dict[str, dict[str, float]] = {}
        suggestions = auto_map_channels(channels, model)
        suggestions.update(existing or {})
        scalar_joints = [
            joint.name
            for joint in model.joints
            if joint.qpos_width == 1 and joint.independent
        ]

        root = QVBoxLayout(self)
        root.addWidget(QLabel("将每个位置通道映射到模型关节。未映射通道不会驱动模型。"))
        content = QWidget()
        form = QFormLayout(content)
        for channel in channels:
            combo = QComboBox()
            combo.addItem("未映射", None)
            for joint in scalar_joints:
                combo.addItem(joint, joint)
            suggested = suggestions.get(channel)
            if suggested:
                index = combo.findData(suggested)
                if index >= 0:
                    combo.setCurrentIndex(index)
            self._combos[channel] = combo
            transform = (transforms or {}).get(channel, {})
            scale = QDoubleSpinBox()
            scale.setRange(-1_000_000, 1_000_000)
            scale.setDecimals(8)
            scale.setValue(float(transform.get("scale", 1.0)))
            offset = QDoubleSpinBox()
            offset.setRange(-1_000_000, 1_000_000)
            offset.setDecimals(8)
            offset.setValue(float(transform.get("offset", 0.0)))
            self._scales[channel] = scale
            self._offsets[channel] = offset
            self._limits[channel] = {
                key: float(transform[key])
                for key in ("minimum", "maximum")
                if key in transform
            }
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(combo, 2)
            row_layout.addWidget(QLabel("scale"))
            row_layout.addWidget(scale, 1)
            row_layout.addWidget(QLabel("offset"))
            row_layout.addWidget(offset, 1)
            form.addRow(channel, row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def mapping(self) -> dict[str, str]:
        return {
            channel: str(combo.currentData())
            for channel, combo in self._combos.items()
            if combo.currentData()
        }

    def transforms(self) -> dict[str, dict[str, float]]:
        result = {
            channel: {
                "scale": self._scales[channel].value(),
                "offset": self._offsets[channel].value(),
                **self._limits[channel],
            }
            for channel in self.mapping()
        }
        return result
