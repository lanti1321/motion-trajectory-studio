"""Independent multi-source real-camera monitor window."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Signal
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)

from studio.ui.message_box import QMessageBox
from studio.ui.real_camera import RealCameraView


class CameraPanel(QGroupBox):
    removeRequested = Signal(object)

    def __init__(self, source: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source_edit = QLineEdit(source)
        self.source_edit.setPlaceholderText("/dev/videoN 或 HTTP(S) MJPEG 地址")
        self.view = RealCameraView(self)
        self.view.setMinimumSize(320, 180)
        connect_button = QPushButton("连接")
        connect_button.clicked.connect(self.connect_stream)
        disconnect_button = QPushButton("断开")
        disconnect_button.clicked.connect(self.disconnect_stream)
        remove_button = QPushButton("移除")
        remove_button.clicked.connect(lambda: self.removeRequested.emit(self))

        buttons = QHBoxLayout()
        buttons.addWidget(connect_button)
        buttons.addWidget(disconnect_button)
        buttons.addStretch(1)
        buttons.addWidget(remove_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.source_edit)
        layout.addWidget(self.view, 1)
        layout.addLayout(buttons)
        self.source_edit.textChanged.connect(self._update_title)
        self._update_title(source)

    @property
    def source(self) -> str:
        return self.source_edit.text().strip()

    def _update_title(self, source: str) -> None:
        self.setTitle(source.strip() or "未命名相机")

    def connect_stream(self) -> None:
        if not self.source:
            QMessageBox.information(self, "连接相机", "请输入相机设备或网络流地址")
            return
        self.view.connect_stream(self.source)

    def disconnect_stream(self) -> None:
        self.view.disconnect_stream()


class MultiCameraWindow(QMainWindow):
    """Grid monitor where every source owns an independent capture lifecycle."""

    COLUMNS = 2

    def __init__(
        self,
        sources: Iterable[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("多相机监视器")
        self.resize(1100, 760)
        self._panels: list[CameraPanel] = []

        container = QWidget()
        root = QVBoxLayout(container)
        controls = QHBoxLayout()
        for text, handler in (
            ("添加相机", self.prompt_add_source),
            ("添加检测到的本机相机", self.add_detected_cameras),
            ("全部连接", self.connect_all),
            ("全部断开", self.disconnect_all),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            controls.addWidget(button)
        controls.addStretch(1)
        root.addLayout(controls)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._grid_host)
        root.addWidget(scroll, 1)
        self.setCentralWidget(container)

        for source in sources:
            if str(source).strip():
                self.add_source(str(source).strip())

    @property
    def panels(self) -> tuple[CameraPanel, ...]:
        return tuple(self._panels)

    def prompt_add_source(self) -> None:
        source, ok = QInputDialog.getText(
            self,
            "添加相机",
            "设备或流地址",
            text="/dev/video0",
        )
        if ok and source.strip():
            self.add_source(source.strip())

    @staticmethod
    def detected_sources() -> list[str]:
        return [
            bytes(device.id()).decode(errors="replace")
            for device in QMediaDevices.videoInputs()
        ]

    def add_detected_cameras(self) -> None:
        detected = self.detected_sources()
        if not detected:
            QMessageBox.information(self, "多相机监视器", "未检测到本机相机")
            return
        for source in detected:
            self.add_source(source)

    def add_source(self, source: str) -> CameraPanel:
        normalized = source.strip()
        existing = next(
            (panel for panel in self._panels if panel.source == normalized),
            None,
        )
        if existing is not None:
            return existing
        panel = CameraPanel(normalized, self._grid_host)
        panel.removeRequested.connect(self.remove_panel)
        self._panels.append(panel)
        self._reflow()
        return panel

    def remove_panel(self, panel: CameraPanel) -> None:
        if panel not in self._panels:
            return
        panel.disconnect_stream()
        self._panels.remove(panel)
        panel.setParent(None)
        panel.deleteLater()
        self._reflow()

    def _reflow(self) -> None:
        while self._grid.count():
            self._grid.takeAt(0)
        for index, panel in enumerate(self._panels):
            self._grid.addWidget(
                panel, index // self.COLUMNS, index % self.COLUMNS
            )
        for column in range(self.COLUMNS):
            self._grid.setColumnStretch(column, 1)

    def connect_all(self) -> None:
        for panel in self._panels:
            panel.connect_stream()

    def disconnect_all(self) -> None:
        for panel in self._panels:
            panel.disconnect_stream()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.disconnect_all()
        super().closeEvent(event)

