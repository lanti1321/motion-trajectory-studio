from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from studio.ui.multi_camera import MultiCameraWindow


def test_multi_camera_window_adds_deduplicates_and_reflows() -> None:
    application = QApplication.instance() or QApplication([])
    window = MultiCameraWindow(["/dev/video0", "/dev/video1"])
    assert [panel.source for panel in window.panels] == [
        "/dev/video0", "/dev/video1"
    ]
    duplicate = window.add_source("/dev/video0")
    assert duplicate is window.panels[0]
    assert len(window.panels) == 2
    third = window.add_source("https://127.0.0.1/cam.mjpg")
    assert window._grid.indexOf(third) >= 0
    window.close()


def test_multi_camera_disconnects_each_panel(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    application = QApplication.instance() or QApplication([])
    window = MultiCameraWindow(["/dev/video0", "/dev/video1"])
    disconnected: list[str] = []
    for panel in window.panels:
        monkeypatch.setattr(
            panel,
            "disconnect_stream",
            lambda source=panel.source: disconnected.append(source),
        )
    window.disconnect_all()
    assert disconnected == ["/dev/video0", "/dev/video1"]
    window.close()
