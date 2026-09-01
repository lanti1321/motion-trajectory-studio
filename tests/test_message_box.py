from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPlainTextEdit

from studio.ui.message_box import QMessageBox, ReadableMessageBox
from studio.ui.style import apply_application_style


def test_normal_message_box_is_wide_and_selectable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    application = QApplication.instance() or QApplication([])
    captured = []

    def fake_exec(box) -> int:  # type: ignore[no-untyped-def]
        captured.append(box)
        return int(QMessageBox.StandardButton.Ok)

    monkeypatch.setattr(ReadableMessageBox, "exec", fake_exec)
    result = QMessageBox.critical(None, "错误", "一条需要完整显示的错误信息")
    assert result == QMessageBox.StandardButton.Ok
    assert captured[0].minimumWidth() >= 520
    assert captured[0].text() == "一条需要完整显示的错误信息"


def test_long_diagnostic_uses_scrollable_copyable_area(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    application = QApplication.instance() or QApplication([])
    captured = []

    def fake_exec(box) -> int:  # type: ignore[no-untyped-def]
        captured.append(box)
        return int(QMessageBox.StandardButton.Ok)

    monkeypatch.setattr(ReadableMessageBox, "exec", fake_exec)
    diagnostic = "\n".join(f"frame {index}: detailed failure" for index in range(30))
    QMessageBox.warning(None, "验证警告", diagnostic)
    editors = captured[0].findChildren(QPlainTextEdit)
    assert len(editors) == 1
    assert editors[0].toPlainText() == diagnostic
    assert editors[0].isReadOnly()


def test_long_diagnostic_summary_obeys_english_startup(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    application = QApplication.instance() or QApplication([])
    captured = []
    monkeypatch.setenv("MOTION_STUDIO_LANG", "en")
    monkeypatch.setattr(
        ReadableMessageBox,
        "exec",
        lambda box: captured.append(box) or int(QMessageBox.StandardButton.Ok),
    )
    QMessageBox.warning(None, "Warning", "\n".join("detail" for _ in range(20)))
    assert captured[0].text().startswith("Detailed diagnostics")


def test_application_style_selects_readable_font() -> None:
    application = QApplication.instance() or QApplication([])
    apply_application_style(application)
    assert application.font().pointSizeF() >= 10.0
    assert "QMessageBox QPushButton" in application.styleSheet()
