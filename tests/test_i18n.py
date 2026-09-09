from __future__ import annotations

import re

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QTableWidget,
)

from studio.ui.i18n import UiLanguageController, language, ui_text
from studio.ui.main_window import MainWindow


HAN = re.compile(r"[\u3400-\u9fff]")


def test_language_defaults_to_chinese(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("MOTION_STUDIO_LANG", raising=False)
    assert language() == "zh"
    assert ui_text("文件") == "文件"
    assert ui_text("左手") == "左手"


def test_language_zh_keeps_chinese(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MOTION_STUDIO_LANG", "zh")
    assert language() == "zh"
    assert ui_text("部件模块") == "部件模块"


def test_dynamic_english_translation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MOTION_STUDIO_LANG", "en")
    assert language() == "en"
    assert ui_text("文件") == "File"
    assert ui_text("左手") == "Left Hand"
    assert ui_text("正在连接真机相机：/dev/video16") == (
        "Connecting to hardware camera：/dev/video16"
    )


def _collect_control_texts(window: MainWindow) -> list[str]:
    texts: list[str] = []
    for widget_type in (QLabel, QAbstractButton):
        for obj in window.findChildren(widget_type):
            texts.append(obj.text())
    for obj in window.findChildren(QLineEdit):
        texts.append(obj.placeholderText())
    for obj in window.findChildren(QComboBox):
        texts.extend(obj.itemText(index) for index in range(obj.count()))
    for obj in window.findChildren(QAction):
        texts.append(obj.text())
    for table in window.findChildren(QTableWidget):
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item is not None:
                    texts.append(item.text())
    return texts


def test_main_window_startup_controls_can_be_chinese(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("MOTION_STUDIO_LANG", raising=False)
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    texts = _collect_control_texts(window)
    chinese = sorted({text for text in texts if HAN.search(text)})
    assert "部件模块" in chinese or any("部件模块" in text for text in texts)
    assert any("打开" == text or "打开" in text for text in texts)
    window.close()


def test_main_window_english_startup_has_no_chinese(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("MOTION_STUDIO_LANG", "en")
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    controller = UiLanguageController(application)
    controller.translate_tree(window)

    texts = _collect_control_texts(window)
    assert not sorted({text for text in texts if HAN.search(text)})
    window.close()
