"""Readable drop-in replacement for the static QMessageBox helpers."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QLabel,
    QMessageBox as QtMessageBox,
    QPlainTextEdit,
    QSizePolicy,
)


class ReadableMessageBox(QtMessageBox):
    """Message box that wraps normal text and scrolls long diagnostics."""

    LONG_TEXT_LENGTH = 480
    LONG_TEXT_LINES = 5

    @staticmethod
    def _details_summary() -> str:
        language = os.environ.get("MOTION_STUDIO_LANG", "zh").lower()
        if language.startswith("en"):
            return "Detailed diagnostics are shown below. Scroll to review or copy them:"
        return "详细诊断信息如下，可滚动查看并复制："

    @classmethod
    def _show(
        cls,
        icon: QtMessageBox.Icon,
        parent,  # type: ignore[no-untyped-def]
        title: str,
        text: str,
        buttons: QtMessageBox.StandardButton = QtMessageBox.StandardButton.Ok,
        default_button: QtMessageBox.StandardButton = QtMessageBox.StandardButton.NoButton,
    ) -> QtMessageBox.StandardButton:
        box = cls(parent)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setStandardButtons(buttons)
        if default_button != QtMessageBox.StandardButton.NoButton:
            box.setDefaultButton(default_button)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        message = str(text)
        if (
            len(message) > cls.LONG_TEXT_LENGTH
            or message.count("\n") >= cls.LONG_TEXT_LINES
        ):
            box.setText(cls._details_summary())
            details = QPlainTextEdit(message, box)
            details.setReadOnly(True)
            details.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
            details.setMinimumSize(680, 260)
            details.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            layout = box.layout()
            if layout is not None:
                layout.addWidget(details, layout.rowCount(), 0, 1, layout.columnCount())
        else:
            box.setText(message)
        box._apply_readable_geometry()
        return QtMessageBox.StandardButton(box.exec())

    def _apply_readable_geometry(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        available_width = screen.availableGeometry().width() if screen else 1200
        target_width = max(600, min(900, int(available_width * 0.72)))
        self.setMinimumWidth(target_width)
        self.setMaximumWidth(max(target_width, int(available_width * 0.88)))
        for label in self.findChildren(QLabel):
            # QMessageBox implements its icon as a QLabel too.  Applying the
            # message text's 360px minimum width to it created the enormous red
            # icon gutter seen on Linux and left almost no space for diagnostics.
            pixmap = label.pixmap()
            has_pixmap = pixmap is not None and not pixmap.isNull()
            if label.objectName() == "qt_msgboxex_icon_label" or has_pixmap:
                label.setMinimumSize(36, 36)
                label.setMaximumSize(36, 36)
                label.setScaledContents(True)
                continue
            label.setWordWrap(True)
            label.setTextInteractionFlags(
                label.textInteractionFlags()
                | Qt.TextInteractionFlag.TextSelectableByMouse
            )
            label.setMinimumWidth(max(440, target_width - 100))
            label.setMaximumWidth(max(500, int(available_width * 0.76)))
        self.adjustSize()

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        self._apply_readable_geometry()

    @classmethod
    def information(cls, parent, title, text, buttons=QtMessageBox.StandardButton.Ok,
                    defaultButton=QtMessageBox.StandardButton.NoButton):  # type: ignore[no-untyped-def]
        return cls._show(cls.Icon.Information, parent, title, text, buttons, defaultButton)

    @classmethod
    def warning(cls, parent, title, text, buttons=QtMessageBox.StandardButton.Ok,
                defaultButton=QtMessageBox.StandardButton.NoButton):  # type: ignore[no-untyped-def]
        return cls._show(cls.Icon.Warning, parent, title, text, buttons, defaultButton)

    @classmethod
    def critical(cls, parent, title, text, buttons=QtMessageBox.StandardButton.Ok,
                 defaultButton=QtMessageBox.StandardButton.NoButton):  # type: ignore[no-untyped-def]
        return cls._show(cls.Icon.Critical, parent, title, text, buttons, defaultButton)

    @classmethod
    def question(cls, parent, title, text,
                 buttons=QtMessageBox.StandardButton.Yes | QtMessageBox.StandardButton.No,
                 defaultButton=QtMessageBox.StandardButton.NoButton):  # type: ignore[no-untyped-def]
        return cls._show(cls.Icon.Question, parent, title, text, buttons, defaultButton)


# Existing modules intentionally import this alias, retaining the familiar API.
QMessageBox = ReadableMessageBox
