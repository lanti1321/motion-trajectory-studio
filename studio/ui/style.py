"""Application-wide font and compact, readable Qt styling."""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


PREFERRED_FONTS = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei UI",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
)


def apply_application_style(application: QApplication) -> None:
    families = set(QFontDatabase.families())
    family = next(
        (candidate for candidate in PREFERRED_FONTS if candidate in families),
        application.font().family(),
    )
    font = QFont(family)
    font.setPointSizeF(10.5)
    application.setFont(font)
    application.setStyleSheet(
        """
        QPushButton { padding: 4px 9px; min-height: 20px; }
        QToolButton { padding: 3px 6px; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 24px; }
        QMessageBox QPushButton { min-width: 82px; }
        QPlainTextEdit { font-family: monospace; font-size: 10pt; }
        QToolTip { padding: 4px 6px; }
        """
    )
