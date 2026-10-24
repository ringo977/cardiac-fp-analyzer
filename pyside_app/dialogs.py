"""Message boxes drawn by Qt instead of the native macOS alert.

On macOS 27 (Qt 6.11) the native ``NSAlert`` behind ``QMessageBox.question`` aborted the process while
rendering its icon (an Objective-C exception in CoreUI/SwiftUI, 24 Oct 2026: 'Rifai l'analisi' confirmation).
``install()`` replaces the four static helpers of ``QMessageBox`` with equivalents of the same signature that
build a ``QMessageBox`` with ``Option.DontUseNativeDialog``; every existing call site keeps working. File
dialogs stay native (they are not affected).
"""
from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

_NATIVE = {}


def _show(icon, parent, title, text, buttons, default):
    box = QMessageBox(parent)
    box.setOption(QMessageBox.Option.DontUseNativeDialog, True)
    box.setIcon(icon)
    box.setWindowTitle(str(title))
    box.setText(str(text))
    box.setStandardButtons(buttons)
    if default != QMessageBox.NoButton:
        box.setDefaultButton(default)
    box.exec()
    clicked = box.clickedButton()
    return box.standardButton(clicked) if clicked is not None else QMessageBox.NoButton


def question(parent, title, text, buttons=QMessageBox.Yes | QMessageBox.No, defaultButton=QMessageBox.NoButton):  # noqa: N803
    return _show(QMessageBox.Question, parent, title, text, buttons, defaultButton)


def information(parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):  # noqa: N803
    return _show(QMessageBox.Information, parent, title, text, buttons, defaultButton)


def warning(parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):  # noqa: N803
    return _show(QMessageBox.Warning, parent, title, text, buttons, defaultButton)


def critical(parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):  # noqa: N803
    return _show(QMessageBox.Critical, parent, title, text, buttons, defaultButton)


def install() -> None:
    """Route ``QMessageBox.question/information/warning/critical`` through Qt-drawn boxes (idempotent)."""
    if _NATIVE:
        return
    for name, fn in (("question", question), ("information", information), ("warning", warning), ("critical", critical)):
        _NATIVE[name] = getattr(QMessageBox, name)
        setattr(QMessageBox, name, staticmethod(fn))


def uninstall() -> None:
    for name, fn in _NATIVE.items():
        setattr(QMessageBox, name, fn)
    _NATIVE.clear()


__all__ = ["install", "uninstall", "question", "information", "warning", "critical"]
