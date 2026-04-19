"""PoC PySide6 entry point for cardiac-fp-analyzer (ADR-0001, Phase 0).

Minimal main window: menu + toolbar (picking mode) + signal viewer.

Run from the repo root:

    python -m pyside_app.main
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QRadioButton, QStatusBar, QVBoxLayout, QWidget,
)

from pyside_app.signal_viewer import SignalViewer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("cardiac-fp-analyzer — PoC PySide6")
        self.resize(1400, 800)

        # ─── central layout ─────────────────────────────────────────
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # Toolbar row: picking mode
        # Two modes only:
        # - Visualizza: read-only, clicks do nothing (zoom/pan still work).
        # - Edit: a left-click opens a contextual popup with the actions that
        #   make sense for that click (see SignalViewer._show_edit_menu).
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Modalità:"))
        self._rb_view = QRadioButton("Visualizza")
        self._rb_edit = QRadioButton("Edit")
        self._rb_view.setChecked(True)
        grp = QButtonGroup(self)
        for rb in (self._rb_view, self._rb_edit):
            grp.addButton(rb)
            bar.addWidget(rb)
        bar.addStretch()
        root.addLayout(bar)

        # Viewer (reads the current radio state on every click)
        self.viewer = SignalViewer(mode_getter=self._current_mode)
        root.addWidget(self.viewer, stretch=1)

        self.setCentralWidget(central)

        # ─── menu ───────────────────────────────────────────────────
        m_file = self.menuBar().addMenu("&File")
        act_open = QAction("&Apri CSV...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open)
        m_file.addAction(act_open)
        m_file.addSeparator()
        act_quit = QAction("&Esci", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        # ─── status bar ─────────────────────────────────────────────
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Pronto — File ▶ Apri CSV per iniziare.")

    # ─── Helpers ───────────────────────────────────────────────────
    def _current_mode(self) -> str:
        # SignalViewer expects "view" or "edit"; anything else is treated
        # as read-only (so future modes default to safe behavior).
        if self._rb_edit.isChecked():
            return "edit"
        return "view"

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Apri CSV di segnale", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return

        self.statusBar().showMessage(f"Analisi in corso: {Path(path).name}...")
        QApplication.processEvents()

        # Import here (not at module top) so the window still opens even
        # if something in the scientific core is broken — easier to diagnose.
        try:
            from cardiac_fp_analyzer.analyze import analyze_single_file
            result = analyze_single_file(path, verbose=False)
        except Exception as e:   # noqa: BLE001 — we want to show *any* error
            QMessageBox.critical(
                self, "Errore analisi", f"{type(e).__name__}: {e}"
            )
            self.statusBar().showMessage("Errore durante l'analisi.")
            return

        if result is None:
            QMessageBox.warning(
                self, "Nessun risultato",
                "La pipeline ha ritornato None (vedi log)."
            )
            self.statusBar().showMessage("Analisi ritornata vuota.")
            return

        self.viewer.set_result(result)
        self.statusBar().showMessage(
            f"{Path(path).name} — {self.viewer.beat_count()} battiti inclusi"
        )


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
