"""PySide6 entry point for cardiac-fp-analyzer (ADR-0001, Phase 1).

Phase 1 scope (task #72): main-window shell with a 4-tab QTabWidget
(Segnale / Battiti / Parametri / Aritmie).  Only the Segnale tab is
functional for now; the others are placeholders that will be wired up
in Phase 2 (#73).

Sidecar persistence of beat overrides is NOT handled here — that lives
in the pipeline integration task (#64).

Run from the repo root:

    python -m pyside_app.main
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QRadioButton, QStatusBar, QTabWidget, QVBoxLayout,
    QWidget,
)

from pyside_app.signal_viewer import SignalViewer


class _SignalTab(QWidget):
    """Segnale tab: picking-mode toolbar + the SignalViewer.

    Kept as its own widget so the tab can be swapped in/out of the
    QTabWidget without reaching into MainWindow internals, and so the
    picking-mode radio buttons live with the viewer they control.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Toolbar row: picking mode + raw overlay toggle.
        # Two picking modes:
        # - Visualizza: read-only, clicks do nothing (zoom/pan still work).
        # - Edit: a left-click opens a contextual popup with the actions
        #   that make sense for that click (see SignalViewer._show_edit_menu).
        # "Mostra raw" overlays the unfiltered trace as a faint grey line.
        bar = QHBoxLayout()
        bar.addWidget(QLabel(self.tr("Modalità:")))
        self._rb_view = QRadioButton(self.tr("Visualizza"))
        self._rb_edit = QRadioButton(self.tr("Edit"))
        self._rb_view.setChecked(True)
        grp = QButtonGroup(self)
        for rb in (self._rb_view, self._rb_edit):
            grp.addButton(rb)
            bar.addWidget(rb)
        bar.addSpacing(24)
        self._cb_raw = QCheckBox(self.tr("Mostra raw"))
        bar.addWidget(self._cb_raw)
        bar.addStretch()

        # Zoom controls.  pyqtgraph's default wheel/right-drag zoom is
        # kept (it works where it works), but explicit buttons are much
        # more discoverable and work the same on every platform
        # regardless of trackpad / mouse configuration.
        self._btn_zoom_in = QPushButton("＋")
        self._btn_zoom_out = QPushButton("−")
        self._btn_fit = QPushButton(self.tr("Auto"))
        self._btn_zoom_in.setToolTip(self.tr("Zoom avanti (Ctrl++)"))
        self._btn_zoom_out.setToolTip(self.tr("Zoom indietro (Ctrl+-)"))
        self._btn_fit.setToolTip(self.tr("Adatta alla finestra (Ctrl+0)"))
        for b in (self._btn_zoom_in, self._btn_zoom_out, self._btn_fit):
            b.setFixedHeight(24)
            bar.addWidget(b)
        layout.addLayout(bar)

        # Viewer (reads the current radio state on every click)
        self.viewer = SignalViewer(mode_getter=self._current_mode)
        layout.addWidget(self.viewer, stretch=1)

        # Wire the raw-overlay checkbox to the viewer.  toggled(bool) is
        # emitted on every state change (keyboard or mouse), which is
        # what we want.
        self._cb_raw.toggled.connect(self.viewer.set_show_raw)

        # Zoom-button wiring.  Factors picked to feel roughly like one
        # notch of a mouse wheel (~30% step each way).
        self._btn_zoom_in.clicked.connect(lambda: self.viewer.zoom_by(0.7))
        self._btn_zoom_out.clicked.connect(lambda: self.viewer.zoom_by(1.4))
        self._btn_fit.clicked.connect(self.viewer.fit_view)

        # Hover-info line — a thin monospace label under the plot that
        # updates as the mouse moves over the signal.  Kept at the tab
        # level (not in the status bar) so it doesn't compete with
        # file-load / error messages the main window writes there.
        self._hover_lbl = QLabel("")
        self._hover_lbl.setStyleSheet(
            "color: #aaaaaa; font-family: monospace;"
            " padding: 2px 6px; min-height: 18px;"
        )
        layout.addWidget(self._hover_lbl)
        self.viewer.hover_info.connect(self._hover_lbl.setText)

    # ─── Helpers ───────────────────────────────────────────────────
    def _current_mode(self) -> str:
        # SignalViewer expects "view" or "edit"; anything else is treated
        # as read-only (so future modes default to safe behavior).
        if self._rb_edit.isChecked():
            return "edit"
        return "view"


def _placeholder_tab(text: str) -> QWidget:
    """Centered italic label — used for tabs not yet implemented."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setAlignment(Qt.AlignCenter)
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #888; font-style: italic; font-size: 14px;")
    lbl.setAlignment(Qt.AlignCenter)
    lay.addWidget(lbl, alignment=Qt.AlignCenter)
    return w


class _BeatsTab(QWidget):
    """Minimal Battiti tab: live count + clickable beat list.

    Phase 1 scope: prove the signal/slot plumbing between viewer edits
    and per-beat views.  Rich per-beat content (templates, RR, amplitude,
    FPD) lands in Phase 2 (#73).  For now every row says "Battito #N
    at t = X.XXX s" and double-clicking a row asks MainWindow to
    center the Signal-tab viewer on that beat.
    """

    jump_requested = Signal(float)   # time in seconds

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self._hdr = QLabel(self.tr("Nessun segnale caricato."))
        self._hdr.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self._hdr)

        hint = QLabel(self.tr(
            "Doppio-click su una riga per centrare il viewer sul battito."
        ))
        hint.setStyleSheet("color: #888; padding: 0 4px 4px 4px;")
        layout.addWidget(hint)

        self._list = QListWidget()
        # itemActivated fires on both double-click and Enter, which
        # matches the "pick one and jump" mental model better than
        # clicked (which would fire on every selection change).
        self._list.itemActivated.connect(self._on_activated)
        layout.addWidget(self._list, stretch=1)

    def set_beats(self, indices, time_vector) -> None:
        """Rebuild the list from current beat indices + time vector.

        ``indices``      : 1-D int array of sample indices into the signal
        ``time_vector``  : 1-D float array, the viewer's time axis (s)
        Either may be empty — empty list is handled gracefully.
        """
        self._list.clear()
        n = int(len(indices)) if indices is not None else 0
        if n == 0 or time_vector is None or len(time_vector) == 0:
            self._hdr.setText(self.tr("Nessun battito."))
            return
        self._hdr.setText(self.tr("{0} battiti inclusi").format(n))
        tv_len = int(len(time_vector))
        for i, idx in enumerate(indices):
            idx_i = int(idx)
            if not (0 <= idx_i < tv_len):
                continue
            t = float(time_vector[idx_i])
            item = QListWidgetItem(
                self.tr("Battito #{0}  —  t = {1:.3f} s").format(i + 1, t)
            )
            # Stash the time on the item so the activation handler
            # doesn't have to look back into the arrays.
            item.setData(Qt.UserRole, t)
            self._list.addItem(item)

    def _on_activated(self, item: QListWidgetItem) -> None:
        t = item.data(Qt.UserRole)
        if isinstance(t, (int, float)):
            self.jump_requested.emit(float(t))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.tr("cardiac-fp-analyzer"))
        self.resize(1400, 800)

        # ─── Tabs ──────────────────────────────────────────────────
        # Order mirrors the current Streamlit UI: Segnale first, then
        # per-beat views, summary parameters, arrhythmia panel.
        self._tabs = QTabWidget()
        self._signal_tab = _SignalTab()
        self._beats_tab = _BeatsTab()
        self._tabs.addTab(self._signal_tab, self.tr("Segnale"))
        self._tabs.addTab(self._beats_tab, self.tr("Battiti"))
        self._tabs.addTab(
            _placeholder_tab(self.tr("Parametri — in arrivo in Fase 2")),
            self.tr("Parametri"),
        )
        self._tabs.addTab(
            _placeholder_tab(self.tr("Aritmie — in arrivo in Fase 2")),
            self.tr("Aritmie"),
        )
        self.setCentralWidget(self._tabs)

        # ─── Cross-tab wiring ──────────────────────────────────────
        # Every edit in the viewer refreshes the Battiti list (live
        # count + rows), and a double-click on the Battiti list jumps
        # the Segnale-tab viewer to that beat.
        self.viewer.beats_changed.connect(self._refresh_beats_tab)
        self._beats_tab.jump_requested.connect(self._jump_to_time)

        # ─── Menu ──────────────────────────────────────────────────
        m_file = self.menuBar().addMenu(self.tr("&File"))
        act_open = QAction(self.tr("&Apri CSV..."), self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open)
        m_file.addAction(act_open)
        m_file.addSeparator()
        act_quit = QAction(self.tr("&Esci"), self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        # ─── Keyboard shortcuts (global) ───────────────────────────
        # Registered at the MainWindow level so they fire no matter
        # which widget has focus.  Ctrl++ and Ctrl+- match the
        # convention of most viewers (and Qt's StandardKey aliases).
        # A plain "+" without modifier is also accepted as a convenience
        # because Ctrl+Shift+= is awkward on Italian layouts.
        QShortcut(QKeySequence(QKeySequence.ZoomIn), self,
                  activated=lambda: self.viewer.zoom_by(0.7))
        QShortcut(QKeySequence(QKeySequence.ZoomOut), self,
                  activated=lambda: self.viewer.zoom_by(1.4))
        QShortcut(QKeySequence("Ctrl++"), self,
                  activated=lambda: self.viewer.zoom_by(0.7))
        QShortcut(QKeySequence("Ctrl+-"), self,
                  activated=lambda: self.viewer.zoom_by(1.4))
        QShortcut(QKeySequence("Ctrl+0"), self,
                  activated=lambda: self.viewer.fit_view())

        # ─── Status bar ────────────────────────────────────────────
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            self.tr("Pronto — File ▶ Apri CSV per iniziare.")
        )

    # ─── Convenience accessors ────────────────────────────────────
    @property
    def viewer(self) -> SignalViewer:
        """Shortcut to the SignalViewer inside the Segnale tab."""
        return self._signal_tab.viewer

    # ─── Slots for cross-tab signals ──────────────────────────────
    def _refresh_beats_tab(self) -> None:
        """Rebuild the Battiti list from the viewer's current state."""
        self._beats_tab.set_beats(
            self.viewer.get_beat_indices(),
            self.viewer.get_time_vector(),
        )

    def _jump_to_time(self, t_s: float) -> None:
        """Center the viewer on ``t_s`` and switch to the Segnale tab
        so the user actually sees where the beat is.
        """
        self.viewer.center_on(float(t_s))
        self._tabs.setCurrentWidget(self._signal_tab)

    # ─── Actions ──────────────────────────────────────────────────
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Apri CSV di segnale"),
            "",
            self.tr("CSV files (*.csv);;All files (*)"),
        )
        if not path:
            return

        self.statusBar().showMessage(
            self.tr("Analisi in corso: {0}...").format(Path(path).name)
        )
        QApplication.processEvents()

        # Import here (not at module top) so the window still opens even
        # if something in the scientific core is broken — easier to diagnose.
        try:
            from cardiac_fp_analyzer.analyze import analyze_single_file
            result = analyze_single_file(path, verbose=False)
        except Exception as e:   # noqa: BLE001 — we want to show *any* error
            QMessageBox.critical(
                self, self.tr("Errore analisi"),
                f"{type(e).__name__}: {e}",
            )
            self.statusBar().showMessage(
                self.tr("Errore durante l'analisi.")
            )
            return

        if result is None:
            QMessageBox.warning(
                self, self.tr("Nessun risultato"),
                self.tr("La pipeline ha ritornato None (vedi log)."),
            )
            self.statusBar().showMessage(
                self.tr("Analisi ritornata vuota.")
            )
            return

        self.viewer.set_result(result)
        # Make sure the Segnale tab is visible after a successful load
        # (the user expects to see the plot right away, even if they had
        # clicked around the placeholder tabs before opening).
        self._tabs.setCurrentWidget(self._signal_tab)
        self.statusBar().showMessage(
            self.tr("{0} — {1} battiti inclusi").format(
                Path(path).name, self.viewer.beat_count()
            )
        )


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
