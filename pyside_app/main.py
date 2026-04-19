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

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyside_app.signal_viewer import SignalViewer

# ═══════════════════════════════════════════════════════════════════════
#   Template representativity constants
# ═══════════════════════════════════════════════════════════════════════
# Mirrored from ui/display.py so the PySide port has the same triggers
# as the Streamlit UI. Kept at module level (not inside the tab) because
# they're policy constants and might be reused if we ever split the
# representativity logic into its own helper.
_TEMPLATE_RISKY_RHYTHM_TYPES = frozenset({
    "chaotic",
    "ambiguous",
    "alternans_2_to_1",
    "trimodal",
})
_FPD_CV_TEMPLATE_WARN = 0.20   # 20 % — beyond this, T-waves jitter enough
#                              #        to cancel in the mean template.


# ═══════════════════════════════════════════════════════════════════════
#   Channel choices — UI dropdown (task #82)
# ═══════════════════════════════════════════════════════════════════════
# Today the Digilent µECG-Pharma loader produces exactly two electrodes
# (el1, el2); "auto" dispatches to ``channel_selection.select_best_channel``
# which picks by SNR.  The list is intentionally kept as module-level
# constants so the future multi-electrode work (task #83) can swap the
# provider to "read from ``metadata['channels']``" without touching the
# tab code.
CHANNEL_AUTO = "auto"
_CHANNEL_VALUES: tuple[str, ...] = (CHANNEL_AUTO, "el1", "el2")


def _channel_label(value: str) -> str:
    """Human-readable label for a channel-selection value.

    ``auto`` stays lowercase because it's a mode, not an electrode;
    electrode IDs are rendered uppercase so they stand out against the
    rest of the toolbar.
    """
    return "Auto" if value == CHANNEL_AUTO else value.upper()


def _make_metric_tile(caption: str) -> dict:
    """Build a label-over-value 'metric' tile and return its parts.

    Module-level so both the Aritmie and Battiti tabs can reuse it
    without cross-class imports. The returned dict carries the frame
    (for layout insertion) and the value label (so callers can update
    the number without rebuilding the widget).
    """
    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setMinimumHeight(60)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(8, 6, 8, 6)
    cap = QLabel(caption)
    cap.setStyleSheet("color: #888; font-size: 11px;")
    val = QLabel("—")
    val.setStyleSheet("font-size: 18px; font-weight: bold;")
    lay.addWidget(cap)
    lay.addWidget(val)
    return {"frame": frame, "value": val}


# ═══════════════════════════════════════════════════════════════════════
#   Zoom-toolbar helpers (Fase 2.E — Plotly-style mode selector)
# ═══════════════════════════════════════════════════════════════════════
# The old Streamlit/Plotly UI exposed a modebar with several discrete
# interaction modes (pan, box-zoom, reset, download-PNG). We recreate
# that in pyqtgraph using ``ViewBox.setMouseMode`` for the pan↔box
# toggle and ``pyqtgraph.exporters.ImageExporter`` for PNG export.
# See memory ``project_streamlit_zoom_modes`` for rationale.

def _export_plot_png(plot_widget: pg.PlotWidget, parent: QWidget) -> None:
    """Ask for a path and export ``plot_widget`` as a PNG.

    The exporter targets the ``PlotItem`` (not the containing widget)
    so axes, grid and legend are included but the pyqtgraph toolbar
    context-menu button is not. ``.png`` is appended if the user
    omits an extension. Any exporter failure surfaces as a QMessageBox
    rather than a silent failure so the user sees why nothing was
    written.
    """
    path, _ = QFileDialog.getSaveFileName(
        parent,
        parent.tr("Esporta PNG"),
        "",
        parent.tr("PNG files (*.png);;All files (*)"),
    )
    if not path:
        return
    if not path.lower().endswith(".png"):
        path = path + ".png"
    try:
        # Import lazily — the exporter subpackage pulls in Qt's image
        # backend, which we don't need for the rest of the UI.
        from pyqtgraph.exporters import ImageExporter
        exporter = ImageExporter(plot_widget.getPlotItem())
        exporter.export(path)
    except Exception as e:   # noqa: BLE001 — surface any export error
        QMessageBox.critical(
            parent,
            parent.tr("Errore esportazione"),
            f"{type(e).__name__}: {e}",
        )


def _add_zoom_mode_buttons(
    plot_widget: pg.PlotWidget, parent: QWidget, bar: QHBoxLayout,
) -> dict:
    """Add Pan / Box-zoom toggle to ``bar`` and wire it to ``plot_widget``.

    The two buttons form a mutually-exclusive QButtonGroup — Pan is the
    pyqtgraph default and stays checked on startup. Callers that want
    a full Plotly-style modebar should combine this with the Reset and
    PNG helpers (the Signal tab already has its own +/−/Auto so it
    only needs Pan/Box/PNG alongside).

    Returns the button dict so callers can tweak (e.g. disable Box
    while Edit mode is active, per ``project_streamlit_zoom_modes``).
    """
    vb = plot_widget.getPlotItem().vb

    lbl = QLabel(parent.tr("Zoom:"))
    lbl.setStyleSheet("color: #888; padding-right: 2px;")
    bar.addWidget(lbl)

    btn_pan = QPushButton(parent.tr("Pan"))
    btn_pan.setToolTip(parent.tr(
        "Pan — trascina il grafico per scorrere"
    ))
    btn_pan.setCheckable(True)
    btn_pan.setChecked(True)

    btn_box = QPushButton(parent.tr("Box"))
    btn_box.setToolTip(parent.tr(
        "Zoom area — trascina un rettangolo per ingrandirlo"
    ))
    btn_box.setCheckable(True)

    # Parent the QButtonGroup to the plot widget so its lifetime is
    # tied to the UI element (not leaked at module scope). Exclusive
    # mode is the default but set explicitly for readability.
    grp = QButtonGroup(plot_widget)
    grp.setExclusive(True)
    grp.addButton(btn_pan)
    grp.addButton(btn_box)

    for b in (btn_pan, btn_box):
        b.setFixedHeight(24)
        bar.addWidget(b)

    # Sync the ViewBox with the default-checked button so the plot
    # starts in the documented state even if pyqtgraph's default
    # changes in a future release.
    vb.setMouseMode(pg.ViewBox.PanMode)
    btn_pan.clicked.connect(lambda: vb.setMouseMode(pg.ViewBox.PanMode))
    btn_box.clicked.connect(lambda: vb.setMouseMode(pg.ViewBox.RectMode))

    return {"pan": btn_pan, "box": btn_box}


class _SignalTab(QWidget):
    """Segnale tab: picking-mode toolbar + the SignalViewer.

    Kept as its own widget so the tab can be swapped in/out of the
    QTabWidget without reaching into MainWindow internals, and so the
    picking-mode radio buttons live with the viewer they control.
    """

    # Fired when the user clicks "Ricalcola".  MainWindow owns the
    # current analysis result + config, so the actual recompute work
    # lives there — the tab just surfaces the request.
    recompute_requested = Signal()

    # Fired when the user picks a different channel in the dropdown
    # (Auto / EL1 / EL2).  Payload is the raw value from
    # ``_CHANNEL_VALUES`` — MainWindow re-runs ``analyze_single_file``
    # with ``channel=<payload>``.
    channel_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Viewer (reads the current radio state on every click). Created
        # BEFORE the toolbar so the zoom-mode buttons can wire directly
        # to its ViewBox — simpler than two-step initialisation with
        # placeholder layouts.
        self.viewer = SignalViewer(mode_getter=self._current_mode)

        # Toolbar row: channel dropdown + picking mode + raw overlay
        # toggle.  Two picking modes:
        # - Visualizza: read-only, clicks do nothing (zoom/pan still work).
        # - Edit: a left-click opens a contextual popup with the actions
        #   that make sense for that click (see SignalViewer._show_edit_menu).
        # "Mostra raw" overlays the unfiltered trace as a faint grey line.
        bar = QHBoxLayout()

        # ── Channel selector (task #82) ──────────────────────────────
        # The CSV loader currently produces EL1 + EL2; ``Auto`` delegates
        # to the SNR-based selector in ``channel_selection``.  The combo
        # is populated from ``_CHANNEL_VALUES`` so the future multi-
        # electrode work (task #83) can swap the provider — nothing else
        # in this tab hard-codes the channel list.  Signal emission is
        # BLOCKED while the combo is programmatically synchronised
        # (``set_channel`` below): without the guard, reflecting the
        # post-analysis "auto → el1" choice would fire a spurious
        # channel_changed and cause a second analyze_single_file call.
        bar.addWidget(QLabel(self.tr("Canale:")))
        self._combo_channel = QComboBox()
        for v in _CHANNEL_VALUES:
            self._combo_channel.addItem(_channel_label(v), userData=v)
        self._combo_channel.setCurrentIndex(0)  # Auto
        self._combo_channel.setToolTip(self.tr(
            "Scegli il canale da analizzare. 'Auto' usa il canale "
            "con SNR migliore (vedi selezione automatica)."
        ))
        self._combo_channel.currentIndexChanged.connect(
            self._on_channel_combo_changed
        )
        bar.addWidget(self._combo_channel)

        # Small label to the right of the combo shows what was actually
        # analysed after load — e.g. "EL1" when the combo says "Auto".
        # Stays empty until the first successful analysis.
        self._lbl_analyzed_ch = QLabel("")
        self._lbl_analyzed_ch.setStyleSheet("color: #8888aa;")
        bar.addWidget(self._lbl_analyzed_ch)
        bar.addSpacing(18)

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

        # Zoom-mode toggle (Plotly-style modebar — Fase 2.E).  Gives
        # the user explicit Pan vs Box-zoom control without relying on
        # trackpad/right-drag conventions that differ per platform.
        # Kept BEFORE the +/−/Auto buttons because discoverability
        # matters more than muscle memory — new users look for "Zoom"
        # modes first. The +/−/Auto buttons stay alongside per
        # ``project_streamlit_zoom_modes`` — they double as the UI for
        # the Ctrl++ / Ctrl+- / Ctrl+0 keyboard shortcuts.
        self._zoom_mode_btns = _add_zoom_mode_buttons(
            self.viewer, self, bar,
        )

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

        # PNG export — replicates the "Download plot as PNG" button that
        # Plotly's modebar offers. Useful when the user wants to drop a
        # figure into a report without screenshotting the window chrome.
        self._btn_png = QPushButton(self.tr("PNG"))
        self._btn_png.setToolTip(self.tr(
            "Esporta la vista corrente come immagine PNG"
        ))
        self._btn_png.setFixedHeight(24)
        bar.addWidget(self._btn_png)

        # ── "Ricalcola" button ───────────────────────────────────
        # Recomputes Parametri + Aritmie from the viewer's current
        # beat indices — the UI half of the Fase 2.F refactor.
        # Design notes:
        #  • Disabled until a result is loaded (MainWindow flips this
        #    via set_recompute_enabled).
        #  • Highlighted in amber when edits are pending (MainWindow
        #    flips this via set_recompute_pending).  Style-sheet is
        #    kept scoped to the button only so the platform theme
        #    still drives the rest of the toolbar.
        bar.addSpacing(12)
        self._btn_recompute = QPushButton(self.tr("Ricalcola"))
        self._btn_recompute.setFixedHeight(24)
        self._btn_recompute.setToolTip(self.tr(
            "Ricalcola Parametri e Aritmie dai battiti attuali"
        ))
        self._btn_recompute.setEnabled(False)
        self._btn_recompute.clicked.connect(self.recompute_requested.emit)
        bar.addWidget(self._btn_recompute)

        layout.addLayout(bar)

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

        # PNG export wiring — done here (not in _add_zoom_mode_buttons)
        # because the toolbar splits the modebar across two groups: the
        # Pan/Box toggle belongs with "mode" (stateful) and PNG belongs
        # with the one-shot +/−/Auto actions.
        self._btn_png.clicked.connect(
            lambda: _export_plot_png(self.viewer, self)
        )

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

    # ─── Recompute-button state ────────────────────────────────────
    def set_recompute_enabled(self, enabled: bool) -> None:
        """Enable or disable the Ricalcola button (and clear pending
        state when disabling — no point highlighting an unclickable
        button).
        """
        self._btn_recompute.setEnabled(bool(enabled))
        if not enabled:
            self.set_recompute_pending(False)

    def set_recompute_pending(self, pending: bool) -> None:
        """Toggle the 'edits pending' visual cue on the Ricalcola button.

        Amber background + bold caption when pending; default theme
        styling when clean.  Scoped via objectName so Qt's style-sheet
        cascade doesn't leak to neighbouring buttons.
        """
        if pending:
            self._btn_recompute.setStyleSheet(
                "QPushButton {"
                " background-color: #b26a00; color: white;"
                " font-weight: bold; border-radius: 3px;"
                " padding: 0 8px;"
                "}"
            )
            self._btn_recompute.setText(self.tr("Ricalcola •"))
            self._btn_recompute.setToolTip(self.tr(
                "Ci sono modifiche non ancora applicate — clicca per "
                "aggiornare Parametri e Aritmie."
            ))
        else:
            self._btn_recompute.setStyleSheet("")
            self._btn_recompute.setText(self.tr("Ricalcola"))
            self._btn_recompute.setToolTip(self.tr(
                "Ricalcola Parametri e Aritmie dai battiti attuali"
            ))

    # ─── Channel selector (task #82) ───────────────────────────────
    def _on_channel_combo_changed(self, _index: int) -> None:
        """Forward a user-driven combo change as ``channel_changed``.

        Programmatic syncs via ``set_channel`` block this signal
        explicitly, so only genuine user interactions reach MainWindow.
        """
        value = self._combo_channel.currentData()
        if isinstance(value, str):
            self.channel_changed.emit(value)

    def channel_choice(self) -> str:
        """Return the current combo value ('auto', 'el1', 'el2'…).

        Used by MainWindow to decide which channel to pass to
        ``analyze_single_file`` — it's the source of truth for "what did
        the user ask for", as opposed to ``analyzed_channel`` which is
        "what did the pipeline pick".
        """
        v = self._combo_channel.currentData()
        return v if isinstance(v, str) else CHANNEL_AUTO

    def set_channel(self, value: str) -> None:
        """Programmatically set the channel combo without emitting a signal.

        Useful on app start / project load when we need to reflect a
        saved preference without triggering an analyze_single_file.
        Falls back to ``Auto`` if ``value`` is not in the allowed list.
        """
        if value not in _CHANNEL_VALUES:
            value = CHANNEL_AUTO
        block = self._combo_channel.blockSignals(True)
        try:
            idx = _CHANNEL_VALUES.index(value)
            self._combo_channel.setCurrentIndex(idx)
        finally:
            self._combo_channel.blockSignals(block)

    def set_analyzed_channel(self, value: str | None) -> None:
        """Render what the pipeline actually analysed next to the combo.

        Called after every successful load / recompute.  Pass ``None`` or
        ``""`` to clear the label (e.g. before a new analysis starts or
        after an error).  When the user already picked a specific
        channel (not Auto) the label is redundant and we hide it.
        """
        if not value:
            self._lbl_analyzed_ch.setText("")
            return
        if self.channel_choice() == CHANNEL_AUTO:
            self._lbl_analyzed_ch.setText(
                self.tr("→ {0}").format(value.upper())
            )
        else:
            self._lbl_analyzed_ch.setText("")


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
    """Battiti tab: per-beat table with include checkboxes and bulk actions.

    Fase 2.C scope (task #76):

    * 5 columns — *Includi* (checkbox), *#*, *t (s)*, *RR (ms)*, *Stato*.
    * Soft toggle on the viewer's current beat set: unchecking a row
      adds the beat's *sample index* to an internal ``_excluded`` set;
      MainWindow filters those out before calling ``recompute_from_beats``.
      Re-checking restores them. The viewer is NOT mutated by this tab
      — checkboxes do not delete beats, only exclude them from the next
      recompute.
    * Bulk actions: *Seleziona tutti* / *Deseleziona tutti*.
    * Status column reflects per-beat quality flags derived from
      ``all_params``: ``OK``, ``No FPD`` (if ``fpd_ms`` is NaN/None), or
      em-dash (``—``) for beats added via the viewer that don't yet
      have parameters (post-edit, pre-recompute).
    * Numeric columns sort numerically (via :class:`_NumericItem`).
    * Double-click on a row centers the Signal-tab viewer on that beat
      (unchanged from the Phase 1 list view).
    """

    # Tells MainWindow to center the viewer on a given time (s).
    jump_requested = Signal(float)
    # Fired when the include-checkbox set changes (used by MainWindow to
    # flip the Ricalcola button into the "pending" state).
    included_changed = Signal()

    # Column indices — single source of truth so the per-row code below
    # doesn't sprinkle magic numbers everywhere.
    _COL_INCLUDE = 0
    _COL_NUM = 1
    _COL_TIME = 2
    _COL_RR = 3
    _COL_STATUS = 4

    # Banner palette — amber matches the "warning" severity used in the
    # Aritmie tab (and Streamlit's ``st.warning``) so the visual language
    # stays coherent across tabs.
    _BANNER_WARN_BG = "#b26a00"
    # Risky-rhythm labels get a short IT translation in the banner body —
    # we avoid exposing the raw classifier enum to the user.
    _RHYTHM_LABELS = {
        "chaotic": "caotico",
        "ambiguous": "ambiguo",
        "alternans_2_to_1": "alternans 2:1",
        "trimodal": "trimodale",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ═══════════════════════════════════════════════════════════
        #   Vertical splitter: template-view on top, table below.
        # ═══════════════════════════════════════════════════════════
        # A QSplitter lets the user drag the divider to trade plot
        # real-estate for table rows without losing either. The default
        # ratio (3 : 2) matches the Streamlit page where the chart sits
        # above the per-beat dataframe.
        splitter = QSplitter(Qt.Vertical)
        # On macOS dark mode the default splitter handle is almost
        # invisible — style it with a muted grey bar so the user can
        # actually find the drag-target.
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(
            "QSplitter::handle { background-color: #3c3c3c; } "
            "QSplitter::handle:hover { background-color: #5a5a5a; }"
        )

        # ─── Upper pane: header + banner + template plot + tiles ──
        upper = QWidget()
        up_lay = QVBoxLayout(upper)
        up_lay.setContentsMargins(0, 0, 0, 0)

        self._hdr = QLabel(self.tr("Nessun segnale caricato."))
        self._hdr.setStyleSheet("font-weight: bold; padding: 4px;")
        up_lay.addWidget(self._hdr)

        # Representativity banner — mirrors ui.display._render_template_
        # representativity_banner.  Hidden unless a risky rhythm type or
        # high FPD dispersion is detected.
        self._banner = QFrame()
        self._banner.setFrameShape(QFrame.StyledPanel)
        self._banner.setStyleSheet(
            f"QFrame {{ background-color: {self._BANNER_WARN_BG}; "
            f"border-radius: 6px; }}"
        )
        ban_lay = QVBoxLayout(self._banner)
        ban_lay.setContentsMargins(10, 6, 10, 6)
        self._banner_title = QLabel("")
        self._banner_title.setStyleSheet(
            "color: white; font-weight: bold; font-size: 13px;"
        )
        self._banner_body = QLabel("")
        self._banner_body.setStyleSheet("color: white; font-size: 12px;")
        self._banner_body.setWordWrap(True)
        ban_lay.addWidget(self._banner_title)
        ban_lay.addWidget(self._banner_body)
        self._banner.setVisible(False)
        up_lay.addWidget(self._banner)

        # pyqtgraph overlay plot — up to 30 beats (semi-transparent
        # blue) plus the mean template (red).  Dark palette identical to
        # the Signal-tab viewer (``#0e1117`` background + ``#cccccc``
        # axes/text) so the two tabs read as one coherent dark UI.
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#0e1117")
        for ax_name in ("left", "bottom"):
            ax = self._plot.getAxis(ax_name)
            ax.setPen("#cccccc")
            ax.setTextPen("#cccccc")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", self.tr("Tempo (ms)"),
                            color="#cccccc")
        self._plot.setLabel("left", self.tr("Ampiezza (mV)"),
                            color="#cccccc")
        self._plot.setMouseEnabled(x=True, y=True)

        # Zoom toolbar (Fase 2.E) — same Pan/Box/Auto/PNG set as the
        # Signal tab, shrunk into a single row above the plot. The
        # template view has no edit mode, so we can expose the full
        # modebar without worrying about click-handler collisions.
        plot_bar = QHBoxLayout()
        plot_bar.setContentsMargins(0, 0, 0, 0)
        self._zoom_mode_btns = _add_zoom_mode_buttons(
            self._plot, self, plot_bar,
        )
        self._btn_plot_auto = QPushButton(self.tr("Auto"))
        self._btn_plot_auto.setToolTip(self.tr("Adatta la vista ai dati"))
        self._btn_plot_auto.setFixedHeight(24)
        plot_bar.addWidget(self._btn_plot_auto)
        self._btn_plot_png = QPushButton(self.tr("PNG"))
        self._btn_plot_png.setToolTip(self.tr(
            "Esporta la vista corrente come immagine PNG"
        ))
        self._btn_plot_png.setFixedHeight(24)
        plot_bar.addWidget(self._btn_plot_png)
        plot_bar.addStretch()
        up_lay.addLayout(plot_bar)

        # Wire the one-shot buttons. ``autoRange()`` on the PlotWidget
        # delegates to the ViewBox, which is the same call the Signal
        # tab uses for its Auto button — identical behaviour, just
        # scoped to this plot.
        self._btn_plot_auto.clicked.connect(self._plot.autoRange)
        self._btn_plot_png.clicked.connect(
            lambda: _export_plot_png(self._plot, self)
        )

        up_lay.addWidget(self._plot, stretch=1)

        # 4 metric tiles (FPD, FPDcF, spike amp, FPD confidence).  Same
        # style as the Aritmie tab, same keys as the Streamlit view, so
        # the bilingual captions only need to be kept in sync once.
        tile_row = QHBoxLayout()
        self._tile_fpd = _make_metric_tile(self.tr("FPD (ms)"))
        self._tile_fpdc = _make_metric_tile(self.tr("FPDcF (ms)"))
        self._tile_spike = _make_metric_tile(self.tr("Spike amp (mV)"))
        self._tile_conf = _make_metric_tile(self.tr("FPD confidence"))
        for tile in (self._tile_fpd, self._tile_fpdc,
                     self._tile_spike, self._tile_conf):
            tile_row.addWidget(tile["frame"], stretch=1)
        up_lay.addLayout(tile_row)

        splitter.addWidget(upper)

        # ─── Lower pane: bulk toolbar + hint + table ─────────────
        lower = QWidget()
        lo_lay = QVBoxLayout(lower)
        lo_lay.setContentsMargins(0, 0, 0, 0)

        # Two buttons that flip every row's checkbox at once. The
        # in-place toggling is done in ``_set_all_included`` which fires
        # ``included_changed`` only once (not once per row) — keeps the
        # Ricalcola pending-state UI snappy on long beat lists.
        bar = QHBoxLayout()
        self._btn_all = QPushButton(self.tr("Seleziona tutti"))
        self._btn_none = QPushButton(self.tr("Deseleziona tutti"))
        for b in (self._btn_all, self._btn_none):
            b.setFixedHeight(24)
            bar.addWidget(b)
        bar.addStretch()
        self._btn_all.clicked.connect(lambda: self._set_all_included(True))
        self._btn_none.clicked.connect(lambda: self._set_all_included(False))
        lo_lay.addLayout(bar)

        hint = QLabel(self.tr(
            "Doppio-click su una riga per centrare il viewer sul battito.  "
            "Deseleziona la checkbox per escludere il battito dal prossimo "
            "ricalcolo."
        ))
        hint.setStyleSheet("color: #888; padding: 0 4px 4px 4px;")
        hint.setWordWrap(True)
        lo_lay.addWidget(hint)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            self.tr("Includi"),
            self.tr("#"),
            self.tr("t (s)"),
            self.tr("RR (ms)"),
            self.tr("Stato"),
        ])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(True)
        # Double-click anywhere on a row jumps to that beat. We use
        # cellDoubleClicked (not itemDoubleClicked) because the include
        # cell is checkable — its double-click would also toggle the
        # checkbox; this way the table-level signal still fires and we
        # can jump consistently from any column.
        self._table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        # itemChanged fires whenever a cell is edited *or* a check state
        # is toggled. Guarded by ``_block_item_changed`` so programmatic
        # population doesn't trigger spurious ``included_changed``
        # cascades.
        self._table.itemChanged.connect(self._on_item_changed)
        lo_lay.addWidget(self._table, stretch=1)

        splitter.addWidget(lower)

        # Default split: more room for the plot than the table — users
        # will still be able to drag the handle to trade space both ways.
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([420, 300])

        root.addWidget(splitter)

        # ─── Internal state ───────────────────────────────────────
        # Sample indices currently excluded from the next recompute.
        # Lives on the tab (not on MainWindow) because the user-visible
        # checkboxes are the source of truth — exposing them as a method
        # keeps MainWindow ignorant of the row layout.
        self._excluded: set[int] = set()
        # Per-beat parameters keyed by sample index — populated by
        # ``set_result``. Used to render RR and Status for each row.
        # Beats added via viewer clicks won't have entries here until
        # the user re-runs the pipeline (Ricalcola).
        self._params_by_idx: dict[int, dict] = {}
        # Re-entrancy guard for itemChanged — see comment on connect().
        self._block_item_changed: bool = False
        # Amplitude scale cached per-render so the tiles can report the
        # spike amplitude in whichever unit the plot is drawn in.
        self._amp_scale: float = 1.0
        self._amp_unit: str = "mV"

    # ─── Public API used by MainWindow ────────────────────────────
    def set_result(self, result: dict | None) -> None:
        """Reset state from a fresh analysis result.

        Called on file open and after each recompute. Reads ``all_params``
        + ``beat_indices_fpd`` to build the per-beat parameter map, then
        rebuilds the banner, template plot, tiles, and table. Clears the
        excluded set — by definition, nothing has been excluded yet on a
        new result.
        """
        self._excluded.clear()
        self._params_by_idx.clear()
        if result is None:
            self._render_banner(None)
            self._render_template_plot(None)
            self._render_tiles(None)
            self._render_rows([], None)
            return
        bi = result.get("beat_indices_fpd")
        if bi is None:
            bi = result.get("beat_indices", [])
        bi = list(int(x) for x in bi)
        all_p = list(result.get("all_params") or [])
        # Pair params with sample indices positionally; the pipeline
        # guarantees ``len(all_params) == len(beat_indices_fpd)`` for a
        # well-formed result, but we defensively skip the tail if not.
        for sample_idx, p in zip(bi, all_p):
            self._params_by_idx[sample_idx] = p
        # Upper pane first — the template plot caches the amplitude
        # scale that the spike-amp tile reads, so order matters.
        self._render_banner(result)
        self._render_template_plot(result)
        self._render_tiles(result)
        self._render_rows(bi, result.get("time_vector"))

    def set_beats(self, indices, time_vector) -> None:
        """Refresh after a viewer-side edit (add / remove / drag).

        Unlike ``set_result``, this is called when the user mutates the
        viewer directly. We keep the per-beat params we already have
        (matched by sample index) and show ``—`` in RR/Status for any
        new beats. The excluded set is intersected with the new indices
        so dormant exclusions for vanished beats don't survive.
        """
        new_idx = list(int(x) for x in (indices or []))
        # Drop excluded entries that no longer correspond to any beat —
        # otherwise re-clicking that sample would silently exclude it.
        self._excluded &= set(new_idx)
        self._render_rows(new_idx, time_vector)

    def get_excluded_indices(self) -> set[int]:
        """Return a defensive copy of the currently-excluded sample indices.

        MainWindow uses this to filter ``viewer.get_beat_indices()``
        before passing the trimmed list to ``recompute_from_beats``.
        """
        return set(self._excluded)

    # ─── Internal: row population ─────────────────────────────────
    def _render_rows(self, indices, time_vector) -> None:
        """(Re-)build every table row from the given beat list."""
        self._block_item_changed = True
        try:
            self._table.setSortingEnabled(False)
            self._table.setRowCount(0)
            n = len(indices)
            if n == 0 or time_vector is None or len(time_vector) == 0:
                self._hdr.setText(self.tr("Nessun battito."))
                self._table.setSortingEnabled(True)
                return
            tv_len = int(len(time_vector))
            self._table.setRowCount(n)
            # Pre-compute RR (ms) sequentially in beat order — the
            # diff-based formula is order-sensitive, so we calculate it
            # before pushing into the table (which may then re-sort).
            sorted_pairs = sorted(enumerate(indices), key=lambda t: int(t[1]))
            sorted_time_order = [
                float(time_vector[int(idx)])
                if 0 <= int(idx) < tv_len else float("nan")
                for _, idx in sorted_pairs
            ]
            rr_ms_by_pos = {}   # original-position → RR (ms)
            prev_t: float | None = None
            for pos, ((orig_pos, _), t) in enumerate(zip(sorted_pairs, sorted_time_order)):
                if prev_t is None or t != t:   # NaN → skip
                    rr_ms_by_pos[orig_pos] = float("nan")
                else:
                    rr_ms_by_pos[orig_pos] = (t - prev_t) * 1000.0
                if t == t:
                    prev_t = t

            for i, idx in enumerate(indices):
                idx_i = int(idx)
                if not (0 <= idx_i < tv_len):
                    # Out-of-range (shouldn't happen post-edit, but
                    # guard anyway). Mark the row as invalid and skip
                    # parameter population.
                    self._table.setItem(
                        i, self._COL_NUM, _NumericItem("—", float("inf")),
                    )
                    continue
                t = float(time_vector[idx_i])

                # ── Include checkbox (col 0) ────────────────────
                inc_item = QTableWidgetItem()
                inc_item.setFlags(
                    Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
                )
                inc_item.setCheckState(
                    Qt.Unchecked if idx_i in self._excluded else Qt.Checked
                )
                # Stash the sample index on the row so itemChanged can
                # find it without scanning the table.
                inc_item.setData(Qt.UserRole, idx_i)
                inc_item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(i, self._COL_INCLUDE, inc_item)

                # ── Beat number (col 1) ─────────────────────────
                num_item = _NumericItem(str(i + 1), float(i + 1))
                num_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                # Stash the time too, so cellDoubleClicked can jump
                # without going through the include cell.
                num_item.setData(Qt.UserRole, t)
                self._table.setItem(i, self._COL_NUM, num_item)

                # ── Time (col 2) ────────────────────────────────
                t_item = _NumericItem(f"{t:.3f}", t)
                t_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t_item.setData(Qt.UserRole, t)
                self._table.setItem(i, self._COL_TIME, t_item)

                # ── RR (col 3) ──────────────────────────────────
                rr = rr_ms_by_pos.get(i, float("nan"))
                rr_item = _NumericItem(_fmt_num(rr, 1), rr if rr == rr else float("inf"))
                rr_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._table.setItem(i, self._COL_RR, rr_item)

                # ── Stato (col 4) ───────────────────────────────
                status_text = self._derive_status(idx_i)
                st_item = QTableWidgetItem(status_text)
                st_item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(i, self._COL_STATUS, st_item)

            # Default order is the natural beat ordering (column #).
            self._table.setSortingEnabled(True)
            self._table.sortItems(self._COL_NUM, Qt.AscendingOrder)
            self._update_header_count()
        finally:
            self._block_item_changed = False

    # ─── Internal: upper-pane rendering (banner / plot / tiles) ───
    def _render_banner(self, result: dict | None) -> None:
        """Show the representativity warning when appropriate.

        Triggers (either is sufficient):

        * rhythm classifier reports a type that mixes morphologies
          (``chaotic``, ``ambiguous``, ``alternans_2_to_1``, ``trimodal``);
        * FPD coefficient-of-variation exceeds
          :data:`_FPD_CV_TEMPLATE_WARN` — wide T-wave jitter cancels in
          the mean template.

        Kept visually minimal: the amber frame is hidden unless one of
        the triggers fires, mirroring the Streamlit behaviour where no
        banner appears for well-behaved rhythms.
        """
        if result is None:
            self._banner.setVisible(False)
            return
        rc = (result.get("detection_info") or {}).get(
            "rhythm_classification"
        ) or {}
        rhythm_type = str(rc.get("rhythm_type") or "")
        summary = result.get("summary") or {}
        try:
            fpd_mean = float(summary.get("fpd_ms_mean", 0) or 0.0)
            fpd_std = float(summary.get("fpd_ms_std", 0) or 0.0)
        except (TypeError, ValueError):
            fpd_mean = 0.0
            fpd_std = 0.0
        fpd_cv = (fpd_std / fpd_mean) if fpd_mean > 0 else 0.0

        risky_rhythm = rhythm_type in _TEMPLATE_RISKY_RHYTHM_TYPES
        dispersive_fpd = fpd_cv > _FPD_CV_TEMPLATE_WARN
        if not (risky_rhythm or dispersive_fpd):
            self._banner.setVisible(False)
            return

        reasons: list[str] = []
        if risky_rhythm:
            label = self._RHYTHM_LABELS.get(rhythm_type, rhythm_type)
            reasons.append(self.tr("ritmo classificato '{0}'").format(label))
        if dispersive_fpd:
            reasons.append(self.tr(
                "dispersione FPD elevata (CV = {0:.0f}%)"
            ).format(fpd_cv * 100.0))

        self._banner_title.setText(self.tr(
            "Template potenzialmente non rappresentativo"
        ))
        self._banner_body.setText(self.tr(
            "Il template medio potrebbe mascherare la morfologia reale: "
            "{0}. Verifica il tracciato sovrapposto prima di trarre "
            "conclusioni sulla FPD."
        ).format(" · ".join(reasons)))
        self._banner.setVisible(True)

    def _render_template_plot(self, result: dict | None) -> None:
        """Port of ``ui.display.plot_beats`` — up to 30 beats + template.

        * Clears the plot on every call — simpler than diffing, and
          the plot has at most ~30 traces anyway.
        * Picks the amplitude unit dynamically (uV vs mV) so ex-vivo
          signals read naturally on both gain settings.
        * Draws the mean template in red on top; skips it when
          ``compute_template`` returns ``None`` (degenerate / short
          recordings).
        """
        self._plot.clear()
        if result is None:
            self._amp_scale = 1.0
            self._amp_unit = "mV"
            self._plot.setLabel("left", self.tr("Ampiezza (mV)"))
            return

        bd = result.get("beats_data") or []
        meta = result.get("metadata") or {}
        try:
            fs = float(meta.get("sample_rate") or 0.0)
        except (TypeError, ValueError):
            fs = 0.0
        if not bd or fs <= 0:
            # No beats yet or missing fs — leave the plot empty.
            self._amp_scale = 1.0
            self._amp_unit = "mV"
            self._plot.setLabel("left", self.tr("Ampiezza (mV)"))
            return

        # Pick uV vs mV from the first beat's peak-to-peak amplitude,
        # matching ``ui.helpers.amplitude_scale``. Kept local to avoid a
        # Streamlit-layer import from the PySide module.
        first = np.asarray(bd[0])
        ptp = float(np.ptp(first)) if first.size > 0 else 0.0
        if ptp < 1e-3:
            self._amp_scale = 1e6
            self._amp_unit = "uV"
            self._plot.setLabel("left", self.tr("Ampiezza (µV)"))
        else:
            self._amp_scale = 1e3
            self._amp_unit = "mV"
            self._plot.setLabel("left", self.tr("Ampiezza (mV)"))

        # Down-sample the beat list to ≤ 30 evenly-spaced entries so
        # heavy recordings don't stall the plot.
        n_plot = min(30, len(bd))
        step_idx = np.linspace(0, len(bd) - 1, n_plot, dtype=int)
        # Slightly brighter blue + higher alpha than the white-bg version
        # — on the dark canvas the original 80/255 alpha washed out into
        # the background. 130/255 keeps the cloud effect without making
        # individual traces noisy.
        pen_beats = pg.mkPen(color=(120, 180, 255, 130), width=1)
        for i in step_idx:
            arr = np.asarray(bd[int(i)])
            if arr.size == 0:
                continue
            x_ms = np.arange(arr.size) / fs * 1000.0
            self._plot.plot(x_ms, arr * self._amp_scale, pen=pen_beats)

        # Template overlay — identical mean definition as the Streamlit
        # view, computed lazily to keep the import surface narrow.
        try:
            from cardiac_fp_analyzer.residual_analysis import compute_template
            tmpl = compute_template(bd)
        except Exception:
            tmpl = None
        if tmpl is not None and len(tmpl) > 0:
            x_tmpl = np.arange(len(tmpl)) / fs * 1000.0
            pen_tmpl = pg.mkPen(color=(220, 40, 40), width=2.5)
            self._plot.plot(
                x_tmpl, np.asarray(tmpl) * self._amp_scale, pen=pen_tmpl,
                name=self.tr("Template"),
            )

    def _render_tiles(self, result: dict | None) -> None:
        """Populate the 4 metric tiles from ``result['summary']``.

        Spike amplitude is shown in the same unit (mV / µV) as the
        template plot, so the tile and the curve agree visually.
        """
        def _set(tile: dict, text: str) -> None:
            tile["value"].setText(text)

        if result is None:
            for t in (self._tile_fpd, self._tile_fpdc,
                      self._tile_spike, self._tile_conf):
                _set(t, "—")
            return

        s = result.get("summary") or {}

        def _f(key: str) -> float:
            try:
                return float(s.get(key, 0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        fpd_m, fpd_s = _f("fpd_ms_mean"), _f("fpd_ms_std")
        fpdc_m, fpdc_s = _f("fpdc_ms_mean"), _f("fpdc_ms_std")
        _set(self._tile_fpd, f"{fpd_m:.1f} ± {fpd_s:.1f}")
        _set(self._tile_fpdc, f"{fpdc_m:.1f} ± {fpdc_s:.1f}")

        # Summary stores spike amplitude in mV. If the plot is drawn in
        # µV, scale accordingly so the tile and the overlay read in the
        # same unit (avoids the "0.002 mV" illegibility of low-gain
        # recordings).
        spike_mv = _f("spike_amplitude_mV_mean")
        if self._amp_unit == "uV":
            _set(self._tile_spike, f"{spike_mv * 1000.0:.1f} µV")
        else:
            _set(self._tile_spike, f"{spike_mv:.4f} mV")

        conf = _f("fpd_confidence")
        _set(self._tile_conf, f"{conf:.2f}")

    def _derive_status(self, sample_idx: int) -> str:
        """Per-beat quality flag for the Stato column.

        Sources:
        * ``all_params[i]['fpd_ms']`` — NaN/None → ``No FPD``.
        * Anything else → ``OK``.
        * Beats with no params yet (added via viewer click, not yet
          ricalcolati) → ``—`` so the user knows the row is provisional.

        Future: incorporate ``rr_outlier_filter`` per-beat flags + QC
        per-beat morph tags. Kept narrow for 2.C to avoid coupling to
        fields that aren't always populated.
        """
        p = self._params_by_idx.get(sample_idx)
        if p is None:
            return "—"
        fpd = p.get("fpd_ms")
        try:
            f = float(fpd) if fpd is not None else float("nan")
        except (TypeError, ValueError):
            f = float("nan")
        if f != f:   # NaN
            return self.tr("No FPD")
        return self.tr("OK")

    def _update_header_count(self) -> None:
        """Refresh the bold header to reflect the current totals."""
        n = self._table.rowCount()
        n_excl = sum(
            1 for r in range(n)
            if self._table.item(r, self._COL_INCLUDE) is not None
            and self._table.item(r, self._COL_INCLUDE).checkState() == Qt.Unchecked
        )
        n_inc = n - n_excl
        if n == 0:
            self._hdr.setText(self.tr("Nessun battito."))
        elif n_excl == 0:
            self._hdr.setText(self.tr("{0} battiti inclusi").format(n))
        else:
            self._hdr.setText(self.tr(
                "{0} battiti inclusi, {1} esclusi (su {2})"
            ).format(n_inc, n_excl, n))

    # ─── Internal: signal handlers ────────────────────────────────
    def _on_cell_double_clicked(self, row: int, _col: int) -> None:
        """Jump the viewer to the beat at ``row`` (any column works)."""
        # The time was stashed on the # item — that one is guaranteed to
        # exist and isn't checkable, so its UserRole holds the float.
        item = self._table.item(row, self._COL_NUM)
        if item is None:
            return
        t = item.data(Qt.UserRole)
        if isinstance(t, (int, float)):
            self.jump_requested.emit(float(t))

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Sync the checkbox state into ``_excluded`` and notify listeners."""
        if self._block_item_changed:
            return
        if item.column() != self._COL_INCLUDE:
            return
        sample_idx = item.data(Qt.UserRole)
        if not isinstance(sample_idx, int):
            return
        if item.checkState() == Qt.Unchecked:
            self._excluded.add(sample_idx)
        else:
            self._excluded.discard(sample_idx)
        self._update_header_count()
        self.included_changed.emit()

    def _set_all_included(self, included: bool) -> None:
        """Bulk-flip every checkbox without firing one signal per row."""
        n = self._table.rowCount()
        if n == 0:
            return
        new_state = Qt.Checked if included else Qt.Unchecked
        # Snapshot BEFORE blocking to know whether the bulk action is
        # actually a no-op (saves a spurious "pending" flag).
        any_changed = False
        self._block_item_changed = True
        try:
            for r in range(n):
                inc = self._table.item(r, self._COL_INCLUDE)
                if inc is None:
                    continue
                if inc.checkState() != new_state:
                    inc.setCheckState(new_state)
                    sample_idx = inc.data(Qt.UserRole)
                    if isinstance(sample_idx, int):
                        if included:
                            self._excluded.discard(sample_idx)
                        else:
                            self._excluded.add(sample_idx)
                        any_changed = True
        finally:
            self._block_item_changed = False
        self._update_header_count()
        if any_changed:
            self.included_changed.emit()


def _fmt_num(x, prec: int) -> str:
    """Format a float-like value with fixed precision; anything that is
    not finite (None, NaN, strings that don't parse) becomes an em-dash
    so the table stays aligned.  Mirrors the empty-cell convention of
    the Streamlit UI (`show_params_table`).
    """
    try:
        if x is None:
            return "—"
        fx = float(x)
    except (TypeError, ValueError):
        return "—"
    if fx != fx:   # NaN — fx != fx is a cheap, import-free NaN test
        return "—"
    return f"{fx:.{prec}f}"


class _NumericItem(QTableWidgetItem):
    """QTableWidgetItem whose sort order is numeric, not lexicographic.

    Default ``QTableWidgetItem`` compares by the displayed text, which
    produces "1, 10, 11, …, 19, 2, 20, …" for integer columns and
    mis-orders decimals with different magnitudes ("1165.5" < "970.5"
    because '1' < '9').  By stashing the raw float in a dedicated
    attribute and overriding ``__lt__``, we keep the pretty-printed
    display text (with its fixed precision / em-dash convention) while
    sorting the way a user expects.

    Non-finite values (NaN, None, unparseable strings) are pushed to the
    end of an ascending sort by using ``+inf`` as their sort key — same
    visual rule as the em-dash placeholder in the display text.
    """

    def __init__(self, text: str, sort_value: float) -> None:
        super().__init__(text)
        self._sort_value = sort_value

    def __lt__(self, other: object) -> bool:   # type: ignore[override]
        if isinstance(other, _NumericItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)   # type: ignore[arg-type]


def _num_item(value, prec: int, *, as_int: bool = False) -> _NumericItem:
    """Build a right-aligned numeric cell that sorts numerically.

    ``prec``   : decimal places for the display text (ignored when
                 ``as_int`` is True).
    ``as_int`` : render the value as an integer (no decimals) — used
                 for the Beat column where "#1" is clearer than "1.000".
    """
    try:
        fx = float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        fx = float("nan")
    if fx != fx:
        text = "—"
        sort_value = float("inf")   # NaN / missing → end of ascending sort
    elif as_int:
        text = f"{int(round(fx))}"
        sort_value = fx
    else:
        text = f"{fx:.{prec}f}"
        sort_value = fx
    item = _NumericItem(text, sort_value)
    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return item


class _ParamsTab(QWidget):
    """Parametri tab: per-beat parameter table + summary stats table.

    Ports ``ui.display.show_params_table`` into PySide6.  Content is
    rebuilt whenever a new analysis result is loaded (see
    ``MainWindow._on_open``).  It does NOT currently refresh on edit
    in the viewer — adding/removing a beat in the Segnale tab will
    leave the parameter tables stale until the pipeline is re-run with
    the edits applied (that wiring belongs in #64 + #73.C).

    The tables are intentionally read-only (NoEditTriggers).  Editing
    of cell values is never the workflow — users edit beats in the
    viewer and the numbers recompute from the signal.
    """

    # Keys used to index the ``summary`` dict returned by the pipeline.
    # The ordering here controls the row order in the Riepilogo table.
    _SUMMARY_KEYS = [
        "spike_amplitude_mV",
        "fpd_ms",
        "fpdc_ms",
        "rise_time_ms",
        "rr_interval_ms",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Per-beat table ------------------------------------------------
        self._hdr_beat = QLabel(self.tr("Parametri per battito"))
        self._hdr_beat.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self._hdr_beat)

        self._per_beat = QTableWidget(0, 7)
        self._per_beat.setHorizontalHeaderLabels([
            self.tr("Beat"),
            self.tr("RR (ms)"),
            self.tr("Spike amp (mV)"),
            self.tr("FPD (ms)"),
            self.tr("FPDcF (ms)"),
            self.tr("Rise time (ms)"),
            self.tr("Max dV/dt"),
        ])
        self._per_beat.verticalHeader().setVisible(False)
        self._per_beat.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._per_beat.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._per_beat.setAlternatingRowColors(True)
        # Sort-by-click: lets the user rank beats by FPD etc. at a glance.
        self._per_beat.setSortingEnabled(True)
        self._per_beat.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        layout.addWidget(self._per_beat, stretch=2)

        # Summary table -------------------------------------------------
        self._hdr_sum = QLabel(self.tr("Riepilogo"))
        self._hdr_sum.setStyleSheet(
            "font-weight: bold; padding: 8px 4px 4px 4px;"
        )
        layout.addWidget(self._hdr_sum)

        self._summary = QTableWidget(0, 4)
        self._summary.setHorizontalHeaderLabels([
            self.tr("Parametro"),
            self.tr("Media"),
            self.tr("SD"),
            self.tr("CV%"),
        ])
        self._summary.verticalHeader().setVisible(False)
        self._summary.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._summary.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._summary.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        # The summary is short (5 rows) — give it a fixed modest height
        # so the per-beat table keeps most of the vertical space.
        self._summary.setMaximumHeight(200)
        layout.addWidget(self._summary)

    # ─── Public API ───────────────────────────────────────────────
    def set_result(self, result: dict | None) -> None:
        """Rebuild both tables from the pipeline's result dict.

        Passing ``None`` (or a result without ``all_params``) clears
        the tables so the tab doesn't show stale data between files.
        """
        all_p = (result or {}).get("all_params") or []
        summary = (result or {}).get("summary") or {}

        # Default row order: by beat_number ascending.  Without this the
        # pipeline's order (processing order, which can be reorganised by
        # later filtering steps) leaks into the UI and makes the table
        # look randomly permuted.  The user can still click a header to
        # re-sort by any other column.
        def _beat_sort_key(p):
            try:
                return (0, int(p.get("beat_number", 0)))
            except (TypeError, ValueError):
                return (1, 0)   # unknown/non-numeric → end
        all_p = sorted(all_p, key=_beat_sort_key)

        # Disable sorting while we populate so setItem doesn't reorder
        # rows mid-loop — Qt enables it again after we're done.
        self._per_beat.setSortingEnabled(False)
        self._per_beat.setRowCount(len(all_p))
        for row, p in enumerate(all_p):
            items = [
                _num_item(p.get("beat_number", row + 1), 0, as_int=True),
                _num_item(p.get("rr_interval_ms"), 1),
                _num_item(p.get("spike_amplitude_mV"), 5),
                _num_item(p.get("fpd_ms"), 1),
                _num_item(p.get("fpdc_ms"), 1),
                _num_item(p.get("rise_time_ms"), 2),
                _num_item(p.get("max_dvdt"), 5),
            ]
            for col, item in enumerate(items):
                self._per_beat.setItem(row, col, item)
        self._per_beat.setSortingEnabled(True)
        # Explicit default: ascending Beat on load.  Without this, a
        # previous sort state (from a prior file) would carry over and
        # confuse the user when a new CSV is opened.
        self._per_beat.sortItems(0, Qt.AscendingOrder)

        # Summary: one row per listed key, "—" if the pipeline didn't
        # produce that stat (for instance when all beats were rejected).
        self._summary.setRowCount(len(self._SUMMARY_KEYS))
        for row, key in enumerate(self._SUMMARY_KEYS):
            pretty = (key.replace("_mV", " (mV)")
                         .replace("_ms", " (ms)")
                         .replace("_", " "))
            m = summary.get(f"{key}_mean")
            s = summary.get(f"{key}_std")
            cv = summary.get(f"{key}_cv")
            cells = [pretty, _fmt_num(m, 3), _fmt_num(s, 3), _fmt_num(cv, 1)]
            for col, s_val in enumerate(cells):
                item = QTableWidgetItem(s_val)
                if col > 0:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._summary.setItem(row, col, item)

        # Keep the header counts in sync with the current content.
        self._hdr_beat.setText(
            self.tr("Parametri per battito — {0} righe").format(len(all_p))
        )


class _ArrhythmiaTab(QWidget):
    """Aritmie tab: classification banner + flags + residual metrics + events.

    Ports ``ui.display.show_arrhythmia`` into PySide6.  Reads
    ``result['arrhythmia_report']`` (an ``ArrhythmiaReport`` object with
    ``classification``, ``risk_score``, ``flags``, ``residual_details``,
    ``events``).  If the key is missing or ``None``, shows an empty
    state so the user knows this file has no arrhythmia data.
    """

    # Risk-score bands (same breakpoints as the Streamlit UI:
    # 🟢 < 30, 🟡 < 60, 🔴 otherwise).  Colours chosen to be readable on
    # both light and dark Qt themes.
    _BANNER_GREEN = "#2e7d32"   # risk < 30
    _BANNER_AMBER = "#b26a00"   # 30 ≤ risk < 60
    _BANNER_RED = "#b71c1c"     # risk ≥ 60
    _BANNER_NEUTRAL = "#555555"  # no report at all

    # Severity → Qt-styled prefix used in the flags list.  Kept in
    # ASCII so it renders consistently on every platform (some Qt
    # styles swallow emoji glyphs).
    _SEV_LABEL = {
        "info": ("INFO", "#1976d2"),
        "warning": ("WARN", "#b26a00"),
        "critical": ("CRIT", "#b71c1c"),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Classification banner ─────────────────────────────────
        # Big, coloured block at the top: classification label on the
        # left, risk score (and a progress bar) on the right.  The
        # risk bar gives a visual cue even when the label doesn't fit.
        self._banner = QFrame()
        self._banner.setFrameShape(QFrame.StyledPanel)
        self._banner.setMinimumHeight(64)
        banner_lay = QHBoxLayout(self._banner)
        banner_lay.setContentsMargins(12, 8, 12, 8)

        self._lbl_class = QLabel(self.tr("Nessun referto."))
        self._lbl_class.setStyleSheet(
            "color: white; font-size: 16px; font-weight: bold;"
        )
        banner_lay.addWidget(self._lbl_class, stretch=2)

        self._lbl_score = QLabel("")
        self._lbl_score.setStyleSheet("color: white; font-size: 14px;")
        self._lbl_score.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        banner_lay.addWidget(self._lbl_score, stretch=1)

        self._bar_score = QProgressBar()
        self._bar_score.setRange(0, 100)
        self._bar_score.setTextVisible(False)
        self._bar_score.setFixedWidth(160)
        self._bar_score.setFixedHeight(16)
        banner_lay.addWidget(self._bar_score)

        self._apply_banner_style(self._BANNER_NEUTRAL)
        root.addWidget(self._banner)

        # ── Flags ────────────────────────────────────────────────
        self._hdr_flags = QLabel(self.tr("Flag rilevate"))
        self._hdr_flags.setStyleSheet(
            "font-weight: bold; padding: 8px 4px 2px 4px;"
        )
        root.addWidget(self._hdr_flags)

        self._flags_list = QListWidget()
        # Fixed-ish height — flags are usually ≤ 5 items; keep more
        # vertical space for the events table below.
        self._flags_list.setMaximumHeight(140)
        root.addWidget(self._flags_list)

        # ── Residual metrics (4 tiles) ───────────────────────────
        self._hdr_resid = QLabel(self.tr("Analisi residuo"))
        self._hdr_resid.setStyleSheet(
            "font-weight: bold; padding: 8px 4px 2px 4px;"
        )
        root.addWidget(self._hdr_resid)

        resid_row = QHBoxLayout()
        self._tile_morph = self._make_metric_tile(
            self.tr("Morphology instability")
        )
        self._tile_ead = self._make_metric_tile(self.tr("EAD incidence"))
        self._tile_stv = self._make_metric_tile(self.tr("STV FPDcF (ms)"))
        self._tile_base = self._make_metric_tile(
            self.tr("Baseline-relative")
        )
        for tile in (self._tile_morph, self._tile_ead,
                     self._tile_stv, self._tile_base):
            resid_row.addWidget(tile["frame"], stretch=1)
        root.addLayout(resid_row)

        # ── Events table ─────────────────────────────────────────
        self._hdr_events = QLabel(self.tr("Eventi"))
        self._hdr_events.setStyleSheet(
            "font-weight: bold; padding: 8px 4px 2px 4px;"
        )
        root.addWidget(self._hdr_events)

        self._events_tbl = QTableWidget(0, 3)
        self._events_tbl.setHorizontalHeaderLabels([
            self.tr("Battito"),
            self.tr("Tipo"),
            self.tr("Dettagli"),
        ])
        self._events_tbl.verticalHeader().setVisible(False)
        self._events_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._events_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._events_tbl.setAlternatingRowColors(True)
        self._events_tbl.setSortingEnabled(True)
        hdr = self._events_tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        root.addWidget(self._events_tbl, stretch=1)

    # ─── Internals ───────────────────────────────────────────────
    def _make_metric_tile(self, caption: str) -> dict:
        """Thin wrapper on the module-level helper so existing call
        sites keep working unchanged.
        """
        return _make_metric_tile(caption)

    def _apply_banner_style(self, color: str) -> None:
        self._banner.setStyleSheet(
            f"QFrame {{ background-color: {color}; border-radius: 6px; }}"
        )

    # ─── Public API ──────────────────────────────────────────────
    def set_result(self, result: dict | None) -> None:
        """Rebuild the whole tab from the pipeline's result dict.

        ``result['arrhythmia_report']`` may be missing (e.g. files with
        too few beats for the analyser) — in that case we show a
        neutral banner and clear all sub-sections.
        """
        ar = (result or {}).get("arrhythmia_report")
        if ar is None:
            self._lbl_class.setText(self.tr("Nessun referto disponibile."))
            self._lbl_score.setText("")
            self._bar_score.setValue(0)
            self._apply_banner_style(self._BANNER_NEUTRAL)
            self._flags_list.clear()
            self._flags_list.addItem(
                QListWidgetItem(self.tr("Nessuna flag (referto assente)."))
            )
            for tile in (self._tile_morph, self._tile_ead,
                         self._tile_stv, self._tile_base):
                tile["value"].setText("—")
            self._events_tbl.setRowCount(0)
            self._hdr_events.setText(self.tr("Eventi"))
            return

        # ── Banner ──────────────────────────────────────────────
        classification = str(getattr(ar, "classification", "") or "—")
        try:
            score = int(getattr(ar, "risk_score", 0) or 0)
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))
        if score < 30:
            color = self._BANNER_GREEN
        elif score < 60:
            color = self._BANNER_AMBER
        else:
            color = self._BANNER_RED
        self._apply_banner_style(color)
        self._lbl_class.setText(classification)
        self._lbl_score.setText(
            self.tr("Risk score: {0}/100").format(score)
        )
        self._bar_score.setValue(score)

        # ── Flags ───────────────────────────────────────────────
        self._flags_list.clear()
        flags = list(getattr(ar, "flags", []) or [])
        if not flags:
            item = QListWidgetItem(
                self.tr("Nessuna flag aritmica rilevata.")
            )
            item.setForeground(Qt.darkGreen)
            self._flags_list.addItem(item)
        else:
            for f in flags:
                sev = str(f.get("severity", "info")).lower()
                label, hex_color = self._SEV_LABEL.get(
                    sev, ("???", "#555555"))
                text = self.tr("[{0}] {1} — {2}").format(
                    label,
                    f.get("type", "?"),
                    f.get("description", ""),
                )
                item = QListWidgetItem(text)
                # Tooltip carries the full raw description even when the
                # row is truncated by the list-widget width.
                item.setToolTip(str(f.get("description", "")))
                # Pinkish background isn't appropriate on all themes —
                # colouring the text alone is safer across light/dark.
                from PySide6.QtGui import QColor
                item.setForeground(QColor(hex_color))
                self._flags_list.addItem(item)

        # ── Residual metrics ───────────────────────────────────
        rd = dict(getattr(ar, "residual_details", {}) or {})
        self._tile_morph["value"].setText(
            _fmt_num(rd.get("morphology_instability"), 3)
        )
        ead_val = rd.get("ead_incidence_pct")
        self._tile_ead["value"].setText(
            "—" if ead_val is None else f"{_fmt_num(ead_val, 1)}%"
        )
        self._tile_stv["value"].setText(
            _fmt_num(rd.get("poincare_stv_fpdc_ms"), 1)
        )
        br = rd.get("baseline_relative")
        if br is None:
            self._tile_base["value"].setText("—")
        else:
            self._tile_base["value"].setText(
                self.tr("Sì") if br else self.tr("No")
            )

        # ── Events table ───────────────────────────────────────
        events = list(getattr(ar, "events", []) or [])
        self._events_tbl.setSortingEnabled(False)
        self._events_tbl.setRowCount(len(events))
        for row, e in enumerate(events):
            # Beat column uses the numeric-sort item so re-ranking by
            # beat number is actually numeric (same issue we hit on
            # the Parametri tab).
            self._events_tbl.setItem(
                row, 0, _num_item(e.get("beat"), 0, as_int=True)
            )
            self._events_tbl.setItem(
                row, 1, QTableWidgetItem(str(e.get("type", "")))
            )
            self._events_tbl.setItem(
                row, 2, QTableWidgetItem(str(e.get("details", "")))
            )
        self._events_tbl.setSortingEnabled(True)
        self._events_tbl.sortItems(0, Qt.AscendingOrder)
        self._hdr_events.setText(
            self.tr("Eventi — {0}").format(len(events))
        )


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
        self._params_tab = _ParamsTab()
        self._arrhythmia_tab = _ArrhythmiaTab()
        self._tabs.addTab(self._signal_tab, self.tr("Segnale"))
        self._tabs.addTab(self._beats_tab, self.tr("Battiti"))
        self._tabs.addTab(self._params_tab, self.tr("Parametri"))
        self._tabs.addTab(self._arrhythmia_tab, self.tr("Aritmie"))
        self.setCentralWidget(self._tabs)

        # ─── Analysis state ────────────────────────────────────────
        # Current result dict from the last successful analyze / recompute.
        # Kept here (not on the viewer) because it outlives individual
        # viewer refreshes and is what the Ricalcola path needs to rebuild
        # Parametri + Aritmie.  ``None`` before the first file is loaded.
        self._current_result: dict | None = None
        self._current_csv_path: Path | None = None
        # Config used for the current analysis; kept at window scope so
        # the Impostazioni dialog (task #74) can mutate ONE shared
        # instance and see the change reflected on the next Apri CSV
        # (or the immediate re-analysis fired by ``applied``).
        # Ricalcola also re-uses the same instance, so per-file edits
        # made in the Battiti tab survive a settings change until the
        # user reloads.  Default ``amplifier_gain = 1e4`` matches the
        # µECG-Pharma Digilent hardware — the field is editable in the
        # Impostazioni dialog.
        from cardiac_fp_analyzer.config import AnalysisConfig
        self._config: AnalysisConfig = AnalysisConfig()
        self._config.amplifier_gain = 1e4
        # ``_current_config`` historically tracked "config used for the
        # last analysis"; now it points to the same ``_config`` object
        # after every ``_run_analysis``.  Ricalcola reads it.
        self._current_config: AnalysisConfig | None = None
        # Guard flag: when True, ``_on_beats_changed`` will NOT flip the
        # Ricalcola button into the "pending" state.  We set it around
        # ``viewer.set_result`` calls (full load + post-recompute refresh)
        # because those trigger ``beats_changed`` even though no user
        # edit happened — so Parametri / Aritmie are already in sync.
        self._suppress_pending: bool = False

        # ─── Cross-tab wiring ──────────────────────────────────────
        # Every edit in the viewer refreshes the Battiti list (live
        # count + rows) AND flags the Ricalcola button as pending so
        # the user knows the Parametri/Aritmie tabs are stale.  A
        # double-click on the Battiti list jumps the Segnale-tab
        # viewer to that beat.  Toggling an include-checkbox on the
        # Battiti tab is also an "edit" for Ricalcola-pending purposes,
        # since the next recompute will see a different beat set.
        self.viewer.beats_changed.connect(self._refresh_beats_tab)
        self.viewer.beats_changed.connect(self._on_beats_changed)
        self._beats_tab.jump_requested.connect(self._jump_to_time)
        self._beats_tab.included_changed.connect(self._on_beats_changed)
        self._signal_tab.recompute_requested.connect(self._on_recompute)
        # Channel dropdown (task #82): re-runs analyze_single_file on the
        # currently-open CSV with the newly-selected channel.
        self._signal_tab.channel_changed.connect(self._on_channel_changed)

        # ─── Menu ──────────────────────────────────────────────────
        m_file = self.menuBar().addMenu(self.tr("&File"))
        act_open = QAction(self.tr("&Apri CSV..."), self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open)
        m_file.addAction(act_open)
        m_file.addSeparator()
        # Impostazioni dialog (task #74) — Cmd+, on macOS, Ctrl+, elsewhere.
        # Qt maps "Ctrl+," to "Cmd+," automatically on mac.
        act_settings = QAction(self.tr("&Impostazioni..."), self)
        act_settings.setShortcut("Ctrl+,")
        act_settings.setMenuRole(QAction.PreferencesRole)  # mac: → app menu
        act_settings.triggered.connect(self._on_settings)
        m_file.addAction(act_settings)
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

    def _on_beats_changed(self) -> None:
        """Flag Parametri / Aritmie as stale when the user edits beats.

        Only runs when a result is already loaded AND the change isn't
        coming from our own ``set_result`` call (``_suppress_pending``).
        This is what makes the Ricalcola button glow amber after an
        add/remove/drag in the viewer.
        """
        if self._suppress_pending:
            return
        if self._current_result is None:
            return
        self._signal_tab.set_recompute_pending(True)

    def _on_recompute(self) -> None:
        """Re-run the post-detection pipeline with the viewer's current
        beats and refresh Parametri + Aritmie.

        The heavy lifting lives in ``recompute_from_beats`` (Fase 2.F
        refactor): it re-uses the cached signal / time arrays and only
        redoes segmentation → QC → rhythm filter → parameters → arrhythmia.
        We keep the viewer's beats as the source of truth — whatever the
        user has added, removed, or moved is what the pipeline sees.
        """
        if self._current_result is None:
            return

        # Build the input beat list for the pipeline:
        #   viewer beats  MINUS  Battiti-tab excluded (soft-toggle)
        # The viewer's beats are the authoritative set of *positions*;
        # the Battiti tab's excluded set is a per-sample-index filter
        # layered on top (checkbox unchecked → skip this beat).
        viewer_bi = self.viewer.get_beat_indices()
        excluded = self._beats_tab.get_excluded_indices()
        bi_edited = np.asarray(
            [int(x) for x in viewer_bi if int(x) not in excluded],
            dtype=int,
        )
        self.statusBar().showMessage(
            self.tr("Ricalcolo in corso ({0} battiti)...").format(len(bi_edited))
        )
        QApplication.processEvents()

        try:
            from cardiac_fp_analyzer.analyze import recompute_from_beats
            new_result = recompute_from_beats(
                self._current_result, bi_edited,
                config=self._current_config, verbose=False,
            )
        except Exception as e:   # noqa: BLE001 — surface any pipeline error
            QMessageBox.critical(
                self, self.tr("Errore ricalcolo"),
                f"{type(e).__name__}: {e}",
            )
            self.statusBar().showMessage(
                self.tr("Errore durante il ricalcolo.")
            )
            return

        # Update the cached result so subsequent recomputes build on the
        # filtered/re-segmented state, not the original detection.
        self._current_result = new_result

        # Populate the Battiti tab's per-beat parameter map BEFORE the
        # viewer fires beats_changed (same ordering as ``_on_open``).
        # Otherwise ``_refresh_beats_tab`` renders rows against the
        # previous-run params map and the Stato column shows stale
        # "OK"/"No FPD" flags until ``_beats_tab.set_result`` catches up.
        self._beats_tab.set_result(new_result)

        # The viewer keeps whatever beats the user had; sync it to the
        # recompute's filtered set so marker positions reflect what the
        # pipeline actually kept (QC/RR filter may drop a few).  Guard
        # with _suppress_pending so the resulting beats_changed doesn't
        # mark the (already-fresh) tabs as stale again.
        self._suppress_pending = True
        try:
            self.viewer.set_result(new_result)
        finally:
            self._suppress_pending = False

        # Refresh Parametri + Aritmie from the new result.
        self._params_tab.set_result(new_result)
        self._arrhythmia_tab.set_result(new_result)

        self._signal_tab.set_recompute_pending(False)

        # Status-bar delta: the QC / rhythm / RR-outlier filters may
        # silently drop beats that the user had explicitly added. Show
        # the delta so the user knows when the pipeline rejected some
        # of their picks — otherwise a "why did my beat disappear?"
        # moment.  When the pipeline kept everything, fall back to the
        # simpler phrasing so the common case isn't noisy.
        n_in = int(len(bi_edited))
        n_out = int(self.viewer.beat_count())
        if n_in > 0 and n_out < n_in:
            dropped = n_in - n_out
            self.statusBar().showMessage(self.tr(
                "Ricalcolo completato — {0} battiti inclusi "
                "({1} filtrati da QC/ritmo/RR su {2})"
            ).format(n_out, dropped, n_in))
        else:
            self.statusBar().showMessage(
                self.tr("Ricalcolo completato — {0} battiti inclusi").format(n_out)
            )

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
        # Respect any channel the user previously picked in this session
        # — preference is sticky across Open actions until they change
        # the combo back to Auto.
        self._run_analysis(path, channel=self._signal_tab.channel_choice())

    def _on_channel_changed(self, choice: str) -> None:
        """User picked a different channel in the Signal-tab dropdown.

        Re-runs ``analyze_single_file`` on the currently-open CSV with
        the new channel.  No-op when no CSV is loaded yet.  The combo's
        own state already reflects the new choice (the signal fires
        *after* the combo updates); ``_run_analysis`` will update the
        side label with whatever the pipeline ends up analysing.
        """
        if self._current_csv_path is None:
            return
        self._run_analysis(str(self._current_csv_path), channel=choice)

    def _on_settings(self) -> None:
        """Open the Impostazioni dialog and re-run on ``applied`` (task #74).

        The dialog mutates ``self._config`` in place via
        ``apply_values_to_config``, so holding a stale snapshot in this
        method would be a bug — we just re-read ``self._config`` on the
        next analysis.  The ``applied`` signal fires on both Applica and
        OK; we only re-run if a CSV is currently open, otherwise the
        settings silently wait for the next File → Apri.
        """
        # Import here (not at module top) to keep the cold-start path
        # minimal — opening the Settings dialog is a conscious user
        # action, so the import cost is fine on first click.
        from pyside_app.settings_dialog import SettingsDialog

        dlg = SettingsDialog(self._config, parent=self)
        dlg.applied.connect(self._on_settings_applied)
        dlg.exec()
        # dlg goes out of scope here; its connections die with it —
        # no manual disconnect needed.

    def _on_settings_applied(self) -> None:
        """Slot: settings were just committed; re-run if a CSV is loaded.

        Uses the current channel preference from the Signal tab so the
        user's EL1/EL2 choice (task #82) survives a settings tweak.
        """
        if self._current_csv_path is None:
            return
        self._run_analysis(
            str(self._current_csv_path),
            channel=self._signal_tab.channel_choice(),
        )

    def _run_analysis(self, path: str, channel: str) -> None:
        """Execute the pipeline and populate all tabs.

        Single entry point for both ``File → Apri CSV`` and channel
        changes from the Signal tab.  Keeping the two paths unified
        matters because any failure-mode polish (status-bar messages,
        downstream-tab clearing, pending-state reset) only has to live
        in one place — divergent copies are how the UI develops stale
        widgets after an error.
        """
        self.statusBar().showMessage(
            self.tr("Analisi in corso: {0}...").format(Path(path).name)
        )
        QApplication.processEvents()

        # Import here (not at module top) so the window still opens even
        # if something in the scientific core is broken — easier to diagnose.
        try:
            from cardiac_fp_analyzer.analyze import analyze_single_file
            # We pass ``self._config`` — the window-scope AnalysisConfig
            # that the Impostazioni dialog (task #74) mutates in place.
            # Ricalcola reads this same instance (via ``_current_config``),
            # so a settings change immediately propagates to both the
            # "Apri" and "Ricalcola" paths without extra plumbing.
            result = analyze_single_file(
                path, channel=channel, verbose=False, config=self._config,
            )
        except Exception as e:   # noqa: BLE001 — we want to show *any* error
            QMessageBox.critical(
                self, self.tr("Errore analisi"),
                f"{type(e).__name__}: {e}",
            )
            self._clear_results(self.tr("Errore durante l'analisi."))
            return

        if result is None:
            QMessageBox.warning(
                self, self.tr("Nessun risultato"),
                self.tr("La pipeline ha ritornato None (vedi log)."),
            )
            self._clear_results(self.tr("Analisi ritornata vuota."))
            return

        # Cache the result for the Ricalcola path BEFORE ``viewer.set_result``
        # fires ``beats_changed`` — the guard relies on both the result
        # being set AND ``_suppress_pending`` being True.
        self._current_result = result
        self._current_csv_path = Path(path)
        self._current_config = self._config
        # Populate the Battiti tab's per-beat parameter map BEFORE the
        # viewer fires beats_changed (which routes through
        # ``_refresh_beats_tab → _beats_tab.set_beats``).  If we don't,
        # the first render shows "—" in RR/Stato until the user touches
        # something.
        self._beats_tab.set_result(result)
        self._suppress_pending = True
        try:
            self.viewer.set_result(result)
        finally:
            self._suppress_pending = False
        self._params_tab.set_result(result)
        self._arrhythmia_tab.set_result(result)
        # Fresh load: Parametri + Aritmie are in sync, so clear any
        # leftover "pending" styling and enable the Ricalcola button.
        self._signal_tab.set_recompute_enabled(True)
        self._signal_tab.set_recompute_pending(False)

        # Reflect what channel the pipeline actually analysed — useful
        # when the user asked for Auto and wants to know which electrode
        # was picked by ``select_best_channel``.
        analyzed = result.get("file_info", {}).get("analyzed_channel", "")
        self._signal_tab.set_analyzed_channel(analyzed)
        self._update_window_title(Path(path).name, analyzed)

        # Make sure the Segnale tab is visible after a successful load
        # (the user expects to see the plot right away, even if they had
        # clicked around the placeholder tabs before opening).
        self._tabs.setCurrentWidget(self._signal_tab)
        ch_label = (
            _channel_label(analyzed) if analyzed else _channel_label(channel)
        )
        if channel == CHANNEL_AUTO and analyzed:
            ch_suffix = self.tr("{0} (auto)").format(ch_label)
        else:
            ch_suffix = ch_label
        self.statusBar().showMessage(
            self.tr("{0} — {1} — {2} battiti inclusi").format(
                Path(path).name, ch_suffix, self.viewer.beat_count(),
            )
        )

    def _clear_results(self, status_msg: str) -> None:
        """Reset all downstream state after a failed / empty analysis.

        Extracted to avoid the error-handling divergence that the
        previous ``_on_open`` had two near-identical copies of.
        """
        self._params_tab.set_result(None)
        self._arrhythmia_tab.set_result(None)
        self._beats_tab.set_result(None)
        self._current_result = None
        self._current_csv_path = None
        self._current_config = None
        self._signal_tab.set_recompute_enabled(False)
        self._signal_tab.set_analyzed_channel(None)
        self._update_window_title(None, None)
        self.statusBar().showMessage(status_msg)

    def _update_window_title(
        self, filename: str | None, channel: str | None,
    ) -> None:
        """Reflect the current file + analysed channel in the window title.

        ``None`` filename resets to the app name.  When the channel is
        known it's appended so the user sees "chipD_ch1.csv — EL1"
        without having to look at the Signal tab.
        """
        base = self.tr("Cardiac FP Analyzer")
        if not filename:
            self.setWindowTitle(base)
            return
        if channel:
            self.setWindowTitle(f"{filename} — {channel.upper()} · {base}")
        else:
            self.setWindowTitle(f"{filename} · {base}")


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
