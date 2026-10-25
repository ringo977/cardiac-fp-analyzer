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

import logging
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
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

from cardiac_fp_analyzer.study import Group, find_group
from cardiac_fp_analyzer.template_quality import (
    FPD_CV_TEMPLATE_WARN,
    TEMPLATE_RISKY_RHYTHM_TYPES,
    template_representativity,
)
from pyside_app import dialogs as _dialogs
from pyside_app import i18n as _i18n
from pyside_app import theme
from pyside_app.chip_map import ChipMapWidget
from pyside_app.i18n import tr
from pyside_app.review_panel import ReviewPanel
from pyside_app.review_queue import ReviewQueue
from pyside_app.signal_viewer import SignalViewer
from pyside_app.study_panel import StudyPanel

logger = logging.getLogger(__name__)
_dialogs.install()        # Qt-drawn message boxes: the native macOS alert crashed (see dialogs.py)

# ═══════════════════════════════════════════════════════════════════════
#   Template representativity constants
# ═══════════════════════════════════════════════════════════════════════
# Policy lives in cardiac_fp_analyzer.template_quality so both UIs agree;
# module-level aliases kept for readability / back-compat.
_TEMPLATE_RISKY_RHYTHM_TYPES = TEMPLATE_RISKY_RHYTHM_TYPES
_FPD_CV_TEMPLATE_WARN = FPD_CV_TEMPLATE_WARN


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
    if value == CHANNEL_AUTO:
        return "Auto"
    return value if value[:1] == 'E' and value[1:].isdigit() else value.upper()


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
        tr("Export PNG"),
        "",
        tr("PNG files (*.png);;All files (*)"),
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
            tr("Export error"),
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

    lbl = QLabel(tr("Zoom:"))
    lbl.setStyleSheet("color: #888; padding-right: 2px;")
    bar.addWidget(lbl)

    btn_pan = QPushButton(tr("Pan"))
    btn_pan.setToolTip(tr(
        "Pan — drag the plot to scroll"
    ))
    btn_pan.setCheckable(True)
    btn_pan.setChecked(True)

    btn_box = QPushButton(tr("Box"))
    btn_box.setToolTip(tr(
        "Zoom area — drag a rectangle to enlarge it"
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
    # Fired when the user picks another chamber of a multi-chamber chip
    # (payload: chamber name, 'A'); MainWindow re-runs the analysis on
    # that chamber with the electrode choice set to Auto.
    chamber_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._channel_values: tuple[str, ...] = _CHANNEL_VALUES

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
        # ── Chamber selector (multi-chamber chips, v3.13) ─────────────
        # Hidden for two-input CSV files; for a µHeart file it lists the
        # chambers of the layout and the channel combo lists the
        # electrodes of the chosen chamber.
        self._lbl_chamber = QLabel(tr("Chamber:"))
        self._combo_chamber = QComboBox()
        self._combo_chamber.setToolTip(tr(
            "Chip chamber (one microtissue per chamber). The analysis uses only the electrodes of "
            "the chamber."
        ))
        self._combo_chamber.currentIndexChanged.connect(self._on_chamber_combo_changed)
        bar.addWidget(self._lbl_chamber)
        bar.addWidget(self._combo_chamber)
        self._lbl_chamber.setVisible(False)
        self._combo_chamber.setVisible(False)

        bar.addWidget(QLabel(tr("Channel:")))
        self._combo_channel = QComboBox()
        for v in _CHANNEL_VALUES:
            self._combo_channel.addItem(_channel_label(v), userData=v)
        self._combo_channel.setCurrentIndex(0)  # Auto
        self._combo_channel.setToolTip(tr(
            "Choose the channel to analyse. 'Auto' uses the channel with the best SNR (see "
            "automatic selection)."
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

        bar.addWidget(QLabel(tr("Mode:")))
        self._rb_view = QRadioButton(tr("View"))
        self._rb_edit = QRadioButton(tr("Edit"))
        self._rb_view.setChecked(True)
        grp = QButtonGroup(self)
        for rb in (self._rb_view, self._rb_edit):
            grp.addButton(rb)
            bar.addWidget(rb)
        bar.addSpacing(24)
        self._cb_raw = QCheckBox(tr("Show raw"))
        bar.addWidget(self._cb_raw)
        bar.addStretch()

        # Second toolbar row (zoom, export, recompute): one long row would set the central widget's
        # minimum width above 1100 px and, with the Revisione dock, push the window beyond a laptop screen.
        bar2 = QHBoxLayout()

        # Zoom-mode toggle (Plotly-style modebar — Fase 2.E).  Gives
        # the user explicit Pan vs Box-zoom control without relying on
        # trackpad/right-drag conventions that differ per platform.
        # Kept BEFORE the +/−/Auto buttons because discoverability
        # matters more than muscle memory — new users look for "Zoom"
        # modes first. The +/−/Auto buttons stay alongside per
        # ``project_streamlit_zoom_modes`` — they double as the UI for
        # the Ctrl++ / Ctrl+- / Ctrl+0 keyboard shortcuts.
        self._zoom_mode_btns = _add_zoom_mode_buttons(
            self.viewer, self, bar2,
        )

        # Zoom controls.  pyqtgraph's default wheel/right-drag zoom is
        # kept (it works where it works), but explicit buttons are much
        # more discoverable and work the same on every platform
        # regardless of trackpad / mouse configuration.
        self._btn_zoom_in = QPushButton("＋")
        self._btn_zoom_out = QPushButton("−")
        self._btn_fit = QPushButton(tr("Auto"))
        self._btn_zoom_in.setToolTip(tr("Zoom in (Ctrl++)"))
        self._btn_zoom_out.setToolTip(tr("Zoom out (Ctrl+-)"))
        self._btn_fit.setToolTip(tr("Fit to window (Ctrl+0)"))
        for b in (self._btn_zoom_in, self._btn_zoom_out, self._btn_fit):
            b.setFixedHeight(24)
            bar2.addWidget(b)

        # PNG export — replicates the "Download plot as PNG" button that
        # Plotly's modebar offers. Useful when the user wants to drop a
        # figure into a report without screenshotting the window chrome.
        self._btn_png = QPushButton(tr("PNG"))
        self._btn_png.setToolTip(tr(
            "Export the current view as a PNG image"
        ))
        self._btn_png.setFixedHeight(24)
        bar2.addWidget(self._btn_png)

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
        bar2.addStretch()
        self._btn_recompute = QPushButton(tr("Recompute"))
        self._btn_recompute.setFixedHeight(24)
        self._btn_recompute.setToolTip(tr(
            "Recompute Parameters and Arrhythmias from the current beats"
        ))
        self._btn_recompute.setEnabled(False)
        self._btn_recompute.clicked.connect(self.recompute_requested.emit)
        bar2.addWidget(self._btn_recompute)

        layout.addLayout(bar)
        layout.addLayout(bar2)

        # Chip map (multi-chamber chips): one row per chamber, one cell per
        # electrode coloured by the quick electrode score; hidden for CSV.
        self.chip_map = ChipMapWidget()
        layout.addWidget(self.chip_map)

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
            self._btn_recompute.setText(tr("Recompute •"))
            self._btn_recompute.setToolTip(tr(
                "There are changes not yet applied — click to update Parameters and Arrhythmias."
            ))
        else:
            self._btn_recompute.setStyleSheet("")
            self._btn_recompute.setText(tr("Recompute"))
            self._btn_recompute.setToolTip(tr(
                "Recompute Parameters and Arrhythmias from the current beats"
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

    def _on_chamber_combo_changed(self, _index: int) -> None:
        value = self._combo_chamber.currentData()
        if isinstance(value, str):
            self.chamber_changed.emit(value)

    def chamber_choice(self) -> str | None:
        """Chamber currently selected, or None for single-tissue files."""
        if self._combo_chamber.isHidden():          # explicit hide, independent of the parent being shown
            return None
        v = self._combo_chamber.currentData()
        return v if isinstance(v, str) else None

    def set_chip(self, layout, chamber: str | None, scores: dict, summary: dict | None = None) -> None:
        """Show the chamber selector and the chip map for a multi-chamber
        file (``layout`` None hides them and restores the el1/el2 list).
        The channel combo then lists Auto plus the electrodes of
        ``chamber`` (stimulation electrodes marked)."""
        if layout is None:
            self._lbl_chamber.setVisible(False)
            self._combo_chamber.setVisible(False)
            self.chip_map.clear()
            self.set_channel_values(_CHANNEL_VALUES)
            return
        block = self._combo_chamber.blockSignals(True)
        try:
            self._combo_chamber.clear()
            for c in layout.chambers:
                txt = tr("{0}").format(c.name)
                if summary and summary.get(c.name):
                    txt += f"  ({summary[c.name]})"
                self._combo_chamber.addItem(txt, userData=c.name)
            names = [c.name for c in layout.chambers]
            if chamber in names:
                self._combo_chamber.setCurrentIndex(names.index(chamber))
        finally:
            self._combo_chamber.blockSignals(block)
        self._lbl_chamber.setVisible(True)
        self._combo_chamber.setVisible(True)
        self.chip_map.set_chip(layout, scores, summary)
        ch = layout[chamber] if chamber in layout.names else layout.chambers[0]
        labels = {}
        for e in ch.electrodes:
            sc = scores.get(e, {})
            if e in ch.stimulation:
                labels[e] = f"{e} (stim)"
            elif sc.get('score') is not None and np.isfinite(sc['score']):
                labels[e] = f"{e}  ·  {sc['score']:.0f}"
            else:
                labels[e] = f"{e}  ·  —"
        self.set_channel_values((CHANNEL_AUTO,) + tuple(ch.electrodes), labels)

    def set_channel_values(self, values: tuple[str, ...], labels: dict | None = None) -> None:
        """Replace the entries of the channel combo (signal blocked)."""
        block = self._combo_channel.blockSignals(True)
        try:
            self._combo_channel.clear()
            for v in values:
                self._combo_channel.addItem((labels or {}).get(v) or _channel_label(v), userData=v)
            self._combo_channel.setCurrentIndex(0)
        finally:
            self._combo_channel.blockSignals(block)
        self._channel_values = tuple(values)

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

        Useful on app start / study load when we need to reflect a
        saved preference without triggering an analyze_single_file.
        Falls back to ``Auto`` if ``value`` is not in the allowed list.
        """
        if value not in self._channel_values:
            value = CHANNEL_AUTO
        block = self._combo_channel.blockSignals(True)
        try:
            idx = self._channel_values.index(value)
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
                tr("→ {0}").format(value.upper())
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
    # Risky-rhythm labels get a short readable form in the banner body (shown through ``tr()``) —
    # we avoid exposing the raw classifier enum to the user.
    _RHYTHM_LABELS = {
        "chaotic": "chaotic",
        "ambiguous": "ambiguous",
        "alternans_2_to_1": "alternans 2:1",
        "trimodal": "trimodal",
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

        self._hdr = QLabel(tr("No signal loaded."))
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
        # blue) plus the mean template (red).  The palette (background
        # + axis pens + grid alpha) comes from the shared ``theme``
        # module so the Battiti tab stays in sync with the Signal-tab
        # viewer and the dose-response dialog.  A user toggle in the
        # Visualizza menu switches Scuro ↔ Chiaro live; see
        # ``_on_theme_changed`` for the re-apply path.
        self._plot = pg.PlotWidget()
        theme.apply_to_plot(self._plot)
        self._apply_template_axis_labels()
        self._plot.setMouseEnabled(x=True, y=True)
        # Subscribe to live theme toggles; the Battiti tab outlives
        # every plot it hosts, so we never disconnect.
        theme.connect(self._on_theme_changed)

        # Zoom toolbar (Fase 2.E) — same Pan/Box/Auto/PNG set as the
        # Signal tab, shrunk into a single row above the plot. The
        # template view has no edit mode, so we can expose the full
        # modebar without worrying about click-handler collisions.
        plot_bar = QHBoxLayout()
        plot_bar.setContentsMargins(0, 0, 0, 0)
        self._zoom_mode_btns = _add_zoom_mode_buttons(
            self._plot, self, plot_bar,
        )
        self._btn_plot_auto = QPushButton(tr("Auto"))
        self._btn_plot_auto.setToolTip(tr("Fit the view to the data"))
        self._btn_plot_auto.setFixedHeight(24)
        plot_bar.addWidget(self._btn_plot_auto)
        self._btn_plot_png = QPushButton(tr("PNG"))
        self._btn_plot_png.setToolTip(tr(
            "Export the current view as a PNG image"
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
        self._tile_fpd = _make_metric_tile(tr("FPD (ms)"))
        self._tile_fpdc = _make_metric_tile(tr("FPDcF (ms)"))
        self._tile_spike = _make_metric_tile(tr("Spike amp (mV)"))
        self._tile_conf = _make_metric_tile(tr("FPD confidence"))
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
        self._btn_all = QPushButton(tr("Select all"))
        self._btn_none = QPushButton(tr("Deselect all"))
        for b in (self._btn_all, self._btn_none):
            b.setFixedHeight(24)
            bar.addWidget(b)
        bar.addStretch()
        self._btn_all.clicked.connect(lambda: self._set_all_included(True))
        self._btn_none.clicked.connect(lambda: self._set_all_included(False))
        lo_lay.addLayout(bar)

        hint = QLabel(tr(
            "Double-click a row to centre the viewer on the beat.  Untick the checkbox to exclude "
            "the beat from the next recompute."
        ))
        hint.setStyleSheet("color: #888; padding: 0 4px 4px 4px;")
        hint.setWordWrap(True)
        lo_lay.addWidget(hint)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            tr("Include"),
            tr("#"),
            tr("t (s)"),
            tr("RR (ms)"),
            tr("Status"),
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
        # Sample indices submitted to the last recompute (i.e. the full
        # set of beats the pipeline *tried* to analyse).  A beat in this
        # set but NOT in ``_params_by_idx`` was dropped by QC / ritmo /
        # RR — rendered as "Scartato" so the user sees their pick was
        # rejected instead of silently vanishing.  A beat added AFTER
        # the recompute is not in this set and shows "—" (provisional).
        self._analyzed_indices: set[int] = set()
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
        self._analyzed_indices.clear()
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
        # Default analyzed-set = the beats the pipeline produced.  On
        # file open this matches the viewer and nothing shows up as
        # "Scartato".  After Ricalcola, MainWindow calls
        # ``set_last_analyzed(bi_edited)`` to widen this to every beat
        # that was *submitted* — so pipeline-rejected picks get flagged.
        self._analyzed_indices = set(bi)
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
        # ``indices or []`` blows up when ``indices`` is a numpy array
        # with more than one element — ndarray.__bool__ raises for
        # multi-element arrays.  The viewer always passes a (possibly
        # empty) ndarray, so do an explicit None check instead.
        new_idx = [int(x) for x in (indices if indices is not None else [])]
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

    def set_last_analyzed(self, indices) -> None:
        """Record the full set of beats submitted to the last recompute.

        Called by MainWindow right after ``set_result`` in the Ricalcola
        path.  ``set_result`` conservatively sets ``_analyzed_indices``
        to the pipeline *output* (kept beats); this widens it to the
        pipeline *input* so user picks that QC / ritmo / RR dropped get
        rendered as "Scartato" in the Stato column instead of silently
        reverting to "—" (which means "needs Ricalcola").

        No re-render — the caller is expected to follow with
        ``set_beats(...)`` to rebuild the rows.
        """
        self._analyzed_indices = set(
            int(x) for x in (indices if indices is not None else [])
        )

    # ─── Theme integration ───────────────────────────────────────
    def _apply_template_axis_labels(self) -> None:
        """Set the template plot's axis titles using the active palette.

        Kept separate from ``theme.apply_to_plot`` so the labels can be
        retranslated independently (in case Qt's language switcher is
        ever wired up) while still tracking the theme colour.
        """
        fg = theme.label_color()
        self._plot.setLabel("bottom", tr("Time (ms)"), color=fg)
        self._plot.setLabel("left", tr("Amplitude (mV)"), color=fg)

    def _on_theme_changed(self, _mode: str) -> None:
        """Live-repaint the Battiti template plot on theme toggle."""
        theme.apply_to_plot(self._plot)
        self._apply_template_axis_labels()

    # ─── Internal: row population ─────────────────────────────────
    def _render_rows(self, indices, time_vector) -> None:
        """(Re-)build every table row from the given beat list."""
        self._block_item_changed = True
        try:
            self._table.setSortingEnabled(False)
            self._table.setRowCount(0)
            n = len(indices)
            if n == 0 or time_vector is None or len(time_vector) == 0:
                self._hdr.setText(tr("No beats."))
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
        tq = template_representativity(result)
        rhythm_type, fpd_cv = tq["rhythm_type"], tq["fpd_cv"]
        risky_rhythm, dispersive_fpd = tq["risky_rhythm"], tq["dispersive_fpd"]
        if tq["representative"]:
            self._banner.setVisible(False)
            return

        reasons: list[str] = []
        if risky_rhythm:
            label = tr(self._RHYTHM_LABELS.get(rhythm_type, rhythm_type))
            reasons.append(tr("rhythm classified '{0}'").format(label))
        if dispersive_fpd:
            reasons.append(tr(
                "high FPD dispersion (CV = {0:.0f}%)"
            ).format(fpd_cv * 100.0))

        self._banner_title.setText(tr(
            "Template possibly not representative"
        ))
        self._banner_body.setText(tr(
            "The mean template may hide the real morphology: {0}. Check the overlaid traces before "
            "drawing conclusions on the FPD."
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
            self._plot.setLabel("left", tr("Amplitude (mV)"))
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
            self._plot.setLabel("left", tr("Amplitude (mV)"))
            return

        # Pick uV vs mV from the first beat's peak-to-peak amplitude,
        # matching ``ui.helpers.amplitude_scale``. Kept local to avoid a
        # Streamlit-layer import from the PySide module.
        first = np.asarray(bd[0])
        ptp = float(np.ptp(first)) if first.size > 0 else 0.0
        if ptp < 1e-3:
            self._amp_scale = 1e6
            self._amp_unit = "uV"
            self._plot.setLabel("left", tr("Amplitude (µV)"))
        else:
            self._amp_scale = 1e3
            self._amp_unit = "mV"
            self._plot.setLabel("left", tr("Amplitude (mV)"))

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
                name=tr("Template"),
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

        Four outcomes:
        * ``OK``       — beat has params AND fpd_ms is a valid number.
        * ``No FPD``   — beat has params but fpd_ms is NaN/None (pipeline
          couldn't detect repolarisation).
        * ``Scartato`` — beat was *submitted* to the last recompute
          (``_analyzed_indices``) but has no params: QC / ritmo / RR
          filter rejected it.  Typical case: user click in a noise
          region, morph-correlation below threshold.
        * ``—``        — beat added to the viewer since the last
          recompute (never analysed): row is provisional until
          Ricalcola.

        Future: expose *why* a Scartato beat was rejected (amp / morph /
        RR / rhythm) by threading the per-beat rejection reason through
        ``recompute_from_beats``.  Out of scope for the POC.
        """
        p = self._params_by_idx.get(sample_idx)
        if p is not None:
            fpd = p.get("fpd_ms")
            try:
                f = float(fpd) if fpd is not None else float("nan")
            except (TypeError, ValueError):
                f = float("nan")
            if f != f:   # NaN
                return tr("No FPD")
            return tr("OK")
        # No params for this beat — either rejected or provisional.
        if sample_idx in self._analyzed_indices:
            return tr("Discarded")
        return "—"

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
            self._hdr.setText(tr("No beats."))
        elif n_excl == 0:
            self._hdr.setText(tr("{0} beats included").format(n))
        else:
            self._hdr.setText(tr(
                "{0} beats included, {1} excluded (out of {2})"
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
        self._hdr_beat = QLabel(tr("Per-beat parameters"))
        self._hdr_beat.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self._hdr_beat)

        self._per_beat = QTableWidget(0, 7)
        self._per_beat.setHorizontalHeaderLabels([
            tr("Beat"),
            tr("RR (ms)"),
            tr("Spike amp (mV)"),
            tr("FPD (ms)"),
            tr("FPDcF (ms)"),
            tr("Rise time (ms)"),
            tr("Max dV/dt"),
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
        self._hdr_sum = QLabel(tr("Summary"))
        self._hdr_sum.setStyleSheet(
            "font-weight: bold; padding: 8px 4px 4px 4px;"
        )
        layout.addWidget(self._hdr_sum)

        self._summary = QTableWidget(0, 4)
        self._summary.setHorizontalHeaderLabels([
            tr("Parameter"),
            tr("Mean"),
            tr("SD"),
            tr("CV%"),
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
            tr("Per-beat parameters — {0} rows").format(len(all_p))
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

        self._lbl_class = QLabel(tr("No report."))
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
        self._hdr_flags = QLabel(tr("Flags detected"))
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
        self._hdr_resid = QLabel(tr("Residual analysis"))
        self._hdr_resid.setStyleSheet(
            "font-weight: bold; padding: 8px 4px 2px 4px;"
        )
        root.addWidget(self._hdr_resid)

        resid_row = QHBoxLayout()
        self._tile_morph = self._make_metric_tile(
            tr("Morphology instability")
        )
        self._tile_ead = self._make_metric_tile(tr("EAD incidence"))
        self._tile_stv = self._make_metric_tile(tr("STV FPDcF (ms)"))
        self._tile_base = self._make_metric_tile(
            tr("Baseline-relative")
        )
        for tile in (self._tile_morph, self._tile_ead,
                     self._tile_stv, self._tile_base):
            resid_row.addWidget(tile["frame"], stretch=1)
        root.addLayout(resid_row)

        # ── Events table ─────────────────────────────────────────
        self._hdr_events = QLabel(tr("Events"))
        self._hdr_events.setStyleSheet(
            "font-weight: bold; padding: 8px 4px 2px 4px;"
        )
        root.addWidget(self._hdr_events)

        self._events_tbl = QTableWidget(0, 3)
        self._events_tbl.setHorizontalHeaderLabels([
            tr("Beat"),
            tr("Type"),
            tr("Details"),
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
            self._lbl_class.setText(tr("No report available."))
            self._lbl_score.setText("")
            self._bar_score.setValue(0)
            self._apply_banner_style(self._BANNER_NEUTRAL)
            self._flags_list.clear()
            self._flags_list.addItem(
                QListWidgetItem(tr("No flags (report missing)."))
            )
            for tile in (self._tile_morph, self._tile_ead,
                         self._tile_stv, self._tile_base):
                tile["value"].setText("—")
            self._events_tbl.setRowCount(0)
            self._hdr_events.setText(tr("Events"))
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
            tr("Risk score: {0}/100").format(score)
        )
        self._bar_score.setValue(score)

        # ── Flags ───────────────────────────────────────────────
        self._flags_list.clear()
        flags = list(getattr(ar, "flags", []) or [])
        if not flags:
            item = QListWidgetItem(
                tr("No arrhythmia flag detected.")
            )
            item.setForeground(Qt.darkGreen)
            self._flags_list.addItem(item)
        else:
            for f in flags:
                sev = str(f.get("severity", "info")).lower()
                label, hex_color = self._SEV_LABEL.get(
                    sev, ("???", "#555555"))
                text = tr("[{0}] {1} — {2}").format(
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
                tr("Yes") if br else tr("No")
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
            tr("Events — {0}").format(len(events))
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("cardiac-fp-analyzer"))
        self.resize(1400, 800)

        # ─── Tabs ──────────────────────────────────────────────────
        # Order mirrors the current Streamlit UI: Segnale first, then
        # per-beat views, summary parameters, arrhythmia panel.
        self._tabs = QTabWidget()
        self._signal_tab = _SignalTab()
        self._beats_tab = _BeatsTab()
        self._params_tab = _ParamsTab()
        self._arrhythmia_tab = _ArrhythmiaTab()
        self._tabs.addTab(self._signal_tab, tr("Signal"))
        self._tabs.addTab(self._beats_tab, tr("Beats"))
        self._tabs.addTab(self._params_tab, tr("Parameters"))
        self._tabs.addTab(self._arrhythmia_tab, tr("Arrhythmias"))
        self.setCentralWidget(self._tabs)

        # ─── Analysis state ────────────────────────────────────────
        # Current result dict from the last successful analyze / recompute.
        # Kept here (not on the viewer) because it outlives individual
        # viewer refreshes and is what the Ricalcola path needs to rebuild
        # Parametri + Aritmie.  ``None`` before the first file is loaded.
        self._current_result: dict | None = None
        self._current_csv_path: Path | None = None
        # Set when the active CSV was opened from the Studi panel
        # (double-click on a file row).  ``None`` when opened via
        # ``File → Apri CSV`` standalone.  Holds the Group so Ricalcola
        # can read ``group.config`` as source-of-truth (issue #6),
        # guaranteeing fingerprint consistency with the batch cache.
        self._current_group: Group | None = None
        # POSIX-style path relative to the study folder, set alongside
        # ``_current_group``.  Needed by Fase 3: after a Ricalcola we
        # call ``StudyPanel.update_entry(group_name, csv_relpath, ...)``
        # to write the fresh FileResult back into the batch cache, so
        # the Studi tree and the dose-response dialog reflect the
        # user's edit immediately.
        self._current_csv_relpath: str | None = None
        # Config used for the current analysis; kept at window scope so
        # the Impostazioni dialog (task #74) can mutate ONE shared
        # instance and see the change reflected on the next Apri CSV
        # (or the immediate re-analysis fired by ``applied``).
        # Ricalcola also re-uses the same instance, so per-file edits
        # made in the Battiti tab survive a settings change until the
        # user reloads.  ``amplifier_gain`` defaults to 1e4 in the library
        # (µECG-Pharma Digilent hardware) — the field is editable in the
        # Impostazioni dialog.
        from cardiac_fp_analyzer.config import AnalysisConfig
        self._config: AnalysisConfig = AnalysisConfig()
        # ``_current_config`` historically tracked "config used for the
        # last analysis"; now it points to the same ``_config`` object
        # after every ``_run_analysis``.  Ricalcola reads it.
        self._current_config: AnalysisConfig | None = None
        # Pure (pre-override) automatic detection captured at open time,
        # kept so ``Ricalcola`` can build a *cumulative* sidecar diff
        # against the automatic baseline — not against whatever beats
        # the last sidecar already injected (task #79).  Populated from
        # ``result['detection_info']['bi_detection']`` in ``_run_analysis``.
        self._bi_auto_at_open: np.ndarray | None = None
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
        self._signal_tab.chamber_changed.connect(self._on_chamber_changed)
        self._signal_tab.chip_map.electrode_clicked.connect(self._on_electrode_clicked)
        # Multi-electrode file kept in memory: (path, metadata, df, layout,
        # scores, summary). Chamber and electrode changes re-run the
        # analysis without reading the file again (v3.13).
        self._chip: dict | None = None

        # ─── Menu ──────────────────────────────────────────────────
        m_file = self.menuBar().addMenu(tr("&File"))
        act_open = QAction(tr("&Open recording..."), self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open)
        m_file.addAction(act_open)
        m_file.addSeparator()
        # Impostazioni dialog (task #74) — Cmd+, on macOS, Ctrl+, elsewhere.
        # Qt maps "Ctrl+," to "Cmd+," automatically on mac.
        act_settings = QAction(tr("&Settings..."), self)
        act_settings.setShortcut("Ctrl+,")
        act_settings.setMenuRole(QAction.PreferencesRole)  # mac: → app menu
        act_settings.triggered.connect(self._on_settings)
        m_file.addAction(act_settings)
        m_file.addSeparator()
        act_quit = QAction(tr("&Quit"), self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        # ─── Studi menu + dock (task #84 step 2) ───────────────────
        # The Studi panel is dockable on the left and hidden by
        # default — single-file Apri CSV is still the primary flow;
        # study-mode is opt-in for users running a dose-response.
        # Terminology: "Studio" (not "Progetto") aligns with GLP /
        # 21 CFR Part 58 and CDISC ``STUDYID`` — see the module
        # docstring of ``cardiac_fp_analyzer.study`` for rationale.
        self._study_panel = StudyPanel()
        self._study_dock = QDockWidget(tr("Studies"), self)
        self._study_dock.setObjectName("studies-dock")
        self._study_dock.setWidget(self._study_panel)
        self._study_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self._study_dock)
        self._study_dock.hide()

        # Double-click on a file in the tree → reuse the normal analyze
        # path, honouring whatever channel the user has selected.  The
        # file entry's own ``channel`` field is not plumbed through in
        # the PoC (step 2): that's a step-3 refinement once batch-per-
        # group lands.  The slot also stashes the Group so Ricalcola
        # can later read ``group.config`` (issue #6).
        self._study_panel.file_activated.connect(self._on_study_file_activated)
        self._study_panel.study_changed.connect(self._on_study_changed)

        m_studies = self.menuBar().addMenu(tr("&Studies"))
        act_toggle_studies = self._study_dock.toggleViewAction()
        act_toggle_studies.setText(tr("&Show studies panel"))
        act_toggle_studies.setShortcut("Ctrl+Shift+P")
        m_studies.addAction(act_toggle_studies)

        # ─── Revisione (v3.17): right dock with measures + per-beat table,
        # left dock with the review queue, menu and keyboard shortcuts.
        self._review_state = None
        self._review_panel = ReviewPanel()
        self._review_dock = QDockWidget(tr("Review"), self)
        self._review_dock.setObjectName("review-dock")
        self._review_dock.setWidget(self._review_panel)
        self._review_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self._review_dock)
        self._review_queue = ReviewQueue(lambda: self._config)
        self._queue_dock = QDockWidget(tr("Review queue"), self)
        self._queue_dock.setObjectName("review-queue-dock")
        self._queue_dock.setWidget(self._review_queue)
        self._queue_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._queue_dock)
        self._queue_dock.hide()
        # Initial widths: the trace keeps most of the window; both docks can be dragged wider.
        self.resizeDocks([self._review_dock, self._queue_dock], [430, 420], Qt.Horizontal)
        self._review_panel.undo_requested.connect(self._review_undo)
        self._review_panel.redo_requested.connect(self._review_redo)
        self._review_panel.recompute_requested.connect(self._on_recompute)
        self._review_panel.save_requested.connect(self._review_save)
        self._review_panel.view_changed.connect(self.viewer.set_view)
        self._review_panel.polarity_requested.connect(self._review_polarity)
        self._review_panel.beat_selected.connect(self._review_select)
        self._review_panel.restore_all_requested.connect(self._review_restore_all)
        self._review_panel.next_problem_requested.connect(self._review_next_problem)
        self._review_panel.approve_requested.connect(self._review_approve)
        self._review_panel.close_requested.connect(self._review_close)
        self._review_panel.reanalyse_requested.connect(self._review_reanalyse)
        self._review_panel.message.connect(lambda m: self.statusBar().showMessage(m, 5000))
        self._ignore_sidecar_once = False
        self._review_queue.open_requested.connect(self._open_from_queue)
        self.viewer.review_report.connect(lambda msg: self.statusBar().showMessage(msg, 8000))
        self.viewer.beats_changed.connect(self._review_panel.refresh)

        m_review = self.menuBar().addMenu(tr("&Review"))
        act_queue = self._queue_dock.toggleViewAction()
        act_queue.setText(tr("Review &queue"))
        act_queue.setShortcut("Ctrl+Shift+R")
        m_review.addAction(act_queue)
        act_panel = self._review_dock.toggleViewAction()
        act_panel.setText(tr("&Measures panel"))
        m_review.addAction(act_panel)
        m_review.addSeparator()
        act_undo = QAction(tr("&Undo correction"), self)
        act_undo.setShortcut(QKeySequence(QKeySequence.Undo))
        act_undo.triggered.connect(self._review_undo)
        m_review.addAction(act_undo)
        act_redo = QAction(tr("&Redo correction"), self)
        act_redo.setShortcut(QKeySequence(QKeySequence.Redo))
        act_redo.triggered.connect(self._review_redo)
        m_review.addAction(act_redo)
        act_save = QAction(tr("&Save corrections"), self)
        act_save.setShortcut(QKeySequence(QKeySequence.Save))
        act_save.triggered.connect(self._review_save)
        m_review.addAction(act_save)
        m_review.addSeparator()
        for text, key, slot in ((tr("Previous beat"), "Left", lambda: self._review_step(-1)),
                                (tr("Next beat"), "Right", lambda: self._review_step(1)),
                                (tr("Next stretch to check"), "Ctrl+G", lambda: self._review_next_problem(1)),
                                (tr("Previous stretch to check"), "Ctrl+Shift+G", lambda: self._review_next_problem(-1)),
                                (tr("Remove the selected beat"), "Delete", self._review_delete_selected),
                                (tr("Repolarisation: positive peak"), "P", lambda: self._review_polarity(1)),
                                (tr("Repolarisation: negative peak"), "N", lambda: self._review_polarity(-1))):
            a = QAction(text, self)
            a.setShortcut(QKeySequence(key))
            a.triggered.connect(slot)
            m_review.addAction(a)

        # ─── Visualizza menu ───────────────────────────────────────
        # Hosts the Tema submenu (Scuro / Chiaro).  Built as exclusive
        # checkable actions via a QActionGroup so the currently-active
        # theme always carries a ✓ — no radio-button widget needed.
        # QSettings persistence + live re-apply to every plot is handled
        # in ``theme.py``; the menu is just a UI over that model.
        m_view = self.menuBar().addMenu(tr("&View"))
        m_theme = m_view.addMenu(tr("&Theme"))
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        self._act_theme_dark = QAction(tr("Dark"), self)
        self._act_theme_dark.setCheckable(True)
        self._act_theme_dark.setData("dark")
        self._theme_group.addAction(self._act_theme_dark)
        m_theme.addAction(self._act_theme_dark)
        self._act_theme_light = QAction(tr("Light"), self)
        self._act_theme_light.setCheckable(True)
        self._act_theme_light.setData("light")
        self._theme_group.addAction(self._act_theme_light)
        m_theme.addAction(self._act_theme_light)
        # Reflect the persisted preference in the menu check-state at
        # startup.  Do this BEFORE wiring ``triggered`` so the initial
        # setChecked() doesn't fire a spurious save.
        current = theme.current_mode()
        (self._act_theme_dark if current == "dark"
         else self._act_theme_light).setChecked(True)
        self._theme_group.triggered.connect(self._on_theme_menu_triggered)

        # Language submenu (v3.19): English / Italiano, exclusive and checkable like the theme.  The
        # choice is saved at once in QSettings (``pyside_app.i18n``) and takes effect at the next start,
        # because every widget sets its texts when it is built.
        m_lang = m_view.addMenu(tr("&Language"))
        m_lang.setToolTip(tr("Language of menus, panels and messages (takes effect at the next start)"))
        self._lang_group = QActionGroup(self)
        self._lang_group.setExclusive(True)
        self._lang_actions = {}
        for code in _i18n.LANGUAGES:
            a = QAction(tr(_i18n.LANGUAGE_NAMES[code]), self)
            a.setCheckable(True)
            a.setData(code)
            self._lang_group.addAction(a)
            m_lang.addAction(a)
            self._lang_actions[code] = a
        self._lang_actions.get(_i18n.configured_language(), self._lang_actions["en"]).setChecked(True)
        self._lang_group.triggered.connect(self._on_language_menu_triggered)

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
            tr("Ready — File ▶ Open recording (CSV, MCS .h5, .npz or NWB) to begin.")
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
            tr("Recomputing ({0} beats)...").format(len(bi_edited))
        )
        QApplication.processEvents()

        # GH #6 Fase 2 — Group.config is source-of-truth for Ricalcola.
        # If the active file belongs to a Group (tracked via
        # ``_current_group`` set by ``_on_study_file_activated``), use
        # that Group's persisted config so the fingerprint of the
        # recomputed result matches the one the batch wrote in the
        # Studi cache.  Falls back to the global app config when the
        # file was opened standalone (outside a Study).
        recompute_config = (
            self._current_group.config
            if self._current_group is not None
            else self._current_config
        )

        try:
            from cardiac_fp_analyzer.analyze import recompute_from_beats
            if self._review_state is not None and not excluded:
                new_result = self._review_state.recompute(self._current_result, config=recompute_config)
            else:
                new_result = recompute_from_beats(
                    self._current_result, bi_edited,
                    config=recompute_config, verbose=False,
                )
        except Exception as e:   # noqa: BLE001 — surface any pipeline error
            QMessageBox.critical(
                self, tr("Recompute error"),
                f"{type(e).__name__}: {e}",
            )
            self.statusBar().showMessage(
                tr("Error during the recompute.")
            )
            return

        # Update the cached result so subsequent recomputes build on the
        # filtered/re-segmented state, not the original detection.
        self._current_result = new_result

        # Populate the Battiti tab's per-beat parameter map.  We pass
        # the full *submitted* beat list (``bi_edited``) as the analysed
        # set so rows whose beat was dropped by QC / ritmo / RR render
        # as "Scartato" instead of silently vanishing.  Then re-render
        # with the viewer's full beat list (which still contains every
        # user pick — we deliberately do NOT resync the viewer to the
        # filtered set, see design note below).
        self._beats_tab.set_result(new_result)
        self._beats_tab.set_last_analyzed(bi_edited)
        self._beats_tab.set_beats(
            self.viewer.get_beat_indices(),
            self.viewer.get_time_vector(),
        )

        # Design note (task #85): we intentionally *do not* call
        # ``viewer.set_result(new_result)``.  The viewer keeps showing
        # every triangle the user placed, including picks the pipeline
        # rejected — otherwise the user's click-picks "disappear" and
        # they lose any sense of what the pipeline is analysing.  The
        # mismatch between "triangles shown" and "beats in Parametri"
        # is surfaced via the Stato column (Scartato) and the status
        # bar delta below.

        # Refresh Parametri + Aritmie from the new result.
        self._params_tab.set_result(new_result)
        self._arrhythmia_tab.set_result(new_result)
        if self._review_state is not None:
            self.viewer.refresh_from_state()
            self._review_panel.set_state(self._review_state, new_result)

        self._signal_tab.set_recompute_pending(False)

        # GH #6 Fase 3 — write the recomputed result back into the
        # Studi batch cache so the tree and the dose-response dialog
        # pick up the user's edits immediately.  Only for files opened
        # from the Studi panel (``_current_group`` and
        # ``_current_csv_relpath`` are set together by
        # ``_on_study_file_activated``).  Standalone files have no
        # cache row to update.
        #
        # Channel canonicalisation is done inside ``update_entry`` by
        # looking up the FileEntry — critical, because the batch
        # stamps the fingerprint with ``fe.channel`` (``'auto'`` by
        # default) while the pipeline output reports the *resolved*
        # channel (``'EL1'``).  Earlier Fase 3 code passed the
        # resolved channel here and the two fingerprints diverged,
        # leaving the badge ● after Ricalcola.
        if (
            self._current_group is not None
            and self._current_csv_relpath is not None
        ):
            self._study_panel.update_entry(
                self._current_group.name,
                self._current_csv_relpath,
                new_result,
                config=recompute_config,
            )

        # Persist the user's corrections as a sidecar JSON next to the
        # CSV (task #79).  The diff is always computed against the pure
        # automatic detection captured at open time — so re-Ricalcola
        # after reopening the same file produces the same sidecar, and
        # an intermediate Ricalcola that ends up identical to the
        # automatic set deletes the sidecar (``save_overrides`` unlinks
        # on empty).  Failure to write is logged but does NOT abort the
        # recompute: the in-memory result is still valid, we just lose
        # persistence until the next Ricalcola attempt.
        sidecar_info: str = ""
        if (
            self._current_csv_path is not None
            and self._bi_auto_at_open is not None
        ):
            from cardiac_fp_analyzer.overrides import (
                diff_to_overrides,
                save_overrides,
            )
            fs = float(
                (new_result.get("metadata") or {}).get("sample_rate") or 0.0
            )
            if fs > 0:
                try:
                    ov = diff_to_overrides(
                        self._bi_auto_at_open, bi_edited, fs,
                    )
                    save_overrides(str(self._current_csv_path), ov)
                    if ov.is_empty():
                        sidecar_info = tr(" · corrections cleared")
                    else:
                        sidecar_info = tr(
                            " · saved +{0}/-{1}"
                        ).format(len(ov.added_s), len(ov.removed_s))
                except OSError as e:   # disk full, permissions, etc.
                    QMessageBox.warning(
                        self, tr("Saving corrections"),
                        tr(
                            "The corrections file could not be saved ({0}).  The recompute is still valid."
                        ).format(e),
                    )

        # Status-bar delta: the QC / rhythm / RR-outlier filters may
        # silently drop beats that the user had explicitly added. Show
        # the delta so the user knows when the pipeline rejected some
        # of their picks — otherwise a "why did my beat disappear?"
        # moment.  When the pipeline kept everything, fall back to the
        # simpler phrasing so the common case isn't noisy.
        #
        # Since #85 the viewer is NOT resynced to the filtered set, so
        # ``viewer.beat_count()`` still reflects the full user pick set.
        # Read the *kept* count directly from the new result.
        kept_bi = new_result.get("beat_indices_fpd")
        if kept_bi is None:
            kept_bi = new_result.get("beat_indices", [])
        n_in = int(len(bi_edited))
        n_out = int(len(kept_bi))
        if n_in > 0 and n_out < n_in:
            dropped = n_in - n_out
            self.statusBar().showMessage(tr(
                "Recompute completed — {0} beats included ({1} filtered by QC/rhythm/RR out of {2}){3}"
            ).format(n_out, dropped, n_in, sidecar_info))
        else:
            self.statusBar().showMessage(
                tr(
                    "Recompute completed — {0} beats included{1}"
                ).format(n_out, sidecar_info)
            )

    # ─── Revisione slots (v3.17) ──────────────────────────────────
    def _review_after_edit(self) -> None:
        self.viewer.refresh_from_state()
        self._review_panel.refresh()
        self._review_panel.select_row(self.viewer.selected_beat())

    def _review_undo(self) -> None:
        if self._review_locked():
            return
        if self._review_state is not None and self._review_state.undo():
            self._review_after_edit()
            self.statusBar().showMessage(tr("Correction undone"), 3000)

    def _review_redo(self) -> None:
        if self._review_locked():
            return
        if self._review_state is not None and self._review_state.redo():
            self._review_after_edit()
            self.statusBar().showMessage(tr("Correction redone"), 3000)

    def _review_step(self, step: int) -> None:
        if self._review_state is None:
            return
        self.viewer.select_next(step)
        self._review_panel.select_row(self.viewer.selected_beat())

    def _review_locked(self) -> bool:
        """True (and a status-bar message) when the current recording is closed."""
        st = self._review_state
        if st is not None and getattr(st, 'locked', False):
            self.statusBar().showMessage(tr("Recording closed: Unlock to change it"), 4000)
            return True
        return False

    def _review_persist_state(self) -> None:
        """Save the workflow state (and the corrections) and tell the queue."""
        st = self._review_state
        if st is None or self._current_csv_path is None:
            return
        try:
            st.save(str(self._current_csv_path))
        except OSError as exc:
            QMessageBox.warning(self, tr("Saving"), str(exc))
            return
        fi = (self._current_result or {}).get("file_info") or {}
        self._review_queue.mark_reviewed(str(self._current_csv_path), st.channel, fi.get("chamber"),
                                         state=st, result=self._current_result)
        self._review_panel.refresh()          # the beats did not change: no 'Ricalcola' pending
        self._update_review_title()

    def _update_review_title(self) -> None:
        st = self._review_state
        title = self.windowTitle().replace(" 🔒", "").replace(" ✓", "")
        if st is not None and getattr(st, 'locked', False):
            title += " 🔒"
        elif st is not None and getattr(st, 'approved', False):
            title += " ✓"
        self.setWindowTitle(title)

    def _review_reanalyse(self) -> None:
        """Redo the automatic analysis of the open recording, dropping the corrections held in memory."""
        st = self._review_state
        if st is None or self._current_csv_path is None:
            return
        if getattr(st, 'locked', False):
            self._review_locked()
            return
        if st.can_undo or st.changes().get('corrected'):
            ans = QMessageBox.question(
                self, tr("Redo analysis"),
                tr("The corrections made to this recording will be discarded (those saved in the "
                   ".review.json stay until you save again). Redo the automatic analysis?"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return
        self._ignore_sidecar_once = True
        channel = st.channel if st.channel in ("el1", "el2") else CHANNEL_AUTO
        self._run_analysis(str(self._current_csv_path), channel)
        self.statusBar().showMessage(tr("Automatic analysis redone: corrections in memory "
                                        "discarded (the .review.json stays until you save)"), 8000)

    def _review_approve(self, on: bool) -> None:
        st = self._review_state
        if st is None or getattr(st, 'locked', False):
            return
        if on:
            st.approve()
            msg = tr("Recording approved by {0}").format(st.approved_by)
        else:
            st.unapprove()
            msg = tr("Approval withdrawn")
        self._review_persist_state()
        self.statusBar().showMessage(msg, 5000)

    def _review_close(self, close: bool) -> None:
        st = self._review_state
        if st is None:
            return
        if close:
            st.close()
            msg = tr("Recording closed (locked)")
        else:
            st.reopen()
            msg = tr("Recording unlocked: it stays approved")
        self._review_persist_state()
        self.statusBar().showMessage(msg, 5000)

    def _review_next_problem(self, step: int = 1) -> None:
        """Centre the trace on the next highlighted stretch (missing beat, detections too close, no
        repolarisation) and say what it is in the status bar."""
        if self._review_state is None:
            return
        from .signal_viewer import PROBLEM_LABELS
        prob = self.viewer.goto_next_problem(step)
        if prob is None:
            self.statusBar().showMessage(tr("No highlighted stretch in this recording"), 4000)
            return
        self._review_panel.select_row(self.viewer.selected_beat())
        t = prob['start'] / self._review_state.fs
        self.statusBar().showMessage(tr("{0} at {1:.1f} s (beat #{2})").format(
            tr(PROBLEM_LABELS.get(prob['tipo'], prob['tipo'])), t, prob['k'] + 1), 6000)

    def _review_select(self, k: int) -> None:
        if self._review_state is None:
            return
        self.viewer.select_beat(k)
        self._tabs.setCurrentWidget(self._signal_tab)

    def _review_delete_selected(self) -> None:
        st = self._review_state
        k = self.viewer.selected_beat()
        if st is None or k is None or self._review_locked():
            return
        st.remove_beat(k)
        self.viewer.select_beat(min(k, st.n_beats() - 1) if st.n_beats() else None, center=False)
        self._review_after_edit()

    def _review_polarity(self, sign: int) -> None:
        st = self._review_state
        if st is None or self._review_locked():
            return
        rep = st.set_polarity(sign)
        self._review_after_edit()
        if rep:
            self.statusBar().showMessage(tr(
                "Repolarisation on the {0} peak: moved {1}, unchanged {2}, ambiguous {3}").format(
                "positivo" if sign > 0 else "negativo", rep.get("spostati", 0), rep.get("invariati", 0),
                rep.get("ambigui", 0)), 8000)

    def _review_restore_all(self) -> None:
        st = self._review_state
        if st is None or not st.vetoed or self._review_locked():
            return
        n = 0
        for v in list(st.vetoed):
            if st.restore_vetoed(v) is not None:
                n += 1
        self._review_after_edit()
        self.statusBar().showMessage(tr("{0} detections restored").format(n), 5000)

    def _review_save(self) -> None:
        st = self._review_state
        if st is None or self._current_csv_path is None:
            return
        try:
            p = st.save(str(self._current_csv_path))
        except OSError as exc:
            QMessageBox.warning(self, tr("Saving corrections"), str(exc))
            return
        fi = (self._current_result or {}).get("file_info") or {}
        # the queue's record (used by the Results export) follows the saved corrections
        self._review_queue.mark_reviewed(str(self._current_csv_path), st.channel, fi.get("chamber"),
                                         state=st, result=self._current_result)
        self.statusBar().showMessage(tr("Corrections saved to {0}").format(p.name), 5000)

    def _tissue_of(self, path: str, channel: str) -> str:
        """'chip E ch2' for an input listed in the review queue, '' otherwise."""
        queue = getattr(self, "_review_queue", None)
        if queue is None or not channel:
            return ""
        path = str(Path(path).resolve()) if Path(path).exists() else str(path)
        e = next(queue._find(path, str(channel).lower()), None)
        if not e or not e.get("chip"):
            return ""
        ch = e.get("camera")
        return f"{tr('chip')} {e['chip']}" + (f" ch{ch}" if ch not in (None, "") else "")

    def _open_from_queue(self, path: str, channel: str, chamber) -> None:
        """A double click in the review queue: open that recording (input or chamber)."""
        self._current_group = None
        self._current_csv_relpath = None
        self._chip = None
        if chamber:
            self._run_analysis(path, CHANNEL_AUTO)
            if self._chip is not None and self._signal_tab.chamber_choice() != chamber:
                self._on_chamber_changed(chamber)
        else:
            if channel in self._signal_tab._channel_values:
                self._signal_tab.set_channel(channel)
            self._run_analysis(path, channel)

    # ─── Actions ──────────────────────────────────────────────────
    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Open a recording (CSV, MCS HDF5, compact .npz or NWB)"),
            "",
            tr("Recordings (*.csv *.h5 *.npz *.nwb);;CSV files (*.csv);;MCS HDF5 (*.h5);;MCS "
               "compact (*.npz);;NWB (*.nwb);;All files (*)"),
        )
        if not path:
            return
        # Standalone open: the file is not associated with a Study /
        # Group, so Ricalcola will fall back to the global
        # ``self._config`` (issue #6) and write-back to the Studi
        # cache is skipped.
        self._current_group = None
        self._current_csv_relpath = None
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

        When the file was opened from a Study (GH #7, task #95), the
        new channel is also **persisted** into the owning
        :class:`FileEntry`.  ``StudyPanel.set_entry_channel`` additionally
        re-stamps the cached :class:`FileResult` fingerprint so the row
        gets the ``●`` stale badge — the visual cue that the group
        aggregate is now out of sync with the viewer's channel choice.
        The user then decides whether to Ricalcola (which refreshes the
        cache against the new channel and removes the badge).

        Persist happens *before* ``_run_analysis`` on purpose: the
        batch-fingerprint contract is "fe.channel is the single source
        of truth for what this file will analyse on", so we want the
        model to be consistent with the run even if the pipeline call
        blows up midway through.
        """
        if self._current_csv_path is None:
            return
        if self._current_group is not None and self._current_csv_relpath:
            # Best-effort persist; returns False when the choice didn't
            # actually change (auto→auto) or when the study / group /
            # file has disappeared underneath us.  Either way we still
            # want the viewer to re-run on the new channel.
            self._study_panel.set_entry_channel(
                self._current_group.name,
                self._current_csv_relpath,
                choice,
            )
        self._run_analysis(str(self._current_csv_path), channel=choice)

    def _on_chamber_changed(self, chamber: str) -> None:
        """Another chamber of the chip: electrode back to Auto, re-run."""
        if self._chip is None or self._current_csv_path is None:
            return
        self._signal_tab.set_chip(self._chip['layout'], chamber, self._chip['scores'], self._chip['summary'])
        self._run_analysis(str(self._current_csv_path), channel=CHANNEL_AUTO)

    def _on_electrode_clicked(self, electrode: str) -> None:
        """A click on the chip map: analyse that electrode (and its chamber)."""
        if self._chip is None or self._current_csv_path is None:
            return
        lay = self._chip['layout']
        ch = lay.chamber_of(electrode)
        if ch is None:
            return
        if ch.name != self._signal_tab.chamber_choice():
            self._signal_tab.set_chip(lay, ch.name, self._chip['scores'], self._chip['summary'])
        self._signal_tab.set_channel(electrode)
        self._run_analysis(str(self._current_csv_path), channel=electrode)

    def _load_chip(self, path: str) -> dict | None:
        """Load a multi-electrode recording once and score every electrode.

        Returns None for CSV files (two inputs) or files without a known
        layout, in which case the ordinary path is used."""
        from cardiac_fp_analyzer.chambers import layout_for
        from cardiac_fp_analyzer.channel_selection import quick_electrode_scores
        from cardiac_fp_analyzer.loader import electrode_columns, is_recording_file, load_recording
        if Path(path).suffix.lower() == '.csv' or not is_recording_file(path):
            return None
        self.statusBar().showMessage(tr("Reading {0}...").format(Path(path).name))
        QApplication.processEvents()
        metadata, df = load_recording(path)
        cols = electrode_columns(df)
        layout = layout_for(cols, getattr(self._config, 'chamber_layout', 'auto'))
        if layout is None:
            return None
        self.statusBar().showMessage(tr("Scoring the {0} electrodes...").format(len(cols)))
        QApplication.processEvents()
        stim = [e for c in layout.chambers for e in c.stimulation]
        scores = quick_electrode_scores(df, metadata['sample_rate'], cols, cfg=self._config, exclude=stim)
        summary = {}
        for c in layout.chambers:
            per = [scores[e]['period_ms'] for e in c.recording
                   if e in scores and np.isfinite(scores[e].get('score', np.nan))]
            summary[c.name] = (tr("{0} ms").format(f"{np.median(per):.0f}") if per
                               else tr("no beats"))
        return {'path': path, 'metadata': metadata, 'df': df, 'layout': layout, 'scores': scores, 'summary': summary}

    def _on_language_menu_triggered(self, action: QAction) -> None:
        """Save the interface language chosen in View → Language; it is applied at the next start."""
        code = action.data()
        if code in _i18n.LANGUAGES and code != _i18n.configured_language():
            _i18n.save_language(code)
            self.statusBar().showMessage(tr("The interface language changes at the next start of the application."), 8000)
            QMessageBox.information(self, tr("Language"),
                                    tr("The interface language changes at the next start of the application."))

    def _on_theme_menu_triggered(self, action: QAction) -> None:
        """Persist the selected theme and notify subscribers.

        Called from the Visualizza → Tema QActionGroup.  The action
        carries the mode string (``'dark'`` | ``'light'``) in its
        ``data()`` slot; ``theme.set_mode`` writes QSettings AND emits
        the ``theme_changed`` signal, so every subscribed plot
        repaints on its own without us having to enumerate them here.
        """
        mode = action.data()
        if mode in ("dark", "light"):
            theme.set_mode(mode)

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
            tr("Analysing: {0}...").format(Path(path).name)
        )
        QApplication.processEvents()

        # Import here (not at module top) so the window still opens even
        # if something in the scientific core is broken — easier to diagnose.
        try:
            from cardiac_fp_analyzer.analyze import analyze_single_file
            # Multi-electrode chip: the file is read once and kept in
            # memory; the analysis is restricted to the chosen chamber.
            if self._chip is None or self._chip['path'] != path:
                self._chip = self._load_chip(path)
                if self._chip is not None:
                    lay = self._chip['layout']
                    # start from the chamber holding the best electrode
                    best = max((v['score'], e) for e, v in self._chip['scores'].items()
                               if np.isfinite(v.get('score', np.nan))) if any(
                        np.isfinite(v.get('score', np.nan)) for v in self._chip['scores'].values()) else None
                    ch0 = lay.chamber_of(best[1]).name if best else lay.chambers[0].name
                    self._signal_tab.set_chip(lay, ch0, self._chip['scores'], self._chip['summary'])
                    if channel not in self._signal_tab._channel_values:
                        channel = CHANNEL_AUTO
                else:
                    self._signal_tab.set_chip(None, None, {})
            update, preloaded = None, None
            if self._chip is not None:
                lay = self._chip['layout']
                ch = lay[self._signal_tab.chamber_choice() or lay.chambers[0].name]
                update = {'chamber': ch.name, 'channel': ch.name, 'electrodes': list(ch.electrodes),
                          'stimulation_electrodes': list(ch.stimulation), 'chamber_layout': lay.name,
                          'two_tissue_file': True}
                preloaded = (self._chip['metadata'], self._chip['df'])
            # We pass ``self._config`` — the window-scope AnalysisConfig
            # that the Impostazioni dialog (task #74) mutates in place.
            # Ricalcola reads this same instance (via ``_current_config``),
            # so a settings change immediately propagates to both the
            # "Apri" and "Ricalcola" paths without extra plumbing.
            result = analyze_single_file(
                path, channel=channel, verbose=False, config=self._config,
                file_info_update=update, preloaded=preloaded,
            )
        except Exception as e:   # noqa: BLE001 — we want to show *any* error
            QMessageBox.critical(
                self, tr("Analysis error"),
                f"{type(e).__name__}: {e}",
            )
            self._clear_results(tr("Error during the analysis."))
            return

        if result is None:
            QMessageBox.warning(
                self, tr("No results"),
                tr("The pipeline returned None (see log)."),
            )
            self._clear_results(tr("The analysis returned nothing."))
            return

        # Cache the result for the Ricalcola path BEFORE ``viewer.set_result``
        # fires ``beats_changed`` — the guard relies on both the result
        # being set AND ``_suppress_pending`` being True.
        self._current_result = result
        self._current_csv_path = Path(path)
        self._current_config = self._config
        # Snapshot the pure automatic detection for the Ricalcola →
        # sidecar save path (task #79).  ``analyze.py`` always writes
        # ``detection_info['bi_detection']``; fall back to the final
        # ``beat_indices`` if a legacy analyzer somehow skipped it so
        # we never crash on KeyError here (sidecar save will just diff
        # against the post-pipeline set — not ideal but safe).
        det_info = result.get("detection_info") or {}
        bi_det = det_info.get("bi_detection")
        if bi_det is not None:
            self._bi_auto_at_open = np.asarray(bi_det, dtype=int)
        else:
            self._bi_auto_at_open = np.asarray(
                result.get("beat_indices", []), dtype=int,
            )
        # Populate the Battiti tab's per-beat parameter map BEFORE the
        # viewer fires beats_changed (which routes through
        # ``_refresh_beats_tab → _beats_tab.set_beats``).  If we don't,
        # the first render shows "—" in RR/Stato until the user touches
        # something.
        self._beats_tab.set_result(result)
        # Review state (v3.17): the software's proposal plus any saved corrections.
        from cardiac_fp_analyzer.review import ReviewState
        try:
            self._review_state = ReviewState.from_result(result, self._config)
            if self._ignore_sidecar_once:
                self._ignore_sidecar_once = False          # 'Rifai l'analisi': the saved corrections stay on disk
            elif ReviewState.load_into(self._review_state, path):
                self.statusBar().showMessage(tr("Saved corrections loaded for {0}").format(
                    self._review_state.channel), 5000)
        except Exception as exc:   # noqa: BLE001 — the viewer must still open
            logger.exception("review state")
            self._review_state = None
            self.statusBar().showMessage(tr("Review not available: {0}").format(exc), 8000)
        self.viewer.set_review_state(self._review_state)
        self._suppress_pending = True
        try:
            self.viewer.set_result(result)
        finally:
            self._suppress_pending = False
        self._params_tab.set_result(result)
        self._arrhythmia_tab.set_result(result)
        self._review_panel.set_state(self._review_state, result)
        self._review_panel.set_view_buttons(self.viewer.view())
        self._update_review_title()
        # Fresh load: Parametri + Aritmie are in sync, so clear any
        # leftover "pending" styling and enable the Ricalcola button.
        self._signal_tab.set_recompute_enabled(True)
        self._signal_tab.set_recompute_pending(False)

        # Reflect what channel the pipeline actually analysed — useful
        # when the user asked for Auto and wants to know which electrode
        # was picked by ``select_best_channel``.
        analyzed = result.get("file_info", {}).get("analyzed_channel", "")
        self._signal_tab.set_analyzed_channel(analyzed)
        # an explicit request (queue row, EL1/EL2 chosen) is reflected in the combo — ``set_chip(None)``
        # above resets it to Auto on every open; a request for Auto keeps Auto with the "→ ELx" hint
        if channel != CHANNEL_AUTO and channel in self._signal_tab._channel_values:
            self._signal_tab.set_channel(channel)
        chamber = result.get("file_info", {}).get("chamber")
        self._signal_tab.chip_map.set_selected(analyzed or None)
        self._update_window_title(
            Path(path).name, f"{tr('chamber')} {chamber} · {analyzed}" if chamber and analyzed else analyzed,
        )

        # Make sure the Segnale tab is visible after a successful load
        # (the user expects to see the plot right away, even if they had
        # clicked around the placeholder tabs before opening).
        self._tabs.setCurrentWidget(self._signal_tab)
        ch_label = (
            _channel_label(analyzed) if analyzed else _channel_label(channel)
        )
        if channel == CHANNEL_AUTO and analyzed:
            ch_suffix = tr("{0} (auto)").format(ch_label)
        else:
            ch_suffix = ch_label
        # a file with two tissues: say which tissue this input is — the channel combo switches tissue
        # there, and the review state (approval, closure) belongs to the input, not to the queue row
        tissue = self._tissue_of(path, analyzed or channel)
        if tissue:
            ch_suffix += f" ({tissue})"
        self.statusBar().showMessage(
            tr("{0} — {1} — {2} beats included").format(
                Path(path).name, ch_suffix, self.viewer.beat_count(),
            )
        )

    def _clear_results(self, status_msg: str) -> None:
        """Reset all downstream state after a failed / empty analysis.

        Extracted to avoid the error-handling divergence that the
        previous ``_on_open`` had two near-identical copies of.
        """
        self._review_state = None
        self.viewer.set_review_state(None)
        self._review_panel.set_state(None, None)
        self._params_tab.set_result(None)
        self._arrhythmia_tab.set_result(None)
        self._beats_tab.set_result(None)
        self._current_result = None
        self._current_csv_path = None
        self._current_config = None
        # Drop the Group reference too — next successful open
        # (studio or standalone) will set it anew (issue #6).
        self._current_group = None
        self._current_csv_relpath = None
        self._signal_tab.set_recompute_enabled(False)
        self._signal_tab.set_analyzed_channel(None)
        self._update_window_title(None, None)
        self.statusBar().showMessage(status_msg)

    def _on_study_file_activated(
        self, path: str, group_name: str, csv_relpath: str,
    ) -> None:
        """Handle double-click on a file row in the Studi panel (issue #6).

        Looks up the Group by name and stashes both the reference in
        ``self._current_group`` and the per-file ``csv_relpath`` so that
        ``_on_recompute`` (Fase 2) can use ``group.config`` as
        source-of-truth — matching what the batch analyzer used for
        this Group, so the result fingerprint stays consistent with
        the cache entry — and (Fase 3) write the recomputed result
        back into the Studi cache via
        :meth:`StudyPanel.update_entry`, unblocking the dose-response
        refresh.

        Falls back gracefully if the Group has been removed between the
        emit and the slot (rare race — study_changed is the cleaner
        signal for that) by leaving ``_current_group = None``; Ricalcola
        will then fall back to the global config, same as standalone.
        """
        study = self._study_panel.current_study()
        if study is not None:
            self._current_group = find_group(study, group_name)
            self._current_csv_relpath = (
                csv_relpath if self._current_group is not None else None
            )
        else:
            self._current_group = None
            self._current_csv_relpath = None
        # Pick the channel from the FileEntry when available (GH #7,
        # task #95) — the Study model owns the per-file preference,
        # so opening a file should respect what the user picked last
        # time they looked at it, not whatever the combo happens to
        # show right now.  Falls back to the current combo value (and
        # thereby to Auto on the first open of a session) when the
        # file entry isn't reachable — preserves prior behaviour.
        #
        # Case note: ``FileEntry.channel`` is stored in batch-style
        # uppercase (``'auto'``/``'EL1'``/``'EL2'`` — see
        # ``StudyPanel.set_entry_channel`` and ``_BatchWorker``), while
        # the Signal-tab combo and ``analyze_single_file`` both speak
        # lowercase (``'auto'``/``'el1'``/``'el2'``).  We lowercase
        # here to bridge the two conventions; unknown values fall back
        # to Auto the same way ``_CHANNEL_VALUES`` enforces.
        channel = self._signal_tab.channel_choice()
        if self._current_group is not None and self._current_csv_relpath:
            fe = next(
                (
                    f for f in self._current_group.files
                    if f.csv_relpath == self._current_csv_relpath
                ),
                None,
            )
            if fe is not None and fe.channel:
                combo_value = fe.channel.lower()
                if combo_value in ('auto', 'el1', 'el2'):
                    channel = combo_value
                    # Sync the combo so the user sees the channel that
                    # will actually be analysed.  set_channel blocks
                    # the ``channel_changed`` signal, so this does NOT
                    # trigger a second ``_on_channel_changed``
                    # round-trip.
                    self._signal_tab.set_channel(combo_value)
        self._run_analysis(path, channel=channel)

        # ── Fix #3 (GH #7 followup) — clear ● after a deep-dive ──────
        # Double-clicking a row in the Studi tree is an implicit
        # "Ricalcola this file on the currently-configured channel".
        # When the user had switched channel from the Signal combo,
        # the Studi cache carries a stale ● fingerprint anchored to
        # the OLD channel; after the fresh analysis completes
        # (``_run_analysis`` populates ``self._current_result``), we
        # mirror the Ricalcola behaviour and push the new result into
        # the cache via ``update_entry``.  ``_is_stale`` then sees a
        # matching fingerprint on the next repaint and the row flips
        # back from ● to ✓ — no user action required.
        #
        # Guards:
        #   * ``_current_result`` may be None if the pipeline errored;
        #     no cache update in that case (the row keeps its ● and
        #     the error will surface in the Signal tab).
        #   * ``_current_group`` / ``_current_csv_relpath`` may be None
        #     if the group was removed between the emit and this slot;
        #     skip silently — the cache has already been pruned.
        if (
            self._current_result is not None
            and self._current_group is not None
            and self._current_csv_relpath
        ):
            self._study_panel.update_entry(
                self._current_group.name,
                self._current_csv_relpath,
                self._current_result,
                self._current_group.config,
            )

    def _on_study_changed(self) -> None:
        """Status-bar hint when a study is opened / created (task #84)."""
        s = self._study_panel.current_study()
        if s is None:
            return
        self.statusBar().showMessage(
            tr("Study: {0} — {1} groups").format(
                s.name, len(s.groups),
            )
        )

    def _update_window_title(
        self, filename: str | None, channel: str | None,
    ) -> None:
        """Reflect the current file + analysed channel in the window title.

        ``None`` filename resets to the app name.  When the channel is
        known it's appended so the user sees "chipD_ch1.csv — EL1"
        without having to look at the Signal tab.
        """
        base = tr("Cardiac FP Analyzer")
        if not filename:
            self.setWindowTitle(base)
            return
        if channel:
            self.setWindowTitle(f"{filename} — {channel if ' ' in channel or channel.startswith('E') else channel.upper()} · {base}")
        else:
            self.setWindowTitle(f"{filename} · {base}")


def main() -> int:
    # ``QSettings`` (used by ``pyside_app.theme`` and future prefs)
    # derives its backing store path from the application's
    # organisation + application name.  Set both here, before any
    # widget is created, so the theme module can read/write the user
    # preference without surprises.
    QApplication.setOrganizationName("CardiacFP")
    QApplication.setOrganizationDomain("cardiac-fp-analyzer")
    QApplication.setApplicationName("Cardiac FP Analyzer")
    app = QApplication(sys.argv)
    # interface language (QSettings ``ui/language``; English by default) — before any widget is built
    _i18n.apply_language(app)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
