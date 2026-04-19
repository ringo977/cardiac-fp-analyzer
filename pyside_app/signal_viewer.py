"""pyqtgraph-based signal viewer with contextual-popup beat picking.

Data contract (keys read from the result dict returned by
:func:`cardiac_fp_analyzer.analyze.analyze_single_file`):

* ``filtered_signal``  : 1-D ndarray, amplitude (µV)
* ``time_vector``      : 1-D ndarray, time (s)  — sampled at constant fs
* ``beat_indices_fpd`` : 1-D int ndarray, indices of included (depol) beats
* ``all_params``       : list of per-beat dicts; we read ``repol_peak_global_idx``
                         from each to draw the repolarization-peak markers.

Editing model (Phase 0, ADR-0001):
- Viewer has two interaction modes, driven by the main window's radio:
  "view" (read-only) and "edit" (click-to-mutate).
- In edit mode, a click opens a *contextual* QMenu with the actions that
  make sense for that click: remove if a marker is near, add-here if not.
- All edits mutate in-memory copies of the beat/repol index arrays and
  redraw. Recomputation of FPD/summary is Phase 1 (task #72).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QMenu


# How close (in seconds) a click must be to an existing marker to be
# treated as "on" that marker (i.e. the menu offers Remove instead of Add).
_SNAP_WINDOW_S = 0.200


class SignalViewer(pg.PlotWidget):
    """Single-plot viewer with click-to-edit via contextual menu."""

    # Emitted when the mouse hovers the plot; the payload is a short,
    # already-formatted string the main window can drop into a label.
    # Empty string means "the cursor has left the viewer".
    hover_info = Signal(str)

    # Emitted whenever the beat set changes — after set_result() loads a
    # new file, after every add/remove in edit mode.  No payload: the
    # receiver queries the viewer for the current state.  Decoupling
    # the signal from the data lets us extend the viewer later without
    # having to rewire every slot.
    beats_changed = Signal()

    def __init__(self, mode_getter: Callable[[], str]):
        super().__init__()
        self._mode_getter = mode_getter

        self._signal: Optional[np.ndarray] = None
        self._raw: Optional[np.ndarray] = None
        self._time: Optional[np.ndarray] = None
        self._fs: float = 1000.0
        # Mutable arrays of sample indices into self._signal.
        self._beats: np.ndarray = np.array([], dtype=int)
        self._repol_idx: np.ndarray = np.array([], dtype=int)
        # Small vertical offset used to lift markers a touch off the signal
        # line so they don't overlap it.
        self._marker_offset: float = 0.0
        # Raw-signal overlay is off by default — matches the Streamlit
        # behaviour the user is migrating from and keeps the plot legible
        # on first load; toggled on demand via set_show_raw().
        self._show_raw: bool = False

        # ─── Style ────────────────────────────────────────────────
        self.setBackground("#0e1117")
        for ax_name in ("left", "bottom"):
            ax = self.getAxis(ax_name)
            ax.setPen("#cccccc")
            ax.setTextPen("#cccccc")
        self.setLabel("bottom", self.tr("Tempo (s)"))
        self.setLabel("left", self.tr("Ampiezza (µV)"))
        self.showGrid(x=True, y=True, alpha=0.2)

        # ─── Plot items (created once, data updated in place) ─────
        # Raw line is drawn BEFORE the filtered line so that, when both
        # are visible, the filtered signal sits on top — the user's
        # primary reference — and the raw trace reads as a faint
        # "original" backdrop.  Semi-transparent grey keeps it readable
        # without fighting the filtered line for attention.
        self._raw_line = self.plot(pen=pg.mkPen((180, 180, 180, 110), width=1))
        self._line = self.plot(pen=pg.mkPen("#4a9eff", width=1))
        self._beats_scatter = pg.ScatterPlotItem(
            symbol="t",   # down-pointing triangle (depol)
            size=12,
            brush=pg.mkBrush("#ff4b4b"),
            pen=pg.mkPen(None),
        )
        self.addItem(self._beats_scatter)
        self._repol_scatter = pg.ScatterPlotItem(
            symbol="t1",  # up-pointing triangle (repol)
            size=10,
            brush=pg.mkBrush("#2ecc71"),
            pen=pg.mkPen(None),
        )
        self.addItem(self._repol_scatter)

        # Mouse clicks on the scene → edit menu (in edit mode).
        self.scene().sigMouseClicked.connect(self._on_mouse_click)

        # Mouse moves → hover-info signal.  Rate-limit with SignalProxy
        # so we don't re-emit on every pixel (pyqtgraph fires this on
        # every mouse-move event, which on a Retina display is a lot).
        self._hover_proxy = pg.SignalProxy(
            self.scene().sigMouseMoved,
            rateLimit=60,   # Hz cap → one update every ~16 ms
            slot=self._on_mouse_moved,
        )

    # ─── Public API ───────────────────────────────────────────────
    def set_result(self, result: dict) -> None:
        """Load a new analysis result and redraw from scratch."""
        self._signal = np.asarray(result["filtered_signal"], dtype=float)
        # Raw signal is optional — analyze_single_file usually returns
        # it, but we don't hard-require it so the viewer stays useful
        # for pipelines that only produce a filtered trace.
        raw = result.get("raw_signal")
        self._raw = np.asarray(raw, dtype=float) if raw is not None else None
        self._time = np.asarray(result["time_vector"], dtype=float)
        if len(self._time) > 1:
            dt = float(np.median(np.diff(self._time)))
            self._fs = (1.0 / dt) if dt > 0 else 1000.0

        self._beats = np.asarray(
            result.get("beat_indices_fpd", []), dtype=int
        ).copy()
        self._repol_idx = self._extract_repol_indices(result)

        amp = float(np.nanmax(self._signal) - np.nanmin(self._signal))
        self._marker_offset = 0.03 * amp if amp > 0 else 1.0

        self._redraw()
        self.autoRange()
        # Let any listeners (e.g. the Battiti tab) refresh on file load.
        self.beats_changed.emit()

    def set_show_raw(self, show: bool) -> None:
        """Toggle the raw-signal overlay.  Safe to call with no data."""
        self._show_raw = bool(show)
        self._redraw_raw()

    def beat_count(self) -> int:
        return int(len(self._beats))

    def repol_count(self) -> int:
        return int(len(self._repol_idx))

    # ─── Read-only accessors (for tabs that want to render our state) ─
    def get_beat_indices(self) -> np.ndarray:
        """Copy of the current depol-beat sample indices."""
        return self._beats.copy()

    def get_time_vector(self) -> Optional[np.ndarray]:
        """The currently loaded time vector, or ``None``."""
        return self._time

    def get_fs(self) -> float:
        return float(self._fs)

    # ─── View control (used by toolbar buttons + Battiti jump) ──────
    def zoom_by(self, factor: float) -> None:
        """Zoom the X-axis around its current center.

        ``factor > 1`` zooms *out* (wider window), ``factor < 1`` zooms
        *in*.  The Y-axis is left alone so the signal stays at full
        height — users care about horizontal detail much more than
        vertical on these traces.  Clamped to the signal's time range.
        """
        if self._time is None or len(self._time) < 2:
            return
        vb = self.getPlotItem().vb
        (x_lo, x_hi), _ = vb.viewRange()
        c = 0.5 * (x_lo + x_hi)
        half = 0.5 * (x_hi - x_lo) * max(1e-4, float(factor))
        t_min = float(self._time[0])
        t_max = float(self._time[-1])
        new_lo = max(t_min, c - half)
        new_hi = min(t_max, c + half)
        # Keep a minimum visible window so a chain of "zoom in" clicks
        # eventually stops instead of collapsing to a point.
        min_span = 2.0 / self._fs   # 2 samples
        if new_hi - new_lo < min_span:
            return
        vb.setXRange(new_lo, new_hi, padding=0)

    def fit_view(self) -> None:
        """Reset the view to show the full signal (X + Y)."""
        if self._time is None or self._signal is None:
            return
        self.autoRange()

    def center_on(self, t_s: float) -> None:
        """Scroll the X-axis so ``t_s`` is at the center of the view,
        preserving the current zoom level (width).  Clamped to the
        signal's time range so we never show an empty area beyond the
        recording.  Silently a no-op if no data is loaded.
        """
        if self._time is None or len(self._time) < 2:
            return
        vb = self.getPlotItem().vb
        (x_lo, x_hi), _ = vb.viewRange()
        half = 0.5 * max(0.0, x_hi - x_lo)
        if half <= 0:
            # Fallback width if the view range is degenerate.
            half = 1.0
        t_min = float(self._time[0])
        t_max = float(self._time[-1])
        new_lo = max(t_min, t_s - half)
        new_hi = min(t_max, t_s + half)
        if new_hi <= new_lo:
            new_lo, new_hi = t_min, t_max
        vb.setXRange(new_lo, new_hi, padding=0)

    # ─── Data extraction helpers ──────────────────────────────────
    @staticmethod
    def _extract_repol_indices(result: dict) -> np.ndarray:
        """Pull repolarization-peak global indices out of ``all_params``.

        The pipeline stores per-beat parameters under ``all_params`` as a
        list of dicts; the key ``repol_peak_global_idx`` is the absolute
        sample index into ``filtered_signal`` (set to ``None`` for beats
        without a valid repolarization peak). See
        ``cardiac_fp_analyzer/parameters.py`` for where this is populated.
        """
        all_p = result.get("all_params")
        if not all_p:
            return np.array([], dtype=int)
        try:
            raw = [p.get("repol_peak_global_idx") for p in all_p]
        except Exception:
            return np.array([], dtype=int)
        out: list[int] = []
        for v in raw:
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                out.append(int(fv))
        return np.asarray(out, dtype=int) if out else np.array([], dtype=int)

    # ─── Drawing ──────────────────────────────────────────────────
    def _redraw_raw(self) -> None:
        """Update (or clear) the raw overlay without touching the rest.

        Called both from the full redraw and from ``set_show_raw`` so
        that toggling the checkbox doesn't force a recomputation of
        the scatter items.
        """
        if (self._show_raw and self._raw is not None
                and self._time is not None
                and len(self._raw) == len(self._time)):
            self._raw_line.setData(self._time, self._raw)
        else:
            self._raw_line.setData([], [])

    def _redraw(self) -> None:
        if self._signal is None or self._time is None:
            return
        self._line.setData(self._time, self._signal)
        self._redraw_raw()

        # Depolarization markers (red, down-triangle) — placed just above
        # each beat's signal value so they track the actual spike.
        if len(self._beats) > 0:
            safe = self._beats[
                (self._beats >= 0) & (self._beats < len(self._signal))
            ]
            xs = self._time[safe]
            ys = self._signal[safe] + self._marker_offset
            self._beats_scatter.setData(xs, ys)
        else:
            self._beats_scatter.setData([], [])

        # Repolarization markers (green, up-triangle) — same principle.
        if len(self._repol_idx) > 0:
            safe = self._repol_idx[
                (self._repol_idx >= 0) & (self._repol_idx < len(self._signal))
            ]
            xs = self._time[safe]
            ys = self._signal[safe] + self._marker_offset
            self._repol_scatter.setData(xs, ys)
        else:
            self._repol_scatter.setData([], [])

    # ─── Hover (tooltip info) ────────────────────────────────────
    def _on_mouse_moved(self, evt) -> None:
        """Emit a one-line hover summary, or an empty string if the
        cursor has left the plot area / no data is loaded yet.

        SignalProxy wraps the original sigMouseMoved payload in a
        1-tuple — the first (and only) element is the QPointF in scene
        coordinates.
        """
        if self._signal is None or self._time is None:
            self.hover_info.emit("")
            return
        try:
            pos = evt[0]
        except (TypeError, IndexError):
            return
        vb = self.getPlotItem().vb
        # The scene includes the axes and margins; only emit when the
        # cursor is actually over the data area.
        if not vb.sceneBoundingRect().contains(pos):
            self.hover_info.emit("")
            return

        view_pt = vb.mapSceneToView(pos)
        x_s = float(view_pt.x())

        # Nearest sample in time.
        i = int(np.argmin(np.abs(self._time - x_s)))
        if not (0 <= i < len(self._signal)):
            self.hover_info.emit("")
            return
        t = float(self._time[i])
        y = float(self._signal[i])

        # Nearest beat (if any).  Offset is in ms, signed so the user
        # can tell whether the cursor is before (-) or after (+) the
        # beat marker.
        extra = ""
        if len(self._beats) > 0:
            nb = int(np.argmin(np.abs(self._beats - i)))
            off_ms = (i - int(self._beats[nb])) * 1000.0 / self._fs
            extra = f"  |  #{nb + 1} ({off_ms:+.0f} ms)"

        self.hover_info.emit(
            f"t = {t:.3f} s  |  A = {y:.1f} µV{extra}"
        )

    # ─── Click / edit-menu ────────────────────────────────────────
    def _on_mouse_click(self, ev) -> None:
        if self._signal is None or self._time is None:
            return
        if ev.button() != Qt.LeftButton:
            return
        if self._mode_getter() != "edit":
            return

        pt = self.getPlotItem().vb.mapSceneToView(ev.scenePos())
        x_click_s = float(pt.x())
        self._show_edit_menu(x_click_s)
        ev.accept()

    def _show_edit_menu(self, x_click_s: float) -> None:
        near_depol_i = self._nearest_marker_i(self._beats, x_click_s)
        near_repol_i = self._nearest_marker_i(self._repol_idx, x_click_s)

        menu = QMenu(self)
        if near_depol_i is not None:
            act = menu.addAction(self.tr("Rimuovi depolarizzazione"))
            act.triggered.connect(
                lambda _checked=False, i=near_depol_i: self._remove_depol_i(i)
            )
        else:
            act = menu.addAction(self.tr("Aggiungi depolarizzazione (qui)"))
            act.triggered.connect(
                lambda _checked=False, x=x_click_s: self._add_depol_at(x)
            )

        if near_repol_i is not None:
            act = menu.addAction(self.tr("Rimuovi ripolarizzazione"))
            act.triggered.connect(
                lambda _checked=False, i=near_repol_i: self._remove_repol_i(i)
            )
        else:
            act = menu.addAction(self.tr("Aggiungi ripolarizzazione (qui)"))
            act.triggered.connect(
                lambda _checked=False, x=x_click_s: self._add_repol_at(x)
            )

        menu.addSeparator()
        menu.addAction(self.tr("Annulla"))   # no handler — dismisses the menu

        menu.exec(QCursor.pos())

    def _nearest_marker_i(
        self, arr: np.ndarray, x_click_s: float
    ) -> Optional[int]:
        """Index (into arr) of the marker nearest to x_click_s, or None
        if there are no markers or the nearest is outside the snap window.
        """
        if self._time is None or len(arr) == 0:
            return None
        dists = np.abs(self._time[arr] - x_click_s)
        nearest = int(np.argmin(dists))
        if float(dists[nearest]) > _SNAP_WINDOW_S:
            return None
        return nearest

    # ─── Mutations (depol) ────────────────────────────────────────
    def _add_depol_at(self, x_click_s: float) -> None:
        """Snap to the local max of |signal| within ±SNAP_WINDOW and add."""
        idx = self._snap_to_local_extremum(x_click_s, use_abs=True)
        if idx is None:
            return
        # Dedupe: if a depol marker is already within half a snap-window here,
        # do nothing so double-clicks don't pile up.
        if self._has_marker_within(self._beats, idx):
            return
        self._beats = np.sort(np.append(self._beats, idx))
        self._redraw()
        self.beats_changed.emit()

    def _remove_depol_i(self, i_in_arr: int) -> None:
        self._beats = np.delete(self._beats, i_in_arr)
        self._redraw()
        self.beats_changed.emit()

    # ─── Mutations (repol) ────────────────────────────────────────
    def _add_repol_at(self, x_click_s: float) -> None:
        """Snap to the local extremum of the *raw* signal in the window.

        Repolarization peaks can be either positive (T+) or negative (T-),
        so we pick whichever has the larger absolute deflection within the
        snap window. The difference from the depol rule is mainly semantic
        — both use |signal|'s argmax — but kept as a separate method in
        case we later want to constrain repol search to AFTER the nearest
        depol beat (Phase 1).
        """
        idx = self._snap_to_local_extremum(x_click_s, use_abs=True)
        if idx is None:
            return
        if self._has_marker_within(self._repol_idx, idx):
            return
        self._repol_idx = np.sort(np.append(self._repol_idx, idx))
        self._redraw()
        self.beats_changed.emit()

    def _remove_repol_i(self, i_in_arr: int) -> None:
        self._repol_idx = np.delete(self._repol_idx, i_in_arr)
        self._redraw()
        self.beats_changed.emit()

    # ─── Low-level snap/dedupe helpers ────────────────────────────
    def _snap_to_local_extremum(
        self, x_click_s: float, use_abs: bool
    ) -> Optional[int]:
        """Find the sample index of the local extremum within ±window."""
        if self._signal is None or self._time is None:
            return None
        win = max(1, int(_SNAP_WINDOW_S * self._fs))
        i_click = int(np.argmin(np.abs(self._time - x_click_s)))
        lo = max(0, i_click - win)
        hi = min(len(self._signal), i_click + win + 1)
        if hi <= lo:
            return None
        segment = self._signal[lo:hi]
        if use_abs:
            local = int(np.argmax(np.abs(segment)))
        else:
            local = int(np.argmax(segment))
        return lo + local

    def _has_marker_within(self, arr: np.ndarray, idx: int) -> bool:
        """True if ``arr`` already has a marker within half a snap-window
        of ``idx`` (prevents near-duplicate insertions on double clicks)."""
        if len(arr) == 0:
            return False
        win = max(1, int(_SNAP_WINDOW_S * self._fs))
        return bool(np.min(np.abs(arr - idx)) < win // 2)
