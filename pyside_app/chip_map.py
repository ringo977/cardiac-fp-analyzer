"""Chip map: the electrodes of a multi-chamber chip, one row per chamber.

Each chamber is drawn as its electrodes in physical order along the tissue
channel: the two stimulation pairs at the ends (grey, hatched) and the
recording electrodes in between, coloured by the quick electrode score
(``channel_selection.quick_electrode_scores``: dark = no beats, bright =
strong spikes, regular rhythm, visible repolarisation). The analysed
electrode has a white frame; a click on an electrode asks for its analysis.

The widget is passive: ``set_chip`` gives it the layout, the scores and
the per-chamber summary, ``set_selected`` the electrode in use.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

CELL = 22          # px per electrode
GAP = 3
ROW_H = 30
LABEL_W = 170      # chamber label on the left
_STIM = QColor('#4a4a52')
_FLAT = QColor('#2b2b33')
_SEL = QColor('#ffffff')


def _score_color(score: float, lo: float, hi: float) -> QColor:
    """Dark blue → cyan → yellow, like a short viridis ramp."""
    if score is None or not math.isfinite(score):
        return _FLAT
    t = 0.0 if hi <= lo else max(0.0, min(1.0, (score - lo) / (hi - lo)))
    stops = [(0.0, (40, 60, 120)), (0.5, (40, 160, 190)), (1.0, (240, 220, 80))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            return QColor(*(int(a + (b - a) * u) for a, b in zip(c0, c1)))
    return QColor(*stops[-1][1])


class ChipMapWidget(QWidget):
    electrode_clicked = Signal(str)       # electrode label ('E24')

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = None
        self._scores: dict = {}
        self._summary: dict = {}          # chamber name -> text ('1407 ms')
        self._selected: str | None = None
        self._cells: list[tuple[QRectF, str]] = []
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setVisible(False)

    # ── API ────────────────────────────────────────────────────────
    def set_chip(self, layout, scores: dict, summary: dict | None = None) -> None:
        self._layout = layout
        self._scores = dict(scores or {})
        self._summary = dict(summary or {})
        self.setVisible(layout is not None)
        if layout is not None:
            n = max(len(c.electrodes) for c in layout.chambers)
            self.setFixedHeight(ROW_H * len(layout.chambers) + 8)
            self.setMinimumWidth(LABEL_W + n * (CELL + GAP) + 10)
        self.update()

    def clear(self) -> None:
        self.set_chip(None, {}, {})

    def set_selected(self, label: str | None) -> None:
        self._selected = label
        self.update()

    # ── Painting ───────────────────────────────────────────────────
    def _order(self, chamber):
        """Electrodes in physical order: first stimulation pair, the row of
        recording electrodes, the last stimulation pair."""
        stim = list(chamber.stimulation)
        return stim[:2] + list(chamber.row) + stim[2:]

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API)
        if self._layout is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        font = QFont(self.font())
        font.setPointSize(max(8, font.pointSize() - 1))
        p.setFont(font)
        finite = [v['score'] for v in self._scores.values() if v.get('score') is not None and math.isfinite(v['score'])]
        lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
        lo = min(lo, 0.0)
        self._cells = []
        y = 4
        for ch in self._layout.chambers:
            # chamber label
            p.setPen(QPen(QColor('#cfcfd8')))
            text = f'camera {ch.name}'
            if self._summary.get(ch.name):
                text += f'  ·  {self._summary[ch.name]}'
            p.drawText(QRectF(4, y, LABEL_W - 8, ROW_H - 8), Qt.AlignVCenter | Qt.AlignLeft, text)
            x = LABEL_W
            for e in self._order(ch):
                r = QRectF(x, y, CELL, CELL)
                sc = self._scores.get(e, {})
                if e in ch.stimulation:
                    p.setBrush(QBrush(_STIM, Qt.BDiagPattern))
                    p.setPen(QPen(_STIM.lighter(130), 1))
                else:
                    p.setBrush(QBrush(_score_color(sc.get('score'), lo, hi)))
                    p.setPen(QPen(QColor('#1e1e24'), 1))
                p.drawRoundedRect(r, 3, 3)
                if e == self._selected:
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(_SEL, 2))
                    p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 3, 3)
                self._cells.append((r, e))
                x += CELL + GAP
            y += ROW_H
        p.end()

    # ── Mouse ──────────────────────────────────────────────────────
    def _cell_at(self, pos):
        for r, e in self._cells:
            if r.contains(pos):
                return e
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        e = self._cell_at(event.position())
        if e is not None and event.button() == Qt.LeftButton:
            self.electrode_clicked.emit(e)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        e = self._cell_at(event.position())
        if e is None:
            self.setToolTip('')
        else:
            sc = self._scores.get(e, {})
            ch = self._layout.chamber_of(e) if self._layout else None
            if ch is not None and e in ch.stimulation:
                self.setToolTip(f'{e} — elettrodo di stimolazione (non usato per le misure)')
            elif sc.get('score') is not None and math.isfinite(sc['score']):
                self.setToolTip(f"{e} — punteggio {sc['score']:.1f}: {sc['n_beats']} spike, periodo {sc['period_ms']:.0f} ms, "
                                f"CV {sc['cv_pct']:.1f} %, SNR {sc['snr']:.1f}, ripolarizzazione {sc['repol_snr']:.1f}")
            else:
                self.setToolTip(f"{e} — {sc.get('reason') or 'nessun battito'}")
        super().mouseMoveEvent(event)
