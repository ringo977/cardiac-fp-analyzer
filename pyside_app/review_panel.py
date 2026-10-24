"""Revisione dock (v3.17): quality badge, review state, measures with the lab's Results labels, per-beat
table, actions.

The dock owns no analysis logic: it renders a ``ReviewState`` (``cardiac_fp_analyzer.review``) plus the
current result, and emits requests the main window fulfils (recompute, re-analysis, save, view change,
polarity, undo/redo, approval, closure). The per-beat table is the sheet ``CHIP X raw``; the measures
are the sheet ``CHIP X``. Measures and beats sit in a vertical splitter so the operator decides how much
of the dock each takes; "Copia misure" / "Copia battiti" put them on the clipboard as tab-separated
text with headers (numbers in the system locale, so Excel reads them as numbers).
"""
from __future__ import annotations

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cardiac_fp_analyzer.quality_summary import LEVEL_COLOURS, summarise_quality
from pyside_app.i18n import tr

from .review_helpers import BEAT_COLUMNS, format_beat_rows, format_measures, processing_text

N_MEASURE_ROWS = 13
MEASURE_HEADERS = ("Measure", "Value", "%", "Δ baseline")


def _locale_number(text: str) -> str:
    """'2.190' → '2,190' on an Italian system (Excel then reads a number); other strings untouched."""
    t = text.strip().replace('−', '-')
    if t.endswith(' %'):
        t = t[:-2]
    try:
        v = float(t.replace('+', ''))
    except ValueError:
        return text
    dec = len(t.split('.')[1]) if '.' in t else 0
    return QLocale().toString(v, 'f', dec)


class ReviewPanel(QWidget):
    """Right-hand dock of the review screen."""

    undo_requested = Signal()
    redo_requested = Signal()
    recompute_requested = Signal()
    reanalyse_requested = Signal()        # redo the automatic analysis, dropping the corrections in memory
    save_requested = Signal()
    view_changed = Signal(str)            # 'filtrato' | 'grezzo' | 'operatori'
    polarity_requested = Signal(int)      # +1 / -1
    beat_selected = Signal(int)           # row double-clicked → beat index
    restore_all_requested = Signal()
    next_problem_requested = Signal(int)  # +1 / -1
    approve_requested = Signal(bool)      # tick / untick 'Approvato'
    close_requested = Signal(bool)        # True = close (lock), False = reopen
    message = Signal(str)                 # one line for the status bar

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = None
        self._result = None
        self._reference = None
        self._quality = None
        self._syncing = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # ── judgement ──
        self._badge = QLabel(tr("No recordings"))
        self._badge.setWordWrap(True)
        self._badge.setStyleSheet("padding: 4px 6px; border-radius: 6px; background: #e6e6e6; color: #222;")
        lay.addWidget(self._badge)
        self._problems = QLabel("")
        self._problems.setWordWrap(True)
        self._problems.setStyleSheet("color: #666; padding: 0 4px;")
        lay.addWidget(self._problems)

        # ── review state ──
        rows = QHBoxLayout()
        self._chk_approved = QCheckBox(tr("Approved"))
        self._chk_approved.setToolTip(tr("Recording reviewed and approved by the operator (saved "
                                         "in the .review.json); a new correction withdraws the "
                                         "approval"))
        self._chk_approved.toggled.connect(self._on_approve_toggled)
        self._btn_close = QPushButton(tr("Close 🔒"))
        self._btn_close.setToolTip(tr("Closes the recording for good: approved and locked against "
                                      "changes (Unlock to reopen it)"))
        self._btn_close.clicked.connect(self._on_close_clicked)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #666;")
        self._status_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        rows.addWidget(self._chk_approved)
        rows.addWidget(self._btn_close)
        rows.addWidget(self._status_lbl, stretch=1)
        lay.addLayout(rows)

        # ── highlighted stretches: legend + navigation ──
        rowl = QHBoxLayout()
        self._legend = QLabel("")
        self._legend.setTextFormat(Qt.RichText)
        self._legend.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._legend.setStyleSheet("padding: 0 4px;")
        self._btn_prev_problem = QPushButton("◀")
        self._btn_next_problem = QPushButton(tr("Next stretch ▶"))
        self._btn_prev_problem.setToolTip(tr("Previous highlighted stretch (Ctrl+Shift+G)"))
        self._btn_next_problem.setToolTip(tr("Goes to the next highlighted stretch: missing beat, "
                                             "beats too close, repolarisation missing (Ctrl+G)"))
        self._btn_prev_problem.setFixedWidth(28)
        self._btn_prev_problem.clicked.connect(lambda: self.next_problem_requested.emit(-1))
        self._btn_next_problem.clicked.connect(lambda: self.next_problem_requested.emit(1))
        rowl.addWidget(self._legend, stretch=1)
        rowl.addWidget(self._btn_prev_problem)
        rowl.addWidget(self._btn_next_problem)
        lay.addLayout(rowl)

        # ── view ──
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("View:")))
        self._rb_filt = QRadioButton(tr("Filtered"))
        self._rb_raw = QRadioButton(tr("Raw"))
        self._rb_ops = QRadioButton(tr("1–40 Hz"))
        self._rb_ops.setToolTip(tr("The operators' view: 50 Hz notch + 1–40 Hz band-pass of order 1"))
        self._rb_filt.setChecked(True)
        grp = QButtonGroup(self)
        for rb, kind in ((self._rb_filt, "filtrato"), (self._rb_raw, "grezzo"), (self._rb_ops, "operatori")):
            grp.addButton(rb)
            row.addWidget(rb)
            rb.toggled.connect(lambda on, k=kind: on and self.view_changed.emit(k))
        row.addStretch()
        lay.addLayout(row)

        # ── actions (short rows: the dock may be narrow) ──
        row2 = QHBoxLayout()
        self._btn_undo = QPushButton(tr("Undo"))
        self._btn_redo = QPushButton(tr("Redo"))
        self._btn_undo.setToolTip(tr("Undo the last correction (Ctrl+Z)"))
        self._btn_redo.setToolTip(tr("Redo the undone correction (Ctrl+Shift+Z)"))
        self._btn_undo.clicked.connect(self.undo_requested.emit)
        self._btn_redo.clicked.connect(self.redo_requested.emit)
        self._btn_pos = QPushButton(tr("peak +"))
        self._btn_neg = QPushButton(tr("peak −"))
        self._btn_pos.setToolTip(tr("Repolarisation on the positive peak for all beats (key P)"))
        self._btn_neg.setToolTip(tr("Repolarisation on the negative peak for all beats (key N)"))
        self._btn_pos.clicked.connect(lambda: self.polarity_requested.emit(1))
        self._btn_neg.clicked.connect(lambda: self.polarity_requested.emit(-1))
        for b in (self._btn_undo, self._btn_redo, self._btn_pos, self._btn_neg):
            row2.addWidget(b)
        lay.addLayout(row2)

        row3 = QHBoxLayout()
        self._btn_restore = QPushButton(tr("Restore removed"))
        self._btn_restore.setToolTip(tr("Puts back all the detections removed by the vetoes (grey crosses)"))
        self._btn_restore.clicked.connect(self.restore_all_requested.emit)
        self._btn_recompute = QPushButton(tr("Recompute measures"))
        self._btn_recompute.setToolTip(tr("Runs the pipeline again (quality, repolarisation, "
                                          "parameters, arrhythmias) on the current beats: the "
                                          "corrections stay"))
        self._btn_recompute.clicked.connect(self.recompute_requested.emit)
        self._btn_reanalyse = QPushButton(tr("Redo analysis"))
        self._btn_reanalyse.setToolTip(tr("Runs the automatic detection again with the current "
                                          "settings and discards the corrections in memory (the "
                                          "saved ones stay in the .review.json until you save "
                                          "again)"))
        self._btn_reanalyse.clicked.connect(self.reanalyse_requested.emit)
        for b in (self._btn_restore, self._btn_recompute, self._btn_reanalyse):
            row3.addWidget(b)
        lay.addLayout(row3)

        row4 = QHBoxLayout()
        self._btn_save = QPushButton(tr("Save corrections"))
        self._btn_save.setToolTip(tr("Writes the .review.json file next to the recording (Ctrl+S)"))
        self._btn_save.clicked.connect(self.save_requested.emit)
        self._btn_copy_measures = QPushButton(tr("Copy measures"))
        self._btn_copy_measures.setToolTip(tr("Copy the measures to the clipboard (tab-separated "
                                              "text with headers, to paste into Excel)"))
        self._btn_copy_measures.clicked.connect(self.copy_measures)
        self._btn_copy_beats = QPushButton(tr("Copy beats"))
        self._btn_copy_beats.setToolTip(tr("Copy the beats table to the clipboard (tab-separated "
                                           "text with headers, to paste into Excel)"))
        self._btn_copy_beats.clicked.connect(self.copy_beats)
        for b in (self._btn_save, self._btn_copy_measures, self._btn_copy_beats):
            row4.addWidget(b)
        lay.addLayout(row4)

        # ── measures (top) and per-beat table (bottom) in a splitter ──
        self._split = QSplitter(Qt.Vertical)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(2)
        self._measures = QTableWidget(N_MEASURE_ROWS, 4)
        self._measures.setHorizontalHeaderLabels([tr(h) for h in MEASURE_HEADERS])
        self._measures.verticalHeader().setVisible(False)
        self._measures.verticalHeader().setDefaultSectionSize(19)
        self._measures.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._measures.setSelectionMode(QAbstractItemView.NoSelection)
        self._measures.setFocusPolicy(Qt.NoFocus)
        self._measures.setShowGrid(False)
        self._measures.setAlternatingRowColors(True)
        self._measures.setStyleSheet("QTableWidget { font-size: 11px; }")
        hh = self._measures.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for j in (1, 2, 3):
            hh.setSectionResizeMode(j, QHeaderView.ResizeToContents)
        hh.setStyleSheet("QHeaderView::section { font-size: 11px; }")
        self._measure_rows: list[tuple[QTableWidgetItem, ...]] = []
        for i in range(N_MEASURE_ROWS):
            cells = tuple(QTableWidgetItem("") for _ in range(4))
            for j, c in enumerate(cells):
                if j:
                    c.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._measures.setItem(i, j, c)
            self._measure_rows.append(cells)
        self._measures.setMinimumHeight(110)
        tl.addWidget(self._measures, stretch=1)
        self._processing = QLabel("")
        self._processing.setStyleSheet("color: #777; font-size: 11px; padding: 0 4px;")
        self._processing.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        tl.addWidget(self._processing)
        self._split.addWidget(top)

        self._table = QTableWidget(0, len(BEAT_COLUMNS))
        self._table.setHorizontalHeaderLabels([tr(h) for h in BEAT_COLUMNS])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(20)
        self._table.setMinimumWidth(0)
        self._table.setMinimumHeight(90)
        self._table.cellDoubleClicked.connect(lambda r, _c: self.beat_selected.emit(r))
        self._split.addWidget(self._table)
        self._split.setStretchFactor(0, 0)
        self._split.setStretchFactor(1, 1)
        self._split.setSizes([300, 260])
        lay.addWidget(self._split, stretch=1)

        self.setMinimumWidth(300)
        self.set_state(None, None)

    # ── rendering ──────────────────────────────────────────────────────

    def set_state(self, state, result, reference=None) -> None:
        """Show ``state`` (ReviewState or None) and the result it came from."""
        self._state, self._result, self._reference = state, result, reference
        self._quality = summarise_quality(result) if (state is not None and result is not None) else None
        enabled = state is not None
        for b in (self._btn_undo, self._btn_redo, self._btn_pos, self._btn_neg, self._btn_restore,
                  self._btn_recompute, self._btn_reanalyse, self._btn_save, self._btn_prev_problem,
                  self._btn_next_problem, self._chk_approved, self._btn_close, self._btn_copy_measures,
                  self._btn_copy_beats):
            b.setEnabled(enabled)
        self.refresh()

    def _elide(self, label: QLabel, text: str, width: int | None = None) -> None:
        label.setToolTip(text)
        w = (width or max(60, label.width() or self.width() - 24)) - 8
        label.setText(QFontMetrics(label.font()).elidedText(text, Qt.ElideRight, w))

    def refresh(self) -> None:
        st = self._state
        if st is None:
            self._badge.setText(tr("No recordings"))
            self._badge.setStyleSheet("padding: 4px 6px; border-radius: 6px; background: #e6e6e6; color: #222;")
            self._problems.setText("")
            self._problems.setToolTip("")
            self._legend.setText("")
            self._status_lbl.setText("")
            for cells in self._measure_rows:
                for c in cells:
                    c.setText("")
            self._table.setRowCount(0)
            self._processing.setText("")
            return
        if self._quality is not None:
            q = self._quality
            col = LEVEL_COLOURS[q["livello"]]
            self._badge.setText(tr("{0} — {1} beats").format(q["titolo"], st.n_beats()))
            self._badge.setStyleSheet(f"padding: 4px 6px; border-radius: 6px; background: {col}; color: white; font-weight: bold;")
            probs = ["• " + p["testo"] for p in q["problemi"]]
            shown = probs[:3]
            if len(probs) > 3:
                shown.append(tr("… and {0} more (hover here)").format(len(probs) - 3))
            self._problems.setText("\n".join(shown))
            self._problems.setToolTip("\n".join(probs))
        locked = bool(getattr(st, 'locked', False))
        self._btn_undo.setEnabled(st.can_undo and not locked)
        self._btn_redo.setEnabled(st.can_redo and not locked)
        for b in (self._btn_pos, self._btn_neg, self._btn_restore, self._btn_recompute, self._btn_reanalyse):
            b.setEnabled(not locked)
        self._syncing = True
        try:
            self._chk_approved.setChecked(bool(getattr(st, 'approved', False)))
            self._chk_approved.setEnabled(not locked)
            self._btn_close.setText(tr("Unlock 🔓") if locked else tr("Close 🔒"))
        finally:
            self._syncing = False
        self._status_lbl.setText(st.status_text() if hasattr(st, 'status_text') else "")
        self._status_lbl.setToolTip(self._status_lbl.text())
        self._legend.setText(self._legend_text(st))
        m = st.measures(reference=self._reference)
        for cells, (label, val, pct, var) in zip(self._measure_rows, format_measures(m, self._reference)):
            cells[0].setText(label)
            cells[0].setToolTip(label)
            cells[1].setText(val)
            cells[2].setText(pct)
            cells[3].setText(var)
        self._elide(self._processing, processing_text(self._result))
        rows = format_beat_rows(st.per_beat_rows())
        self._table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, txt in enumerate(r):
                item = QTableWidgetItem(txt)
                if j >= 6 and txt != "auto":
                    item.setForeground(QBrush(QColor("#1f4e9c" if txt in ("corretto", "propagato", "aggiunto", "ripristinato")
                                                     else "#b23a3a")))
                self._table.setItem(i, j, item)

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt API
        super().resizeEvent(event)
        if self._result is not None:
            self._elide(self._processing, processing_text(self._result))

    def select_row(self, k: int | None) -> None:
        if k is None or k >= self._table.rowCount():
            self._table.clearSelection()
            return
        self._table.selectRow(k)
        self._table.scrollToItem(self._table.item(k, 0))

    def set_view_buttons(self, kind: str) -> None:
        {"filtrato": self._rb_filt, "grezzo": self._rb_raw, "operatori": self._rb_ops}.get(kind, self._rb_filt).setChecked(True)

    # ── clipboard ──────────────────────────────────────────────────────

    def beats_tsv(self) -> str:
        """The per-beat table as tab-separated text with a header row (numbers in the system locale)."""
        if self._state is None:
            return ""
        lines = ["\t".join(tr(h) for h in BEAT_COLUMNS)]
        for r in format_beat_rows(self._state.per_beat_rows()):
            lines.append("\t".join(_locale_number(c) if j < 6 else c for j, c in enumerate(r)))
        return "\n".join(lines) + "\n"

    def measures_tsv(self) -> str:
        if self._state is None:
            return ""
        lines = ["\t".join(tr(h) for h in MEASURE_HEADERS)]
        for label, val, pct, var in format_measures(self._state.measures(reference=self._reference), self._reference):
            lines.append("\t".join((label, _locale_number(val), _locale_number(pct), _locale_number(var))))
        return "\n".join(lines) + "\n"

    def copy_beats(self) -> None:
        text = self.beats_tsv()
        if text:
            QApplication.clipboard().setText(text)
            self.message.emit(tr("{0} beats copied to the clipboard (paste into Excel)").format(text.count("\n") - 1))

    def copy_measures(self) -> None:
        text = self.measures_tsv()
        if text:
            QApplication.clipboard().setText(text)
            self.message.emit(tr("Measures copied to the clipboard (paste into Excel)"))

    # ── slots ──────────────────────────────────────────────────────────

    def _on_approve_toggled(self, on: bool) -> None:
        if not self._syncing and self._state is not None:
            self.approve_requested.emit(bool(on))

    def _on_close_clicked(self) -> None:
        if self._state is not None:
            self.close_requested.emit(not bool(getattr(self._state, 'locked', False)))

    @staticmethod
    def _legend_text(st) -> str:
        from .signal_viewer import PROBLEM_COLOURS, PROBLEM_LABELS
        counts = {}
        for p in st.problems():
            counts[p['tipo']] = counts.get(p['tipo'], 0) + 1
        parts = []
        if st.bursts:
            parts.append('<span style="color:#aaaaaa">■</span> burst')
        for tipo, label in PROBLEM_LABELS.items():
            if counts.get(tipo):
                r, g, b = PROBLEM_COLOURS[tipo]
                parts.append(f'<span style="color:rgb({r},{g},{b})">■</span> {tr(label)} ({counts[tipo]})')
        return ('<span style="color:#666">' + ' · '.join(parts) + '</span>') if parts else ''
