"""Revisione dock (v3.17): quality badge, measures with the lab's Results labels, per-beat table, actions.

The dock owns no analysis logic: it renders a ``ReviewState`` (``cardiac_fp_analyzer.review``) plus the
current result, and emits requests the main window fulfils (recompute, save, view change, polarity,
undo/redo). The per-beat table is the sheet ``CHIP X raw``; the measures are the sheet ``CHIP X``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cardiac_fp_analyzer.quality_summary import LEVEL_COLOURS, summarise_quality

from .review_helpers import BEAT_COLUMNS, format_beat_rows, format_measures, processing_text


class ReviewPanel(QWidget):
    """Right-hand dock of the review screen."""

    undo_requested = Signal()
    redo_requested = Signal()
    recompute_requested = Signal()
    save_requested = Signal()
    view_changed = Signal(str)            # 'filtrato' | 'grezzo' | 'operatori'
    polarity_requested = Signal(int)      # +1 / -1
    beat_selected = Signal(int)           # row double-clicked → beat index
    restore_all_requested = Signal()
    next_problem_requested = Signal(int)  # +1 / -1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = None
        self._result = None
        self._reference = None
        self._quality = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        # ── quality badge ──
        self._badge = QLabel(self.tr("Nessuna registrazione"))
        self._badge.setWordWrap(True)
        self._badge.setStyleSheet("padding: 6px; border-radius: 6px; background: #e6e6e6; color: #222;")
        lay.addWidget(self._badge)
        self._problems = QLabel("")
        self._problems.setWordWrap(True)
        self._problems.setStyleSheet("color: #666; padding: 0 4px;")
        lay.addWidget(self._problems)
        # legend of the highlighted stretches + navigation
        rowl = QHBoxLayout()
        self._legend = QLabel("")
        self._legend.setWordWrap(True)
        self._legend.setTextFormat(Qt.RichText)
        self._legend.setStyleSheet("padding: 0 4px;")
        rowl.addWidget(self._legend, stretch=1)
        self._btn_prev_problem = QPushButton("◀")
        self._btn_next_problem = QPushButton(self.tr("Prossimo tratto ▶"))
        self._btn_prev_problem.setToolTip(self.tr("Tratto evidenziato precedente"))
        self._btn_next_problem.setToolTip(self.tr("Va al prossimo tratto evidenziato: battito mancante, battiti troppo "
                                                  "vicini, ripolarizzazione assente (Ctrl+G)"))
        self._btn_prev_problem.setFixedWidth(28)
        self._btn_prev_problem.clicked.connect(lambda: self.next_problem_requested.emit(-1))
        self._btn_next_problem.clicked.connect(lambda: self.next_problem_requested.emit(1))
        rowl.addWidget(self._btn_prev_problem)
        rowl.addWidget(self._btn_next_problem)
        lay.addLayout(rowl)

        # ── view + actions ──
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Vista:")))
        self._rb_filt = QRadioButton(self.tr("Filtrato"))
        self._rb_raw = QRadioButton(self.tr("Grezzo"))
        self._rb_ops = QRadioButton(self.tr("1–40 Hz (operatori)"))
        self._rb_filt.setChecked(True)
        grp = QButtonGroup(self)
        for rb, kind in ((self._rb_filt, "filtrato"), (self._rb_raw, "grezzo"), (self._rb_ops, "operatori")):
            grp.addButton(rb)
            row.addWidget(rb)
            rb.toggled.connect(lambda on, k=kind: on and self.view_changed.emit(k))
        row.addStretch()
        lay.addLayout(row)

        row2 = QHBoxLayout()
        self._btn_undo = QPushButton(self.tr("Annulla"))
        self._btn_redo = QPushButton(self.tr("Ripeti"))
        self._btn_undo.setToolTip(self.tr("Annulla l'ultima correzione (Ctrl+Z)"))
        self._btn_redo.setToolTip(self.tr("Ripeti la correzione annullata (Ctrl+Shift+Z)"))
        self._btn_undo.clicked.connect(self.undo_requested.emit)
        self._btn_redo.clicked.connect(self.redo_requested.emit)
        row2.addWidget(self._btn_undo)
        row2.addWidget(self._btn_redo)
        row2.addStretch()
        lay.addLayout(row2)

        # Short rows: a long single row of buttons would set the dock's minimum width and push the
        # main window beyond a laptop screen.
        row2b = QHBoxLayout()
        row2b.addWidget(QLabel(self.tr("Ripolarizzazione:")))
        self._btn_pos = QPushButton(self.tr("picco +"))
        self._btn_neg = QPushButton(self.tr("picco −"))
        self._btn_pos.setToolTip(self.tr("Usa il picco positivo per tutti i battiti (tasto P)"))
        self._btn_neg.setToolTip(self.tr("Usa il picco negativo per tutti i battiti (tasto N)"))
        self._btn_pos.clicked.connect(lambda: self.polarity_requested.emit(1))
        self._btn_neg.clicked.connect(lambda: self.polarity_requested.emit(-1))
        row2b.addWidget(self._btn_pos)
        row2b.addWidget(self._btn_neg)
        row2b.addStretch()
        lay.addLayout(row2b)

        row3 = QHBoxLayout()
        self._btn_restore = QPushButton(self.tr("Ripristina tolti"))
        self._btn_restore.setToolTip(self.tr("Rimette tutti i rilevamenti tolti dai veti (grigi)"))
        self._btn_restore.clicked.connect(self.restore_all_requested.emit)
        self._btn_recompute = QPushButton(self.tr("Ricalcola"))
        self._btn_recompute.setToolTip(self.tr("Rilancia la pipeline sui battiti correnti (le ripolarizzazioni corrette restano)"))
        self._btn_recompute.clicked.connect(self.recompute_requested.emit)
        self._btn_save = QPushButton(self.tr("Salva correzioni"))
        self._btn_save.setToolTip(self.tr("Scrive il file .review.json accanto alla registrazione (Ctrl+S)"))
        self._btn_save.clicked.connect(self.save_requested.emit)
        for b in (self._btn_restore, self._btn_recompute, self._btn_save):
            row3.addWidget(b)
        row3.addStretch()
        lay.addLayout(row3)

        # ── measures ──
        box = QFrame()
        box.setFrameShape(QFrame.StyledPanel)
        g = QGridLayout(box)
        g.setContentsMargins(6, 4, 6, 4)
        g.setColumnStretch(0, 1)
        for j, h in enumerate((self.tr("Misura"), self.tr("Valore"), "%", self.tr("Δ baseline"))):
            lbl = QLabel(h)
            lbl.setStyleSheet("font-weight: bold;")
            g.addWidget(lbl, 0, j)
        self._measure_rows: list[tuple[QLabel, QLabel, QLabel, QLabel]] = []
        for i in range(11):
            cells = tuple(QLabel("") for _ in range(4))
            for j, c in enumerate(cells):
                if j:
                    c.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    # The label column wraps and never dictates the dock's minimum width: long
                    # Results labels would otherwise push the main window beyond a laptop screen.
                    c.setWordWrap(True)
                    c.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
                g.addWidget(c, i + 1, j)
            self._measure_rows.append(cells)
        lay.addWidget(box)

        self._processing = QLabel("")
        self._processing.setWordWrap(True)
        self._processing.setStyleSheet("color: #777; font-size: 11px; padding: 0 4px;")
        lay.addWidget(self._processing)

        # ── per-beat table ──
        self._table = QTableWidget(0, len(BEAT_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(BEAT_COLUMNS))
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumWidth(0)
        self._table.cellDoubleClicked.connect(lambda r, _c: self.beat_selected.emit(r))
        lay.addWidget(self._table, stretch=1)

        self.setMinimumWidth(300)
        self.set_state(None, None)

    # ── rendering ──────────────────────────────────────────────────────

    def set_state(self, state, result, reference=None) -> None:
        """Show ``state`` (ReviewState or None) and the result it came from."""
        self._state, self._result, self._reference = state, result, reference
        self._quality = summarise_quality(result) if (state is not None and result is not None) else None
        enabled = state is not None
        for b in (self._btn_undo, self._btn_redo, self._btn_pos, self._btn_neg, self._btn_restore,
                  self._btn_recompute, self._btn_save, self._btn_prev_problem, self._btn_next_problem):
            b.setEnabled(enabled)
        self.refresh()

    def refresh(self) -> None:
        st = self._state
        if st is None:
            self._badge.setText(self.tr("Nessuna registrazione"))
            self._badge.setStyleSheet("padding: 6px; border-radius: 6px; background: #e6e6e6; color: #222;")
            self._problems.setText("")
            self._legend.setText("")
            for cells in self._measure_rows:
                for c in cells:
                    c.setText("")
            self._table.setRowCount(0)
            self._processing.setText("")
            return
        if self._quality is not None:
            q = self._quality
            col = LEVEL_COLOURS[q["livello"]]
            self._badge.setText(self.tr("{0} — {1} battiti").format(q["titolo"], st.n_beats()))
            self._badge.setStyleSheet(f"padding: 6px; border-radius: 6px; background: {col}; color: white; font-weight: bold;")
            self._problems.setText("\n".join("• " + p["testo"] for p in q["problemi"][:6]))
        self._btn_undo.setEnabled(st.can_undo)
        self._btn_redo.setEnabled(st.can_redo)
        self._legend.setText(self._legend_text(st))
        m = st.measures(reference=self._reference)
        for cells, (label, val, pct, var) in zip(self._measure_rows, format_measures(m, self._reference)):
            cells[0].setText(label)
            cells[1].setText(val)
            cells[2].setText(pct)
            cells[3].setText(var)
        self._processing.setText(processing_text(self._result))
        rows = format_beat_rows(st.per_beat_rows())
        self._table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, txt in enumerate(r):
                item = QTableWidgetItem(txt)
                if j >= 6 and txt != "auto":
                    item.setForeground(QBrush(QColor("#1f4e9c" if txt in ("corretto", "propagato", "aggiunto", "ripristinato")
                                                     else "#b23a3a")))
                self._table.setItem(i, j, item)

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
                parts.append(f'<span style="color:rgb({r},{g},{b})">■</span> {label} ({counts[tipo]})')
        return ('<span style="color:#666">Tratti evidenziati: ' + ' · '.join(parts) + '</span>') if parts else ''

    def select_row(self, k: int | None) -> None:
        if k is None or k >= self._table.rowCount():
            self._table.clearSelection()
            return
        self._table.selectRow(k)
        self._table.scrollToItem(self._table.item(k, 0))

    def set_view_buttons(self, kind: str) -> None:
        {"filtrato": self._rb_filt, "grezzo": self._rb_raw, "operatori": self._rb_ops}.get(kind, self._rb_filt).setChecked(True)
