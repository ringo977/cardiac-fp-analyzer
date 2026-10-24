"""Coda di revisione (v3.17): recordings analysed in the background, listed by quality judgement.

Add files or a folder; the recordings they hold are planned like the batch does
(``cardiac_fp_analyzer.results_export.plan_recordings``: one per tissue input of a two-tissue CSV, one per
chamber of a multi-chamber chip, identities from the ``samples.csv`` of the experiment when there is one)
and analysed with the window's configuration in a background thread. Every row carries its *record*
(identity, measures, per-beat rows, judgement, corrections — see ``results_export.record_from_state``),
updated when the operator saves corrections; "Esporta Results…" writes the lab's Results workbook from
the records. A double click opens the recording in the viewer.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cardiac_fp_analyzer.quality_summary import LEVEL_COLOURS

from .review_helpers import queue_entry, queue_sort_key
from .workers import analysis_pool

logger = logging.getLogger(__name__)
_EXT = ('.csv', '.h5', '.npz')


class _QueueSignals(QObject):
    done = Signal(int, object)        # (job index, entry dict)


class _QueueWorker(QRunnable):
    def __init__(self, index, unit, config, signals):
        super().__init__()
        self._i, self._unit, self._config, self._signals = index, unit, config, signals

    def run(self) -> None:   # noqa: D401
        from cardiac_fp_analyzer.analyze import analyze_single_file
        from cardiac_fp_analyzer.results_export import record_from_state
        from cardiac_fp_analyzer.review import ReviewState
        u = self._unit
        try:
            result = analyze_single_file(u['file'], channel=u['electrode'], verbose=False, config=self._config,
                                         file_info_update=u.get('update'))
            state = ReviewState.from_result(result, self._config)
            ReviewState.load_into(state, u['file'])
            record = record_from_state(state, result, file=u['file'], electrode=u['electrode'], identity=u.get('update'))
            entry = queue_entry(u['file'], u['electrode'], (u.get('update') or {}).get('chamber'), result, record=record)
        except Exception as exc:   # noqa: BLE001 — the queue must survive one bad file
            logger.exception("review queue: %s", u['file'])
            entry = queue_entry(u['file'], u['electrode'], (u.get('update') or {}).get('chamber'), None,
                                error=f"{type(exc).__name__}: {exc}")
        self._signals.done.emit(self._i, entry)


class ReviewQueue(QWidget):
    """Left-hand dock: the recordings to review."""

    open_requested = Signal(str, str, object)      # (file, channel, chamber or None)

    def __init__(self, config_getter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_getter = config_getter
        self._entries: dict[int, dict] = {}
        self._pending = 0
        self._signals = _QueueSignals()
        self._signals.done.connect(self._on_done)
        # 16 MB stacks: OpenBLAS needs more than macOS's 512 KB default (see workers.py).
        self._pool = analysis_pool(self, max_threads=min(3, QThreadPool.globalInstance().maxThreadCount() - 1))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        row = QHBoxLayout()
        self._btn_files = QPushButton(self.tr("Aggiungi file…"))
        self._btn_folder = QPushButton(self.tr("Aggiungi cartella…"))
        self._btn_clear = QPushButton(self.tr("Svuota"))
        self._btn_export = QPushButton(self.tr("Esporta Results…"))
        self._btn_export.setToolTip(self.tr("Scrive il foglio Excel nel formato Results del laboratorio "
                                            "(CHIP X, CHIP X raw) con le registrazioni della coda"))
        self._btn_files.clicked.connect(self._add_files)
        self._btn_folder.clicked.connect(self._add_folder)
        self._btn_clear.clicked.connect(self.clear)
        self._btn_export.clicked.connect(self._export)
        for b in (self._btn_files, self._btn_folder, self._btn_clear):
            row.addWidget(b)
        row.addStretch()
        row.addWidget(self._btn_export)
        lay.addLayout(row)
        self._status = QLabel(self.tr("Aggiungi file o una cartella: ogni ingresso o camera viene analizzato e giudicato."))
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #777;")
        lay.addWidget(self._status)
        self._tree = QTreeWidget()
        self._tree.setColumnCount(8)
        self._tree.setHeaderLabels([self.tr("Registrazione"), self.tr("Tessuto"), self.tr("Condizione"),
                                    self.tr("Ingresso"), self.tr("Giudizio"), self.tr("BP [s]"), self.tr("FPD [s]"),
                                    self.tr("Problemi")])
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setTextElideMode(Qt.ElideMiddle)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        lay.addWidget(self._tree, stretch=1)
        self._btn_export.setEnabled(False)

    # ── adding work ──────────────────────────────────────────────────

    def add_paths(self, paths) -> int:
        """Queue every recording of ``paths`` (files or folders); returns the number of jobs submitted."""
        from cardiac_fp_analyzer.results_export import plan_recordings, sheet_root
        files = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                files += sorted(f for f in p.rglob('*') if f.suffix.lower() in _EXT and '_report' not in f.parts
                                and not f.name.endswith('.review.json'))
            elif p.suffix.lower() in _EXT:
                files.append(p)
        if not files:
            return 0
        config = self._config_getter()
        root = sheet_root(Path(paths[0]) if Path(paths[0]).is_dir() else Path(paths[0]).parent)
        units, plan = plan_recordings(files, root, config)
        self._last_plan = plan
        n = 0
        for u in units:
            upd = u.get('update') or {}
            i = len(self._entries)
            self._entries[i] = {'file': str(u['file']), 'nome': Path(u['file']).name, 'ingresso': u['electrode'],
                                'camera': upd.get('chamber'), 'chip': upd.get('chip'), 'item': upd.get('item'),
                                'livello': None, 'titolo': self.tr("in analisi…"), 'riassunto': '', 'rivista': False,
                                'bp_s': None, 'fpd_s': None, 'record': None, 'condizione': ''}
            self._pool.start(_QueueWorker(i, u, config, self._signals))
            self._pending += 1
            n += 1
        self._render()
        msg = self.tr("{0} registrazioni in analisi…").format(self._pending)
        if plan.get('sheets'):
            msg += " " + self.tr("(identità da {0})").format(', '.join(plan['sheets']))
        self._status.setText(msg)
        return n

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, self.tr("Registrazioni da rivedere"), "",
                                                self.tr("Registrazioni (*.csv *.h5 *.npz);;Tutti i file (*)"))
        if paths:
            self.add_paths(paths)

    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, self.tr("Cartella con le registrazioni"))
        if d:
            self.add_paths([d])

    def clear(self) -> None:
        self._entries.clear()
        self._pending = 0
        self._render()
        self._status.setText(self.tr("Coda vuota."))

    # ── results ──────────────────────────────────────────────────────

    def _on_done(self, i: int, entry: dict) -> None:
        if i in self._entries:
            old = self._entries[i]
            entry['rivista'] = old.get('rivista', False)
            for k in ('chip', 'item', 'camera'):
                entry.setdefault(k, old.get(k))
            self._entries[i] = entry
        self._pending = max(0, self._pending - 1)
        self._render()
        done = sum(1 for e in self._entries.values() if e.get('livello'))
        counts = {lv: sum(1 for e in self._entries.values() if e.get('livello') == lv) for lv in ('rosso', 'giallo', 'verde')}
        self._status.setText(self.tr("{0} analizzate ({1} in corso) — rosse {2}, gialle {3}, verdi {4}").format(
            done, self._pending, counts['rosso'], counts['giallo'], counts['verde']))

    def _find(self, path: str, channel: str, chamber=None):
        for e in self._entries.values():
            if e['file'] == str(path) and str(e.get('ingresso')) == str(channel) and (chamber is None or e.get('camera') == chamber):
                yield e

    def mark_reviewed(self, path: str, channel: str, chamber=None, state=None, result=None) -> None:
        """Flag the row as reviewed; with ``state`` and ``result`` its record is rebuilt from the corrections."""
        from cardiac_fp_analyzer.results_export import record_from_state
        for e in self._find(path, channel, chamber):
            e['rivista'] = True
            if state is not None and result is not None:
                rec = e.get('record') or {}
                identity = {k: rec.get(k) for k in ('experiment', 'day', 'chip', 'chamber', 'item', 'is_baseline')
                            if rec.get(k) is not None}
                if rec.get('condition'):
                    identity['concentration'] = None if rec['condition'] == 'baseline' else rec['condition']
                new = record_from_state(state, result, file=path, electrode=channel, identity=identity)
                if rec.get('condition'):
                    new['condition'] = rec['condition']
                e['record'] = new
                m = new['measures']
                e['bp_s'], e['fpd_s'] = m.get('bp_s'), m.get('fpd_s')
        self._render()

    def entries(self) -> list[dict]:
        return sorted(self._entries.values(), key=queue_sort_key)

    def records(self) -> list[dict]:
        return [e['record'] for e in self.entries() if e.get('record')]

    def _render(self) -> None:
        self._tree.clear()
        for e in self.entries():
            rec = e.get('record') or {}
            chip, ch, item = rec.get('chip', e.get('chip')), rec.get('chamber', e.get('camera')), rec.get('item', e.get('item'))
            tissue = ' '.join(s for s in (f"chip {chip}" if chip else '', f"ch{ch}" if ch not in (None, '') else '',
                                          f"({item})" if item else '') if s)
            title = e.get('titolo') or ''
            if e.get('rivista'):
                title = self.tr("rivista") + " · " + title
            item_w = QTreeWidgetItem([e['nome'], tissue, rec.get('condition') or e.get('condizione') or '',
                                      str(e.get('analizzato') or e.get('ingresso') or ''), title,
                                      f"{e['bp_s']:.3f}" if e.get('bp_s') else "—",
                                      f"{e['fpd_s']:.3f}" if e.get('fpd_s') else "—",
                                      '; '.join(p['testo'] for p in e.get('problemi', [])[:3]) or e.get('riassunto', '')])
            lv = e.get('livello')
            if lv in LEVEL_COLOURS:
                item_w.setForeground(4, QBrush(QColor(LEVEL_COLOURS[lv])))
                item_w.setData(4, Qt.UserRole, lv)
            item_w.setData(0, Qt.UserRole, (e['file'], e.get('ingresso'), e.get('camera')))
            item_w.setToolTip(0, e['file'])
            item_w.setToolTip(7, e.get('riassunto', ''))
            self._tree.addTopLevelItem(item_w)
        for c in range(8):
            self._tree.resizeColumnToContents(c)
        self._tree.setColumnWidth(0, min(self._tree.columnWidth(0), 260))
        self._btn_export.setEnabled(any(e.get('record') for e in self._entries.values()))

    def _on_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.UserRole)
        if data:
            self.open_requested.emit(data[0], str(data[1]), data[2])

    # ── export ───────────────────────────────────────────────────────

    def export_results(self, path) -> int:
        """Write the Results workbook of the queue's records to ``path``; returns the number of records."""
        from cardiac_fp_analyzer.results_export import write_results_workbook
        recs = self.records()
        if not recs:
            return 0
        write_results_workbook(recs, path)
        return len(recs)

    def _export(self) -> None:
        recs = self.records()
        if not recs:
            QMessageBox.information(self, self.tr("Esporta Results"), self.tr("Nessuna registrazione analizzata."))
            return
        first = Path(recs[0]['file'])
        exp = str(recs[0].get('experiment') or '').replace(' ', '') or 'Results'
        suggested = str(first.parent / f"{exp}_FP_analysis.xlsx")
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Foglio Results"), suggested, self.tr("Excel (*.xlsx)"))
        if not path:
            return
        try:
            n = self.export_results(path)
        except Exception as exc:    # noqa: BLE001 — report, don't crash the GUI
            logger.exception("results export")
            QMessageBox.warning(self, self.tr("Esporta Results"), str(exc))
            return
        pending = sum(1 for e in self._entries.values() if not e.get('livello'))
        msg = self.tr("{0} registrazioni scritte in {1}").format(n, Path(path).name)
        if pending:
            msg += " " + self.tr("({0} ancora in analisi, non incluse)").format(pending)
        self._status.setText(msg)
