"""Coda di revisione (v3.17): recordings analysed in the background, listed by quality judgement.

Add files or a folder; the recordings they hold are planned like the batch does
(``cardiac_fp_analyzer.results_export.plan_recordings``: one per tissue input of a two-tissue CSV, one per
chamber of a multi-chamber chip, identities from the ``samples.csv`` of the experiment when there is one)
and analysed with the window's configuration in a background thread.

One row per **tissue**. When a file holds the two electrodes of the same construct (a single-tissue CSV
with two inputs) both are analysed and listed as sub-rows EL1 / EL2, each with its judgement and
measures, and a tick marks the electrode used for the measures (default: the software's choice by
signal quality, or the one saved earlier); the choice is written to the recording's ``.review.json``
(``elettrodo_scelto``) and honoured by the analysis in ``auto`` mode, the Results export and the batch.
Every row carries its *record* (identity, measures, per-beat rows, judgement, corrections — see
``results_export.record_from_state``), refreshed when the operator saves corrections; "Esporta
Results…" writes the lab's Results workbook from the chosen records. A double click opens the
recording (the ticked electrode for a tissue row, that electrode for a sub-row).
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
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
_EXT = ('.csv', '.h5', '.npz', '.nwb')
COL_FILE, COL_TISSUE, COL_COND, COL_INPUT, COL_USE, COL_STATE, COL_JUDGE, COL_BP, COL_FPD, COL_PROBLEMS = range(10)
_STATE_MARK = {'chiuso': '🔒', 'approvato': '✓', 'da_rivedere': ''}


class _QueueSignals(QObject):
    done = Signal(int, object)        # (job index, {'electrodes': {label: entry}, 'chosen': label})


def _analyse(unit, config, channel):
    """One analysis of a unit's file on ``channel``; returns (channel analysed, queue entry with its record)."""
    from cardiac_fp_analyzer.analyze import analyze_single_file
    from cardiac_fp_analyzer.results_export import record_from_state
    from cardiac_fp_analyzer.review import ReviewState
    result = analyze_single_file(unit['file'], channel=channel, verbose=False, config=config,
                                 file_info_update=unit.get('update'))
    state = ReviewState.from_result(result, config)
    ReviewState.load_into(state, unit['file'])
    actual = (result.get('file_info') or {}).get('analyzed_channel') or channel
    record = record_from_state(state, result, file=unit['file'], electrode=actual, identity=unit.get('update'))
    return actual, queue_entry(unit['file'], actual, (unit.get('update') or {}).get('chamber'), result, record=record)


class _QueueWorker(QRunnable):
    """Analyses one tissue: its input (or chamber) — or both inputs of a single-tissue file with two."""

    def __init__(self, index, unit, config, signals, both_inputs):
        super().__init__()
        self._i, self._unit, self._config, self._signals, self._both = index, unit, config, signals, both_inputs

    def run(self) -> None:   # noqa: D401
        import dataclasses

        from cardiac_fp_analyzer.review import ReviewState
        u = self._unit
        electrodes, chosen = {}, None
        try:
            if self._both:
                # the software's own choice first ('auto' without the sidecar's choice), then the other input
                saved = ReviewState.chosen_electrode(u['file'])
                cfg_auto = dataclasses.replace(self._config, use_overrides=False) if saved else self._config
                first, entry = _analyse(u, cfg_auto, 'auto')
                electrodes[first] = entry
                other = 'el2' if first == 'el1' else 'el1'
                try:
                    lab, entry2 = _analyse(u, self._config, other)
                    electrodes[lab] = entry2
                except Exception as exc:    # noqa: BLE001 — one bad input must not lose the other
                    logger.exception("review queue: %s (%s)", u['file'], other)
                    electrodes[other] = queue_entry(u['file'], other, None, None, error=f"{type(exc).__name__}: {exc}")
                chosen = saved if saved in electrodes else first
                for lab, e in electrodes.items():
                    e['scelta_software'] = (lab == first)
            else:
                lab, entry = _analyse(u, self._config, u['electrode'])
                electrodes[lab] = entry
                chosen = lab
        except Exception as exc:   # noqa: BLE001 — the queue must survive one bad file
            logger.exception("review queue: %s", u['file'])
            lab = u['electrode'] if u['electrode'] != 'auto' else 'el1'
            electrodes = {lab: queue_entry(u['file'], lab, (u.get('update') or {}).get('chamber'), None,
                                           error=f"{type(exc).__name__}: {exc}")}
            chosen = lab
        self._signals.done.emit(self._i, {'electrodes': electrodes, 'chosen': chosen})


class ReviewQueue(QWidget):
    """Left-hand dock: the recordings to review."""

    open_requested = Signal(str, str, object)      # (file, channel, chamber or None)

    def __init__(self, config_getter, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_getter = config_getter
        self._entries: dict[int, dict] = {}        # tissue rows
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
        self._status = QLabel(self.tr("Aggiungi file o una cartella: ogni tessuto viene analizzato e giudicato; "
                                      "con due elettrodi dello stesso tessuto scegli quale usare con la spunta."))
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #777;")
        lay.addWidget(self._status)
        self._tree = QTreeWidget()
        self._tree.setColumnCount(10)
        self._tree.setHeaderLabels([self.tr("Registrazione"), self.tr("Tessuto"), self.tr("Condizione"),
                                    self.tr("Ingresso"), self.tr("Usa"), self.tr("Stato"), self.tr("Giudizio"),
                                    self.tr("BP [s]"), self.tr("FPD [s]"), self.tr("Problemi")])
        self._tree.setRootIsDecorated(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setTextElideMode(Qt.ElideMiddle)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._rendering = False
        lay.addWidget(self._tree, stretch=1)
        self._btn_export.setEnabled(False)

    # ── adding work ──────────────────────────────────────────────────

    def add_paths(self, paths) -> int:
        """Queue every recording of ``paths`` (files or folders); returns the number of tissues submitted."""
        from cardiac_fp_analyzer.loader import input_columns
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
            both = u['electrode'] == 'auto' and not upd.get('electrodes') and len(input_columns(u['file'])) == 2
            i = len(self._entries)
            self._entries[i] = {'file': str(u['file']), 'nome': Path(u['file']).name, 'ingresso': u['electrode'],
                                'camera': upd.get('chamber'), 'chip': upd.get('chip'), 'item': upd.get('item'),
                                'livello': None, 'titolo': self.tr("in analisi…"), 'riassunto': '', 'rivista': False,
                                'bp_s': None, 'fpd_s': None, 'record': None, 'condizione': '',
                                'elettrodi': {}, 'scelto': None, 'due_elettrodi': both}
            self._pool.start(_QueueWorker(i, u, config, self._signals, both))
            self._pending += 1
            n += 1
        self._render()
        msg = self.tr("{0} tessuti in analisi…").format(self._pending)
        if plan.get('sheets'):
            msg += " " + self.tr("(identità da {0})").format(', '.join(plan['sheets']))
        self._status.setText(msg)
        return n

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, self.tr("Registrazioni da rivedere"), "",
                                                self.tr("Registrazioni (*.csv *.h5 *.npz *.nwb);;Tutti i file (*)"))
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

    @staticmethod
    def _apply_choice(e: dict) -> None:
        """Copy the chosen electrode's judgement and measures onto the tissue row."""
        sub = e['elettrodi'].get(e.get('scelto'))
        if not sub:
            return
        for k in ('livello', 'titolo', 'riassunto', 'problemi', 'bp_s', 'fpd_s', 'n_battiti', 'record', 'errore',
                  'analizzato', 'condizione', 'chip', 'item'):
            if k in sub:
                e[k] = sub[k]
        if sub.get('camera') is not None:
            e['camera'] = sub['camera']
        e['ingresso'] = e['scelto']

    def _on_done(self, i: int, payload: dict) -> None:
        if i in self._entries:
            e = self._entries[i]
            e['elettrodi'] = payload['electrodes']
            e['scelto'] = payload['chosen']
            self._apply_choice(e)
        self._pending = max(0, self._pending - 1)
        self._render()
        done = sum(1 for e in self._entries.values() if e.get('livello'))
        counts = {lv: sum(1 for e in self._entries.values() if e.get('livello') == lv) for lv in ('rosso', 'giallo', 'verde')}
        self._status.setText(self.tr("{0} tessuti analizzati ({1} in corso) — rossi {2}, gialli {3}, verdi {4}").format(
            done, self._pending, counts['rosso'], counts['giallo'], counts['verde']))

    def _find(self, path: str, channel: str, chamber=None):
        """Tissue rows holding ``channel`` of ``path`` (and the chamber when given)."""
        for e in self._entries.values():
            if e['file'] != str(path):
                continue
            if chamber is not None and e.get('camera') != chamber:
                continue
            if str(channel) in e['elettrodi'] or str(e.get('ingresso')) == str(channel):
                yield e

    def mark_reviewed(self, path: str, channel: str, chamber=None, state=None, result=None) -> None:
        """Flag the row as reviewed; with ``state`` and ``result`` the record of that electrode is rebuilt."""
        from cardiac_fp_analyzer.results_export import record_from_state
        for e in self._find(path, channel, chamber):
            e['rivista'] = True
            sub = e['elettrodi'].get(str(channel))
            if state is not None and result is not None and sub is not None:
                rec = sub.get('record') or {}
                identity = {k: rec.get(k) for k in ('experiment', 'day', 'chip', 'chamber', 'item', 'is_baseline')
                            if rec.get(k) is not None}
                if rec.get('condition'):
                    identity['concentration'] = None if rec['condition'] == 'baseline' else rec['condition']
                new = record_from_state(state, result, file=path, electrode=channel, identity=identity)
                if rec.get('condition'):
                    new['condition'] = rec['condition']
                sub['record'] = new
                m = new['measures']
                sub['bp_s'], sub['fpd_s'] = m.get('bp_s'), m.get('fpd_s')
            if e.get('scelto') == str(channel):
                self._apply_choice(e)
        self._render()

    def choose_electrode(self, path: str, electrode: str, chamber=None, render: bool = True) -> bool:
        """Use ``electrode`` for the measures of the tissue recorded in ``path``; saved in the sidecar."""
        from cardiac_fp_analyzer.review import ReviewState
        changed = False
        for e in self._find(path, electrode, chamber):
            if electrode in e['elettrodi'] and e.get('scelto') != electrode:
                e['scelto'] = electrode
                self._apply_choice(e)
                changed = True
                if e.get('due_elettrodi'):
                    try:
                        ReviewState.save_choice(path, electrode)
                    except OSError as exc:
                        logger.warning("could not save the electrode choice for %s: %s", path, exc)
        if changed and render:
            self._render()
        return changed

    def entries(self) -> list[dict]:
        return sorted(self._entries.values(), key=queue_sort_key)

    def records(self) -> list[dict]:
        """The export records of the chosen electrode of every tissue."""
        return [e['record'] for e in self.entries() if e.get('record')]

    def _item(self, e: dict, label: str | None, sub: dict | None) -> QTreeWidgetItem:
        src = sub if sub is not None else e
        rec = src.get('record') or {}
        chip, ch, item = rec.get('chip', e.get('chip')), rec.get('chamber', e.get('camera')), rec.get('item', e.get('item'))
        tissue = ' '.join(s for s in (f"chip {chip}" if chip else '', f"ch{ch}" if ch not in (None, '') else '',
                                      f"({item})" if item else '') if s)
        title = src.get('titolo') or ''
        state = rec.get('stato') or 'da_rivedere'
        state_txt = _STATE_MARK.get(state, '')
        if sub is None and e.get('rivista') and state == 'da_rivedere':
            state_txt = self.tr("rivista")
        if sub is None:
            inp = str(e.get('scelto') or e.get('analizzato') or e.get('ingresso') or '').upper()
            if e.get('due_elettrodi') and e.get('scelto'):
                inp += ' ' + self.tr("(usato)")
        else:
            inp = str(label).upper() + (' · ' + self.tr("scelta del software") if sub.get('scelta_software') else '')
        it = QTreeWidgetItem([e['nome'] if sub is None else '', tissue if sub is None else '',
                              (rec.get('condition') or src.get('condizione') or '') if sub is None else '', inp, '',
                              state_txt, title, f"{src['bp_s']:.3f}" if src.get('bp_s') else "—",
                              f"{src['fpd_s']:.3f}" if src.get('fpd_s') else "—",
                              '; '.join(p['testo'] for p in src.get('problemi', [])[:3]) or src.get('riassunto', '')])
        lv = src.get('livello')
        if lv in LEVEL_COLOURS:
            it.setForeground(COL_JUDGE, QBrush(QColor(LEVEL_COLOURS[lv])))
            it.setData(COL_JUDGE, Qt.UserRole, lv)
        it.setData(COL_FILE, Qt.UserRole, (e['file'], label or e.get('scelto') or e.get('ingresso'), e.get('camera')))
        it.setToolTip(COL_FILE, e['file'])
        it.setToolTip(COL_PROBLEMS, src.get('riassunto', ''))
        it.setToolTip(COL_STATE, rec.get('stato_testo') or self.tr("da rivedere"))
        if state == 'chiuso':
            it.setForeground(COL_STATE, QBrush(QColor('#888888')))
        if sub is not None:
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(COL_USE, Qt.Checked if e.get('scelto') == label else Qt.Unchecked)
            it.setToolTip(COL_USE, self.tr("Usa questo elettrodo per le misure del tessuto (salvato nel .review.json)"))
        return it

    def _render(self) -> None:
        self._rendering = True
        try:
            self._tree.clear()
            for e in self.entries():
                top = self._item(e, None, None)
                self._tree.addTopLevelItem(top)
                if e.get('due_elettrodi') and e.get('elettrodi'):
                    for label in sorted(e['elettrodi']):
                        top.addChild(self._item(e, label, e['elettrodi'][label]))
                    top.setExpanded(True)
            for c in range(10):
                self._tree.resizeColumnToContents(c)
            self._tree.setColumnWidth(COL_FILE, min(self._tree.columnWidth(COL_FILE), 260))
        finally:
            self._rendering = False
        self._btn_export.setEnabled(any(e.get('record') for e in self._entries.values()))

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._rendering or column != COL_USE or item.parent() is None:
            return
        data = item.data(COL_FILE, Qt.UserRole)
        if not data:
            return
        if item.checkState(COL_USE) == Qt.Checked:
            self.choose_electrode(data[0], str(data[1]), data[2], render=False)
        # The tree is rebuilt after this handler returns: clearing it from inside the signal of one of
        # its items would destroy the emitter while Qt is still using it. (An unticked box comes back
        # ticked: a choice can only be moved to the other electrode.)
        QTimer.singleShot(0, self._render)

    def _on_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(COL_FILE, Qt.UserRole)
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
