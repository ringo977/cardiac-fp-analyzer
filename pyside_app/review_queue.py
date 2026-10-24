"""Coda di revisione (v3.17): files and inputs analysed in the background, listed by quality judgement.

Add files or a folder; every recording a file holds (the two inputs of a WaveForms CSV, the chambers of a
chip) is analysed with the window's configuration in a background thread; the list shows a coloured
judgement (verde / giallo / rosso), the problems, beating period and FPD, sorted problems-first. A double
click opens the recording in the viewer for review; saving corrections marks the row as reviewed.
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
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cardiac_fp_analyzer.quality_summary import LEVEL_COLOURS

from .review_helpers import inputs_of, queue_entry, queue_sort_key
from .workers import analysis_pool

logger = logging.getLogger(__name__)
_EXT = ('.csv', '.h5', '.npz')


class _QueueSignals(QObject):
    done = Signal(int, object)        # (job index, entry dict)


class _QueueWorker(QRunnable):
    def __init__(self, index, path, channel, chamber, config, signals):
        super().__init__()
        self._i, self._path, self._channel, self._chamber, self._config, self._signals = index, path, channel, chamber, config, signals

    def run(self) -> None:   # noqa: D401
        from cardiac_fp_analyzer.analyze import analyze_single_file
        try:
            update = None
            if self._chamber:
                from cardiac_fp_analyzer.chambers import layout_for
                from cardiac_fp_analyzer.loader import electrode_columns, load_recording
                metadata, df = load_recording(self._path)
                lay = layout_for(electrode_columns(df), 'auto')
                ch = lay[self._chamber]
                update = {'chamber': ch.name, 'channel': ch.name, 'electrodes': list(ch.electrodes),
                          'stimulation_electrodes': list(ch.stimulation), 'chamber_layout': lay.name, 'two_tissue_file': True}
                result = analyze_single_file(self._path, channel=self._channel, verbose=False, config=self._config,
                                             file_info_update=update, preloaded=(metadata, df))
            else:
                result = analyze_single_file(self._path, channel=self._channel, verbose=False, config=self._config)
            entry = queue_entry(self._path, self._channel, self._chamber, result)
        except Exception as exc:   # noqa: BLE001 — the queue must survive one bad file
            logger.exception("review queue: %s", self._path)
            entry = queue_entry(self._path, self._channel, self._chamber, None, error=f"{type(exc).__name__}: {exc}")
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
        self._btn_files.clicked.connect(self._add_files)
        self._btn_folder.clicked.connect(self._add_folder)
        self._btn_clear.clicked.connect(self.clear)
        for b in (self._btn_files, self._btn_folder, self._btn_clear):
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        self._status = QLabel(self.tr("Aggiungi file o una cartella: ogni ingresso o camera viene analizzato e giudicato."))
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #777;")
        lay.addWidget(self._status)
        self._tree = QTreeWidget()
        self._tree.setColumnCount(6)
        self._tree.setHeaderLabels([self.tr("Registrazione"), self.tr("Ingresso"), self.tr("Giudizio"),
                                    self.tr("BP [s]"), self.tr("FPD [s]"), self.tr("Problemi")])
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        lay.addWidget(self._tree, stretch=1)

    # ── adding work ──────────────────────────────────────────────────

    def add_paths(self, paths) -> int:
        """Queue every recording of ``paths`` (files or folders); returns the number of jobs submitted."""
        files = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                files += sorted(f for f in p.rglob('*') if f.suffix.lower() in _EXT and '_report' not in f.parts
                                and not f.name.endswith('.review.json'))
            elif p.suffix.lower() in _EXT:
                files.append(p)
        n = 0
        for f in files:
            for channel, chamber in inputs_of(f):
                i = len(self._entries)
                self._entries[i] = {'file': str(f), 'nome': f.name, 'ingresso': channel, 'camera': chamber,
                                    'livello': None, 'titolo': self.tr("in analisi…"), 'riassunto': '', 'rivista': False,
                                    'bp_s': None, 'fpd_s': None}
                self._pool.start(_QueueWorker(i, str(f), channel, chamber, self._config_getter(), self._signals))
                self._pending += 1
                n += 1
        self._render()
        self._status.setText(self.tr("{0} registrazioni in analisi…").format(self._pending))
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
            entry['rivista'] = self._entries[i].get('rivista', False)
            self._entries[i] = entry
        self._pending = max(0, self._pending - 1)
        self._render()
        done = sum(1 for e in self._entries.values() if e.get('livello'))
        counts = {lv: sum(1 for e in self._entries.values() if e.get('livello') == lv) for lv in ('rosso', 'giallo', 'verde')}
        self._status.setText(self.tr("{0} analizzate ({1} in corso) — rosse {2}, gialle {3}, verdi {4}").format(
            done, self._pending, counts['rosso'], counts['giallo'], counts['verde']))

    def mark_reviewed(self, path: str, channel: str, chamber=None) -> None:
        for e in self._entries.values():
            if e['file'] == str(path) and str(e.get('ingresso')) == str(channel) and (chamber is None or e.get('camera') == chamber):
                e['rivista'] = True
        self._render()

    def entries(self) -> list[dict]:
        return sorted(self._entries.values(), key=queue_sort_key)

    def _render(self) -> None:
        self._tree.clear()
        for e in self.entries():
            where = e.get('ingresso', '')
            if e.get('camera'):
                where = self.tr("camera {0}").format(e['camera'])
            elif e.get('analizzato'):
                where = e['analizzato']
            title = e.get('titolo') or ''
            if e.get('rivista'):
                title = self.tr("rivista") + " · " + title
            item = QTreeWidgetItem([e['nome'], where, title,
                                    f"{e['bp_s']:.3f}" if e.get('bp_s') else "—",
                                    f"{e['fpd_s']:.3f}" if e.get('fpd_s') else "—",
                                    '; '.join(p['testo'] for p in e.get('problemi', [])[:3]) or e.get('riassunto', '')])
            lv = e.get('livello')
            if lv in LEVEL_COLOURS:
                item.setForeground(2, QBrush(QColor(LEVEL_COLOURS[lv])))
                item.setData(2, Qt.UserRole, lv)
            item.setData(0, Qt.UserRole, (e['file'], e.get('ingresso'), e.get('camera')))
            item.setToolTip(5, e.get('riassunto', ''))
            self._tree.addTopLevelItem(item)
        for c in range(6):
            self._tree.resizeColumnToContents(c)

    def _on_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.UserRole)
        if data:
            self.open_requested.emit(data[0], str(data[1]), data[2])
