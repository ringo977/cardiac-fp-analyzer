"""Small stand-alone window to convert WaveForms CSV recordings to NWB 2 (``cardiac-fp-nwb-gui``).

Pick one or more CSV files or a folder (searched recursively; with an output folder the sub-folders are kept; CSV files that are not WaveForms recordings are skipped), fill in what is known — every field except the
amplifier gain and the time zone may be left empty — and convert.  The work is done by
``cardiac_fp_analyzer.nwb_io.convert_many`` (the same code as the ``cardiac-fp-nwb`` command) on a worker
thread; a failing file is reported and the others go on.  What was typed is remembered for the next run.
The identity of each recording (experiment, chip, chamber, test item, condition) is read from the
``samples.csv`` found next to the files (or up to three folders above) or from the file names, exactly as in
the analysis.  The language follows the one chosen in the main application (View ▸ Language).

This module is also the entry point of the packaged Windows / macOS application (``packaging/``).
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyside_app import i18n
from pyside_app.i18n import tr

ORG, APP = "CardiacFP", "NWB Converter"
MAIN_APP = "Cardiac FP Analyzer"          # the main application's settings hold the interface language
WORKER_STACK_BYTES = 16 * 1024 * 1024     # macOS gives secondary threads 512 KB; BLAS needs more
DEFAULT_GAIN, DEFAULT_TZ = 10000.0, "Europe/Rome"
DEFAULT_SPECIES, DEFAULT_SUBJECT = "Homo sapiens", "hiPSC-derived cardiac microtissue"


def split_names(text: str) -> list[str]:
    """``'Rossi, M.; Bianchi'`` → ``['Rossi, M.', 'Bianchi']``: names are separated by ``;`` or new lines
    (a comma belongs to the name, as in 'Surname, N.')."""
    out = []
    for chunk in str(text or "").replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if chunk:
            out.append(chunk)
    return out


def options_from_fields(*, gain, tz, lab="", institution="", experimenter="", species="", subject="",
                        notes="", overwrite=False) -> dict:
    """Keyword options for ``csv_to_nwb`` from the window's fields: empty ones fall back to the defaults or are
    left out.  Raises ``ValueError`` for a time zone the system does not know."""
    tz = (tz or "").strip() or DEFAULT_TZ
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(tz)
    except Exception as exc:    # noqa: BLE001 — unknown key, missing tz database, bad format
        raise ValueError(tr("Unknown time zone: {0}").format(tz)) from exc
    return {"gain": float(gain) if gain else DEFAULT_GAIN, "tz": tz, "lab": (lab or "").strip(),
            "institution": (institution or "").strip(), "experimenter": split_names(experimenter),
            "species": (species or "").strip() or DEFAULT_SPECIES,
            "subject_description": (subject or "").strip() or DEFAULT_SUBJECT,
            "notes": (notes or "").strip(), "overwrite": bool(overwrite)}


class _Worker(QThread):
    progress = Signal(int, int, str, str, str)     # i, n, csv, written ('' on error), message
    finished_all = Signal(list)

    def __init__(self, paths, out_dir, check, options, parent=None):
        super().__init__(parent)
        self.setStackSize(WORKER_STACK_BYTES)
        self._paths, self._out_dir, self._check, self._options = paths, out_dir, check, options

    def run(self):
        from cardiac_fp_analyzer.nwb_io import convert_many
        try:
            results = convert_many(
                self._paths, out_dir=self._out_dir or None, check=self._check,
                progress=lambda i, n, csv, written, msg: self.progress.emit(i, n, csv, written or "", msg),
                **self._options)
        except Exception as exc:    # noqa: BLE001 — e.g. unreadable folder
            results = [{"csv": "", "nwb": None, "ok": False, "message": f"{type(exc).__name__}: {exc}", "issues": None}]
        self.finished_all.emit(results)


class NwbConverterWindow(QMainWindow):
    """The converter window (see the module docstring)."""

    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self._settings = settings if settings is not None else QSettings(ORG, APP)
        self._paths: list[str] = []
        self._worker: _Worker | None = None
        self.setWindowTitle(tr("NWB converter"))
        root = QWidget(self)
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)

        # ── what to convert / where to write ──
        files = QGroupBox(tr("Recordings"))
        fl = QVBoxLayout(files)
        row = QHBoxLayout()
        self._src = QLineEdit()
        self._src.setReadOnly(True)
        self._src.setPlaceholderText(tr("CSV files or a folder (searched recursively)"))
        b_files = QPushButton(tr("Choose files…"))
        b_folder = QPushButton(tr("Choose folder…"))
        b_files.clicked.connect(self._choose_files)
        b_folder.clicked.connect(self._choose_folder)
        row.addWidget(self._src, 1)
        row.addWidget(b_files)
        row.addWidget(b_folder)
        fl.addLayout(row)
        row = QHBoxLayout()
        self._out = QLineEdit()
        self._out.setPlaceholderText(tr("Output folder (empty: next to each CSV)"))
        b_out = QPushButton(tr("Choose…"))
        b_out.clicked.connect(self._choose_out)
        row.addWidget(self._out, 1)
        row.addWidget(b_out)
        fl.addLayout(row)
        lay.addWidget(files)

        # ── acquisition ──
        acq = QGroupBox(tr("Acquisition"))
        af = QFormLayout(acq)
        self._gain = QDoubleSpinBox()
        self._gain.setRange(1.0, 1e7)
        self._gain.setDecimals(1)
        self._gain.setSingleStep(100.0)
        self._gain.setValue(DEFAULT_GAIN)
        self._gain.setToolTip(tr("Hardware gain between electrode and oscilloscope (raw / gain = volts at the "
                                 "electrode). 10000 for the µECG-Pharma Digilent; 1 if the CSV is already in volts."))
        self._tz = QLineEdit(DEFAULT_TZ)
        self._tz.setToolTip(tr("Time zone of the computer clock that wrote the CSV (e.g. Europe/Rome); the file's "
                               "start time is stored with it."))
        af.addRow(tr("Amplifier gain"), self._gain)
        af.addRow(tr("Time zone"), self._tz)
        lay.addWidget(acq)

        # ── metadata, all optional ──
        meta = QGroupBox(tr("Metadata (optional — leave empty what you do not need)"))
        mf = QFormLayout(meta)
        self._lab = QLineEdit()
        self._institution = QLineEdit()
        self._experimenter = QLineEdit()
        self._experimenter.setPlaceholderText(tr("Surname, N.; Surname, N."))
        self._experimenter.setToolTip(tr("One or more names separated by ; (semicolon)"))
        self._species = QLineEdit(DEFAULT_SPECIES)
        self._subject = QLineEdit(DEFAULT_SUBJECT)
        self._notes = QPlainTextEdit()
        self._notes.setFixedHeight(54)
        mf.addRow(tr("Laboratory"), self._lab)
        mf.addRow(tr("Institution"), self._institution)
        mf.addRow(tr("Experimenter(s)"), self._experimenter)
        mf.addRow(tr("Species"), self._species)
        mf.addRow(tr("Subject description"), self._subject)
        mf.addRow(tr("Notes"), self._notes)
        lay.addWidget(meta)

        # ── options + run ──
        self._overwrite = QCheckBox(tr("Overwrite existing .nwb files"))
        self._check = QCheckBox(tr("Check each file with nwbinspector (if installed)"))
        lay.addWidget(self._overwrite)
        lay.addWidget(self._check)
        row = QHBoxLayout()
        self._convert = QPushButton(tr("Convert"))
        self._convert.setDefault(True)
        self._convert.clicked.connect(self.start)
        self._bar = QProgressBar()
        self._bar.setTextVisible(True)
        row.addWidget(self._convert)
        row.addWidget(self._bar, 1)
        lay.addLayout(row)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(110)
        lay.addWidget(self._log, 1)
        self._status = QLabel("")
        lay.addWidget(self._status)
        self.resize(720, 760)
        self._restore()

    # ── settings ──
    _FIELDS = ("lab", "institution", "experimenter", "species", "subject", "tz", "out")

    def _restore(self):
        s = self._settings
        for name in self._FIELDS:
            v = s.value(f"fields/{name}")
            if v not in (None, ""):
                getattr(self, "_" + name).setText(str(v))
        g = s.value("fields/gain")
        if g not in (None, ""):
            with contextlib.suppress(TypeError, ValueError):
                self._gain.setValue(float(g))
        self._notes.setPlainText(str(s.value("fields/notes", "") or ""))

    def _save(self):
        s = self._settings
        for name in self._FIELDS:
            s.setValue(f"fields/{name}", getattr(self, "_" + name).text())
        s.setValue("fields/gain", self._gain.value())
        s.setValue("fields/notes", self._notes.toPlainText())

    # ── choosing ──
    def set_paths(self, paths):
        self._paths = [str(p) for p in paths]
        self._src.setText("; ".join(self._paths))

    def _choose_files(self):
        start = self._settings.value("last_dir", "") or ""
        files, _ = QFileDialog.getOpenFileNames(self, tr("Choose CSV files"), str(start), tr("CSV files (*.csv)"))
        if files:
            self._settings.setValue("last_dir", str(Path(files[0]).parent))
            self.set_paths(files)

    def _choose_folder(self):
        start = self._settings.value("last_dir", "") or ""
        folder = QFileDialog.getExistingDirectory(self, tr("Choose the folder with the recordings"), str(start))
        if folder:
            self._settings.setValue("last_dir", folder)
            self.set_paths([folder])

    def _choose_out(self):
        folder = QFileDialog.getExistingDirectory(self, tr("Choose the output folder"), self._out.text() or "")
        if folder:
            self._out.setText(folder)

    # ── running ──
    def options(self) -> dict:
        return options_from_fields(
            gain=self._gain.value(), tz=self._tz.text(), lab=self._lab.text(), institution=self._institution.text(),
            experimenter=self._experimenter.text(), species=self._species.text(), subject=self._subject.text(),
            notes=self._notes.toPlainText(), overwrite=self._overwrite.isChecked())

    def start(self):
        if self._worker is not None:
            return
        if not self._paths:
            QMessageBox.information(self, tr("NWB converter"), tr("Choose the CSV files or the folder first."))
            return
        try:
            options = self.options()
        except ValueError as exc:
            QMessageBox.warning(self, tr("NWB converter"), str(exc))
            return
        self._save()
        self._log.clear()
        self._bar.setValue(0)
        self._status.setText(tr("Converting…"))
        self._convert.setEnabled(False)
        self._worker = _Worker(self._paths, self._out.text().strip(), self._check.isChecked(), options, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_all.connect(self._on_done)
        self._worker.start()

    def _on_progress(self, i, n, csv, written, message):
        self._bar.setMaximum(max(n, 1))
        self._bar.setValue(i)
        name = Path(csv).name
        if written:
            self._log.appendPlainText(f"✓ {name} → {written}")
        elif message == "not a WaveForms recording":
            self._log.appendPlainText(f"– {name}: " + tr("skipped (not a WaveForms recording)"))
        else:
            self._log.appendPlainText(f"✗ {name}: {message}")

    def _on_done(self, results):
        n_ok = sum(1 for r in results if r["ok"])
        n_skip = sum(1 for r in results if r.get("skipped"))
        results = [r for r in results if not r.get("skipped")]
        for r in results:
            if r["ok"] and r.get("issues"):
                self._log.appendPlainText(f"  {Path(r['nwb']).name}: " + "; ".join(r["issues"]))
        if not results:
            self._log.appendPlainText(tr("No CSV file found."))
        elif not results[0]["csv"] and not results[0]["ok"]:
            self._log.appendPlainText(f"✗ {results[0]['message']}")
            results = []
        self._status.setText(tr("{0} of {1} files converted").format(n_ok, len(results))
                             + (tr(" — {0} skipped (not WaveForms recordings)").format(n_skip) if n_skip else ""))
        self._convert.setEnabled(True)
        w, self._worker = self._worker, None
        if w is not None:
            w.wait()
            w.deleteLater()

    def closeEvent(self, event):    # noqa: N802 (Qt)
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(5000)
        self._save()
        super().closeEvent(event)


def self_test() -> int:
    """Convert a tiny synthetic recording and read it back — proves that a packaged build carries everything
    (pynwb's schema files, h5py, the time-zone database).  ``NWB converter --self-test``; no window."""
    import tempfile

    import numpy as np

    from cardiac_fp_analyzer.nwb_io import csv_to_nwb, read_nwb
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "selftest.csv"
        n, fs = 2000, 1000.0
        t = -1.0 + np.arange(n) / fs
        with open(csv, "w") as f:
            f.write("#Digilent WaveForms Oscilloscope Acquisition\n#Device Name: Discovery2\n"
                    "#Serial Number: SN:0\n#Date Time: 2026-04-15 15:36:06.081\n#Sample rate: 1000Hz\n"
                    f"#Samples: {n}\n#Channel 1: Range: 500 mV/div Offset: 0 V Sample Mode: Average\n\n"
                    "Time (s),Channel 1 (V)\n")
            for ti in t:
                f.write(f"{ti:.6f},{np.sin(40 * ti) * 1e-4:.9e}\n")
        out = csv_to_nwb(csv, Path(d) / "selftest.nwb", gain=10000.0, tz=DEFAULT_TZ)
        meta, df = read_nwb(out)
        assert len(df) == n and "el1" in df.columns, "round trip failed"
    print("self-test OK")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    QApplication.setOrganizationName(ORG)
    QApplication.setApplicationName(APP)
    app = QApplication(argv)
    i18n.apply_language(app, i18n.configured_language(QSettings(ORG, MAIN_APP)))
    win = NwbConverterWindow()
    win.show()
    win.raise_()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
