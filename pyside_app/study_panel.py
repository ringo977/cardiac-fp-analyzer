"""Study panel — dockable tree for Studi/Gruppi/File (task #84).

PoC UI on top of :mod:`cardiac_fp_analyzer.study`.  The panel is a
dockable ``QWidget`` that shows the current study as a ``QTreeWidget``:

    Study "Exp6-DofRR"
    ├─ baseline   [Control]
    │  └─ Exp6_ChipD_ch2_Ti10_C.csv  (EL1)
    └─ dof-10nM   [Dofetilide 0.010 µM]
       ├─ Exp6_ChipD_ch2_Ti10_dose1.csv  (EL1)
       └─ Exp6_ChipD_ch2_Ti10_dose2.csv  (EL2)

Terminology note
----------------
"Studio" (not "Progetto") is the regulatory term: GLP / OECD / 21 CFR
Part 58 refer to a protocolled nonclinical investigation as a
*nonclinical laboratory study*, and CDISC SDTM uses ``STUDYID`` as the
top-level identifier.  "Progetto" has no such anchoring in the
regulations that a dose-response dataset may eventually be submitted
under, so we align the on-disk format (``.cfp-study.json``) and the
UI vocabulary with the audit language now, while the PoC is still
cheap to rename.

Features (step 2 + follow-up #86 — per *feedback_poc_ergonomics*,
ergonomics can be rough):

* Toolbar: *Nuovo studio* / *Apri studio* / *Chiudi studio* /
  *Aggiungi gruppo* / *Aggiungi CSV al gruppo* / *Impostazioni gruppo*
  / *Rimuovi (gruppo/file)* / *Elimina sidecar studio* (destructive).
* Double-click on a file row → emits :attr:`file_activated` with the
  absolute CSV path, so :class:`MainWindow` can reuse ``_run_analysis``.
* Every mutation autosaves the ``.cfp-study.json`` sidecar via
  :func:`cardiac_fp_analyzer.study.save_study` — no explicit Save
  button in the PoC (Marco prefers fewer clicks).
* No batch-analyse button here yet: that's step 3 (#84 step 3); this
  panel only manages the *structure*.

Out of scope on purpose:

* Custom :class:`QAbstractItemModel` / drag-drop between groups.  A
  plain :class:`QTreeWidget` is enough for dose-response workflows
  where the tree rarely exceeds ~20 rows, and it keeps the diff small.
* Per-file config overrides (only per-group ``AnalysisConfig`` today).
* Aggregate reports / dose-response plots (step #84b).
* Deleting the study *folder* (incl. CSV data) — that stays in Finder.
  The panel can only delete the **sidecar** ``.cfp-study.json``, not
  the experimental data; see :meth:`StudyPanel._on_delete_study`.

Integration contract with :class:`MainWindow`:

* The panel owns no signal/result state — it just hands back a CSV
  path on double-click.
* The study sidecar is SSOT; anything the user edits in the tree
  is reflected back onto disk synchronously.

Audit & data safety (GxP-adjacent context — cardiac-electrophysiology
research, corrections are part of the scientific record):

* **Destructive actions are explicit.**  Only *two* UI actions remove
  bytes from disk — "Rimuovi" on a group/file (drops rows from the
  sidecar JSON, never touches CSVs) and "Elimina sidecar studio"
  (deletes exactly one file: ``<folder>/.cfp-study.json``).  Neither
  can touch raw experimental data; neither can touch per-file override
  sidecars (``*.overrides.json``) — those remain next to the CSVs and
  are keyed by CSV path, not by study membership.
* **Logging.**  ``logger.warning`` is emitted with the full absolute
  path whenever we delete or rewrite the sidecar, so a terminal /
  log-file audit shows *what was deleted and when* even if the UI is
  gone.
* **Confirmation dialogs.**  Every destructive action uses a Yes/No
  dialog with default = No, and the body text spells out *which
  artefacts are affected and which are not* (CSV, overrides, cartella).
* **No silent fallbacks.**  If :func:`save_study` fails mid-edit we
  surface it; we never swallow an ``OSError`` and pretend the edit
  was persisted.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import (
    QEventLoop,
    QObject,
    QRunnable,
    Qt,
    QThread,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QAction, QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cardiac_fp_analyzer.study import (
    STUDY_FILENAME,
    Group,
    Study,
    add_file_to_group,
    add_group,
    find_group,
    load_study,
    remove_group,
    resolve_file_path,
    save_study,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
#  Role constants for QTreeWidgetItem.data() — which layer does the row
#  represent, and what's its key in the Study model.
# ─────────────────────────────────────────────────────────────────────────
_KIND_ROLE = Qt.UserRole + 1      # "study" | "group" | "file"
_GROUP_NAME_ROLE = Qt.UserRole + 2   # str (for group + file rows)
_FILE_INDEX_ROLE = Qt.UserRole + 3   # int (position within group.files)


# ─────────────────────────────────────────────────────────────────────────
#  Tuning knob for the batch-analysis thread pool (step 3b).
#
#  4 is a pragmatic ceiling: most laptops have 4–8 performance cores,
#  and past 4 the gain flattens because ``analyze_single_file`` is
#  disk + numpy bound and workers start fighting over the same CSV
#  cache / memory bandwidth.  If you need to parallelise harder (big
#  Mac / batch server), raise it here — the UI logic doesn't care.
# ─────────────────────────────────────────────────────────────────────────
_BATCH_MAX_THREADS = 4


# ═══════════════════════════════════════════════════════════════════════
#  Batch-analysis result cache (step 3 of task #84)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FileResult:
    """Summary of a batch-analyze pass for one file.

    Step 3 deliberately caches **only the summary** (channel + beat
    counts + error string), not the full :class:`AnalysisResult`:

    * keeping the full result for every file in a 20-file study would
      chew through memory (each result holds the full filtered signal,
      time vector, and beats arrays);
    * the panel only needs enough to render badges and tooltips;
    * a double-click on a file row re-runs :func:`analyze_single_file`
      and populates the editing tabs, so the user still has a path to
      inspect the raw result — batch gives the overview, double-click
      gives the deep dive.

    Attributes
    ----------
    status : str
        ``'ok'`` — pipeline succeeded; ``'error'`` — raised, timed out,
        or returned ``None`` (reason stored in ``error``).
    channel_analyzed : str
        What ``select_best_channel`` (or the user-pinned override)
        actually ran on — e.g. ``'EL1'``.  Empty for errors.
    n_included : int
        Beats that fed the parameter extraction
        (``len(result['beat_indices_fpd'])``) — the "useful" count
        the user cares about for QC.
    n_total : int
        Beats surviving the detection + clean-up pass
        (``len(result['beat_indices'])``).  ``>= n_included``.
    error : str
        Short user-facing message.  Empty when ``status == 'ok'``.
    """

    status: str                      # 'ok' | 'error'
    channel_analyzed: str = ''
    n_included: int = 0
    n_total: int = 0
    error: str = ''


# ═══════════════════════════════════════════════════════════════════════
#  Background batch-analysis workers (step 3b of task #84)
# ═══════════════════════════════════════════════════════════════════════
#
#  Step 3 ran ``analyze_single_file`` in the UI thread with a modal
#  QProgressDialog + QApplication.processEvents().  That froze the UI
#  for the duration of each individual file — tolerable for a 3-file
#  group, painful for 20 files × multi-second pipeline.
#
#  Step 3b moves the work to a :class:`QThreadPool`.  Design notes:
#
#  * We use ``QRunnable`` (fire-and-forget) rather than ``QThread`` per
#    file because we want a bounded-parallelism pool, not N threads.
#    The cap is ``min(idealThreadCount(), 4)`` — 4 is a pragmatic
#    ceiling that keeps the Mac's fan quiet and avoids hammering the
#    same disk with many concurrent CSV reads.
#  * ``QRunnable`` isn't a ``QObject`` so it can't emit signals; we
#    pipe results through a separate :class:`_BatchWorkerSignals`
#    object, connected via ``Qt.QueuedConnection`` so the slot runs
#    on the UI thread.  That way the slot can safely touch
#    ``self._results``, ``QProgressDialog``, and the tree.
#  * Cancellation only removes *pending* runnables via
#    ``pool.clear()``.  Runnables already executing finish their
#    ``analyze_single_file`` call (which is CPU-bound numpy/pandas
#    with no cooperative cancellation point) — the caller then
#    ``waitForDone()``s and drops their results on the floor.
#  * ``analyze_single_file`` is thread-safe in practice: it re-reads
#    the CSV, builds a fresh :class:`AnalysisResult`, and touches no
#    module-level mutable state.  Logging is thread-safe by design.
#  * Writes to ``self._results`` happen only from the UI slot, so the
#    GIL-safety of dict writes isn't even load-bearing — the worker
#    just emits the result and lets the UI thread store it.


class _BatchWorkerSignals(QObject):
    """Signal carrier for :class:`_BatchWorker` — one result per file.

    ``QRunnable`` is not a ``QObject``, so it cannot own signals.
    The pattern is to hand each worker a shared ``_BatchWorkerSignals``
    instance and have the worker emit through it; the UI-thread
    consumer connects once to the shared object.

    Using a single shared instance (instead of one per worker) keeps
    the connection / disconnection story simple and avoids leaking
    per-file ``QObject``\\s if a cancel drops pending runnables.
    """

    # (file_index, FileResult) — file_index is the position within
    # ``group.files`` at submit time, used to key the cache.
    done = Signal(int, object)


class _BatchWorker(QRunnable):
    """One-shot worker: run :func:`analyze_single_file` on one CSV.

    All per-file state is captured in ``__init__``; ``run`` is the
    only method that touches the pipeline.  Both success and failure
    paths emit through ``signals.done`` — the UI slot decides how to
    render the :class:`FileResult` and never needs to distinguish
    "worker crashed" from "pipeline returned error" (both surface as
    ``status='error'``).
    """

    def __init__(
        self,
        *,
        file_index: int,
        abspath: str,
        channel: str,
        config,              # AnalysisConfig — kept untyped to avoid a
                             # hard import of the scientific core at
                             # module load time.
        signals: _BatchWorkerSignals,
    ) -> None:
        super().__init__()
        self._file_index = file_index
        self._abspath = abspath
        self._channel = channel
        self._config = config
        self._signals = signals
        # ``setAutoDelete(True)`` is the QRunnable default, which we
        # want: the worker is disposable and only lives for one call.

    def run(self) -> None:   # noqa: D401 — Qt virtual; not a "returns"
        # Import here (not at module top) so the panel still loads
        # even if the scientific core has an import error — same
        # rationale as the original _on_analyze_group.
        from cardiac_fp_analyzer.analyze import analyze_single_file

        try:
            result = analyze_single_file(
                self._abspath,
                channel=self._channel,
                verbose=False,
                config=self._config,
            )
        except Exception as exc:   # noqa: BLE001 — batch must survive
            logger.exception(
                "StudyPanel: analyze_single_file failed for %s",
                self._abspath,
            )
            fr = FileResult(
                status='error',
                error=f"{type(exc).__name__}: {exc}",
            )
            self._signals.done.emit(self._file_index, fr)
            return

        if result is None:
            fr = FileResult(status='error', error="Pipeline ritornata vuota")
            self._signals.done.emit(self._file_index, fr)
            return

        analyzed = (
            (result.get('file_info') or {}).get('analyzed_channel') or ''
        )
        n_total = _len_or_zero(result.get('beat_indices'))
        # ``beat_indices_fpd`` = beats that actually produced FPD (the
        # user-meaningful "inclusi" count).  Fall back to
        # ``beat_indices`` if the schema is older than task #73.
        fpd_len = _len_or_zero(result.get('beat_indices_fpd'))
        n_included = fpd_len if fpd_len > 0 else n_total

        fr = FileResult(
            status='ok',
            channel_analyzed=str(analyzed),
            n_included=n_included,
            n_total=n_total,
        )
        self._signals.done.emit(self._file_index, fr)


# ═══════════════════════════════════════════════════════════════════════
#  Group-edit dialog (New group / Group settings)
# ═══════════════════════════════════════════════════════════════════════

class _GroupDialog(QDialog):
    """Modal form to create or edit the metadata of a :class:`Group`.

    Only the *descriptive* metadata lives here — drug, dose, condition,
    analysis date, notes.  The per-group :class:`AnalysisConfig` has its
    own dedicated dialog (see ``pyside_app.settings_dialog``) and is
    wired separately through "Impostazioni gruppo" on the toolbar: mixing
    them in a single form would bury the ~20 numeric analysis knobs
    under the 5 descriptive fields and make both worse.
    """

    def __init__(
        self,
        *,
        name: str = "",
        drug: str = "",
        dose_uM: float | None = None,
        condition: str = "",
        analysis_date: str = "",
        notes: str = "",
        title: str = "",
        lock_name: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or self.tr("Gruppo"))

        form = QFormLayout()
        self._name = QLineEdit(name)
        self._name.setReadOnly(lock_name)
        self._drug = QLineEdit(drug)
        self._condition = QLineEdit(condition)
        self._date = QLineEdit(analysis_date)
        self._date.setPlaceholderText("YYYY-MM-DD")
        self._notes = QLineEdit(notes)

        # Dose is optional — a checkbox gates the spinbox so a baseline
        # group can persist "dose_uM = null" rather than "0.0 µM", which
        # would be ambiguous with a true zero-concentration control.
        self._has_dose = QCheckBox(self.tr("Dose (µM)"))
        self._has_dose.setChecked(dose_uM is not None)
        self._dose = QDoubleSpinBox()
        self._dose.setDecimals(4)
        self._dose.setRange(0.0, 1e6)
        self._dose.setSingleStep(0.001)
        self._dose.setValue(float(dose_uM) if dose_uM is not None else 0.0)
        self._dose.setEnabled(dose_uM is not None)
        self._has_dose.toggled.connect(self._dose.setEnabled)

        dose_row = QHBoxLayout()
        dose_row.addWidget(self._has_dose)
        dose_row.addWidget(self._dose, 1)
        dose_host = QWidget()
        dose_host.setLayout(dose_row)

        form.addRow(self.tr("Nome"), self._name)
        form.addRow(self.tr("Farmaco"), self._drug)
        form.addRow("", dose_host)
        form.addRow(self.tr("Condizione"), self._condition)
        form.addRow(self.tr("Data analisi"), self._date)
        form.addRow(self.tr("Note"), self._notes)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(buttons)

    def values(self) -> dict:
        """Return the edited fields as a dict (names match :class:`Group`)."""
        return {
            "name": self._name.text().strip(),
            "drug": self._drug.text().strip(),
            "dose_uM": (
                float(self._dose.value()) if self._has_dose.isChecked() else None
            ),
            "condition": self._condition.text().strip(),
            "analysis_date": self._date.text().strip(),
            "notes": self._notes.text().strip(),
        }


# ═══════════════════════════════════════════════════════════════════════
#  Study panel
# ═══════════════════════════════════════════════════════════════════════

class StudyPanel(QWidget):
    """Dockable panel that manages a single :class:`Study`.

    Signals
    -------
    file_activated(str)
        Emitted when the user double-clicks a file row.  Payload is the
        **absolute** CSV path, ready to hand to ``MainWindow._run_analysis``.

    study_changed()
        Emitted after every successful mutation (add / remove / edit /
        load).  Primarily so the window title / status-bar can reflect
        the current study name; the model on disk has already been
        saved by the time this fires.
    """

    file_activated = Signal(str)
    study_changed = Signal()

    # ── Construction ──────────────────────────────────────────────────

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._study: Study | None = None

        # Batch-analysis cache — keyed by (group_name, csv_relpath) so
        # it stays stable under file reordering within a group.  Pruned
        # on structural edits by :meth:`_prune_cache`.
        self._results: dict[tuple[str, str], FileResult] = {}

        # ── Toolbar ───────────────────────────────────────────────────
        # Layout in four groups, separated visually:
        #   1. Study lifecycle     : New / Open / Close
        #   2. Study editing       : Add group / Add CSV / Group settings /
        #                            Remove (group or file)
        #   3. Analysis            : Analizza gruppo (batch)
        #   4. Destructive study   : Delete sidecar (kept at the far end so
        #                            accidental click-streaks don't land on
        #                            it — see audit notes in module docstring).
        self._toolbar = QToolBar(self.tr("Studio"))
        self._toolbar.setIconSize(self._toolbar.iconSize())  # keep defaults

        self._act_new = QAction(self.tr("Nuovo studio..."), self)
        self._act_open = QAction(self.tr("Apri studio..."), self)
        self._act_close = QAction(self.tr("Chiudi studio"), self)
        self._act_add_group = QAction(self.tr("Aggiungi gruppo..."), self)
        self._act_add_file = QAction(self.tr("Aggiungi CSV al gruppo..."), self)
        self._act_group_settings = QAction(
            self.tr("Impostazioni gruppo..."), self,
        )
        self._act_remove = QAction(self.tr("Rimuovi"), self)
        self._act_analyze_group = QAction(self.tr("Analizza gruppo..."), self)
        self._act_delete_study = QAction(
            self.tr("Elimina sidecar studio..."), self,
        )

        # Group 1 — lifecycle.
        self._toolbar.addAction(self._act_new)
        self._toolbar.addAction(self._act_open)
        self._toolbar.addAction(self._act_close)
        self._toolbar.addSeparator()
        # Group 2 — editing.
        self._toolbar.addAction(self._act_add_group)
        self._toolbar.addAction(self._act_add_file)
        self._toolbar.addAction(self._act_group_settings)
        self._toolbar.addAction(self._act_remove)
        self._toolbar.addSeparator()
        # Group 3 — analysis (batch).
        self._toolbar.addAction(self._act_analyze_group)
        self._toolbar.addSeparator()
        # Group 4 — destructive. Distanced from the editing group on purpose.
        self._toolbar.addAction(self._act_delete_study)

        self._act_new.triggered.connect(self._on_new_study)
        self._act_open.triggered.connect(self._on_open_study)
        self._act_close.triggered.connect(self._on_close_study)
        self._act_add_group.triggered.connect(self._on_add_group)
        self._act_add_file.triggered.connect(self._on_add_file)
        self._act_group_settings.triggered.connect(self._on_group_settings)
        self._act_remove.triggered.connect(self._on_remove)
        self._act_analyze_group.triggered.connect(self._on_analyze_group)
        self._act_delete_study.triggered.connect(self._on_delete_study)

        # ── Tree ──────────────────────────────────────────────────────
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(
            [self.tr("Elemento"), self.tr("Dettagli")]
        )
        self._tree.setColumnCount(2)
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.currentItemChanged.connect(self._refresh_actions)

        # ── Empty-state label, shown instead of the tree until a study
        #   is opened or created.  Kept as a sibling widget rather than
        #   stuffed into the tree header because PyQtGraph-style "empty
        #   plot" placeholders are much clearer than a 1-row dummy tree.
        self._hint = QLabel(self.tr(
            "Nessuno studio aperto.  Usa “Nuovo studio…” o "
            "“Apri studio…” per iniziare."
        ))
        self._hint.setWordWrap(True)
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet("color: #888; padding: 18px;")

        # ── Layout ────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._toolbar)
        root.addWidget(self._hint)
        root.addWidget(self._tree, 1)
        self._tree.hide()   # start in empty state

        self._refresh_actions()

    # ── Public API ────────────────────────────────────────────────────

    def current_study(self) -> Study | None:
        """Return the currently-loaded :class:`Study`, or ``None``."""
        return self._study

    def set_study(self, study: Study | None) -> None:
        """Replace the loaded study.  Does NOT save — caller's problem.

        This is the single choke point for swapping studies in the
        panel, so keep it side-effect-light: refresh the tree, enable
        actions, fire ``study_changed``.  Persistence is handled on
        the *caller* side via :func:`save_study`.

        Any previous batch-analysis results are dropped on purpose:
        they were keyed by (group_name, csv_relpath) of the *old*
        study and would mis-attribute to unrelated files if we kept
        them around.
        """
        self._study = study
        self._results.clear()
        self._rebuild_tree()
        self._refresh_actions()
        self.study_changed.emit()

    # ═════════════════════════════════════════════════════════════════
    #  Toolbar handlers
    # ═════════════════════════════════════════════════════════════════

    def _on_new_study(self) -> None:
        """Create a new study sidecar in a user-chosen folder.

        Refuses to overwrite an existing ``.cfp-study.json`` so we
        never clobber someone else's study by accident — the user can
        pick the same folder again explicitly via "Apri studio".
        """
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Cartella del nuovo studio"),
        )
        if not folder:
            return
        folder_path = Path(folder)

        # Existing sidecar → bail out and suggest "Apri studio".  We
        # do not auto-adopt because differing schema versions or stale
        # sidecars would silently override the user's intent.
        if (folder_path / STUDY_FILENAME).is_file():
            QMessageBox.warning(
                self, self.tr("Studio già presente"),
                self.tr(
                    "Questa cartella contiene già uno studio "
                    "({0}).  Usa “Apri studio…” per caricarlo."
                ).format(STUDY_FILENAME),
            )
            return

        # Default name = folder basename so users don't have to type it
        # twice; they can still rename later by re-saving with a
        # different study.name (not exposed in the PoC UI).
        default_name = folder_path.name or "new-study"
        name, ok = QInputDialog.getText(
            self, self.tr("Nome studio"),
            self.tr("Nome:"),
            text=default_name,
        )
        if not ok:
            return
        name = name.strip() or default_name

        study = Study(name=name, folder=str(folder_path))
        try:
            save_study(study)
        except OSError as e:
            QMessageBox.critical(
                self, self.tr("Errore salvataggio"),
                self.tr("Impossibile creare lo studio: {0}").format(e),
            )
            return

        self.set_study(study)

    def _on_open_study(self) -> None:
        """Load an existing study sidecar from a user-chosen folder."""
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Cartella dello studio"),
        )
        if not folder:
            return
        study = load_study(folder)
        if study is None:
            QMessageBox.warning(
                self, self.tr("Studio non trovato"),
                self.tr(
                    "Nessun file {0} valido in questa cartella."
                ).format(STUDY_FILENAME),
            )
            return
        self.set_study(study)

    def _on_close_study(self) -> None:
        """Unload the currently-open study from the panel.

        Scope (important for audit):

        * Touches **nothing** on disk — this is purely an in-memory
          operation.  The ``.cfp-study.json`` sidecar, the CSVs, the
          per-file ``*.overrides.json`` sidecars all stay put.
        * Since every structural edit autosaves (see
          :meth:`_save_and_refresh`), there is no "unsaved work" state
          to guard against here — at the moment the user clicks Close,
          the sidecar on disk is already the authoritative copy of the
          in-memory model.  Consequence: **no confirmation dialog**.
          We keep the UX friction-free precisely because this action
          is safe.

        After close, the panel shows its empty-state hint again; the
        user can open a different study or create a new one.
        """
        if self._study is None:
            return
        logger.info(
            "StudyPanel: closing study %r (folder=%s) — in-memory "
            "unload only, no disk I/O",
            self._study.name, self._study.folder,
        )
        self.set_study(None)

    def _on_add_group(self) -> None:
        if self._study is None:
            return
        dlg = _GroupDialog(
            title=self.tr("Nuovo gruppo"),
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        vals = dlg.values()
        if not vals["name"]:
            QMessageBox.warning(
                self, self.tr("Nome mancante"),
                self.tr("Il nome del gruppo non può essere vuoto."),
            )
            return

        group = Group(**vals)
        try:
            add_group(self._study, group)
        except ValueError as e:   # duplicate name / empty name
            QMessageBox.warning(
                self, self.tr("Impossibile aggiungere"), str(e),
            )
            return

        self._save_and_refresh()

    def _on_add_file(self) -> None:
        """Attach one or more CSV files to the currently-selected group.

        Files must live under the study folder (:func:`make_file_entry`
        enforces it).  We point the file dialog at the study root by
        default so the user doesn't have to re-navigate every time.
        """
        if self._study is None:
            return
        group = self._selected_group()
        if group is None:
            QMessageBox.information(
                self, self.tr("Seleziona un gruppo"),
                self.tr(
                    "Seleziona prima un gruppo nell'albero, poi "
                    "“Aggiungi CSV al gruppo”."
                ),
            )
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("Aggiungi CSV al gruppo “{0}”").format(group.name),
            self._study.folder,
            self.tr("CSV files (*.csv);;All files (*)"),
        )
        if not paths:
            return

        outside: list[str] = []
        added = 0
        for p in paths:
            try:
                add_file_to_group(self._study, group.name, p)
            except ValueError:
                # File outside the study folder — collect, report
                # once at the end rather than a spam-dialog per file.
                outside.append(p)
            except KeyError:
                # Can't happen: we just resolved ``group`` above.
                logger.exception(
                    "StudyPanel._on_add_file: group %r vanished", group.name,
                )
            else:
                added += 1

        if outside:
            QMessageBox.warning(
                self, self.tr("File fuori dallo studio"),
                self.tr(
                    "I seguenti file non si trovano dentro la cartella "
                    "dello studio e non sono stati aggiunti:\n\n{0}"
                ).format("\n".join(outside)),
            )

        if added:
            self._save_and_refresh()

    def _on_group_settings(self) -> None:
        """Edit the selected group's metadata (drug/dose/condition/...)."""
        if self._study is None:
            return
        group = self._selected_group()
        if group is None:
            return
        dlg = _GroupDialog(
            title=self.tr("Impostazioni gruppo"),
            name=group.name,
            drug=group.drug,
            dose_uM=group.dose_uM,
            condition=group.condition,
            analysis_date=group.analysis_date,
            notes=group.notes,
            lock_name=True,   # renaming requires care re: Excel sheet names
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        vals = dlg.values()
        # Name is locked in the dialog; defend against future tampering.
        if vals["name"] != group.name:
            logger.warning(
                "StudyPanel: group rename attempted via locked dialog "
                "(%r → %r); ignored", group.name, vals["name"],
            )
        group.drug = vals["drug"]
        group.dose_uM = vals["dose_uM"]
        group.condition = vals["condition"]
        group.analysis_date = vals["analysis_date"]
        group.notes = vals["notes"]
        self._save_and_refresh()

    def _on_remove(self) -> None:
        """Remove the selected group or file (with confirmation).

        Scope is intentionally limited: we refuse to remove the study
        itself from here — that's a "delete the folder in Finder" kind
        of operation, not a UI action.
        """
        if self._study is None:
            return
        item = self._tree.currentItem()
        if item is None:
            return
        kind = item.data(0, _KIND_ROLE)
        if kind == "group":
            name = item.data(0, _GROUP_NAME_ROLE)
            if QMessageBox.question(
                self, self.tr("Rimuovi gruppo"),
                self.tr("Rimuovere il gruppo “{0}” e tutti i suoi file "
                         "dall'elenco?  I file CSV su disco NON verranno "
                         "eliminati.").format(name),
            ) != QMessageBox.Yes:
                return
            remove_group(self._study, name)
            self._save_and_refresh()
        elif kind == "file":
            group_name = item.data(0, _GROUP_NAME_ROLE)
            file_idx = item.data(0, _FILE_INDEX_ROLE)
            group = find_group(self._study, group_name)
            if group is None or not (0 <= file_idx < len(group.files)):
                return
            if QMessageBox.question(
                self, self.tr("Rimuovi file"),
                self.tr("Rimuovere il file dall'elenco del gruppo?  "
                         "Il CSV su disco NON verrà eliminato."),
            ) != QMessageBox.Yes:
                return
            del group.files[file_idx]
            self._save_and_refresh()
        # kind == "study" or None: nothing to do.

    def _on_analyze_group(self) -> None:
        """Run the full pipeline on every file of the selected group.

        Step 3 of task #84 with the step-3b upgrade: work runs on a
        :class:`QThreadPool` so the UI thread stays responsive for
        the duration of the batch (tree redraws, mouse moves, even
        tooltips still work while 4 files process in parallel).

        Behaviour:

        * Channel selection follows :attr:`FileEntry.channel`.  The
          default value is ``'auto'`` (assigned by ``make_file_entry``)
          so out-of-the-box batch picks the best electrode per file via
          :func:`select_best_channel`.  Users who have explicitly pinned
          EL1 / EL2 on a file (e.g. because they know the other
          electrode is dead on that chip) keep that preference.
        * Each group carries its own :class:`AnalysisConfig` — batch
          honours it, so a baseline group and a dose group can use
          different QC thresholds without touching global settings.
        * Files whose CSV is missing are rejected up-front (before any
          worker is submitted) so the pool only ever runs useful work.
        * Parallelism is capped at
          ``min(idealThreadCount(), _BATCH_MAX_THREADS)`` — pragmatic
          ceiling, see the top-of-module rationale.
        * Results are cached by (group_name, csv_relpath) in
          :attr:`_results`; badges in the tree reflect ✓ / ✗ / — and
          tooltips carry the error message or beat counts.  The full
          :class:`AnalysisResult` is **not** cached (memory) — a
          double-click on a file still re-runs the pipeline to
          populate the editing tabs.
        * Cancel aborts *pending* workers immediately via
          ``pool.clear()``; workers already running finish their
          current file and their results are discarded.  The tree is
          still rebuilt so the already-completed files show their
          badges.

        Audit note: exceptions from the scientific pipeline are
        logged via ``logger.exception`` with the file path, so a
        terminal audit shows *which* file failed and why.
        """
        if self._study is None:
            return
        group = self._selected_group()
        if group is None:
            QMessageBox.information(
                self, self.tr("Seleziona un gruppo"),
                self.tr(
                    "Seleziona prima un gruppo nell'albero, poi "
                    "“Analizza gruppo”."
                ),
            )
            return
        if not group.files:
            QMessageBox.information(
                self, self.tr("Gruppo vuoto"),
                self.tr(
                    "Il gruppo “{0}” non contiene file.  Aggiungi "
                    "almeno un CSV prima di avviare l'analisi."
                ).format(group.name),
            )
            return

        n = len(group.files)

        # ── 1. Resolve paths + split into (ready | missing) up-front ──
        # Missing files don't become runnables — we write their error
        # straight into the cache and account for them in the progress
        # total.  This keeps the pool honest (no fake work) and means
        # the per-file progress bar represents real pipeline work.
        ready: list[tuple[int, Path, str]] = []  # (idx, abspath, channel)
        for i, fe in enumerate(group.files):
            key = (group.name, fe.csv_relpath)
            abspath = resolve_file_path(self._study, fe)
            if not abspath.is_file():
                self._results[key] = FileResult(
                    status='error',
                    error=self.tr(
                        "File non trovato: {0}"
                    ).format(abspath),
                )
                continue
            # Clamp the per-file channel field.  We only trust the
            # three known values; anything else (legacy / corrupt
            # sidecar) falls back to 'auto' rather than propagating a
            # bogus label into the pipeline.
            ch = fe.channel if fe.channel in ('auto', 'EL1', 'EL2') else 'auto'
            ready.append((i, abspath, ch))

        logger.info(
            "StudyPanel: batch analyze — group=%r, n_files=%d "
            "(ready=%d, missing=%d), study=%s",
            group.name, n, len(ready), n - len(ready), self._study.folder,
        )

        # ── 2. Short-circuit: every file missing → just refresh tree ──
        if not ready:
            self._rebuild_tree()
            return

        # ── 3. Wire up progress dialog + result collector + event loop ──
        progress = QProgressDialog(
            self.tr("Analisi gruppo “{0}”...").format(group.name),
            self.tr("Annulla"),
            0, n, self,
        )
        progress.setWindowTitle(self.tr("Analisi gruppo"))
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)   # show immediately for small N
        # Missing files already counted in the cache — reflect them in
        # the progress bar so the user sees "(k/n) …" honestly.
        already_done = n - len(ready)
        progress.setValue(already_done)

        loop = QEventLoop(self)
        signals = _BatchWorkerSignals()
        pool = QThreadPool(self)
        pool.setMaxThreadCount(
            min(QThread.idealThreadCount() or 1, _BATCH_MAX_THREADS)
        )

        # Mutable flags packed in a dict so the inner closures can
        # mutate without needing ``nonlocal`` (keeps the diff local).
        state = {'completed': already_done, 'cancelled': False}

        def _on_file_done(file_idx: int, result: FileResult) -> None:
            # Runs on the UI thread (QueuedConnection), so it's safe to
            # touch ``self._results``, the progress dialog, and the
            # tree.  We swallow results that come in after a cancel —
            # the worker was already committed to its file, but the
            # user has moved on.
            if state['cancelled']:
                return
            fe = group.files[file_idx]
            key = (group.name, fe.csv_relpath)
            self._results[key] = result
            state['completed'] += 1
            progress.setValue(state['completed'])
            progress.setLabelText(
                self.tr("({0}/{1}) {2}").format(
                    state['completed'], n, fe.csv_relpath,
                )
            )
            if state['completed'] >= n:
                loop.quit()

        signals.done.connect(_on_file_done, Qt.QueuedConnection)

        def _on_cancel() -> None:
            state['cancelled'] = True
            logger.info(
                "StudyPanel: batch analyze cancelled — completed=%d/%d",
                state['completed'], n,
            )
            pool.clear()   # drop pending runnables; in-flight workers
                           # still finish their current file.
            loop.quit()

        progress.canceled.connect(_on_cancel)

        # ── 4. Submit all workers ─────────────────────────────────────
        for file_idx, abspath, ch in ready:
            worker = _BatchWorker(
                file_index=file_idx,
                abspath=str(abspath),
                channel=ch,
                config=group.config,
                signals=signals,
            )
            pool.start(worker)

        # ── 5. Block UI thread here (but keep the event loop pumping)
        # so signal/slot still dispatches, tree repaints, etc.
        loop.exec()

        # ── 6. On cancel, wait for in-flight workers so we don't tear
        # down ``signals`` while they're still emitting.  Their results
        # are dropped in ``_on_file_done`` via the ``cancelled`` guard.
        if state['cancelled']:
            pool.waitForDone()

        # Final book-keeping: make sure the bar reaches 100% and the
        # tree picks up whatever results landed before the cancel.
        progress.setValue(n)
        self._rebuild_tree()

    def _on_delete_study(self) -> None:
        """Delete the study **sidecar** from disk, keep CSVs intact.

        This is the only panel action that removes a *file* (as opposed
        to rewriting one).  Spelled out in full because it will be
        audited:

        Affects
            Exactly one file on disk: ``<folder>/.cfp-study.json``
            (``STUDY_FILENAME`` constant).

        Does NOT affect
            * the study **folder** itself (only the sidecar inside it);
            * the CSV files of the experimental signals;
            * per-file override sidecars (``*.overrides.json``) — those
              live next to the CSVs, are keyed by CSV path, and remain
              valid even if the study mapping is removed;
            * any other file in the folder.

        Intended use-cases
            * The user wants to stop treating a folder as a study but
              keep the raw data (renaming, re-organising, hand-off).
            * The study JSON is corrupt and the user wants a clean
              slate without losing the CSVs that the sidecar was
              indexing.

        UX safety rails
            * Button lives at the far end of the toolbar, separated by
              a toolbar separator from editing actions, so accidental
              click-streaks land on ``Rimuovi`` (non-destructive for
              CSVs) before reaching here.
            * Modal dialog uses :class:`QMessageBox.warning` (amber
              icon) with explicit wording about what is and is not
              affected.  Default button is **No**, so hammering ↵ on an
              open dialog will NOT delete anything.
            * After deletion, the panel unloads the in-memory study
              so the stale state can't be re-saved and silently
              resurrect the sidecar.

        Audit trail
            * On confirm we log ``logger.warning`` with the absolute
              path *and* the study name; the log line precedes the
              ``os.remove`` call so the audit record exists even if
              the unlink then fails.
            * Failure modes (``OSError``) surface to the user via
              :class:`QMessageBox.critical` and are re-logged with
              ``logger.exception`` — no silent swallow.
        """
        if self._study is None:
            return
        sidecar = Path(self._study.folder) / STUDY_FILENAME
        study_name = self._study.name
        study_folder = self._study.folder

        # Short-circuit: sidecar already gone (user removed it from
        # Finder while the panel was open).  Offer to clear the panel
        # rather than erroring, because that is almost certainly what
        # the user wanted.
        if not sidecar.exists():
            logger.info(
                "StudyPanel: delete requested but sidecar %s already "
                "missing — unloading panel", sidecar,
            )
            self.set_study(None)
            return

        # Explicit confirmation.  Text lists every artefact the user
        # might care about — CSVs, overrides, cartella — so "Yes" is
        # an informed click, not a default-to-destructive.
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(self.tr("Elimina sidecar studio"))
        box.setText(self.tr(
            "Eliminare il file di studio “{0}”?"
        ).format(STUDY_FILENAME))
        box.setInformativeText(self.tr(
            "Verrà eliminato SOLO il file:\n\n"
            "    {0}\n\n"
            "NON verranno toccati:\n"
            "  • la cartella dello studio\n"
            "  • i file CSV di segnale\n"
            "  • i file di correzione per-segnale "
            "(*.overrides.json)\n\n"
            "Operazione non annullabile."
        ).format(sidecar))
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)   # ↵ = Annulla, non elimina.
        if box.exec() != QMessageBox.Yes:
            return

        # ── Audit record (emitted BEFORE the unlink on purpose, so the
        #    trail exists even if the OS then refuses the delete).
        logger.warning(
            "StudyPanel: deleting study sidecar — user=%s, "
            "study=%r, folder=%s, sidecar=%s",
            os.getlogin() if _safe_getlogin_available() else "unknown",
            study_name, study_folder, sidecar,
        )

        try:
            os.remove(sidecar)
        except OSError as e:
            logger.exception(
                "StudyPanel: failed to delete sidecar %s", sidecar,
            )
            QMessageBox.critical(
                self, self.tr("Eliminazione fallita"),
                self.tr(
                    "Impossibile eliminare il file di studio:\n\n"
                    "    {0}\n\n"
                    "Errore: {1}"
                ).format(sidecar, e),
            )
            return

        # Unload the now-orphaned in-memory study so a later edit
        # cannot silently resurrect the sidecar via autosave.
        self.set_study(None)

    # ═════════════════════════════════════════════════════════════════
    #  Tree events
    # ═════════════════════════════════════════════════════════════════

    def _on_item_double_clicked(
        self, item: QTreeWidgetItem, _column: int,
    ) -> None:
        """Double-clicking a file row emits :attr:`file_activated`."""
        if self._study is None:
            return
        if item.data(0, _KIND_ROLE) != "file":
            return
        group_name = item.data(0, _GROUP_NAME_ROLE)
        file_idx = item.data(0, _FILE_INDEX_ROLE)
        group = find_group(self._study, group_name)
        if group is None or not (0 <= file_idx < len(group.files)):
            return
        abspath = resolve_file_path(self._study, group.files[file_idx])
        if not abspath.is_file():
            QMessageBox.warning(
                self, self.tr("File mancante"),
                self.tr(
                    "Il file non esiste più sul disco:\n{0}"
                ).format(abspath),
            )
            return
        self.file_activated.emit(str(abspath))

    # ═════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ═════════════════════════════════════════════════════════════════

    def _save_and_refresh(self) -> None:
        """Persist the study to disk, then rebuild the tree.

        On save failure we leave the in-memory model edited but warn
        the user — better than silently reverting, which would lose
        work the user thought they'd committed.

        Also prunes stale batch-analysis cache entries (files that no
        longer exist in the study after the mutation that just fired).
        We *don't* clear the whole cache — other files in the group
        still have valid results, and the PoC does best-effort
        preservation.
        """
        if self._study is None:
            return
        try:
            save_study(self._study)
        except OSError as e:
            QMessageBox.warning(
                self, self.tr("Salvataggio fallito"),
                self.tr(
                    "Impossibile salvare lo studio su disco ({0}).  "
                    "Le modifiche restano in memoria ma potrebbero "
                    "andare perse alla chiusura."
                ).format(e),
            )
        self._prune_cache()
        self._rebuild_tree()
        self._refresh_actions()
        self.study_changed.emit()

    def _prune_cache(self) -> None:
        """Drop cache entries whose (group, file) pair no longer exists.

        Called after every mutation so removing a file / group /
        study doesn't leave ghost badges behind.  Entries for files
        that still exist keep their last known status (✓ / ✗) — the
        user has to re-run batch to refresh.
        """
        if self._study is None:
            self._results.clear()
            return
        valid = {
            (g.name, f.csv_relpath)
            for g in self._study.groups
            for f in g.files
        }
        stale = [k for k in self._results if k not in valid]
        for k in stale:
            del self._results[k]

    def _selected_group(self) -> Group | None:
        """Return the currently-selected group (walks up from file rows)."""
        if self._study is None:
            return None
        item = self._tree.currentItem()
        while item is not None:
            kind = item.data(0, _KIND_ROLE)
            if kind == "group":
                return find_group(
                    self._study, item.data(0, _GROUP_NAME_ROLE),
                )
            item = item.parent()
        return None

    def _rebuild_tree(self) -> None:
        """Repopulate the tree from :attr:`_study`.  O(N) rows.

        Batch-analysis state (step 3) is layered in:

        * File rows get a status badge prefix in column 0 — ``✓`` /
          ``✗`` / no prefix for "never analysed".
        * Error status colours the row red; ok leaves the default
          theme colour (don't fight the user's OS theme).
        * Tooltip on column 0 carries the full error message or the
          channel + beat counts (gives the user something to read
          before committing to a double-click re-analyse).
        * Group rows get an aggregate suffix in column 1 —
          ``"… · 3 file · 2✓ 1✗"`` — so the user sees at a glance
          which groups still have work to do.
        """
        self._tree.clear()
        if self._study is None:
            self._hint.show()
            self._tree.hide()
            return
        self._hint.hide()
        self._tree.show()

        root = QTreeWidgetItem(self._tree)
        root.setText(0, self._study.name)
        root.setText(1, self.tr("{0} gruppi").format(len(self._study.groups)))
        root.setData(0, _KIND_ROLE, "study")
        root.setExpanded(True)

        for group in self._study.groups:
            g_item = QTreeWidgetItem(root)
            g_item.setText(0, group.name)

            # Aggregate batch status for the group, appended after the
            # normal details — kept as a suffix so existing logic that
            # reads ``_format_group_details`` stays pure.
            details = _format_group_details(group)
            agg = _aggregate_status(
                [self._results.get((group.name, f.csv_relpath))
                 for f in group.files]
            )
            if agg:
                details = f"{details}  ·  {agg}"
            g_item.setText(1, details)
            g_item.setData(0, _KIND_ROLE, "group")
            g_item.setData(0, _GROUP_NAME_ROLE, group.name)
            g_item.setExpanded(True)

            for idx, entry in enumerate(group.files):
                f_item = QTreeWidgetItem(g_item)
                f_details = entry.csv_relpath
                if entry.channel and entry.channel != "auto":
                    f_details = f"{f_details}  ·  {entry.channel.upper()}"
                if entry.note:
                    f_details = f"{f_details}  ·  {entry.note}"
                f_item.setData(0, _KIND_ROLE, "file")
                f_item.setData(0, _GROUP_NAME_ROLE, group.name)
                f_item.setData(0, _FILE_INDEX_ROLE, idx)

                # Badge + tooltip from the batch-result cache.
                res = self._results.get((group.name, entry.csv_relpath))
                badge = _badge_for(res)
                base_name = Path(entry.csv_relpath).name
                f_item.setText(
                    0,
                    f"{badge} {base_name}" if badge else base_name,
                )
                f_item.setText(1, f_details)
                tip = _tooltip_for(res)
                if tip:
                    f_item.setToolTip(0, tip)
                    f_item.setToolTip(1, tip)
                # Colour cue for errors only — keep success on the
                # default theme palette so we don't fight dark mode.
                if res is not None and res.status == 'error':
                    brush = QBrush(QColor("#b00020"))   # matte red
                    f_item.setForeground(0, brush)
                    f_item.setForeground(1, brush)

        # Stretch the first column to fit the deepest row, then let the
        # details column take the remaining width.
        self._tree.resizeColumnToContents(0)

    def _refresh_actions(self, *_args) -> None:
        """Enable/disable toolbar actions based on current selection.

        Rule of thumb:

        * *New / Open* are always enabled (lifecycle entry points).
        * *Close / Delete sidecar* require a loaded study.
        * *Add group* requires a loaded study.
        * *Add CSV / Group settings / Remove* additionally require a
          group-or-file selection in the tree (we walk up from file
          rows via :meth:`_selected_group` for file-level selections,
          so either kind is a valid anchor).
        """
        has_study = self._study is not None

        # Lifecycle group — Close / Delete depend on "study loaded".
        self._act_close.setEnabled(has_study)
        self._act_delete_study.setEnabled(has_study)

        # Editing group.
        self._act_add_group.setEnabled(has_study)

        item = self._tree.currentItem()
        kind = item.data(0, _KIND_ROLE) if item is not None else None
        can_edit_group = has_study and kind in ("group", "file")
        self._act_add_file.setEnabled(can_edit_group)
        self._act_group_settings.setEnabled(can_edit_group)
        self._act_remove.setEnabled(can_edit_group)
        # Batch analyze is gated on the same "a group (or a file within
        # a group) is selected" rule — we walk up to the group in
        # ``_selected_group`` so either kind is a valid anchor.
        self._act_analyze_group.setEnabled(can_edit_group)


# ─────────────────────────────────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────────────────────────────────

def _format_group_details(group: Group) -> str:
    """Human-readable one-liner for a :class:`Group` tree row.

    Kept as a module-level helper so it can be unit-tested without
    spinning up Qt (important: tests run headless in CI).
    """
    parts: list[str] = []
    if group.drug:
        parts.append(group.drug)
    if group.dose_uM is not None:
        parts.append(f"{group.dose_uM:g} µM")
    if group.condition:
        parts.append(group.condition)
    parts.append(_n_files_label(len(group.files)))
    return " · ".join(parts)


def _n_files_label(n: int) -> str:
    """Italian-aware pluralisation for "N file"."""
    return "1 file" if n == 1 else f"{n} file"


def _len_or_zero(obj) -> int:
    """``len(obj)`` with a safe fallback to 0 for ``None`` / unsized.

    Used to pull beat counts out of an ``AnalysisResult`` dict where
    keys might be missing on older schemas or ``None`` on edge cases
    (analyse-empty-signal).  Keeps ``_on_analyze_group`` free of
    nested ``if`` ladders.
    """
    try:
        return len(obj) if obj is not None else 0
    except TypeError:
        return 0


def _badge_for(res: FileResult | None) -> str:
    """Return the column-0 prefix glyph for a file row.

    * ``None`` (never analysed) → empty string, no prefix.
    * ``ok`` → ``'✓'``.
    * ``error`` → ``'✗'``.

    Kept as a module-level pure function so it's unit-testable without
    spinning up Qt — same rationale as :func:`_format_group_details`.
    """
    if res is None:
        return ""
    if res.status == 'ok':
        return "✓"
    if res.status == 'error':
        return "✗"
    return ""


def _tooltip_for(res: FileResult | None) -> str:
    """Tooltip text for a file row in the tree.

    For successful runs we surface the channel chosen by
    ``select_best_channel`` and the included-beats count, which is
    what the user cares about before committing to a double-click
    deep-dive.  For errors we dump the full error message (what /
    where — the user can then re-open the CSV and iterate).
    """
    if res is None:
        return ""
    if res.status == 'error':
        return f"Errore: {res.error}" if res.error else "Errore"
    # status == 'ok'
    parts: list[str] = []
    if res.channel_analyzed:
        parts.append(f"Canale: {res.channel_analyzed.upper()}")
    if res.n_total:
        if res.n_included != res.n_total:
            parts.append(f"{res.n_included}/{res.n_total} battiti inclusi")
        else:
            parts.append(f"{res.n_total} battiti")
    return "  ·  ".join(parts)


def _aggregate_status(results: list[FileResult | None]) -> str:
    """Summary string for a group row — ``"2✓ 1✗"`` / ``"all ✓"`` / ``""``.

    The PoC convention:

    * no results at all → empty string (don't clutter the row before
      the user has run batch).
    * all ok → compact ``"✓ {n}/{n}"``.
    * mixed → ``"{ok}✓ {err}✗"``, with the untouched remainder
      implicit (they have no prefix on the individual rows either).
    """
    if not results:
        return ""
    n_ok = sum(1 for r in results if r is not None and r.status == 'ok')
    n_err = sum(1 for r in results if r is not None and r.status == 'error')
    if n_ok == 0 and n_err == 0:
        return ""
    if n_err == 0 and n_ok == len(results):
        return f"✓ {n_ok}/{len(results)}"
    parts: list[str] = []
    if n_ok:
        parts.append(f"{n_ok}✓")
    if n_err:
        parts.append(f"{n_err}✗")
    return " ".join(parts)


def _safe_getlogin_available() -> bool:
    """Return True iff ``os.getlogin()`` works in this environment.

    Used to stamp audit-log lines with the OS user.  On some minimal
    container / systemd / no-controlling-tty setups ``os.getlogin``
    raises ``OSError`` ("Inappropriate ioctl for device"), so we probe
    once at call time rather than crashing the delete path.  Getting
    "unknown" in the log is acceptable; crashing the unlink isn't.
    """
    try:
        os.getlogin()
    except OSError:
        return False
    return True
