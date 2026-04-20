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

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, is_dataclass
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
    QComboBox,
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
    FileEntry,
    Group,
    Study,
    add_file_to_group,
    add_group,
    find_group,
    load_study,
    make_file_entry,
    remove_group,
    resolve_file_path,
    save_study,
)

from . import theme

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
    fingerprint : str
        Hash (:func:`_result_fingerprint`) of the inputs that produced
        this result — currently ``(group.config, file.channel)``.
        The tree compares it to the *current* fingerprint to decide
        whether to show the ``●`` stale badge.  Empty for legacy
        cached entries pre-3c: those are tolerated as "fresh" rather
        than flagged stale, on the assumption that the user hasn't
        touched the config since loading the study.
    fpd_ms, fpdc_ms, bpm, stv_ms : float
        Scalar batch metrics extracted from ``result['summary']``
        (step 4 — per-group aggregate).  All default to ``NaN`` so
        legacy cached entries pre-step-4 (and error results) keep
        rendering without breaking the aggregate helpers — aggregate
        / tooltip code must tolerate NaN via :func:`math.isnan`.

        Choice of these four is scientific:

        * ``fpd_ms``   — field-potential duration median, raw
          (primary electrophysiology endpoint).
        * ``fpdc_ms``  — rate-corrected FPD median (dose-response
          primary endpoint; analogue of QTc in ECG).
        * ``bpm``      — mean beat rate (chronotropic effect).
        * ``stv_ms``   — short-term variability (arrhythmogenic
          risk proxy — small ≈ stable, high ≈ unstable).

        We deliberately store point estimates (not arrays) — the
        per-beat distribution stays in the pipeline's result dict
        and the deep-dive tab re-loads it on double-click.  The
        study-panel cache is for *decisions at a glance*, not for
        re-running statistics.
    """

    status: str                      # 'ok' | 'error'
    channel_analyzed: str = ''
    n_included: int = 0
    n_total: int = 0
    error: str = ''
    fingerprint: str = ''
    # Scientific scalar metrics — NaN sentinel for "missing / error".
    fpd_ms: float = float('nan')
    fpdc_ms: float = float('nan')
    bpm: float = float('nan')
    stv_ms: float = float('nan')


# ═══════════════════════════════════════════════════════════════════════
#  Dose-response data model (step 5 of task #84)
# ═══════════════════════════════════════════════════════════════════════
#
#  A dose-response curve is the scientific end-point of the Study model:
#  once each group has been batch-analysed, we aggregate across files
#  (done in step 4) and then across groups (here) to see how the metric
#  of interest moves with increasing drug concentration.
#
#  The aggregated per-group point is kept as its own dataclass instead
#  of a bare tuple so (a) tests can match on names and (b) future
#  extensions (e.g. Hill-fit residuals, CI halfwidths, CDISC ``PPTEST``
#  tags) can add fields without breaking call sites.


@dataclass
class DoseResponsePoint:
    """One aggregated (drug, dose) point in a dose-response curve.

    Attributes
    ----------
    group_name : str
        Name of the :class:`Group` this point was aggregated from —
        used to label the scatter point in the plot and to trace back
        from a chart click to the study tree.
    drug : str
        Free-text drug name from :attr:`Group.drug`.  Empty string
        when the group is a pure condition control (e.g. "vehicle").
        Used to split the plot into one curve per drug.
    dose_uM : float | None
        Dose in µM.  ``None`` means the group is a *baseline* (pre-dose
        control) and is plotted as a reference horizontal line instead
        of as a curve point — log-scale x axes can't render x=0 and
        dose-response figures conventionally show baseline separately.
    mean : float
        Aggregate central tendency across the group's fresh-ok files
        for the selected metric (mean of per-file medians; see
        :func:`_aggregate_group_metrics`).  Always finite — callers
        that would produce NaN are dropped before assembling the list.
    sd : float
        Sample standard deviation (ddof=1) of the per-file values.
        ``0.0`` when the aggregate is built from a single file — that
        special case is noted explicitly in the plot via ``n=1``
        annotation in the tooltip so the zero-error bar isn't read
        as "perfectly consistent".
    n : int
        Number of fresh-ok files that contributed a finite value for
        the selected metric.  May be smaller than the group's total
        file count when some files had stale or NaN results for this
        specific metric.
    """

    group_name: str
    drug: str
    dose_uM: float | None
    mean: float
    sd: float
    n: int


# Dose-response metric whitelist, in the order shown in the dropdown.
# Kept as a module-level constant so tests and the dialog agree on
# ``(key, label, unit, decimals)`` without duplicating the mapping.
#
# * ``key`` matches :class:`FileResult` field + the
#   :func:`_aggregate_group_metrics` dict key — the rename-safe
#   single source of truth.
# * ``label`` is the *Italian* human-readable name (FPDc / FPD / BPM / STV);
#   ``unit`` is appended on the y-axis.  BPM already encodes its unit
#   in the label, so ``unit=''``.
# * ``decimals`` match the tree-row formatting from step 4 — integers
#   for FPD* and BPM, one decimal for STV.
_DOSE_RESPONSE_METRICS: tuple[tuple[str, str, str, int], ...] = (
    ('fpdc_ms', 'FPDc', 'ms', 0),
    ('fpd_ms', 'FPD', 'ms', 0),
    ('bpm', 'BPM', '', 0),
    ('stv_ms', 'STV', 'ms', 1),
)


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


def _collect_batch_inputs(
    study: Study,
    groups: list[Group],
) -> tuple[
    list[tuple[Group, FileEntry, Path, str]],
    dict[tuple[str, str], FileResult],
]:
    """Split every ``(group, file)`` pair across ``groups`` into ready + missing.

    Pure helper pulled out of :meth:`StudyPanel._on_analyze_group` so
    both the per-group and the study-wide batch flows can share one
    ready/missing split, and so the collection logic can be unit-tested
    without spinning up Qt.

    Parameters
    ----------
    study
        The owning :class:`Study` — used by :func:`resolve_file_path`
        to anchor relative CSV paths against ``study.folder``.
    groups
        Subset of ``study.groups`` to batch.  Pass a single-group list
        for *Analizza gruppo*, or ``study.groups`` for *Analizza studio*.
        The caller decides scope; this helper doesn't second-guess the
        selection.

    Returns
    -------
    ready
        One tuple per resolvable CSV: ``(group, file_entry, abspath,
        channel)``.  ``channel`` is clamped to one of
        ``'auto' | 'EL1' | 'EL2'`` — legacy or corrupted sidecar values
        fall back to ``'auto'`` rather than propagate a bogus label
        into the pipeline.  The order mirrors ``groups`` then
        ``group.files`` so the batch's submission order is deterministic
        (matters only for the progress label — pool scheduling is not
        order-preserving).
    missing
        Dict keyed by ``(group_name, csv_relpath)`` mapping to a
        pre-built ``FileResult(status='error', ...)``.  The caller is
        expected to merge this into its result cache before running
        the pool, so the tree shows ✗ badges even for files that
        never got submitted (no phantom "pending" rows).
    """
    ready: list[tuple[Group, FileEntry, Path, str]] = []
    missing: dict[tuple[str, str], FileResult] = {}
    for group in groups:
        for fe in group.files:
            abspath = resolve_file_path(study, fe)
            key = (group.name, fe.csv_relpath)
            if not abspath.is_file():
                # Plain Italian here (no ``tr()``): the string is
                # assembled outside a ``QObject`` context so we can
                # keep this helper Qt-free.  The wider i18n pass
                # (bilingual UI, see project_i18n memory) will route
                # this through ``QCoreApplication.translate`` later.
                missing[key] = FileResult(
                    status='error',
                    error=f"File non trovato: {abspath}",
                )
                continue
            ch = fe.channel if fe.channel in ('auto', 'EL1', 'EL2') else 'auto'
            ready.append((group, fe, abspath, ch))
    return ready, missing


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
        fingerprint: str,
        signals: _BatchWorkerSignals,
    ) -> None:
        super().__init__()
        self._file_index = file_index
        self._abspath = abspath
        self._channel = channel
        self._config = config
        # Fingerprint of (config, channel) computed on the UI thread at
        # submit time — captured here so we stamp the *current* inputs
        # into the :class:`FileResult`, not whatever the user might
        # have edited in the config dialog by the time we emit.
        self._fingerprint = fingerprint
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
                fingerprint=self._fingerprint,
            )
            self._signals.done.emit(self._file_index, fr)
            return

        if result is None:
            fr = FileResult(
                status='error',
                error="Pipeline ritornata vuota",
                fingerprint=self._fingerprint,
            )
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

        # Step 4 — pull the four per-group-aggregate scalars out of
        # ``summary``.  All may be NaN on degenerate signals (e.g.
        # no beats with valid repol); downstream aggregate/tooltip
        # tolerates that.
        fpd, fpdc, bpm, stv = _extract_summary_metrics(result.get('summary'))

        fr = FileResult(
            status='ok',
            channel_analyzed=str(analyzed),
            n_included=n_included,
            n_total=n_total,
            fingerprint=self._fingerprint,
            fpd_ms=fpd,
            fpdc_ms=fpdc,
            bpm=bpm,
            stv_ms=stv,
        )
        self._signals.done.emit(self._file_index, fr)


# ═══════════════════════════════════════════════════════════════════════
#  Background export worker (step 6 of task #84 / #89)
# ═══════════════════════════════════════════════════════════════════════
#
#  The export pipeline re-runs :func:`analyze_single_file` on every
#  fresh-ok file in the study, enriches the per-file ``file_info`` with
#  study-level metadata, and calls :func:`export_send_package` on the
#  collected list.  All of that is CPU-bound (numpy / pandas / xport
#  writers), so we do it on a background :class:`QRunnable` and keep
#  the UI event loop responsive via a modal :class:`QProgressDialog`.
#
#  Unlike :class:`_BatchWorker` — which parallelises one worker per
#  file — the export runs sequentially: the CDISC build step is not
#  embarrassingly parallel (it walks a single list of results and
#  merges them into shared DataFrames), and running the per-file
#  pipeline sequentially keeps the progress reporting linear and the
#  error attribution obvious ("fails on file 7" rather than "fails
#  somewhere in the pool").  Studies are small (≤ ~50 files), so
#  single-threaded re-analysis finishes in under a minute on Marco's
#  Mac — the serialisation cost is worth the simpler control flow.
#
#  The worker emits three kinds of signals:
#
#  * ``progress(int done, int total, str filename)`` — per-file tick,
#    drives the progress dialog label + bar.
#  * ``finished(result)`` — terminal success: ``result`` is the dict
#    returned by :func:`export_send_package`.
#  * ``failed(str)`` — terminal failure: ``str`` is a user-facing
#    message.  Exceptions are caught and logged (``logger.exception``)
#    so the audit log captures the full traceback.
#
#  Only one of ``finished`` / ``failed`` is ever emitted; ``progress``
#  fires only up to the failure point.


class _ExportWorkerSignals(QObject):
    """Signal carrier for :class:`_ExportWorker`.

    Uses a separate QObject because ``QRunnable`` isn't a ``QObject``
    and cannot own signals — same pattern as :class:`_BatchWorkerSignals`.
    """

    # (completed_index, total, csv_relpath) — relpath is the label
    # the worker wants the UI to show next to the progress bar.
    progress = Signal(int, int, str)
    # Terminal success: exporter result dict (see export_send_package).
    finished = Signal(object)
    # Terminal failure: user-facing message (already localised in the
    # worker — keeps the UI slot free of format-string logic).
    failed = Signal(str)


class _ExportWorker(QRunnable):
    """One-shot worker that builds a CDISC SEND package for a study.

    Parameters
    ----------
    study : Study
        Study being exported — consumed read-only.  The worker takes
        no lock; the UI thread must not mutate ``study.groups`` while
        the export is in flight (enforced in practice by the modal
        progress dialog, which blocks all toolbar actions).
    pairs : list[tuple[Group, FileEntry]]
        Fresh-ok ``(group, file)`` pairs produced by
        :func:`_collect_export_inputs`.  Captured here rather than
        re-walked inside ``run`` so the snapshot is deterministic
        even if the UI somehow mutates the study concurrently.
    output_dir : str
        Destination folder for the SEND package (``.xpt`` + define.xml).
        The :func:`export_send_package` function creates it if missing.
    study_id : str
        User-chosen CDISC ``STUDYID`` — typically coming from
        :func:`_suggest_study_id` with a chance for the user to edit.
    study_title, sponsor : str
        Pass-through to ``export_send_package``.
    signals : _ExportWorkerSignals
        Shared signal carrier for progress / finished / failed.
    """

    def __init__(
        self,
        *,
        study: Study,
        pairs: list[tuple[Group, FileEntry]],
        output_dir: str,
        study_id: str,
        study_title: str,
        sponsor: str,
        signals: _ExportWorkerSignals,
    ) -> None:
        super().__init__()
        self._study = study
        self._pairs = pairs
        self._output_dir = output_dir
        self._study_id = study_id
        self._study_title = study_title
        self._sponsor = sponsor
        self._signals = signals
        # Cancellation flag written from the UI thread (setter).  GIL-
        # protected bool read in the loop — no lock needed.
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation.  Safe to call from any thread.

        The worker checks the flag between files; an already-running
        ``analyze_single_file`` call finishes (no cooperative cancel
        inside numpy), but subsequent files are skipped and the worker
        terminates with a ``failed`` signal.
        """
        self._cancelled = True

    def run(self) -> None:   # noqa: D401 — Qt virtual
        # Import here (not at module top) so the panel still loads
        # even if the scientific / CDISC cores have an import error —
        # same rationale as :class:`_BatchWorker`.
        try:
            from cardiac_fp_analyzer.analyze import analyze_single_file
            from cardiac_fp_analyzer.cdisc_export import export_send_package
        except Exception as exc:   # noqa: BLE001
            logger.exception("ExportWorker: failed to import scientific core")
            self._signals.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )
            return

        enriched_results: list[dict] = []
        total = len(self._pairs)

        for i, (group, fe) in enumerate(self._pairs):
            if self._cancelled:
                self._signals.failed.emit("__cancelled__")
                return

            abspath = resolve_file_path(self._study, fe)
            self._signals.progress.emit(i, total, fe.csv_relpath)

            try:
                raw = analyze_single_file(
                    str(abspath),
                    channel=fe.channel if fe.channel in ('auto', 'EL1', 'EL2') else 'auto',
                    verbose=False,
                    config=group.config,
                )
            except Exception as exc:   # noqa: BLE001
                logger.exception(
                    "ExportWorker: analyze_single_file failed for %s", abspath,
                )
                self._signals.failed.emit(
                    f"{fe.csv_relpath}: {type(exc).__name__}: {exc}"
                )
                return

            if raw is None:
                # Pipeline returned empty — don't silently drop the
                # file; surface it so the user knows the export isn't
                # a complete snapshot of the study.
                self._signals.failed.emit(
                    f"{fe.csv_relpath}: pipeline ritornata vuota"
                )
                return

            enriched = _enrich_file_info_for_export(raw, group, fe)
            enriched_results.append(enriched)

        # Final tick so the progress bar reaches N/N before the modal
        # dialog closes.
        self._signals.progress.emit(total, total, '')

        if self._cancelled:
            self._signals.failed.emit("__cancelled__")
            return

        if not enriched_results:
            self._signals.failed.emit("Nessun risultato fresco-ok da esportare.")
            return

        try:
            out = export_send_package(
                enriched_results,
                self._output_dir,
                study_id=self._study_id,
                study_title=self._study_title,
                sponsor=self._sponsor,
            )
        except Exception as exc:   # noqa: BLE001
            logger.exception("ExportWorker: export_send_package failed")
            self._signals.failed.emit(
                f"export_send_package: {type(exc).__name__}: {exc}"
            )
            return

        self._signals.finished.emit(out)


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
#  Dose-response dialog (step 5 of task #84)
# ═══════════════════════════════════════════════════════════════════════
#
#  Minimal PoC-grade plot: scatter + error bars + dashed baseline.
#  Scope was deliberately kept small (no Hill / EC50 fit, no custom PNG
#  button — right-click menu from pyqtgraph already exports) so the
#  feature ships and Marco can validate the numbers against his bench
#  notebook before investing in curve fitting.
#
#  Colour palette is 6 colours chosen for simultaneous-drug
#  distinguishability on both light and dark backgrounds; we cycle
#  through it when a study contains > 6 drugs.  Palette is the
#  ColorBrewer "Set1" (six-class) minus the "neutral grey" slot,
#  because grey risks reading as "disabled" on the scatter.


# ColorBrewer Set1 minus the neutral grey — 6 colours.
_DRUG_PALETTE: tuple[str, ...] = (
    '#e41a1c',   # red
    '#377eb8',   # blue
    '#4daf4a',   # green
    '#984ea3',   # purple
    '#ff7f00',   # orange
    '#a65628',   # brown
)


def _drug_colour(drug_index: int) -> str:
    """Return a hex colour string for the N-th drug in the legend.

    Cycles through :data:`_DRUG_PALETTE` modulo its length — studies
    with more than 6 drugs will reuse colours, which is a known limit
    of the PoC palette (out of scope to solve for now).
    """
    return _DRUG_PALETTE[drug_index % len(_DRUG_PALETTE)]


class DoseResponseDialog(QDialog):
    """Modal dialog with a dose-response plot for one metric at a time.

    Inputs
    ------
    study : Study
        The loaded :class:`Study`; groups + config drive the aggregate.
    results_cache : dict
        ``StudyPanel._results`` (same object, not a copy) — read-only
        from this dialog's perspective; the panel still owns it.

    Interaction
    -----------
    * Metric dropdown (FPDc / FPD / BPM / STV) — re-aggregates and
      re-plots without closing the dialog.
    * "Asse X logaritmico" checkbox — toggle between log and linear
      x scale.  Disabled when any dose ≤ 0 in the data (log can't
      render those).  Default on when all doses are positive — the
      scientific convention for drug dose-response figures.
    * pyqtgraph's built-in right-click menu on the plot still works
      for zoom/pan/export-PNG, so we don't duplicate those as
      toolbar buttons.

    Out of scope for this PoC
    -------------------------
    * Hill equation / EC50 fit — step 6 when we wire CDISC export.
    * Per-file drill-down on scatter click — the tree already does
      that via double-click.
    * Custom baseline picker — baseline is always ``dose_uM is None``
      groups.
    """

    def __init__(
        self,
        *,
        study: Study,
        results_cache: dict,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Dose-risposta"))
        self.resize(720, 480)
        self._study = study
        self._results_cache = results_cache

        # Lazy import — pyqtgraph pulls in numpy extras that we don't
        # want on the unit-test import path (the helpers above test
        # headless without any Qt app).
        import pyqtgraph as pg
        self._pg = pg

        # ── Controls row ──────────────────────────────────────────────
        self._metric_combo = QComboBox()
        for key, label, unit, _dec in _DOSE_RESPONSE_METRICS:
            display = f"{label} ({unit})" if unit else label
            self._metric_combo.addItem(display, userData=key)
        self._metric_combo.setCurrentIndex(0)   # FPDc by default
        self._metric_combo.currentIndexChanged.connect(self._refresh_plot)

        self._chk_logx = QCheckBox(self.tr("Asse X logaritmico"))
        self._chk_logx.setChecked(True)
        self._chk_logx.toggled.connect(self._refresh_plot)

        top = QHBoxLayout()
        top.addWidget(QLabel(self.tr("Metrica:")))
        top.addWidget(self._metric_combo)
        top.addSpacing(12)
        top.addWidget(self._chk_logx)
        top.addStretch(1)

        # ── Plot widget ───────────────────────────────────────────────
        # Palette (background + axis pens + grid alpha) comes from the
        # shared ``theme`` module so the dose-response dialog stays
        # coherent with the Signal viewer and the Battiti template
        # plot.  Live updates on toggle are wired via
        # ``theme.connect(self._on_theme_changed)`` below.
        self._plot = pg.PlotWidget()
        theme.apply_to_plot(self._plot)
        for ax_name in ('left', 'bottom'):
            ax = self._plot.getAxis(ax_name)
            # Disable SI auto-prefixing — in log-x mode pyqtgraph
            # applies its own prefix to the transformed tick values
            # (which are log10 of the dose), producing nonsensical
            # axes like ``10⁻²⁷…10⁷³`` with a ``(x1e+27)`` suffix.
            # For concentration data we always want raw µM values on
            # the tick labels, so we keep the prefixer off on both
            # axes for symmetry.
            ax.enableAutoSIPrefix(False)
        # Legend — text colour comes from the palette so the entries
        # are readable against the current background.  The legend is
        # re-created at most once (here), never in ``_refresh_plot``
        # (see the long comment there explaining the orphaned-legend
        # bug that motivated the simplification).
        self._legend = self._plot.addLegend(offset=(10, 10))
        theme.apply_to_legend(self._legend)
        # Subscribe to live theme toggles — the dialog can stay open
        # while the user flips Visualizza → Tema.  Unsubscribe on
        # close so we don't leak if the dialog is re-opened.
        theme.connect(self._on_theme_changed)

        # ── Empty-state overlay ───────────────────────────────────────
        # Shown when no group has fresh-ok results for the chosen
        # metric.  Kept as a plain QLabel layered behind the plot so
        # we don't have to manipulate pyqtgraph's text item machinery.
        self._empty_label = QLabel(self.tr(
            "Nessun gruppo con risultati fresh-ok per la metrica scelta.\n"
            "Esegui “Analizza gruppo” su almeno un gruppo."
        ))
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #888; padding: 24px;")
        self._empty_label.setWordWrap(True)
        self._empty_label.hide()

        # ── Close button ──────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        close_btn = btns.button(QDialogButtonBox.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self._plot, 1)
        root.addWidget(self._empty_label)
        root.addWidget(btns)

        self._refresh_plot()

    # ── Theme integration ──────────────────────────────────────────

    def _on_theme_changed(self, _mode: str) -> None:
        """Live-repaint the dose-response plot on theme toggle.

        Re-applies the palette to both plot and legend and triggers a
        full redraw so the axis titles pick up ``theme.label_color()``
        too — ``setLabel(color=...)`` is only honoured at label-set
        time, hence the ``_refresh_plot`` call.
        """
        theme.apply_to_plot(self._plot)
        theme.apply_to_legend(self._legend)
        self._refresh_plot()

    # ── Refresh logic ────────────────────────────────────────────────

    def _refresh_plot(self, *_args) -> None:
        """Recompute aggregates for the current metric and redraw."""
        pg = self._pg
        # Clear previous plot items.  ``PlotItem.clear()`` iterates
        # ``self.items`` and for each one calls ``legend.removeItem``,
        # so the existing legend is emptied automatically — we must
        # NOT detach the legend from the scene here.  A previous
        # attempt did ``self._legend.scene().removeItem(self._legend)``
        # followed by ``addLegend(...)``, but ``PlotItem.addLegend``
        # only creates a new ``LegendItem`` when ``self.legend is
        # None``; after removing from the scene the PlotItem still
        # holds the now-orphaned reference, and every subsequent
        # ``plot(name=...)`` registered to the orphan, so no legend
        # entries ever showed up on screen.
        self._plot.clear()

        metric = self._metric_combo.currentData() or 'fpdc_ms'
        label, unit, decimals = _metric_meta(metric)
        points = _collect_dose_response_points(
            self._study, self._results_cache, metric=metric,
        )

        if not points:
            self._plot.hide()
            self._empty_label.show()
            return
        self._plot.show()
        self._empty_label.hide()

        # Axis labels — the x label is constant ("Dose (µM)"); y
        # encodes the chosen metric + unit.
        #
        # NOTE: we pass the unit INSIDE the label string rather than via
        # the ``units=`` kwarg.  Passing ``units='µM'`` triggers
        # pyqtgraph's SI-prefix auto-scaling, which parses the leading
        # ``µ`` as the micro prefix (10⁻⁶) and then re-applies another
        # prefix on top — in log-x mode this collapses every real
        # concentration to a single column at ~10⁻²⁷ and labels the axis
        # ``Dose (e27µM)``.  Putting "(µM)" in the label text keeps
        # pyqtgraph from interpreting the string as an SI unit.
        y_axis = label if not unit else f"{label} ({unit})"
        fg = theme.label_color()
        self._plot.setLabel('bottom', self.tr("Dose (µM)"), color=fg)
        self._plot.setLabel('left', y_axis, color=fg)

        # Log-x only makes sense when every dose point is > 0; disable
        # the checkbox otherwise so the UI is honest about what's
        # possible.
        log_possible = _can_plot_log_x(points)
        self._chk_logx.setEnabled(log_possible)
        use_log = bool(self._chk_logx.isChecked()) and log_possible
        self._plot.setLogMode(x=use_log, y=False)

        series = _assemble_dose_response_series(points)

        # ── Dose curves — one per drug ────────────────────────────────
        # Drugs are plotted in insertion order (Python dict preserves it
        # since 3.7); pick colour by index so the legend stays stable
        # across redraws for the same study.
        for drug_idx, (drug, curve) in enumerate(series['doses'].items()):
            colour = _drug_colour(drug_idx)
            xs = [p.dose_uM for p in curve]
            ys = [p.mean for p in curve]
            errs = [p.sd for p in curve]

            pen = pg.mkPen(colour, width=2)
            brush = pg.mkBrush(colour)
            # Line connecting the points — read left-to-right as
            # increasing dose; the polyline makes the trend visible
            # at a glance.
            legend_name = drug if drug else self.tr("(senza nome)")
            self._plot.plot(
                xs, ys, pen=pen, symbol='o',
                symbolBrush=brush, symbolPen=pg.mkPen('k', width=0.5),
                symbolSize=9, name=legend_name,
            )
            # Error bars — one segment per point.  Unlike
            # ``PlotDataItem`` (used by ``self._plot.plot(...)``),
            # ``ErrorBarItem`` does NOT auto-transform its x values
            # when the PlotItem is in log-x mode: the ViewBox
            # coordinate system is already log10, so we must feed it
            # ``log10(xs)``.  Passing raw µM values would plot the
            # bars at plot-space positions 0.1…100 — which the axis
            # renders as 10^0.1…10^100 and drags the auto-range up
            # to 10^100, collapsing every real dose to the left edge.
            import numpy as np
            xs_arr = np.asarray(xs, dtype=float)
            xs_plot = np.log10(xs_arr) if use_log else xs_arr
            err_item = pg.ErrorBarItem(
                x=xs_plot,
                y=np.asarray(ys, dtype=float),
                height=2.0 * np.asarray(errs, dtype=float),
                beam=0.0,
                pen=pg.mkPen(colour, width=1.2),
            )
            self._plot.addItem(err_item)

        # ── Baselines — dashed horizontal lines ───────────────────────
        # One line per baseline group, in insertion order.  When the
        # baseline has no drug label ('' / "vehicle"), we tag the
        # legend entry "Baseline" so it's unambiguous; otherwise we
        # prefix "Baseline" to the drug name for multi-drug runs.
        for bl_idx, bl in enumerate(series['baselines']):
            colour = '#555555' if not bl.drug else _drug_colour(
                # Match the colour of the dose curve for the same drug
                # if present — visual anchor between line + baseline.
                list(series['doses'].keys()).index(bl.drug)
                if bl.drug in series['doses'] else bl_idx
            )
            pen = pg.mkPen(colour, width=1.5, style=Qt.DashLine)
            name = (
                self.tr("Baseline {0}").format(bl.drug)
                if bl.drug else self.tr("Baseline")
            )
            line = pg.InfiniteLine(
                pos=bl.mean, angle=0, pen=pen, label=None,
            )
            # pyqtgraph InfiniteLine doesn't show in the default
            # legend; add a proxy plot for legend purposes (empty
            # data, just the pen) and hide its ViewBox footprint.
            self._plot.addItem(line)
            self._plot.plot(
                [], [], pen=pen, name=name,
            )


# ═══════════════════════════════════════════════════════════════════════
#  Study panel
# ═══════════════════════════════════════════════════════════════════════

class StudyPanel(QWidget):
    """Dockable panel that manages a single :class:`Study`.

    Signals
    -------
    file_activated(str, str, str)
        Emitted when the user double-clicks a file row.  Payload is
        (absolute CSV path, group name, csv_relpath).  The group name
        + relpath let ``MainWindow`` track which ``(group, file)`` the
        active file belongs to, so Ricalcola can (a) use
        ``group.config`` as source-of-truth instead of the global app
        config — keeping the fingerprint consistent with the batch
        cache entry — and (b) write the recomputed result back into
        the batch cache via :meth:`update_entry` (issue #6).

    study_changed()
        Emitted after every successful mutation (add / remove / edit /
        load).  Primarily so the window title / status-bar can reflect
        the current study name; the model on disk has already been
        saved by the time this fires.
    """

    file_activated = Signal(str, str, str)
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
        self._act_add_folder = QAction(
            self.tr("Aggiungi cartella al gruppo..."), self,
        )
        self._act_group_settings = QAction(
            self.tr("Impostazioni gruppo..."), self,
        )
        self._act_remove = QAction(self.tr("Rimuovi"), self)
        self._act_analyze_study = QAction(self.tr("Analizza studio..."), self)
        self._act_analyze_group = QAction(self.tr("Analizza gruppo..."), self)
        self._act_dose_response = QAction(self.tr("Dose-risposta..."), self)
        self._act_export_cdisc = QAction(self.tr("Esporta CDISC..."), self)
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
        self._toolbar.addAction(self._act_add_folder)
        self._toolbar.addAction(self._act_group_settings)
        self._toolbar.addAction(self._act_remove)
        self._toolbar.addSeparator()
        # Group 3 — analysis (batch + cross-group plot + CDISC export).
        # "Analizza studio" first because it's the broadest action —
        # Marco's complaint was that re-analysing groups one by one was
        # "un inferno", so the single-click "whole study" path is the
        # primary affordance and "Analizza gruppo" is the surgical
        # re-run after config tweaks.
        self._toolbar.addAction(self._act_analyze_study)
        self._toolbar.addAction(self._act_analyze_group)
        self._toolbar.addAction(self._act_dose_response)
        self._toolbar.addAction(self._act_export_cdisc)
        self._toolbar.addSeparator()
        # Group 4 — destructive. Distanced from the editing group on purpose.
        self._toolbar.addAction(self._act_delete_study)

        self._act_new.triggered.connect(self._on_new_study)
        self._act_open.triggered.connect(self._on_open_study)
        self._act_close.triggered.connect(self._on_close_study)
        self._act_add_group.triggered.connect(self._on_add_group)
        self._act_add_file.triggered.connect(self._on_add_file)
        self._act_add_folder.triggered.connect(self._on_add_folder)
        self._act_group_settings.triggered.connect(self._on_group_settings)
        self._act_remove.triggered.connect(self._on_remove)
        self._act_analyze_study.triggered.connect(self._on_analyze_study)
        self._act_analyze_group.triggered.connect(self._on_analyze_group)
        self._act_dose_response.triggered.connect(self._on_dose_response)
        self._act_export_cdisc.triggered.connect(self._on_export_cdisc)
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

    def _on_add_folder(self) -> None:
        """Recursively attach every CSV under a folder to the selected group.

        Companion to :meth:`_on_add_file`: the native
        ``QFileDialog.getOpenFileNames`` dialog does not let the user
        multi-select CSVs across different subfolders in one pass, which
        is exactly the shape Marco's on-disk data takes (``baseline/``,
        ``dose1/``, ``dose2/`` siblings under the study root).  This
        action asks for a *folder*, globs ``*.csv`` recursively, and
        adds each one.

        Behaviour:

        * Hidden files (name starts with ``'.'``) are skipped — keeps
          macOS ``.DS_Store`` artefacts and the user's own dotfiles out.
        * Duplicates (CSV already in this group, identified by the
          POSIX relative path) are skipped silently and counted in the
          summary — no "file already exists" per-file dialog spam.
        * CSVs outside the study folder (can happen if the user
          navigates out of the study root before confirming) are
          collected and reported once at the end.
        * Scanning is sorted deterministically so the tree order
          follows the filesystem alphabetically, which matches how
          Marco lays dose-response chips out.

        If the dialog returns nothing, we bail silently.  If the scan
        returns zero CSVs, we show a polite "nothing to add" dialog so
        the user doesn't wonder whether the action fired.
        """
        if self._study is None:
            return
        group = self._selected_group()
        if group is None:
            QMessageBox.information(
                self, self.tr("Seleziona un gruppo"),
                self.tr(
                    "Seleziona prima un gruppo nell'albero, poi "
                    "“Aggiungi cartella al gruppo”."
                ),
            )
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            self.tr("Aggiungi cartella al gruppo “{0}”").format(group.name),
            self._study.folder,
        )
        if not folder:
            return

        candidates = _enumerate_csvs_under(Path(folder))
        if not candidates:
            QMessageBox.information(
                self, self.tr("Nessun CSV trovato"),
                self.tr(
                    "Nessun file .csv trovato (ricorsivamente) in “{0}”."
                ).format(folder),
            )
            return

        # Dedup against what's already in the group.
        existing = {fe.csv_relpath for fe in group.files}

        added = 0
        duplicates = 0
        outside: list[str] = []
        for p in candidates:
            try:
                # Use make_file_entry to resolve + validate "inside
                # study folder" without mutating the study yet — this
                # way we can decide to skip duplicates before bloating
                # ``group.files``.
                entry = make_file_entry(self._study, str(p))
            except ValueError:
                outside.append(str(p))
                continue
            if entry.csv_relpath in existing:
                duplicates += 1
                continue
            try:
                add_file_to_group(self._study, group.name, str(p))
            except (ValueError, KeyError):
                # Can't happen: we just resolved ``group`` and just
                # validated ``entry`` above.  Guard is defensive.
                logger.exception(
                    "StudyPanel._on_add_folder: unexpected add failure for %s",
                    p,
                )
                continue
            existing.add(entry.csv_relpath)
            added += 1

        logger.info(
            "StudyPanel: add-folder group=%r folder=%s "
            "added=%d duplicates=%d outside=%d",
            group.name, folder, added, duplicates, len(outside),
        )

        if outside:
            # Trim long lists so the dialog doesn't dwarf the screen —
            # full list is in the audit log anyway.
            preview = "\n".join(outside[:20])
            if len(outside) > 20:
                preview += self.tr("\n… ({0} altri)").format(len(outside) - 20)
            QMessageBox.warning(
                self, self.tr("File fuori dallo studio"),
                self.tr(
                    "I seguenti CSV non si trovano dentro la cartella "
                    "dello studio e non sono stati aggiunti:\n\n{0}"
                ).format(preview),
            )

        if added:
            self._save_and_refresh()
            # A concise summary line so the user knows what happened —
            # posted once, not per-file.
            QMessageBox.information(
                self, self.tr("Aggiunta cartella completata"),
                self.tr(
                    "Aggiunti {0} CSV al gruppo “{1}”.\n"
                    "Duplicati saltati: {2}.\n"
                    "Fuori studio: {3}."
                ).format(added, group.name, duplicates, len(outside)),
            )
        elif duplicates and not outside:
            QMessageBox.information(
                self, self.tr("Nessun nuovo file"),
                self.tr(
                    "Tutti i {0} CSV trovati erano già nel gruppo “{1}”."
                ).format(duplicates, group.name),
            )

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

        This is the single-group entry point; the actual batch
        machinery lives in :meth:`_run_batch`, which is shared with
        :meth:`_on_analyze_study`.  See that method's docstring for
        the cross-group batch contract.  Per-group-specific bits
        handled here:

        * Pre-flight: a group must be selected in the tree (unlike
          *Analizza studio* which is selection-agnostic).
        * Empty group → info dialog instead of a silent no-op; the
          user almost certainly mis-targeted the action.
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

        self._run_batch(
            groups=[group],
            window_title=self.tr("Analisi gruppo"),
            header_text=self.tr("Analisi gruppo “{0}”...").format(group.name),
            include_group_in_row_label=False,
        )

    def _on_analyze_study(self) -> None:
        """Run the full pipeline on every file of every group in the study.

        One-click equivalent of *Analizza gruppo* repeated over the
        whole study — the UX Marco asked for after complaining that
        re-running groups one by one was "un inferno".  Critical
        differences vs. the per-group flow:

        * One progress dialog, one Cancel button, one
          :class:`QThreadPool` for the entire study.  Inter-group
          order does not matter for correctness; the pool schedules
          workers by availability, not by group boundary, so the
          overall wall-clock is governed by file count and
          ``_BATCH_MAX_THREADS`` — not by the number of groups.
        * Per-file label includes the group name so the user can see
          at a glance which group is currently processing.
        * A confirmation dialog *always* precedes the batch, doubling
          as a pre-flight readout of totals (``N file in M gruppi``).
          The dialog defaults to Yes so one Enter keypress proceeds;
          the purpose is to surface the scope, not add friction.

        Gating: the toolbar action is disabled via ``_refresh_actions``
        when the study has no files anywhere, so the empty-state
        branch here is only reached via programmatic triggering (or
        as a race if the last file was just removed).
        """
        if self._study is None:
            return
        study = self._study
        groups_with_files = [g for g in study.groups if g.files]
        if not groups_with_files:
            QMessageBox.information(
                self, self.tr("Studio vuoto"),
                self.tr(
                    "Lo studio “{0}” non contiene file da analizzare.  "
                    "Aggiungi almeno un CSV a un gruppo prima di "
                    "avviare l'analisi dell'intero studio."
                ).format(study.name),
            )
            return

        n_files = sum(len(g.files) for g in groups_with_files)
        n_groups = len(groups_with_files)

        answer = QMessageBox.question(
            self, self.tr("Analizza studio"),
            self.tr(
                "Analizzare {0} file in {1} gruppi dello studio “{2}”?\n\n"
                "L'operazione analizzerà tutti i file di tutti i gruppi "
                "e può richiedere alcuni minuti."
            ).format(n_files, n_groups, study.name),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return

        self._run_batch(
            groups=groups_with_files,
            window_title=self.tr("Analisi studio"),
            header_text=self.tr("Analisi studio “{0}”...").format(study.name),
            include_group_in_row_label=True,
        )

    def _run_batch(
        self,
        *,
        groups: list[Group],
        window_title: str,
        header_text: str,
        include_group_in_row_label: bool,
    ) -> None:
        """Drive one :class:`QThreadPool` batch across the given groups.

        Shared machinery behind *Analizza gruppo* (``groups=[selected]``)
        and *Analizza studio* (``groups=study.groups``).  Kept as a
        single method so both entry points exercise identical code for
        cancellation, fingerprinting, progress accounting and cache
        writes — subtle divergence here would be exactly the kind of
        bug that shows up only on real data.

        Behaviour (invariant across both callers):

        * Channel selection follows :attr:`FileEntry.channel`.  The
          default ``'auto'`` (assigned by ``make_file_entry``) lets the
          pipeline pick the best electrode per file; users who pinned
          EL1 / EL2 on a file keep that preference.
        * Each group carries its own :class:`AnalysisConfig` — baseline
          and dose groups can use different QC thresholds, and each
          worker is submitted with its own group's config so the
          fingerprint and the staleness logic work cross-group.
        * Files whose CSV is missing are rejected up-front: their ✗
          rows land in the cache before any worker starts, so the
          progress bar's denominator matches the tree's badge count.
        * Parallelism is capped at
          ``min(idealThreadCount(), _BATCH_MAX_THREADS)`` — pragmatic
          ceiling, see the top-of-module rationale.
        * Results are cached by ``(group_name, csv_relpath)`` in
          :attr:`_results`; only the summary is kept — double-click
          re-runs the pipeline to populate the editing tabs.
        * Cancel aborts *pending* workers immediately via
          ``pool.clear()``; in-flight workers finish their current
          file and their results are dropped.  The tree is rebuilt
          either way so completed files show their badges.

        Audit note: exceptions from the scientific pipeline are
        logged via ``logger.exception`` with the file path (inside
        :class:`_BatchWorker.run`), so a terminal / log-file audit
        shows *which* file failed and why across any size of batch.

        Parameters
        ----------
        groups
            Groups to iterate.  Empty groups are tolerated
            (``_collect_batch_inputs`` just skips them) but a caller
            that's already filtered them out gets cleaner logs.
        window_title, header_text
            Title bar and initial body text for the progress dialog.
            The body text is overwritten with a per-file label once
            the first worker completes.
        include_group_in_row_label
            ``True`` for study-wide batch — the label becomes
            ``"group / file.csv"`` so the user can track which group
            the currently-displayed file belongs to.  ``False`` for
            single-group batch where the group name is already in
            the header and repeating it per row is noise.
        """
        # Callers gate on ``self._study is not None`` before arriving
        # here; the assert is a defensive anchor for future callers
        # who might forget.
        assert self._study is not None
        study = self._study

        # ── 1. Resolve paths + split into (ready | missing) up-front ──
        ready, missing = _collect_batch_inputs(study, groups)
        # Missing files don't become runnables — merging their errors
        # into the cache up-front means the pool only ever runs useful
        # work, and the progress bar's (k/n) honestly reflects what's
        # left to do on the pipeline side.
        self._results.update(missing)

        n = sum(len(g.files) for g in groups)

        logger.info(
            "StudyPanel: batch analyze — groups=%s, n_files=%d "
            "(ready=%d, missing=%d), study=%s",
            [g.name for g in groups], n, len(ready), n - len(ready),
            study.folder,
        )

        # ── 2. Short-circuit: every file missing → just refresh tree ──
        if not ready:
            self._rebuild_tree()
            return

        # ── 3. Wire up progress dialog + result collector + event loop ──
        progress = QProgressDialog(
            header_text, self.tr("Annulla"), 0, n, self,
        )
        progress.setWindowTitle(window_title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)   # show immediately for small N
        already_done = n - len(ready)
        progress.setValue(already_done)

        # Parallel lookup table for the done-slot.  ``_BatchWorker``
        # echoes back an opaque submission id (reusing its
        # ``file_index`` parameter), and we resolve it here into
        # ``(group, file_entry)`` so the cache key survives the
        # cross-group batch — the worker itself doesn't need to know
        # which group owns its file.
        submissions: list[tuple[Group, FileEntry]] = [
            (group, fe) for group, fe, _abs, _ch in ready
        ]

        loop = QEventLoop(self)
        signals = _BatchWorkerSignals()
        pool = QThreadPool(self)
        pool.setMaxThreadCount(
            min(QThread.idealThreadCount() or 1, _BATCH_MAX_THREADS)
        )

        # Mutable flags packed in a dict so the inner closures can
        # mutate without needing ``nonlocal`` (keeps the diff local).
        state = {'completed': already_done, 'cancelled': False}

        def _on_file_done(submission_id: int, result: FileResult) -> None:
            # Runs on the UI thread (QueuedConnection), so it's safe to
            # touch ``self._results``, the progress dialog, and the
            # tree.  Results that arrive after a cancel are swallowed —
            # the worker was already committed to its file, but the
            # user has moved on.
            if state['cancelled']:
                return
            group, fe = submissions[submission_id]
            key = (group.name, fe.csv_relpath)
            self._results[key] = result
            state['completed'] += 1
            progress.setValue(state['completed'])
            row_label = (
                f"{group.name} / {fe.csv_relpath}"
                if include_group_in_row_label
                else fe.csv_relpath
            )
            progress.setLabelText(
                self.tr("({0}/{1}) {2}").format(
                    state['completed'], n, row_label,
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
        # Compute the per-file fingerprint on the UI thread so we
        # snapshot the config *as submitted*.  If the user opens the
        # group-settings dialog while batch is running and tweaks a
        # knob, the in-flight workers still stamp their results with
        # the old fingerprint — the rebuilt tree will then correctly
        # flag them ● against the new current config.
        for submission_id, (group, _fe, abspath, ch) in enumerate(ready):
            fp = _result_fingerprint(group.config, ch)
            worker = _BatchWorker(
                file_index=submission_id,
                abspath=str(abspath),
                channel=ch,
                config=group.config,
                fingerprint=fp,
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

    def _on_dose_response(self) -> None:
        """Open the dose-response plot dialog over the current study.

        Step 5 of task #84.  Opens even when the study has no fresh-ok
        results — the dialog itself renders an empty-state message in
        that case, which is friendlier than disabling the action
        without explanation.  We still gate the toolbar enablement in
        :meth:`_refresh_actions` so the button looks disabled until
        there's something useful to plot.

        The dialog reads ``self._results`` by reference; it doesn't
        own or mutate the cache, so closing the dialog doesn't lose
        any batch-analysis state.
        """
        if self._study is None:
            return
        dlg = DoseResponseDialog(
            study=self._study,
            results_cache=self._results,
            parent=self,
        )
        dlg.exec()

    def _on_export_cdisc(self) -> None:
        """Export a CDISC SEND package for the whole study (step 6 / #89).

        Flow:

        1. Collect the fresh-ok ``(group, file)`` pairs via
           :func:`_collect_export_inputs`.  If the result is empty we
           show a "nothing to export" dialog and bail — no silent no-op.
        2. Ask for a CDISC ``STUDYID`` (prefilled from
           :func:`_suggest_study_id`) and an output folder.  The
           folder can be anywhere on disk; we don't force it under
           the study folder because regulatory submissions usually
           live in a separate tree.
        3. Launch :class:`_ExportWorker` on the shared ``QThreadPool``
           with a modal :class:`QProgressDialog`.  The dialog's Cancel
           button calls :meth:`_ExportWorker.cancel`; already-running
           ``analyze_single_file`` finishes but subsequent files are
           skipped and the worker terminates with a ``failed`` signal
           carrying the sentinel ``"__cancelled__"``.
        4. On success, show a summary dialog with the backend used
           (``pyreadstat`` / ``xport`` / ``csv_fallback``) and the
           output folder.  The user can then open the folder in
           Finder / Explorer and ship the ``.xpt`` package to
           Pinnacle 21.

        Why we re-run the pipeline at export time (instead of caching
        the full result): see module-level rationale above
        :func:`_enrich_file_info_for_export`.  The short version:
        caching ~50 full result dicts would blow the memory floor
        :class:`FileResult` was introduced to protect; re-running
        costs a minute and the user expects a wait for "make me a
        regulatory package" anyway.
        """
        if self._study is None:
            return

        # ── 1. Harvest fresh-ok pairs ─────────────────────────────────
        pairs = _collect_export_inputs(self._study, self._results)
        if not pairs:
            QMessageBox.information(
                self, self.tr("Nessun risultato esportabile"),
                self.tr(
                    "Lo studio non ha ancora risultati freschi (✓) da "
                    "esportare.  Esegui “Analizza gruppo” su almeno un "
                    "gruppo prima di creare il pacchetto CDISC."
                ),
            )
            return

        # ── 2. Ask the user for STUDYID + output folder ───────────────
        suggested = _suggest_study_id(self._study.name)
        study_id, ok = QInputDialog.getText(
            self, self.tr("Export CDISC SEND"),
            self.tr(
                "STUDYID (identificativo CDISC — massimo 32 caratteri "
                "A-Z / 0-9 / -):"
            ),
            text=suggested,
        )
        if not ok:
            return
        study_id = _suggest_study_id(study_id)   # normalise user input

        out_folder = QFileDialog.getExistingDirectory(
            self,
            self.tr("Cartella di destinazione per il pacchetto CDISC"),
        )
        if not out_folder:
            return

        # ── 3. Progress dialog + worker ───────────────────────────────
        n = len(pairs)
        progress = QProgressDialog(
            self.tr("Preparazione export CDISC..."),
            self.tr("Annulla"),
            0, n, self,
        )
        progress.setWindowTitle(self.tr("Export CDISC SEND"))
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        signals = _ExportWorkerSignals()
        worker = _ExportWorker(
            study=self._study,
            pairs=pairs,
            output_dir=out_folder,
            study_id=study_id,
            study_title='CiPA In Vitro Cardiac Safety - hiPSC-CM Field Potential',
            sponsor='',
            signals=signals,
        )

        loop = QEventLoop(self)
        outcome = {'result': None, 'error': None}

        def _on_progress(done: int, total: int, name: str) -> None:
            # Runs on the UI thread (QueuedConnection).
            progress.setMaximum(total)
            progress.setValue(done)
            if name:
                progress.setLabelText(
                    self.tr("({0}/{1}) {2}").format(done, total, name)
                )

        def _on_finished(result: object) -> None:
            outcome['result'] = result
            loop.quit()

        def _on_failed(msg: str) -> None:
            outcome['error'] = msg
            loop.quit()

        signals.progress.connect(_on_progress, Qt.QueuedConnection)
        signals.finished.connect(_on_finished, Qt.QueuedConnection)
        signals.failed.connect(_on_failed, Qt.QueuedConnection)

        progress.canceled.connect(worker.cancel)

        logger.info(
            "StudyPanel: CDISC export started — study=%r, study_id=%r, "
            "n_pairs=%d, output_dir=%s",
            self._study.name, study_id, n, out_folder,
        )

        pool = QThreadPool(self)
        pool.start(worker)

        loop.exec()
        # Worker has either emitted finished or failed; either way the
        # pool has no more work.  We still waitForDone so any log
        # flushes from the worker settle before we pop the result dialog.
        pool.waitForDone()
        progress.close()

        # ── 4. Report outcome ─────────────────────────────────────────
        if outcome['error'] is not None:
            err = outcome['error']
            if err == "__cancelled__":
                logger.info(
                    "StudyPanel: CDISC export cancelled by user — "
                    "study=%r", self._study.name,
                )
                QMessageBox.information(
                    self, self.tr("Export annullato"),
                    self.tr(
                        "Export CDISC annullato.  Nessun file "
                        "garantito completo nella cartella scelta."
                    ),
                )
                return
            logger.error(
                "StudyPanel: CDISC export failed — study=%r, error=%s",
                self._study.name, err,
            )
            QMessageBox.critical(
                self, self.tr("Export CDISC fallito"),
                self.tr(
                    "Impossibile completare l'export CDISC.\n\n"
                    "Dettaglio:\n{0}"
                ).format(err),
            )
            return

        result = outcome['result']
        if not isinstance(result, dict):
            # Belt-and-braces: the worker must emit a dict on
            # ``finished`` by contract, but we refuse to deref if
            # something went wrong upstream.
            logger.error(
                "StudyPanel: CDISC export returned non-dict result: %r",
                type(result).__name__,
            )
            QMessageBox.critical(
                self, self.tr("Export CDISC fallito"),
                self.tr("Risultato dell'export in formato inatteso."),
            )
            return

        backend = result.get('backend_used', '?')
        files = result.get('files', []) or []
        n_files = len(files)
        logger.info(
            "StudyPanel: CDISC export OK — study=%r, backend=%s, "
            "files=%d, output_dir=%s",
            self._study.name, backend, n_files, out_folder,
        )

        # Backend-aware messaging: CSV fallback is NOT regulatory-grade
        # and the user should know up-front.
        if backend == 'csv_fallback':
            QMessageBox.warning(
                self, self.tr("Export CDISC completato (fallback CSV)"),
                self.tr(
                    "Il pacchetto è stato creato in {0} ma usa il "
                    "fallback CSV: né pyreadstat né xport sono "
                    "installati, quindi i file .xpt sono in realtà "
                    "CSV e NON sono conformi alla sottomissione "
                    "regolatoria.\n\n"
                    "Per produrre .xpt veri installa pyreadstat "
                    "(consigliato) o xport."
                ).format(out_folder),
            )
        else:
            QMessageBox.information(
                self, self.tr("Export CDISC completato"),
                self.tr(
                    "Pacchetto CDISC SEND generato con backend "
                    "“{0}”.\n\n"
                    "Cartella: {1}\n"
                    "File scritti: {2}"
                ).format(backend, out_folder, n_files),
            )

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
        fe = group.files[file_idx]
        abspath = resolve_file_path(self._study, fe)
        if not abspath.is_file():
            QMessageBox.warning(
                self, self.tr("File mancante"),
                self.tr(
                    "Il file non esiste più sul disco:\n{0}"
                ).format(abspath),
            )
            return
        self.file_activated.emit(str(abspath), group_name, fe.csv_relpath)

    # ── Cache write-back (GH #6 Fase 3) ──────────────────────────────

    def update_entry(
        self,
        group_name: str,
        csv_relpath: str,
        result_dict: dict,
        channel: str,
        config,   # AnalysisConfig — not imported at module scope on purpose
    ) -> None:
        """Replace the cached :class:`FileResult` for one file.

        Invoked by :class:`MainWindow` after a successful Ricalcola
        (GH #6 Fase 3).  Builds a fresh :class:`FileResult` from the
        pipeline output dict using the exact same extraction pattern
        as :class:`_BatchWorker.run` (keeps the two paths in lockstep
        so the Studi tree and the dose-response dialog see the same
        numbers after a manual edit).

        The new fingerprint is computed from the *actual* config used
        by the recompute — which, post GH #6 Fase 2, is
        ``group.config`` when the file belongs to a Group.  This is
        what makes the cached entry match the current batch
        fingerprint: the staleness gate stops flagging the row as ●,
        and aggregates / dose-response pick up the user's edits
        immediately.

        Parameters
        ----------
        group_name : str
            Name of the :class:`Group` the file belongs to.  Part of
            the cache key.
        csv_relpath : str
            POSIX-style path relative to the study folder.  Part of
            the cache key.
        result_dict : dict
            Pipeline result dict (same shape as
            :func:`analyze_single_file` output).  Must contain
            ``file_info``, ``beat_indices``, ``beat_indices_fpd``,
            and ``summary``.
        channel : str
            Channel that was analysed (``"EL1"``, ``"EL2"``, ...).
            Needed for the fingerprint — ``_result_fingerprint``
            hashes ``(config, channel)``.
        config : AnalysisConfig
            Configuration that produced ``result_dict``.  Typically
            ``group.config`` for group-scoped recomputes.

        Notes
        -----
        No-op if ``self._study is None`` — can't sensibly write into
        a closed study.  Silently ignores unknown ``group_name`` or
        ``csv_relpath`` (caller is responsible for passing keys that
        exist in the current study; we log at debug level but don't
        raise, since a stale MainWindow state is survivable).
        """
        if self._study is None:
            return
        group = find_group(self._study, group_name)
        if group is None:
            logger.debug(
                "StudyPanel.update_entry: unknown group %r — ignored",
                group_name,
            )
            return
        if not any(f.csv_relpath == csv_relpath for f in group.files):
            logger.debug(
                "StudyPanel.update_entry: unknown file %r in group %r — ignored",
                csv_relpath, group_name,
            )
            return

        analyzed = (
            (result_dict.get('file_info') or {}).get('analyzed_channel')
            or ''
        )
        n_total = _len_or_zero(result_dict.get('beat_indices'))
        fpd_len = _len_or_zero(result_dict.get('beat_indices_fpd'))
        n_included = fpd_len if fpd_len > 0 else n_total
        fpd, fpdc, bpm, stv = _extract_summary_metrics(
            result_dict.get('summary'),
        )

        fr = FileResult(
            status='ok',
            channel_analyzed=str(analyzed),
            n_included=n_included,
            n_total=n_total,
            fingerprint=_result_fingerprint(config, channel),
            fpd_ms=fpd,
            fpdc_ms=fpdc,
            bpm=bpm,
            stv_ms=stv,
        )
        self._results[(group_name, csv_relpath)] = fr
        # Rebuild so the tree's badge / tooltip / metric columns
        # reflect the new result straight away — otherwise the user
        # has to trigger some other refresh (new batch, reopen study)
        # to see the edit land.
        self._rebuild_tree()

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
            # Collect cached results + parallel staleness flags vs the
            # *current* group.config so the aggregate can distinguish
            # "all fresh ✓" from "mixed with stale ●".  One pass here
            # keeps the file-row loop from re-computing staleness per
            # row.
            group_results: list[FileResult | None] = []
            group_stale: list[bool] = []
            for f in group.files:
                r = self._results.get((group.name, f.csv_relpath))
                group_results.append(r)
                group_stale.append(_is_stale(r, group.config, f.channel))
            agg = _aggregate_status(group_results, group_stale)
            if agg:
                details = f"{details}  ·  {agg}"
            # Step 4 — per-group scalar metrics (FPDc/BPM/STV mean±SD),
            # appended only when there's at least one fresh-ok result
            # to average over.  Dose-response readout at a glance,
            # which is the whole point of grouping files by dose.
            metrics_agg = _aggregate_group_metrics(group_results, group_stale)
            metrics_line = _format_group_metrics_line(metrics_agg)
            if metrics_line:
                details = f"{details}  ·  {metrics_line}"
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
                res = group_results[idx]
                is_stale = group_stale[idx]
                badge = _badge_for(res, stale=is_stale)
                base_name = Path(entry.csv_relpath).name
                f_item.setText(
                    0,
                    f"{badge} {base_name}" if badge else base_name,
                )
                f_item.setText(1, f_details)
                tip = _tooltip_for(res, stale=is_stale)
                if tip:
                    f_item.setToolTip(0, tip)
                    f_item.setToolTip(1, tip)
                # Colour cue for errors only — keep success on the
                # default theme palette so we don't fight dark mode.
                # Stale ok-results keep the default colour too: the ●
                # badge already carries the signal, and tinting would
                # over-index on "something is wrong".
                if res is not None and res.status == 'error':
                    brush = QBrush(QColor("#b00020"))   # matte red
                    f_item.setForeground(0, brush)
                    f_item.setForeground(1, brush)

        # Stretch the first column to fit the deepest row, then let the
        # details column take the remaining width.
        self._tree.resizeColumnToContents(0)

        # Dose-response enablement depends on whether any fresh-ok
        # result is in cache — that state only changes on tree rebuild
        # (after batch analyze) or on study-level swaps, both of which
        # already call into here.  Cheaper than wiring a dedicated
        # "cache changed" signal.
        self._refresh_actions()

    def _any_fresh_ok_result(self) -> bool:
        """True iff at least one cached result is non-stale and status='ok'.

        Used by :meth:`_refresh_actions` to gate the Dose-response button:
        the plot has nothing to show until batch-analyze has landed at
        least one fresh result, and we'd rather disable the action than
        pop a modal dialog only to show "no data" inside it.

        ``O(total files)`` — studies are small (≤ ~50 files), so walking
        every (group, file) pair is effectively free.  A more elaborate
        incremental flag was considered and rejected as premature.
        """
        if self._study is None:
            return False
        for group in self._study.groups:
            for f in group.files:
                r = self._results.get((group.name, f.csv_relpath))
                if r is None or r.status != 'ok':
                    continue
                if _is_stale(r, group.config, f.channel):
                    continue
                return True
        return False

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
        self._act_add_folder.setEnabled(can_edit_group)
        self._act_group_settings.setEnabled(can_edit_group)
        self._act_remove.setEnabled(can_edit_group)
        # Batch analyze is gated on the same "a group (or a file within
        # a group) is selected" rule — we walk up to the group in
        # ``_selected_group`` so either kind is a valid anchor.
        self._act_analyze_group.setEnabled(can_edit_group)
        # "Analizza studio" is selection-agnostic — it iterates over
        # every group in the study — so we gate it on "study loaded
        # AND at least one group has at least one file".  The second
        # half catches the edge case where the user creates a study,
        # adds empty groups, then clicks Analizza studio: we'd rather
        # grey the button out than pop an empty-state dialog.
        has_any_file = has_study and any(
            bool(g.files) for g in self._study.groups
        )
        self._act_analyze_study.setEnabled(has_any_file)
        # Dose-response is a *study-wide* action — it doesn't need a
        # selection, only a loaded study with at least one fresh-ok
        # result somewhere.  Walking the whole cache once per selection
        # change is O(N files) which is cheap (studies are small); the
        # payoff is the button grays out until there's data to plot.
        has_fresh_ok = has_study and self._any_fresh_ok_result()
        self._act_dose_response.setEnabled(has_fresh_ok)
        # CDISC export follows the same gating as dose-response: we
        # only let the user click it once at least one fresh-ok
        # result exists.  Export runs the scientific pipeline again
        # under the hood (see :class:`_ExportWorker`), so having
        # anything staler than ``ok`` would either be wasted work
        # (errors) or confused by re-run-then-mismatch logic (stale).
        self._act_export_cdisc.setEnabled(has_fresh_ok)


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


def _enumerate_csvs_under(folder: Path) -> list[Path]:
    """Recursively list *visible* ``*.csv`` files under ``folder``.

    Module-level and pure so it can be unit-tested without spinning up
    Qt: :meth:`StudyPanel._on_add_folder` delegates here for the
    filesystem scan.

    Returns
    -------
    list[Path]
        Sorted by POSIX path for deterministic tree order.  Hidden
        files (any component starting with ``'.'``) are excluded so
        macOS ``.DS_Store`` siblings and the user's own dotfiles
        don't pollute the study.  Symlinks are followed by ``rglob``
        — that's Python's default and matches the user's mental
        model of "it's inside the folder, add it".

    Notes
    -----
    Returning the paths as ``Path`` objects (not strings) lets the
    caller stringify them only when handing them to Qt / study API,
    and keeps the tests filesystem-agnostic.
    """
    try:
        all_csvs = list(Path(folder).rglob('*.csv'))
    except OSError:
        # Unreadable folder (permissions) — let the caller show an
        # empty-result dialog rather than crashing the UI thread.
        return []
    visible = [
        p for p in all_csvs
        # Skip anything whose path contains a hidden component.  That
        # also filters out files inside hidden subtrees (``.cache/`` …).
        if not any(part.startswith('.') for part in p.parts)
    ]
    return sorted(visible)


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


def _to_float_or_nan(x) -> float:
    """Coerce ``x`` to float, returning NaN for ``None`` / non-numeric / NaN.

    The scientific pipeline mostly stores scalars as numpy floats.
    :class:`FileResult` stores native Python floats so the dataclass
    stays json-dumpable if we ever persist it.  This helper is the
    single coercion point — keeps the worker clean of try/except
    scatter.  Handles:

    * ``None`` → NaN (missing field).
    * strings → tries ``float(x)``; falls back to NaN on parse error.
    * numpy scalars → accepted via ``float(x)``.
    * already-NaN → NaN (no spurious conversion).
    """
    if x is None:
        return float('nan')
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float('nan')
    # math.isnan rejects complex / ambiguous; our try/except above
    # already filtered those, so this is a pure finite-ness check.
    if math.isnan(f):
        return float('nan')
    return f


def _extract_summary_metrics(summary: dict | None) -> tuple[float, float, float, float]:
    """Pull ``(fpd_ms, fpdc_ms, bpm, stv_ms)`` out of a pipeline summary.

    The scientific-core ``summary`` dict is rich (~30 keys); we only
    surface the four scalars the study-panel UI shows at a glance.
    For FPD / FPDc we prefer **median** over mean because the per-beat
    distribution is heavy-tailed on marginal recordings (a handful of
    ectopic beats drag the mean more than clinicians expect).  Median
    is also what gets reported in papers.

    Units:

    * ``fpd_*`` / ``stv_ms`` are in *milliseconds* in the UI cache.
      The scientific core stores ``fpd_values`` in *seconds* (see
      ``parameters.py:456``) but also writes ``fpd_median`` in ms
      via the generic parameter loop (``*_median`` lives alongside
      ``*_mean``).  We read the ms one directly.
    * ``bpm`` is beats-per-minute (``summary['bpm_mean']``).

    Any missing / NaN source becomes NaN in the output — downstream
    aggregate / tooltip code already tolerates NaN.
    """
    if not summary:
        return (float('nan'),) * 4

    # FPD/FPDc are ms in the summary (see parameters.py generic loop
    # over ``['fpd_ms', 'fpdc_ms', ...]`` filling ``_median`` keys).
    fpd = _to_float_or_nan(summary.get('fpd_ms_median'))
    fpdc = _to_float_or_nan(summary.get('fpdc_ms_median'))
    # Fallback to mean when median missing on older result schemas.
    if math.isnan(fpd):
        fpd = _to_float_or_nan(summary.get('fpd_ms_mean'))
    if math.isnan(fpdc):
        fpdc = _to_float_or_nan(summary.get('fpdc_ms_mean'))

    bpm = _to_float_or_nan(summary.get('bpm_mean'))
    stv = _to_float_or_nan(summary.get('stv_ms'))
    return fpd, fpdc, bpm, stv


def _result_fingerprint(config, channel: str) -> str:
    """Short deterministic hash of the inputs that produced a result.

    The fingerprint is written into :class:`FileResult.fingerprint` at
    analysis time and re-computed on every tree repaint.  When the two
    diverge, the row gets the ``●`` stale badge: the cache stays
    visible (we don't throw work away), but the user is told *which*
    files need re-analysis.

    Inputs
        * ``config`` — the group's :class:`AnalysisConfig` (dataclass
          or plain object; we fall back to ``repr`` for anything we
          can't json-encode, which is good enough for a change detector
          — it doesn't have to be canonical, only deterministic).
        * ``channel`` — the per-file channel override ("auto" or
          ``ch1`` / ``ch2`` / …).  Changing it re-picks a different
          signal, so it's part of the fingerprint even though it
          lives on the ``FileEntry``, not on the config.

    Why SHA-1 truncated to 12 hex chars
        Fingerprints are never written to disk (they live inside the
        in-memory ``_results`` cache).  SHA-1 is not a security claim
        here — any stable hash would do; SHA-1 just ships with the
        stdlib and has the right output shape.  We truncate because
        the full 40-char digest has no extra collision resistance for
        our use case (a handful of configs per study).
    """
    if is_dataclass(config):
        payload = asdict(config)
    elif hasattr(config, '__dict__'):
        payload = dict(config.__dict__)
    else:
        payload = {'repr': repr(config)}

    payload['__channel__'] = channel or ''

    try:
        blob = json.dumps(payload, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        # Belt-and-braces: if something deep in the config isn't
        # json-encodable even with ``default=repr``, fall back to a
        # sorted repr.  Still deterministic for equal inputs.
        blob = repr(sorted(payload.items()))

    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]


def _is_stale(
    res: FileResult | None,
    config,
    channel: str,
) -> bool:
    """Is the cached ``res`` out-of-date relative to the current inputs?

    Rules:

    * ``None`` (never analysed) → not stale (the row has no badge;
      an absence of result is not a "stale" result).
    * Empty ``res.fingerprint`` → not stale.  These are legacy cache
      entries from before step 3c; flagging them stale on first load
      would nag the user about perfectly good results.  They get a
      fingerprint on the next analyse.
    * Otherwise stale iff the stored fingerprint differs from the
      current ``_result_fingerprint(config, channel)``.
    """
    if res is None:
        return False
    if not res.fingerprint:
        return False
    return res.fingerprint != _result_fingerprint(config, channel)


def _badge_for(res: FileResult | None, *, stale: bool = False) -> str:
    """Return the column-0 prefix glyph for a file row.

    * ``None`` (never analysed) → empty string, no prefix.
    * ``stale=True`` → ``'●'`` (result exists but inputs changed).
      Takes precedence over ok/error so the user sees "this is out
      of date" before "this succeeded last time".
    * ``ok`` → ``'✓'``.
    * ``error`` → ``'✗'``.

    Kept as a module-level pure function so it's unit-testable without
    spinning up Qt — same rationale as :func:`_format_group_details`.
    """
    if res is None:
        return ""
    if stale:
        return "●"
    if res.status == 'ok':
        return "✓"
    if res.status == 'error':
        return "✗"
    return ""


def _tooltip_for(res: FileResult | None, *, stale: bool = False) -> str:
    """Tooltip text for a file row in the tree.

    For successful runs we surface the channel chosen by
    ``select_best_channel`` and the included-beats count, which is
    what the user cares about before committing to a double-click
    deep-dive.  For errors we dump the full error message (what /
    where — the user can then re-open the CSV and iterate).

    When ``stale=True`` we prepend a short marker explaining *why*
    the ● badge is showing — otherwise the user has to remember what
    ● means vs ✓ / ✗.
    """
    if res is None:
        return ""
    if res.status == 'error':
        body = f"Errore: {res.error}" if res.error else "Errore"
    else:
        # status == 'ok'
        parts: list[str] = []
        if res.channel_analyzed:
            parts.append(f"Canale: {res.channel_analyzed.upper()}")
        if res.n_total:
            if res.n_included != res.n_total:
                parts.append(f"{res.n_included}/{res.n_total} battiti inclusi")
            else:
                parts.append(f"{res.n_total} battiti")
        # Step 4 — surface the scalar metrics alongside the beat
        # counts so the user can compare FPDc/BPM across files in
        # the same group without clicking each one.
        metrics = _format_file_metrics(res)
        if metrics:
            parts.append(metrics)
        body = "  ·  ".join(parts)
    if stale:
        prefix = "Risultato non aggiornato (config cambiata)"
        return f"{prefix}  ·  {body}" if body else prefix
    return body


def _aggregate_status(
    results: list[FileResult | None],
    stale: list[bool] | None = None,
) -> str:
    """Summary string for a group row — ``"2✓ 1✗"`` / ``"all ✓"`` / ``""``.

    The PoC convention:

    * no results at all → empty string (don't clutter the row before
      the user has run batch).
    * all ok + all fresh → compact ``"✓ {n}/{n}"``.
    * mixed → ``"{ok}✓ {st}● {err}✗"`` with each term omitted when
      the count is zero.  ``st`` is the number of stale-but-otherwise
      countable results (a stale error still counts as ✗, not ●; a
      stale ok counts as ● and is *not* double-counted in ✓).

    ``stale`` is optional and parallel to ``results`` when given.
    When ``None`` (pre-3c callers) the method behaves as before.
    """
    if not results:
        return ""
    if stale is None:
        stale = [False] * len(results)
    n_ok_fresh = 0
    n_stale = 0
    n_err = 0
    for r, is_stale_flag in zip(results, stale):
        if r is None:
            continue
        if r.status == 'error':
            n_err += 1
        elif is_stale_flag:
            n_stale += 1
        elif r.status == 'ok':
            n_ok_fresh += 1
    if n_ok_fresh == 0 and n_stale == 0 and n_err == 0:
        return ""
    if n_err == 0 and n_stale == 0 and n_ok_fresh == len(results):
        return f"✓ {n_ok_fresh}/{len(results)}"
    parts: list[str] = []
    if n_ok_fresh:
        parts.append(f"{n_ok_fresh}✓")
    if n_stale:
        parts.append(f"{n_stale}●")
    if n_err:
        parts.append(f"{n_err}✗")
    return " ".join(parts)


def _mean_sd_n(values: list[float]) -> tuple[float, float, int]:
    """Return ``(mean, sd, n)`` over the finite entries of ``values``.

    NaNs are *dropped* before computing stats: a file whose pipeline
    couldn't produce a given metric (e.g. no valid repol → NaN FPD)
    simply doesn't contribute to the aggregate instead of polluting
    it.  ``n`` reflects the finite count actually used, not the
    input length — so the caller can tell "aggregate of 3/5 files"
    from "aggregate of 5/5".

    Rules:

    * empty / all-NaN input → ``(NaN, NaN, 0)``.
    * n=1 → ``(value, 0.0, 1)`` — can't compute SD from one point;
      returning 0 is the common-sense read "no spread observed yet".
    * n≥2 → sample SD (ddof=1), matching what scientists expect when
      they see "mean ± SD".
    """
    finite = [v for v in values if not (v is None or math.isnan(v))]
    n = len(finite)
    if n == 0:
        return float('nan'), float('nan'), 0
    mean = sum(finite) / n
    if n == 1:
        return mean, 0.0, 1
    # Sample SD (ddof=1) — matches scientific convention.
    var = sum((v - mean) ** 2 for v in finite) / (n - 1)
    return mean, math.sqrt(var), n


def _aggregate_group_metrics(
    results: list[FileResult | None],
    stale: list[bool] | None = None,
) -> dict:
    """Compute ``mean ± SD`` per metric over fresh-ok results in a group.

    Contract for "fresh-ok":

    * ``r is not None``
    * ``r.status == 'ok'``
    * ``stale[i] is False`` (or ``stale`` is ``None`` → all fresh)

    Only fresh-ok results feed the aggregate.  Stale and error rows
    are deliberately excluded — if the user edited the config,
    presenting a mean over old numbers would be misleading; once they
    re-run, the fresh results will repopulate the aggregate.

    Returns a flat dict shaped for the formatter:

        {
            'n': <int, count of fresh-ok files contributing>,
            'n_total': <int, total files in the group>,
            'fpd_ms':  (mean, sd, n_finite),
            'fpdc_ms': (mean, sd, n_finite),
            'bpm':     (mean, sd, n_finite),
            'stv_ms':  (mean, sd, n_finite),
        }

    ``n_finite`` per metric may be lower than ``n`` when some files
    had NaN for that specific metric (e.g. chronic baseline with no
    valid FPD but still a measurable rate).  That keeps the UI
    honest about how much each metric is actually averaging over.
    """
    n_total = len(results)
    if stale is None:
        stale = [False] * n_total

    fpd_vals: list[float] = []
    fpdc_vals: list[float] = []
    bpm_vals: list[float] = []
    stv_vals: list[float] = []
    n_fresh_ok = 0
    for r, s in zip(results, stale):
        if r is None or r.status != 'ok' or s:
            continue
        n_fresh_ok += 1
        fpd_vals.append(r.fpd_ms)
        fpdc_vals.append(r.fpdc_ms)
        bpm_vals.append(r.bpm)
        stv_vals.append(r.stv_ms)

    return {
        'n': n_fresh_ok,
        'n_total': n_total,
        'fpd_ms': _mean_sd_n(fpd_vals),
        'fpdc_ms': _mean_sd_n(fpdc_vals),
        'bpm': _mean_sd_n(bpm_vals),
        'stv_ms': _mean_sd_n(stv_vals),
    }


def _fmt_mean_sd(mean_sd_n: tuple[float, float, int], *, unit: str = '', decimals: int = 1) -> str:
    """Format a ``(mean, sd, n)`` tuple as ``"385 ± 12 ms"``.

    Rules:

    * ``n == 0`` → empty string (caller can skip the whole term).
    * ``n == 1`` → ``"385 ms (n=1)"`` — no ``± 0`` for a single
      sample, since it's misleading.
    * ``n >= 2`` → ``"385 ± 12 ms"``.

    Integer formatting when ``decimals=0``, else fixed decimals.
    """
    mean, sd, n = mean_sd_n
    if n == 0 or math.isnan(mean):
        return ""
    fmt = f"{{:.{decimals}f}}"
    u = f" {unit}" if unit else ""
    if n == 1:
        return f"{fmt.format(mean)}{u} (n=1)"
    return f"{fmt.format(mean)} ± {fmt.format(sd)}{u}"


def _format_group_metrics_line(agg: dict) -> str:
    """One-line ``"FPDc 385±12 ms · BPM 58±3 · STV 2.1 ms · n=4/5"`` summary.

    Only terms with ``n >= 1`` are included.  The ``n=K/N`` suffix is
    always appended when ``agg['n'] > 0`` so the user knows how many
    files contributed — stale/error rows are not in the aggregate
    but are in the denominator (``n_total``).

    Returns empty string when the group has no fresh-ok aggregates,
    so the caller can skip appending anything to the tree row.
    """
    if agg.get('n', 0) == 0:
        return ""
    parts: list[str] = []
    # FPDc first — clinically the primary endpoint.
    fpdc = _fmt_mean_sd(agg['fpdc_ms'], unit='ms', decimals=0)
    if fpdc:
        parts.append(f"FPDc {fpdc}")
    fpd = _fmt_mean_sd(agg['fpd_ms'], unit='ms', decimals=0)
    if fpd and not fpdc:
        # Fall back to raw FPD only when FPDc is unavailable —
        # otherwise two near-duplicate numbers clutter the row.
        parts.append(f"FPD {fpd}")
    bpm = _fmt_mean_sd(agg['bpm'], unit='BPM', decimals=0)
    if bpm:
        parts.append(bpm)
    stv = _fmt_mean_sd(agg['stv_ms'], unit='ms', decimals=1)
    if stv:
        parts.append(f"STV {stv}")
    parts.append(f"n={agg['n']}/{agg['n_total']}")
    return "  ·  ".join(parts)


def _format_file_metrics(res: FileResult | None) -> str:
    """Single-file metrics block for the tooltip — ``"FPDc: 385 ms · 58 BPM"``.

    Returns empty string when ``res`` is None / error / all-NaN, so
    the tooltip call site can skip appending the separator.  Used
    alongside the existing channel+beat-count tooltip text, not as
    a replacement — both pieces of info help the user decide whether
    a deep-dive is warranted.
    """
    if res is None or res.status != 'ok':
        return ""
    parts: list[str] = []
    if not math.isnan(res.fpdc_ms):
        parts.append(f"FPDc: {res.fpdc_ms:.0f} ms")
    elif not math.isnan(res.fpd_ms):
        parts.append(f"FPD: {res.fpd_ms:.0f} ms")
    if not math.isnan(res.bpm):
        parts.append(f"{res.bpm:.0f} BPM")
    if not math.isnan(res.stv_ms):
        parts.append(f"STV: {res.stv_ms:.1f} ms")
    return "  ·  ".join(parts)


def _metric_meta(key: str) -> tuple[str, str, int]:
    """Return ``(label, unit, decimals)`` for a dose-response metric key.

    Raises ``KeyError`` on unknown keys — callers are expected to pick
    from :data:`_DOSE_RESPONSE_METRICS` and should fail loudly if they
    pass a typo.  Used by the dialog to build the axis label and format
    tooltips consistently with step-4 tree rendering.
    """
    for k, label, unit, decimals in _DOSE_RESPONSE_METRICS:
        if k == key:
            return label, unit, decimals
    raise KeyError(f"unknown metric key: {key!r}")


def _collect_dose_response_points(
    study: Study,
    results_cache: dict,
    *,
    metric: str = 'fpdc_ms',
) -> list[DoseResponsePoint]:
    """Build one :class:`DoseResponsePoint` per group for a given metric.

    Iterates :attr:`Study.groups`, computes per-metric aggregate with
    :func:`_aggregate_group_metrics` (reusing the same staleness logic
    as the tree rows so the plot is never inconsistent with what the
    user sees in the panel), and drops groups where the metric has
    ``n == 0`` finite values — those groups simply don't appear in the
    plot, rather than showing as a visually misleading "point at NaN".

    Parameters
    ----------
    study : Study
        The study being plotted.  ``study.groups`` is walked in order;
        the sort by dose happens downstream in
        :func:`_assemble_dose_response_series`.
    results_cache : dict
        ``StudyPanel._results`` — maps ``(group_name, csv_relpath)`` to
        :class:`FileResult`.  Missing keys are handled as "never
        analysed" and simply don't contribute to the aggregate.
    metric : str
        One of the keys in :data:`_DOSE_RESPONSE_METRICS`.  Validated
        against the whitelist to prevent silent no-ops on typos.

    Returns
    -------
    list[DoseResponsePoint]
        Unordered.  Baseline groups (``dose_uM is None``) are included
        as regular entries — the plot code filters them into a reference
        line separately.
    """
    _metric_meta(metric)   # validate early; raises KeyError on typo

    points: list[DoseResponsePoint] = []
    for group in study.groups:
        group_results: list[FileResult | None] = []
        group_stale: list[bool] = []
        for f in group.files:
            r = results_cache.get((group.name, f.csv_relpath))
            group_results.append(r)
            group_stale.append(_is_stale(r, group.config, f.channel))
        agg = _aggregate_group_metrics(group_results, group_stale)
        mean, sd, n_finite = agg.get(metric, (float('nan'), float('nan'), 0))
        if n_finite < 1 or math.isnan(mean):
            continue
        points.append(DoseResponsePoint(
            group_name=group.name,
            drug=group.drug or '',
            dose_uM=group.dose_uM,
            mean=float(mean),
            sd=float(sd) if not math.isnan(sd) else 0.0,
            n=int(n_finite),
        ))
    return points


def _assemble_dose_response_series(
    points: list[DoseResponsePoint],
) -> dict:
    """Split points into baselines + per-drug dose curves, ready to plot.

    The dose-response figure has two visual layers:

    1. **Dose curves** — one per distinct drug, with x=dose_µM on a
       log scale when all doses are strictly positive.  Points are
       sorted by dose so the polyline doesn't jump back and forth.
    2. **Baselines** — a dashed horizontal reference line per drug
       (``dose_uM is None`` groups), plus an "overall" baseline when
       the baseline group has an empty ``drug`` (the typical "vehicle
       only" control that precedes *any* dose series).

    Returns
    -------
    dict with two keys:

    * ``'doses'`` : ``dict[str, list[DoseResponsePoint]]``
        Per-drug curves, each sorted by ``dose_uM`` ascending.
        Drugs with no dose points (baseline-only) are omitted.
    * ``'baselines'`` : ``list[DoseResponsePoint]``
        Baseline groups (``dose_uM is None``).  Order of insertion
        (== order in ``Study.groups``) is preserved, so the user-chosen
        study layout drives the legend order.

    Both sides of the split are derived from the same input — a point
    is *either* a dose point *or* a baseline (by its ``dose_uM``), so
    no point is double-counted between the two categories.
    """
    baselines: list[DoseResponsePoint] = []
    doses: dict[str, list[DoseResponsePoint]] = {}
    for p in points:
        if p.dose_uM is None:
            baselines.append(p)
            continue
        doses.setdefault(p.drug, []).append(p)
    for drug, curve in doses.items():
        # Sort by dose ascending so pyqtgraph can draw a polyline that
        # doesn't zig-zag.  Tie-break by group_name for stable ordering
        # when two groups share a dose (rare but legal).
        curve.sort(key=lambda pt: (pt.dose_uM, pt.group_name))
    return {'doses': doses, 'baselines': baselines}


def _can_plot_log_x(points: list[DoseResponsePoint]) -> bool:
    """Return True iff every dose point has ``dose_uM > 0``.

    Log-scale x axes can't render x ≤ 0, so a zero-dose group (which
    should really have been flagged baseline with ``dose_uM=None``,
    but users sometimes write "0" literally) forces the caller to
    fall back to linear scale.  Empty input → False (there's nothing
    to plot, so log doesn't apply).
    """
    dose_points = [p for p in points if p.dose_uM is not None]
    if not dose_points:
        return False
    return all(p.dose_uM > 0.0 for p in dose_points)


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


# ═══════════════════════════════════════════════════════════════════════
#  Step 6 — CDISC SEND export per study (task #89)
# ═══════════════════════════════════════════════════════════════════════
#
#  The step-5 dose-response dialog renders cached scalars; step 6 hands
#  the *same* fresh-ok universe to :func:`cdisc_export.export_send_package`
#  so the user can drop a SEND submission into Pinnacle 21.
#
#  FileResult only caches scalars by documented design (memory floor):
#  ``export_send_package`` needs the full pipeline dict per file
#  (``file_info`` / ``summary`` / beat arrays).  We therefore re-run
#  :func:`analyze_single_file` at export time on every fresh-ok
#  (group, file) pair — same contract as a double-click deep-dive, but
#  repeated in a background worker with a progress dialog so the UI
#  stays responsive.  The re-run is pure: the returned dict is a fresh
#  object built from CSV + AnalysisConfig, and no module-level mutable
#  state is touched.  Costs: ~1-3 s per file × ≤ ~50 files per study =
#  at most a minute, which matches the user's export mental model
#  ("make me the regulatory package, I'll grab coffee").
#
#  Why "fresh-ok only":
#
#  * Stale rows may have been analysed with a config the user has since
#    rewritten — exporting them would mislabel the dataset against the
#    current AnalysisConfig.  The user sees the ● badge in the tree
#    and knows those files need a re-run before export.
#  * Error rows have no scientific signal; including them would produce
#    rows full of NaN that waste reviewer time.
#
#  The per-file ``file_info`` from the pipeline encodes
#  (experiment, chip, chamber, analyzed_channel) for USUBJID, but NOT
#  the study-level (drug, concentration, is_baseline) trio.  Those
#  belong to the :class:`Group` in our sidecar — the analyse step
#  doesn't get to read group metadata, by design (files are portable,
#  the study is the overlay).  The enrichment step below patches those
#  three fields into a shallow copy of ``file_info`` on the way into
#  ``export_send_package``, so the CDISC builders see a coherent view.
#  We never mutate the original pipeline result — the worker returns
#  its own dict and we hand that dict to the exporter.


def _format_concentration_from_dose_uM(
    dose_uM: float | None, *, drug: str = ''
) -> str:
    """Render a :class:`Group.dose_uM` as a CDISC-parseable string.

    Called at export time to synthesise the ``file_info.concentration``
    string expected by :func:`cdisc_export._parse_concentration` (which
    wants input like ``"10 uM"`` → ``(10.0, 'umol/L')``).

    Rules:

    * ``dose_uM is None`` (baseline group) → ``''``.  The exporter's
      ``_is_baseline`` / ``_is_control`` short-circuits on empty drug
      AND baseline heuristics, and EX rows are skipped entirely for
      baseline subjects — so returning an empty string is correct.
    * ``drug`` is empty → still render the dose (user may use the
      drug field as a ``condition`` label, e.g. "vehicle-2x").  The
      exporter will treat it as CONTROL and drop it from EX, but a
      numeric dose_uM may still be informative if it gets echoed to
      TX (it is today).
    * Otherwise → ``"<value> uM"`` with ``:g`` formatting so 0.010
      renders as ``"0.01 uM"`` (scientists read that as "10 nM") and
      100 renders as ``"100 uM"``.

    We always emit "uM" (not "nM" / "mM") because:

    * the study sidecar stores dose in µM as its single unit of truth
      (see :class:`Group`);
    * :func:`cdisc_export._parse_concentration` maps ``um`` →
      ``umol/L`` in its unit whitelist, so the round-trip is lossless;
    * emitting a unit keeps ``EXDOSU`` informative in the exported
      submission instead of falling back to the default ``'nmol/L'``.
    """
    if dose_uM is None:
        return ''
    # ``:g`` strips trailing zeros for clean output (0.010 → "0.01").
    return f"{dose_uM:g} uM"


def _enrich_file_info_for_export(
    result: dict, group: Group, file: FileEntry
) -> dict:
    """Return a copy of ``result`` with ``file_info`` overlaid from the study.

    The scientific pipeline doesn't know that a CSV belongs to a group
    named ``dof-10nM`` with drug ``Dofetilide`` and dose ``0.010 µM`` —
    those live in the study sidecar.  Before handing the result to
    :func:`cdisc_export.export_send_package` we patch the three
    study-level fields the exporter reads:

    * ``drug``          → :attr:`Group.drug` (raw string; the exporter
      normalises to CDISC CT in ``_canonical_drug_send``).
    * ``concentration`` → derived from :attr:`Group.dose_uM`; see
      :func:`_format_concentration_from_dose_uM`.
    * ``is_baseline``   → True when ``dose_uM is None`` (study-level
      baseline).  The exporter ORs its own heuristic with this flag so
      baseline rows land in DM/ARMCD=CONTROL instead of spurious EX
      rows.  We write it even if the exporter already has its own
      ``_is_baseline`` (drug/filename substring), because making the
      study model the single source of truth avoids ambiguity when
      the user named the group "baseline" but kept the drug field
      populated for traceability.

    The ``file.note`` field is intentionally ignored: it's free-text
    user annotation, not a CDISC input.

    Returns a shallow copy of ``result`` with a new ``file_info``
    dict — does NOT mutate the caller's ``result``.  We deep-copy
    only ``file_info`` (not ``summary`` or beat arrays) because
    nothing downstream of the exporter re-reads those nested structures.
    """
    enriched = dict(result)
    fi_src = result.get('file_info') or {}
    fi = dict(fi_src)  # shallow-copy the sub-dict so we don't mutate
    fi['drug'] = group.drug or ''
    fi['concentration'] = _format_concentration_from_dose_uM(
        group.dose_uM, drug=group.drug or '',
    )
    fi['is_baseline'] = group.dose_uM is None
    enriched['file_info'] = fi
    return enriched


def _collect_export_inputs(
    study: Study,
    results_cache: dict,
) -> list[tuple[Group, FileEntry]]:
    """Walk a study and return the ``(group, file)`` pairs that are ready to export.

    "Ready" = the cache holds a fresh, ``status='ok'`` :class:`FileResult`
    for ``(group.name, file.csv_relpath)``.  Stale and error rows are
    dropped by design (see module-level rationale for step 6).  Missing
    cache entries (user never clicked "Analizza gruppo" on this group)
    are also dropped — the caller can surface a "nothing fresh to
    export" dialog instead of producing a misleading half-dataset.

    The result preserves study/group/file ordering so the exported
    USUBJID / EXSEQ sequences come out in the same order the user sees
    in the tree, which is easier to audit against the study sidecar.

    ``O(total files)`` — studies are small (≤ ~50 files).
    """
    pairs: list[tuple[Group, FileEntry]] = []
    for group in study.groups:
        for f in group.files:
            r = results_cache.get((group.name, f.csv_relpath))
            if r is None or r.status != 'ok':
                continue
            if _is_stale(r, group.config, f.channel):
                continue
            pairs.append((group, f))
    return pairs


def _suggest_study_id(name: str) -> str:
    """Slugify a free-form study name into a CDISC-safe STUDYID.

    SENDIG 3.1 requires ``STUDYID`` to be a short identifier — there
    is no formal regex, but Pinnacle 21 and reviewers expect
    ``[A-Z0-9_-]{1,32}`` in practice.  We:

    1. Uppercase everything so the result is stable under
       case-insensitive filesystems when the sidecar round-trips.
    2. Replace any non-alphanumeric run with a single ``-`` so
       ``"Exp6 · dof-rr"`` → ``"EXP6-DOF-RR"``.
    3. Strip leading/trailing dashes so the result doesn't start
       with ``-`` (would trip some CDISC validators).
    4. Cap at 32 characters (pragmatic ceiling — SENDIG variable
       length for STUDYID is 40 chars, but reviewers prefer short).
    5. Fall back to ``'STUDY'`` when the slug is empty (e.g. the
       user named the study with only punctuation).

    The result is a *default* the user can overwrite in the input
    dialog — we never commit it to disk without their confirmation.
    """
    # Walk the string and replace any non-[A-Z0-9] sequence with '-'.
    out_chars: list[str] = []
    prev_dash = False
    for ch in (name or '').upper():
        if ch.isalnum():
            out_chars.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out_chars.append('-')
                prev_dash = True
    slug = ''.join(out_chars).strip('-')
    if not slug:
        return 'STUDY'
    return slug[:32]
