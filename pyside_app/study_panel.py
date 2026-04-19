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
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
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

        # ── Toolbar ───────────────────────────────────────────────────
        # Layout in three groups, separated visually:
        #   1. Study lifecycle     : New / Open / Close
        #   2. Study editing       : Add group / Add CSV / Group settings /
        #                            Remove (group or file)
        #   3. Destructive study   : Delete sidecar (kept at the far end so
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
        # Group 3 — destructive. Distanced from the editing group on purpose.
        self._toolbar.addAction(self._act_delete_study)

        self._act_new.triggered.connect(self._on_new_study)
        self._act_open.triggered.connect(self._on_open_study)
        self._act_close.triggered.connect(self._on_close_study)
        self._act_add_group.triggered.connect(self._on_add_group)
        self._act_add_file.triggered.connect(self._on_add_file)
        self._act_group_settings.triggered.connect(self._on_group_settings)
        self._act_remove.triggered.connect(self._on_remove)
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
        """
        self._study = study
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
        self._rebuild_tree()
        self._refresh_actions()
        self.study_changed.emit()

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
        """Repopulate the tree from :attr:`_study`.  O(N) rows."""
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
            g_item.setText(1, _format_group_details(group))
            g_item.setData(0, _KIND_ROLE, "group")
            g_item.setData(0, _GROUP_NAME_ROLE, group.name)
            g_item.setExpanded(True)

            for idx, entry in enumerate(group.files):
                f_item = QTreeWidgetItem(g_item)
                # Show just the filename — the full relpath goes in the
                # details column so long paths don't blow out column 0.
                f_item.setText(0, Path(entry.csv_relpath).name)
                details = entry.csv_relpath
                if entry.channel and entry.channel != "auto":
                    details = f"{details}  ·  {entry.channel.upper()}"
                if entry.note:
                    details = f"{details}  ·  {entry.note}"
                f_item.setText(1, details)
                f_item.setData(0, _KIND_ROLE, "file")
                f_item.setData(0, _GROUP_NAME_ROLE, group.name)
                f_item.setData(0, _FILE_INDEX_ROLE, idx)

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
