"""Study / Group / File domain model for dose-response workflows.

Terminology note
----------------
Top-level container is **Study** — the GLP / OECD / 21 CFR Part 58 term
for a protocolled nonclinical investigation, and the top-level key
(``STUDYID``) in CDISC SDTM datasets.  "Project" was considered and
rejected: it has no regulatory semantics, and in CDISC ADaM the word
"Analysis" already has a precise meaning (post-raw derivation).  If an
umbrella concept above Study is ever needed (e.g. a multi-study
campaign on the same compound), that is where "project" would
correctly belong.

Rationale
---------
The single-file ``analyze_single_file`` pipeline is the right unit for a
signal-by-signal QC pass, but it doesn't know about the **experimental
structure** that users actually care about: a drug's dose-response curve
typically needs *baseline → dose₁ → dose₂ …* measured across several
chips / electrodes on the same day, and then aggregated across
biological replicates.

This module introduces three dataclasses that sit *above* the
single-file analysis:

* :class:`FileEntry`  — one CSV + channel choice + free-text note.
* :class:`Group`      — a set of file entries sharing a drug/dose/
  condition, with its own :class:`AnalysisConfig`.  Maps to a
  CDISC "treatment arm" / dose group.
* :class:`Study`      — a collection of groups persisted next to the
  data as ``.cfp-study.json``.

Persistence
-----------
A study is a folder with a ``.cfp-study.json`` sidecar.  The JSON
stores *relative* CSV paths so the whole folder can be moved or
renamed and the study still loads.  Atomic write via ``tempfile`` +
``os.replace`` — same pattern as :mod:`cardiac_fp_analyzer.overrides`.

The in-memory :class:`Study` carries an absolute ``folder`` attribute
that is **not serialised**: it is set by :func:`load_study` from
whichever path you give it.  That keeps the on-disk format portable
while the in-memory object knows how to resolve relative CSV paths
without round-tripping through the caller.

Design notes
------------
* Per-file config overrides are out of scope for step 1 — each Group
  carries one :class:`AnalysisConfig` that applies to all of its files.
  If a user wants a different config for one replicate they can split
  it into its own Group.  A future enhancement could add
  ``FileEntry.config_override: AnalysisConfig | None``.
* Analysis results are **not** cached on disk.  ``.overrides.json``
  sidecars next to each CSV remain the single source of truth for
  manual corrections; re-opening a study re-runs ``analyze_single_file``
  on each entry, which re-applies any sidecars present.
* No Qt import here — this module is meant to be driven by the
  PySide UI but is trivially testable headless.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from cardiac_fp_analyzer.config import AnalysisConfig

logger = logging.getLogger(__name__)

STUDY_FILENAME = '.cfp-study.json'
SCHEMA_VERSION = '1'


# ═══════════════════════════════════════════════════════════════════════
#  FileEntry
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FileEntry:
    """A single CSV within a group.

    Attributes
    ----------
    csv_relpath : str
        Path relative to the study folder (e.g. ``data/day1/exp6_ch2.csv``).
        Stored relative so the study is portable across machines /
        folder renames.  Always forward-slash-separated on disk — the
        POSIX form round-trips on Windows without surprises.
    channel : str
        Channel choice passed to ``analyze_single_file`` — ``'auto'``
        (pipeline decides), ``'EL1'`` / ``'EL2'``, or a future
        multi-electrode label.
    note : str
        Free-text user annotation (e.g. "noisy baseline",
        "moved electrode mid-recording").  Never interpreted.
    """

    csv_relpath: str
    channel: str = 'auto'
    note: str = ''

    def to_dict(self) -> dict:
        return {
            'csv_relpath': str(self.csv_relpath),
            'channel': str(self.channel),
            'note': str(self.note),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FileEntry:
        return cls(
            csv_relpath=str(d.get('csv_relpath', '')),
            channel=str(d.get('channel', 'auto')),
            note=str(d.get('note', '')),
        )


# ═══════════════════════════════════════════════════════════════════════
#  Group
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Group:
    """A set of files sharing a drug/dose/condition.

    ``dose_uM`` is optional because the common baseline / control groups
    have no dose.  We keep it as ``float | None`` rather than a sentinel
    like ``-1`` so ``"dose": null`` is what hits the JSON — no chance of
    confusing a "0 µM control" with "unknown".

    ``analysis_date`` is stored as an ISO-ish string (e.g.
    ``"2026-04-19"``).  We avoid ``datetime`` here so the JSON survives
    round-tripping without any timezone / format surprises; the UI
    layer is the right place to validate / format dates.

    Each group carries its own :class:`AnalysisConfig` so a baseline
    and a dose group can use different repolarisation search windows,
    QC thresholds, etc.  Step 1 keeps it at group granularity — per-file
    overrides are a future extension.
    """

    name: str
    drug: str = ''
    dose_uM: float | None = None
    condition: str = ''
    analysis_date: str = ''
    notes: str = ''
    files: list[FileEntry] = field(default_factory=list)
    config: AnalysisConfig = field(default_factory=AnalysisConfig)

    def to_dict(self) -> dict:
        return {
            'name': str(self.name),
            'drug': str(self.drug),
            'dose_uM': (
                None if self.dose_uM is None else float(self.dose_uM)
            ),
            'condition': str(self.condition),
            'analysis_date': str(self.analysis_date),
            'notes': str(self.notes),
            'files': [f.to_dict() for f in self.files],
            'config': self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Group:
        raw_dose = d.get('dose_uM')
        dose: float | None
        if raw_dose is None:
            dose = None
        else:
            try:
                dose = float(raw_dose)
            except (TypeError, ValueError):
                logger.warning(
                    "study: ignoring non-numeric dose_uM=%r for group %r",
                    raw_dose, d.get('name'),
                )
                dose = None

        raw_files = d.get('files') or []
        files = [FileEntry.from_dict(f) for f in raw_files if isinstance(f, dict)]

        raw_config = d.get('config')
        if isinstance(raw_config, dict):
            cfg = AnalysisConfig.from_dict(raw_config)
        else:
            cfg = AnalysisConfig()

        return cls(
            name=str(d.get('name', '')),
            drug=str(d.get('drug', '')),
            dose_uM=dose,
            condition=str(d.get('condition', '')),
            analysis_date=str(d.get('analysis_date', '')),
            notes=str(d.get('notes', '')),
            files=files,
            config=cfg,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Study
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Study:
    """Top-level container persisted to ``<folder>/.cfp-study.json``.

    ``folder`` is the absolute on-disk path of the study directory.
    It is set by :func:`load_study` / :func:`save_study` — do NOT
    serialise it, because a study moved to a different machine would
    carry a stale path.
    """

    name: str
    folder: str
    groups: list[Group] = field(default_factory=list)
    version: str = SCHEMA_VERSION

    # ── Path helpers ──────────────────────────────────────────────────

    @property
    def sidecar_path(self) -> Path:
        """Absolute path of the ``.cfp-study.json`` sidecar."""
        return Path(self.folder) / STUDY_FILENAME

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a plain dict ready for :func:`json.dump`.

        ``folder`` is deliberately excluded — see class docstring.
        """
        return {
            'version': self.version,
            'name': str(self.name),
            'groups': [g.to_dict() for g in self.groups],
        }

    @classmethod
    def from_dict(cls, d: dict, folder: str | os.PathLike) -> Study:
        """Build from a loaded dict, combined with the on-disk folder.

        ``folder`` is supplied by the caller (the loader) because it's
        not stored inside the JSON — see the class docstring.
        """
        raw_groups = d.get('groups') or []
        groups = [
            Group.from_dict(g) for g in raw_groups if isinstance(g, dict)
        ]
        return cls(
            name=str(d.get('name', Path(folder).name)),
            folder=str(Path(folder).resolve()),
            groups=groups,
            version=str(d.get('version', SCHEMA_VERSION)),
        )


# ═══════════════════════════════════════════════════════════════════════
#  Load / save
# ═══════════════════════════════════════════════════════════════════════

def load_study(folder: str | os.PathLike) -> Study | None:
    """Load the study from ``<folder>/.cfp-study.json``.

    Returns ``None`` if the sidecar is missing, malformed, or not a JSON
    object.  A missing sidecar is not an error — callers should treat a
    ``None`` return as "this folder is not a study yet".
    """
    folder_path = Path(folder)
    sidecar = folder_path / STUDY_FILENAME
    if not sidecar.is_file():
        return None
    try:
        with sidecar.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "study: failed to read %s (%s); treating as missing",
            sidecar, exc,
        )
        return None
    if not isinstance(data, dict):
        logger.warning(
            "study: %s is not a JSON object; treating as missing", sidecar)
        return None
    return Study.from_dict(data, folder=folder_path)


def save_study(study: Study) -> Path:
    """Atomically write the study sidecar.

    The destination folder must already exist — we don't silently
    create it, since a study targets a specific data directory and
    silently creating a new folder elsewhere would be surprising.

    Returns the sidecar path (useful for the UI to display).
    """
    folder = Path(study.folder)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"save_study: folder does not exist: {folder}"
        )
    sidecar = folder / STUDY_FILENAME
    fd, tmp_path = tempfile.mkstemp(
        prefix=sidecar.name + '.', suffix='.tmp', dir=str(folder),
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(
                study.to_dict(), fh, indent=2, sort_keys=True,
                ensure_ascii=False,
            )
            fh.write('\n')
        os.replace(tmp_path, sidecar)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return sidecar


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def resolve_file_path(study: Study, entry: FileEntry) -> Path:
    """Return the absolute path of a :class:`FileEntry` on disk.

    Joins the study folder with the entry's relative path.  Does
    *not* check for existence — callers that need that should call
    ``.is_file()`` themselves.
    """
    return (Path(study.folder) / entry.csv_relpath).resolve()


def make_file_entry(
    study: Study,
    csv_path: str | os.PathLike,
    channel: str = 'auto',
    note: str = '',
) -> FileEntry:
    """Build a :class:`FileEntry` whose path is relative to the study.

    Accepts both absolute paths (typical from a file-open dialog) and
    paths already relative to the study folder.  Raises ``ValueError``
    if the target isn't within the study folder — keeps the study
    self-contained and portable.
    """
    study_root = Path(study.folder).resolve()
    csv_abs = Path(csv_path)
    if not csv_abs.is_absolute():
        csv_abs = (study_root / csv_abs).resolve()
    else:
        csv_abs = csv_abs.resolve()
    try:
        rel = csv_abs.relative_to(study_root)
    except ValueError as exc:
        raise ValueError(
            f"CSV is outside the study folder: {csv_abs} "
            f"(study root: {study_root})"
        ) from exc
    # POSIX-style separator on disk — stable across OSes.
    return FileEntry(
        csv_relpath=rel.as_posix(), channel=str(channel), note=str(note),
    )


def find_group(study: Study, name: str) -> Group | None:
    """Return the first group whose ``name`` matches exactly, else ``None``."""
    for g in study.groups:
        if g.name == name:
            return g
    return None


def add_group(study: Study, group: Group) -> None:
    """Append a group, raising ``ValueError`` on duplicate names.

    Group names are the stable user-visible identifier (used in reports,
    Excel sheet names, etc.) so we enforce uniqueness at mutation time
    rather than pushing it onto the UI.
    """
    if not group.name:
        raise ValueError("add_group: group name must be non-empty")
    if find_group(study, group.name) is not None:
        raise ValueError(f"add_group: duplicate group name: {group.name!r}")
    study.groups.append(group)


def remove_group(study: Study, name: str) -> bool:
    """Remove the group with the given name.  Returns True if removed."""
    for i, g in enumerate(study.groups):
        if g.name == name:
            del study.groups[i]
            return True
    return False


def add_file_to_group(
    study: Study,
    group_name: str,
    csv_path: str | os.PathLike,
    channel: str = 'auto',
    note: str = '',
) -> FileEntry:
    """Convenience: build a :class:`FileEntry` and attach it to a group.

    Raises ``KeyError`` if the group doesn't exist, ``ValueError`` from
    :func:`make_file_entry` if the CSV is outside the study folder.
    """
    group = find_group(study, group_name)
    if group is None:
        raise KeyError(f"add_file_to_group: unknown group {group_name!r}")
    entry = make_file_entry(study, csv_path, channel=channel, note=note)
    group.files.append(entry)
    return entry


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    'STUDY_FILENAME',
    'SCHEMA_VERSION',
    'FileEntry',
    'Group',
    'Study',
    'load_study',
    'save_study',
    'resolve_file_path',
    'make_file_entry',
    'find_group',
    'add_group',
    'remove_group',
    'add_file_to_group',
]

# Keep the ``dataclasses`` import used by pyright/mypy even if no
# bare ``dataclasses.asdict`` / ``fields`` call remains above.
_ = dataclasses
