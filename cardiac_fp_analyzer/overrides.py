"""Manual beat override persistence + application (v3.5.x, manual picking).

Users may correct the automatic detection by adding missed beats or
removing spurious ones via the Streamlit UI.  Those corrections are
persisted next to the CSV as a ``.overrides.json`` sidecar so they
survive re-analysis and can be moved with the data.

Design notes
------------
* Overrides are stored in **seconds** (sample-rate independent), so a
  sidecar written at fs=10 kHz keeps working if the user re-loads the
  CSV at fs=2 kHz or after a resampling step.
* ``removed_s`` matches by **nearest automatic beat within ``tol_s``** —
  the UI click may land a few ms off the true sample, and we don't want
  to throw the correction away on float jitter.
* ``added_s`` inserts a new index at ``round(t * fs)`` iff no existing
  beat already sits within ``tol_s`` (prevents double-counting when the
  click lands near an already-detected beat).
* The sidecar is versioned so we can evolve the schema without breaking
  existing files.  ``load_overrides`` is permissive: unknown fields are
  ignored, missing fields fall back to sensible defaults, malformed JSON
  returns ``None`` with a warning log (treat as "no overrides").

The module is pure Python — no Streamlit, no numpy beyond array ops —
so it is easy to unit-test and reusable outside the UI.
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

import numpy as np

logger = logging.getLogger(__name__)

OVERRIDES_SUFFIX = '.overrides.json'
SCHEMA_VERSION = '1'

# Default tolerance for matching a ``removed`` timestamp to an automatic
# beat, and for de-duplicating ``added`` timestamps against existing
# beats.  50 ms is comfortably above click jitter (~1–5 px at typical
# zoom levels ≈ 5–30 ms) and below any realistic inter-beat interval.
DEFAULT_TOL_S = 0.050


@dataclass
class BeatOverrides:
    """Container for manual beat corrections on a single signal.

    Attributes
    ----------
    added_s : list[float]
        Timestamps (seconds) of beats the user added manually.
    removed_s : list[float]
        Timestamps (seconds) of automatic beats the user rejected.
    tol_s : float
        Tolerance (seconds) for the nearest-match used when applying
        ``removed_s`` to an automatic beat set.
    version : str
        Schema version of the sidecar file.  Not user-editable.
    """

    added_s: list[float] = field(default_factory=list)
    removed_s: list[float] = field(default_factory=list)
    tol_s: float = DEFAULT_TOL_S
    version: str = SCHEMA_VERSION

    # ── Serialisation helpers ─────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a plain dict ready for ``json.dump``."""
        return {
            'version': self.version,
            'added_s': [float(x) for x in self.added_s],
            'removed_s': [float(x) for x in self.removed_s],
            'tol_s': float(self.tol_s),
        }

    @classmethod
    def from_dict(cls, d: dict) -> BeatOverrides:
        """Build from a dict, tolerating missing / unknown keys."""
        return cls(
            added_s=[float(x) for x in (d.get('added_s') or [])],
            removed_s=[float(x) for x in (d.get('removed_s') or [])],
            tol_s=float(d.get('tol_s', DEFAULT_TOL_S)),
            version=str(d.get('version', SCHEMA_VERSION)),
        )

    def is_empty(self) -> bool:
        """True if nothing was added/removed — sidecar can be deleted."""
        return not (self.added_s or self.removed_s)


# ═══════════════════════════════════════════════════════════════════════
#  Path handling
# ═══════════════════════════════════════════════════════════════════════

def overrides_path_for(csv_path: str | os.PathLike) -> Path:
    """Return the sidecar path for a given CSV — ``<csv>.overrides.json``.

    We append the suffix instead of replacing ``.csv`` so that
    ``foo.csv`` → ``foo.csv.overrides.json`` remains adjacent and
    unambiguous even when users have files like ``foo.csv.bak``.
    """
    p = Path(csv_path)
    return p.with_name(p.name + OVERRIDES_SUFFIX)


# ═══════════════════════════════════════════════════════════════════════
#  Load / save
# ═══════════════════════════════════════════════════════════════════════

def load_overrides(csv_path: str | os.PathLike) -> BeatOverrides | None:
    """Load overrides for a CSV, returning ``None`` if none exist.

    A missing sidecar is not an error — it simply means the user has
    never corrected this file.  A malformed sidecar is logged as a
    warning and treated the same way, so a corrupt file never blocks
    analysis.
    """
    path = overrides_path_for(csv_path)
    if not path.is_file():
        return None
    try:
        with path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "overrides: failed to read %s (%s); ignoring sidecar", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning(
            "overrides: %s is not a JSON object; ignoring sidecar", path)
        return None
    return BeatOverrides.from_dict(data)


def save_overrides(csv_path: str | os.PathLike,
                   overrides: BeatOverrides) -> Path:
    """Atomically write the sidecar next to the CSV.

    If ``overrides.is_empty()`` the existing sidecar is *deleted*
    instead of being written as an empty stub — keeps the directory
    tidy and the semantics clean ("no sidecar" ≡ "no corrections").

    Returns the sidecar path (useful for the UI to display).
    """
    path = overrides_path_for(csv_path)
    if overrides.is_empty():
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("overrides: unlink failed for %s (%s)", path, exc)
        return path

    # Atomic write: write to a temp file in the same directory, then
    # rename.  Keeps readers from seeing a half-written file and works
    # across the filesystem boundary on Linux / macOS.
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(directory))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(overrides.to_dict(), fh, indent=2, sort_keys=True)
            fh.write('\n')
        os.replace(tmp_path, path)
    except OSError:
        # Best-effort cleanup of the tmp file if rename failed.
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return path


# ═══════════════════════════════════════════════════════════════════════
#  Apply to a beat set
# ═══════════════════════════════════════════════════════════════════════

def apply_overrides(
    bi_auto: np.ndarray | list[int],
    overrides: BeatOverrides,
    fs: float,
) -> tuple[np.ndarray, dict]:
    """Apply user overrides on top of the automatic beat indices.

    Parameters
    ----------
    bi_auto : 1-D int array
        Automatic beat sample indices.
    overrides : BeatOverrides
        User corrections.  Times are in seconds.
    fs : float
        Sample rate (Hz), used to convert between seconds and samples.

    Returns
    -------
    bi_manual : 1-D int array, sorted, de-duplicated
        Beat set after corrections.
    info : dict
        Diagnostics: ``n_added``, ``n_removed``, ``unmatched_removals``
        (removals that did not find a near-enough automatic beat — kept
        in the sidecar so a future re-detection that *does* expose them
        still removes them).
    """
    if fs <= 0:
        raise ValueError(f"apply_overrides: fs must be positive, got {fs}")

    bi_auto_arr = np.asarray(sorted(set(int(x) for x in bi_auto)), dtype=int)
    tol_samples = max(1, int(round(overrides.tol_s * fs)))

    # ── Step 1: remove ────────────────────────────────────────────────
    kept_mask = np.ones(bi_auto_arr.shape, dtype=bool)
    unmatched_removals: list[float] = []
    for t_rm in overrides.removed_s:
        if bi_auto_arr.size == 0:
            unmatched_removals.append(float(t_rm))
            continue
        target = int(round(t_rm * fs))
        dists = np.abs(bi_auto_arr - target)
        j = int(np.argmin(dists))
        if dists[j] <= tol_samples and kept_mask[j]:
            kept_mask[j] = False
        else:
            unmatched_removals.append(float(t_rm))
    bi_after_remove = bi_auto_arr[kept_mask]

    # ── Step 2: add ───────────────────────────────────────────────────
    bi_set = set(int(x) for x in bi_after_remove.tolist())
    n_added = 0
    for t_add in overrides.added_s:
        new_idx = int(round(t_add * fs))
        # Reject duplicates: any existing beat within tol_samples blocks
        # the insert.  Numerical re-clicks and floating-point noise
        # should not multiply the beat set.
        if bi_set:
            arr = np.fromiter(bi_set, dtype=int)
            if np.min(np.abs(arr - new_idx)) <= tol_samples:
                continue
        bi_set.add(new_idx)
        n_added += 1

    bi_manual = np.asarray(sorted(bi_set), dtype=int)

    info = {
        'n_added': int(n_added),
        'n_removed': int(len(overrides.removed_s) - len(unmatched_removals)),
        'unmatched_removals': unmatched_removals,
        'n_auto': int(bi_auto_arr.size),
        'n_manual': int(bi_manual.size),
    }
    return bi_manual, info


# ═══════════════════════════════════════════════════════════════════════
#  Convenience: update from a (kept, added) pair captured by the UI
# ═══════════════════════════════════════════════════════════════════════

def diff_to_overrides(
    bi_auto: np.ndarray | list[int],
    bi_final: np.ndarray | list[int],
    fs: float,
    tol_s: float = DEFAULT_TOL_S,
) -> BeatOverrides:
    """Infer ``BeatOverrides`` from (automatic, user-final) beat sets.

    Useful for the UI: the beat editor tracks the "current" set via
    checkboxes and a timestamp input box.  When the user hits "Save
    overrides", we diff that set against the original automatic set to
    build the minimal override record.

    ``tol_s`` is only stored on the sidecar (used later by
    ``apply_overrides``); it does NOT loosen the diff itself.
    """
    auto = set(int(x) for x in bi_auto)
    final = set(int(x) for x in bi_final)
    added_idx = sorted(final - auto)
    removed_idx = sorted(auto - final)
    return BeatOverrides(
        added_s=[i / float(fs) for i in added_idx],
        removed_s=[i / float(fs) for i in removed_idx],
        tol_s=tol_s,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Public API for ``from cardiac_fp_analyzer.overrides import *``
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    'BeatOverrides',
    'OVERRIDES_SUFFIX',
    'SCHEMA_VERSION',
    'DEFAULT_TOL_S',
    'overrides_path_for',
    'load_overrides',
    'save_overrides',
    'apply_overrides',
    'diff_to_overrides',
]

# Ensure ``dataclasses`` import is used even if linters can't see the
# decorator reference above on some setups.
_ = dataclasses
