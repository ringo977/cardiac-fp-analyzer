"""
rhythm_integration.py — Sprint 2 #3, second PR.

Consumers of the rhythm topology classification produced by
`_classify_rhythm_topology()` live here so the classifier module stays
purely analytical and the downstream code paths (parameter extraction,
QC grading, UI) contain only thin hooks.

Four responsibilities:

1. ``apply_rhythm_filter``
     Given the QC-cleaned beats and the rhythm classification, drop beats
     that should not contribute to FPD / amplitude estimation:

       * ``regular_with_ectopics`` → keep dominant cluster only
       * ``alternans_2_to_1``      → keep dominant cluster only
       * ``regular_with_noise``    → keep dominant cluster only
       * ``trimodal``              → keep dominant cluster only
       * everything else           → passthrough (no filtering)

     Returns the (possibly shortened) beat arrays plus a diagnostic dict.

2. ``build_rhythm_summary_fields``
     Project the classification + filter outcome into a flat dict of
     fields that can be merged into ``summary`` produced by
     ``extract_all_parameters``. Fields are *additive* — nothing clashes
     with existing keys, so downstream exports stay backward compatible.

3. ``apply_rhythm_qc_downgrade``
     Mutates a ``QualityReport.grade`` when noise contamination exceeds a
     configurable ratio. Implemented here (instead of inside
     `quality_control.py`) so `quality_control.py` stays unaware of the
     rhythm classifier's existence.

4. ``render_rhythm_badge_html``
     Build a small Streamlit-friendly HTML pill rendering the rhythm
     type and any flags, with clinical severity colour-coding.

All four functions are pure (apart from ``apply_rhythm_qc_downgrade``
which mutates the supplied QC report) and tolerate missing/partial
classification dicts so the integration can be toggled off in config
without breaking callers.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

# Rhythm types for which we restrict FPD / amplitude stats to the
# dominant amplitude cluster only.
_DOMINANT_ONLY_TYPES = frozenset({
    'regular_with_ectopics',
    'regular_with_noise',
    'alternans_2_to_1',
    'trimodal',
})

# Rhythm types that unambiguously pass beats through unchanged.
_PASSTHROUGH_TYPES = frozenset({
    'regular',
    'chaotic',
    'unimodal_insufficient',
    'degenerate',
    'ambiguous',
    'disabled',
    'error',
})

# Severity order for UI badge stacking (first match wins when composing
# the short label; flag pills render all applicable).
_SEVERITY_ORDER: list[tuple[str, str, str]] = [
    # (rhythm_type, italian_label, css_colour)
    ('alternans_2_to_1',       'Alternans 2:1',               '#c0392b'),
    ('chaotic',                'Ritmo caotico',               '#c0392b'),
    ('trimodal',               'Trimodale',                   '#e67e22'),
    ('regular_with_noise',     'Contaminazione rumore',       '#e67e22'),
    ('regular_with_ectopics',  'Battiti ectopici',            '#f1c40f'),
    ('ambiguous',              'Ritmo ambiguo',               '#7f8c8d'),
    ('unimodal_insufficient',  'Beat insufficienti',          '#7f8c8d'),
    ('degenerate',             'Finestra degenere',           '#7f8c8d'),
    ('regular',                'Ritmo regolare',              '#27ae60'),
    ('disabled',               'Classificazione disattivata', '#7f8c8d'),
    ('error',                  'Errore classificazione',     '#7f8c8d'),
]

_FLAG_LABELS = {
    'alternans_pattern':      ('Alternans',              '#c0392b'),
    'arrhythmia_candidate':   ('Candidato aritmia',      '#e67e22'),
    'noise_contamination':    ('Rumore',                 '#e67e22'),
    'manual_review_required': ('Review manuale',         '#c0392b'),
}


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _get_dominant_bi_values(rc: dict[str, Any],
                            bi_raw: np.ndarray) -> set | None:
    """Return the set of sample indices of the dominant cluster as stored
    in ``bi_raw``. ``None`` if no dominant cluster is present.
    """
    clusters = rc.get('clusters') or []
    for cl in clusters:
        if cl.get('role') == 'dominant':
            idx = cl.get('indices_in_bi') or []
            if len(idx) == 0 or len(bi_raw) == 0:
                return None
            try:
                return set(int(v) for v in bi_raw[np.asarray(idx, dtype=int)])
            except (IndexError, ValueError):
                return None
    return None


def _count_by_role(rc: dict[str, Any]) -> dict[str, int]:
    counts = {'dominant': 0, 'secondary': 0, 'noise': 0, 'review': 0}
    for cl in (rc.get('clusters') or []):
        role = cl.get('role')
        if role in counts:
            counts[role] += int(cl.get('n', 0))
    return counts


# ═══════════════════════════════════════════════════════════════════════
#  1. Filter beats based on rhythm classification
# ═══════════════════════════════════════════════════════════════════════


def apply_rhythm_filter(
    bd_clean: np.ndarray,
    btm_clean: np.ndarray,
    bi_clean: np.ndarray,
    bi_raw: np.ndarray,
    rhythm_classification: dict[str, Any] | None,
    enable: bool = True,
    min_retention_ratio: float = 0.5,
    min_retention_beats: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Filter QC-cleaned beats so FPD / amplitude statistics use only the
    dominant amplitude cluster when the rhythm topology calls for it.

    Parameters
    ----------
    bd_clean, btm_clean : 2-D arrays
        Beat data and beat-time matrices after QC.
    bi_clean : 1-D int array
        Sample indices of QC-accepted beats.
    bi_raw : 1-D int array
        Sample indices of ALL detected beats before QC. The classifier's
        ``indices_in_bi`` point into this array.
    rhythm_classification : dict
        ``info['rhythm_classification']`` from ``detect_beats``.
    enable : bool
        Master switch. When False, returns inputs unchanged.
    min_retention_ratio : float, default 0.5
        Safety bail: if applying "dominant cluster only" would keep
        fewer than ``min_retention_ratio × len(bi_clean)`` beats, fall
        back to passthrough. A rhythm whose dominant cluster carries
        less than half the QC-accepted signal is almost certainly a
        trimodal/ectopic classification artefact (observed on
        Exp6_chipD_ch1 EL1: 34 QC beats → 7 trimodal-kept).
    min_retention_beats : int, default 3
        Safety bail: also fall back to passthrough if the absolute
        number of kept beats would be below this threshold.

    Returns
    -------
    bd_f, btm_f, bi_f : arrays
        Filtered beat data / times / indices (subset of inputs).
    filter_info : dict
        Diagnostics — always contains at least ``filter_applied`` (bool)
        and ``rhythm_type`` (str).
    """
    rc = rhythm_classification or {}
    rtype = rc.get('rhythm_type', 'unknown')
    info: dict[str, Any] = {
        'filter_applied': False,
        'rhythm_type': rtype,
        'n_input': int(len(bi_clean)),
        'n_kept': int(len(bi_clean)),
        'n_dropped': 0,
        'kept_role': None,
        'reason': None,
    }

    if not enable:
        info['reason'] = 'disabled_in_config'
        return bd_clean, btm_clean, bi_clean, info

    if rtype in _PASSTHROUGH_TYPES or rtype not in _DOMINANT_ONLY_TYPES:
        info['reason'] = 'passthrough'
        return bd_clean, btm_clean, bi_clean, info

    if len(bi_clean) == 0 or len(bi_raw) == 0:
        info['reason'] = 'empty_beats'
        return bd_clean, btm_clean, bi_clean, info

    dom_values = _get_dominant_bi_values(rc, np.asarray(bi_raw, dtype=int))
    if dom_values is None or len(dom_values) == 0:
        info['reason'] = 'no_dominant_cluster'
        return bd_clean, btm_clean, bi_clean, info

    keep_mask = np.array(
        [int(v) in dom_values for v in np.asarray(bi_clean, dtype=int)],
        dtype=bool,
    )
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        # Safety: QC may have removed all dominant-cluster beats (rare).
        # Fall back to passthrough rather than returning an empty beat
        # set that would crash parameter extraction.
        info['reason'] = 'qc_removed_all_dominant'
        return bd_clean, btm_clean, bi_clean, info

    # Safety bail (Sprint 3 #3 — bradycardia trimodal fix):
    # if keeping only the dominant cluster would discard too much of
    # the QC-accepted signal, the classification is almost certainly a
    # trimodal artefact of bradycardic R+T double-spikes (Exp6_chipD_ch1
    # EL1: 34 → 7 beats). Fall back to passthrough and annotate the
    # diagnostic dict so downstream UI / summary can still report the
    # rhythm type.
    n_input = int(len(bi_clean))
    min_ratio = float(min_retention_ratio) if min_retention_ratio is not None else 0.0
    min_abs = int(min_retention_beats) if min_retention_beats is not None else 0
    retention_ratio = n_kept / n_input if n_input > 0 else 1.0
    if retention_ratio < min_ratio or n_kept < min_abs:
        info.update({
            'filter_applied': False,
            'n_kept': n_input,
            'n_dropped': 0,
            'kept_role': None,
            'reason': 'safety_bail_low_retention',
            'safety_bail': {
                'n_would_keep': n_kept,
                'retention_ratio': round(retention_ratio, 4),
                'min_retention_ratio': min_ratio,
                'min_retention_beats': min_abs,
            },
        })
        return bd_clean, btm_clean, bi_clean, info

    info.update({
        'filter_applied': True,
        'n_kept': n_kept,
        'n_dropped': int(len(bi_clean) - n_kept),
        'kept_role': 'dominant',
        'reason': 'kept_dominant_only',
    })

    # Apply mask — bd_clean/btm_clean may be numpy arrays OR Python lists of
    # variable-length beat arrays; handle both.
    def _apply_mask(seq, mask):
        if isinstance(seq, np.ndarray):
            return seq[mask]
        return [item for item, keep in zip(seq, mask) if keep]

    bi_out = np.asarray(bi_clean)[keep_mask] if isinstance(bi_clean, np.ndarray) \
        else np.asarray([v for v, k in zip(bi_clean, keep_mask) if k], dtype=int)
    return _apply_mask(bd_clean, keep_mask), _apply_mask(btm_clean, keep_mask), bi_out, info


# ═══════════════════════════════════════════════════════════════════════
#  1-bis. RR outlier filter
# ═══════════════════════════════════════════════════════════════════════


def apply_rr_outlier_filter(
    bd: Any,
    btm: Any,
    bi: np.ndarray,
    fs: float,
    max_rr_ratio: float = 5.0,
    enable: bool = True,
) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    """Drop beats whose adjacent RR interval is pathologically long.

    Rationale
    ---------
    A beat whose preceding RR is more than ``max_rr_ratio`` times the
    median RR is almost always one of:

      * a missed detection (detector skipped intermediate beats during
        a noise burst or a dropout)
      * a quiescent gap followed by a spontaneous reactivation beat
      * an artefact that happened to fire the detector

    Keeping such beats contaminates FPD / amplitude statistics: on
    Exp6_chipD_ch1 a single 24-second dropout gap produced the only
    "valid" FPD in the recording (918 ms → FPDcF 318.8 ± 0, QC grade F)
    because every other beat failed repolarisation detection.

    The filter drops a beat when BOTH:
      * its preceding RR (``bi[i] - bi[i-1]``) exceeds
        ``max_rr_ratio × median_RR``, or
      * (first beat only) its following RR does,
    after computing the median on the raw differences (so the median
    itself is robust to a single large outlier — the MAD-like pattern
    of robust statistics).

    Parameters
    ----------
    bd, btm : 2-D arrays or Python lists of 1-D arrays
        Beat-data / beat-time matrices, aligned with ``bi``.
    bi : 1-D int array
        Sample indices of beats (same length as ``bd`` / ``btm``).
    fs : float
        Sampling rate (Hz). Only used for diagnostics.
    max_rr_ratio : float, default 5.0
        Beats with RR > this × median RR are dropped.
    enable : bool, default True
        Master switch.

    Returns
    -------
    bd_f, btm_f, bi_f : filtered copies (subset of inputs).
    info : dict with keys
        ``filter_applied`` (bool), ``n_input``, ``n_kept``, ``n_dropped``,
        ``median_rr_ms``, ``max_rr_ratio``, ``reason``.
    """
    info: dict[str, Any] = {
        'filter_applied': False,
        'n_input': int(len(bi)) if bi is not None else 0,
        'n_kept': int(len(bi)) if bi is not None else 0,
        'n_dropped': 0,
        'median_rr_ms': None,
        'max_rr_ratio': float(max_rr_ratio),
        'reason': None,
    }

    if not enable:
        info['reason'] = 'disabled'
        return bd, btm, bi, info

    bi_arr = np.asarray(bi, dtype=int) if bi is not None else np.array([], dtype=int)

    # Need at least 3 beats to derive a meaningful median RR.
    if len(bi_arr) < 3:
        info['reason'] = 'insufficient_beats'
        return bd, btm, bi_arr, info

    if max_rr_ratio <= 0 or not np.isfinite(max_rr_ratio):
        info['reason'] = 'ratio_disabled'
        return bd, btm, bi_arr, info

    rr_samples = np.diff(bi_arr).astype(float)
    if not np.any(rr_samples > 0):
        info['reason'] = 'degenerate_rr'
        return bd, btm, bi_arr, info

    median_rr = float(np.median(rr_samples[rr_samples > 0]))
    if median_rr <= 0 or not np.isfinite(median_rr):
        info['reason'] = 'degenerate_median'
        return bd, btm, bi_arr, info

    threshold = max_rr_ratio * median_rr

    # A beat is dropped when its *preceding* RR exceeds the threshold:
    # this models the "reactivation after a long gap" pattern, where the
    # beat immediately after the gap is the contaminant (it's the
    # artefact / lone spontaneous beat that breaks FPD statistics). The
    # beat BEFORE the gap is legitimate and is kept.
    # On genuinely bradycardic signals every RR is uniformly long, so
    # no individual RR exceeds the median by ``max_rr_ratio`` and the
    # filter is a no-op — which is what we want.
    n = len(bi_arr)
    keep = np.ones(n, dtype=bool)
    for i in range(1, n):
        prev_rr = rr_samples[i - 1]
        if prev_rr > threshold:
            keep[i] = False

    n_kept = int(keep.sum())
    info['median_rr_ms'] = median_rr / fs * 1000 if fs > 0 else None
    if n_kept == n:
        info['reason'] = 'no_outliers'
        return bd, btm, bi_arr, info

    # Safety net: if the filter wipes out more than half the beats, the
    # median RR itself is unreliable (signal is mostly dropout). Bail
    # out and keep the input unchanged rather than produce a nearly
    # empty beat set.
    if n_kept < max(3, n // 2):
        info['reason'] = 'too_many_dropped_safety_bail'
        return bd, btm, bi_arr, info

    info.update({
        'filter_applied': True,
        'n_kept': n_kept,
        'n_dropped': int(n - n_kept),
        'reason': 'rr_outlier_dropped',
    })

    def _apply_mask(seq: Any, mask: np.ndarray) -> Any:
        if isinstance(seq, np.ndarray):
            return seq[mask]
        return [item for item, k in zip(seq, mask) if k]

    bi_out = bi_arr[keep]
    return _apply_mask(bd, keep), _apply_mask(btm, keep), bi_out, info


# ═══════════════════════════════════════════════════════════════════════
#  2. Summary enrichment
# ═══════════════════════════════════════════════════════════════════════


def build_rhythm_summary_fields(
    rhythm_classification: dict[str, Any] | None,
    filter_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a flat dict of rhythm-derived fields to merge into the
    per-file ``summary`` dict. Returns an empty dict when classification
    is missing.

    Notes
    -----
    * ``bpm_effective`` is only present for ``alternans_2_to_1``; for
      other rhythm types it is stored as ``None`` (so the CDISC / Excel
      exports can skip it cleanly).
    * ``noise_contamination_ratio`` is in [0, 1] and is 0 when no noise
      cluster is present.
    """
    rc = rhythm_classification or {}
    if not rc or not rc.get('rhythm_type'):
        return {}

    rtype = rc.get('rhythm_type', 'unknown')
    metrics = rc.get('metrics') or {}
    flags = list(rc.get('flags') or [])
    counts = _count_by_role(rc)
    n_bio = counts['dominant'] + counts['secondary']
    n_total = n_bio + counts['noise']
    noise_ratio = (counts['noise'] / n_total) if n_total > 0 else 0.0
    ectopic_rate = (counts['secondary'] / n_bio) if n_bio > 0 else 0.0

    fields: dict[str, Any] = {
        'rhythm_type': rtype,
        'rhythm_flags': ';'.join(flags) if flags else '',
        'n_beats_dominant': counts['dominant'],
        'n_beats_secondary': counts['secondary'],
        'n_beats_noise': counts['noise'],
        'n_beats_classified_total': n_total,
        'noise_contamination_ratio': round(noise_ratio, 4),
        'ectopic_rate_pct': round(ectopic_rate * 100.0, 2),
        'bpm_dominant': metrics.get('bpm_dominant'),
        'cv_rr_dominant': metrics.get('cv_rr_dominant'),
        'bpm_effective': metrics.get('bpm_effective'),  # alternans only
        'alternans_flag': rtype == 'alternans_2_to_1',
        'alternans_phase_median': metrics.get('alternans_phase_median'),
        'alternans_phase_std': metrics.get('alternans_phase_std'),
    }

    if filter_info:
        fields.update({
            'rhythm_filter_applied': bool(filter_info.get('filter_applied', False)),
            'rhythm_filter_n_dropped': int(filter_info.get('n_dropped', 0)),
            'rhythm_filter_kept_role': filter_info.get('kept_role'),
        })

    return fields


# ═══════════════════════════════════════════════════════════════════════
#  3. QC downgrade for noise-contaminated signals
# ═══════════════════════════════════════════════════════════════════════

_GRADE_ORDER = ['A', 'B', 'C', 'D', 'F']


def apply_rhythm_qc_downgrade(
    qc_report,
    rhythm_classification: dict[str, Any] | None,
    downgrade_threshold: float = 0.30,
    downgrade_steps: int = 1,
    enable: bool = True,
) -> dict[str, Any]:
    """Mutate ``qc_report.grade`` one or more steps toward 'F' when noise
    contamination exceeds ``downgrade_threshold``.

    Returns a diagnostic dict describing whether the downgrade was
    applied and by how many grade steps.
    """
    info: dict[str, Any] = {
        'applied': False,
        'reason': None,
        'noise_ratio': 0.0,
        'steps': 0,
        'grade_before': getattr(qc_report, 'grade', None),
        'grade_after': getattr(qc_report, 'grade', None),
    }
    if not enable:
        info['reason'] = 'disabled_in_config'
        return info
    if qc_report is None:
        info['reason'] = 'no_qc_report'
        return info

    rc = rhythm_classification or {}
    counts = _count_by_role(rc)
    n_noise = counts['noise']
    n_total = counts['dominant'] + counts['secondary'] + n_noise
    noise_ratio = (n_noise / n_total) if n_total > 0 else 0.0
    info['noise_ratio'] = round(noise_ratio, 4)

    if noise_ratio < downgrade_threshold:
        info['reason'] = 'below_threshold'
        return info

    grade_before = qc_report.grade
    if grade_before not in _GRADE_ORDER:
        info['reason'] = 'unknown_grade'
        return info

    cur = _GRADE_ORDER.index(grade_before)
    new_idx = min(cur + max(1, int(downgrade_steps)), len(_GRADE_ORDER) - 1)
    grade_after = _GRADE_ORDER[new_idx]
    qc_report.grade = grade_after

    note = (f"Rhythm QC downgrade: noise_contamination_ratio="
            f"{noise_ratio:.2f} ≥ {downgrade_threshold:.2f} → "
            f"{grade_before} → {grade_after}")
    notes = getattr(qc_report, 'notes', None)
    if isinstance(notes, list):
        notes.append(note)

    info.update({
        'applied': True,
        'reason': 'noise_above_threshold',
        'steps': new_idx - cur,
        'grade_after': grade_after,
    })
    return info


# ═══════════════════════════════════════════════════════════════════════
#  4. UI badge rendering
# ═══════════════════════════════════════════════════════════════════════


def _pill(text: str, color: str) -> str:
    return (f'<span style="display:inline-block;padding:2px 10px;'
            f'margin:2px 4px 2px 0;border-radius:10px;'
            f'background:{color};color:white;font-size:0.82rem;'
            f'font-weight:600;">{text}</span>')


def render_rhythm_badge_html(
    rhythm_type: str | None,
    flags: list[str] | None = None,
) -> str:
    """Return an inline HTML string suitable for
    ``st.markdown(..., unsafe_allow_html=True)``.

    Returns an empty string when no rhythm type is available.
    """
    if not rhythm_type:
        return ''

    label = None
    color = '#7f8c8d'
    for rt, lbl, clr in _SEVERITY_ORDER:
        if rt == rhythm_type:
            label, color = lbl, clr
            break
    if label is None:
        label, color = rhythm_type, '#7f8c8d'

    parts = [_pill(label, color)]
    for f in (flags or []):
        if f in _FLAG_LABELS:
            flbl, fclr = _FLAG_LABELS[f]
            parts.append(_pill(flbl, fclr))
    return ''.join(parts)
