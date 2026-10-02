"""
normalization.py — Baseline normalization and TdP risk scoring.

After batch analysis, pairs each drug recording with its baseline
(same tissue: experiment + day + chip + chamber, same electrode) and computes:
  - %change in BP, FPDcF, AMP relative to baseline
  - TdP risk score based on FPDcF prolongation thresholds (ICH S7B)

Thresholds follow Visone et al. (Tox Sci 2023):
  LOW:  %FPDcF change ≥ 10%
  MID:  %FPDcF change ≥ 15%  (optimal threshold per paper)
  HIGH: %FPDcF change ≥ 20%

TdP risk scoring (from Ando et al. 2017):
  Score -1: shortening (↓ FPDcF)
  Score  0: no significant effect (|%FPDcF| < LOW threshold)
  Score  1: mild prolongation (LOW ≤ %FPDcF < MID)
  Score  2: moderate prolongation (MID ≤ %FPDcF < HIGH)
  Score  3: strong prolongation (%FPDcF ≥ HIGH) or arrhythmic events
"""

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ─── Drug names ───
# File names abbreviate drugs in many ways ('terfe', 'DOFE', 'Quinid',
# 'MEXILITINE', 'Alfuso', 'NIFEDIPINE_10'). classify_drug grouped on the raw
# string, so before Oct 2026 one drug could be called positive under one
# spelling and negative under another (8 drugs became 17 on the Visone 2023
# data). Prefixes are matched on the first word, longest first; prefixes that
# could belong to another compound ('quin' → quinine) are deliberately absent.
_DRUG_ALIASES = (
    ('terfe', 'terfenadine'), ('quinid', 'quinidine'), ('dofe', 'dofetilide'),
    ('alfu', 'alfuzosin'), ('mexi', 'mexiletine'), ('nife', 'nifedipine'),
    ('rano', 'ranolazine'), ('cisa', 'cisapride'), ('sotal', 'sotalol'),
    ('verap', 'verapamil'), ('aspir', 'aspirin'), ('bepri', 'bepridil'),
    ('torem', 'toremifene'), ('chloro', 'chloroquine'), ('cloro', 'chloroquine'),
    ('dmso', 'dmso'), ('vehicle', 'vehicle'),
)
VEHICLE_NAMES = frozenset({'dmso', 'vehicle'})


def canonical_drug_name(raw):
    """Canonical lower-case drug name: 'DOFE' → 'dofetilide',
    'nifedipine 10' → 'nifedipine'. Unknown names are returned lower-cased
    without a trailing separated number; codes such as 'Ti07' are kept."""
    s = str(raw or '').strip().lower()
    s = re.sub(r'[\s_-]+\d[\d._-]*$', '', s).strip()
    if not s:
        return ''
    word = re.split(r'[\s_]+', s)[0]
    for prefix, canon in sorted(_DRUG_ALIASES, key=lambda a: -len(a[0])):
        if word.startswith(prefix):
            return canon
    return s


def is_washout(result):
    """Washout / recovery recordings are not a concentration of the drug."""
    fi = result.get('file_info', {}) or {}
    text = f"{fi.get('drug', '') or ''} {(result.get('metadata', {}) or {}).get('filename', '')}".lower()
    return 'wash' in text or 'recovery' in text

# ─── Thresholds (module-level defaults, overridden by NormalizationConfig) ───
THRESHOLD_LOW = 10.0    # %FPDcF change ≥ 10%
THRESHOLD_MID = 15.0    # %FPDcF change ≥ 15% (paper's optimal)
THRESHOLD_HIGH = 20.0   # %FPDcF change ≥ 20%


def _get_norm_thresholds(cfg=None):
    """Return (LOW, MID, HIGH) thresholds from config or module defaults."""
    if cfg is None:
        return THRESHOLD_LOW, THRESHOLD_MID, THRESHOLD_HIGH
    return cfg.threshold_low, cfg.threshold_mid, cfg.threshold_high


def _get_base_key(result):
    """
    Extract the base grouping key (experiment + day + chip + chamber) WITHOUT electrode.
    E.g. Exp5/Day7/chipA_ch1_terfe_300nM → "exp5/day7/chipA_ch1"

    Results produced by analyze_single_file carry ``file_info['tissue']``
    (loader.describe_recording), which is used as is. Results built by
    hand or by older versions fall back to experiment + chip + chamber
    from the file name, as before.
    """
    fi = result.get('file_info', {})
    if fi.get('tissue'):
        return fi['tissue']
    exp = fi.get('experiment', '')
    chip = fi.get('chip', '')
    channel_label = fi.get('channel_label', '')

    meta = result.get('metadata', {})
    fname = meta.get('filename', '')

    parts = fname.split('_')
    if len(parts) >= 2 and parts[0].startswith('chip'):
        chip_ch = f"{parts[0]}_{parts[1]}"
    else:
        chip_ch = f"chip{chip}_{channel_label}" if chip and channel_label else fname

    return f"{exp}/{chip_ch}"


def get_group_key(result):
    """
    Extract the grouping key (experiment + chip + chamber + electrode) from a result.
    E.g. chipA_ch1_terfe_300nM analyzed on el1 → group = "EXP 5/chipA_ch1/el1"

    The group key includes the electrode (el1/el2) so that dual-electrode
    analyses of the same file are kept in separate normalization groups.
    """
    base = _get_base_key(result)
    electrode = result.get('file_info', {}).get('analyzed_channel', '')
    if electrode:
        return f"{base}/{electrode}"
    return base

# Back-compat alias for internal callers
_get_group_key = get_group_key


def is_baseline(result):
    """Check if a result is a baseline recording."""
    fi = result.get('file_info', {})
    drug = str(fi.get('drug', '') or '').lower()
    fname = str(result.get('metadata', {}).get('filename', '')).lower()

    return ('baseline' in drug or 'basline' in drug or
            'baseline' in fname or 'basline' in fname)

# Back-compat alias for internal callers
_is_baseline = is_baseline


def _is_control(result):
    """Check if a result is a control recording (CTRL/CTR)."""
    fi = result.get('file_info', {})
    drug = str(fi.get('drug', '') or '').lower()
    return drug.startswith('ctrl') or drug.startswith('ctr')


def recording_key(result):
    """Unique key of one analysed recording: file path + electrode.

    Pairing used to be keyed by the file *stem*, so two recordings with the
    same name in different folders ('ChipE/chipE_ch1_baseline' and
    'ChipE/new baseline/chipE_ch1_baseline'), or the el1 and el2 analyses
    of one file in 'both' mode, overwrote each other's baseline.
    """
    md = result.get('metadata', {}) or {}
    fi = result.get('file_info', {}) or {}
    return f"{md.get('filepath') or md.get('filename', '')}#{fi.get('analyzed_channel', '')}"


def _folder(result):
    fp = (result.get('metadata', {}) or {}).get('filepath')
    return str(Path(fp).parent) if fp else None


def pair_with_baselines(results_list, details=None):
    """
    Pair each drug recording with its baseline.

    Groups results by tissue (experiment + day + chip + chamber) and
    electrode. Among the baselines of the tissue, a recording is paired
    with the one in its own folder; if there are several, or none in its
    folder, with the best QC grade (then the file name). Baselines the
    analysability verdict rejected are never used. If the chosen baseline
    failed the inclusion criteria the recording is left unpaired.

    **Fallback**: when the electrode-specific group has no baseline (e.g.
    an explicit el1/el2 run where the baseline was analysed on the other
    electrode), the same tissue on any electrode is used, and the pairing
    record says so. In 'auto' batch mode every recording of a tissue is
    analysed on the same electrode, so this does not arise.

    Files whose name carries two tissues ('…chipC_ch1_chipA_ch1…') are not
    paired: each electrode is a different microtissue.

    Returns a dict: recording_key(result) → baseline_result (or None).
    When ``details`` is a dict it is filled with recording_key →
    {'baseline_file', 'reason', 'candidates'}.
    """
    groups = defaultdict(list)
    base_groups = defaultdict(list)
    for r in results_list:
        groups[_get_group_key(r)].append(r)
        base_groups[_get_base_key(r)].append(r)

    grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}

    def _grade(r):
        return grade_order.get(getattr(r.get('qc_report'), 'grade', 'F'), 5)

    def _usable(baselines):
        return [b for b in baselines
                if not (b.get('summary') or {}).get('not_analysable', False)]

    def _find_baseline(group_results):
        """Find baselines in a group, falling back to controls."""
        baselines = [r for r in group_results if _is_baseline(r)]
        if not baselines:
            controls = [r for r in group_results if _is_control(r)]
            if controls:
                baselines = controls[:1]
        return baselines

    def _choose(r, candidates):
        cands = _usable(candidates)
        if not cands:
            return None, ('no baseline for this tissue' if not candidates
                          else 'baseline not analysable')
        here = _folder(r)
        same = [b for b in cands if here is not None and _folder(b) == here]
        pool = same or cands
        best = min(pool, key=lambda b: (_grade(b), (b.get('metadata', {}) or {}).get('filename', '')))
        why = 'baseline in the same folder' if same else 'baseline in another folder of the same tissue'
        if len(pool) > 1:
            why += f', best QC grade of {len(pool)}'
        return best, why

    baseline_map = {}
    for key, group_results in groups.items():
        candidates = _find_baseline(group_results)
        other_electrode = False
        if not _usable(candidates):
            fallback = _find_baseline(base_groups.get(_get_base_key(group_results[0]), []))
            if _usable(fallback):
                candidates, other_electrode = fallback, True

        for r in group_results:
            rk = recording_key(r)
            fi = r.get('file_info', {}) or {}
            if _is_baseline(r) or _is_control(r):
                baseline_map[rk] = None
                continue
            if fi.get('dual_tissue'):
                bl, why = None, 'two tissues in one file (one per electrode): not paired automatically'
            else:
                bl, why = _choose(r, candidates)
                if bl is not None and not (bl.get('inclusion', {}) or {}).get('passed', True):
                    why = f"baseline failed inclusion ({(bl.get('inclusion', {}) or {}).get('reason', '')})"
                    bl = None
                elif bl is not None and other_electrode:
                    why += ' (baseline analysed on the other electrode)'
            baseline_map[rk] = bl
            if details is not None:
                details[rk] = {
                    'baseline_file': (bl.get('metadata', {}) or {}).get('filename', '') if bl is not None else '',
                    'reason': why,
                    'candidates': [(b.get('metadata', {}) or {}).get('filename', '') for b in candidates],
                }

    return baseline_map


def compute_normalized_parameters(result, baseline_result, cfg=None):
    """
    Compute percentage changes relative to baseline.

    Returns a dict with:
      - pct_bp_change: % change in BP
      - pct_fpdc_change: % change in FPDcF
      - pct_amp_change: % change in amplitude
      - baseline_bp_ms, baseline_fpdc_ms, baseline_amp_mV
      - fpdc_threshold_low/mid/high: bool flags
      - tdp_score: -1 to 3
    """
    norm = {
        'has_baseline': False,
        'baseline_file': '',
        'baseline_bp_ms': np.nan,
        'baseline_fpdc_ms': np.nan,
        'baseline_amp_mV': np.nan,
        'pct_bp_change': np.nan,
        'pct_fpdc_change': np.nan,
        'pct_amp_change': np.nan,
        'exceeds_LOW': False,
        'exceeds_MID': False,
        'exceeds_HIGH': False,
        'tdp_score': 0,
        # FPD reliability provenance — see ``norm_require_fpd_reliable`` in
        # config.py.  Always present so downstream consumers (reports, CDISC
        # export, UI badges) can surface the condition without having to
        # reach back into the two source summaries.
        'baseline_fpd_reliable': True,
        'drug_fpd_reliable': True,
        'fpd_reliable': True,
        # Why %ΔFPDcF was not computed although both recordings exist
        # (near cessation, see max_beat_period_for_fpdc_ms); '' otherwise.
        'fpdc_withheld': '',
    }

    if baseline_result is None:
        return norm

    # Get baseline values
    bl_summary = baseline_result.get('summary', {})
    bl_bp = bl_summary.get('beat_period_ms_mean')
    bl_fpdc = bl_summary.get('fpdc_ms_mean')
    bl_amp = bl_summary.get('spike_amplitude_mV_mean')

    # Get drug values
    dr_summary = result.get('summary', {})
    dr_bp = dr_summary.get('beat_period_ms_mean')
    dr_fpdc = dr_summary.get('fpdc_ms_mean')
    dr_amp = dr_summary.get('spike_amplitude_mV_mean')

    norm['has_baseline'] = True
    norm['baseline_file'] = baseline_result.get('metadata', {}).get('filename', '')

    # ── FPD reliability provenance ──
    # A %ΔFPDcF is only as trustworthy as the weaker of its two operands,
    # so the pair reliability is the AND of both.  Default True keeps
    # results produced before the reliability gate existed usable.
    bl_reliable = bool(bl_summary.get('fpd_reliable', True))
    dr_reliable = bool(dr_summary.get('fpd_reliable', True))
    norm['baseline_fpd_reliable'] = bl_reliable
    norm['drug_fpd_reliable'] = dr_reliable
    norm['fpd_reliable'] = bl_reliable and dr_reliable

    if not norm['fpd_reliable']:
        which = []
        if not bl_reliable:
            which.append('baseline')
        if not dr_reliable:
            which.append('drug')
        logger.warning(
            "%%ΔFPDcF computed from unreliable FPD (%s): %s vs baseline %s. "
            "Enable AnalysisConfig.norm_require_fpd_reliable to exclude such "
            "recordings from drug classification.",
            '+'.join(which),
            result.get('metadata', {}).get('filename', '?'),
            norm['baseline_file'] or '?',
        )

    # BP change
    if bl_bp and not np.isnan(bl_bp) and bl_bp > 0:
        norm['baseline_bp_ms'] = bl_bp
        if dr_bp and not np.isnan(dr_bp):
            norm['pct_bp_change'] = (dr_bp - bl_bp) / bl_bp * 100

    # Near-cessation guard: Fridericia divides FPD by RR^(1/3), so on a
    # tissue that has almost stopped (beating period of tens of seconds) the
    # corrected FPD is meaningless — +580 % for a cisapride recording at
    # RR = 40 s on the Visone 2023 data. BP and amplitude changes are kept.
    limit = getattr(cfg, 'max_beat_period_for_fpdc_ms', 6000.0) if cfg is not None else 6000.0
    slow = []
    for side, s in (('baseline', bl_summary), ('drug', dr_summary)):
        bp = s.get('beat_period_ms_median', s.get('beat_period_ms_mean'))
        if bp is not None and not np.isnan(bp) and bp > limit:
            slow.append(f"{side} {bp / 1000:.1f} s")
    if slow:
        norm['fpdc_withheld'] = (f"beating period above {limit / 1000:g} s ({', '.join(slow)}): "
                                 f"near cessation, FPDc not compared")

    # FPDcF change (the key metric for QT prolongation)
    if bl_fpdc and not np.isnan(bl_fpdc) and bl_fpdc > 0:
        norm['baseline_fpdc_ms'] = bl_fpdc
        if not slow and dr_fpdc and not np.isnan(dr_fpdc):
            pct = (dr_fpdc - bl_fpdc) / bl_fpdc * 100
            norm['pct_fpdc_change'] = pct

            # Threshold flags
            t_low, t_mid, t_high = _get_norm_thresholds(cfg)
            norm['exceeds_LOW'] = pct >= t_low
            norm['exceeds_MID'] = pct >= t_mid
            norm['exceeds_HIGH'] = pct >= t_high

    # AMP change
    if bl_amp and not np.isnan(bl_amp) and bl_amp > 0:
        norm['baseline_amp_mV'] = bl_amp
        if dr_amp and not np.isnan(dr_amp):
            norm['pct_amp_change'] = (dr_amp - bl_amp) / bl_amp * 100

    # TdP score
    norm['tdp_score'] = _compute_tdp_score(result, norm, cfg=cfg)

    return norm


def _compute_tdp_score(result, norm, cfg=None):
    """
    Compute TdP risk score (-1 to 3) based on FPDcF change and arrhythmia.

    Scoring (adapted from Ando et al. 2017 / Visone et al. 2023):
      -1: significant shortening (< -LOW threshold)
       0: no effect (|change| < LOW)
       1: prolongation above LOW but below MID
       2: prolongation above MID but below HIGH
       3: prolongation above HIGH, OR confirmed proarrhythmic events, OR cessation

    Arrhythmia events only upgrade the score if they are severe (cessation,
    EADs with FPDcF already trending up, or confirmed premature beats).
    Simple beat-rate irregularity is NOT counted — the arrhythmia module
    flags too many benign recordings.
    """
    pct = norm.get('pct_fpdc_change', np.nan)

    if np.isnan(pct):
        return 0

    # Check for severe arrhythmic events only
    ar = result.get('arrhythmia_report')
    has_cessation = False
    has_severe_arrhythmia = False  # EAD + prolongation, or cessation
    if ar:
        for flag in ar.flags:
            ftype = flag.get('type', '')
            severity = flag.get('severity', '')
            if ftype == 'cessation' or ftype == 'beat_cessation':
                has_cessation = True
            # Only count EADs and premature beats as proarrhythmic
            # if FPDcF is also trending upward (confirming drug effect)
            if ftype in ('ead_events',) and severity == 'critical':
                if pct > 0:  # Must show some prolongation trend
                    has_severe_arrhythmia = True

    # Also check cessation detection module (more robust)
    cess = result.get('cessation_report')
    if cess is not None and cess.has_cessation and cess.cessation_confidence > 0.5:
        has_cessation = True

    # Score based on FPDcF change (primary criterion)
    t_low, t_mid, t_high = _get_norm_thresholds(cfg)
    if has_cessation or pct >= t_high or has_severe_arrhythmia and pct >= t_low:
        return 3
    elif pct >= t_mid:
        return 2
    elif pct >= t_low:
        return 1
    elif pct <= -t_low:
        return -1
    else:
        return 0


def classify_drug(results_list, cfg=None):
    """
    Classify each drug as positive/negative for QT prolongation.

    Groups drug recordings by drug name (across all chips/channels/concentrations)
    and applies the classification method from config:
      - 'max'     : positive if ANY concentration exceeds threshold (default, most sensitive)
      - 'mean'    : positive if MEAN %FPDcF change exceeds threshold (reduces borderline FPs)
      - 'n_above' : positive if ≥ N concentrations exceed threshold (strict)

    Returns dict: drug_name → {
        'positive': bool,
        'method': str,
        'max_pct_change': float,
        'mean_pct_change': float,
        'n_above': int,
        'concentrations': list of (conc, pct_change),
        'threshold_used': float,
    }
    """
    if cfg is None:
        from .config import NormalizationConfig
        cfg = NormalizationConfig()

    t_low, t_mid, t_high = _get_norm_thresholds(cfg)
    threshold_map = {'low': t_low, 'mid': t_mid, 'high': t_high}
    threshold = threshold_map.get(cfg.classification_threshold, t_mid)

    # ── QC / CV filters for drug-level classification ──
    grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
    qc_filter_on = getattr(cfg, 'norm_min_qc_enabled', False)
    qc_min_grade = getattr(cfg, 'norm_min_qc_grade', 'D')
    qc_min_rank = grade_order.get(qc_min_grade, 3)

    cv_filter_on = getattr(cfg, 'norm_max_cv_enabled', False)
    cv_max = getattr(cfg, 'norm_max_cv_bp', 50.0)

    fpd_filter_on = getattr(cfg, 'norm_require_fpd_reliable', False)

    # Group by drug — collect FPDcF data for classification
    drug_data = defaultdict(list)
    n_qc_excluded = 0
    n_fpd_excluded = 0
    n_na_excluded = 0
    for r in results_list:
        if _is_baseline(r) or _is_control(r) or is_washout(r):
            continue
        if canonical_drug_name((r.get('file_info', {}) or {}).get('drug')) in VEHICLE_NAMES:
            continue   # vehicle: %Δ kept in the normalisation table, not a drug call
        norm = r.get('normalization', {})
        if not norm.get('has_baseline'):
            continue
        # Analysability verdict is a hard exclusion (not a quality preference):
        # the recording has no depolarisation pattern distinguishable from
        # noise, so it cannot contribute a %ΔFPDcF.
        if (r.get('summary') or {}).get('not_analysable', False):
            n_na_excluded += 1
            continue
        inc = r.get('inclusion', {})
        if not inc.get('passed', True):
            continue

        fi = r.get('file_info', {})
        drug = canonical_drug_name(fi.get('drug'))
        conc = fi.get('concentration', '')
        pct = norm.get('pct_fpdc_change', np.nan)

        # Apply QC grade filter
        if qc_filter_on:
            qc = r.get('qc_report')
            qc_grade = getattr(qc, 'grade', 'F') if qc else 'F'
            if grade_order.get(qc_grade, 4) > qc_min_rank:
                n_qc_excluded += 1
                continue

        # Apply CV filter
        if cv_filter_on:
            bp = r.get('beat_periods', [])
            if len(bp) > 1:
                cv_val = (np.std(bp) / np.mean(bp) * 100) if np.mean(bp) > 0 else 999
                if cv_val > cv_max:
                    n_qc_excluded += 1
                    continue

        # Apply FPD reliability filter (opt-in).
        # ``fpd_reliable`` is the AND of the drug and baseline recordings —
        # a %ΔFPDcF is only as trustworthy as the weaker operand. Recordings
        # analysed before this key existed default to True.
        if fpd_filter_on and not norm.get('fpd_reliable', True):
            n_fpd_excluded += 1
            logger.info(
                "Excluded from classification (FPD unreliable): %s",
                r.get('metadata', {}).get('filename', '?'),
            )
            continue

        if drug and not np.isnan(pct):
            drug_data[drug].append({
                'concentration': conc,
                'pct_fpdc_change': pct,
                'tdp_score': norm.get('tdp_score', 0),
            })

    # Surface how many recordings the opt-in filters removed. Without this
    # an enabled filter silently shrinks the classification denominator,
    # which is exactly the kind of change that must be visible in a log.
    if n_qc_excluded or n_fpd_excluded or n_na_excluded:
        logger.info(
            "Classification filters excluded %d recording(s): "
            "%d by QC/CV, %d by FPD reliability, %d not analysable.",
            n_qc_excluded + n_fpd_excluded + n_na_excluded,
            n_qc_excluded, n_fpd_excluded, n_na_excluded,
        )

    # Collect cessation data per drug (from ALL drug recordings, not just those
    # with valid FPD — the whole point is to catch drugs that destroy waveforms)
    drug_cessation = defaultdict(lambda: {'has_cessation': False, 'min_fpd_conf': 1.0,
                                           'cessation_details': []})
    enable_cess = getattr(cfg, 'enable_cessation_override', True)
    cess_max_conf = getattr(cfg, 'cessation_override_max_fpd_confidence', 0.60)

    if enable_cess:
        for r in results_list:
            if _is_baseline(r) or _is_control(r) or is_washout(r):
                continue
            fi = r.get('file_info', {})
            drug = canonical_drug_name(fi.get('drug'))
            if not drug or drug in VEHICLE_NAMES:
                continue

            # Check cessation
            cess = r.get('cessation_report')
            if cess is not None and cess.has_cessation and cess.cessation_confidence > 0.5:
                drug_cessation[drug]['has_cessation'] = True
                drug_cessation[drug]['cessation_details'].append({
                    'concentration': fi.get('concentration', ''),
                    'type': cess.cessation_type,
                    'confidence': cess.cessation_confidence,
                })

            # Track min FPD confidence across all concentrations
            summary = r.get('summary', {})
            fpd_conf = summary.get('fpd_confidence', 1.0)
            if fpd_conf is not None and not np.isnan(fpd_conf):
                drug_cessation[drug]['min_fpd_conf'] = min(
                    drug_cessation[drug]['min_fpd_conf'], fpd_conf)

            # Track spectral morphology change (from normalization step)
            norm = r.get('normalization', {})
            spec_score = norm.get('spectral_change_score', np.nan)
            if spec_score is not None and not np.isnan(spec_score):
                if 'spectral_scores' not in drug_cessation[drug]:
                    drug_cessation[drug]['spectral_scores'] = []
                drug_cessation[drug]['spectral_scores'].append(spec_score)

    # Classify each drug
    classifications = {}
    for drug, entries in drug_data.items():
        pct_values = [e['pct_fpdc_change'] for e in entries]
        conc_list = [(e['concentration'], e['pct_fpdc_change']) for e in entries]

        max_pct = max(pct_values)
        mean_pct = np.mean(pct_values)
        n_above = sum(1 for p in pct_values if p >= threshold)

        if cfg.classification_method == 'mean':
            positive = mean_pct >= threshold
        elif cfg.classification_method == 'n_above':
            positive = n_above >= cfg.classification_n_above
        else:  # 'max' (default)
            positive = max_pct >= threshold

        # Smart cessation override: if the drug causes cessation AND waveform
        # destruction (low FPD confidence), elevate to positive.
        # This catches drugs like dofetilide that destroy waveform morphology
        # so FPD can't be measured, but does NOT trigger for drugs like
        # ranolazine that have cessation at extreme doses with intact FPD.
        cessation_override = False
        cess_info = drug_cessation.get(drug, {})
        if (enable_cess and cess_info.get('has_cessation', False)
                and cess_info.get('min_fpd_conf', 1.0) < cess_max_conf):
            cessation_override = True
            positive = True

        # Spectral morphology change summary for this drug
        spec_scores = cess_info.get('spectral_scores', [])
        max_spec = max(spec_scores) if spec_scores else np.nan
        mean_spec = np.mean(spec_scores) if spec_scores else np.nan

        classifications[drug] = {
            'positive': positive,
            'method': cfg.classification_method,
            'max_pct_change': max_pct,
            'mean_pct_change': mean_pct,
            'n_above_threshold': n_above,
            'n_concentrations': len(pct_values),
            'concentrations': conc_list,
            'threshold_used': threshold,
            'threshold_name': cfg.classification_threshold,
            'cessation_override': cessation_override,
            'cessation_info': cess_info if cessation_override else {},
            'max_spectral_change': max_spec,
            'mean_spectral_change': mean_spec,
        }

    # Also check for drugs that ONLY have cessation (no valid FPD data at all)
    # but were detected in the cessation scan
    if enable_cess:
        for drug, cess_info in drug_cessation.items():
            if drug not in classifications and cess_info['has_cessation']:
                if cess_info['min_fpd_conf'] < cess_max_conf:
                    classifications[drug] = {
                        'positive': True,
                        'method': 'cessation_only',
                        'max_pct_change': np.nan,
                        'mean_pct_change': np.nan,
                        'n_above_threshold': 0,
                        'n_concentrations': 0,
                        'concentrations': [],
                        'threshold_used': threshold,
                        'threshold_name': cfg.classification_threshold,
                        'cessation_override': True,
                        'cessation_info': cess_info,
                    }

    return classifications


def normalize_all_results(results_list, cfg=None):
    """
    Run baseline normalization on all results.

    Adds 'normalization' key to each result dict.
    Also computes spectral comparison vs baseline (spectral_change_score).
    Runs drug-level classification and adds 'drug_classification' to each drug result.
    Returns the modified results_list.
    """
    pairing = {}
    baseline_map = pair_with_baselines(results_list, details=pairing)

    # Lazy import to avoid circular dependency
    try:
        from .spectral import (
            SpectralConfig,
            _compare_with_baseline,
            compute_morphology_change_score,
        )
    except ImportError:
        compute_morphology_change_score = None

    for r in results_list:
        rk = recording_key(r)
        bl = baseline_map.get(rk)

        if bl is not None:
            r['normalization'] = compute_normalized_parameters(r, bl, cfg=cfg)

            # ─── Spectral comparison vs baseline ───
            if compute_morphology_change_score is not None:
                bl_spec = bl.get('spectral_report')
                dr_spec = r.get('spectral_report')
                if bl_spec is not None and dr_spec is not None:
                    # Run PSD comparison if not already done (baseline_spectral
                    # wasn't available during single-file analysis)
                    if np.isnan(dr_spec.spectral_correlation):
                        bl_freqs = bl_spec.details.get('freqs')
                        dr_freqs = dr_spec.details.get('freqs')
                        dr_psd = dr_spec.details.get('psd')
                        if bl_freqs is not None and dr_freqs is not None and dr_psd is not None:
                            _compare_with_baseline(
                                dr_freqs, dr_psd, bl_spec, SpectralConfig(), dr_spec)

                    score = compute_morphology_change_score(dr_spec, bl_spec)
                    r['normalization']['spectral_change_score'] = score
                else:
                    r['normalization']['spectral_change_score'] = np.nan
        else:
            r['normalization'] = {
                'has_baseline': False,
                'baseline_file': '',
                'baseline_bp_ms': np.nan,
                'baseline_fpdc_ms': np.nan,
                'baseline_amp_mV': np.nan,
                'pct_bp_change': np.nan,
                'pct_fpdc_change': np.nan,
                'pct_amp_change': np.nan,
                'exceeds_LOW': False,
                'exceeds_MID': False,
                'exceeds_HIGH': False,
                'tdp_score': 0,
                'spectral_change_score': np.nan,
                'fpdc_withheld': '',
            }
        if rk in pairing:
            r['normalization']['pairing'] = pairing[rk]

    # Drug-level classification
    drug_cls = classify_drug(results_list, cfg=cfg)

    # Annotate each result with its drug classification
    for r in results_list:
        fi = r.get('file_info', {})
        drug = canonical_drug_name(fi.get('drug'))
        if drug in drug_cls and not is_washout(r):
            r['normalization']['drug_classification'] = drug_cls[drug]

    return results_list
