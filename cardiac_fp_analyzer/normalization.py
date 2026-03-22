"""
normalization.py — Baseline normalization and TdP risk scoring.

After batch analysis, pairs each drug recording with its baseline
(same experiment + chip + channel) and computes:
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

import numpy as np
from collections import defaultdict


# ─── Thresholds (module-level defaults, overridden by NormalizationConfig) ───
THRESHOLD_LOW = 10.0    # %FPDcF change ≥ 10%
THRESHOLD_MID = 15.0    # %FPDcF change ≥ 15% (paper's optimal)
THRESHOLD_HIGH = 20.0   # %FPDcF change ≥ 20%


def _get_norm_thresholds(cfg=None):
    """Return (LOW, MID, HIGH) thresholds from config or module defaults."""
    if cfg is None:
        return THRESHOLD_LOW, THRESHOLD_MID, THRESHOLD_HIGH
    return cfg.threshold_low, cfg.threshold_mid, cfg.threshold_high


def _get_group_key(result):
    """
    Extract the grouping key (experiment + chip + chamber + electrode) from a result.
    E.g. chipA_ch1_terfe_300nM analyzed on el1 → group = "EXP 5/chipA_ch1/el1"

    The group key includes the electrode (el1/el2) so that dual-electrode
    analyses of the same file are kept in separate normalization groups.
    """
    fi = result.get('file_info', {})
    exp = fi.get('experiment', '')
    chip = fi.get('chip', '')
    channel_label = fi.get('channel_label', '')  # e.g. ch1, ch2, ch3 (chamber from filename)
    electrode = fi.get('analyzed_channel', '')     # e.g. el1, el2 (electrode)

    # Also extract from filename
    meta = result.get('metadata', {})
    fname = meta.get('filename', '')

    # Parse chip_chamber from filename (e.g. chipA_ch1_terfe_300nM → chipA_ch1)
    parts = fname.split('_')
    if len(parts) >= 2 and parts[0].startswith('chip'):
        chip_ch = f"{parts[0]}_{parts[1]}"
    else:
        chip_ch = f"chip{chip}_{channel_label}" if chip and channel_label else fname

    key = f"{exp}/{chip_ch}"
    if electrode:
        key = f"{key}/{electrode}"
    return key


def _is_baseline(result):
    """Check if a result is a baseline recording."""
    fi = result.get('file_info', {})
    drug = str(fi.get('drug', '') or '').lower()
    fname = str(result.get('metadata', {}).get('filename', '')).lower()

    return ('baseline' in drug or 'basline' in drug or
            'baseline' in fname or 'basline' in fname)


def _is_control(result):
    """Check if a result is a control recording (CTRL/CTR)."""
    fi = result.get('file_info', {})
    drug = str(fi.get('drug', '') or '').lower()
    return drug.startswith('ctrl') or drug.startswith('ctr')


def pair_with_baselines(results_list):
    """
    Pair each drug recording with its baseline.

    Groups results by experiment + chip + channel, finds the baseline
    in each group, and pairs drug recordings with it.

    Returns a dict: filename → baseline_result (or None if no baseline found).
    """
    # Group results
    groups = defaultdict(list)
    for r in results_list:
        key = _get_group_key(r)
        groups[key].append(r)

    # For each group, find baseline and pair
    baseline_map = {}  # filename → baseline result

    for key, group_results in groups.items():
        # Find baseline(s) in this group
        baselines = [r for r in group_results if _is_baseline(r)]

        if not baselines:
            # No baseline — check if controls exist
            controls = [r for r in group_results if _is_control(r)]
            if controls:
                baselines = controls[:1]  # Use first control as pseudo-baseline

        if not baselines:
            # No baseline found for this group
            for r in group_results:
                fname = r.get('metadata', {}).get('filename', '')
                baseline_map[fname] = None
            continue

        # Use the first baseline (or the one with best QC grade)
        best_bl = baselines[0]
        if len(baselines) > 1:
            # Prefer the one with better QC grade
            grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
            best_bl = min(baselines,
                         key=lambda r: grade_order.get(
                             r.get('qc_report', type('', (), {'grade': 'F'})()).grade, 5))

        # Check if baseline passes inclusion criteria
        bl_inclusion = best_bl.get('inclusion', {})
        bl_passed = bl_inclusion.get('passed', True)  # default: pass if no criteria applied

        for r in group_results:
            fname = r.get('metadata', {}).get('filename', '')
            if _is_baseline(r) or _is_control(r):
                baseline_map[fname] = None  # baselines don't have a reference
            elif not bl_passed:
                baseline_map[fname] = None  # baseline excluded → skip normalization
            else:
                baseline_map[fname] = best_bl

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

    # BP change
    if bl_bp and not np.isnan(bl_bp) and bl_bp > 0:
        norm['baseline_bp_ms'] = bl_bp
        if dr_bp and not np.isnan(dr_bp):
            norm['pct_bp_change'] = (dr_bp - bl_bp) / bl_bp * 100

    # FPDcF change (the key metric for QT prolongation)
    if bl_fpdc and not np.isnan(bl_fpdc) and bl_fpdc > 0:
        norm['baseline_fpdc_ms'] = bl_fpdc
        if dr_fpdc and not np.isnan(dr_fpdc):
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
    if cess is not None and cess.has_cessation:
        if cess.cessation_confidence > 0.5:
            has_cessation = True

    # Score based on FPDcF change (primary criterion)
    t_low, t_mid, t_high = _get_norm_thresholds(cfg)
    if has_cessation:
        return 3
    elif pct >= t_high:
        return 3
    elif has_severe_arrhythmia and pct >= t_low:
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

    # Group by drug — collect FPDcF data for classification
    drug_data = defaultdict(list)
    for r in results_list:
        if _is_baseline(r) or _is_control(r):
            continue
        norm = r.get('normalization', {})
        if not norm.get('has_baseline'):
            continue
        inc = r.get('inclusion', {})
        if not inc.get('passed', True):
            continue

        fi = r.get('file_info', {})
        drug = str(fi.get('drug', '') or '').lower()
        conc = fi.get('concentration', '')
        pct = norm.get('pct_fpdc_change', np.nan)

        if drug and not np.isnan(pct):
            drug_data[drug].append({
                'concentration': conc,
                'pct_fpdc_change': pct,
                'tdp_score': norm.get('tdp_score', 0),
            })

    # Collect cessation data per drug (from ALL drug recordings, not just those
    # with valid FPD — the whole point is to catch drugs that destroy waveforms)
    drug_cessation = defaultdict(lambda: {'has_cessation': False, 'min_fpd_conf': 1.0,
                                           'cessation_details': []})
    enable_cess = getattr(cfg, 'enable_cessation_override', True)
    cess_max_conf = getattr(cfg, 'cessation_override_max_fpd_confidence', 0.60)

    if enable_cess:
        for r in results_list:
            if _is_baseline(r) or _is_control(r):
                continue
            fi = r.get('file_info', {})
            drug = str(fi.get('drug', '') or '').lower()
            if not drug:
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
    baseline_map = pair_with_baselines(results_list)

    # Lazy import to avoid circular dependency
    try:
        from .spectral import compute_morphology_change_score, _compare_with_baseline, SpectralConfig
    except ImportError:
        compute_morphology_change_score = None

    for r in results_list:
        fname = r.get('metadata', {}).get('filename', '')
        bl = baseline_map.get(fname)

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
            }

    # Drug-level classification
    drug_cls = classify_drug(results_list, cfg=cfg)

    # Annotate each result with its drug classification
    for r in results_list:
        fi = r.get('file_info', {})
        drug = str(fi.get('drug', '') or '').lower()
        if drug in drug_cls:
            r['normalization']['drug_classification'] = drug_cls[drug]

    return results_list
