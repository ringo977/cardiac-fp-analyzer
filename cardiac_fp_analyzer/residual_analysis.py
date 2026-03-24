"""
residual_analysis.py — Residual-based arrhythmia analysis (Visone et al. 2023).

For each beat, compute residual = beat − template and analyse for:
  - Morphological instability (RMS of residuals)
  - EAD detection (peaks in the repolarisation window of the residual)
  - Repolarisation variability (Poincaré STV of FPD values)
  - Triggered-activity bursts (clusters of abnormal residuals)

Extracted from arrhythmia.py to keep modules under ~450 lines.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#   TEMPLATE & RESIDUAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_template(beats_data, max_beats=50):
    """Robust median template from up to *max_beats* beats."""
    if not beats_data or len(beats_data) == 0:
        return None
    n = min(len(beats_data), max_beats)
    # Use beats from the middle of the recording (most stable region)
    mid = len(beats_data) // 2
    start = max(0, mid - n // 2)
    sel = beats_data[start:start + n]
    min_len = min(len(b) for b in sel)
    if min_len < 20:
        return None
    aligned = np.array([b[:min_len] for b in sel])
    return np.median(aligned, axis=0)


def compute_residuals(beats_data, template):
    """Compute residual = beat − template for each beat.

    Returns list of residual arrays (same length as template).
    """
    tlen = len(template)
    residuals = []
    for b in beats_data:
        n = min(len(b), tlen)
        res = b[:n] - template[:n]
        residuals.append(res)
    return residuals


def residual_rms(residuals):
    """Per-beat RMS of residual (morphological instability)."""
    return np.array([np.sqrt(np.mean(r ** 2)) for r in residuals])


# ═══════════════════════════════════════════════════════════════════════
#   EAD DETECTION FROM RESIDUALS
# ═══════════════════════════════════════════════════════════════════════

def detect_ead_from_residual(residual, fs, template_range=None,
                              template=None,
                              repol_start_ms=150, repol_end_ms=500,
                              min_prominence_factor=6.0,
                              min_amplitude_frac=0.08,
                              min_width_ms=8.0,
                              max_width_ms=150.0):
    """Detect EAD-like bumps in the repolarisation window of the residual.

    An EAD manifests as a **positive** deflection (secondary depolarisation)
    in the residual during the repolarisation phase.  Phase-2 EADs occur
    during the AP plateau (150–400 ms post-spike); phase-3 EADs slightly
    later.  Non-EAD drug effects (Na block shortening, Ca block morphology
    changes) tend to produce broad, symmetric shifts — NOT sharp humps.

    Detection criteria (ALL must be met):
      1. **Statistical** : peak > prominence_factor × σ  (noise floor)
      2. **Absolute**    : peak > amp_frac × template peak-to-peak range
      3. **Width**       : half-max width ∈ [min_width, max_width] ms
         (too narrow → noise; too wide → global shape change, not EAD)
      4. **Polarity**    : peak must be positive (depolarising bump)
      5. **Location**    : within the plateau/repolarisation window
         (150–500 ms), avoiding the spike region where residuals are
         dominated by depolarisation alignment jitter.

    Parameters
    ----------
    residual : 1-D array
    fs : float — sampling rate
    template_range : float or None — peak-to-peak amplitude of template.
    template : 1-D array or None — reserved for future refinement.
    repol_start_ms, repol_end_ms : detection window (ms post-alignment).
    min_prominence_factor : float — peak > N × σ of full residual.
    min_amplitude_frac : float — peak > frac × template amplitude.
    min_width_ms : float — minimum half-max width (ms).
    max_width_ms : float — maximum half-max width (ms); wider peaks
        are considered global morphology shifts, not focal EADs.

    Returns
    -------
    list of dict with keys: sample_idx, time_ms, amplitude, width_ms
    """
    s = int(repol_start_ms * fs / 1000)
    e = min(int(repol_end_ms * fs / 1000), len(residual))
    if e - s < 20:
        return []

    seg = residual[s:e]

    # Noise estimate: MAD → σ, from the FULL residual (not just the
    # repol segment — avoids circular bias in noisy residuals)
    full_mad = np.median(np.abs(residual - np.median(residual)))
    if full_mad < 1e-12:
        return []
    sigma = full_mad * 1.4826
    stat_threshold = min_prominence_factor * sigma

    # Absolute amplitude threshold (% of template range)
    abs_threshold = 0.0
    if template_range is not None and template_range > 0:
        abs_threshold = min_amplitude_frac * template_range

    # Combined threshold: must exceed BOTH
    threshold = max(stat_threshold, abs_threshold)

    # Width bounds in samples
    min_width_samp = max(3, int(min_width_ms * fs / 1000))
    max_width_samp = int(max_width_ms * fs / 1000)

    # Peak detection: local maxima above threshold with width check
    eads = []
    i = 1
    while i < len(seg) - 1:
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > threshold:
            # Check peak width at half-maximum
            half_max = seg[i] / 2.0
            # Scan left
            left = i
            while left > 0 and seg[left] > half_max:
                left -= 1
            # Scan right
            right = i
            while right < len(seg) - 1 and seg[right] > half_max:
                right += 1
            width = right - left

            if min_width_samp <= width <= max_width_samp:
                width_ms = width / fs * 1000
                eads.append({
                    'sample_idx': s + i,
                    'time_ms': (s + i) / fs * 1000,
                    'amplitude': float(seg[i]),
                    'width_ms': float(width_ms),
                })
                # Skip past this peak to avoid double-counting
                i = right + 1
                continue
        i += 1

    return eads


# ═══════════════════════════════════════════════════════════════════════
#   POINCARÉ STV
# ═══════════════════════════════════════════════════════════════════════

def poincare_stv(values):
    """Short-term variability from Poincaré plot.

    STV = mean |x_{i+1} - x_i| / √2

    Standard measure of beat-to-beat repolarisation instability
    (Thomsen et al. 2004, Hondeghem et al. 2001).
    """
    if len(values) < 3:
        return np.nan
    v = np.array(values, dtype=float)
    diffs = np.abs(np.diff(v))
    return float(np.mean(diffs) / np.sqrt(2))


# ═══════════════════════════════════════════════════════════════════════
#   MAIN RESIDUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze_residual(beats_data, fs, all_params, cfg=None,
                     baseline_template=None):
    """Residual-based arrhythmia analysis (Visone et al. 2023 approach).

    Parameters
    ----------
    beats_data : list of 1-D arrays — individual beat segments (aligned
        to spike peak).
    fs : float — sampling rate (Hz).
    all_params : list of dict — per-beat parameters from extract_all_parameters.
    cfg : ArrhythmiaConfig or None.
    baseline_template : 1-D array or None — if provided, use this as the
        reference template instead of computing one from the current recording.
        This implements the paper's baseline-relative residual analysis:
        residual = drug_beat − baseline_template, which captures drug-induced
        morphology changes rather than just beat-to-beat jitter.

    Returns
    -------
    dict with keys:
        template : 1-D array or None
        residual_rms : 1-D array — per-beat RMS of residual
        mean_residual_rms : float
        morphology_instability : float — normalised 0-1 score
        ead_beats : list of (beat_index, [ead_events])
        n_ead_beats : int — number of beats with ≥1 EAD
        ead_incidence_pct : float — % of beats with EAD
        poincare_stv_fpd : float — STV of FPD (ms)
        poincare_stv_fpdc : float — STV of FPDcF (ms)
        baseline_relative : bool — True if baseline template was used
    """
    if cfg is None:
        from .config import ArrhythmiaConfig
        cfg = ArrhythmiaConfig()

    result = {
        'template': None,
        'residual_rms': np.array([]),
        'mean_residual_rms': np.nan,
        'morphology_instability': 0.0,
        'ead_beats': [],
        'n_ead_beats': 0,
        'ead_incidence_pct': 0.0,
        'poincare_stv_fpd': np.nan,
        'poincare_stv_fpdc': np.nan,
        'baseline_relative': False,
    }

    if not beats_data or len(beats_data) < 5:
        return result

    # ── Template ──
    # If a baseline template is provided, use it (baseline-relative mode).
    # Otherwise, compute from the current recording (intra-recording mode).
    if baseline_template is not None:
        template = baseline_template
        result['baseline_relative'] = True
    else:
        template = compute_template(beats_data)
    if template is None:
        return result
    result['template'] = template

    # ── Residuals ──
    residuals = compute_residuals(beats_data, template)
    rms_vals = residual_rms(residuals)
    result['residual_rms'] = rms_vals
    result['mean_residual_rms'] = float(np.mean(rms_vals))

    # Morphology instability: normalise RMS by template amplitude range
    template_range = np.ptp(template)
    if template_range > 0:
        # Fraction of template amplitude that the residual represents
        norm_rms = np.mean(rms_vals) / template_range
        # Sigmoid mapping: 5% → ~0.12, 15% → ~0.5, 30% → ~0.88
        result['morphology_instability'] = float(
            1.0 / (1.0 + np.exp(-15.0 * (norm_rms - 0.15)))
        )

    # ── EAD detection from residuals ──
    ead_factor = getattr(cfg, 'ead_residual_prominence', 6.0)
    ead_min_amp = getattr(cfg, 'ead_residual_min_amp_frac', 0.08)
    ead_min_width = getattr(cfg, 'ead_residual_min_width_ms', 8.0)
    ead_max_width = getattr(cfg, 'ead_residual_max_width_ms', 150.0)
    ead_beats = []
    for i, res in enumerate(residuals):
        eads = detect_ead_from_residual(
            res, fs,
            template_range=template_range,
            template=template,
            min_prominence_factor=ead_factor,
            min_amplitude_frac=ead_min_amp,
            min_width_ms=ead_min_width,
            max_width_ms=ead_max_width,
        )
        if eads:
            ead_beats.append((i, eads))
    result['ead_beats'] = ead_beats
    result['n_ead_beats'] = len(ead_beats)
    result['ead_incidence_pct'] = (
        len(ead_beats) / len(beats_data) * 100 if beats_data else 0.0
    )

    # ── Poincaré STV of repolarisation ──
    fpd_vals = [p.get('fpd_ms', np.nan) for p in all_params]
    fpd_valid = [v for v in fpd_vals if not np.isnan(v)]
    result['poincare_stv_fpd'] = poincare_stv(fpd_valid)

    fpdc_vals = [p.get('fpdc_ms', np.nan) for p in all_params]
    fpdc_valid = [v for v in fpdc_vals if not np.isnan(v)]
    result['poincare_stv_fpdc'] = poincare_stv(fpdc_valid)

    return result


# ── Back-compat aliases (private names used in arrhythmia.py) ──
_compute_template = compute_template
_compute_residuals = compute_residuals
_residual_rms = residual_rms
_detect_ead_from_residual = detect_ead_from_residual
_poincare_stv = poincare_stv
