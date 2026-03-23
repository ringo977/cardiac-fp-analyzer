"""
arrhythmia.py — Arrhythmia detection and classification for hiPSC-CM FP.

Two complementary approaches:

  1. **Statistical** (original): beat-period variability, STV, premature/
     delayed beat detection, amplitude instability, FPD outlier EADs.

  2. **Residual-based** (paper approach — Visone et al. 2023): for each
     beat, compute residual = beat − template.  Analyse the residual for:
       - Morphological instability (RMS of residuals)
       - EAD detection (peaks in the repolarisation window of the residual)
       - Repolarisation variability (Poincaré STV of FPD values)
       - Triggered-activity bursts (clusters of abnormal residuals)

Both approaches contribute to a single ArrhythmiaReport with a 0–100
risk score.
"""

import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Module-level constants (legacy, kept for backward compat) ──
TACHYCARDIA_BP_MS = 300
BRADYCARDIA_BP_MS = 2500
RR_IRREGULARITY_CV = 15.0
PREMATURE_BEAT_FACTOR = 0.7
DELAYED_BEAT_FACTOR = 1.5
CESSATION_FACTOR = 3.0
STV_HIGH_RISK_MS = 10.0
FPD_PROLONGATION_THRESHOLD = 1.3


class ArrhythmiaReport:
    def __init__(self):
        self.flags = []
        self.events = []
        self.classification = 'Normal Sinus Rhythm'
        self.risk_score = 0
        self.details = {}
        # Residual-analysis results (populated if beats_data available)
        self.residual_details = {}

    def add_flag(self, flag_type, severity, description):
        """Record a diagnostic flag. Risk score is computed separately
        at the end via compute_risk_score() using incidence-based metrics,
        not accumulated per-flag (Blinova 2017, Hondeghem 2001)."""
        self.flags.append({'type': flag_type, 'severity': severity,
                           'description': description})

    def add_event(self, beat_number, event_type, details=''):
        self.events.append({'beat': beat_number, 'type': event_type,
                            'details': details})

    def compute_risk_score(self, mode='manual'):
        """Compute risk score (0-100) from incidence-normalised metrics.

        Parameters
        ----------
        mode : str — 'manual' or 'data_driven'
            'manual': expert-assigned weights based on literature
            'data_driven': weights from logistic regression on CiPA dataset

        Literature basis:
          - CV of beat period (Thomsen 2004): beat-to-beat variability
          - EAD incidence % (Visone 2023): fraction of beats with EADs
          - Morphology instability (Visone 2023): 0-1, normalised by
            template amplitude
          - Amplitude instability (Visone 2023): CV of spike amplitude
            — captures drug-induced depolarisation changes (hERG block,
            Ca²⁺ block, progressive tissue degradation)
          - Poincaré STV (Hondeghem 2001): ms, inherently per-beat
          - Cessation: binary flag with confidence

        All metrics are rate-based or normalised, so a 30-second
        recording with 2/15 premature beats (13%) correctly scores
        higher than a 5-minute recording with 2/150 (1.3%).
        """
        d = self.details
        n_beats = d.get('n_beats', 0)
        if n_beats < 3:
            self.risk_score = 0
            return

        if mode == 'data_driven':
            self._compute_risk_score_data_driven(d, n_beats)
        else:
            self._compute_risk_score_manual(d, n_beats)

    def _compute_risk_score_manual(self, d, n_beats):
        """Manual (expert) weights — default mode.

        Component weights (sum=100):
          CV beat period:     20  (Thomsen 2004)
          Premature/delayed:  10  (Hondeghem 2001)
          Morphology:         20  (Visone 2023)
          EAD incidence:      20  (Visone 2023)
          Amplitude CV:       10  (Visone 2023)
          Poincaré STV:       10  (Hondeghem 2001)
          Cessation:          10  (binary)
        """
        score = 0.0

        # 1. Rhythm irregularity: CV of beat period (0-20)
        cv_bp = d.get('cv_bp_pct', 0)
        if cv_bp > 10:
            score += min(20, (cv_bp - 10) / 30 * 20)

        # 2. Premature/delayed beat incidence (0-10)
        n_prem = d.get('n_premature', 0)
        n_del = d.get('n_delayed', 0)
        prem_pct = (n_prem + n_del) / n_beats * 100
        score += min(10, prem_pct * 1.0)

        # 3. Morphology instability (0-1 → 0-20)
        morph = d.get('morphology_instability', 0)
        score += morph * 20

        # 4. EAD incidence (% → 0-20)
        ead_pct = d.get('ead_incidence_pct', 0)
        score += min(20, ead_pct * 2.0)

        # 5. Amplitude instability (CV → 0-10)
        amp_cv = d.get('amplitude_cv_pct', 0)
        if amp_cv > 10:
            score += min(10, (amp_cv - 10) / 30 * 10)

        # 6. Poincaré STV of FPDcF (ms → 0-10)
        stv = d.get('poincare_stv_fpdc_ms', 0)
        if not np.isnan(stv) and stv > 5:
            score += min(10, (stv - 5) / 15 * 10)

        # 7. Cessation / pauses (binary → 0-10)
        if d.get('has_pauses', False):
            score += 10

        self.risk_score = int(min(100, round(score)))

    def _compute_risk_score_data_driven(self, d, n_beats):
        """Data-driven weights from CiPA logistic regression.

        Uses the fitted model to compute P(proarrhythmic) and maps
        the probability to a 0-100 score.

        If fitted_weights.json is not available, falls back to manual.
        """
        import json
        from pathlib import Path

        weights_path = Path(__file__).parent / 'fitted_weights.json'
        if not weights_path.exists():
            # Fallback to manual if no fitted weights
            self._compute_risk_score_manual(d, n_beats)
            return

        try:
            with open(weights_path) as f:
                w = json.load(f)

            coefs = np.array(w['coef_original_scale'])
            intercept = w['intercept_original_scale']

            # Extract features in the same order as training
            n_prem = d.get('n_premature', 0)
            n_del = d.get('n_delayed', 0)
            prem_pct = (n_prem + n_del) / n_beats * 100
            stv = d.get('poincare_stv_fpdc_ms', 0)
            if np.isnan(stv):
                stv = 0

            features = np.array([
                d.get('cv_bp_pct', 0),
                prem_pct,
                d.get('morphology_instability', 0),
                d.get('ead_incidence_pct', 0),
                d.get('amplitude_cv_pct', 0),
                stv,
                1.0 if d.get('has_pauses', False) else 0.0,
            ])

            # Logistic regression: P(+) = sigmoid(coefs · features + intercept)
            logit = np.dot(coefs, features) + intercept
            prob = 1.0 / (1.0 + np.exp(-logit))

            # Map probability to 0-100 score
            self.risk_score = int(min(100, max(0, round(prob * 100))))

        except (ValueError, KeyError, IndexError, TypeError) as e:
            logger.debug("Logistic risk score failed, using manual fallback: %s", e)
            self._compute_risk_score_manual(d, n_beats)

    def summary_text(self):
        lines = [f"Classification: {self.classification}",
                 f"Risk Score: {self.risk_score}/100"]
        for f in self.flags:
            lines.append(f"  [{f['severity'].upper()}] {f['type']}: "
                         f"{f['description']}")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════
#   RESIDUAL ANALYSIS  (paper approach)
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

# Back-compat alias
_compute_template = compute_template


def _compute_residuals(beats_data, template):
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


def _residual_rms(residuals):
    """Per-beat RMS of residual (morphological instability)."""
    return np.array([np.sqrt(np.mean(r ** 2)) for r in residuals])


def _detect_ead_from_residual(residual, fs, template_range=None,
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
    template : 1-D array or None — if provided, used for additional
        context (not currently used, reserved for future refinement).
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


def _poincare_stv(values):
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
    residuals = _compute_residuals(beats_data, template)
    rms_vals = _residual_rms(residuals)
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
        eads = _detect_ead_from_residual(
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
    result['poincare_stv_fpd'] = _poincare_stv(fpd_valid)

    fpdc_vals = [p.get('fpdc_ms', np.nan) for p in all_params]
    fpdc_valid = [v for v in fpdc_vals if not np.isnan(v)]
    result['poincare_stv_fpdc'] = _poincare_stv(fpdc_valid)

    return result


# ═══════════════════════════════════════════════════════════════════════
#   MAIN ANALYSIS  (statistical + residual combined)
# ═══════════════════════════════════════════════════════════════════════

def analyze_arrhythmia(beat_indices, beat_periods, all_params, summary, fs,
                       baseline_summary=None, cfg=None,
                       beats_data=None, baseline_template=None):
    """Analyse arrhythmia combining statistical and residual approaches.

    Parameters
    ----------
    beat_indices, beat_periods, all_params, summary, fs :
        Standard pipeline outputs.
    baseline_summary : dict or None — baseline parameters for comparison.
    cfg : ArrhythmiaConfig or None.
    beats_data : list of 1-D arrays or None — if provided, residual
        analysis is performed (paper approach).
    baseline_template : 1-D array or None — if provided, used as the
        reference template for residual analysis (baseline-relative mode).
        When None, the template is computed from the current recording.

    Returns
    -------
    ArrhythmiaReport
    """
    if cfg is None:
        from .config import ArrhythmiaConfig
        cfg = ArrhythmiaConfig()

    report = ArrhythmiaReport()
    bp_ms = beat_periods * 1000 if len(beat_periods) > 0 else np.array([])

    if len(bp_ms) < 3:
        report.add_flag('insufficient_data', 'warning',
                        f'Only {len(bp_ms)} beat periods')
        report.classification = 'Insufficient Data'
        return report

    mean_bp, std_bp = np.mean(bp_ms), np.std(bp_ms)
    cv_bp = std_bp / mean_bp * 100 if mean_bp > 0 else 0
    report.details.update({
        'mean_bp_ms': mean_bp, 'cv_bp_pct': cv_bp,
        'n_beats': len(beat_indices),
        'bpm': 60000 / mean_bp if mean_bp > 0 else 0,
    })

    # ── Statistical analysis (original) ────────────────────────────────

    # Heart rate
    if mean_bp < cfg.tachycardia_bp_ms:
        report.add_flag('tachycardia', 'warning',
                        f'Mean BP = {mean_bp:.0f} ms '
                        f'({60000/mean_bp:.0f} BPM)')
    elif mean_bp > cfg.bradycardia_bp_ms:
        report.add_flag('bradycardia', 'warning',
                        f'Mean BP = {mean_bp:.0f} ms '
                        f'({60000/mean_bp:.0f} BPM)')

    # Rhythm regularity
    if cv_bp > cfg.rr_irregularity_cv:
        sev = 'critical' if cv_bp > cfg.rr_critical_cv else 'warning'
        report.add_flag('rr_irregular', sev, f'RR CV = {cv_bp:.1f}%')

    # Premature / delayed beats
    n_prem, n_del = 0, 0
    for i, bp in enumerate(bp_ms):
        if bp < mean_bp * cfg.premature_beat_factor:
            n_prem += 1
            report.add_event(i + 1, 'premature_beat', f'BP={bp:.0f}ms')
        elif bp > mean_bp * cfg.delayed_beat_factor:
            n_del += 1
            report.add_event(i + 1, 'delayed_beat', f'BP={bp:.0f}ms')

    report.details['n_premature'] = n_prem
    report.details['n_delayed'] = n_del

    if n_prem > 0:
        pct = n_prem / len(bp_ms) * 100
        report.add_flag('premature_beats',
                        'critical' if pct > 10 else 'warning',
                        f'{n_prem} premature ({pct:.1f}%)')
    if n_del > 0:
        pct = n_del / len(bp_ms) * 100
        report.add_flag('delayed_beats',
                        'warning' if pct > 5 else 'info',
                        f'{n_del} delayed ({pct:.1f}%)')

    # Pauses / cessation
    pauses = bp_ms[bp_ms > mean_bp * cfg.cessation_factor]
    report.details['has_pauses'] = len(pauses) > 0
    if len(pauses) > 0:
        report.add_flag('beat_cessation', 'critical',
                        f'{len(pauses)} pause(s), '
                        f'max={np.max(pauses):.0f}ms')

    # STV (from summary — beat-period STV)
    stv = summary.get('stv_ms', np.nan)
    if not np.isnan(stv) and stv > cfg.stv_high_risk_ms:
        report.add_flag('high_stv', 'warning', f'STV = {stv:.1f} ms')

    # FPD prolongation vs baseline
    fpd_vals = [p['fpd_ms'] for p in all_params
                if not np.isnan(p.get('fpd_ms', np.nan))]
    if fpd_vals:
        mean_fpd = np.mean(fpd_vals)
        if baseline_summary and 'fpd_ms_mean' in baseline_summary:
            ratio = mean_fpd / baseline_summary['fpd_ms_mean']
            if ratio > cfg.fpd_prolongation_threshold:
                report.add_flag('fpd_prolongation', 'critical',
                                f'FPD {ratio:.0%} of baseline')
        if mean_fpd > cfg.fpd_critical_length_ms:
            report.add_flag('fpd_very_long', 'critical',
                            f'FPD = {mean_fpd:.0f} ms')

    # EAD detection (FPD outlier approach — original)
    n_ead_stat = 0
    if len(fpd_vals) > 5:
        med_fpd = np.median(fpd_vals)
        mad_fpd = np.median(np.abs(np.array(fpd_vals) - med_fpd))
        for i, p in enumerate(all_params):
            fpd = p.get('fpd_ms', np.nan)
            if (not np.isnan(fpd) and mad_fpd > 0
                    and (fpd - med_fpd) > cfg.ead_mad_factor * mad_fpd * 1.4826):
                n_ead_stat += 1
                report.add_event(i + 1, 'ead_suspect_stat',
                                 f'FPD={fpd:.0f}ms')

    # Amplitude instability (Visone et al. 2023: amplitude variation
    # signals altered depolarization — hERG blockers, Ca²⁺ channel
    # blockers, or progressive tissue degradation)
    amps = [p['spike_amplitude_mV'] for p in all_params
            if not np.isnan(p.get('spike_amplitude_mV', np.nan))]
    amp_cv = 0.0
    if len(amps) > 5:
        amp_cv = np.std(amps) / np.mean(amps) * 100
        report.details['amplitude_cv_pct'] = amp_cv
        if amp_cv > cfg.amplitude_instability_cv:
            report.add_flag('amplitude_instability', 'warning',
                            f'Amplitude CV = {amp_cv:.1f}%')
    else:
        report.details['amplitude_cv_pct'] = 0.0

    # ── Residual analysis (paper approach) ─────────────────────────────

    n_ead_resid = 0
    if beats_data is not None and len(beats_data) >= 5:
        res_result = analyze_residual(beats_data, fs, all_params, cfg=cfg,
                                      baseline_template=baseline_template)
        report.residual_details = res_result

        morph_inst = res_result.get('morphology_instability', 0)
        n_ead_resid = res_result.get('n_ead_beats', 0)
        ead_pct = res_result.get('ead_incidence_pct', 0)
        stv_fpd = res_result.get('poincare_stv_fpd', np.nan)
        stv_fpdc = res_result.get('poincare_stv_fpdc', np.nan)

        # Store in details for downstream use (risk map, reports)
        report.details['morphology_instability'] = morph_inst
        report.details['ead_incidence_pct'] = ead_pct
        report.details['n_ead_residual'] = n_ead_resid
        report.details['poincare_stv_fpd_ms'] = stv_fpd
        report.details['poincare_stv_fpdc_ms'] = stv_fpdc
        report.details['mean_residual_rms'] = res_result.get(
            'mean_residual_rms', np.nan)

        # Flags from residual analysis
        if morph_inst > 0.6:
            report.add_flag('morphology_instability', 'critical',
                            f'Residual instability = {morph_inst:.2f}')
        elif morph_inst > 0.3:
            report.add_flag('morphology_instability', 'warning',
                            f'Residual instability = {morph_inst:.2f}')

        if n_ead_resid > 0:
            for beat_i, ead_events in res_result.get('ead_beats', []):
                for ead in ead_events:
                    report.add_event(beat_i + 1, 'ead_residual',
                                     f't={ead["time_ms"]:.0f}ms, '
                                     f'amp={ead["amplitude"]:.4f}')

        if not np.isnan(stv_fpdc) and stv_fpdc > cfg.stv_high_risk_ms:
            report.add_flag('high_repol_stv', 'warning',
                            f'Poincaré STV(FPDcF) = {stv_fpdc:.1f} ms')

    # ── Combine EAD counts from both approaches ──
    n_ead_total = max(n_ead_stat, n_ead_resid)
    ead_pct = report.details.get('ead_incidence_pct', 0)
    if n_ead_total > 0:
        source = 'residual' if n_ead_resid >= n_ead_stat else 'statistical'
        report.add_flag('ead_events',
                        'critical' if ead_pct > 10 else 'warning',
                        f'{n_ead_total} EAD-like event(s) '
                        f'({ead_pct:.1f}% of beats) [{source}]')
    report.details['n_ead_total'] = n_ead_total

    # ── Classification (based on incidence, not absolute counts) ──────

    morph_inst = report.details.get('morphology_instability', 0)
    prem_pct = (n_prem / len(bp_ms) * 100) if len(bp_ms) > 0 else 0

    if len(pauses) > 0 and cv_bp > cfg.fibrillation_cv:
        report.classification = 'Fibrillation-like / Chaotic Rhythm'
    elif ead_pct > 20:
        report.classification = 'EAD with Triggered Activity'
    elif ead_pct > 5 and cv_bp > cfg.rr_irregularity_cv:
        report.classification = 'Proarrhythmic (EAD-prone)'
    elif morph_inst > 0.6 and cv_bp > cfg.rr_irregularity_cv:
        report.classification = 'Morphologically Unstable + Irregular'
    elif len(pauses) > 0:
        report.classification = 'Intermittent Cessation'
    elif cv_bp > cfg.rr_critical_cv:
        report.classification = 'Highly Irregular Rhythm'
    elif morph_inst > 0.6:
        report.classification = 'Morphologically Unstable'
    elif prem_pct > 10:
        report.classification = 'Frequent Premature Beats'
    elif cv_bp > cfg.rr_irregularity_cv:
        report.classification = 'Irregular Rhythm'
    elif mean_bp < cfg.tachycardia_bp_ms:
        report.classification = 'Tachycardia'
    elif mean_bp > cfg.bradycardia_bp_ms:
        report.classification = 'Bradycardia'
    elif [f for f in report.flags if f['severity'] == 'warning']:
        report.classification = 'Borderline / Mild Abnormalities'

    # ── Compute risk score from incidence-based metrics ──
    score_mode = getattr(cfg, 'risk_score_mode', 'manual')
    report.compute_risk_score(mode=score_mode)

    return report
