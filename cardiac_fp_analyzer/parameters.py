"""
parameters.py — Extraction of electrophysiological parameters from FP beats.

Key parameters:
  - Beat Period (BP) / RR interval
  - Signal Amplitude (Vmax)
  - FPD (Field Potential Duration) ~ QT
  - FPDc (Fridericia): FPDc = FPD / (RR)^(1/3)
  - FPDc (Bazett): FPDc = FPD / sqrt(RR)
  - Rise Time
  - Repolarization amplitude
  - STV (short-term variability)

FPD measurement strategy (inspired by Visone et al., Tox Sci 2023):
  1. Build an averaged beat template via cross-correlation alignment
  2. Detect the repolarization wave on the clean template → reference FPD
  3. Refine FPD per-beat using a template-guided search window

All parameters are configurable via RepolarizationConfig (see config.py).
"""

import logging

import numpy as np

from .repolarization import (
    _get_repol_cfg,
)
from .repolarization import (
    find_repolarization_on_template as _find_repolarization_on_template,
)
from .repolarization import (
    find_repolarization_per_beat as _find_repolarization_per_beat,
)

logger = logging.getLogger(__name__)


# ─── Template averaging ───

def _align_beats_xcorr(beats_data, fs, cfg=None):
    """
    Align beats via cross-correlation to a reference beat.

    The reference is initially the median beat, then refined iteratively.
    Returns aligned beat waveforms (all same length).
    """
    rc = _get_repol_cfg(cfg)
    if len(beats_data) < 3:
        return beats_data

    max_shift = int(rc.alignment_max_shift_ms * fs / 1000)

    # Ensure uniform length
    min_len = min(len(b) for b in beats_data)
    beats = [b[:min_len].copy() for b in beats_data]

    # Reference: median of all beats (robust starting point)
    ref = np.median(np.array(beats), axis=0)

    # Align each beat to reference via cross-correlation
    dep_len = min(int(rc.alignment_depol_region_ms / 1000 * fs), min_len)
    aligned = []
    for beat in beats:
        corr = np.correlate(ref[:dep_len + 2*max_shift],
                           beat[:dep_len], mode='valid')
        if len(corr) == 0:
            aligned.append(beat)
            continue
        shift = np.argmax(corr) - max_shift
        shift = np.clip(shift, -max_shift, max_shift)

        if shift > 0:
            padded = np.concatenate([beat[shift:], np.full(shift, beat[-1])])
        elif shift < 0:
            padded = np.concatenate([np.full(-shift, beat[0]), beat[:shift]])
        else:
            padded = beat

        aligned.append(padded[:min_len])

    return aligned


def build_beat_template(beats_data, fs, cfg=None):
    """
    Build a clean averaged beat template.

    1. Select up to max_beats from the recording (evenly spaced)
    2. Align via cross-correlation
    3. Compute robust median template

    Returns: template (array), or None if too few beats.
    """
    rc = _get_repol_cfg(cfg)
    if len(beats_data) < 5:
        return None

    n = min(len(beats_data), rc.max_beats_template)
    if n < len(beats_data):
        indices = np.linspace(0, len(beats_data)-1, n, dtype=int)
        selected = [beats_data[i] for i in indices]
    else:
        selected = list(beats_data)

    aligned = _align_beats_xcorr(selected, fs, cfg=cfg)

    if not aligned:
        return None

    min_len = min(len(b) for b in aligned)
    mat = np.array([b[:min_len] for b in aligned])

    template = np.median(mat, axis=0)
    return template


# ─── Per-beat parameter extraction ───

def extract_beat_parameters(beat_data, beat_time, fs, rr_interval=None,
                           template_fpd_samples=None, template_peak_samples=None,
                           template_repol_sign=1, cfg=None,
                           beat_period_s=None):
    """
    Extract parameters from a single segmented beat.

    If template_fpd_samples is provided, uses template-guided FPD detection.
    """
    rc = _get_repol_cfg(cfg)
    params = {}
    data = np.array(beat_data, dtype=np.float64)
    t = np.array(beat_time, dtype=np.float64)
    zero_idx = np.argmin(np.abs(t))

    # Spike amplitude
    sp = max(0, zero_idx - int(rc.spike_pre_ms / 1000 * fs))
    ep = min(len(data), zero_idx + int(rc.spike_post_ms / 1000 * fs))
    spike_region = data[sp:ep]
    spike_max, spike_min = np.max(spike_region), np.min(spike_region)
    params['spike_amplitude_mV'] = (spike_max - spike_min) * 1000

    # Rise time (10-90%)
    spike_amp = spike_max - spike_min
    if spike_amp > 0:
        t10, t90 = spike_min + 0.1 * spike_amp, spike_min + 0.9 * spike_amp
        a10 = np.where(spike_region >= t10)[0]
        a90 = np.where(spike_region >= t90)[0]
        params['rise_time_ms'] = (max(0, (a90[0] - a10[0]) / fs * 1000)
                                  if len(a10) > 0 and len(a90) > 0 else np.nan)
    else:
        params['rise_time_ms'] = np.nan

    # FPD (template-guided, configured method)
    fpd, repol_amp, repol_peak_i, fpd_end_i = _find_repolarization_per_beat(
        data, t, zero_idx, fs,
        template_fpd_samples=template_fpd_samples,
        template_peak_samples=template_peak_samples,
        template_repol_sign=template_repol_sign,
        cfg=cfg,
        beat_period_s=beat_period_s
    )
    params['fpd_ms'] = fpd * 1000 if fpd is not None else np.nan
    params['repol_amplitude_mV'] = repol_amp * 1000 if not np.isnan(repol_amp) else np.nan
    params['repol_peak_idx_in_beat'] = repol_peak_i
    params['fpd_endpoint_idx_in_beat'] = fpd_end_i

    # FPDc Fridericia & Bazett (always compute both; config selects which to report)
    if fpd is not None and rr_interval is not None and rr_interval > 0:
        params['fpdc_ms'] = (fpd / (rr_interval ** (1/3))) * 1000
        params['fpdc_bazett_ms'] = (fpd / np.sqrt(rr_interval)) * 1000
    else:
        params['fpdc_ms'] = np.nan
        params['fpdc_bazett_ms'] = np.nan

    # Max dV/dt
    deriv = np.gradient(data, 1.0 / fs)
    params['max_dvdt'] = np.max(np.abs(deriv[sp:ep]))

    return params


def extract_all_parameters(beats_data, beats_time, beat_indices, fs, cfg=None):
    """
    Extract parameters for all beats and compute summary statistics.

    Uses template averaging for robust FPD measurement:
    1. Build averaged template from aligned beats
    2. Detect repolarization on template → reference FPD
    3. Guide per-beat FPD with template reference

    Parameters
    ----------
    cfg : RepolarizationConfig or None
    """
    rc = _get_repol_cfg(cfg)
    from .beat_detection import compute_beat_periods

    # Alignment guard: beats_data, beats_time and beat_indices must be in sync
    if not (len(beats_data) == len(beats_time) == len(beat_indices)):
        raise ValueError(
            f"Parameter extraction alignment error: beats_data={len(beats_data)}, "
            f"beats_time={len(beats_time)}, beat_indices={len(beat_indices)}"
        )

    # Guard against empty input
    if len(beats_data) == 0:
        logger.warning("extract_all_parameters: no beats to process")
        return [], {}

    beat_periods = compute_beat_periods(beat_indices, fs)

    # ─── Template averaging for FPD reference ───
    template = build_beat_template(beats_data, fs, cfg=cfg)
    template_fpd_samples = None
    template_repol_sign = 1

    repol_confidence = 0.0
    template_peak_samples = None

    consensus_info = None
    # Compute median beat period for adaptive min FPD
    median_bp_s = None
    if len(beat_periods) > 0:
        median_bp_s = float(np.median(beat_periods))

    # ─── Minimal signal amplitude gate ───
    # If median spike amplitude is below threshold, skip FPD detection entirely.
    min_amp_uV = getattr(rc, 'min_signal_amplitude_uV', 0.0)
    signal_too_weak = False
    if min_amp_uV > 0 and len(beats_data) > 0:
        pre_samples = int(rc.segment_pre_ms / 1000 * fs)
        spike_amps = []
        for bd in beats_data:
            sp = max(0, pre_samples - int(rc.spike_pre_ms / 1000 * fs))
            ep = min(len(bd), pre_samples + int(rc.spike_post_ms / 1000 * fs))
            if ep > sp:
                spike_region = bd[sp:ep]
                spike_amps.append((np.max(spike_region) - np.min(spike_region)) * 1e6)  # V → µV
        if spike_amps:
            median_amp_uV = float(np.median(spike_amps))
            if median_amp_uV < min_amp_uV:
                signal_too_weak = True
                logger.info("Signal amplitude gate: median %.1f µV < %.1f µV → "
                            "FPD analysis skipped (signal too weak)",
                            median_amp_uV, min_amp_uV)

    if template is not None and not signal_too_weak:
        pre_ms = rc.segment_pre_ms
        fpd_result = _find_repolarization_on_template(
            template, fs, pre_ms=pre_ms, cfg=cfg,
            median_bp_s=median_bp_s)
        if fpd_result[0] is not None:
            template_fpd_samples = fpd_result[0]
            template_repol_sign = fpd_result[1]
            repol_confidence = fpd_result[3] if len(fpd_result) > 3 else 0.5
            template_peak_samples = fpd_result[4] if len(fpd_result) > 4 else None
            consensus_info = fpd_result[5] if len(fpd_result) > 5 else None
            _template_fpd_ms = template_fpd_samples / fs * 1000  # noqa: F841

    # ─── Per-beat extraction ───
    all_params = []
    fpd_vals, fpdc_vals, fpdc_bazett_vals, amp_vals = [], [], [], []
    rise_time_vals, rr_interval_vals = [], []
    pre_samples = int(rc.segment_pre_ms / 1000 * fs)

    # Detect template spike polarity for per-beat inversion detection.
    # If the template's depolarization spike is positive, and a beat's spike
    # is negative, the repolarization sign must be flipped for that beat.
    template_spike_positive = True
    if template is not None:
        t_pre = int(rc.segment_pre_ms / 1000 * fs)
        t_sp = max(0, t_pre - int(rc.spike_pre_ms / 1000 * fs))
        t_ep = min(len(template), t_pre + int(rc.spike_post_ms / 1000 * fs))
        t_spike = template[t_sp:t_ep]
        if len(t_spike) > 0:
            t_baseline = np.median(template)
            template_spike_positive = (
                abs(np.max(t_spike) - t_baseline) >= abs(np.min(t_spike) - t_baseline)
            )

    for i, (bd, bt) in enumerate(zip(beats_data, beats_time)):
        rr = beat_periods[i-1] if i > 0 and i-1 < len(beat_periods) else None
        # Use per-beat RR for adaptive min FPD; fall back to median
        bp_for_beat = rr if rr is not None else median_bp_s

        # Per-beat polarity detection: check if this beat's spike is inverted
        # relative to the template.  If so, flip the repol sign and drop the
        # template guidance (peak position) since the morphology differs.
        beat_repol_sign = template_repol_sign
        beat_tpl_fpd = template_fpd_samples
        beat_tpl_peak = template_peak_samples
        zero_i = np.argmin(np.abs(np.array(bt)))
        b_sp = max(0, zero_i - int(rc.spike_pre_ms / 1000 * fs))
        b_ep = min(len(bd), zero_i + int(rc.spike_post_ms / 1000 * fs))
        b_spike = bd[b_sp:b_ep]
        if len(b_spike) > 0:
            b_baseline = np.median(bd)
            beat_spike_positive = (
                abs(np.max(b_spike) - b_baseline) >= abs(np.min(b_spike) - b_baseline)
            )
            if beat_spike_positive != template_spike_positive:
                # Inverted beat: flip repol sign but KEEP template FPD
                # timing as a guide — the repolarization timing is similar
                # even when the morphology is inverted.  Drop peak_samples
                # (exact peak position depends on morphology) but keep
                # fpd_samples (approximate timing window).
                beat_repol_sign = -template_repol_sign
                beat_tpl_peak = None
                # beat_tpl_fpd stays as template_fpd_samples

        params = extract_beat_parameters(
            bd, bt, fs, rr_interval=rr,
            template_fpd_samples=beat_tpl_fpd,
            template_peak_samples=beat_tpl_peak,
            template_repol_sign=beat_repol_sign,
            cfg=cfg,
            beat_period_s=bp_for_beat
        )
        params['beat_number'] = i + 1
        params['rr_interval_ms'] = rr * 1000 if rr is not None else np.nan
        bi_g = beat_indices[i]
        rp = params.get('repol_peak_idx_in_beat')
        fe = params.get('fpd_endpoint_idx_in_beat')
        if rp is not None:
            params['repol_peak_global_idx'] = int(bi_g - pre_samples + rp)
        else:
            params['repol_peak_global_idx'] = None
        if fe is not None:
            params['fpd_endpoint_global_idx'] = int(bi_g - pre_samples + fe)
        else:
            params['fpd_endpoint_global_idx'] = None
        all_params.append(params)
        if not np.isnan(params['fpd_ms']):
            fpd_vals.append(params['fpd_ms'])
        if not np.isnan(params['fpdc_ms']):
            fpdc_vals.append(params['fpdc_ms'])
        if not np.isnan(params.get('fpdc_bazett_ms', np.nan)):
            fpdc_bazett_vals.append(params['fpdc_bazett_ms'])
        amp_vals.append(params['spike_amplitude_mV'])
        if not np.isnan(params.get('rise_time_ms', np.nan)):
            rise_time_vals.append(params['rise_time_ms'])
        if not np.isnan(params.get('rr_interval_ms', np.nan)):
            rr_interval_vals.append(params['rr_interval_ms'])

    # ── Repolarization diagnostic trace ──
    n_repol_ok = sum(1 for p in all_params if not np.isnan(p['fpd_ms']))
    n_repol_fail = sum(1 for p in all_params if np.isnan(p['fpd_ms']))
    fpd_measured = [p['fpd_ms'] for p in all_params if not np.isnan(p['fpd_ms'])]
    fpd_stats = ""
    if fpd_measured:
        fpd_arr_diag = np.array(fpd_measured)
        fpd_stats = (f", FPD: median={np.median(fpd_arr_diag):.0f}ms "
                     f"range={np.min(fpd_arr_diag):.0f}-{np.max(fpd_arr_diag):.0f}ms")
    print(f"     Repol: {n_repol_ok}/{len(all_params)} detected, "
          f"{n_repol_fail} not detectable"
          f" (template FPD={'%.0f ms' % (template_fpd_samples / fs * 1000) if template_fpd_samples else 'None'}"
          f"{fpd_stats})")

    # ─── Summary statistics ───
    summary = {}
    bp_ms = beat_periods * 1000 if len(beat_periods) > 0 else np.array([])
    for name, vals in [('beat_period_ms', bp_ms), ('spike_amplitude_mV', amp_vals),
                       ('fpd_ms', fpd_vals), ('fpdc_ms', fpdc_vals),
                       ('fpdc_bazett_ms', fpdc_bazett_vals),
                       ('rise_time_ms', rise_time_vals),
                       ('rr_interval_ms', rr_interval_vals)]:
        v = np.array(vals)
        v = v[~np.isnan(v)]
        if len(v) > 0:
            summary[f'{name}_mean'] = np.mean(v)
            summary[f'{name}_std'] = np.std(v)
            summary[f'{name}_median'] = np.median(v)
            summary[f'{name}_cv'] = np.std(v) / np.mean(v) * 100 if np.mean(v) != 0 else np.nan
            summary[f'{name}_min'] = np.min(v)
            summary[f'{name}_max'] = np.max(v)
            summary[f'{name}_n'] = len(v)
        else:
            for s in ['_mean', '_std', '_median', '_cv', '_min', '_max', '_n']:
                summary[f'{name}{s}'] = np.nan

    summary['bpm_mean'] = 60000 / np.mean(bp_ms) if len(bp_ms) > 0 and np.mean(bp_ms) > 0 else np.nan
    summary['stv_ms'] = np.mean(np.abs(np.diff(bp_ms))) / np.sqrt(2) if len(bp_ms) > 1 else np.nan
    summary['beat_periods'] = beat_periods
    summary['fpd_values'] = np.array([p['fpd_ms'] / 1000 for p in all_params if not np.isnan(p['fpd_ms'])])
    summary['fpdc_values'] = np.array([p['fpdc_ms'] / 1000 for p in all_params if not np.isnan(p['fpdc_ms'])])

    # ─── Repolarization detectability statistics ───
    # Count beats where FPD could not be measured (repolarization not detectable).
    # This is a clinically meaningful parameter: a drug that abolishes visible
    # repolarization is high-risk for proarrhythmic effects.
    n_total_beats = len(all_params)
    n_no_repol = sum(1 for p in all_params if np.isnan(p['fpd_ms']))
    summary['n_beats_no_repol'] = n_no_repol
    summary['pct_beats_no_repol'] = (n_no_repol / n_total_beats * 100
                                      if n_total_beats > 0 else 0.0)

    # ─── FPD confidence score ───
    fpd_arr = np.array(fpd_vals)
    fpd_arr = fpd_arr[~np.isnan(fpd_arr)]
    if len(fpd_arr) > 3 and np.mean(fpd_arr) > 0:
        fpd_cv = np.std(fpd_arr) / np.mean(fpd_arr)
        consistency_conf = max(0, 1.0 - fpd_cv / rc.fpd_cv_max_for_confidence)
    else:
        consistency_conf = 0.0

    summary['fpd_confidence'] = (rc.fpd_conf_weight_template * repol_confidence +
                                  rc.fpd_conf_weight_consistency * consistency_conf)

    # Add template info and config to summary
    if template_fpd_samples is not None:
        summary['template_fpd_ms'] = template_fpd_samples / fs * 1000
    summary['fpd_method'] = rc.fpd_method
    summary['correction'] = rc.correction

    # Multi-method consensus info
    if consensus_info is not None:
        summary['consensus'] = consensus_info

    return all_params, summary
