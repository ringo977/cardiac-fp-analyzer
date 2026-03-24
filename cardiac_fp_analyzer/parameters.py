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
from scipy import signal as sig

from .repolarization import (
    _get_repol_cfg,
    linear_detrend_endpoints as _linear_detrend_endpoints,
    apply_fpd_method as _apply_fpd_method,
    consensus_fpd as _consensus_fpd,
    find_repolarization_on_template as _find_repolarization_on_template,
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
                           template_repol_sign=1, cfg=None):
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
    fpd, repol_amp = _find_repolarization_per_beat(
        data, t, zero_idx, fs,
        template_fpd_samples=template_fpd_samples,
        template_peak_samples=template_peak_samples,
        template_repol_sign=template_repol_sign,
        cfg=cfg
    )
    params['fpd_ms'] = fpd * 1000 if fpd is not None else np.nan
    params['repol_amplitude_mV'] = repol_amp * 1000 if not np.isnan(repol_amp) else np.nan

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
    assert len(beats_data) == len(beats_time) == len(beat_indices), (
        f"Parameter extraction alignment error: beats_data={len(beats_data)}, "
        f"beats_time={len(beats_time)}, beat_indices={len(beat_indices)}"
    )

    beat_periods = compute_beat_periods(beat_indices, fs)

    # ─── Template averaging for FPD reference ───
    template = build_beat_template(beats_data, fs, cfg=cfg)
    template_fpd_samples = None
    template_repol_sign = 1

    repol_confidence = 0.0
    template_peak_samples = None

    consensus_info = None
    if template is not None:
        pre_ms = rc.segment_pre_ms
        fpd_result = _find_repolarization_on_template(template, fs, pre_ms=pre_ms, cfg=cfg)
        if fpd_result[0] is not None:
            template_fpd_samples = fpd_result[0]
            template_repol_sign = fpd_result[1]
            repol_confidence = fpd_result[3] if len(fpd_result) > 3 else 0.5
            template_peak_samples = fpd_result[4] if len(fpd_result) > 4 else None
            consensus_info = fpd_result[5] if len(fpd_result) > 5 else None
            template_fpd_ms = template_fpd_samples / fs * 1000

    # ─── Per-beat extraction ───
    all_params = []
    fpd_vals, fpdc_vals, fpdc_bazett_vals, amp_vals = [], [], [], []

    for i, (bd, bt) in enumerate(zip(beats_data, beats_time)):
        rr = beat_periods[i-1] if i > 0 and i-1 < len(beat_periods) else None
        params = extract_beat_parameters(
            bd, bt, fs, rr_interval=rr,
            template_fpd_samples=template_fpd_samples,
            template_peak_samples=template_peak_samples,
            template_repol_sign=template_repol_sign,
            cfg=cfg
        )
        params['beat_number'] = i + 1
        params['rr_interval_ms'] = rr * 1000 if rr is not None else np.nan
        all_params.append(params)
        if not np.isnan(params['fpd_ms']):
            fpd_vals.append(params['fpd_ms'])
        if not np.isnan(params['fpdc_ms']):
            fpdc_vals.append(params['fpdc_ms'])
        if not np.isnan(params.get('fpdc_bazett_ms', np.nan)):
            fpdc_bazett_vals.append(params['fpdc_bazett_ms'])
        amp_vals.append(params['spike_amplitude_mV'])

    # ─── Summary statistics ───
    summary = {}
    bp_ms = beat_periods * 1000 if len(beat_periods) > 0 else np.array([])
    for name, vals in [('beat_period_ms', bp_ms), ('spike_amplitude_mV', amp_vals),
                       ('fpd_ms', fpd_vals), ('fpdc_ms', fpdc_vals),
                       ('fpdc_bazett_ms', fpdc_bazett_vals)]:
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
