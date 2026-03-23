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

logger = logging.getLogger(__name__)


def _get_repol_cfg(cfg):
    """Return a RepolarizationConfig, creating defaults if cfg is None."""
    if cfg is None:
        from .config import RepolarizationConfig
        return RepolarizationConfig()
    return cfg


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


# ─── Repolarization detection ───

def _linear_detrend_endpoints(segment, margin_frac=0.08):
    """
    Linear detrend using the first and last margin_frac of the segment.

    Unlike polynomial detrend (which can absorb the repolarization wave),
    this only subtracts a straight line anchored at the segment boundaries,
    preserving the central bump.
    """
    n = len(segment)
    margin = max(3, int(n * margin_frac))
    y_start = np.mean(segment[:margin])
    y_end = np.mean(segment[-margin:])
    baseline = np.linspace(y_start, y_end, n)
    return segment - baseline


def _apply_fpd_method(seg_det, best_pk, best_sign, fs, search_start, spike_idx, cfg=None):
    """
    Apply the configured FPD measurement method after finding the repol peak.

    Methods:
      'tangent'         : tangent at max downslope → baseline intersection
      'peak'            : repolarization peak position only
      'max_slope'       : point of maximum downslope
      '50pct'           : 50% amplitude on descending side
      'baseline_return' : zero-crossing after peak

    Returns: fpd_samples (int)
    """
    rc = _get_repol_cfg(cfg)
    method = rc.fpd_method
    peak_samples = search_start + best_pk - spike_idx

    if method == 'peak':
        return peak_samples

    after_peak = seg_det[best_pk:]
    if len(after_peak) <= 10:
        return peak_samples

    dt = 1.0 / fs
    after_signed = best_sign * after_peak

    if method == 'baseline_return':
        zc = np.where(after_signed < 0)[0]
        if len(zc) > 0:
            return search_start + best_pk + zc[0] - spike_idx
        return peak_samples

    if method == '50pct':
        peak_amp = after_signed[0]
        target = peak_amp * 0.5
        cross = np.where(after_signed < target)[0]
        if len(cross) > 0:
            return search_start + best_pk + cross[0] - spike_idx
        return peak_samples

    # Methods that need derivative
    deriv = np.gradient(after_signed, dt)
    max_slope_idx = np.argmin(deriv)
    max_slope_ms = max_slope_idx / fs * 1000

    # Minimum meaningful slope: relative to peak amplitude, not absolute.
    # This makes the tangent method robust to amplifier gain scaling.
    peak_amp = after_signed[0] if len(after_signed) > 0 else 1.0
    min_slope = max(abs(peak_amp) * 0.01 / dt, 1e-12)  # 1% of peak per sample

    if method == 'max_slope':
        if max_slope_ms < rc.tangent_max_slope_window_ms and np.abs(deriv[max_slope_idx]) > min_slope:
            return search_start + best_pk + max_slope_idx - spike_idx
        return peak_samples

    # Default: 'tangent'
    if max_slope_ms < rc.tangent_max_slope_window_ms and np.abs(deriv[max_slope_idx]) > min_slope:
        y0 = after_signed[max_slope_idx]
        slope = deriv[max_slope_idx]
        if abs(slope) > min_slope:
            x_intersect = max_slope_idx - y0 / slope * fs
            x_intersect = max(0, x_intersect)
            max_extension = int(rc.tangent_max_extension_ms / 1000 * fs)
            if x_intersect <= max_extension:
                return search_start + best_pk + int(x_intersect) - spike_idx
            else:
                return search_start + best_pk + max_slope_idx - spike_idx

    return peak_samples


def _consensus_fpd(seg_det, best_pk, best_sign, fs, search_start, spike_idx, cfg=None):
    """
    Run multiple FPD methods and return consensus result.

    Runs all available methods (tangent, peak, max_slope, 50pct, baseline_return),
    computes agreement statistics, and selects the best result based on:
      - Agreement with majority of methods (cluster analysis)
      - Individual method confidence

    Returns: (fpd_samples, method_details)
      method_details: dict with per-method results and agreement info
    """
    rc = _get_repol_cfg(cfg)
    methods = ['tangent', 'peak', 'max_slope', '50pct', 'baseline_return']
    results = {}

    for method in methods:
        # Temporarily override fpd_method in a copy
        from .config import RepolarizationConfig
        from dataclasses import replace
        temp_cfg = replace(rc, fpd_method=method)
        try:
            fpd = _apply_fpd_method(seg_det, best_pk, best_sign, fs,
                                     search_start, spike_idx, cfg=temp_cfg)
            if fpd is not None and fpd > 0:
                results[method] = fpd
        except (ValueError, IndexError, RuntimeError) as e:
            logger.debug("FPD method %s failed: %s", method, e)

    if not results:
        return None, {'methods_tried': methods, 'methods_ok': 0}

    # Cluster analysis: find the largest group of methods that agree
    fpd_values = np.array(list(results.values()))
    fpd_methods = list(results.keys())
    fpd_ms = fpd_values / fs * 1000  # convert to ms for comparison

    # Agreement: methods within ±50ms of each other
    agreement_radius_ms = 50.0
    best_cluster = []
    best_cluster_center = 0

    for i, val in enumerate(fpd_ms):
        cluster = [j for j, v in enumerate(fpd_ms) if abs(v - val) <= agreement_radius_ms]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
            best_cluster_center = np.mean(fpd_ms[cluster])

    # Preferred method priority (if in the best cluster)
    priority = ['tangent', 'max_slope', '50pct', 'baseline_return', 'peak']
    cluster_methods = [fpd_methods[i] for i in best_cluster]

    selected_method = None
    selected_fpd = None
    for pref in priority:
        if pref in cluster_methods:
            selected_method = pref
            selected_fpd = results[pref]
            break

    if selected_fpd is None:
        # Fallback to tangent or first available
        selected_method = fpd_methods[0]
        selected_fpd = fpd_values[0]

    agreement = len(best_cluster) / len(results) if results else 0

    method_details = {
        'methods_tried': methods,
        'methods_ok': len(results),
        'per_method': {m: v / fs * 1000 for m, v in results.items()},
        'selected_method': selected_method,
        'cluster_size': len(best_cluster),
        'cluster_methods': cluster_methods,
        'cluster_center_ms': best_cluster_center,
        'agreement': agreement,
        'spread_ms': float(np.ptp(fpd_ms)) if len(fpd_ms) > 1 else 0,
    }

    return selected_fpd, method_details


def _find_repolarization_on_template(template, fs, pre_ms=50, cfg=None):
    """
    Find the repolarization wave on a clean averaged template.

    Uses the configured fpd_method (default: tangent) to determine the
    FPD endpoint after finding the repolarization peak.

    Returns: fpd_samples (int), repol_sign (+1 or -1), repol_amplitude,
             confidence (float 0-1), peak_samples (int)
    """
    rc = _get_repol_cfg(cfg)
    pre_samples = int(pre_ms * fs / 1000)

    # Depolarization spike: find the sharpest feature near expected position
    spike_search = int(rc.spike_search_window_ms / 1000 * fs)
    s0 = max(0, pre_samples - spike_search)
    s1 = min(len(template), pre_samples + spike_search)
    spike_region = template[s0:s1]

    spike_local_idx = np.argmax(np.abs(spike_region - np.mean(spike_region)))
    spike_idx = s0 + spike_local_idx

    # Search for repolarization
    search_start = spike_idx + int(rc.search_start_ms / 1000 * fs)
    search_end = spike_idx + int(rc.search_end_ms / 1000 * fs)

    if search_end > len(template):
        search_end = len(template)
    if search_start >= search_end or (search_end - search_start) < int(0.05 * fs):
        return None, 1, 0.0, 0.0, None

    segment = template[search_start:search_end].copy()

    # Low-pass filter
    nyq = 0.5 * fs
    cutoff = min(rc.repol_lowpass_hz, nyq * 0.8)
    try:
        b, a = sig.butter(rc.repol_filter_order, cutoff / nyq, btype='low')
        seg_smooth = sig.filtfilt(b, a, segment)
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.debug("Template repol filter failed (cutoff=%.1f Hz): %s", cutoff, e)
        seg_smooth = segment

    # LINEAR detrend (preserves repolarization bump)
    seg_det = _linear_detrend_endpoints(seg_smooth, margin_frac=rc.detrend_margin_frac)

    # ─── Step 1: Find the repolarization peak ───
    best_pk = None
    best_prom = 0
    best_sign = 1

    peak_dist = int(rc.peak_min_distance_ms / 1000 * fs)
    for sign in [1, -1]:
        threshold = np.std(seg_det) * rc.peak_prominence_factor
        pks, props = sig.find_peaks(sign * seg_det,
                                     prominence=threshold,
                                     distance=peak_dist)
        if len(pks) > 0:
            idx = np.argmax(props['prominences'])
            if props['prominences'][idx] > best_prom:
                best_prom = props['prominences'][idx]
                best_pk = pks[idx]
                best_sign = sign

    if best_pk is None:
        best_pk = np.argmax(np.abs(seg_det))
        best_sign = 1 if seg_det[best_pk] > 0 else -1
        best_prom = np.abs(seg_det[best_pk])

    peak_samples = search_start + best_pk - spike_idx
    repol_amp = seg_smooth[best_pk]

    # ─── Step 2: Apply configured FPD method (with optional consensus) ───
    consensus_details = None
    if rc.fpd_method == 'consensus':
        fpd_samples, consensus_details = _consensus_fpd(
            seg_det, best_pk, best_sign, fs, search_start, spike_idx, cfg=cfg)
        if fpd_samples is None:
            fpd_samples = peak_samples
    else:
        fpd_samples = _apply_fpd_method(seg_det, best_pk, best_sign, fs,
                                         search_start, spike_idx, cfg=cfg)

    # ─── Step 3: Confidence scoring ───
    noise_level = np.std(seg_det)
    prominence_ratio = best_prom / noise_level if noise_level > 0 else 0

    # Find return-to-baseline for agreement check
    after_signed = best_sign * seg_det[best_pk:]
    zc_after = np.where(after_signed < 0)[0]
    fpd_end = None
    if len(zc_after) > 0:
        fpd_end = search_start + best_pk + zc_after[0] - spike_idx

    endpoints = [peak_samples, fpd_samples]
    if fpd_end is not None:
        endpoints.append(fpd_end)
    endpoint_range_ms = (max(endpoints) - min(endpoints)) / fs * 1000

    conf_prominence = min(1.0, prominence_ratio / rc.confidence_prominence_scale)
    conf_agreement = max(0, 1.0 - endpoint_range_ms / rc.confidence_agreement_range_ms)
    confidence = (rc.confidence_weight_prominence * conf_prominence +
                  rc.confidence_weight_agreement * conf_agreement)

    # Boost confidence when consensus methods agree well
    if consensus_details is not None and consensus_details.get('agreement', 0) > 0.6:
        consensus_bonus = 0.1 * consensus_details['agreement']
        confidence = min(1.0, confidence + consensus_bonus)

    if fpd_samples is not None and fpd_samples > 0:
        return fpd_samples, best_sign, repol_amp, confidence, peak_samples, consensus_details

    return None, 1, 0.0, 0.0, None, None


def _find_repolarization_per_beat(data, t, spike_idx, fs,
                                   template_fpd_samples=None,
                                   template_peak_samples=None,
                                   template_repol_sign=1,
                                   cfg=None):
    """
    Find repolarization in a single beat, optionally guided by template.

    Uses the configured FPD method (consistent with template detection).
    """
    rc = _get_repol_cfg(cfg)
    search_tolerance = int(rc.per_beat_tolerance_ms * fs / 1000)

    # Define search window
    if template_peak_samples is not None:
        peak_search_start = spike_idx + max(int(0.1 * fs), template_peak_samples - search_tolerance)
        peak_search_end = spike_idx + template_peak_samples + search_tolerance + int(0.200 * fs)
    elif template_fpd_samples is not None:
        peak_search_start = spike_idx + max(int(0.1 * fs), template_fpd_samples - search_tolerance - int(0.100 * fs))
        peak_search_end = spike_idx + template_fpd_samples + search_tolerance + int(0.100 * fs)
    else:
        peak_search_start = spike_idx + int(rc.search_start_ms / 1000 * fs)
        peak_search_end = spike_idx + int(rc.search_end_ms / 1000 * fs)

    if peak_search_end > len(data):
        peak_search_end = len(data)
    if peak_search_start >= peak_search_end or (peak_search_end - peak_search_start) < 10:
        return None, np.nan

    segment = data[peak_search_start:peak_search_end]

    # Low-pass filter
    nyq = 0.5 * fs
    cutoff = min(rc.repol_lowpass_hz, nyq * 0.8)
    if len(segment) > 20:
        try:
            b, a = sig.butter(rc.repol_filter_order, cutoff / nyq, btype='low')
            seg_smooth = sig.filtfilt(b, a, segment)
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.debug("Per-beat repol filter failed (cutoff=%.1f Hz): %s", cutoff, e)
            seg_smooth = segment
    else:
        seg_smooth = segment

    seg_det = _linear_detrend_endpoints(seg_smooth, margin_frac=rc.detrend_margin_frac)

    # ─── Step 1: Find the repolarization peak ───
    best_idx = None
    best_score = 0
    peak_dist = int(rc.per_beat_peak_distance_ms / 1000 * fs)
    dist_penalty_scale = rc.per_beat_distance_penalty_ms / 1000 * fs

    for sign in [template_repol_sign, -template_repol_sign]:
        pks, props = sig.find_peaks(sign * seg_det,
                                     prominence=np.std(seg_det) * rc.peak_prominence_factor,
                                     distance=peak_dist)
        if len(pks) > 0:
            if template_peak_samples is not None:
                expected_local = template_peak_samples - (peak_search_start - spike_idx)
                distances = np.abs(pks - expected_local)
                scores = props['prominences'] / (1 + distances / dist_penalty_scale)
                best_pk = pks[np.argmax(scores)]
                score = np.max(scores)
            elif template_fpd_samples is not None:
                expected_local = template_fpd_samples - int(0.050 * fs) - (peak_search_start - spike_idx)
                distances = np.abs(pks - expected_local)
                scores = props['prominences'] / (1 + distances / dist_penalty_scale)
                best_pk = pks[np.argmax(scores)]
                score = np.max(scores)
            else:
                best_pk = pks[np.argmax(props['prominences'])]
                score = np.max(props['prominences'])

            if score > best_score:
                best_score = score
                best_idx = best_pk

    if best_idx is None:
        best_idx = np.argmax(np.abs(seg_det))

    # ─── Step 2: Apply configured FPD method ───
    fpd_idx = _apply_fpd_method(seg_det, best_idx, template_repol_sign, fs,
                                 peak_search_start, spike_idx, cfg=cfg)
    # Convert from "samples from spike" to absolute index
    actual_idx = spike_idx + fpd_idx
    if actual_idx < len(t):
        fpd = abs(t[actual_idx] - t[spike_idx])
        return fpd, seg_smooth[min(best_idx, len(seg_smooth)-1)]

    return None, np.nan


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
