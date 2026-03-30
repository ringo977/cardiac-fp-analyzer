"""
repolarization.py — Repolarization wave detection and FPD measurement.

Contains all methods for detecting the repolarization wave on both
averaged templates and individual beats, including:
  - Linear detrend
  - Multiple FPD endpoint methods (tangent, peak, max_slope, 50pct, baseline_return)
  - Multi-method consensus
  - Template-based and per-beat detection

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


def linear_detrend_endpoints(segment, margin_frac=0.08):
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


def apply_fpd_method(seg_det, best_pk, best_sign, fs, search_start, spike_idx, cfg=None):
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


def consensus_fpd(seg_det, best_pk, best_sign, fs, search_start, spike_idx, cfg=None):
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
        from dataclasses import replace
        temp_cfg = replace(rc, fpd_method=method)
        try:
            fpd = apply_fpd_method(seg_det, best_pk, best_sign, fs,
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


def find_repolarization_on_template(template, fs, pre_ms=50, cfg=None):
    """
    Find the repolarization wave on a clean averaged template.

    Uses the configured fpd_method (default: tangent) to determine the
    FPD endpoint after finding the repolarization peak.

    Returns: fpd_samples (int), repol_sign (+1 or -1), repol_amplitude,
             confidence (float 0-1), peak_samples (int), consensus_details
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
        return None, 1, 0.0, 0.0, None, None

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
    seg_det = linear_detrend_endpoints(seg_smooth, margin_frac=rc.detrend_margin_frac)

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
        # No peak found by find_peaks — try argmax as last resort
        best_pk = np.argmax(np.abs(seg_det))
        best_sign = 1 if seg_det[best_pk] > 0 else -1
        best_prom = np.abs(seg_det[best_pk])

    # ─── Repolarization detectability gate ───
    # If the best candidate's prominence is too low relative to noise,
    # the repolarization is not detectable (flat T-wave / drug effect).
    # Return None instead of forcing a spurious FPD measurement.
    # Reference: EFP Analyzer (Patel et al., Sci Rep 2025) — traces with
    # non-detectable repolarization are excluded, not force-measured.
    noise_level_gate = np.std(seg_det)
    if rc.enable_repol_gate and noise_level_gate > 0:
        repol_snr = best_prom / noise_level_gate
        if repol_snr < rc.repol_gate_min_snr:
            logger.debug("Template repol gate: prominence %.4f < %.1f × noise %.4f "
                         "(SNR=%.2f < %.1f) → repolarization not detectable",
                         best_prom, rc.repol_gate_min_snr, noise_level_gate,
                         repol_snr, rc.repol_gate_min_snr)
            return None, 1, 0.0, 0.0, None, None

    peak_samples = search_start + best_pk - spike_idx
    repol_amp = seg_smooth[best_pk]

    # ─── Step 2: Apply configured FPD method (with optional consensus) ───
    consensus_details = None
    if rc.fpd_method == 'consensus':
        fpd_samples, consensus_details = consensus_fpd(
            seg_det, best_pk, best_sign, fs, search_start, spike_idx, cfg=cfg)
        if fpd_samples is None:
            fpd_samples = peak_samples
    else:
        fpd_samples = apply_fpd_method(seg_det, best_pk, best_sign, fs,
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


def find_repolarization_per_beat(data, t, spike_idx, fs,
                                 template_fpd_samples=None,
                                 template_peak_samples=None,
                                 template_repol_sign=1,
                                 cfg=None):
    """
    Find repolarization in a single beat, optionally guided by template.

    Uses the configured FPD method (consistent with template detection).

    Returns
    -------
    fpd : float or None
        Field potential duration (s), or None if not measurable.
    repol_amp : float
        Smoothed signal amplitude at repolarization peak (V); NaN if unknown.
    repol_peak_idx : int or None
        Sample index *within the beat segment* ``data`` where the repolarization
        peak was identified (for overlay on full trace: ``bi - pre + repol_peak_idx``).
    fpd_endpoint_idx : int or None
        Sample index *within the beat segment* where the configured FPD endpoint
        lies (tangent / peak / etc.); None if FPD could not be computed.
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
        return None, np.nan, None, None

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

    seg_det = linear_detrend_endpoints(seg_smooth, margin_frac=rc.detrend_margin_frac)

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

    # ─── Repolarization detectability gate (per-beat) ───
    # Same logic as template gate but with more lenient threshold,
    # since individual beats are noisier than the averaged template.
    if rc.enable_repol_gate:
        noise_level_beat = np.std(seg_det)
        if noise_level_beat > 0:
            beat_prom = np.abs(seg_det[best_idx])
            beat_repol_snr = beat_prom / noise_level_beat
            if beat_repol_snr < rc.repol_gate_min_snr_beat:
                logger.debug("Per-beat repol gate: SNR=%.2f < %.1f → not detectable",
                             beat_repol_snr, rc.repol_gate_min_snr_beat)
                return None, np.nan, None, None

    # ─── Step 2: Apply configured FPD method ───
    fpd_idx = apply_fpd_method(seg_det, best_idx, template_repol_sign, fs,
                               peak_search_start, spike_idx, cfg=cfg)
    # Convert from "samples from spike" to absolute index (within beat segment ``data``)
    repol_peak_idx = int(peak_search_start + best_idx)
    actual_idx = int(spike_idx + fpd_idx)
    repol_amp_val = float(seg_smooth[min(int(best_idx), len(seg_smooth) - 1)])

    if actual_idx < len(t):
        fpd = abs(t[actual_idx] - t[spike_idx])
        return fpd, repol_amp_val, repol_peak_idx, actual_idx

    return None, np.nan, repol_peak_idx, None


# ── Back-compat aliases (private names used by older code) ──
_linear_detrend_endpoints = linear_detrend_endpoints
_apply_fpd_method = apply_fpd_method
_consensus_fpd = consensus_fpd
_find_repolarization_on_template = find_repolarization_on_template
_find_repolarization_per_beat = find_repolarization_per_beat
