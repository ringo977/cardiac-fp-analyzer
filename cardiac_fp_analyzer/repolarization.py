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


def search_window_end_ms(rc, rr_s=None, rr_low_s=None):
    """End of the repolarisation search window, in ms after the spike.

    ``max(search_end_ms, safe × RR, min(pct × RR, RR_low − margin))`` with
    the fractions of ``RepolarizationConfig`` (see ``search_end_pct_rr``);
    ``rr_s`` is the median beat period and ``rr_low_s`` the low percentile
    of the beat periods (where the next beat can start). Without a rhythm
    the fixed ``search_end_ms``.
    """
    end = float(rc.search_end_ms)
    pct = float(getattr(rc, 'search_end_pct_rr', 0.0) or 0.0)
    if pct <= 0 or rr_s is None or not rr_s > 0:
        return end
    safe = float(getattr(rc, 'search_end_pct_rr_safe', pct) or 0.0)
    ext = pct * rr_s * 1000.0
    if rr_low_s is not None and rr_low_s > 0:
        margin = float(getattr(rc, 'search_end_next_beat_margin_ms', 0.0) or 0.0)
        ext = min(ext, rr_low_s * 1000.0 - margin)
    return max(end, min(safe, pct) * rr_s * 1000.0, ext)


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


def next_spike_cut_index(template, fs, spike_idx, scan_from_idx, scan_to_idx, cfg=None):
    """First index in ``[scan_from_idx, scan_to_idx)`` where the template
    contains a depolarisation-like event (the next beat), or ``None``.

    Shape-based, not derivative-based: on noisy templates the derivative of
    the noise is only 3–6× smaller than the spike's, so a dV/dt threshold
    fires on noise. Here a window must correlate with the main spike's shape
    (≥ ``next_spike_guard_min_corr``) AND reach a comparable peak-to-peak
    (≥ ``next_spike_guard_min_amp`` × main) — random noise does neither.
    Returns the index of the event's onset (start of the matching window).
    """
    rc = _get_repol_cfg(cfg)
    x = np.asarray(template, dtype=float)
    a, b = int(0.005 * fs), int(0.015 * fs)
    if spike_idx - a < 0 or spike_idx + b > len(x):
        return None
    k = x[spike_idx - a:spike_idx + b]
    k = k - k.mean()
    kn = float(np.linalg.norm(k))
    kptp = float(np.ptp(k))
    if kn == 0.0 or kptp == 0.0:
        return None
    L = len(k)
    lo = max(int(scan_from_idx), spike_idx + b)
    hi = min(int(scan_to_idx), len(x) - L)
    if hi <= lo:
        return None
    # Vectorised sliding correlation over the scan range
    win = np.lib.stride_tricks.sliding_window_view(x[lo:hi + L - 1], L)
    win = win - win.mean(axis=1, keepdims=True)
    wn = np.linalg.norm(win, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = (win @ k) / (wn * kn)
    amp = np.ptp(win, axis=1)
    hits = np.where((corr >= rc.next_spike_guard_min_corr)
                    & (amp >= rc.next_spike_guard_min_amp * kptp))[0]
    if len(hits) == 0:
        return None
    return int(lo + hits[0])


def _select_candidate(cands, rc, fs):
    """Pick the repolarisation wave among eligible candidates.

    ``cands`` : list of (peak_idx, prominence, sign). See
    ``RepolarizationConfig.repol_candidate_rule`` for the rationale.
    Returns the chosen tuple or None.
    """
    if not cands:
        return None
    top = max(cands, key=lambda c: c[1])
    rule = getattr(rc, 'repol_candidate_rule', 'max_prominence')
    if rule == 'max_prominence':
        return top
    if rule == 'prefer_positive':
        f = float(getattr(rc, 'repol_positive_min_rel_prom', 0.5))
        w = getattr(rc, 'repol_positive_max_offset_ms', 400.0)
        w_samples = None if w is None else w / 1000.0 * fs
        pos = [c for c in cands
               if c[2] > 0 and c[1] >= f * top[1]
               and (w_samples is None or abs(c[0] - top[0]) <= w_samples)]
        return max(pos, key=lambda c: c[1]) if pos else top
    raise ValueError(f"Unknown repol_candidate_rule: {rule!r}")


def find_repolarization_on_template(template, fs, pre_ms=50, cfg=None,
                                    median_bp_s=None, guard_after_ms=None, rr_low_s=None):
    """
    Find the repolarization wave on a clean averaged template.

    Uses the configured fpd_method (default: 'peak', the latency of the
    wave's peak; 'tangent' and the others in apply_fpd_method) after
    choosing the repolarization wave among the candidates
    (repol_candidate_rule).

    ``guard_after_ms`` : when the caller has lengthened the search window
    beyond what it would otherwise be (see
    ``RepolarizationConfig.window_rr_from_template_beats``), the latency
    where the old window ended. The extension beyond it is cut before the
    first depolarisation-like event, so the window never crosses into the
    next beat.

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
    # Adaptive search end: up to 85 % of the cycle on a regular rhythm,
    # never closer than a margin to where the next beat can start
    # (search_window_end_ms; ``rr_low_s`` = low percentile of the periods).
    fixed_end_ms = rc.search_end_ms
    effective_end_ms = search_window_end_ms(rc, median_bp_s, rr_low_s)
    if effective_end_ms > fixed_end_ms:
        logger.debug("Template search_end: adaptive %.0f ms > fixed %.0f ms (RR=%.0f ms, RR_low=%s)",
                     effective_end_ms, fixed_end_ms, (median_bp_s or 0) * 1000,
                     f"{rr_low_s * 1000:.0f} ms" if rr_low_s else "n/a")
    search_end = spike_idx + int(effective_end_ms / 1000 * fs)

    if search_end > len(template):
        search_end = len(template)

    # Guard the window EXTENSION (if any) against the next depolarisation.
    if guard_after_ms is not None:
        old_end = spike_idx + int(guard_after_ms / 1000 * fs)
        if old_end < search_end:
            cut = next_spike_cut_index(template, fs, spike_idx, old_end, search_end, cfg=rc)
            if cut is not None:
                margin = int(rc.next_spike_guard_margin_ms / 1000 * fs)
                new_end = max(old_end, cut - margin)
                logger.debug("Template window extension cut at %.0f ms (next-beat event at %.0f ms)",
                             (new_end - spike_idx) / fs * 1000, (cut - spike_idx) / fs * 1000)
                search_end = min(search_end, new_end)

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
    # Minimum FPD constraint: exclude peaks that would yield FPD below
    # the physiological floor (afterpotential rebound, not real T-wave).
    # Effective floor = max(fixed min_fpd_ms,
    #                        min(adaptive min_fpd_pct_rr × RR,
    #                            max_adaptive_min_fpd_ms)).
    # The cap keeps the adaptive contribution within physiological
    # bounds on bradycardic signals (RR > 2.5 s) where the raw product
    # would otherwise overshoot the real T-wave latency and force every
    # beat into the argmax fallback.
    fixed_min_fpd_ms = getattr(rc, 'min_fpd_ms', 0)
    adaptive_min_fpd_ms = 0.0
    pct_rr = getattr(rc, 'min_fpd_pct_rr', 0.0)
    if pct_rr > 0 and median_bp_s is not None and median_bp_s > 0:
        adaptive_min_fpd_ms = pct_rr * median_bp_s * 1000  # convert to ms
    adaptive_cap_ms = getattr(rc, 'max_adaptive_min_fpd_ms', 0.0)
    if adaptive_cap_ms and adaptive_cap_ms > 0 and adaptive_min_fpd_ms > adaptive_cap_ms:
        logger.debug("Template min_fpd: adaptive %.0f ms capped at %.0f ms "
                     "(%.0f%% of RR=%.0f ms)",
                     adaptive_min_fpd_ms, adaptive_cap_ms,
                     pct_rr * 100, median_bp_s * 1000)
        adaptive_min_fpd_ms = adaptive_cap_ms
    effective_min_fpd_ms = max(fixed_min_fpd_ms, adaptive_min_fpd_ms)
    min_fpd_samples = int(effective_min_fpd_ms / 1000 * fs)
    if adaptive_min_fpd_ms > fixed_min_fpd_ms:
        logger.debug("Template min_fpd: adaptive %.0f ms > fixed %.0f ms "
                     "(%.0f%% of RR=%.0f ms)",
                     adaptive_min_fpd_ms, fixed_min_fpd_ms,
                     pct_rr * 100, median_bp_s * 1000)
    # Convert to minimum peak index within the search segment:
    # peak_idx in seg → FPD_samples = search_start + peak_idx - spike_idx
    # We want FPD_samples >= min_fpd_samples, i.e.
    # peak_idx >= min_fpd_samples - (search_start - spike_idx)
    min_pk_idx = max(0, min_fpd_samples - (search_start - spike_idx))

    best_pk = None
    best_prom = 0
    best_sign = 1

    # Collect every eligible candidate of both polarities, then let the
    # configured rule pick (see RepolarizationConfig.repol_candidate_rule).
    peak_dist = int(rc.peak_min_distance_ms / 1000 * fs)
    candidates = []
    for sign in [1, -1]:
        threshold = np.std(seg_det) * rc.peak_prominence_factor
        pks, props = sig.find_peaks(sign * seg_det,
                                     prominence=threshold,
                                     distance=peak_dist)
        for j in range(len(pks)):
            # Exclude peaks that would give FPD < min_fpd_ms
            if pks[j] < min_pk_idx:
                continue
            candidates.append((int(pks[j]), float(props['prominences'][j]), sign))
    chosen = _select_candidate(candidates, rc, fs)
    if chosen is not None:
        best_pk, best_prom, best_sign = chosen

    if best_pk is None:
        # No qualifying peak found by find_peaks in the search window →
        # the repolarization (T-wave) is not measurable for this template.
        # Return None instead of forcing a spurious FPD via argmax(|seg_det|).
        # The old argmax fallback fabricated an FPD wherever the T-wave was
        # flat/absent (e.g. slow-RR after-potential signals): it picked the
        # largest excursion of the segment, which then passed the SNR gate
        # and was reported as a normal FPD with noRepol≈0%. find_peaks with
        # its prominence threshold IS the detectability test — if it finds
        # nothing qualifying, there is nothing to measure.
        # Reference: EFP Analyzer (Patel et al., Sci Rep 2025) — traces with
        # non-detectable repolarization are excluded, not force-measured.
        logger.debug("Template: no qualifying repolarization peak in window "
                     "→ repolarization not measurable")
        return None, 1, 0.0, 0.0, None, None

    # Defensive sanity: best_pk must be a valid index into seg_det by now.
    if best_pk < 0 or best_pk >= len(seg_det):
        return None, 1, 0.0, 0.0, None, None

    # ─── Repolarization detectability gate ───
    # If the best candidate's prominence is too low relative to noise,
    # the repolarization is not detectable (flat T-wave / drug effect).
    # Return None instead of forcing a spurious FPD measurement.
    # Reference: EFP Analyzer (Patel et al., Sci Rep 2025) — traces with
    # non-detectable repolarization are excluded, not force-measured.
    # Noise is estimated excluding the peak region so the repol bump
    # itself doesn't inflate the noise estimate.
    if best_pk is not None:
        pk_excl_s = max(0, best_pk - int(0.030 * fs))
        pk_excl_e = min(len(seg_det), best_pk + int(0.030 * fs))
        noise_region_t = np.concatenate([
            seg_det[:pk_excl_s], seg_det[pk_excl_e:]
        ]) if pk_excl_s > 5 or pk_excl_e < len(seg_det) - 5 else seg_det
        noise_level_gate = np.std(noise_region_t) if len(noise_region_t) > 10 else np.std(seg_det)
    else:
        noise_level_gate = np.std(seg_det)
    if np.isnan(noise_level_gate) or noise_level_gate <= 0:
        noise_level_gate = 0.0
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
                                 cfg=None,
                                 beat_period_s=None,
                                 next_rr_s=None):
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
        # Adaptive search end for the unguided per-beat search (the beat's
        # own next period bounds it, when known)
        effective_end_ms = search_window_end_ms(rc, beat_period_s, next_rr_s)
        peak_search_end = spike_idx + int(effective_end_ms / 1000 * fs)

    # The beat's own next depolarisation bounds every per-beat window: the
    # segment is long enough to hold it on short cycles, and a spike is
    # more prominent than any T-wave (v3.14.2).
    if next_rr_s is not None and next_rr_s > 0:
        margin_ms = float(getattr(rc, 'search_end_next_beat_margin_ms', 0.0) or 0.0)
        cap = spike_idx + int((next_rr_s * 1000.0 - margin_ms) / 1000 * fs)
        if cap < peak_search_end:
            peak_search_end = cap

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
    # Minimum FPD constraint: exclude peaks in the afterpotential zone.
    # Effective floor = max(fixed min_fpd_ms,
    #                        min(adaptive min_fpd_pct_rr × RR,
    #                            max_adaptive_min_fpd_ms)).
    # See the matching comment in find_repolarization_on_template for a
    # detailed explanation of the cap; it prevents bradycardic signals
    # (RR > 2.5 s) from producing a floor that overshoots the real
    # T-wave latency.
    fixed_min_fpd_ms = getattr(rc, 'min_fpd_ms', 0)
    adaptive_min_fpd_ms = 0.0
    pct_rr = getattr(rc, 'min_fpd_pct_rr', 0.0)
    if pct_rr > 0 and beat_period_s is not None and beat_period_s > 0:
        adaptive_min_fpd_ms = pct_rr * beat_period_s * 1000
    adaptive_cap_ms = getattr(rc, 'max_adaptive_min_fpd_ms', 0.0)
    if adaptive_cap_ms and adaptive_cap_ms > 0 and adaptive_min_fpd_ms > adaptive_cap_ms:
        logger.debug("Per-beat min_fpd: adaptive %.0f ms capped at %.0f ms "
                     "(%.0f%% of RR=%.0f ms)",
                     adaptive_min_fpd_ms, adaptive_cap_ms,
                     pct_rr * 100, beat_period_s * 1000)
        adaptive_min_fpd_ms = adaptive_cap_ms
    effective_min_fpd_ms = max(fixed_min_fpd_ms, adaptive_min_fpd_ms)
    min_fpd_samples = int(effective_min_fpd_ms / 1000 * fs)
    min_pk_idx = max(0, min_fpd_samples - (peak_search_start - spike_idx))

    best_idx = None
    best_score = 0
    # Polarity of the peak that won the search below.  This MUST be carried
    # into apply_fpd_method: the tangent / 50% / baseline-return maths is
    # polarity-dependent, so running it with the template's sign on a beat
    # whose repolarization is inverted produces a meaningless endpoint that
    # is then reported as a normal FPD.
    #
    # Defaults to the template sign so that the argmax fallback path below
    # (which does not run find_peaks and so has no winning polarity) keeps
    # its previous behaviour.
    #
    # The template path already does this correctly — see ``best_sign`` in
    # find_repolarization_on_template — and the asymmetry between the two
    # was the bug.
    best_sign = template_repol_sign
    peak_dist = int(rc.per_beat_peak_distance_ms / 1000 * fs)
    dist_penalty_scale = rc.per_beat_distance_penalty_ms / 1000 * fs

    # Use MAD-based noise estimate for the prominence threshold.
    # MAD is more robust than std to outliers (depol remnants, artifacts)
    # that inflate the noise estimate and cause find_peaks to miss real
    # repolarization bumps.  Use the lower of MAD-based and std-based to
    # be more sensitive, but keep a floor so that truly flat (no-repol)
    # signals don't trigger on pure noise.
    std_noise = np.std(seg_det)
    mad_noise = np.median(np.abs(seg_det - np.median(seg_det)))
    robust_noise = 1.4826 * mad_noise if mad_noise > 0 else std_noise
    # Use the lower estimate (more sensitive) but not less than 40% of std
    effective_noise = max(min(robust_noise, std_noise), 0.4 * std_noise)
    # Per-beat uses a lower prominence factor than template (noisier)
    prom_factor = getattr(rc, 'per_beat_prominence_factor', rc.peak_prominence_factor)
    repol_prom_threshold = effective_noise * prom_factor

    prefer_same_sign = getattr(rc, 'per_beat_prefer_template_sign', False)
    for sign in [template_repol_sign, -template_repol_sign]:
        if prefer_same_sign and sign != template_repol_sign and best_idx is not None:
            # A same-sign repolarisation peak qualified: do not let the
            # opposite lobe of a biphasic wave compete with it.
            break
        pks, props = sig.find_peaks(sign * seg_det,
                                     prominence=repol_prom_threshold,
                                     distance=peak_dist)
        if len(pks) > 0:
            # Filter out peaks that would give FPD < min_fpd_ms
            valid = pks >= min_pk_idx
            pks = pks[valid]
            proms = props['prominences'][valid]
            if len(pks) == 0:
                continue

            if template_peak_samples is not None:
                expected_local = template_peak_samples - (peak_search_start - spike_idx)
                distances = np.abs(pks - expected_local)
                scores = proms / (1 + distances / dist_penalty_scale)
                best_pk = pks[np.argmax(scores)]
                score = np.max(scores)
            elif template_fpd_samples is not None:
                expected_local = template_fpd_samples - int(0.050 * fs) - (peak_search_start - spike_idx)
                distances = np.abs(pks - expected_local)
                scores = proms / (1 + distances / dist_penalty_scale)
                best_pk = pks[np.argmax(scores)]
                score = np.max(scores)
            else:
                best_pk = pks[np.argmax(proms)]
                score = np.max(proms)

            if score > best_score:
                best_score = score
                best_idx = best_pk
                best_sign = sign

    if best_idx is None:
        # No qualifying repolarization peak found by find_peaks → the T-wave
        # is not measurable for this beat. Return None instead of forcing a
        # spurious FPD via argmax(|seg_det|). The old fallback picked the
        # largest excursion of the segment (either polarity), which then
        # passed the SNR gate and was reported as a normal per-beat FPD even
        # on flat/absent T-waves. find_peaks with its prominence threshold is
        # the detectability test — nothing qualifying means nothing to measure.
        logger.debug("Per-beat: no qualifying repolarization peak in window "
                     "→ repolarization not measurable")
        return None, np.nan, None, None

    # Defensive sanity: any path that produced best_idx out-of-bounds
    # means the repolarization search is not reliable for this beat.
    if best_idx < 0 or best_idx >= len(seg_det):
        return None, np.nan, None, None

    # ─── Repolarization detectability gate (per-beat) ───
    # Prominence: use the best_score from find_peaks if it came from
    # find_peaks (actual prominence), otherwise fall back to abs(peak value).
    # Noise is estimated excluding ±50ms around the peak.
    # The score tested is the distance-penalised prominence (best_score),
    # not the raw prominence. Guided and unguided searches have their own
    # thresholds (repol_gate_min_snr_beat, repol_gate_min_snr_beat_unguided;
    # both 1.5 by default).
    if rc.enable_repol_gate:
        excl_half = int(0.050 * fs)
        peak_excl_start = max(0, best_idx - excl_half)
        peak_excl_end = min(len(seg_det), best_idx + excl_half)
        beat_prom = best_score if best_score > 0 else np.abs(seg_det[best_idx])
        noise_region = np.concatenate([
            seg_det[:peak_excl_start],
            seg_det[peak_excl_end:]
        ]) if (peak_excl_start > 5 and peak_excl_end < len(seg_det) - 5) else seg_det
        noise_level_beat = np.std(noise_region) if len(noise_region) > 10 else np.std(seg_det)
        if noise_level_beat > 0:
            # Template-guided → lenient; unguided → stricter
            is_guided = (template_fpd_samples is not None or template_peak_samples is not None)
            snr_threshold = rc.repol_gate_min_snr_beat if is_guided else (
                getattr(rc, 'repol_gate_min_snr_beat_unguided', rc.repol_gate_min_snr_beat)
            )
            beat_repol_snr = beat_prom / noise_level_beat
            if beat_repol_snr < snr_threshold:
                logger.debug("Per-beat repol gate: SNR=%.2f < %.1f (guided=%s) → not detectable",
                             beat_repol_snr, snr_threshold, is_guided)
                return None, np.nan, None, None

    # ─── Step 2: Apply configured FPD method ───
    # ``best_sign`` — not ``template_repol_sign`` — is the polarity of the
    # peak actually selected above.  They differ on beats whose T-wave is
    # inverted relative to the template, which is a real regime here:
    # parameters.py deliberately flips the sign for inverted beats.
    # 'consensus' votes among five methods; before v3.14.1 the per-beat
    # call fell through to 'tangent' silently, so template and per-beat
    # values came from different methods. Now the same vote runs per beat.
    if rc.fpd_method == 'consensus':
        fpd_idx = consensus_fpd(seg_det, best_idx, best_sign, fs,
                                peak_search_start, spike_idx, cfg=cfg)[0]
    else:
        fpd_idx = apply_fpd_method(seg_det, best_idx, best_sign, fs,
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
