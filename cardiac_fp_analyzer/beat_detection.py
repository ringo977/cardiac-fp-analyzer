"""
beat_detection.py — Robust beat detection for hiPSC-CM field potentials.

Multi-strategy approach:
  1. Prominence-based peak detection on the signal envelope
  2. Derivative confirmation (fast dV/dt at depolarization)
  3. Adaptive thresholding with local noise estimation
  4. Auto-selection of best method
"""

import logging

import numpy as np
from scipy import signal as sig

logger = logging.getLogger(__name__)


def _get_bd_cfg(cfg=None):
    """Return a BeatDetectionConfig, creating default if needed."""
    if cfg is not None:
        return cfg
    from .config import BeatDetectionConfig
    return BeatDetectionConfig()


def detect_beats(data, fs, method='auto', min_distance_ms=400,
                 threshold_factor=4.0, channel_data=None, cfg=None):
    """
    Detect beat (depolarization spike) locations in FP signal.

    Parameters
    ----------
    data : array — filtered FP signal
    fs : float — sampling rate
    method : str — 'auto' (recommended), 'prominence', 'derivative', or 'peak'
    min_distance_ms : float — minimum inter-beat interval in ms
    threshold_factor : float — multiplier for adaptive threshold
    cfg : BeatDetectionConfig or None — if provided, overrides method/min_distance_ms/threshold_factor

    Returns
    -------
    beat_indices : array of int
    beat_times : array of float
    info : dict — detection diagnostics
    """
    if cfg is not None:
        method = cfg.method
        min_distance_ms = cfg.min_distance_ms
        threshold_factor = cfg.threshold_factor

    data = np.array(data, dtype=np.float64)

    # ── Input validation ──
    if len(data) == 0:
        logger.warning("detect_beats: empty signal")
        return np.array([], dtype=int), np.array([]), {'method': method, 'n_beats': 0, 'polarity': 'unknown'}
    if fs <= 0:
        raise ValueError(f"detect_beats: sampling rate must be positive, got {fs}")
    if np.all(np.isnan(data)):
        logger.warning("detect_beats: signal is all NaN")
        return np.array([], dtype=int), np.array([]), {'method': method, 'n_beats': 0, 'polarity': 'unknown'}
    if len(data) < int(fs * 0.5):
        logger.warning("detect_beats: signal shorter than 0.5 s (%d samples at %.0f Hz)",
                        len(data), fs)
        return np.array([], dtype=int), np.array([]), {'method': method, 'n_beats': 0, 'polarity': 'unknown'}

    # Replace any NaN with 0 to prevent propagation
    if np.any(np.isnan(data)):
        logger.warning("detect_beats: %d NaN samples replaced with 0", np.sum(np.isnan(data)))
        data = np.nan_to_num(data, nan=0.0)

    min_dist = int(min_distance_ms * fs / 1000)

    if method == 'auto':
        return _detect_auto(data, fs, min_dist, threshold_factor, cfg=cfg)
    elif method == 'prominence':
        return _detect_prominence(data, fs, min_dist, threshold_factor)
    elif method == 'derivative':
        return _detect_derivative(data, fs, min_dist, threshold_factor, cfg=cfg)
    elif method == 'peak':
        return _detect_peak(data, fs, min_dist, threshold_factor)
    else:
        raise ValueError(f"Unknown method: {method}")


def _recover_missed_beats(data, fs, beat_indices, cfg=None):
    """Recover missed beats using periodicity-guided search.

    After initial detection + morphological validation, this function:
    1. Estimates the median beat period from validated beats.
    2. Computes reference amplitude and dV/dt from validated beats.
    3. Identifies gaps where a beat is expected but missing.
    4. In each gap, finds ALL local peaks and accepts the best candidate
       that passes ANY of three criteria:
       - template correlation ≥ threshold (shape match)
       - local amplitude ≥ fraction of reference (strong enough peak)
       - local |dV/dt| ≥ fraction of reference (fast enough deflection)
    This multi-criteria approach is robust on noisy 3D construct signals
    where template correlation alone fails due to low SNR.

    Parameters
    ----------
    data : array — filtered FP signal
    fs : float — sampling rate
    beat_indices : array of int — initially detected beat locations
    cfg : BeatDetectionConfig or None

    Returns
    -------
    recovered_indices : array of int — merged and sorted beat locations
    recovery_info : dict — diagnostics
    """
    c = _get_bd_cfg(cfg)

    if not c.enable_beat_recovery or len(beat_indices) < 4:
        return beat_indices, {'recovery': 'skipped', 'reason': 'too_few_beats'}

    # ── Step 1: Estimate median beat period ──
    bp_samples = np.diff(beat_indices)
    median_bp = np.median(bp_samples)

    if median_bp <= 0:
        return beat_indices, {'recovery': 'skipped', 'reason': 'invalid_period'}

    # ── Step 2: Build template + reference metrics from detected beats ──
    pre = int(20 * fs / 1000)   # 20 ms before spike
    post = int(80 * fs / 1000)  # 80 ms after spike
    window = int(20 * fs / 1000)  # ±20 ms for amplitude measurement

    template_segs = []
    beat_amplitudes = []
    beat_dvdt_max = []
    dt = 1.0 / fs

    for bi in beat_indices:
        s, e = bi - pre, bi + post
        ws, we = max(0, bi - window), min(len(data), bi + window)
        if s >= 0 and e < len(data):
            seg = data[s:e]
            template_segs.append(seg)
            # Amplitude: peak-to-peak in ±20ms window
            local_seg = data[ws:we]
            beat_amplitudes.append(np.max(local_seg) - np.min(local_seg))
            # Max |dV/dt| in ±20ms window
            local_deriv = np.abs(np.gradient(local_seg, dt))
            beat_dvdt_max.append(np.max(local_deriv))

    if len(template_segs) < 3:
        return beat_indices, {'recovery': 'skipped', 'reason': 'cannot_build_template'}

    min_len = min(len(seg) for seg in template_segs)
    mat = np.array([seg[:min_len] for seg in template_segs])
    template = np.median(mat, axis=0)
    t_c = template - np.mean(template)
    t_c_ss = np.sum(t_c**2)

    # Reference values: median (robust to outliers)
    ref_amplitude = np.median(beat_amplitudes)
    ref_dvdt = np.median(beat_dvdt_max)

    # Adaptive jitter for template matching
    jitter_max = 0
    if c.enable_jitter_correction and t_c_ss > 0:
        if c.jitter_adaptive:
            jitter_max = _estimate_jitter_from_template(
                template, fs, c.jitter_adaptive_fraction
            )
        if jitter_max < 1:
            jitter_max = max(1, int(c.jitter_max_shift_ms / 1000.0 * fs))

    # ── Step 3: Find gaps and search for missed beats ──
    tolerance = c.recovery_search_tolerance * median_bp
    min_gap_samples = int(median_bp * 0.3)
    recovered = []
    all_bi = list(beat_indices)

    def _is_too_close(cg):
        """Check if candidate is too close to existing or recovered beats."""
        for b in all_bi:
            if abs(cg - b) < min_gap_samples:
                return True
        return any(abs(cg - r) < min_gap_samples for r in recovered)

    def _score_candidate(cg):
        """Score a candidate position. Returns (score, corr, amp_ratio, dvdt_ratio).
        Score > 0 means at least one acceptance criterion is met."""
        score = 0.0

        # Local amplitude
        ws = max(0, cg - window)
        we = min(len(data), cg + window)
        local_seg = data[ws:we]
        amp = np.max(local_seg) - np.min(local_seg)
        amp_ratio = amp / ref_amplitude if ref_amplitude > 0 else 0

        # Local |dV/dt|
        local_deriv = np.abs(np.gradient(local_seg, dt))
        dvdt = np.max(local_deriv)
        dvdt_ratio = dvdt / ref_dvdt if ref_dvdt > 0 else 0

        # Template correlation with jitter
        # Use abs(corr) when morphology_accept_inverted is True, so that
        # inverted beats (negative correlation) can still be recovered.
        corr = -1.0
        if t_c_ss > 0:
            cs = cg - pre - jitter_max
            ce = cg + post + jitter_max
            if cs >= 0 and ce < len(data):
                beat_wide = data[cs:ce]
                accept_inv = c.morphology_accept_inverted
                for lag in range(-jitter_max, jitter_max + 1):
                    offset = jitter_max + lag
                    beat_seg = beat_wide[offset:offset + min_len]
                    if len(beat_seg) < min_len:
                        continue
                    b_c = beat_seg - np.mean(beat_seg)
                    denom = np.sqrt(np.sum(b_c**2) * t_c_ss)
                    if denom > 0:
                        r = np.sum(b_c * t_c) / denom
                        effective_r = abs(r) if accept_inv else r
                        if effective_r > corr:
                            corr = effective_r

        # Acceptance logic:
        #   - correlation alone is sufficient (shape match is reliable)
        #   - amplitude + dV/dt together are sufficient (two independent
        #     physiological features agree — unlikely to both be noise)
        #   - any single non-correlation criterion alone is NOT sufficient
        #     (noise spikes can have high amplitude or high dV/dt, but
        #     rarely both)
        criteria_met = 0
        if corr >= c.recovery_min_corr:
            criteria_met += 1
            score += 2.0 + corr  # correlation is the strongest signal
        if amp_ratio >= c.recovery_min_amplitude_ratio:
            criteria_met += 1
            score += amp_ratio
        if dvdt_ratio >= c.recovery_min_dvdt_ratio:
            criteria_met += 1
            score += dvdt_ratio

        # Need correlation alone, or both amplitude + dV/dt
        if corr < c.recovery_min_corr and criteria_met < 2:
            score = 0.0

        return score, corr, amp_ratio, dvdt_ratio

    def _search_gap(expected):
        """Search for best beat candidate near an expected position."""
        search_start = max(0, int(expected - tolerance))
        search_end = min(len(data), int(expected + tolerance))
        if search_end - search_start < window * 2:
            return None

        seg = data[search_start:search_end]

        # Find ALL local peaks in the search window (both polarities)
        noise_std = np.std(seg)
        low_prom = max(noise_std * 0.3, 1e-6)
        candidates_local = set()

        for s_data in [seg, -seg]:
            pks, _ = sig.find_peaks(s_data, prominence=low_prom,
                                    distance=max(3, int(2 * fs / 1000)))
            candidates_local.update(pks.tolist())

        # Also add argmax of absolute deviation
        abs_seg = np.abs(seg - np.median(seg))
        candidates_local.add(int(np.argmax(abs_seg)))

        # Also add the position closest to expected
        center = int(expected - search_start)
        if 0 <= center < len(seg):
            candidates_local.add(center)

        # Score each candidate — prefer those close to expected position
        best_score = 0.0
        best_global = None

        for cl in candidates_local:
            cg = search_start + cl
            if _is_too_close(cg):
                continue
            # Penalise candidates far from expected position
            dist_from_expected = abs(cg - expected)
            if dist_from_expected > tolerance:
                continue
            score, _, _, _ = _score_candidate(cg)
            # Distance penalty: reduce score for candidates far from center
            proximity = 1.0 - (dist_from_expected / tolerance)
            score *= (0.5 + 0.5 * proximity)
            if score > best_score:
                best_score = score
                best_global = cg

        if best_global is not None and best_score > 0:
            return best_global
        return None

    # ── Scan gaps between consecutive beats ──
    # Only fill gaps up to 3.5× the median period (1–2 missed beats).
    # Larger gaps indicate cessation or discontinuity — do not fill.
    max_gap = median_bp * 3.5
    for i in range(len(all_bi) - 1):
        gap = all_bi[i + 1] - all_bi[i]
        if gap > max_gap:
            continue
        n_missing = round(gap / median_bp) - 1
        if n_missing < 1:
            continue
        for k in range(1, n_missing + 1):
            expected = int(all_bi[i] + k * median_bp)
            result = _search_gap(expected)
            if result is not None:
                recovered.append(result)
                all_bi = sorted(all_bi + [result])

    # ── Scan before first beat ──
    # Only look if the gap is 1.8–3.5× median_bp (same ceiling as inter-beat).
    first = beat_indices[0]
    if median_bp * 1.8 < first < max_gap:
        n_before = round(first / median_bp)
        for k in range(1, n_before + 1):
            expected = int(first - k * median_bp)
            if expected < window:
                break
            result = _search_gap(expected)
            if result is not None:
                recovered.append(result)
                all_bi = sorted(all_bi + [result])

    # ── Scan after last beat ──
    last = beat_indices[-1]
    remaining = len(data) - last
    if median_bp * 1.8 < remaining < max_gap:
        n_after = round(remaining / median_bp)
        for k in range(1, n_after + 1):
            expected = int(last + k * median_bp)
            if expected + window >= len(data):
                break
            result = _search_gap(expected)
            if result is not None:
                recovered.append(result)
                all_bi = sorted(all_bi + [result])

    # Merge
    merged = np.array(sorted(set(list(beat_indices) + recovered)), dtype=int)

    info = {
        'recovery': 'applied',
        'n_initially_detected': len(beat_indices),
        'n_recovered': len(recovered),
        'n_total': len(merged),
        'median_bp_ms': median_bp / fs * 1000,
        'ref_amplitude': float(ref_amplitude),
        'ref_dvdt': float(ref_dvdt),
    }
    logger.info("Beat recovery: %d detected + %d recovered = %d total "
                "(ref_amp=%.1f, ref_dvdt=%.0f)",
                len(beat_indices), len(recovered), len(merged),
                ref_amplitude, ref_dvdt)

    return merged, info


def _detect_auto(data, fs, min_dist, threshold_factor, cfg=None):
    """Auto-select best method based on physiological plausibility."""
    c = _get_bd_cfg(cfg)
    results = []
    for method_fn, name in [(_detect_prominence, 'prominence'),
                             (_detect_derivative, 'derivative'),
                             (_detect_peak, 'peak')]:
        try:
            if name == 'derivative':
                bi, bt, info = method_fn(data, fs, min_dist, threshold_factor, cfg=cfg)
            else:
                bi, bt, info = method_fn(data, fs, min_dist, threshold_factor)
            bp = np.diff(bi) / fs if len(bi) > 1 else np.array([])
            score = 0
            if len(bp) > 2:
                mean_bp = np.mean(bp)
                cv_bp = np.std(bp) / mean_bp if mean_bp > 0 else 999
                if c.bp_ideal_range_s[0] <= mean_bp <= c.bp_ideal_range_s[1]:
                    score += c.score_bp_ideal
                elif c.bp_extended_range_s[0] <= mean_bp <= c.bp_extended_range_s[1]:
                    score += c.score_bp_extended
                if cv_bp < c.cv_good_frac: score += c.score_cv_good
                elif cv_bp < c.cv_fair_frac: score += c.score_cv_fair
                elif cv_bp < c.cv_marginal_frac: score += c.score_cv_marginal
                duration_s = len(data) / fs
                if duration_s/3 <= len(bi) <= duration_s/0.3:
                    score += c.score_rate_ok
                elif len(bi) > 3:
                    score += c.score_rate_low
                if len(bi) > duration_s/0.3:
                    score += c.score_rate_excess
            else:
                score = c.score_too_few
            # Derivative bonus: dV/dt is the physiologically correct way
            # to find depolarisation spikes (fastest deflection).  This
            # prevents prominence from winning when a large, slow
            # repolarisation wave is mistaken for a beat.
            if name == 'derivative' and score > c.score_too_few:
                score += c.score_derivative_bonus

            info['_score'] = score
            info['_method_name'] = name
            results.append((bi, bt, info))
        except (ValueError, IndexError, RuntimeError) as e:
            logger.debug("Beat detection method %s failed: %s", name, e)
            continue

    if not results:
        return np.array([], dtype=int), np.array([]), {'method': 'auto', 'n_beats': 0, 'polarity': 'unknown'}

    best = max(results, key=lambda x: x[2].get('_score', -999))
    bi, bt, info = best

    # ── Bimodal BP correction ──
    # Detect alternating short/long beat periods (T-wave false positives).
    # If the short group ≈ half the long group, the short intervals are
    # T-wave repolarisation peaks mistaken for depolarisation spikes.
    # Fix: increase min_distance to midpoint between groups and re-detect.
    bi, bt, info = _fix_bimodal_bp(data, fs, bi, bt, info, threshold_factor, cfg)

    # ── Post-detection morphological validation ──
    # Remove noise spikes before they reach segmentation/parameter extraction.
    # Pass detected polarity so validation can adapt for mixed-polarity signals.
    detected_polarity = info.get('polarity', 'positive')
    bi, val_info = validate_beats_morphology(data, fs, bi, cfg=cfg,
                                              detected_polarity=detected_polarity)
    bt = bi / fs
    info['beat_validation'] = val_info

    # ── Beat recovery: find missed beats using periodicity ──
    # Runs AFTER morphological validation so that:
    #   1. The template is built from clean, validated beats.
    #   2. Recovered beats are not re-validated (recovery already checks
    #      template correlation), preventing the validator from discarding
    #      weaker but genuine beats that were just recovered.
    bi, recovery_info = _recover_missed_beats(data, fs, bi, cfg=cfg)
    bt = bi / fs
    info['n_beats'] = len(bi)
    info['beat_recovery'] = recovery_info

    # (No post-detection amplitude/quality gate — false positive rejection
    # is handled by morphological validation and bimodal correction.)

    info['n_beats'] = len(bi)

    info['method'] = f"auto({info.get('_method_name', '?')})"
    for k in list(info.keys()):
        if k.startswith('_'): del info[k]
    return bi, bt, info


def _fix_bimodal_bp(data, fs, bi, bt, info, threshold_factor, cfg=None):
    """If beat periods show bimodal short/long pattern, re-detect with larger min_distance.

    Uses Otsu-style threshold to find the optimal split that minimises
    within-group variance.  If the two groups have a ratio ≈ 1.5–2.8×
    (consistent with spike + T-wave double-counting), the short intervals
    are T-wave artefacts and we re-detect with a larger min_distance.
    """
    if len(bi) < 10:
        return bi, bt, info

    bp_samples = np.diff(bi)
    bp_ms = bp_samples / fs * 1000

    # Otsu-style bimodal split: sweep percentiles, minimise total within-group variance
    best_thresh, best_score = 0, float('inf')
    for pct in range(15, 85, 3):
        thresh = np.percentile(bp_ms, pct)
        lo = bp_ms[bp_ms <= thresh]
        hi = bp_ms[bp_ms > thresh]
        if len(lo) < 3 or len(hi) < 3:
            continue
        score = np.var(lo) * len(lo) + np.var(hi) * len(hi)
        if score < best_score:
            best_score, best_thresh = score, thresh

    short = bp_ms[bp_ms <= best_thresh]
    long_ = bp_ms[bp_ms > best_thresh]

    if len(short) < 3 or len(long_) < 3:
        return bi, bt, info

    mean_short = np.mean(short)
    mean_long = np.mean(long_)
    ratio = mean_long / mean_short if mean_short > 0 else 0

    # Check separation quality: groups must be well-separated
    # (gap between groups relative to their spread)
    combined_std = (np.std(short) + np.std(long_)) / 2
    gap = mean_long - mean_short
    separation = gap / combined_std if combined_std > 0 else 0

    # Bimodal pattern: long ≈ 1.5–2.8× short, both groups sizeable, well-separated
    if (1.4 < ratio < 3.0
            and min(len(short), len(long_)) > 0.15 * len(bp_ms)
            and separation > 2.0):
        # The true beat period is the long one; short ones are T-wave artefacts.
        # Set min_distance to midpoint so T-wave peaks are suppressed.
        new_min_dist_ms = (mean_short + mean_long) / 2
        new_min_dist = int(new_min_dist_ms * fs / 1000)

        method_name = info.get('_method_name', 'prominence')
        method_map = {
            'prominence': _detect_prominence,
            'derivative': _detect_derivative,
            'peak': _detect_peak,
        }
        method_fn = method_map.get(method_name, _detect_prominence)
        try:
            if method_name == 'derivative':
                bi2, bt2, info2 = method_fn(data, fs, new_min_dist, threshold_factor, cfg=cfg)
            else:
                bi2, bt2, info2 = method_fn(data, fs, new_min_dist, threshold_factor)

            # Verify improvement: CV should decrease
            bp2 = np.diff(bi2) / fs if len(bi2) > 1 else np.array([])
            old_cv = np.std(bp_ms) / np.mean(bp_ms) if np.mean(bp_ms) > 0 else 999
            new_cv = np.std(bp2 * 1000) / np.mean(bp2 * 1000) if len(bp2) > 1 and np.mean(bp2) > 0 else 999

            if new_cv < old_cv and len(bi2) >= 5:
                info2['_method_name'] = method_name
                info2['_score'] = info.get('_score', 0) + 10
                info2['bimodal_correction'] = (
                    f'T-wave artefact detected (short={mean_short:.0f}ms, '
                    f'long={mean_long:.0f}ms, ratio={ratio:.1f}x). '
                    f'min_distance adjusted to {new_min_dist_ms:.0f}ms. '
                    f'CV improved {old_cv*100:.1f}% → {new_cv*100:.1f}%'
                )
                return bi2, bt2, info2
        except (ValueError, IndexError, RuntimeError) as e:
            logger.debug("Bimodal correction failed: %s", e)

    return bi, bt, info


def _detect_prominence(data, fs, min_dist, threshold_factor):
    """Prominence-based: robust to baseline drift and noise."""
    pos_ext = np.percentile(data, 99.9) - np.median(data)
    neg_ext = np.median(data) - np.percentile(data, 0.1)
    signal_for_peaks = data if pos_ext >= neg_ext else -data
    polarity = 'positive' if pos_ext >= neg_ext else 'negative'

    p5, p95 = np.percentile(signal_for_peaks, [5, 95])
    noise_mask = (signal_for_peaks >= p5) & (signal_for_peaks <= p95)
    noise_std = np.std(signal_for_peaks[noise_mask]) if np.sum(noise_mask) > 100 else np.std(signal_for_peaks)
    min_prom = threshold_factor * noise_std

    peaks, props = sig.find_peaks(signal_for_peaks, distance=min_dist, prominence=min_prom, width=1)
    if len(peaks) < len(data)/fs/5.0 and threshold_factor > 2:
        peaks2, _ = sig.find_peaks(signal_for_peaks, distance=min_dist, prominence=min_prom*0.5, width=1)
        if len(peaks2) > len(peaks): peaks = peaks2

    return peaks, peaks / fs, {'method': 'prominence', 'polarity': polarity, 'n_beats': len(peaks)}


def _detect_derivative(data, fs, min_dist, threshold_factor, cfg=None):
    """Derivative-based: detects steepest slope at depolarization.

    Two-pass approach:
      Pass 1 — standard threshold (threshold_factor × noise_std) finds
               strong beats and establishes the rhythm.
      Pass 2 — lower threshold (threshold_factor × 0.5) searches only
               in gaps where a beat is expected from Pass 1's rhythm.
               This recovers weak beats that fall below the main threshold
               without adding false positives elsewhere.
    """
    c = _get_bd_cfg(cfg)
    dt = 1.0 / fs
    deriv = np.gradient(data, dt)
    win = max(3, int(c.deriv_smooth_ms / 1000.0 * fs))
    if win % 2 == 0: win += 1
    deriv_smooth = sig.savgol_filter(deriv, win, 2) if fs >= 1000 else deriv

    best_peaks, best_pol = np.array([], dtype=int), 'positive'
    best_score = -999
    best_ns = 1.0  # noise std for best polarity

    # Per-polarity detection results (used for merged approach)
    polarity_results = {}

    # Try positive and negative derivative separately.
    # Use two noise estimates: std-based (for threshold) and MAD-based
    # (more robust to large spikes).  The lower of the two is used for
    # prominence, so small but real beats are not lost.
    for sign, pol in [(1, 'positive'), (-1, 'negative')]:
        d = sign * deriv_smooth
        p5, p95 = np.percentile(d, [5, 95])
        mask = (d >= p5) & (d <= p95)
        ns_std = np.std(d[mask]) if np.sum(mask) > 100 else np.std(d)
        # MAD-based noise: robust to outliers (large spikes)
        mad = np.median(np.abs(d - np.median(d)))
        ns_mad = 1.4826 * mad if mad > 0 else ns_std
        # Use the lower (MAD-based) estimate so that small beats in
        # mixed-amplitude signals are not missed.  Morphological
        # validation downstream removes false positives.
        # Floor at 50% of std to avoid near-zero thresholds on very
        # clean signals where MAD ≈ 0.
        ns = max(min(ns_std, ns_mad), 0.5 * ns_std)
        thresh = np.median(d) + threshold_factor * ns
        pks, _ = sig.find_peaks(d, height=thresh, distance=min_dist, prominence=ns)
        polarity_results[pol] = (pks, ns)
        if len(pks) > 2:
            bp = np.diff(pks) / fs
            cv = np.std(bp) / np.mean(bp) if np.mean(bp) > 0 else 999
            score = -cv
            if score > best_score:
                best_score, best_peaks, best_pol = score, pks, pol
                best_ns = ns

    # ── Pass 2: low-threshold search in rhythm gaps (same polarity) ──
    if len(best_peaks) >= 4:
        bp_samples = np.diff(best_peaks)
        median_bp = np.median(bp_samples)
        if median_bp > 0:
            sign = 1 if best_pol == 'positive' else -1
            d = sign * deriv_smooth
            low_thresh = np.median(d) + threshold_factor * 0.6 * best_ns
            low_pks, _ = sig.find_peaks(d, height=low_thresh, distance=min_dist // 2,
                                         prominence=best_ns * 0.5)
            tolerance = median_bp * 0.25
            # Only search in gaps that are 1.5–3× the median period
            # (one or two missed beats). Larger gaps indicate cessation
            # or other discontinuities — do not fill them.
            gap_regions = []
            for gi in range(len(best_peaks) - 1):
                gap = best_peaks[gi + 1] - best_peaks[gi]
                if 1.5 * median_bp < gap < 3.5 * median_bp:
                    gap_regions.append((best_peaks[gi], best_peaks[gi + 1]))

            pass2_added = []
            existing = set(best_peaks.tolist())
            for lp in low_pks:
                if lp in existing:
                    continue
                # Only consider peaks inside a gap region
                in_gap = any(gs < lp < ge for gs, ge in gap_regions)
                if not in_gap:
                    continue
                # Too close to an existing peak?
                if any(abs(lp - ep) < min_dist for ep in best_peaks):
                    continue
                if any(abs(lp - ep) < min_dist for ep in pass2_added):
                    continue
                # Is this peak at a rhythmically expected position?
                dists = np.abs(best_peaks.astype(float) - float(lp))
                min_dist_to_existing = np.min(dists)
                n_periods = min_dist_to_existing / median_bp
                residual = abs(n_periods - round(n_periods)) * median_bp
                if residual < tolerance and round(n_periods) >= 1:
                    pass2_added.append(lp)

            if pass2_added:
                best_peaks = np.sort(np.concatenate([
                    best_peaks, np.array(pass2_added, dtype=int)
                ]))
                logger.info("Derivative pass 2: +%d beats from low-threshold gaps",
                            len(pass2_added))

    # ── Pass 3: Interleaving-based opposite-polarity merge ──
    # For mixed-polarity (biphasic) signals where depolarization spikes
    # alternate between positive and negative, single-polarity detection
    # only catches every other beat.  The other polarity's beats sit
    # exactly in between (interleaved).
    #
    # Strategy:
    #   1. Check if the other polarity's beats cleanly interleave with ours
    #      (each sits roughly midway between two primary beats).
    #   2. If a high fraction interleave, merge them — this is an
    #      alternating-polarity signal.
    #   3. Also fall back to the old gap-based approach for signals that
    #      are mostly single-polarity but have occasional inverted beats.
    if len(best_peaks) >= 3 and best_pol in ('positive', 'negative'):
        bp_samples = np.diff(best_peaks)
        median_bp = np.median(bp_samples)
        other_pol = 'negative' if best_pol == 'positive' else 'positive'
        other_pks, other_ns = polarity_results.get(other_pol, (np.array([], dtype=int), 1.0))

        if len(other_pks) > 0 and median_bp > 0:
            # Reference sharpness from the primary detection
            dvdt_abs = np.abs(deriv_smooth)
            w_dvdt = max(3, int(5 / 1000.0 * fs))
            ref_dvdt_vals = [
                np.max(dvdt_abs[max(0, p - w_dvdt):min(len(dvdt_abs), p + w_dvdt + 1)])
                for p in best_peaks
            ]
            ref_dvdt_median = np.median(ref_dvdt_vals) if ref_dvdt_vals else 1.0

            # ── Interleaving check ──
            # For each other-polarity peak, check if it falls between two
            # primary peaks at roughly the midpoint (within 35% of half-period).
            interleaved = []
            non_interleaved = []
            existing = set(best_peaks.tolist())

            for op in other_pks:
                if op in existing:
                    continue
                # Too close to an existing beat?
                if any(abs(op - ep) < min_dist for ep in best_peaks):
                    continue
                # Sharpness check: must have at least 20% of primary sharpness
                local_dvdt = np.max(
                    dvdt_abs[max(0, op - w_dvdt):min(len(dvdt_abs), op + w_dvdt + 1)])
                if local_dvdt < 0.20 * ref_dvdt_median:
                    continue
                # Find surrounding primary beats
                left_beats = best_peaks[best_peaks < op]
                right_beats = best_peaks[best_peaks > op]
                if len(left_beats) > 0 and len(right_beats) > 0:
                    gap = right_beats[0] - left_beats[-1]
                    midpoint = (left_beats[-1] + right_beats[0]) / 2.0
                    deviation_from_mid = abs(op - midpoint) / (gap / 2.0) if gap > 0 else 999
                    # Accept if within 35% of midpoint AND gap is at least
                    # 1.3× median (room for a beat) — but for truly alternating
                    # signals the median period IS the doubled period, so
                    # every gap qualifies.
                    if deviation_from_mid < 0.35 and gap > 0.8 * median_bp:
                        interleaved.append(op)
                    else:
                        non_interleaved.append(op)
                elif len(left_beats) > 0:
                    # At the right edge: check distance to last primary beat
                    dist = op - left_beats[-1]
                    if 0.35 * median_bp < dist < 0.65 * median_bp:
                        interleaved.append(op)
                elif len(right_beats) > 0:
                    # At the left edge: check distance to first primary beat
                    dist = right_beats[0] - op
                    if 0.35 * median_bp < dist < 0.65 * median_bp:
                        interleaved.append(op)

            # Decide: is this an interleaving signal?
            # If ≥40% of other-polarity peaks interleave AND the count is
            # meaningful (at least 2), accept the merge.
            n_valid_other = len(interleaved) + len(non_interleaved)
            interleave_ratio = len(interleaved) / n_valid_other if n_valid_other > 0 else 0

            if len(interleaved) >= 2 and interleave_ratio >= 0.40:
                # Check that merged result has good regularity
                merged_candidate = np.sort(np.concatenate([
                    best_peaks, np.array(interleaved, dtype=int)
                ]))
                # Remove duplicates closer than min_dist
                keep = [merged_candidate[0]]
                for p in merged_candidate[1:]:
                    if p - keep[-1] >= min_dist:
                        keep.append(p)
                merged_candidate = np.array(keep, dtype=int)

                if len(merged_candidate) >= 4:
                    merged_bp = np.diff(merged_candidate) / fs
                    merged_cv = np.std(merged_bp) / np.mean(merged_bp) if np.mean(merged_bp) > 0 else 999
                    orig_bp = np.diff(best_peaks) / fs
                    orig_cv = np.std(orig_bp) / np.mean(orig_bp) if np.mean(orig_bp) > 0 else 999

                    # Accept merge if CV is reasonable (< 0.5) OR better than original
                    if merged_cv < 0.50 or merged_cv < orig_cv * 1.1:
                        best_peaks = merged_candidate
                        best_pol = 'mixed'
                        logger.info(
                            "Derivative pass 3 (interleave merge): +%d beats "
                            "(ratio=%.2f, merged_cv=%.3f, orig_cv=%.3f)",
                            len(interleaved), interleave_ratio, merged_cv, orig_cv)
                    else:
                        logger.debug(
                            "Derivative pass 3: interleave merge rejected "
                            "(merged_cv=%.3f > orig_cv=%.3f × 1.1)",
                            merged_cv, orig_cv)
                else:
                    logger.debug("Derivative pass 3: merged candidate too short (%d)",
                                 len(merged_candidate))
            else:
                # ── Fallback: gap-based opposite-polarity search ──
                # For signals that are mostly single-polarity but have
                # occasional inverted beats in large gaps.
                pass3_added = []
                for op in other_pks:
                    if op in existing:
                        continue
                    if any(abs(op - ep) < min_dist for ep in best_peaks):
                        continue
                    if any(abs(op - ep) < min_dist for ep in pass3_added):
                        continue
                    left_beats = best_peaks[best_peaks < op]
                    right_beats = best_peaks[best_peaks > op]
                    if len(left_beats) == 0 or len(right_beats) == 0:
                        continue
                    gap = right_beats[0] - left_beats[-1]
                    if gap < 1.6 * median_bp:
                        continue
                    local_dvdt = np.max(
                        dvdt_abs[max(0, op - w_dvdt):min(len(dvdt_abs), op + w_dvdt + 1)])
                    if local_dvdt < 0.30 * ref_dvdt_median:
                        continue
                    pass3_added.append(op)

                if pass3_added:
                    best_peaks = np.sort(np.concatenate([
                        best_peaks, np.array(pass3_added, dtype=int)
                    ]))
                    best_pol = 'mixed'
                    logger.info("Derivative pass 3 (gap-fill): +%d opposite-polarity "
                                "beats in gaps", len(pass3_added))

    # ── Pass 4: Post-merge gap fill (rhythm-guided, both polarities) ──
    # After the interleaving merge, the median period reflects the TRUE
    # beat period.  For each gap ≥ 1.4× median, compute where beats SHOULD
    # be based on the rhythm, then find the strongest derivative peak (of
    # either polarity) within a tolerance window of each expected position.
    # This is much more sensitive than re-running find_peaks because it
    # targets specific positions rather than requiring global threshold.
    if len(best_peaks) >= 4:
        bp_samples = np.diff(best_peaks)
        merged_median_bp = np.median(bp_samples)
        if merged_median_bp > 0:
            tol_samples = int(merged_median_bp * 0.30)  # ±30% tolerance
            pass4_added = []
            existing = set(best_peaks.tolist())
            dvdt_abs = np.abs(deriv_smooth)

            # Reference: median absolute derivative at existing beats
            w_dvdt = max(3, int(5 / 1000.0 * fs))
            ref_dvdt_vals = [
                np.max(dvdt_abs[max(0, p - w_dvdt):min(len(dvdt_abs), p + w_dvdt + 1)])
                for p in best_peaks
            ]
            ref_dvdt_median = np.median(ref_dvdt_vals) if ref_dvdt_vals else 1.0

            for gi in range(len(best_peaks) - 1):
                gap = best_peaks[gi + 1] - best_peaks[gi]
                if gap < 1.4 * merged_median_bp:
                    continue
                # How many beats are missing in this gap?
                n_missing = round(gap / merged_median_bp) - 1
                if n_missing < 1 or n_missing > 5:
                    continue
                # Expected positions of missing beats
                step = gap / (n_missing + 1)
                for k in range(1, n_missing + 1):
                    expected_pos = int(best_peaks[gi] + k * step)
                    # Search window around expected position
                    ws = max(0, expected_pos - tol_samples)
                    we = min(len(dvdt_abs), expected_pos + tol_samples + 1)
                    if we - ws < 5:
                        continue
                    # Too close to an existing or already-added beat?
                    all_existing = list(existing) + pass4_added
                    if any(abs(expected_pos - ep) < min_dist * 0.7 for ep in all_existing):
                        continue
                    # Find strongest derivative peak in window (either polarity)
                    seg_dvdt = dvdt_abs[ws:we]
                    local_max_dvdt = np.max(seg_dvdt)
                    # Must have at least 15% of reference sharpness
                    if local_max_dvdt < 0.15 * ref_dvdt_median:
                        continue
                    # Find the exact position: look at both positive and
                    # negative derivative, pick the one with the sharper peak
                    best_candidate = None
                    best_candidate_dvdt = 0
                    for sign in [1, -1]:
                        d = sign * deriv_smooth
                        seg_d = d[ws:we]
                        local_pks, local_props = sig.find_peaks(
                            seg_d, prominence=0, distance=max(1, min_dist // 4))
                        if len(local_pks) == 0:
                            continue
                        # Pick peak with highest prominence
                        best_local_idx = np.argmax(local_props['prominences'])
                        pk_pos = ws + local_pks[best_local_idx]
                        pk_dvdt = np.max(dvdt_abs[max(0, pk_pos - w_dvdt):
                                                    min(len(dvdt_abs), pk_pos + w_dvdt + 1)])
                        if pk_dvdt > best_candidate_dvdt:
                            best_candidate = pk_pos
                            best_candidate_dvdt = pk_dvdt

                    if best_candidate is not None and best_candidate_dvdt >= 0.15 * ref_dvdt_median:
                        if not any(abs(best_candidate - ep) < min_dist * 0.7 for ep in all_existing):
                            pass4_added.append(best_candidate)

            if pass4_added:
                best_peaks = np.sort(np.concatenate([
                    best_peaks, np.array(pass4_added, dtype=int)
                ]))
                logger.info("Derivative pass 4 (rhythm-guided gap-fill): +%d beats",
                            len(pass4_added))

    # Refine to signal peaks.
    # For mixed-polarity signals, decide per-beat whether the spike is
    # positive or negative and refine to the correct extreme.
    w = int(c.peak_refine_window_ms / 1000.0 * fs)
    refined = []
    beat_polarities = []  # track per-beat polarity for downstream use
    for p in best_peaks:
        s, e = max(0, p-w), min(len(data), p+w+1)
        seg = data[s:e]
        seg_min, seg_max = np.min(seg), np.max(seg)
        median_val = np.median(data)
        # Per-beat polarity: pick the extreme that is farthest from median
        if abs(seg_min - median_val) > abs(seg_max - median_val):
            idx = s + np.argmin(seg)
            beat_polarities.append('negative')
        else:
            idx = s + np.argmax(seg)
            beat_polarities.append('positive')
        refined.append(idx)
    refined = np.array(sorted(set(refined)), dtype=int)
    if len(refined) > 1:
        keep = [refined[0]]
        for p in refined[1:]:
            if p - keep[-1] >= min_dist: keep.append(p)
        refined = np.array(keep, dtype=int)

    return refined, refined / fs, {'method': 'derivative', 'polarity': best_pol, 'n_beats': len(refined)}


def _detect_peak(data, fs, min_dist, threshold_factor):
    """Simple amplitude-based peak detection as fallback."""
    p5, p95 = np.percentile(data, [5, 95])
    mask = (data >= p5) & (data <= p95)
    ns = np.std(data[mask]) if np.sum(mask) > 100 else np.std(data)
    thresh = threshold_factor * ns
    peaks_pos, _ = sig.find_peaks(data, height=np.median(data)+thresh, distance=min_dist, prominence=ns)
    peaks_neg, _ = sig.find_peaks(-data, height=-np.median(data)+thresh, distance=min_dist, prominence=ns)
    if len(peaks_pos) >= len(peaks_neg):
        bi, pol = peaks_pos, 'positive'
    else:
        bi, pol = peaks_neg, 'negative'
    return bi, bi / fs, {'method': 'peak', 'polarity': pol, 'n_beats': len(bi)}


def _estimate_jitter_from_template(template, fs, fraction):
    """Estimate adaptive jitter window from template spike width.

    Computes the half-width at half-maximum (HWHM) of the dominant
    deflection in the template, then returns ``fraction × HWHM``
    in samples.  Returns 0 if estimation fails.
    """
    try:
        # Find the dominant peak (largest absolute deflection from mean)
        baseline = np.median(template)
        deviation = np.abs(template - baseline)
        peak_idx = np.argmax(deviation)
        peak_val = deviation[peak_idx]
        if peak_val <= 0:
            return 0
        half_max = peak_val * 0.5

        # Walk left from peak to find half-max crossing
        left = peak_idx
        while left > 0 and deviation[left] > half_max:
            left -= 1
        # Walk right from peak
        right = peak_idx
        while right < len(deviation) - 1 and deviation[right] > half_max:
            right += 1

        fwhm_samples = right - left  # full width at half max
        hwhm_samples = fwhm_samples / 2.0
        jitter = max(1, int(fraction * hwhm_samples))

        hwhm_ms = hwhm_samples / fs * 1000
        logger.debug("Adaptive jitter: FWHM=%.1f ms, HWHM=%.1f ms, "
                      "jitter=±%d samples (±%.1f ms)",
                      fwhm_samples / fs * 1000, hwhm_ms,
                      jitter, jitter / fs * 1000)
        return jitter
    except (ValueError, IndexError):
        return 0


def validate_beats_morphology(data, fs, beat_indices, cfg=None, detected_polarity=None):
    """
    Post-detection morphological validation (CardioMDA-inspired).

    Builds a template from the strongest beat candidates (by amplitude),
    then rejects beats whose correlation with the template is below threshold
    AND beats whose amplitude is far below the robust reference.

    For mixed-polarity signals (detected_polarity='mixed'), morphological
    validation is skipped entirely — these signals have beats of fundamentally
    different shapes that cannot be validated against a single template.
    Only a lenient amplitude gate is applied.

    Parameters
    ----------
    data : array — filtered FP signal
    fs : float — sampling rate
    beat_indices : array of int — detected beat locations
    cfg : BeatDetectionConfig or None
    detected_polarity : str or None — detection polarity ('positive', 'negative', 'mixed')

    Returns
    -------
    validated_indices : array of int — indices of accepted beats
    validation_info : dict — diagnostics
    """
    c = _get_bd_cfg(cfg)

    if not c.enable_morphology_validation or len(beat_indices) < 3:
        return beat_indices, {'validation': 'skipped', 'n_input': len(beat_indices)}

    # ── Mixed-polarity bypass ──
    # For mixed-polarity signals, beats have fundamentally different shapes
    # (large/small, positive/negative).  Template-based morphological
    # validation cannot work — it will always reject ~half the beats.
    # Use only a lenient amplitude gate instead.
    if detected_polarity == 'mixed':
        window = int(20 * fs / 1000)
        amplitudes = np.zeros(len(beat_indices))
        for i, bi in enumerate(beat_indices):
            s = max(0, bi - window)
            e = min(len(data), bi + window)
            seg = data[s:e]
            amplitudes[i] = np.max(seg) - np.min(seg)
        ref_amplitude = np.median(amplitudes) if len(amplitudes) > 0 else 1.0
        if ref_amplitude > 0:
            amp_ratios = amplitudes / ref_amplitude
        else:
            amp_ratios = np.ones(len(amplitudes))
        # Very lenient amplitude gate: 10% of median amplitude
        amp_ok = amp_ratios >= 0.10
        validated_indices = beat_indices[amp_ok]
        n_rej = int(np.sum(~amp_ok))
        logger.info("Mixed-polarity validation: skip morphology, amplitude-only "
                     "%d → %d (-%d)", len(beat_indices), len(validated_indices), n_rej)
        return validated_indices, {
            'validation': 'mixed_polarity_amplitude_only',
            'n_input': len(beat_indices),
            'n_accepted': len(validated_indices),
            'n_rejected_amplitude': n_rej,
            'n_rejected_morphology': 0,
            'ref_amplitude': float(ref_amplitude),
        }

    # ── Step 1: Compute amplitude for each beat ──
    window = int(20 * fs / 1000)  # ±20 ms around spike
    amplitudes = np.zeros(len(beat_indices))
    for i, bi in enumerate(beat_indices):
        s = max(0, bi - window)
        e = min(len(data), bi + window)
        seg = data[s:e]
        amplitudes[i] = np.max(seg) - np.min(seg)

    # ── Pre-step: Classify each beat by polarity ──
    # Must happen before amplitude gate so we can adjust thresholds
    # for mixed-polarity signals.
    median_val = np.median(data)
    beat_is_positive = np.zeros(len(beat_indices), dtype=bool)
    for i, bi in enumerate(beat_indices):
        s = max(0, bi - window)
        e = min(len(data), bi + window)
        seg = data[s:e]
        beat_is_positive[i] = abs(np.max(seg) - median_val) >= abs(np.min(seg) - median_val)

    n_pos = np.sum(beat_is_positive)
    n_neg = len(beat_indices) - n_pos
    n_total = len(beat_indices)
    min_group = max(3, int(0.25 * n_total))
    is_mixed = n_pos >= min_group and n_neg >= min_group

    # Robust reference: median of upper 50% of amplitudes
    sorted_amps = np.sort(amplitudes)
    upper_half = sorted_amps[len(sorted_amps) // 2:]
    ref_amplitude = np.median(upper_half) if len(upper_half) > 0 else np.median(amplitudes)

    # ── Step 2: Amplitude gate ──
    if ref_amplitude > 0:
        amp_ratios = amplitudes / ref_amplitude
    else:
        amp_ratios = np.ones(len(amplitudes))

    # For mixed-polarity signals, use a much lower amplitude threshold.
    # Large spikes inflate the reference, making small but real beats fail.
    effective_min_amp = c.min_amplitude_ratio * 0.4 if is_mixed else c.min_amplitude_ratio
    amp_ok = amp_ratios >= effective_min_amp

    # ── Step 3: Morphological validation (template correlation) ──
    morph_ok = np.ones(len(beat_indices), dtype=bool)
    beat_corrs = np.full(len(beat_indices), -1.0)
    beat_corrs_min = np.full(len(beat_indices), 0.0)

    if len(beat_indices) >= c.morphology_min_beats:
        pre = int(20 * fs / 1000)   # 20 ms before spike
        post = int(80 * fs / 1000)  # 80 ms after spike (depolarization region)

        # ── Build template(s) ──
        # For mixed-polarity signals: build separate templates for positive
        # and negative beats.  For single-polarity: build one template.
        templates = {}  # key: 'positive' or 'negative', value: (template, t_c, t_c_ss, min_len, jitter_max)

        polarity_groups = [('positive', beat_is_positive), ('negative', ~beat_is_positive)] if is_mixed else [('all', np.ones(len(beat_indices), dtype=bool))]

        for group_name, group_mask in polarity_groups:
            group_indices = np.where(group_mask)[0]
            if len(group_indices) < 3:
                continue
            group_amps = amplitudes[group_indices]
            # Select top 50% by amplitude within this group
            n_for_template = max(3, len(group_indices) // 2)
            top_in_group = np.argsort(group_amps)[::-1][:n_for_template]
            template_beats = []
            for tidx in top_in_group:
                bi = beat_indices[group_indices[tidx]]
                s, e = bi - pre, bi + post
                if s >= 0 and e < len(data):
                    template_beats.append(data[s:e])
            if len(template_beats) < 3:
                continue
            tpl_min_len = min(len(b) for b in template_beats)
            mat = np.array([b[:tpl_min_len] for b in template_beats])
            tpl = np.median(mat, axis=0)

            # Jitter correction
            jmax = 0
            if c.enable_jitter_correction:
                if c.jitter_adaptive:
                    jmax = _estimate_jitter_from_template(tpl, fs, c.jitter_adaptive_fraction)
                if jmax < 1:
                    jmax = max(1, int(c.jitter_max_shift_ms / 1000.0 * fs))

            tc = tpl - np.mean(tpl)
            tc_ss = np.sum(tc**2)
            templates[group_name] = (tpl, tc, tc_ss, tpl_min_len, jmax)

        if not templates:
            # Fallback: build single template from top amplitudes (original logic)
            n_for_template = max(5, len(beat_indices) // 2)
            top_indices = np.argsort(amplitudes)[::-1][:n_for_template]
            template_beats = []
            for idx in top_indices:
                bi = beat_indices[idx]
                s, e = bi - pre, bi + post
                if s >= 0 and e < len(data):
                    template_beats.append(data[s:e])
            if len(template_beats) >= 3:
                tpl_min_len = min(len(b) for b in template_beats)
                mat = np.array([b[:tpl_min_len] for b in template_beats])
                tpl = np.median(mat, axis=0)
                jmax = 0
                if c.enable_jitter_correction:
                    if c.jitter_adaptive:
                        jmax = _estimate_jitter_from_template(tpl, fs, c.jitter_adaptive_fraction)
                    if jmax < 1:
                        jmax = max(1, int(c.jitter_max_shift_ms / 1000.0 * fs))
                tc = tpl - np.mean(tpl)
                tc_ss = np.sum(tc**2)
                templates['all'] = (tpl, tc, tc_ss, tpl_min_len, jmax)

        # ── Correlate each beat with its matching template ──
        if templates:
            for i, bi in enumerate(beat_indices):
                # Select the right template for this beat
                if is_mixed:
                    group_key = 'positive' if beat_is_positive[i] else 'negative'
                    if group_key not in templates:
                        # No template for this group — try the other
                        group_key = 'negative' if beat_is_positive[i] else 'positive'
                    if group_key not in templates:
                        continue  # no template available
                else:
                    group_key = 'all'
                    if group_key not in templates:
                        continue

                tpl, t_c, t_c_ss, min_len, jitter_max = templates[group_key]

                s, e = bi - pre, bi + post
                s_ext = s - jitter_max
                e_ext = e + jitter_max
                if s_ext < 0 or e_ext >= len(data):
                    continue
                beat_wide = data[s_ext:e_ext]

                best_corr = -1.0
                worst_corr = 1.0
                for lag in range(-jitter_max, jitter_max + 1):
                    offset = jitter_max + lag
                    beat_seg = beat_wide[offset:offset + min_len]
                    if len(beat_seg) < min_len:
                        continue
                    b_c = beat_seg - np.mean(beat_seg)
                    denom = np.sqrt(np.sum(b_c**2) * t_c_ss)
                    if denom > 0:
                        corr = np.sum(b_c * t_c) / denom
                    else:
                        corr = 0.0
                    if corr > best_corr:
                        best_corr = corr
                    if corr < worst_corr:
                        worst_corr = corr
                beat_corrs[i] = best_corr
                beat_corrs_min[i] = worst_corr

                # For mixed signals: use a much lower correlation threshold.
                # Mixed-polarity signals have beats of widely varying
                # morphology (large/small, positive/negative) — strict
                # template correlation would reject genuine beats.
                # Rhythm + amplitude already provide strong validation.
                # For single-template mode: keep the inverted-beat logic.
                if is_mixed:
                    mixed_min_corr = c.morphology_min_corr * 0.35
                    morph_ok[i] = best_corr >= mixed_min_corr
                else:
                    if (c.morphology_accept_inverted
                            and worst_corr < -c.morphology_min_corr
                            and amp_ratios[i] >= c.min_amplitude_ratio * 0.6):
                        effective_corr = abs(worst_corr)
                    else:
                        effective_corr = best_corr
                    morph_ok[i] = effective_corr >= c.morphology_min_corr

        if is_mixed:
            logger.info("Mixed-polarity validation: %d positive + %d negative beats, "
                        "%d templates built", n_pos, n_neg, len(templates))

    # ── Step 4: Combine gates ──
    accepted = amp_ok & morph_ok
    n_rej_amp = int(np.sum(~amp_ok))
    n_rej_morph = int(np.sum(amp_ok & ~morph_ok))  # morphology-only rejections

    validated_indices = beat_indices[accepted]

    info = {
        'validation': 'applied',
        'n_input': len(beat_indices),
        'n_accepted': len(validated_indices),
        'n_rejected_amplitude': n_rej_amp,
        'n_rejected_morphology': n_rej_morph,
        'ref_amplitude': float(ref_amplitude),
    }

    if len(validated_indices) < 3 and len(beat_indices) >= 5:
        # Validation too aggressive — fall back to amplitude-only
        validated_indices = beat_indices[amp_ok]
        info['validation'] = 'fallback_amplitude_only'
        info['n_accepted'] = len(validated_indices)
        info['note'] = 'Morphology validation rejected too many beats; using amplitude-only'

    # ── Step 5: Period-aware re-admission ──
    # After validation, use the CLEAN rhythm (from validated beats only)
    # to re-admit rejected beats that sit at expected intervals.
    # This is safe because the rhythm estimate comes from validated beats,
    # not the contaminated initial set.
    # Re-admit beats rejected by morphology (but NOT amplitude) if they sit
    # at rhythmically expected positions.
    rejected_mask = ~accepted & amp_ok  # rejected by morphology, not by amplitude
    n_readmitted = 0
    if np.sum(rejected_mask) > 0 and len(validated_indices) >= 4:
        clean_bp = np.diff(validated_indices)
        clean_median_bp = np.median(clean_bp)
        if clean_median_bp > 0:
            tol = clean_median_bp * 0.3
            relaxed_corr = c.morphology_min_corr * 0.5
            rejected_indices = beat_indices[rejected_mask]
            rejected_corrs = beat_corrs[rejected_mask]
            rejected_corrs_min = beat_corrs_min[rejected_mask]
            readmit = []
            # Get amplitude ratios for rejected beats
            rejected_amp_ratios = amp_ratios[rejected_mask]
            for ri, rc_val, rc_min, ar in zip(rejected_indices, rejected_corrs,
                                               rejected_corrs_min, rejected_amp_ratios):
                # For inverted beats: use most-negative correlation (rc_min)
                if (c.morphology_accept_inverted
                        and rc_min < -relaxed_corr
                        and ar >= c.min_amplitude_ratio * 0.5):
                    effective_rc = abs(rc_min)
                else:
                    effective_rc = rc_val
                if effective_rc < relaxed_corr:
                    continue
                # Check if this beat is at a rhythmically expected position
                # relative to validated beats
                dists = np.abs(validated_indices.astype(float) - float(ri))
                min_dist = np.min(dists)
                if min_dist < clean_median_bp * 0.3:
                    # Too close to an already-accepted beat — skip
                    continue
                # Check if min_dist is a near-integer multiple of period
                n_periods = min_dist / clean_median_bp
                residual = abs(n_periods - round(n_periods)) * clean_median_bp
                if residual < tol:
                    readmit.append(ri)
            if readmit:
                validated_indices = np.sort(
                    np.concatenate([validated_indices, np.array(readmit, dtype=int)])
                )
                n_readmitted = len(readmit)
                info['n_readmitted'] = n_readmitted
                info['n_accepted'] = len(validated_indices)

    logger.info("Beat validation: %d → %d (-%d amp, -%d morph, +%d readmitted)",
                len(beat_indices), len(validated_indices),
                n_rej_amp, n_rej_morph, n_readmitted)

    return validated_indices, info


def segment_beats(data, time, beat_indices, fs, pre_ms=50, post_ms=600, cfg=None):
    """Segment signal into individual beats aligned to depolarization spike.

    Each beat is extracted as ``data[idx - pre : idx + post]`` where *idx*
    is the depolarisation peak index.  Beats whose window falls outside
    the signal boundaries are silently excluded; a warning is logged when
    more than 10 % of beats are lost this way.

    Parameters
    ----------
    data : array — filtered FP signal
    time : array — time axis (same length as data)
    beat_indices : array of int — sample indices of detected beats
    fs : float — sampling rate (Hz)
    pre_ms, post_ms : float — window before/after spike (ms)
    cfg : RepolarizationConfig or None — overrides pre_ms/post_ms

    Returns
    -------
    beats_data : list of 1-D arrays
    beats_time : list of 1-D arrays (relative time, spike at t=0)
    valid : list of int — indices into *beat_indices* that were kept
    """
    if cfg is not None:
        pre_ms = cfg.segment_pre_ms
        # post_ms should cover the full repolarization search + some margin
        post_ms = max(post_ms, cfg.search_end_ms + 50)
    pre, post = int(pre_ms * fs / 1000), int(post_ms * fs / 1000)
    beats_data, beats_time, valid = [], [], []
    for i, idx in enumerate(beat_indices):
        s, e = idx - pre, idx + post
        if s < 0 or e >= len(data):
            continue
        beats_data.append(data[s:e].copy())
        beats_time.append(np.arange(-pre, post) / fs)
        valid.append(i)

    n_total = len(beat_indices)
    n_excluded = n_total - len(valid)
    if n_total > 0 and n_excluded / n_total > 0.10:
        logger.warning(
            "segment_beats: %d/%d beats (%.0f%%) excluded because they are "
            "too close to signal edges (pre=%.0fms, post=%.0fms)",
            n_excluded, n_total, n_excluded / n_total * 100, pre_ms, post_ms,
        )

    return beats_data, beats_time, valid


def compute_beat_periods(beat_indices, fs):
    """Compute inter-beat intervals in seconds."""
    if len(beat_indices) < 2: return np.array([])
    return np.diff(beat_indices) / fs
