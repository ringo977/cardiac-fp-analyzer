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
                if cv_bp < c.cv_good: score += c.score_cv_good
                elif cv_bp < c.cv_fair: score += c.score_cv_fair
                elif cv_bp < c.cv_marginal: score += c.score_cv_marginal
                duration_s = len(data) / fs
                if duration_s/3 <= len(bi) <= duration_s/0.3:
                    score += c.score_rate_ok
                elif len(bi) > 3:
                    score += c.score_rate_low
                if len(bi) > duration_s/0.3:
                    score += c.score_rate_excess
            else:
                score = c.score_too_few
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
    bi, val_info = validate_beats_morphology(data, fs, bi, cfg=cfg)
    bt = bi / fs
    info['n_beats'] = len(bi)
    info['beat_validation'] = val_info

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
    """Derivative-based: detects steepest slope at depolarization."""
    c = _get_bd_cfg(cfg)
    dt = 1.0 / fs
    deriv = np.gradient(data, dt)
    win = max(3, int(c.deriv_smooth_ms / 1000.0 * fs))
    if win % 2 == 0: win += 1
    deriv_smooth = sig.savgol_filter(deriv, win, 2) if fs >= 1000 else deriv

    best_peaks, best_pol = np.array([], dtype=int), 'positive'
    best_score = -999

    for sign, pol in [(1, 'positive'), (-1, 'negative')]:
        d = sign * deriv_smooth
        p5, p95 = np.percentile(d, [5, 95])
        mask = (d >= p5) & (d <= p95)
        ns = np.std(d[mask]) if np.sum(mask) > 100 else np.std(d)
        thresh = np.median(d) + threshold_factor * ns
        pks, _ = sig.find_peaks(d, height=thresh, distance=min_dist, prominence=ns*2)
        if len(pks) > 2:
            bp = np.diff(pks) / fs
            cv = np.std(bp) / np.mean(bp) if np.mean(bp) > 0 else 999
            score = -cv
            if score > best_score:
                best_score, best_peaks, best_pol = score, pks, pol

    # Refine to signal peaks
    w = int(c.peak_refine_window_ms / 1000.0 * fs)
    refined = []
    for p in best_peaks:
        s, e = max(0, p-w), min(len(data), p+w+1)
        seg = data[s:e]
        idx = s + (np.argmin(seg) if np.abs(np.min(seg)) > np.abs(np.max(seg)) else np.argmax(seg))
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


def validate_beats_morphology(data, fs, beat_indices, cfg=None):
    """
    Post-detection morphological validation (CardioMDA-inspired).

    Builds a template from the strongest beat candidates (by amplitude),
    then rejects beats whose correlation with the template is below threshold
    AND beats whose amplitude is far below the robust reference.

    This runs BEFORE segmentation/parameter extraction — unlike the QC module
    which runs after.  The goal is to remove noise spikes mistakenly identified
    as depolarization spikes, so downstream modules never see them.

    References:
      - Clements & Thomas, PLOS ONE 2013 (CardioMDA): correlation ≥ 0.95–0.98
      - Patel et al., Sci Rep 2025 (EFP Analyzer): stringent quality criteria

    Parameters
    ----------
    data : array — filtered FP signal
    fs : float — sampling rate
    beat_indices : array of int — detected beat locations
    cfg : BeatDetectionConfig or None

    Returns
    -------
    validated_indices : array of int — indices of accepted beats
    validation_info : dict — diagnostics (n_rejected_amplitude, n_rejected_morphology, etc.)
    """
    c = _get_bd_cfg(cfg)

    if not c.enable_morphology_validation or len(beat_indices) < 3:
        return beat_indices, {'validation': 'skipped', 'n_input': len(beat_indices)}

    # ── Step 1: Compute amplitude for each beat ──
    window = int(20 * fs / 1000)  # ±20 ms around spike
    amplitudes = np.zeros(len(beat_indices))
    for i, bi in enumerate(beat_indices):
        s = max(0, bi - window)
        e = min(len(data), bi + window)
        seg = data[s:e]
        amplitudes[i] = np.max(seg) - np.min(seg)

    # Robust reference: median of upper 50% of amplitudes
    sorted_amps = np.sort(amplitudes)
    upper_half = sorted_amps[len(sorted_amps) // 2:]
    ref_amplitude = np.median(upper_half) if len(upper_half) > 0 else np.median(amplitudes)

    # ── Step 2: Amplitude gate ──
    if ref_amplitude > 0:
        amp_ratios = amplitudes / ref_amplitude
    else:
        amp_ratios = np.ones(len(amplitudes))

    amp_ok = amp_ratios >= c.min_amplitude_ratio

    # ── Step 3: Morphological validation (template correlation) ──
    morph_ok = np.ones(len(beat_indices), dtype=bool)

    if len(beat_indices) >= c.morphology_min_beats:
        # Build template from highest-amplitude beats
        pre = int(20 * fs / 1000)   # 20 ms before spike
        post = int(80 * fs / 1000)  # 80 ms after spike (depolarization region)

        # Select top 50% by amplitude for template
        n_for_template = max(5, len(beat_indices) // 2)
        top_indices = np.argsort(amplitudes)[::-1][:n_for_template]
        template_beats = []
        for idx in top_indices:
            bi = beat_indices[idx]
            s, e = bi - pre, bi + post
            if s >= 0 and e < len(data):
                template_beats.append(data[s:e])

        if len(template_beats) >= 3:
            # Median template (robust to outliers)
            min_len = min(len(b) for b in template_beats)
            mat = np.array([b[:min_len] for b in template_beats])
            template = np.median(mat, axis=0)

            # Correlate each beat with template
            for i, bi in enumerate(beat_indices):
                s, e = bi - pre, bi + post
                if s < 0 or e >= len(data):
                    # Edge beat — skip morphology check, keep amplitude gate
                    continue
                beat_seg = data[s:e][:min_len]
                # Pearson correlation
                b_c = beat_seg - np.mean(beat_seg)
                t_c = template - np.mean(template)
                denom = np.sqrt(np.sum(b_c**2) * np.sum(t_c**2))
                if denom > 0:
                    corr = np.sum(b_c * t_c) / denom
                else:
                    corr = 0.0
                morph_ok[i] = corr >= c.morphology_min_corr

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

    logger.info("Beat validation: %d → %d (-%d amp, -%d morph)",
                len(beat_indices), len(validated_indices), n_rej_amp, n_rej_morph)

    return validated_indices, info


def segment_beats(data, time, beat_indices, fs, pre_ms=50, post_ms=600, cfg=None):
    """Segment signal into individual beats aligned to depolarization spike.

    If cfg (RepolarizationConfig) is provided, pre_ms is taken from
    cfg.segment_pre_ms and post_ms from cfg.search_end_ms + margin.
    Explicit pre_ms/post_ms still override when cfg is None.
    """
    if cfg is not None:
        pre_ms = cfg.segment_pre_ms
        # post_ms should cover the full repolarization search + some margin
        post_ms = max(post_ms, cfg.search_end_ms + 50)
    pre, post = int(pre_ms * fs / 1000), int(post_ms * fs / 1000)
    beats_data, beats_time, valid = [], [], []
    for i, idx in enumerate(beat_indices):
        s, e = idx - pre, idx + post
        if s < 0 or e >= len(data): continue
        beats_data.append(data[s:e].copy())
        beats_time.append(np.arange(-pre, post) / fs)
        valid.append(i)
    return beats_data, beats_time, valid


def compute_beat_periods(beat_indices, fs):
    """Compute inter-beat intervals in seconds."""
    if len(beat_indices) < 2: return np.array([])
    return np.diff(beat_indices) / fs
