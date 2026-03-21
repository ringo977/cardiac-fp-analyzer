"""
beat_detection.py — Robust beat detection for hiPSC-CM field potentials.

Multi-strategy approach:
  1. Prominence-based peak detection on the signal envelope
  2. Derivative confirmation (fast dV/dt at depolarization)
  3. Adaptive thresholding with local noise estimation
  4. Auto-selection of best method
"""

import numpy as np
from scipy import signal as sig


def detect_beats(data, fs, method='auto', min_distance_ms=400,
                 threshold_factor=4.0, channel_data=None):
    """
    Detect beat (depolarization spike) locations in FP signal.

    Parameters
    ----------
    data : array — filtered FP signal
    fs : float — sampling rate
    method : str — 'auto' (recommended), 'prominence', 'derivative', or 'peak'
    min_distance_ms : float — minimum inter-beat interval in ms
    threshold_factor : float — multiplier for adaptive threshold

    Returns
    -------
    beat_indices : array of int
    beat_times : array of float
    info : dict — detection diagnostics
    """
    data = np.array(data, dtype=np.float64)
    min_dist = int(min_distance_ms * fs / 1000)

    if method == 'auto':
        return _detect_auto(data, fs, min_dist, threshold_factor)
    elif method == 'prominence':
        return _detect_prominence(data, fs, min_dist, threshold_factor)
    elif method == 'derivative':
        return _detect_derivative(data, fs, min_dist, threshold_factor)
    elif method == 'peak':
        return _detect_peak(data, fs, min_dist, threshold_factor)
    else:
        raise ValueError(f"Unknown method: {method}")


def _detect_auto(data, fs, min_dist, threshold_factor):
    """Auto-select best method based on physiological plausibility."""
    results = []
    for method_fn, name in [(_detect_prominence, 'prominence'),
                             (_detect_derivative, 'derivative'),
                             (_detect_peak, 'peak')]:
        try:
            bi, bt, info = method_fn(data, fs, min_dist, threshold_factor)
            bp = np.diff(bi) / fs if len(bi) > 1 else np.array([])
            score = 0
            if len(bp) > 2:
                mean_bp = np.mean(bp)
                cv_bp = np.std(bp) / mean_bp if mean_bp > 0 else 999
                if 0.4 <= mean_bp <= 3.0: score += 30
                elif 0.3 <= mean_bp <= 5.0: score += 15
                if cv_bp < 0.15: score += 30
                elif cv_bp < 0.30: score += 20
                elif cv_bp < 0.50: score += 10
                duration_s = len(data) / fs
                if duration_s/3 <= len(bi) <= duration_s/0.3: score += 20
                elif len(bi) > 3: score += 10
                if len(bi) > duration_s/0.3: score -= 20
            else:
                score = -10
            info['_score'] = score
            info['_method_name'] = name
            results.append((bi, bt, info))
        except: continue

    if not results:
        return np.array([], dtype=int), np.array([]), {'method': 'auto', 'n_beats': 0, 'polarity': 'unknown'}

    best = max(results, key=lambda x: x[2].get('_score', -999))
    bi, bt, info = best
    info['method'] = f"auto({info.get('_method_name', '?')})"
    for k in list(info.keys()):
        if k.startswith('_'): del info[k]
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


def _detect_derivative(data, fs, min_dist, threshold_factor):
    """Derivative-based: detects steepest slope at depolarization."""
    dt = 1.0 / fs
    deriv = np.gradient(data, dt)
    win = max(3, int(0.002 * fs))
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
    w = int(0.01 * fs)
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


def segment_beats(data, time, beat_indices, fs, pre_ms=50, post_ms=600):
    """Segment signal into individual beats aligned to depolarization spike."""
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
