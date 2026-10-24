"""Opt-in vetoes on the detections leaving ``detect_beats`` (v3.17, Oct 2026).

Two independent rules, both OFF by default (``BeatDetectionConfig``), applied
in this order by :func:`apply_post_detection_vetoes`. They only ever remove
detections, never add or move one, and abstain when fewer than
``post_veto_min_kept`` detections would remain.

1. Weak-population split (``enable_weak_population_split``)
   On noisy, slow recordings the detector also picks repolarisation waves and
   noise peaks. Within one recording these form a second, weaker population.
   Every detection gets a score ``log(local SNR) + w × correlation`` where the
   local SNR is its amplitude over the robust noise of the surrounding second
   (its own ±50 ms excluded) and the correlation is taken against the median
   shape of the strongest half of the detections. An Otsu threshold splits the
   scores in two groups; the lower group is removed only when the two groups
   are well separated (Cohen's d ≥ ``weak_split_min_separation``), the upper
   group has at least ``weak_split_min_strong`` detections and the lower
   group's median local SNR is below ``weak_split_max_weak_snr``.

2. Burst veto (``enable_burst_veto``)
   Some set-ups show noise bursts from the incubator (a 230 V load switched on
   a fixed 10 s cycle: heater, CO2 valve) with a switching transient at both
   borders. Burst windows are found from the 100–450 Hz envelope of the
   filtered trace. Detections inside a burst or within ``burst_veto_edge_s``
   of an inner border are compared, on a 30 Hz low-pass copy of the trace, with
   the detections that lie well away from every burst: a detection is removed
   when its shape correlation (polarity-normalised) is below
   ``burst_veto_corr`` or its slope or amplitude is below the given fraction of
   the references. Without at least 3 clean references the veto abstains.
   Port of the "passata sottrattiva" v0.2.1 (no fallback, min_kept 5).

Development on the manual gold standard GG (Exp5/8/10), frozen, then a single
held-out run on Exp6/7/9 together with the gap-tolerant rhythm criterion of
``QualityConfig.not_analysable_gap_tolerant``: beat period within ±10 % of the
operator 65 → 79 %, FPD within ±10 % 74 → 74 %, false detections −42 %, missed
+13 %, recordings wrongly not analysable 4 → 6, operator's not-analysable
recognised 12/13 → 13/13.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sg

__all__ = ['local_beat_features', 'weak_population_split', 'burst_windows', 'lowpass_30',
           'burst_veto', 'enforce_min_rr', 'apply_post_detection_vetoes']


# ─────────────────────────────────────────────────────────────────────────────
#   Weak-population split
# ─────────────────────────────────────────────────────────────────────────────

def local_beat_features(x, fs, bi):
    """Per-detection features on the filtered trace.

    amp     max |x| within ±10 ms of the detection (anchor = that sample)
    slope   max |dx/dt| within ±15 ms of the anchor
    snr     amp / (1.4826 × MAD) of x within ±0.5 s of the anchor, ±50 ms excluded
    w       polarity-normalised, mean-removed ±50 ms window around the anchor (None at the borders)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    h10, h15, L = int(.01 * fs), int(.015 * fs), int(.05 * fs)
    half_s = int(.5 * fs)
    dx = np.gradient(x) * fs
    out = []
    for i in np.asarray(bi, dtype=int):
        a, b = max(0, i - h10), min(n, i + h10 + 1)
        j = a + int(np.argmax(np.abs(x[a:b])))
        amp = abs(x[j])
        pol = np.sign(x[j]) or 1.0
        slope = np.max(np.abs(dx[max(0, j - h15):min(n, j + h15 + 1)]))
        lo, hi = max(0, j - half_s), min(n, j + half_s)
        seg = np.r_[x[lo:max(lo, j - L)], x[min(hi, j + L):hi]]
        noise = 1.4826 * np.median(np.abs(seg - np.median(seg))) if len(seg) > 10 else np.nan
        snr = amp / noise if noise and noise > 0 else np.nan
        w = None
        if j - L >= 0 and j + L <= n:
            w = pol * x[j - L:j + L]
            w = w - np.mean(w)
        out.append({'anchor': j, 'amp': amp, 'slope': slope, 'snr': snr, 'w': w})
    return out


def _otsu_threshold(v):
    v = np.sort(np.asarray(v, dtype=float))
    best_w, best_t = -1.0, None
    for k in range(2, len(v) - 1):
        a, b = v[:k], v[k:]
        w = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if w > best_w:
            best_w, best_t = w, (v[k - 1] + v[k]) / 2
    return best_t


def weak_population_split(x, fs, bi, cfg):
    """Remove the weaker of two well-separated detection populations (see module docstring)."""
    bi = np.asarray(bi, dtype=int)
    info = {'applied': False, 'n_input': int(len(bi)), 'n_removed': 0, 'removed': [], 'separation': None}
    if len(bi) < 6:
        info['reason'] = 'fewer than 6 detections'
        return bi, info
    feats = local_beat_features(x, fs, bi)
    amps = np.array([f['amp'] for f in feats])
    slopes = np.array([f['slope'] for f in feats])
    top = [k for k in np.argsort(-(amps * slopes))[:max(3, len(feats) // 2)] if feats[k]['w'] is not None]
    corr = np.full(len(feats), np.nan)
    if len(top) >= 3:
        tpl = np.median(np.array([feats[k]['w'] for k in top]), axis=0)
        tn = np.linalg.norm(tpl) + 1e-30
        corr = np.array([f['w'] @ tpl / (np.linalg.norm(f['w']) * tn + 1e-30) if f['w'] is not None else np.nan
                         for f in feats])
    snr = np.array([f['snr'] for f in feats], dtype=float)
    snr = np.where(np.isfinite(snr), snr, np.nanmedian(snr) if np.isfinite(snr).any() else 1.0)
    score = np.log(np.clip(snr, 1e-3, None)) + cfg.weak_split_corr_weight * np.nan_to_num(corr, nan=1.0)
    th = _otsu_threshold(score)
    if th is None:
        info['reason'] = 'no split'
        return bi, info
    lo = score < th
    hi = ~lo
    if lo.sum() < 2 or hi.sum() < cfg.weak_split_min_strong:
        info['reason'] = 'groups too small'
        return bi, info
    sd = np.sqrt((score[lo].var(ddof=1) * (lo.sum() - 1) + score[hi].var(ddof=1) * (hi.sum() - 1))
                 / (len(score) - 2))
    d = float((score[hi].mean() - score[lo].mean()) / (sd + 1e-9))
    info['separation'] = round(d, 3)
    info['weak_median_snr'] = round(float(np.median(snr[lo])), 3)
    if d < cfg.weak_split_min_separation or np.median(snr[lo]) >= cfg.weak_split_max_weak_snr:
        info['reason'] = 'populations not separated enough'
        return bi, info
    if hi.sum() < cfg.post_veto_min_kept:
        info['reason'] = 'would keep too few'
        return bi, info
    info.update(applied=True, n_removed=int(lo.sum()), removed=[int(i) for i in bi[lo]], reason='')
    return bi[hi], info


# ─────────────────────────────────────────────────────────────────────────────
#   Burst veto
# ─────────────────────────────────────────────────────────────────────────────

def burst_windows(x, fs, cfg):
    """Noise-burst windows [start, end) in samples, from the 100–450 Hz envelope of the filtered trace.

    Envelope = rms in ``burst_window_ms`` windows; a burst where it exceeds the geometric mean of its 10th and
    90th percentile (only if p90/p10 ≥ ``burst_min_ratio``), after a 11-window median, lasting at least
    ``burst_min_duration_s``.
    """
    x = np.asarray(x, dtype=float)
    lo_hz, hi_hz = cfg.burst_band_hz
    hi_hz = min(hi_hz, 0.45 * fs)
    if hi_hz <= lo_hz:
        return []
    b, a = sg.butter(4, [lo_hz / (fs / 2), hi_hz / (fs / 2)], 'band')
    y = sg.filtfilt(b, a, x)
    w = max(1, int(fs * cfg.burst_window_ms / 1000))
    n = len(y) // w
    if n < 20:
        return []
    env = np.sqrt((y[:n * w].reshape(n, w) ** 2).mean(1))
    p10, p90 = np.percentile(env, 10), np.percentile(env, 90)
    if p10 <= 0 or p90 / p10 < cfg.burst_min_ratio:
        return []
    on = sg.medfilt((env > np.sqrt(p10 * p90)).astype(float), 11) > .5
    d = np.diff(np.r_[0, on.astype(int), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(int(s * w), int(min(e * w, len(x)))) for s, e in zip(st, en)
            if (e - s) * w >= fs * cfg.burst_min_duration_s]


def lowpass_30(x, fs, cutoff_hz=30.0, baseline_hz=0.5):
    """Zero-phase Butterworth low-pass (order 3 per pass) minus a 0.5 Hz (order 2) baseline."""
    y = sg.sosfiltfilt(sg.butter(3, cutoff_hz, fs=fs, btype='lowpass', output='sos'), np.asarray(x, dtype=float))
    return y - sg.sosfiltfilt(sg.butter(2, baseline_hz, fs=fs, btype='lowpass', output='sos'), y)


def burst_veto(x, fs, bi, cfg, windows=None):
    """Remove detections inside bursts or at their borders that do not look like the clean beats."""
    x = np.asarray(x, dtype=float)
    bi = np.asarray(bi, dtype=int)
    n = len(x)
    win = burst_windows(x, fs, cfg) if windows is None else list(windows)
    info = {'applied': False, 'n_input': int(len(bi)), 'n_bursts': len(win), 'windows': [list(w_) for w_ in win],
            'n_scoped': 0, 'n_refs': 0, 'n_removed': 0, 'removed': [], 'status': 'ok'}
    burst = np.zeros(n, bool)
    edges = np.zeros(n, bool)
    margin = int(round(cfg.burst_veto_edge_s * fs))
    for s, e in win:
        burst[s:e] = True
        for border in (s, e):
            if 0 < border < n:
                edges[max(0, border - margin):min(n, border + margin + 1)] = True
    forbidden = burst | edges
    in_scope = forbidden[bi] if len(bi) else np.zeros(0, bool)
    info['n_scoped'] = int(in_scope.sum())
    if not len(bi) or not win or not in_scope.any():
        info['status'] = 'nothing_in_scope'
        return bi, info
    if not np.all(np.isfinite(x)) or n < 64 or fs <= 60:
        info['status'] = 'unsuitable_signal'
        return bi, info
    L = max(1, int(round(.05 * fs)))
    h = max(1, int(round(.015 * fs)))
    prefix = np.r_[0, np.cumsum(forbidden)]

    def clean(c, r):
        a, b = c - r, c + r + 1
        return a >= 0 and b <= n and prefix[b] == prefix[a]

    refs = [int(i) for i in bi if clean(int(i), L + h)]
    if len(refs) < 3:
        info['status'] = 'insufficient_clean_references'
        return bi, info
    y = lowpass_30(x, fs)
    dy = np.gradient(y) * fs

    def feature(i):
        a, b = max(0, i - h), min(n, i + h + 1)
        j = a + int(np.argmax(np.abs(y[a:b])))
        if j - L < 0 or j + L > n:
            return None
        sign = float(np.sign(y[j]) or 1)
        wf = sign * y[j - L:j + L]
        wf = wf - wf.mean()
        norm = float(np.linalg.norm(wf))
        amp = float(abs(y[j]))
        slope = float(np.max(np.abs(dy[max(0, j - h):min(n, j + h + 1)])))
        if not (np.isfinite(norm + amp + slope) and norm > 0 and amp > 0 and slope > 0):
            return None
        return {'anchor': j, 'w': wf, 'norm': norm, 'amp': amp, 'slope': slope}

    usable = []
    for i in refs:
        f = feature(i)
        if f is not None and clean(f['anchor'], L + h):
            usable.append(f)
    info['n_refs'] = len(usable)
    if len(usable) < 3:
        info['status'] = 'insufficient_usable_references'
        return bi, info
    tpl = np.median(np.array([f['w'] for f in usable]), axis=0)
    tpl = tpl - tpl.mean()
    tn = float(np.linalg.norm(tpl))
    ref_amp = float(np.median([f['amp'] for f in usable]))
    ref_slope = float(np.median([f['slope'] for f in usable]))
    if not (np.isfinite(tn + ref_amp + ref_slope) and tn > 0 and ref_amp > 0 and ref_slope > 0):
        info['status'] = 'degenerate_reference_model'
        return bi, info
    reject = set()
    for i in bi[in_scope]:
        f = feature(int(i))
        if f is None:
            continue
        corr = float(np.clip(f['w'] @ tpl / (f['norm'] * tn), -1, 1))
        if corr < cfg.burst_veto_corr or f['slope'] / ref_slope < cfg.burst_veto_slope \
                or f['amp'] / ref_amp < cfg.burst_veto_amp:
            reject.add(int(i))
    if reject and len(bi) - len(reject) < cfg.post_veto_min_kept:
        info['status'] = 'abstain_min_kept'
        info['n_proposed'] = len(reject)
        return bi, info
    keep = np.array([int(i) not in reject for i in bi], dtype=bool)
    info.update(applied=bool(reject), n_removed=len(reject), removed=sorted(reject))
    return bi[keep], info


def enforce_min_rr(x, fs, bi, min_rr_ms):
    """Physiological guard: of two detections closer than ``min_rr_ms`` keep the one with the larger spike
    (max |x| within ±10 ms); repeated until every interval respects the limit."""
    x = np.asarray(x, dtype=float)
    bi = np.asarray(bi, dtype=int)
    info = {'min_rr_ms': float(min_rr_ms), 'n_input': int(len(bi)), 'n_removed': 0, 'removed': []}
    if min_rr_ms <= 0 or len(bi) < 2:
        return bi, info
    h = int(.01 * fs)
    n = len(x)
    amp = np.array([np.max(np.abs(x[max(0, i - h):min(n, i + h + 1)])) for i in bi])
    keep = np.ones(len(bi), bool)
    lim = min_rr_ms / 1000 * fs
    changed = True
    while changed:
        changed = False
        idx = np.where(keep)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if bi[b] - bi[a] < lim:
                keep[a if amp[a] < amp[b] else b] = False
                changed = True
                break
    info['n_removed'] = int((~keep).sum())
    info['removed'] = [int(i) for i in bi[~keep]]
    return bi[keep], info


def apply_post_detection_vetoes(x, fs, bi, cfg):
    """Weak-population split, burst veto, physiological minimum RR, as enabled in ``cfg`` (a BeatDetectionConfig)."""
    bi = np.asarray(bi, dtype=int)
    info = {'n_input': int(len(bi))}
    if getattr(cfg, 'enable_weak_population_split', False):
        bi, info['weak_population_split'] = weak_population_split(x, fs, bi, cfg)
    if getattr(cfg, 'enable_burst_veto', False):
        bi, info['burst_veto'] = burst_veto(x, fs, bi, cfg)
    min_rr = float(getattr(cfg, 'min_rr_physiological_ms', 0.0) or 0.0)
    if min_rr > 0:
        bi, info['min_rr'] = enforce_min_rr(x, fs, bi, min_rr)
    info['n_output'] = int(len(bi))
    return bi, info
