"""The operators' depolarisation convention (their MATLAB FP analyzer v1.5), as an option.

``BeatDetectionConfig.dep_time_rule``:

* ``'peak'`` (default) — the software's convention: the depolarisation time is the crest of the spike;
* ``'steepest_fall'`` — the lab's: the time is the **steepest downstroke** of the depolarisation complex (minimum
  of dV/dt), the same phase whether the complex is negative-first or positive-first, so no "side of the spike"
  has to be chosen; the amplitude is the highest minus the lowest point of the complex read on the
  **unfiltered** signal, with the lab's windows.

What is reproduced (``fp_filter_signal``, ``fp_detect_beats``, ``fp_measure_beat``, ``fp_derive``):

* traces: every mains harmonic notched (50 … 450 Hz, Q 30), then a 20–300 Hz band-pass (Butterworth 2, zero
  phase) for the spike and a 100 Hz low-pass (Butterworth 4, zero phase) for the extrema; dV/dt of the latter;
* time: the 20–300 Hz |peak| of the complex, then the minimum of dV/dt within ±12 ms of it;
* lowest point: minimum of the 100 Hz trace in the 50 ms after the downstroke, moved to the raw minimum within
  ±20 ms, then ±2 ms;
* highest point: the taller, on the raw signal, of the peak in the 40 ms before the downstroke and the overshoot
  in the 100 ms after the lowest point (each a genuine local maximum of the 100 Hz trace, moved to the raw
  maximum within ±20 ms), then ±2 ms.

On the Exp1 baseline (16 tissues, 422 beats) this gives the operators' depolarisation time to the sample in
99 % of the beats, their "Dep. Voltage up" in 98 % and "down" in 80 % (the rest are their hand corrections).
"""
from __future__ import annotations

import numpy as np
import scipy.signal as sg

RULE_PEAK = 'peak'
RULE_STEEPEST_FALL = 'steepest_fall'
RULES = (RULE_PEAK, RULE_STEEPEST_FALL)


def _p(cfg, name, default):
    v = getattr(cfg, name, None) if cfg is not None else None
    return default if v is None else v


def rule_of(cfg) -> str:
    """The configured rule (``AnalysisConfig`` or ``BeatDetectionConfig``), ``'peak'`` when unset or unknown."""
    bd = getattr(cfg, 'beat_detection', cfg)
    r = getattr(bd, 'dep_time_rule', RULE_PEAK) if bd is not None else RULE_PEAK
    return r if r in RULES else RULE_PEAK


def lab_traces(x, fs, cfg=None) -> dict:
    """The lab's traces from the untouched signal ``x``: ``notched``, ``vspike`` (20–300 Hz), ``v`` (100 Hz
    low-pass) and ``dv`` (dV/dt of ``v``, per second)."""
    x = np.asarray(x, dtype=float)
    fs = float(fs)
    nyq = fs / 2.0
    y = x.copy()
    mains = float(_p(cfg, 'lab_mains_hz', 50.0))
    q = float(_p(cfg, 'lab_notch_q', 30.0))
    fmax = min(float(_p(cfg, 'lab_mains_max_hz', 450.0)), 0.95 * nyq)
    if len(y) > 30 and mains > 0:
        for f0 in np.arange(mains, fmax + 1e-9, mains):
            b, a = sg.iirnotch(f0, q, fs=fs)
            y = sg.filtfilt(b, a, y)
    lo, hi = _p(cfg, 'lab_spike_band_hz', (20.0, 300.0))
    hi = min(float(hi), 0.45 * fs)
    b, a = sg.butter(2, [float(lo) / nyq, hi / nyq], btype='band')
    vspike = sg.filtfilt(b, a, y)
    b, a = sg.butter(4, min(float(_p(cfg, 'lab_wide_lp_hz', 100.0)), 0.45 * fs) / nyq, btype='low')
    v = sg.filtfilt(b, a, y)
    return {'notched': y, 'vspike': vspike, 'v': v, 'dv': np.gradient(v, 1.0 / fs), 'fs': fs}


def steepest_fall(tr, i, cfg=None, search_s=None) -> int:
    """The depolarisation time near sample ``i``: the 20–300 Hz |peak| within ±``search_s`` (default
    ``lab_search_ms`` = 25 ms), then the minimum of dV/dt within ±``lab_cplx_half_ms`` (12 ms) of it."""
    fs = tr['fs']
    n = len(tr['v'])
    i = int(i)
    s = int(round((float(_p(cfg, 'lab_search_ms', 25.0)) / 1000.0 if search_s is None else search_s) * fs))
    a, b = max(0, i - s), min(n, i + s + 1)
    if b - a < 3:
        return i
    c = a + int(np.argmax(np.abs(tr['vspike'][a:b])))
    w = max(1, int(round(float(_p(cfg, 'lab_cplx_half_ms', 12.0)) / 1000.0 * fs)))
    lo, hi = max(0, c - w), min(n, c + w + 1)
    return lo + int(np.argmin(tr['dv'][lo:hi]))


def _refine(x, j, w, sgn):
    lo, hi = max(0, j - w), min(len(x), j + w + 1)
    seg = x[lo:hi]
    return lo + int(np.argmax(seg) if sgn > 0 else np.argmin(seg))


def _local_max(v, lo, hi, fallback):
    lo, hi = max(0, int(round(lo))), min(len(v) - 1, int(round(hi)))
    if hi - lo < 2:
        return fallback
    seg = v[lo:hi + 1]
    pk, _ = sg.find_peaks(seg)
    if len(pk) == 0:
        return fallback
    return lo + int(pk[int(np.argmax(seg[pk]))])


def extrema(raw, tr, i_dep, cfg=None):
    """(lowest, highest) sample of the depolarisation complex whose steepest downstroke is ``i_dep``, read
    on the untouched signal ``raw`` (see the module docstring)."""
    fs = tr['fs']
    v = tr['v']
    raw = np.asarray(raw, dtype=float)
    n = len(v)
    i = int(i_dep)
    ms = lambda name, d: int(round(float(_p(cfg, name, d)) / 1000.0 * fs))  # noqa: E731
    wr, w2 = ms('lab_peak_refine_ms', 20.0), ms('lab_amp_refine_ms', 2.0)
    hi_t = min(n - 1, i + ms('lab_trough_ms', 50.0))
    if hi_t <= i:
        return i, i
    i_min = _refine(raw, i + int(np.argmin(v[i:hi_t + 1])), wr, -1)
    i_pre = _local_max(v, i - ms('lab_prepeak_ms', 40.0), i, i)
    i_post = _local_max(v, i_min, i_min + ms('lab_postpeak_ms', 100.0), None)
    cands = [_refine(raw, i_pre, wr, +1)]
    if i_post is not None:
        cands.append(_refine(raw, i_post, wr, +1))
    i_max = max(cands, key=lambda q: raw[q])
    return _refine(raw, i_min, w2, -1), _refine(raw, i_max, w2, +1)


def amplitude(raw, tr, i_dep, cfg=None):
    """(highest − lowest, highest, lowest, i_low, i_high) of the complex on ``raw`` (same units as ``raw``)."""
    i_lo, i_hi = extrema(raw, tr, i_dep, cfg)
    raw = np.asarray(raw, dtype=float)
    return float(raw[i_hi] - raw[i_lo]), float(raw[i_hi]), float(raw[i_lo]), int(i_lo), int(i_hi)


def anchors(bi, x, fs, cfg=None, tr=None) -> np.ndarray:
    """The detections ``bi`` moved to the steepest downstroke of their complex (sorted, unique)."""
    bi = np.asarray(bi, dtype=int)
    if len(bi) == 0:
        return bi
    tr = tr if tr is not None else lab_traces(x, fs, cfg)
    return np.unique(np.asarray([steepest_fall(tr, b, cfg) for b in bi], dtype=int))
