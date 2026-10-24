"""Drift of the beating rate (and of the FPD) within one recording — e.g. a tissue warming up or cooling.

``rate_drift(beat_times_s, fpd_s=None)`` fits a robust straight line (Theil–Sen, with its 95 % confidence
band) to the instantaneous rate — 60 / interval, placed at the middle of each interval — against time,
after leaving out the intervals that are not two consecutive beats (longer than 1.5 × the median: a
missing beat; shorter than 0.5 ×: a detection in excess). It reports the slope in bpm per minute, the
rate at the start and at the end of the recording from the fitted line, the change over the recording in
% of the starting rate and the same change per minute (an extrapolation when the recording is shorter
than a minute, said so in ``nota``), whether the slope is distinguishable from zero (the confidence band
excludes 0) and whether the estimate is reliable at all (at least ``min_intervals`` usable intervals over
at least ``min_duration_s``). The FPD gets the same treatment (ms per minute, % over the recording).
"""
from __future__ import annotations

import numpy as np

from .i18n import tr

MIN_INTERVALS = 10
MIN_DURATION_S = 20.0
GAP_FACTOR = 1.5
CLOSE_FACTOR = 0.5


def _theil_sen(x, y):
    """Slope, intercept and 95 % slope bounds (scipy's Theil–Sen); NaN bounds when they cannot be computed."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res = stats.theilslopes(y, x, alpha=0.95)
    slope, intercept, lo, hi = (float(v) for v in (res[0], res[1], res[2], res[3]))
    return slope, intercept, lo, hi


def _line(x, y, min_points, duration_ok):
    """Fit and describe one series (rate or FPD) against time."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    out = {'n': int(len(x)), 'pendenza_per_min': None, 'ci_per_min': None, 'inizio': None, 'fine': None,
           'variazione_pct': None, 'variazione_pct_min': None, 'significativa': False, 'affidabile': False, 'motivo': ''}
    if len(x) < 3 or np.ptp(x) <= 0:
        out['motivo'] = tr('too few points')
        return out
    slope, intercept, lo, hi = _theil_sen(x, y)
    span = float(x.max() - x.min())
    start, end = intercept + slope * x.min(), intercept + slope * x.max()
    out.update(pendenza_per_min=slope * 60.0, ci_per_min=(lo * 60.0, hi * 60.0), inizio=float(start), fine=float(end),
               durata_s=span)
    if start != 0 and np.isfinite(start):
        out['variazione_pct'] = float((end - start) / abs(start) * 100.0)
        out['variazione_pct_min'] = float(slope * 60.0 / abs(start) * 100.0)
    out['significativa'] = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
    if len(x) < min_points:
        out['motivo'] = tr('only {0} usable intervals (minimum {1})').format(len(x), min_points)
    elif not duration_ok:
        out['motivo'] = tr('recording too short')
    else:
        out['affidabile'] = True
    return out


def rate_drift(beat_times_s, fpd_s=None, *, min_intervals=MIN_INTERVALS, min_duration_s=MIN_DURATION_S,
               gap_factor=GAP_FACTOR, close_factor=CLOSE_FACTOR):
    """Drift of the rate (and of the FPD when ``fpd_s`` — one value per beat, NaN when missing — is given).

    Returns a dict: ``n_intervalli``, ``durata_s``, ``bpm_inizio``, ``bpm_fine``, ``pendenza_bpm_min``,
    ``ci_bpm_min`` (lower, upper), ``variazione_pct`` (over the recording), ``variazione_pct_min``,
    ``significativa``, ``affidabile``, ``motivo``, ``nota`` ('stima estrapolata al minuto' when the recording is
    shorter than 60 s), ``fpd`` (same keys for the FPD in ms: ``pendenza_ms_min``, ``ci_ms_min``,
    ``fpd_inizio_ms``, ``fpd_fine_ms``, ``variazione_pct``, …) or None.
    """
    t = np.asarray(beat_times_s, dtype=float)
    t = t[np.isfinite(t)]
    t.sort()
    out = {'n_intervalli': 0, 'durata_s': float(t[-1] - t[0]) if len(t) > 1 else 0.0, 'bpm_inizio': None,
           'bpm_fine': None, 'pendenza_bpm_min': None, 'ci_bpm_min': None, 'variazione_pct': None,
           'variazione_pct_min': None, 'significativa': False, 'affidabile': False, 'motivo': '', 'nota': '', 'fpd': None}
    if len(t) < 3:
        out['motivo'] = tr('too few beats')
        return out
    rr = np.diff(t)
    med = float(np.median(rr))
    keep = (rr <= gap_factor * med) & (rr >= close_factor * med) & (rr > 0)
    mid = 0.5 * (t[:-1] + t[1:])
    rate = 60.0 / rr[keep]
    duration_ok = out['durata_s'] >= min_duration_s
    line = _line(mid[keep], rate, min_intervals, duration_ok)
    out.update(n_intervalli=line['n'], bpm_inizio=line['inizio'], bpm_fine=line['fine'],
               pendenza_bpm_min=line['pendenza_per_min'], ci_bpm_min=line['ci_per_min'],
               variazione_pct=line['variazione_pct'], variazione_pct_min=line['variazione_pct_min'],
               significativa=line['significativa'], affidabile=line['affidabile'], motivo=line['motivo'])
    if out['durata_s'] < 60.0 and line['pendenza_per_min'] is not None:
        out['nota'] = tr('estimate extrapolated to one minute (recording of {0:.0f} s)').format(out['durata_s'])
    if fpd_s is not None:
        f = np.asarray(fpd_s, dtype=float)
        if len(f) == len(beat_times_s):
            tf = np.asarray(beat_times_s, dtype=float)
            ok = np.isfinite(f) & np.isfinite(tf)
            if ok.sum() >= 3:
                lf = _line(tf[ok], f[ok] * 1000.0, min_intervals, duration_ok)
                out['fpd'] = {'n': lf['n'], 'pendenza_ms_min': lf['pendenza_per_min'], 'ci_ms_min': lf['ci_per_min'],
                              'fpd_inizio_ms': lf['inizio'], 'fpd_fine_ms': lf['fine'],
                              'variazione_pct': lf['variazione_pct'], 'variazione_pct_min': lf['variazione_pct_min'],
                              'significativa': lf['significativa'], 'affidabile': lf['affidabile'], 'motivo': lf['motivo']}
    return out


def drift_text(d, *, threshold_pct=5.0):
    """One line for the operator: e.g. 'rate +4.1 % in 58 s (+1.3 bpm/min, 95 % CI −0.2…2.8), estimate
    extrapolated to one minute; FPD −2.0 %'. Empty when there is no estimate."""
    if not d or d.get('pendenza_bpm_min') is None:
        return ''
    ci = d.get('ci_bpm_min') or (float('nan'), float('nan'))
    parts = [tr('rate {0:+.1f} % in {1:.0f} s ({2:+.2f} bpm/min, 95 % CI {3:+.2f}…{4:+.2f})').format(
        d['variazione_pct'], d['durata_s'], d['pendenza_bpm_min'], ci[0], ci[1])]
    if not d.get('affidabile'):
        parts.append(tr('unreliable estimate: {0}').format(d.get('motivo', '')))
    elif not d.get('significativa'):
        parts.append(tr('not distinguishable from zero'))
    if d.get('nota'):
        parts.append(d['nota'])
    f = d.get('fpd')
    if f and f.get('pendenza_ms_min') is not None:
        parts.append(f"FPD {f['variazione_pct']:+.1f} % ({f['pendenza_ms_min']:+.1f} ms/min)"
                     + ('' if f.get('significativa') else ', ' + tr('not distinguishable from zero')))
    return ', '.join(parts)


__all__ = ['rate_drift', 'drift_text', 'MIN_INTERVALS', 'MIN_DURATION_S']
