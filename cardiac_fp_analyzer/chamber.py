"""
chamber.py — Chamber-level measurements of a multi-electrode tissue.

A chamber of a µHeart chip is one microtissue seen by up to 12 recording
electrodes (plus 4 stimulation electrodes). The single-electrode pipeline
measures one of them; this module uses all of them for the two quantities
where one electrode is fragile:

* **Rhythm.** Beats seen by at least ``MIN_ELECTRODES`` electrodes within
  ``SYNC_TOL_S`` are the chamber's beats. Their median interval is the beat
  period, the robust CV (1.4826 MAD / median) its regularity, and the
  synchrony (fraction of one electrode's spikes that another electrode sees
  at the same time, median over pairs) tells whether the beat still travels
  through the whole tissue. From these a ``rhythm_status``:

    regular           robust CV <= ``CV_IRREGULAR_PCT`` and synchrony >= ``SYNC_LOST``
    irregular         robust CV above the limit (proarrhythmic sign)
    conduction_lost   spikes present but seen by few electrodes together
    silent            fewer than ``MIN_COMMON_BEATS`` common beats, or spikes on
                      fewer than 30 % of the electrodes that beat at the reference
                      (tissue stopped)
    insufficient      fewer than ``MIN_ELECTRODES`` usable electrodes

* **FPD by consensus.** On a reference recording (baseline) the FPD is
  measured on every usable electrode's median beat; those shorter than
  ``FPD_MAX_FRACTION_BP`` × period are valid, their median is the chamber
  FPD, and electrodes within ``FPD_AGREE_PCT`` of it keep their own value.
  Each electrode's repolarisation wave (± ``TEMPLATE_HALF_S`` around its
  FPD) becomes a template. On a dose recording the same wave is followed:
  each template is located in the dose median beat of the same electrode by
  correlation (``locate_wave``); electrodes with correlation >=
  ``MIN_CORR`` and within ``FPD_AGREE_PCT`` of the median give the chamber
  FPD. Fridericia FPDc uses the chamber period. This is the method whose
  values matched the PHOENIX D10.1 manual analysis (Oct 2026).

``analyze_chamber`` returns a plain dict; ``analyze.analyze_single_file``
merges it into the recording's summary when ``AnalysisConfig.chamber_consensus``
is on, and the batch hands the baseline's templates to the dose recordings
of the same tissue (``reference``).
"""

import numpy as np
from scipy.signal import find_peaks

from .filtering import full_filter_pipeline, lowpass_filter
from .repolarization import find_repolarization_on_template

MIN_ELECTRODES = 3
SYNC_TOL_S = 0.06
MIN_COMMON_BEATS = 10
CV_IRREGULAR_PCT = 15.0
SYNC_LOST = 0.5
FPD_MAX_FRACTION_BP = 0.8
FPD_AGREE_PCT = 15.0
MIN_CORR = 0.8
TEMPLATE_HALF_S = 0.12
TEMPLATE_HALF_SHORT_S = 0.06
SPIKE_MIN_DISTANCE_S = 0.15
SPIKE_THRESHOLD_NOISE = 5.0
SPIKE_HIGHPASS_HZ = 10.0
PRE_S = 0.05


# ── Electrode level ───────────────────────────────────────────────────
def spikes(filtered, fs):
    """Spike times (s): peaks of the fast component of the signal (slow
    waves below SPIKE_HIGHPASS_HZ removed, so a large repolarisation wave
    is not a second spike) above SPIKE_THRESHOLD_NOISE × robust noise,
    SPIKE_MIN_DISTANCE_S apart."""
    fast = filtered - lowpass_filter(filtered, fs, cutoff=SPIKE_HIGHPASS_HZ)
    noise = 1.4826 * np.median(np.abs(fast - np.median(fast)))
    if noise <= 0:
        return np.array([])
    pk, _ = find_peaks(np.abs(fast), height=SPIKE_THRESHOLD_NOISE * noise, distance=int(SPIKE_MIN_DISTANCE_S * fs))
    return pk / fs


def median_beat(x, beats_s, fs, post_s, pre_s=PRE_S, window=None):
    """Median of the beats of one electrode, from pre_s before the spike to post_s after."""
    b = (np.asarray(beats_s) * fs).astype(int)
    if window is not None:
        b = b[(b >= window[0] * fs) & (b < window[1] * fs)]
    pre, post = int(pre_s * fs), int(post_s * fs)
    b = b[(b - pre >= 0) & (b + post <= len(x))]
    if len(b) < 8:
        return None
    return np.median(np.stack([x[i - pre:i + post] for i in b]), 0)


def locate_wave(template, beat, fs, lo_s, hi_s, half):
    """Position (ms after the spike) of ``template`` in ``beat`` (both start
    PRE_S before the spike), searched between lo_s and hi_s; returns (ms, corr)."""
    t0 = template - template.mean()
    nt = np.sqrt((t0 * t0).sum())
    best = (-2.0, None)
    pre = int(PRE_S * fs)
    for p in range(pre + int(lo_s * fs), min(pre + int(hi_s * fs), len(beat) - half)):
        s = beat[p - half:p + half]
        if len(s) != len(template):
            continue
        s0 = s - s.mean()
        den = np.sqrt((s0 * s0).sum()) * nt
        c = (s0 * t0).sum() / den if den > 0 else -2.0
        if c > best[0]:
            best = (c, p)
    if best[1] is None:
        return np.nan, best[0]
    return (best[1] - pre) / fs * 1000, best[0]


def common_beats(spike_times, k_min=MIN_ELECTRODES, tol=SYNC_TOL_S):
    """Beat times seen on at least k_min electrodes within tol seconds."""
    ev = sorted((t, j) for j, ts in enumerate(spike_times) for t in ts)
    out, i = [], 0
    while i < len(ev):
        k = i + 1
        while k < len(ev) and ev[k][0] - ev[i][0] <= tol:
            k += 1
        if len({g[1] for g in ev[i:k]}) >= k_min:
            out.append(float(np.median([g[0] for g in ev[i:k]])))
            i = k
        else:
            i += 1
    return np.array(out)


def synchrony(spike_times, tol=SYNC_TOL_S):
    """Median over electrode pairs of the fraction of one electrode's spikes
    that the other electrode also has within tol."""
    fr = []
    for a, ta in enumerate(spike_times):
        for b, tb in enumerate(spike_times):
            if a == b or len(ta) < 5 or len(tb) < 5:
                continue
            k = np.searchsorted(tb, ta)
            d = np.minimum(np.abs(ta - tb[np.clip(k - 1, 0, len(tb) - 1)]), np.abs(ta - tb[np.clip(k, 0, len(tb) - 1)]))
            fr.append(float(np.mean(d <= tol)))
    return float(np.median(fr)) if fr else np.nan


def agreeing(values, pct=FPD_AGREE_PCT):
    """Largest group of values within ±pct of one of them (the electrodes
    that agree on the FPD); ties go to the group with the longer FPD. With a
    bimodal spread (fast rhythms: 100 vs 200 ms) a plain median would fall
    between the groups and agree with nobody."""
    best = {}
    for e0, v0 in values.items():
        grp = {e: v for e, v in values.items() if abs(v / v0 - 1) <= pct / 100}
        if len(grp) > len(best) or (len(grp) == len(best) and np.median(list(grp.values())) > np.median(list(best.values()))):
            best = grp
    return best


# ── Chamber level ─────────────────────────────────────────────────────
def analyze_chamber(df, fs, electrodes, stimulation=(), cfg=None, reference=None):
    """Rhythm status and consensus FPD of one chamber.

    Parameters
    ----------
    df : DataFrame with the electrode columns (volts)
    fs : sampling rate
    electrodes : labels of the chamber's electrodes (stimulation included)
    stimulation : labels of the stimulation electrodes (used for the rhythm,
        not for the FPD templates)
    cfg : AnalysisConfig (filter and repolarisation settings) or None
    reference : None for a reference recording (consensus from this
        recording, templates built), or the 'reference' dict of the
        baseline's result to follow the same wave on a dose recording

    Returns a dict: electrodes_usable, n_common_beats, bp_ms, bp_cv_pct
    (std/mean), cv_robust_pct, synchrony, rhythm_status, status_reason,
    fpd_ms, fpdc_ms, fpd_n (electrodes agreeing), fpd_method, electrode_fpd
    ({label: ms}), electrode_corr, reference (templates to pass to the dose
    recordings; None on a dose), and 'ok' (an FPD was measured).
    """
    from .config import AnalysisConfig
    cfg = cfg or AnalysisConfig()
    fc = cfg.filtering
    cols = [e for e in electrodes if e in df.columns]
    out = {'electrodes': cols, 'electrodes_usable': [], 'n_common_beats': 0, 'bp_ms': np.nan, 'bp_cv_pct': np.nan,
           'cv_robust_pct': np.nan, 'synchrony': np.nan, 'rhythm_status': 'insufficient', 'status_reason': '',
           'fpd_ms': np.nan, 'fpdc_ms': np.nan, 'fpd_n': 0, 'fpd_method': '', 'electrode_fpd': {}, 'electrode_corr': {},
           'reference': None, 'ok': False}
    if len(cols) < MIN_ELECTRODES:
        out['status_reason'] = f'{len(cols)} electrodes in the file'
        return out

    # per electrode: filtered signal and spikes
    filt, sp, sd = {}, {}, {}
    for e in cols:
        x = np.asarray(df[e].values, dtype=np.float64)
        sd[e] = float(np.std(x[: int(60 * fs)]))
        if not (1e-6 < sd[e] < 1e-4):           # flat or saturated
            continue
        filt[e] = full_filter_pipeline(x, fs, cfg=fc)
        sp[e] = spikes(filt[e], fs)
    with_spikes = [e for e in sp if len(sp[e]) >= 10]
    periods = {e: float(np.median(np.diff(sp[e]))) for e in with_spikes}
    ref_n = int((reference or {}).get('n_usable', 0) or 0)
    if len(with_spikes) < MIN_ELECTRODES:
        out['electrodes_usable'] = with_spikes
        if ref_n >= 2 * MIN_ELECTRODES:
            out['rhythm_status'] = 'silent'
            out['status_reason'] = f'spikes on {len(with_spikes)} electrodes, {ref_n} at the reference'
        else:
            out['status_reason'] = f'{len(with_spikes)} electrodes with spikes'
        return out
    med = float(np.median(list(periods.values())))
    usable = [e for e in with_spikes if abs(periods[e] / med - 1) <= 0.20]
    out['electrodes_usable'] = usable
    # A tissue that beat on many electrodes at the reference and now shows
    # spikes on a few is a tissue that stopped, not a one-electrode tissue.
    if ref_n >= 2 * MIN_ELECTRODES and len(with_spikes) < max(MIN_ELECTRODES, int(round(0.3 * ref_n))):
        out['rhythm_status'] = 'silent'
        out['status_reason'] = f'spikes on {len(with_spikes)} electrodes, {ref_n} at the reference'
        return out

    # rhythm of the chamber from the common beats
    cb = common_beats([sp[e] for e in usable]) if len(usable) >= MIN_ELECTRODES else np.array([])
    out['n_common_beats'] = int(len(cb))
    d = np.diff(cb)
    if len(d) >= 3:
        bp = float(np.median(d))
        out['bp_ms'] = bp * 1000
        out['bp_cv_pct'] = float(np.std(d) / np.mean(d) * 100)
        out['cv_robust_pct'] = float(1.4826 * np.median(np.abs(d - bp)) / bp * 100)
    out['synchrony'] = synchrony([sp[e] for e in usable]) if len(usable) >= 2 else np.nan

    if len(usable) < MIN_ELECTRODES:
        out['rhythm_status'] = 'insufficient'
        out['status_reason'] = f'{len(usable)} electrodes with a consistent rhythm'
    elif len(cb) < MIN_COMMON_BEATS:
        out['rhythm_status'] = 'silent'
        out['status_reason'] = f'{len(cb)} beats seen by {MIN_ELECTRODES} or more electrodes'
    elif np.isfinite(out['synchrony']) and out['synchrony'] < SYNC_LOST:
        out['rhythm_status'] = 'conduction_lost'
        out['status_reason'] = f"beats seen together by {out['synchrony'] * 100:.0f} % of the electrode pairs"
    elif out['cv_robust_pct'] > CV_IRREGULAR_PCT:
        out['rhythm_status'] = 'irregular'
        out['status_reason'] = f"robust CV of the beat interval {out['cv_robust_pct']:.0f} %"
    else:
        out['rhythm_status'] = 'regular'

    if out['rhythm_status'] != 'regular':
        return out

    # FPD per electrode on the median beat (20 Hz low-pass), recording electrodes only
    bp = out['bp_ms'] / 1000
    post = min(4.0, max(1.15, 0.95 * bp))
    hw, hws = int(TEMPLATE_HALF_S * fs), int(TEMPLATE_HALF_SHORT_S * fs)
    rec = [e for e in usable if e not in stimulation] or usable
    beats_lp = {}
    for e in rec:
        lp = lowpass_filter(filt[e], fs, cutoff=20.0)
        m = median_beat(lp, sp[e], fs, post)
        if m is not None:
            beats_lp[e] = m

    if reference is None:
        # consensus from this recording
        fpd_e = {}
        for e, m in beats_lp.items():
            try:
                f_s, _sign, _amp, _conf, _pk, _det = find_repolarization_on_template(m, fs, pre_ms=PRE_S * 1000,
                                                                                        cfg=cfg.repolarization, median_bp_s=bp)
            except (ValueError, IndexError):
                f_s = None
            if f_s is not None and f_s > 0:
                fpd_e[e] = f_s / fs * 1000
        valid = {e: f for e, f in fpd_e.items() if f < FPD_MAX_FRACTION_BP * out['bp_ms']}
        out['electrode_fpd'] = fpd_e
        if len(valid) < MIN_ELECTRODES:
            out['fpd_method'] = 'consensus'
            out['status_reason'] = f'{len(valid)} electrodes with a valid FPD'
            return out
        agree = agreeing(valid)
        if len(agree) < MIN_ELECTRODES:
            out['fpd_method'] = 'consensus'
            out['status_reason'] = f'{len(valid)} electrodes with a valid FPD, {len(agree)} agreeing'
            return out
        cons = float(np.median(list(agree.values())))
        out.update({'fpd_ms': cons, 'fpd_n': len(agree), 'fpd_method': 'consensus', 'ok': True,
                    'fpd_spread_note': '' if len(agree) >= 0.5 * len(valid) else
                    f'{len(valid) - len(agree)} of {len(valid)} electrodes give another FPD'})
        ref = {}
        for e, m in beats_lp.items():
            f0 = agree.get(e, cons)
            k = int(PRE_S * fs) + int(f0 / 1000 * fs)
            if k - hw >= 0 and k + hw <= len(m):
                ref[e] = {'f0_ms': float(f0), 'template': m[k - hw:k + hw].astype(np.float32).tolist(),
                          'template_short': m[k - hws:k + hws].astype(np.float32).tolist()}
        out['reference'] = {'fs': fs, 'bp_ms': out['bp_ms'], 'fpd_ms': out['fpd_ms'], 'electrodes': ref,
                            'n_usable': len(usable)}
    else:
        # same wave as the reference, per electrode
        refs = reference.get('electrodes', {})
        if abs(float(reference.get('fs', fs)) - fs) > 1e-6:
            out['fpd_method'] = 'same wave'
            out['status_reason'] = 'reference at another sample rate'
            return out
        res, corr = {}, {}
        for e, m in beats_lp.items():
            r = refs.get(e)
            if r is None:
                continue
            short = 0.9 * bp - TEMPLATE_HALF_S <= 0.27
            tpl = np.asarray(r['template_short' if short else 'template'], dtype=np.float64)
            half = hws if short else hw
            lo = 0.07 if short else 0.25
            hi = min(3.0, 0.9 * bp - half / fs)
            if hi <= lo + 0.02:
                continue
            f, c = locate_wave(tpl, m, fs, lo, hi, half)
            if np.isfinite(f):
                res[e], corr[e] = f, c
        out['electrode_fpd'], out['electrode_corr'] = res, corr
        good = {e: f for e, f in res.items() if corr[e] >= MIN_CORR}
        out['fpd_method'] = 'same wave'
        if len(good) < MIN_ELECTRODES:
            out['status_reason'] = f'{len(good)} electrodes follow the reference wave (correlation >= {MIN_CORR})'
            return out
        agree = agreeing(good)
        if len(agree) < MIN_ELECTRODES:
            out['status_reason'] = f'{len(good)} electrodes follow the reference wave but {len(agree)} agree on the FPD'
            return out
        out.update({'fpd_ms': float(np.median(list(agree.values()))), 'fpd_n': len(agree), 'ok': True,
                    'fpd_spread_note': '' if len(agree) >= 0.5 * len(good) else
                    f'{len(good) - len(agree)} of {len(good)} electrodes give another FPD'})
    out['fpdc_ms'] = out['fpd_ms'] / (out['bp_ms'] / 1000) ** (1 / 3)
    return out


def describe(ch):
    """One line for logs and the UI."""
    st = ch.get('rhythm_status', '')
    s = f"camera: {len(ch.get('electrodes_usable', []))} elettrodi, periodo {ch['bp_ms']:.0f} ms" if np.isfinite(ch.get('bp_ms', np.nan)) \
        else 'camera: periodo non misurabile'
    s += f", ritmo {st}"
    if ch.get('status_reason'):
        s += f" ({ch['status_reason']})"
    if ch.get('ok'):
        s += f"; FPD {ch['fpd_ms']:.0f} ms su {ch['fpd_n']} elettrodi ({ch['fpd_method']}), FPDc {ch['fpdc_ms']:.0f} ms"
    return s
