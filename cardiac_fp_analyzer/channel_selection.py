"""
channel_selection.py — Automatic electrode channel selection.

Scores each available electrode (el1, el2 of a CSV, or every electrode
column of an MCS HDF5 file) by beat quality, morphology correlation,
amplitude and regularity, then returns the best channel.

All scoring weights are configurable via ChannelSelectionConfig.
"""

import logging

import numpy as np

from .beat_detection import compute_beat_periods, detect_beats, segment_beats
from .filtering import full_filter_pipeline

logger = logging.getLogger(__name__)


def select_best_channel(df, fs, cfg=None, channels=None):
    """Select the best electrode based on beat detection quality.

    Parameters
    ----------
    df : DataFrame with 'time' and the electrode columns ('el1', 'el2', or
        electrode labels)
    fs : sampling rate (Hz)
    cfg : AnalysisConfig or None
    channels : list of column names to score, default all electrode columns

    Returns
    -------
    best_ch : str (column name)
    details : dict  channel -> description string
    """
    if cfg is not None:
        cs = cfg.channel_selection
        fc = cfg.filtering
    else:
        from .config import ChannelSelectionConfig, FilterConfig
        cs = ChannelSelectionConfig()
        fc = FilterConfig()

    if channels is None:
        channels = [c for c in df.columns if c != 'time']
    best_ch, best_score = channels[0], -999
    gain = cfg.amplifier_gain if cfg is not None else 1.0
    details = {}
    for ch in channels:
        try:
            raw_ch = df[ch].values
            if gain != 1.0:
                raw_ch = raw_ch / gain
            filt = full_filter_pipeline(raw_ch, fs, cfg=fc)
            bi, bt, info = detect_beats(filt, fs, method='auto', min_distance_ms=400)
            bp = compute_beat_periods(bi, fs)
            score = 0
            if len(bp) > 2:
                mbp = np.mean(bp)
                cv = np.std(bp) / mbp if mbp > 0 else 999

                # ── 1. Beat period in physiological range ──
                if cs.bp_ideal_range_s[0] <= mbp <= cs.bp_ideal_range_s[1]:
                    score += cs.w_bp_range

                # ── 2. Beat rate reasonable ──
                rate = len(bi) / (len(df) / fs)
                if cs.rate_range_per_s[0] <= rate <= cs.rate_range_per_s[1]:
                    score += cs.w_rate_ok

                # ── 3. Template correlation — dominant criterion ──
                rep_cfg = cfg.repolarization if cfg else None
                pre_ms = rep_cfg.segment_pre_ms if rep_cfg else 50
                # Adaptive post_ms (mirrors analyze.py): cover both fixed
                # search_end_ms AND adaptive search_end_pct_rr × RR window,
                # so template-correlation scoring is computed on the full
                # repolarization tail for slow rhythms. `mbp` is already
                # computed above from this channel's beats.
                if rep_cfg is not None:
                    pct_rr = getattr(rep_cfg, 'search_end_pct_rr', 0.0)
                    adaptive_end_ms = (pct_rr * mbp * 1000.0
                                       if (pct_rr > 0 and mbp > 0) else 0.0)
                    post_ms = max(rep_cfg.search_end_ms + 50.0,
                                  adaptive_end_ms + 50.0)
                else:
                    post_ms = 900
                bd, btm, vi = segment_beats(filt, df['time'].values, bi, fs,
                                            pre_ms=pre_ms, post_ms=post_ms)
                if len(bd) >= 3:
                    min_len = min(len(b) for b in bd)
                    template = np.mean([b[:min_len] for b in bd], axis=0)
                    corrs = [np.corrcoef(b[:min_len], template)[0, 1]
                             for b in bd if len(b) >= min_len]
                    mean_corr = np.nanmean(corrs) if corrs else 0
                else:
                    mean_corr = 0
                score += max(0, min(cs.w_corr_max,
                                    round(mean_corr * cs.w_corr_scale - cs.w_corr_offset, 1)))

                # ── 4. Beat-period regularity ──
                cv_pct = cv * 100
                score += max(0, round(cs.w_regularity_max - cv_pct * cs.w_regularity_slope, 1))

                # ── 5. Spike amplitude ──
                ptp_per_beat = [np.ptp(b) for b in bd] if len(bd) > 0 else [0]
                median_ptp_mV = np.median(ptp_per_beat) * 1000
                score += min(cs.w_amplitude_max,
                             round(median_ptp_mV / cs.w_amplitude_ref_mV * cs.w_amplitude_max, 1))

                p5, p95 = np.percentile(filt, [5, 95])
                nm = (filt >= p5) & (filt <= p95)
                ns = np.std(filt[nm]) if np.sum(nm) > 100 else np.std(filt)
                snr = np.mean(np.abs(filt[bi])) / ns if ns > 0 else 0

                details[ch] = (f'{len(bi)} beats, BP={mbp*1000:.0f}ms, CV={cv_pct:.1f}%, '
                               f'ptp={median_ptp_mV:.0f}mV, corr={mean_corr:.3f}, '
                               f'SNR={snr:.1f}, score={score:.1f}')
            else:
                details[ch] = f'{len(bi)} beats (too few)'
            if score > best_score:
                best_score, best_ch = score, ch
        except (ValueError, IndexError, RuntimeError) as e:
            logger.debug("Channel %s scoring failed: %s", ch, e)
            details[ch] = f'error: {e}'
    return best_ch, details


# ──────────────────────────────────────────────────────────────────────
#   Quick scoring for many electrodes (multi-chamber MCS files)
# ──────────────────────────────────────────────────────────────────────

def quick_electrode_scores(df, fs, channels, cfg=None, exclude=(), min_beats=10, bp_range_s=(0.2, 6.0)):
    """Score electrodes cheaply (about 0.15 s each at 2 kHz) to pick the one to
    analyse in a chamber of a multi-electrode chip.

    For each electrode: band-pass as the pipeline, spikes = peaks of the fast
    component (slow waves below 10 Hz removed) above 5 × the robust noise
    (1.4826 MAD) at least 150 ms apart; then

      snr          median spike peak-to-peak / noise
      cv_pct       robust CV of the intervals (1.4826 MAD / median)
      repol_snr    peak-to-peak of the median beat (20 Hz low-pass) between
                   200 ms and 0.9 × period, over the low-pass noise
      score        min(snr, 30) + 1.5·min(repol_snr, 10) − 0.4·cv_pct,
                   −inf when fewer than ``min_beats`` spikes or the period is
                   outside ``bp_range_s``

    Electrodes in ``exclude`` (stimulation electrodes) get score −inf and
    reason 'stimulation'. Returns {label: dict(score, snr, cv_pct, repol_snr,
    n_beats, period_ms, reason)}.
    """
    from .filtering import lowpass_filter
    fc = cfg.filtering if cfg is not None else None
    if fc is None:
        from .config import FilterConfig
        fc = FilterConfig()
    from scipy.signal import find_peaks
    out = {}
    for ch in channels:
        r = {'score': -np.inf, 'snr': np.nan, 'cv_pct': np.nan, 'repol_snr': np.nan, 'n_beats': 0,
             'period_ms': np.nan, 'reason': ''}
        out[ch] = r
        if ch in exclude:
            r['reason'] = 'stimulation'
            continue
        x = np.asarray(df[ch].values, dtype=np.float64)
        if not np.isfinite(x).all() or np.ptp(x) == 0:
            r['reason'] = 'flat'
            continue
        f = full_filter_pipeline(x, fs, cfg=fc)
        fast_part = f - lowpass_filter(f, fs, cutoff=10.0)      # spikes only: slow repolarisation waves removed
        noise = 1.4826 * np.median(np.abs(fast_part - np.median(fast_part)))
        if noise <= 0:
            r['reason'] = 'flat'
            continue
        pk, _ = find_peaks(np.abs(fast_part), height=5.0 * noise, distance=int(0.15 * fs))
        r['n_beats'] = int(len(pk))
        if len(pk) < min_beats:
            r['reason'] = 'too few spikes'
            continue
        d = np.diff(pk) / fs
        bp = float(np.median(d))
        r['period_ms'] = bp * 1000
        r['cv_pct'] = float(1.4826 * np.median(np.abs(d - bp)) / bp * 100)
        w = int(0.02 * fs)
        amp = np.median([np.ptp(f[max(0, k - w):k + w]) for k in pk])
        r['snr'] = float(amp / noise)
        if not (bp_range_s[0] <= bp <= bp_range_s[1]):
            r['reason'] = 'period out of range'
            continue
        lp = lowpass_filter(f, fs, cutoff=20.0)
        post = int(min(0.9 * bp, 1.5) * fs)
        pre = int(0.05 * fs)
        segs = [lp[k - pre:k + post] for k in pk if k - pre >= 0 and k + post <= len(lp)]
        if len(segs) >= 5:
            m = np.median(np.stack(segs), 0)
            a, b = pre + int(0.2 * fs), len(m)
            if b - a > int(0.1 * fs):
                lp_noise = 1.4826 * np.median(np.abs(lp - np.median(lp)))
                r['repol_snr'] = float(np.ptp(m[a:b]) / lp_noise) if lp_noise > 0 else np.nan
        rep = 0.0 if not np.isfinite(r['repol_snr']) else min(r['repol_snr'], 10.0)
        r['score'] = float(min(r['snr'], 30.0) + 1.5 * rep - 0.4 * min(r['cv_pct'], 100.0))
    return out


def select_electrode_quick(df, fs, channels, cfg=None, exclude=()):
    """Best electrode among ``channels`` by ``quick_electrode_scores``; the
    excluded ones are used only when no other electrode scores. Returns
    (label, scores)."""
    scores = quick_electrode_scores(df, fs, channels, cfg=cfg, exclude=exclude)
    ranked = sorted(scores.items(), key=lambda kv: kv[1]['score'], reverse=True)
    if ranked and np.isfinite(ranked[0][1]['score']):
        return ranked[0][0], scores
    if exclude:
        fallback = quick_electrode_scores(df, fs, [c for c in channels if c in exclude], cfg=cfg)
        ranked = sorted(fallback.items(), key=lambda kv: kv[1]['score'], reverse=True)
        if ranked and np.isfinite(ranked[0][1]['score']):
            scores.update(fallback)
            return ranked[0][0], scores
    return channels[0], scores
