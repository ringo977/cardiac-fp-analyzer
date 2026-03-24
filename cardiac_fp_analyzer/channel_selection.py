"""
channel_selection.py — Automatic electrode channel selection.

Scores each available electrode (el1, el2) by beat quality, morphology
correlation, amplitude and regularity, then returns the best channel.

All scoring weights are configurable via ChannelSelectionConfig.
"""

import logging

import numpy as np

from .beat_detection import compute_beat_periods, detect_beats, segment_beats
from .filtering import full_filter_pipeline

logger = logging.getLogger(__name__)


def select_best_channel(df, fs, cfg=None):
    """Select the best electrode based on beat detection quality.

    Parameters
    ----------
    df : DataFrame with columns 'el1', 'el2', 'time'
    fs : sampling rate (Hz)
    cfg : AnalysisConfig or None

    Returns
    -------
    best_ch : str ('el1' or 'el2')
    details : dict  channel -> description string
    """
    if cfg is not None:
        cs = cfg.channel_selection
        fc = cfg.filtering
    else:
        from .config import ChannelSelectionConfig, FilterConfig
        cs = ChannelSelectionConfig()
        fc = FilterConfig()

    best_ch, best_score = 'el1', -999
    gain = cfg.amplifier_gain if cfg is not None else 1.0
    details = {}
    for ch in ['el1', 'el2']:
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
                post_ms = rep_cfg.search_end_ms + 50 if rep_cfg else 900
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
