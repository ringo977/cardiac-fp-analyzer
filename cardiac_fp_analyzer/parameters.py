"""
parameters.py — Extraction of electrophysiological parameters from FP beats.

Key parameters:
  - Beat Period (BP) / RR interval
  - Signal Amplitude (Vmax)
  - FPD (Field Potential Duration) ~ QT
  - FPDc (Fridericia): FPDc = FPD / (RR)^(1/3)
  - FPDc (Bazett): FPDc = FPD / sqrt(RR)
  - Rise Time
  - Repolarization amplitude
  - STV (short-term variability)
"""

import numpy as np
from scipy import signal as sig


def extract_beat_parameters(beat_data, beat_time, fs, rr_interval=None):
    """Extract parameters from a single segmented beat."""
    params = {}
    data = np.array(beat_data, dtype=np.float64)
    t = np.array(beat_time, dtype=np.float64)
    zero_idx = np.argmin(np.abs(t))

    # Spike amplitude
    sp, ep = max(0, zero_idx - int(0.01*fs)), min(len(data), zero_idx + int(0.02*fs))
    spike_region = data[sp:ep]
    spike_max, spike_min = np.max(spike_region), np.min(spike_region)
    params['spike_amplitude_mV'] = (spike_max - spike_min) * 1000

    # Rise time (10-90%)
    spike_amp = spike_max - spike_min
    if spike_amp > 0:
        t10, t90 = spike_min + 0.1*spike_amp, spike_min + 0.9*spike_amp
        a10 = np.where(spike_region >= t10)[0]
        a90 = np.where(spike_region >= t90)[0]
        params['rise_time_ms'] = max(0, (a90[0]-a10[0])/fs*1000) if len(a10)>0 and len(a90)>0 else np.nan
    else:
        params['rise_time_ms'] = np.nan

    # FPD (repolarization detection)
    fpd, repol_amp = _find_repolarization(data, t, zero_idx, fs)
    params['fpd_ms'] = fpd * 1000 if fpd is not None else np.nan
    params['repol_amplitude_mV'] = repol_amp * 1000

    # FPDc Fridericia & Bazett
    if fpd is not None and rr_interval is not None and rr_interval > 0:
        params['fpdc_ms'] = (fpd / (rr_interval ** (1/3))) * 1000
        params['fpdc_bazett_ms'] = (fpd / np.sqrt(rr_interval)) * 1000
    else:
        params['fpdc_ms'] = np.nan
        params['fpdc_bazett_ms'] = np.nan

    # Max dV/dt
    deriv = np.gradient(data, 1.0/fs)
    params['max_dvdt'] = np.max(np.abs(deriv[sp:ep]))

    return params


def _find_repolarization(data, t, spike_idx, fs):
    """Find the repolarization wave (T-wave equivalent) after the depol spike."""
    search_start = spike_idx + int(0.05 * fs)
    search_end = spike_idx + int(0.600 * fs)
    if search_end > len(data): search_end = len(data)
    if search_start >= search_end: return None, np.nan

    segment = data[search_start:search_end]
    if len(segment) > 20:
        try:
            nyq = 0.5 * fs
            cutoff = min(30.0, nyq * 0.8)
            b, a = sig.butter(3, cutoff / nyq, btype='low')
            seg_smooth = sig.filtfilt(b, a, segment)
        except:
            seg_smooth = segment
    else:
        seg_smooth = segment

    x = np.arange(len(seg_smooth))
    if len(x) > 2:
        coeffs = np.polyfit(x, seg_smooth, 1)
        seg_det = seg_smooth - np.polyval(coeffs, x)
    else:
        seg_det = seg_smooth

    best_idx, best_prom = None, 0
    for sign in [1, -1]:
        pks, props = sig.find_peaks(sign * seg_det, prominence=0.0001)
        if len(pks) > 0 and len(props['prominences']) > 0:
            idx = pks[np.argmax(props['prominences'])]
            prom = np.max(props['prominences'])
            if prom > best_prom:
                best_prom, best_idx = prom, idx

    if best_idx is not None:
        fpd = abs(t[search_start + best_idx] - t[spike_idx])
        return fpd, seg_smooth[best_idx]
    return None, np.nan


def extract_all_parameters(beats_data, beats_time, beat_indices, fs):
    """Extract parameters for all beats and compute summary statistics."""
    from .beat_detection import compute_beat_periods
    beat_periods = compute_beat_periods(beat_indices, fs)

    all_params = []
    fpd_vals, fpdc_vals, amp_vals = [], [], []

    for i, (bd, bt) in enumerate(zip(beats_data, beats_time)):
        rr = beat_periods[i-1] if i > 0 and i-1 < len(beat_periods) else None
        params = extract_beat_parameters(bd, bt, fs, rr_interval=rr)
        params['beat_number'] = i + 1
        params['rr_interval_ms'] = rr * 1000 if rr is not None else np.nan
        all_params.append(params)
        if not np.isnan(params['fpd_ms']): fpd_vals.append(params['fpd_ms'])
        if not np.isnan(params['fpdc_ms']): fpdc_vals.append(params['fpdc_ms'])
        amp_vals.append(params['spike_amplitude_mV'])

    summary = {}
    bp_ms = beat_periods * 1000 if len(beat_periods) > 0 else np.array([])
    for name, vals in [('beat_period_ms', bp_ms), ('spike_amplitude_mV', amp_vals),
                        ('fpd_ms', fpd_vals), ('fpdc_ms', fpdc_vals)]:
        v = np.array(vals); v = v[~np.isnan(v)]
        if len(v) > 0:
            summary[f'{name}_mean'] = np.mean(v)
            summary[f'{name}_std'] = np.std(v)
            summary[f'{name}_median'] = np.median(v)
            summary[f'{name}_cv'] = np.std(v)/np.mean(v)*100 if np.mean(v)!=0 else np.nan
            summary[f'{name}_min'] = np.min(v)
            summary[f'{name}_max'] = np.max(v)
            summary[f'{name}_n'] = len(v)
        else:
            for s in ['_mean','_std','_median','_cv','_min','_max','_n']:
                summary[f'{name}{s}'] = np.nan

    summary['bpm_mean'] = 60000/np.mean(bp_ms) if len(bp_ms)>0 and np.mean(bp_ms)>0 else np.nan
    summary['stv_ms'] = np.mean(np.abs(np.diff(bp_ms)))/np.sqrt(2) if len(bp_ms)>1 else np.nan
    summary['beat_periods'] = beat_periods
    summary['fpd_values'] = np.array([p['fpd_ms']/1000 for p in all_params if not np.isnan(p['fpd_ms'])])
    summary['fpdc_values'] = np.array([p['fpdc_ms']/1000 for p in all_params if not np.isnan(p['fpdc_ms'])])

    return all_params, summary
