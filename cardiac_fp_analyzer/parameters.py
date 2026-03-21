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

FPD measurement strategy (inspired by Visone et al., Tox Sci 2023):
  1. Build an averaged beat template via cross-correlation alignment
  2. Detect the repolarization wave on the clean template → reference FPD
  3. Refine FPD per-beat using a template-guided search window
"""

import numpy as np
from scipy import signal as sig


# ─── Template averaging ───

def _align_beats_xcorr(beats_data, fs, max_shift_ms=50):
    """
    Align beats via cross-correlation to a reference beat.

    The reference is initially the median beat, then refined iteratively.
    Returns aligned beat waveforms (all same length).
    """
    if len(beats_data) < 3:
        return beats_data

    max_shift = int(max_shift_ms * fs / 1000)

    # Ensure uniform length
    min_len = min(len(b) for b in beats_data)
    beats = [b[:min_len].copy() for b in beats_data]

    # Reference: median of all beats (robust starting point)
    ref = np.median(np.array(beats), axis=0)

    # Align each beat to reference via cross-correlation
    aligned = []
    for beat in beats:
        # Cross-correlate only the depolarization region (first 100ms)
        dep_len = min(int(0.1 * fs), min_len)
        corr = np.correlate(ref[:dep_len + 2*max_shift],
                           beat[:dep_len], mode='valid')
        if len(corr) == 0:
            aligned.append(beat)
            continue
        shift = np.argmax(corr) - max_shift
        shift = np.clip(shift, -max_shift, max_shift)

        # Apply shift
        if shift > 0:
            padded = np.concatenate([beat[shift:], np.full(shift, beat[-1])])
        elif shift < 0:
            padded = np.concatenate([np.full(-shift, beat[0]), beat[:shift]])
        else:
            padded = beat

        aligned.append(padded[:min_len])

    return aligned


def build_beat_template(beats_data, fs, max_beats=60):
    """
    Build a clean averaged beat template.

    1. Select up to max_beats from the recording (evenly spaced)
    2. Align via cross-correlation
    3. Compute robust median template

    Returns: template (array), or None if too few beats.
    """
    if len(beats_data) < 5:
        return None

    # Select beats evenly across the recording
    n = min(len(beats_data), max_beats)
    if n < len(beats_data):
        indices = np.linspace(0, len(beats_data)-1, n, dtype=int)
        selected = [beats_data[i] for i in indices]
    else:
        selected = list(beats_data)

    # Align beats
    aligned = _align_beats_xcorr(selected, fs)

    if not aligned:
        return None

    # Uniform length
    min_len = min(len(b) for b in aligned)
    mat = np.array([b[:min_len] for b in aligned])

    # Robust median template
    template = np.median(mat, axis=0)

    return template


# ─── Repolarization detection ───

def _find_repolarization_on_template(template, fs, pre_ms=50):
    """
    Find the repolarization wave on a clean averaged template.

    The repolarization in cardiac FP signals appears as a smooth positive
    (or negative) deflection 200-800ms after the depolarization spike.

    Strategy:
    1. Identify the depolarization spike (sharpest feature near t=0)
    2. Low-pass filter the post-spike region to isolate the slow repolarization
    3. Remove baseline drift with polynomial detrend
    4. Find the most prominent peak in the 150-800ms window

    Returns: fpd_samples (int), repol_sign (+1 or -1), repol_amplitude
    """
    pre_samples = int(pre_ms * fs / 1000)

    # Depolarization spike: find the sharpest negative or positive deflection
    # in the first 50ms around the expected spike location
    spike_search = int(0.05 * fs)
    spike_region = template[max(0, pre_samples - spike_search):
                           min(len(template), pre_samples + spike_search)]

    # The spike is the point of maximum absolute deviation
    spike_local_idx = np.argmax(np.abs(spike_region - np.mean(spike_region)))
    spike_idx = max(0, pre_samples - spike_search) + spike_local_idx

    # Search for repolarization: 150ms to 800ms after spike
    search_start = spike_idx + int(0.150 * fs)
    search_end = spike_idx + int(0.800 * fs)

    if search_end > len(template):
        search_end = len(template)
    if search_start >= search_end or (search_end - search_start) < int(0.05 * fs):
        return None, 1, 0.0

    segment = template[search_start:search_end].copy()

    # Low-pass filter at 15 Hz to isolate repolarization (slow wave)
    nyq = 0.5 * fs
    cutoff = min(15.0, nyq * 0.8)
    try:
        b, a = sig.butter(3, cutoff / nyq, btype='low')
        seg_smooth = sig.filtfilt(b, a, segment)
    except:
        seg_smooth = segment

    # Polynomial detrend (2nd order) to remove slow drift
    x = np.arange(len(seg_smooth))
    if len(x) > 10:
        coeffs = np.polyfit(x, seg_smooth, 2)
        seg_det = seg_smooth - np.polyval(coeffs, x)
    else:
        seg_det = seg_smooth - np.mean(seg_smooth)

    # Find the most prominent peak (try both polarities)
    best_fpd_samples = None
    best_prom = 0
    best_sign = 1
    best_amp = 0.0

    for sign in [1, -1]:
        pks, props = sig.find_peaks(sign * seg_det,
                                     prominence=np.std(seg_det) * 0.3,
                                     distance=int(0.05 * fs))
        if len(pks) > 0:
            # Prefer the most prominent peak
            best_pk_idx = np.argmax(props['prominences'])
            pk = pks[best_pk_idx]
            prom = props['prominences'][best_pk_idx]

            if prom > best_prom:
                best_prom = prom
                best_fpd_samples = search_start + pk - spike_idx
                best_sign = sign
                best_amp = seg_smooth[pk]

    if best_fpd_samples is not None and best_fpd_samples > 0:
        return best_fpd_samples, best_sign, best_amp

    # Fallback: use the point of maximum absolute deviation from detrended mean
    max_abs_idx = np.argmax(np.abs(seg_det))
    return search_start + max_abs_idx - spike_idx, 1, seg_smooth[max_abs_idx]


def _find_repolarization_per_beat(data, t, spike_idx, fs,
                                   template_fpd_samples=None,
                                   template_repol_sign=1,
                                   search_tolerance_ms=150):
    """
    Find repolarization in a single beat, optionally guided by template.

    If template_fpd_samples is provided, searches in a window around the
    expected position (±tolerance). Otherwise falls back to a broad search.
    """
    # Define search window
    if template_fpd_samples is not None:
        # Template-guided: search ±tolerance around expected FPD
        tolerance = int(search_tolerance_ms * fs / 1000)
        search_start = spike_idx + max(int(0.1 * fs), template_fpd_samples - tolerance)
        search_end = spike_idx + min(len(data) - spike_idx, template_fpd_samples + tolerance)
    else:
        # Broad search: 150ms to 800ms
        search_start = spike_idx + int(0.150 * fs)
        search_end = spike_idx + int(0.800 * fs)

    if search_end > len(data):
        search_end = len(data)
    if search_start >= search_end or (search_end - search_start) < 10:
        return None, np.nan

    segment = data[search_start:search_end]

    # Low-pass filter
    nyq = 0.5 * fs
    cutoff = min(15.0, nyq * 0.8)
    if len(segment) > 20:
        try:
            b, a = sig.butter(3, cutoff / nyq, btype='low')
            seg_smooth = sig.filtfilt(b, a, segment)
        except:
            seg_smooth = segment
    else:
        seg_smooth = segment

    # Detrend
    x = np.arange(len(seg_smooth))
    if len(x) > 5:
        coeffs = np.polyfit(x, seg_smooth, 2)
        seg_det = seg_smooth - np.polyval(coeffs, x)
    else:
        seg_det = seg_smooth - np.mean(seg_smooth)

    # Search for peak with the same polarity as template repolarization
    # (but also check opposite polarity as fallback)
    best_idx = None
    best_prom = 0

    for sign in [template_repol_sign, -template_repol_sign]:
        pks, props = sig.find_peaks(sign * seg_det,
                                     prominence=np.std(seg_det) * 0.2,
                                     distance=int(0.03 * fs))
        if len(pks) > 0:
            # If template-guided, prefer the peak closest to expected FPD
            if template_fpd_samples is not None:
                expected_local = template_fpd_samples - (search_start - spike_idx)
                distances = np.abs(pks - expected_local)
                # Weight by prominence and proximity
                scores = props['prominences'] / (1 + distances / (0.05 * fs))
                best_pk = pks[np.argmax(scores)]
                score = np.max(scores)
            else:
                best_pk = pks[np.argmax(props['prominences'])]
                score = np.max(props['prominences'])

            if score > best_prom:
                best_prom = score
                best_idx = best_pk

    if best_idx is not None:
        actual_idx = search_start + best_idx
        fpd = abs(t[actual_idx] - t[spike_idx])
        return fpd, seg_smooth[best_idx]

    return None, np.nan


# ─── Per-beat parameter extraction ───

def extract_beat_parameters(beat_data, beat_time, fs, rr_interval=None,
                           template_fpd_samples=None, template_repol_sign=1):
    """
    Extract parameters from a single segmented beat.

    If template_fpd_samples is provided, uses template-guided FPD detection.
    """
    params = {}
    data = np.array(beat_data, dtype=np.float64)
    t = np.array(beat_time, dtype=np.float64)
    zero_idx = np.argmin(np.abs(t))

    # Spike amplitude
    sp = max(0, zero_idx - int(0.01 * fs))
    ep = min(len(data), zero_idx + int(0.02 * fs))
    spike_region = data[sp:ep]
    spike_max, spike_min = np.max(spike_region), np.min(spike_region)
    params['spike_amplitude_mV'] = (spike_max - spike_min) * 1000

    # Rise time (10-90%)
    spike_amp = spike_max - spike_min
    if spike_amp > 0:
        t10, t90 = spike_min + 0.1 * spike_amp, spike_min + 0.9 * spike_amp
        a10 = np.where(spike_region >= t10)[0]
        a90 = np.where(spike_region >= t90)[0]
        params['rise_time_ms'] = (max(0, (a90[0] - a10[0]) / fs * 1000)
                                  if len(a10) > 0 and len(a90) > 0 else np.nan)
    else:
        params['rise_time_ms'] = np.nan

    # FPD (template-guided or standalone)
    fpd, repol_amp = _find_repolarization_per_beat(
        data, t, zero_idx, fs,
        template_fpd_samples=template_fpd_samples,
        template_repol_sign=template_repol_sign
    )
    params['fpd_ms'] = fpd * 1000 if fpd is not None else np.nan
    params['repol_amplitude_mV'] = repol_amp * 1000 if not np.isnan(repol_amp) else np.nan

    # FPDc Fridericia & Bazett
    if fpd is not None and rr_interval is not None and rr_interval > 0:
        params['fpdc_ms'] = (fpd / (rr_interval ** (1/3))) * 1000
        params['fpdc_bazett_ms'] = (fpd / np.sqrt(rr_interval)) * 1000
    else:
        params['fpdc_ms'] = np.nan
        params['fpdc_bazett_ms'] = np.nan

    # Max dV/dt
    deriv = np.gradient(data, 1.0 / fs)
    params['max_dvdt'] = np.max(np.abs(deriv[sp:ep]))

    return params


def extract_all_parameters(beats_data, beats_time, beat_indices, fs):
    """
    Extract parameters for all beats and compute summary statistics.

    Uses template averaging for robust FPD measurement:
    1. Build averaged template from aligned beats
    2. Detect repolarization on template → reference FPD
    3. Guide per-beat FPD with template reference
    """
    from .beat_detection import compute_beat_periods
    beat_periods = compute_beat_periods(beat_indices, fs)

    # ─── Template averaging for FPD reference ───
    template = build_beat_template(beats_data, fs)
    template_fpd_samples = None
    template_repol_sign = 1

    if template is not None:
        pre_ms = 50  # must match segment_beats pre_ms
        fpd_result = _find_repolarization_on_template(template, fs, pre_ms=pre_ms)
        if fpd_result[0] is not None:
            template_fpd_samples = fpd_result[0]
            template_repol_sign = fpd_result[1]
            template_fpd_ms = template_fpd_samples / fs * 1000

    # ─── Per-beat extraction ───
    all_params = []
    fpd_vals, fpdc_vals, amp_vals = [], [], []

    for i, (bd, bt) in enumerate(zip(beats_data, beats_time)):
        rr = beat_periods[i-1] if i > 0 and i-1 < len(beat_periods) else None
        params = extract_beat_parameters(
            bd, bt, fs, rr_interval=rr,
            template_fpd_samples=template_fpd_samples,
            template_repol_sign=template_repol_sign
        )
        params['beat_number'] = i + 1
        params['rr_interval_ms'] = rr * 1000 if rr is not None else np.nan
        all_params.append(params)
        if not np.isnan(params['fpd_ms']):
            fpd_vals.append(params['fpd_ms'])
        if not np.isnan(params['fpdc_ms']):
            fpdc_vals.append(params['fpdc_ms'])
        amp_vals.append(params['spike_amplitude_mV'])

    # ─── Summary statistics ───
    summary = {}
    bp_ms = beat_periods * 1000 if len(beat_periods) > 0 else np.array([])
    for name, vals in [('beat_period_ms', bp_ms), ('spike_amplitude_mV', amp_vals),
                       ('fpd_ms', fpd_vals), ('fpdc_ms', fpdc_vals)]:
        v = np.array(vals)
        v = v[~np.isnan(v)]
        if len(v) > 0:
            summary[f'{name}_mean'] = np.mean(v)
            summary[f'{name}_std'] = np.std(v)
            summary[f'{name}_median'] = np.median(v)
            summary[f'{name}_cv'] = np.std(v) / np.mean(v) * 100 if np.mean(v) != 0 else np.nan
            summary[f'{name}_min'] = np.min(v)
            summary[f'{name}_max'] = np.max(v)
            summary[f'{name}_n'] = len(v)
        else:
            for s in ['_mean', '_std', '_median', '_cv', '_min', '_max', '_n']:
                summary[f'{name}{s}'] = np.nan

    summary['bpm_mean'] = 60000 / np.mean(bp_ms) if len(bp_ms) > 0 and np.mean(bp_ms) > 0 else np.nan
    summary['stv_ms'] = np.mean(np.abs(np.diff(bp_ms))) / np.sqrt(2) if len(bp_ms) > 1 else np.nan
    summary['beat_periods'] = beat_periods
    summary['fpd_values'] = np.array([p['fpd_ms'] / 1000 for p in all_params if not np.isnan(p['fpd_ms'])])
    summary['fpdc_values'] = np.array([p['fpdc_ms'] / 1000 for p in all_params if not np.isnan(p['fpdc_ms'])])

    # Add template info to summary
    if template_fpd_samples is not None:
        summary['template_fpd_ms'] = template_fpd_samples / fs * 1000

    return all_params, summary
