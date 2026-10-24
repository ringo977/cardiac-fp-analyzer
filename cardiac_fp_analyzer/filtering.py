"""
filtering.py — Signal conditioning for cardiac field potentials.

Provides:
  - Notch filter (50 Hz + harmonics) for powerline interference
  - Bandpass filter for FP signal extraction
  - Baseline drift removal
  - Savitzky-Golay smoothing for gentle noise reduction

All parameters are configurable via FilterConfig (see config.py).
Filter coefficients are cached per (fs, parameters) to avoid
redundant recomputation across files with the same sample rate.
"""

from functools import lru_cache

import numpy as np
from scipy import signal

# ── Cached coefficient computation ──────────────────────────────────

@lru_cache(maxsize=16)
def _notch_coeffs(freq, Q, fs):
    """Compute and cache notch filter coefficients."""
    return signal.iirnotch(freq, Q, fs)


@lru_cache(maxsize=8)
def _butter_coeffs(order, low, high, btype):
    """Compute and cache Butterworth filter coefficients."""
    return signal.butter(order, [low, high] if btype == 'band' else low, btype=btype)


@lru_cache(maxsize=8)
def _butter_sos(order, low, high, btype):
    return signal.butter(order, [low, high] if btype == 'band' else low, btype=btype, output='sos')


@lru_cache(maxsize=8)
def _ba_is_stable(order, low, high, btype):
    """(b, a) designs with a very low normalised cut-off (0.5 Hz at 20 kHz)
    round a pole onto or outside the unit circle and filtfilt overflows."""
    _, a = _butter_coeffs(order, low, high, btype)
    return bool(np.all(np.abs(np.roots(a)) < 1.0))


def _butter_filtfilt(data, order, low, high, btype):
    """filtfilt with the (b, a) design used since v1 — results at the usual
    ~2 kHz are unchanged — and second-order sections when that design is
    numerically unstable (Oct 2026)."""
    if _ba_is_stable(order, low, high, btype):
        b, a = _butter_coeffs(order, low, high, btype)
        return signal.filtfilt(b, a, data)
    return signal.sosfiltfilt(_butter_sos(order, low, high, btype), data)


# ── Filter functions ────────────────────────────────────────────────

def notch_filter(data, fs, freq=50.0, n_harmonics=3, Q=30):
    """Remove powerline interference at `freq` Hz and its harmonics."""
    y = np.array(data, dtype=np.float64)
    for i in range(1, n_harmonics + 1):
        f_notch = freq * i
        if f_notch >= fs / 2: break
        b, a = _notch_coeffs(f_notch, Q, fs)
        y = signal.filtfilt(b, a, y)
    return y


def bandpass_filter(data, fs, lowcut=0.5, highcut=500.0, order=4):
    """Butterworth bandpass filter tuned for cardiac FP."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 0.9999)
    return _butter_filtfilt(data, order, low, high, 'band')


def highpass_filter(data, fs, cutoff=0.5, order=4):
    """Remove baseline drift with a high-pass Butterworth filter."""
    nyq = 0.5 * fs
    return _butter_filtfilt(data, order, cutoff / nyq, 0.0, 'high')


def lowpass_filter(data, fs, cutoff=200.0, order=4):
    """Low-pass Butterworth filter for gentle smoothing."""
    nyq = 0.5 * fs
    return _butter_filtfilt(data, order, cutoff / nyq, 0.0, 'low')


def smooth_savgol(data, window_length=11, polyorder=3):
    """Savitzky-Golay smoothing — preserves peak shapes."""
    if window_length % 2 == 0: window_length += 1
    if window_length > len(data):
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
    return signal.savgol_filter(data, window_length, polyorder)


def full_filter_pipeline(data, fs, cfg=None):
    """
    Complete filtering pipeline:
      1. Notch at 50 Hz (+ harmonics)
      2. Bandpass 0.5–500 Hz
      3. Light Savitzky-Golay smoothing

    Parameters
    ----------
    data : array-like — raw signal
    fs : float — sample rate (Hz)
    cfg : FilterConfig or None — if None, uses defaults
    """
    if cfg is None:
        from .config import FilterConfig
        cfg = FilterConfig()

    y = np.array(data, dtype=np.float64)
    if cfg.notch_freq_hz and cfg.notch_freq_hz > 0:
        y = notch_filter(y, fs, freq=cfg.notch_freq_hz,
                         n_harmonics=cfg.notch_harmonics, Q=cfg.notch_q)
    y = bandpass_filter(y, fs, lowcut=cfg.bandpass_low_hz,
                        highcut=cfg.bandpass_high_hz, order=cfg.bandpass_order)
    y = smooth_savgol(y, window_length=cfg.savgol_window,
                      polyorder=cfg.savgol_polyorder)
    return y


def describe_filtering(config, fs, metadata=None):
    """What was done to the signal before beat detection, as data and as one line of text.

    Returns a dict with ``dispositivo`` (acquisition-side filters and sample mode, when the file says so),
    ``decimazione`` (loader anti-alias decimation, if any), ``software`` (the steps of ``full_filter_pipeline``
    with their parameters at this sample rate), ``guadagno`` and ``testo`` (Italian summary). Stored in
    ``result['metadata']['processing']`` by ``analyze_single_file``.
    """
    md = metadata or {}
    fc = config.filtering
    dev = {}
    if md.get('device'):
        dev['strumento'] = md['device']
    if md.get('sample_mode'):
        dev['modalita_campionamento'] = md['sample_mode']
    if md.get('device_filters'):
        dev['filtri'] = md['device_filters']
    dec = None
    if md.get('decimation_factor') and md.get('decimation_factor') > 1:
        dec = {'fattore': int(md['decimation_factor']), 'da_hz': float(md.get('original_sample_rate') or 0),
               'a_hz': float(fs), 'metodo': 'FIR anti-alias, fase zero'}
    steps = []
    if fc.notch_freq_hz and fc.notch_freq_hz > 0:
        freqs = [fc.notch_freq_hz * (k + 1) for k in range(int(fc.notch_harmonics))]
        steps.append({'passo': 'notch', 'frequenze_hz': freqs, 'q': fc.notch_q})
    steps.append({'passo': 'passa-banda', 'basso_hz': fc.bandpass_low_hz, 'alto_hz': min(fc.bandpass_high_hz, 0.49 * fs),
                  'ordine': fc.bandpass_order, 'tipo': 'Butterworth, fase zero (avanti e indietro)'})
    steps.append({'passo': 'Savitzky-Golay', 'finestra_campioni': fc.savgol_window,
                  'finestra_ms': round(fc.savgol_window / fs * 1000, 2), 'grado': fc.savgol_polyorder})
    gain = float(md.get('gain_applied', getattr(config, 'amplifier_gain', 1.0)) or 1.0)
    parts = []
    if md.get('format') == 'nwb':
        parts.append('file NWB (volt all\'elettrodo, conversione del file applicata)')
    if dev.get('filtri'):
        parts.append('strumento: ' + dev['filtri'] if isinstance(dev['filtri'], str) else 'strumento: ' + ', '.join(
            f"{k} {v}" for k, v in dev['filtri'].items()))
    if dev.get('modalita_campionamento'):
        parts.append(f"campionamento {dev['modalita_campionamento']}")
    if dec:
        parts.append(f"decimazione ×{dec['fattore']} ({dec['da_hz']:.0f} → {dec['a_hz']:.0f} Hz, FIR fase zero)")
    for st in steps:
        if st['passo'] == 'notch':
            parts.append('notch ' + '/'.join(f'{f:g}' for f in st['frequenze_hz']) + f" Hz (Q {st['q']:g})")
        elif st['passo'] == 'passa-banda':
            parts.append(f"passa-banda {st['basso_hz']:g}–{st['alto_hz']:g} Hz Butterworth ordine {st['ordine']} fase zero")
        else:
            parts.append(f"Savitzky–Golay {st['finestra_campioni']} campioni ({st['finestra_ms']:g} ms) grado {st['grado']}")
    if gain != 1.0:
        parts.append(f'guadagno ÷{gain:g}')
    return {'dispositivo': dev, 'decimazione': dec, 'software': steps, 'guadagno': gain,
            'frequenza_campionamento_hz': float(fs), 'testo': 'Filtraggio: ' + '; '.join(parts)}
