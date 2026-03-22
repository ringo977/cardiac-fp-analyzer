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

import numpy as np
from functools import lru_cache
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
    b, a = _butter_coeffs(order, low, high, 'band')
    return signal.filtfilt(b, a, data)


def highpass_filter(data, fs, cutoff=0.5, order=4):
    """Remove baseline drift with a high-pass Butterworth filter."""
    nyq = 0.5 * fs
    b, a = _butter_coeffs(order, cutoff / nyq, 0.0, 'high')
    return signal.filtfilt(b, a, data)


def lowpass_filter(data, fs, cutoff=200.0, order=4):
    """Low-pass Butterworth filter for gentle smoothing."""
    nyq = 0.5 * fs
    b, a = _butter_coeffs(order, cutoff / nyq, 0.0, 'low')
    return signal.filtfilt(b, a, data)


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

    y = notch_filter(data, fs, freq=cfg.notch_freq_hz,
                     n_harmonics=cfg.notch_harmonics, Q=cfg.notch_q)
    y = bandpass_filter(y, fs, lowcut=cfg.bandpass_low_hz,
                        highcut=cfg.bandpass_high_hz, order=cfg.bandpass_order)
    y = smooth_savgol(y, window_length=cfg.savgol_window,
                      polyorder=cfg.savgol_polyorder)
    return y
