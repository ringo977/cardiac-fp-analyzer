"""
filtering.py — Signal conditioning for cardiac field potentials.

Provides:
  - Notch filter (50 Hz + harmonics) for powerline interference
  - Bandpass filter for FP signal extraction
  - Baseline drift removal
  - Savitzky-Golay smoothing for gentle noise reduction
"""

import numpy as np
from scipy import signal


def notch_filter(data, fs, freq=50.0, n_harmonics=3, Q=30):
    """Remove powerline interference at `freq` Hz and its harmonics."""
    y = np.array(data, dtype=np.float64)
    for i in range(1, n_harmonics + 1):
        f_notch = freq * i
        if f_notch >= fs / 2: break
        b, a = signal.iirnotch(f_notch, Q, fs)
        y = signal.filtfilt(b, a, y)
    return y


def bandpass_filter(data, fs, lowcut=0.5, highcut=500.0, order=4):
    """Butterworth bandpass filter tuned for cardiac FP."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 0.9999)
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def highpass_filter(data, fs, cutoff=0.5, order=4):
    """Remove baseline drift with a high-pass Butterworth filter."""
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype='high')
    return signal.filtfilt(b, a, data)


def lowpass_filter(data, fs, cutoff=200.0, order=4):
    """Low-pass Butterworth filter for gentle smoothing."""
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype='low')
    return signal.filtfilt(b, a, data)


def smooth_savgol(data, window_length=11, polyorder=3):
    """Savitzky-Golay smoothing — preserves peak shapes."""
    if window_length % 2 == 0: window_length += 1
    if window_length > len(data):
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
    return signal.savgol_filter(data, window_length, polyorder)


def full_filter_pipeline(data, fs, notch_freq=50.0, bp_low=0.5, bp_high=500.0):
    """
    Complete filtering pipeline:
      1. Notch at 50 Hz (+ harmonics)
      2. Bandpass 0.5–500 Hz
      3. Light Savitzky-Golay smoothing
    """
    y = notch_filter(data, fs, freq=notch_freq)
    y = bandpass_filter(y, fs, lowcut=bp_low, highcut=bp_high)
    y = smooth_savgol(y, window_length=7, polyorder=3)
    return y
