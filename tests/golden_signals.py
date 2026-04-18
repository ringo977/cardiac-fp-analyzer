"""
golden_signals.py — Synthetic FP signal generators for golden-reference tests.

Generates realistic hiPSC-CM field potential signals with configurable:
  - Beat rate, amplitude, noise level
  - Repolarization shape and amplitude
  - Pathological features (EADs, cessation, irregular rhythm)

These are used by test_golden_reference.py to verify that the full
pipeline produces stable, expected results.
"""

import numpy as np


def _single_fp_beat(fs, depol_amp=0.05, repol_amp=0.008, fpd_ms=300,
                    depol_width_ms=2.0, repol_width_ms=40.0):
    """Generate one synthetic FP beat waveform.

    Returns a 1-D array representing one beat from 0 to fpd_ms + 200 ms.
    The depolarization spike is at t=0, repolarization peak at t=fpd_ms.
    """
    duration_ms = fpd_ms + 200  # Extra room after repolarization
    n = int(duration_ms / 1000 * fs)
    t_ms = np.arange(n) / fs * 1000

    beat = np.zeros(n)

    # Depolarization: sharp biphasic spike at t=0
    depol_sigma = depol_width_ms / 2
    beat += depol_amp * np.exp(-0.5 * (t_ms / depol_sigma) ** 2)
    beat -= depol_amp * 0.6 * np.exp(-0.5 * ((t_ms - depol_width_ms) / (depol_sigma * 1.5)) ** 2)

    # Repolarization: broad positive hump at t=fpd_ms
    repol_sigma = repol_width_ms / 2
    beat += repol_amp * np.exp(-0.5 * ((t_ms - fpd_ms) / repol_sigma) ** 2)

    return beat


def generate_regular_fp(fs=2000.0, duration_s=10.0, beat_period_ms=800.0,
                        depol_amp=0.05, repol_amp=0.008, fpd_ms=300.0,
                        noise_std=0.0005, seed=42):
    """Generate a clean, regular FP signal — the baseline golden reference.

    Parameters
    ----------
    fs : float — sampling rate (Hz)
    duration_s : float — total signal duration (s)
    beat_period_ms : float — inter-beat interval (ms)
    depol_amp : float — depolarization spike amplitude (V)
    repol_amp : float — repolarization hump amplitude (V)
    fpd_ms : float — field potential duration (ms)
    noise_std : float — Gaussian noise std (V)
    seed : int — random seed for reproducibility

    Returns
    -------
    signal : 1-D array (V)
    time : 1-D array (s)
    expected : dict with ground-truth values
    """
    rng = np.random.RandomState(seed)
    n_total = int(duration_s * fs)
    signal = np.zeros(n_total)
    time = np.arange(n_total) / fs

    beat_template = _single_fp_beat(fs, depol_amp, repol_amp, fpd_ms)
    beat_period_samples = int(beat_period_ms / 1000 * fs)

    true_beat_indices = []
    pos = int(beat_period_ms / 1000 * fs)  # Start after one period
    while pos + len(beat_template) < n_total:
        signal[pos:pos + len(beat_template)] += beat_template
        true_beat_indices.append(pos)
        pos += beat_period_samples

    # Add noise
    signal += rng.randn(n_total) * noise_std

    n_beats = len(true_beat_indices)
    expected = {
        'n_beats': n_beats,
        'beat_period_ms': beat_period_ms,
        'bpm': 60000 / beat_period_ms,
        'fpd_ms_approx': fpd_ms,  # Approximate — pipeline may differ by method
        'classification': 'Normal Sinus Rhythm',
        'risk_score_max': 15,  # Should be low for regular signal
        'beat_indices': np.array(true_beat_indices),
    }
    return signal, time, expected


def generate_irregular_fp(fs=2000.0, duration_s=10.0, mean_period_ms=800.0,
                          cv_period_pct=25.0, depol_amp=0.05, repol_amp=0.008,
                          fpd_ms=300.0, noise_std=0.0005, seed=42):
    """Generate an irregular-rhythm FP signal.

    Beat periods are drawn from a normal distribution with specified CV.
    """
    rng = np.random.RandomState(seed)
    n_total = int(duration_s * fs)
    signal = np.zeros(n_total)
    time = np.arange(n_total) / fs

    beat_template = _single_fp_beat(fs, depol_amp, repol_amp, fpd_ms)

    period_std = mean_period_ms * cv_period_pct / 100
    true_beat_indices = []
    pos = int(mean_period_ms / 1000 * fs)

    while pos + len(beat_template) < n_total:
        signal[pos:pos + len(beat_template)] += beat_template
        true_beat_indices.append(pos)
        next_period = max(400, rng.normal(mean_period_ms, period_std))
        pos += int(next_period / 1000 * fs)

    signal += rng.randn(n_total) * noise_std

    expected = {
        'n_beats_min': len(true_beat_indices) - 2,
        'n_beats_max': len(true_beat_indices) + 2,
        'classification_not': 'Normal Sinus Rhythm',
        'risk_score_min': 5,
    }
    return signal, time, expected


def generate_cessation_fp(fs=2000.0, duration_s=10.0, beat_period_ms=800.0,
                          cessation_at_s=5.0, depol_amp=0.05, repol_amp=0.008,
                          fpd_ms=300.0, noise_std=0.0005, seed=42):
    """Generate a signal that stops beating midway (cessation).

    Beats are present up to cessation_at_s, then only noise.
    """
    rng = np.random.RandomState(seed)
    n_total = int(duration_s * fs)
    signal = np.zeros(n_total)
    time = np.arange(n_total) / fs

    beat_template = _single_fp_beat(fs, depol_amp, repol_amp, fpd_ms)
    beat_period_samples = int(beat_period_ms / 1000 * fs)
    cessation_sample = int(cessation_at_s * fs)

    true_beat_indices = []
    pos = int(beat_period_ms / 1000 * fs)
    while pos + len(beat_template) < cessation_sample:
        signal[pos:pos + len(beat_template)] += beat_template
        true_beat_indices.append(pos)
        pos += beat_period_samples

    signal += rng.randn(n_total) * noise_std

    expected = {
        'n_beats': len(true_beat_indices),
        'has_cessation': True,
    }
    return signal, time, expected
