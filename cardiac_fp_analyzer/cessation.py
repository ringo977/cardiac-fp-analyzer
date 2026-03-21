"""
cessation.py — Beating cessation and quiescence detection.

Detects when a drug causes cardiomyocytes to stop beating, which is a
strong indicator of cardiotoxicity (e.g. dofetilide at high concentrations).

Strategies:
  1. Signal energy analysis: compare energy in sliding windows to baseline noise
  2. Gap detection: find prolonged periods without detected beats
  3. Progressive deterioration: detect declining beat amplitude over time
  4. Terminal cessation: beating stops and never resumes

This module is critical for drugs that destroy waveform morphology — in those
cases the standard FPD measurement fails (low confidence), but cessation
detection captures the drug effect via a different mechanism.
"""

import numpy as np
from scipy import signal as sig
from dataclasses import dataclass, field


@dataclass
class CessationConfig:
    """Configuration for cessation/quiescence detection."""

    # --- Energy analysis ---
    # Sliding window duration for energy computation (seconds)
    energy_window_s: float = 2.0
    # Step between windows (seconds)
    energy_step_s: float = 0.5
    # Energy drop threshold: region is "silent" if energy < threshold × baseline_energy
    energy_silence_ratio: float = 0.15
    # Minimum silent duration to count as cessation (seconds)
    min_silence_duration_s: float = 3.0

    # --- Gap detection ---
    # A gap is an inter-beat interval > gap_factor × median_bp
    gap_factor: float = 3.0
    # Minimum gap duration to flag (seconds)
    min_gap_s: float = 2.0

    # --- Progressive deterioration ---
    # Split recording into N segments for amplitude trend analysis
    n_trend_segments: int = 4
    # Amplitude must drop below this fraction of initial to flag deterioration
    amplitude_decay_threshold: float = 0.40
    # Minimum beats per segment for trend analysis
    min_beats_per_segment: int = 3

    # --- Terminal cessation ---
    # If the last N% of the recording has no beats, flag terminal cessation
    terminal_fraction: float = 0.20
    # Minimum silent tail duration (seconds) to flag terminal cessation
    min_terminal_silence_s: float = 5.0

    # --- Confidence / scoring ---
    # Weight of each sub-detector for overall cessation confidence
    weight_energy: float = 0.35
    weight_gaps: float = 0.25
    weight_deterioration: float = 0.20
    weight_terminal: float = 0.20


@dataclass
class CessationReport:
    """Results of cessation detection."""

    # Overall
    has_cessation: bool = False
    cessation_confidence: float = 0.0    # 0-1 composite score
    cessation_type: str = 'none'         # 'none', 'intermittent', 'terminal', 'progressive', 'full'

    # Sub-detector results
    silent_periods: list = field(default_factory=list)  # list of (start_s, end_s, duration_s)
    total_silent_s: float = 0.0
    silent_fraction: float = 0.0

    gaps: list = field(default_factory=list)  # list of (start_s, end_s, duration_s)
    max_gap_s: float = 0.0

    amplitude_trend: list = field(default_factory=list)  # per-segment mean amplitudes
    amplitude_ratio: float = 1.0   # last_segment_amp / first_segment_amp

    terminal_silence_s: float = 0.0
    is_terminal: bool = False

    # Detail info
    details: dict = field(default_factory=dict)


def detect_cessation(filtered_signal, fs, beat_indices, all_params=None,
                     beat_indices_clean=None, qc_report=None, cfg=None):
    """
    Detect beating cessation in a recording.

    Parameters
    ----------
    filtered_signal : array — filtered FP signal
    fs : float — sampling rate
    beat_indices : array — ALL detected beat sample indices (raw, before QC)
    all_params : list of dicts — per-beat parameters (for amplitude trend)
    beat_indices_clean : array or None — QC-filtered beat indices (for gap detection)
    qc_report : QualityReport or None — QC info (rejection rate as indicator)
    cfg : CessationConfig or None

    Returns
    -------
    CessationReport
    """
    if cfg is None:
        cfg = CessationConfig()

    report = CessationReport()
    duration_s = len(filtered_signal) / fs

    if duration_s < 2:
        return report

    # Use clean beats for gap detection (raw beats include noise spikes)
    # Fall back to raw beats if clean not provided
    gap_beats = beat_indices_clean if beat_indices_clean is not None else beat_indices

    # ═══════════════════════════════════════════════════════════════
    #  1. Energy-based silence detection
    # ═══════════════════════════════════════════════════════════════
    energy_conf = _detect_energy_silence(filtered_signal, fs, beat_indices, cfg, report)

    # ═══════════════════════════════════════════════════════════════
    #  2. Gap detection (on CLEAN beats — noise spikes removed)
    # ═══════════════════════════════════════════════════════════════
    gap_conf = _detect_gaps(gap_beats, fs, duration_s, cfg, report)

    # ═══════════════════════════════════════════════════════════════
    #  3. Progressive amplitude deterioration
    # ═══════════════════════════════════════════════════════════════
    deterior_conf = _detect_deterioration(beat_indices, all_params, fs, duration_s, cfg, report)

    # ═══════════════════════════════════════════════════════════════
    #  4. Terminal cessation (on clean beats)
    # ═══════════════════════════════════════════════════════════════
    terminal_conf = _detect_terminal_cessation(gap_beats, fs, duration_s, cfg, report)

    # ═══════════════════════════════════════════════════════════════
    #  5. QC-based waveform destruction indicator
    # ═══════════════════════════════════════════════════════════════
    qc_conf = 0.0
    if qc_report is not None:
        rejection_rate = qc_report.rejection_rate if hasattr(qc_report, 'rejection_rate') else 0
        # Very high rejection rate → waveform is destroyed
        if rejection_rate > 0.80:
            qc_conf = 1.0
        elif rejection_rate > 0.60:
            qc_conf = 0.6
        elif rejection_rate > 0.40:
            qc_conf = 0.3
        report.details['qc_rejection_rate'] = rejection_rate

    # ═══════════════════════════════════════════════════════════════
    #  Composite scoring
    # ═══════════════════════════════════════════════════════════════
    # Redistribute weights to include QC indicator
    w_total = (cfg.weight_energy + cfg.weight_gaps +
               cfg.weight_deterioration + cfg.weight_terminal)
    # Add 15% weight for QC indicator, scale others proportionally
    qc_weight = 0.15
    scale = (1.0 - qc_weight) / w_total if w_total > 0 else 0

    composite = (cfg.weight_energy * scale * energy_conf +
                 cfg.weight_gaps * scale * gap_conf +
                 cfg.weight_deterioration * scale * deterior_conf +
                 cfg.weight_terminal * scale * terminal_conf +
                 qc_weight * qc_conf)

    report.cessation_confidence = min(1.0, composite)
    report.details['sub_scores'] = {
        'energy': energy_conf,
        'gaps': gap_conf,
        'deterioration': deterior_conf,
        'terminal': terminal_conf,
        'qc_destruction': qc_conf,
    }

    # Classify — any single strong signal is enough
    if report.is_terminal and terminal_conf > 0.6:
        report.cessation_type = 'terminal'
        report.has_cessation = True
    elif report.silent_fraction > 0.5:
        report.cessation_type = 'full'
        report.has_cessation = True
    elif report.amplitude_ratio < cfg.amplitude_decay_threshold and deterior_conf > 0.5:
        report.cessation_type = 'progressive'
        report.has_cessation = True
    elif gap_conf >= 0.7 or (len(report.gaps) > 0 and report.max_gap_s > 10):
        report.cessation_type = 'intermittent'
        report.has_cessation = True
    elif qc_conf >= 0.8 and gap_conf > 0.3:
        # QC says waveform is destroyed + some gaps in clean beats
        report.cessation_type = 'waveform_destruction'
        report.has_cessation = True
    elif len(report.silent_periods) > 0 and energy_conf > 0.4:
        report.cessation_type = 'intermittent'
        report.has_cessation = True
    elif composite > 0.35:
        report.cessation_type = 'intermittent'
        report.has_cessation = True

    return report


def _detect_energy_silence(signal, fs, beat_indices, cfg, report):
    """Detect silent periods using sliding-window RMS energy."""
    win_samples = int(cfg.energy_window_s * fs)
    step_samples = int(cfg.energy_step_s * fs)

    if win_samples >= len(signal):
        return 0.0

    # Compute energy in sliding windows
    n_windows = (len(signal) - win_samples) // step_samples + 1
    energies = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * step_samples
        window = signal[start:start + win_samples]
        energies[i] = np.sqrt(np.mean(window ** 2))

    if len(energies) == 0:
        return 0.0

    # Baseline energy: robust estimate from windows containing beats
    beat_set = set(beat_indices)
    windows_with_beats = []
    for i in range(n_windows):
        start = i * step_samples
        end = start + win_samples
        if any(start <= b < end for b in beat_set):
            windows_with_beats.append(energies[i])

    if len(windows_with_beats) > 3:
        baseline_energy = np.median(windows_with_beats)
    else:
        baseline_energy = np.median(energies)

    if baseline_energy <= 0:
        return 0.0

    # Find silent windows
    threshold = cfg.energy_silence_ratio * baseline_energy
    is_silent = energies < threshold

    # Group consecutive silent windows into periods
    silent_periods = []
    in_silence = False
    start_t = 0
    for i, s in enumerate(is_silent):
        t = i * cfg.energy_step_s
        if s and not in_silence:
            in_silence = True
            start_t = t
        elif not s and in_silence:
            in_silence = False
            duration = t - start_t
            if duration >= cfg.min_silence_duration_s:
                silent_periods.append((start_t, t, duration))

    # Handle trailing silence
    if in_silence:
        end_t = (len(is_silent) - 1) * cfg.energy_step_s + cfg.energy_window_s
        duration = end_t - start_t
        if duration >= cfg.min_silence_duration_s:
            silent_periods.append((start_t, end_t, duration))

    report.silent_periods = silent_periods
    total_silent = sum(p[2] for p in silent_periods)
    report.total_silent_s = total_silent
    duration_s = len(signal) / fs
    report.silent_fraction = total_silent / duration_s if duration_s > 0 else 0

    # Confidence: proportion of recording that is silent
    if report.silent_fraction > 0.5:
        return 1.0
    elif report.silent_fraction > 0.2:
        return 0.7
    elif len(silent_periods) > 0:
        return 0.4
    return 0.0


def _detect_gaps(beat_indices, fs, duration_s, cfg, report):
    """Detect long gaps between beats."""
    if len(beat_indices) < 2:
        # No beats at all — the entire recording is a gap
        if duration_s > cfg.min_gap_s:
            report.gaps = [(0, duration_s, duration_s)]
            report.max_gap_s = duration_s
            return 1.0
        return 0.0

    beat_times = beat_indices / fs
    ibis = np.diff(beat_times)
    median_ibi = np.median(ibis)

    gap_threshold = max(cfg.gap_factor * median_ibi, cfg.min_gap_s)

    gaps = []
    for i, ibi in enumerate(ibis):
        if ibi > gap_threshold:
            start = beat_times[i]
            end = beat_times[i + 1]
            gaps.append((start, end, ibi))

    # Check for gap at start (first beat late)
    if beat_times[0] > gap_threshold:
        gaps.insert(0, (0, beat_times[0], beat_times[0]))

    # Check for gap at end (last beat early)
    tail = duration_s - beat_times[-1]
    if tail > gap_threshold:
        gaps.append((beat_times[-1], duration_s, tail))

    report.gaps = gaps
    report.max_gap_s = max((g[2] for g in gaps), default=0.0)

    total_gap = sum(g[2] for g in gaps)
    gap_fraction = total_gap / duration_s if duration_s > 0 else 0

    # Confidence scaling: consider both fraction AND absolute duration
    if gap_fraction > 0.3 or report.max_gap_s > 30:
        return 1.0
    elif gap_fraction > 0.1 or report.max_gap_s > 10:
        return 0.7
    elif len(gaps) > 0:
        # Scale by how long the longest gap is
        return min(0.6, 0.3 + report.max_gap_s / 20.0)
    return 0.0


def _detect_deterioration(beat_indices, all_params, fs, duration_s, cfg, report):
    """Detect progressive amplitude decline across the recording."""
    if all_params is None or len(all_params) < cfg.n_trend_segments * cfg.min_beats_per_segment:
        return 0.0

    # Get amplitudes in time order
    amps = []
    for p in all_params:
        a = p.get('spike_amplitude_mV', np.nan)
        if not np.isnan(a):
            amps.append(a)

    if len(amps) < cfg.n_trend_segments * cfg.min_beats_per_segment:
        return 0.0

    # Split into segments
    segment_size = len(amps) // cfg.n_trend_segments
    segment_means = []
    for i in range(cfg.n_trend_segments):
        start = i * segment_size
        end = start + segment_size if i < cfg.n_trend_segments - 1 else len(amps)
        segment_means.append(np.mean(amps[start:end]))

    report.amplitude_trend = segment_means

    if segment_means[0] > 0:
        report.amplitude_ratio = segment_means[-1] / segment_means[0]
    else:
        report.amplitude_ratio = 1.0

    # Check for monotonic decline
    n_declining = sum(1 for i in range(1, len(segment_means))
                      if segment_means[i] < segment_means[i-1])

    if report.amplitude_ratio < cfg.amplitude_decay_threshold:
        if n_declining >= cfg.n_trend_segments - 1:
            return 1.0   # Strong monotonic decline
        return 0.7
    elif report.amplitude_ratio < 0.6:
        return 0.4
    return 0.0


def _detect_terminal_cessation(beat_indices, fs, duration_s, cfg, report):
    """Detect if beating stops at the end of the recording."""
    if duration_s < cfg.min_terminal_silence_s:
        return 0.0

    if len(beat_indices) == 0:
        # No beats at all
        report.terminal_silence_s = duration_s
        report.is_terminal = True
        return 1.0

    last_beat_time = beat_indices[-1] / fs
    tail_silence = duration_s - last_beat_time

    report.terminal_silence_s = tail_silence

    terminal_threshold = max(cfg.min_terminal_silence_s,
                             duration_s * cfg.terminal_fraction)

    if tail_silence >= terminal_threshold:
        report.is_terminal = True
        # Scale confidence by how long the silence is
        confidence = min(1.0, tail_silence / (2 * terminal_threshold))
        return confidence

    return 0.0
